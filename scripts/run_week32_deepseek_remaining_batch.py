"""Run the frozen Week 32 unseen batch against the official DeepSeek API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_week32_remaining_batch as prepared
from scripts import run_week14_behavior_similarity as rebuild
from scripts import run_week16_local6_openai_top5 as live
from scripts import run_week31_deepseek_adaptive_capacity as week31
from scripts.re_evaluate_localization_truth_repair_v1 import _truths_for_transactions
from src.cloc_v2.phase1_llm_client import OpenAICompatibleTransport
from src.cloc_v2.source_cache import atomic_json


SOURCE_AUDIT = prepared.SCREEN_ROOT / "post_sourcify_source_evaluability.json"
SOURCE_MANIFEST = prepared.OUTPUT / "inference_manifest.json"
OUTPUT = ROOT / "data" / "prototypes" / "week32_deepseek_remaining_20260911"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
EXPECTED_SAMPLES = 19
EXPECTED_UNIQUE_INPUTS = 13


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trial_manifest() -> dict[str, Any]:
    groups = {row["sample_alias"]: row["group"] for row in _load(SOURCE_AUDIT)["samples"]}
    source = _load(SOURCE_MANIFEST)
    samples = [
        row for row in source["samples"]
        if row["preparation_status"] == "READY_FOR_LLM"
        and groups[row["sample_alias"]] == "UNRUN_REMAINDER"
    ]
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        by_prompt[row["prompt_sha256"]].append(row)
    inputs = []
    for number, prompt_hash in enumerate(sorted(by_prompt), 1):
        rows = sorted(by_prompt[prompt_hash], key=lambda row: row["sample_alias"])
        inputs.append({
            "input_alias": f"week32_input_{number:03d}",
            "representative_alias": rows[0]["sample_alias"],
            "mapped_sample_aliases": [row["sample_alias"] for row in rows],
            "prompt_sha256": prompt_hash,
            "prompt_mode": rows[0]["prompt_mode"],
        })
    if len(samples) != EXPECTED_SAMPLES or len(inputs) != EXPECTED_UNIQUE_INPUTS:
        raise ValueError("WEEK32_TRIAL_SELECTION_DRIFT")
    return {
        "schema_version": "week32_deepseek_remaining_manifest_v1",
        "mode": "FROZEN_UNSEEN_PROMPTS_DEDUPLICATED_BY_PROMPT_SHA256",
        "source_manifest_sha256": _file_sha(SOURCE_MANIFEST),
        "sample_count": len(samples),
        "unique_input_count": len(inputs),
        "provider": "DEEPSEEK_OFFICIAL",
        "provider_host": "api.deepseek.com",
        "model": source["model"],
        "thinking_mode": source["thinking_mode"],
        "temperature": source["temperature"],
        "max_output_tokens": live.MAX_OUTPUT_TOKENS,
        "max_requests_total": len(inputs),
        "max_requests_per_unique_input": 1,
        "retry_enabled": False,
        "truth_read_before_inference": False,
        "samples": samples,
        "unique_inputs": inputs,
    }


def prepare(output: Path) -> int:
    manifest = _trial_manifest()
    live._write_manifest_once(output / "inference_manifest.json", manifest)
    print(json.dumps({
        "sample_count": manifest["sample_count"],
        "unique_input_count": manifest["unique_input_count"],
        "max_requests_total": manifest["max_requests_total"],
        "model": manifest["model"],
        "retry_enabled": manifest["retry_enabled"],
    }, ensure_ascii=False, indent=2))
    return 0


def _context(manifest: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[Path, str]], tuple[Any, ...], Any]:
    samples = {row["sample_alias"]: dict(row) for row in manifest["samples"]}
    trees = prepared._tree_index()
    normal_records = rebuild.replay._load_normal_records(rebuild.replay.DEFAULT_NORMAL)
    cache_index = rebuild.replay.build_verified_source_cache_index(
        rebuild.replay.DEFAULT_CACHE,
        blockchain_chain_id=1,
        chain_key="eth",
    )
    return samples, trees, normal_records, cache_index


def _preparation(row: Mapping[str, Any], trees: Mapping[str, tuple[Path, str]], normal_records: tuple[Any, ...], cache_index: Any) -> Any:
    options = {}
    if row["prompt_mode"] == "ADAPTIVE_40":
        options = {"max_prompt_candidates": prepared.EXPANDED_CANDIDATES, "max_prompt_bytes": prepared.EXPANDED_BYTES}
    return rebuild._build_sample(
        dict(row), trees[row["tx_hash"].lower()][0], normal_records,
        cache_index, rebuild.replay.DEFAULT_CACHE, **options,
    )[1]


def _freeze(output: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    predictions = [_load(output / "predictions" / f"{row['input_alias']}.json") for row in manifest["unique_inputs"]]
    core = {
        "schema_version": "week32_deepseek_remaining_frozen_v1",
        "inference_manifest_sha256": _file_sha(output / "inference_manifest.json"),
        "predictions": predictions,
    }
    frozen = {**core, "payload_sha256": live._sha(core)}
    live._write_manifest_once(output / "frozen_predictions.json", frozen)
    return frozen


def infer(output: Path, timeout_seconds: int, resume: bool) -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY") or ""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY_MISSING")
    expected = _trial_manifest()
    manifest_path = output / "inference_manifest.json"
    if not manifest_path.exists() or live._canonical(_load(manifest_path)) != live._canonical(expected):
        raise ValueError("RUN_PREPARE_OR_MANIFEST_DRIFT")
    samples, trees, normal_records, cache_index = _context(expected)
    transport = OpenAICompatibleTransport(DEEPSEEK_BASE_URL)
    completed = []
    for group in expected["unique_inputs"]:
        input_alias = group["input_alias"]
        path = output / "predictions" / f"{input_alias}.json"
        if path.exists():
            if not resume:
                raise ValueError("PREDICTIONS_EXIST_USE_RESUME")
            completed.append(input_alias)
            print(f"input={input_alias} status=RESUMED", flush=True)
            continue
        sample = samples[group["representative_alias"]]
        preparation = _preparation(sample, trees, normal_records, cache_index)
        if preparation.prompt is None or preparation.prompt_sha256 != group["prompt_sha256"]:
            raise ValueError(f"WEEK32_PROMPT_DRIFT:{input_alias}")
        print(f"input={input_alias} status=RUNNING", flush=True)
        options = {}
        if sample["prompt_mode"] == "ADAPTIVE_40":
            options = {"max_prompt_candidates": prepared.EXPANDED_CANDIDATES, "max_prompt_bytes": prepared.EXPANDED_BYTES}
        prediction = live._infer_one(
            sample, trees[sample["tx_hash"].lower()][0], normal_records, cache_index,
            rebuild.replay.DEFAULT_CACHE, expected["model"], transport, api_key,
            timeout_seconds, expected["thinking_mode"], **options,
        )
        if prediction.get("prompt_sha256") != group["prompt_sha256"]:
            raise ValueError(f"WEEK32_INFERENCE_PROMPT_DRIFT:{input_alias}")
        prediction = week31._apply_frozen_guard(prediction, preparation)
        prediction.update({
            "schema_version": "week32_deepseek_remaining_prediction_v1",
            "input_alias": input_alias,
            "mapped_sample_aliases": group["mapped_sample_aliases"],
        })
        atomic_json(path, prediction)
        completed.append(input_alias)
        atomic_json(output / "checkpoint.json", {
            "schema_version": "week32_deepseek_remaining_checkpoint_v1",
            "completed_inputs": completed,
            "physical_request_count": sum(int(_load(output / "predictions" / f"{item}.json").get("request_count", 0)) for item in completed),
        })
        print(f"input={input_alias} status={prediction['llm_status']} topk={len(prediction.get('final_topk', ())) }", flush=True)
    frozen = _freeze(output, expected)
    print(json.dumps({
        "terminal_input_count": len(frozen["predictions"]),
        "physical_request_count": sum(int(row.get("request_count", 0)) for row in frozen["predictions"]),
        "valid_input_count": sum(bool(row.get("topk_valid")) for row in frozen["predictions"]),
    }, ensure_ascii=False, indent=2))
    return 0


def _hits(topk: list[Mapping[str, Any]], truths: list[Mapping[str, Any]]) -> dict[int, bool]:
    return {k: any(live._function_match(candidate, truth) for candidate in topk[:k] for truth in truths) for k in (1, 3, 5)}


def evaluate(output: Path) -> int:
    manifest = _load(output / "inference_manifest.json")
    frozen = _load(output / "frozen_predictions.json")
    if frozen["payload_sha256"] != live._sha({key: frozen[key] for key in ("schema_version", "inference_manifest_sha256", "predictions")}):
        raise ValueError("WEEK32_FROZEN_PREDICTIONS_INVALID")
    by_input = {row["input_alias"]: row for row in frozen["predictions"]}
    input_for_sample = {alias: group["input_alias"] for group in manifest["unique_inputs"] for alias in group["mapped_sample_aliases"]}
    transactions = {row["sample_alias"]: row["tx_hash"].lower() for row in manifest["samples"]}
    truths = _truths_for_transactions(transactions, ROOT / "dataset", rebuild.replay.DEFAULT_CACHE)
    labels = [{"sample_alias": alias, "truths": rows} for alias, rows in sorted(truths.items())]
    truth_core = {"schema_version": "week32_remaining_evaluation_truth_v1", "labels": labels}
    atomic_json(output / "evaluation_ground_truth.json", {**truth_core, "ground_truth_sha256": live._sha(truth_core)})

    rows = []
    for sample in manifest["samples"]:
        alias = sample["sample_alias"]
        prediction = by_input[input_for_sample[alias]]
        prompt_rows = [row for row in prediction.get("pre_llm_candidates", ()) if row.get("included_in_prompt")]
        prompt_hit = any(live._function_match(candidate, truth) for candidate in prompt_rows for truth in truths[alias])
        topk = sorted(prediction.get("final_topk", ()), key=lambda row: row.get("rank", 999)) if prediction.get("topk_valid") else []
        hits = _hits(topk, truths[alias])
        layer = "HIT" if hits[5] else "LLM_RANKING_MISS" if prompt_hit and prediction.get("topk_valid") else "LLM_INVALID" if prompt_hit else "PROMPT_CANDIDATE_MISS"
        rows.append({
            "sample_alias": alias,
            "input_alias": input_for_sample[alias],
            "status": prediction["llm_status"],
            "prompt_truth_hit": prompt_hit,
            "failure_layer": layer,
            **{f"function_hit_at_{k}": hits[k] for k in (1, 3, 5)},
            "predicted_top5": [item.get("function_signature") for item in topk[:5]],
        })
    by_alias = {row["sample_alias"]: row for row in rows}
    representative_rows = [by_alias[group["representative_alias"]] for group in manifest["unique_inputs"]]
    counts = {k: sum(row[f"function_hit_at_{k}"] for row in rows) for k in (1, 3, 5)}
    distinct_counts = {k: sum(row[f"function_hit_at_{k}"] for row in representative_rows) for k in (1, 3, 5)}
    valid_samples = sum(by_input[input_for_sample[row["sample_alias"]]].get("topk_valid", False) for row in manifest["samples"])
    valid_inputs = sum(row.get("topk_valid", False) for row in frozen["predictions"])
    report = {
        "schema_version": "week32_deepseek_remaining_evaluation_v1",
        "sample_count": EXPECTED_SAMPLES,
        "unique_input_count": EXPECTED_UNIQUE_INPUTS,
        "physical_request_count": sum(int(row.get("request_count", 0)) for row in frozen["predictions"]),
        "valid_transaction_count": valid_samples,
        "valid_unique_input_count": valid_inputs,
        "prompt_function_recall_count": sum(row["prompt_truth_hit"] for row in rows),
        "transaction_function_recall_counts": {f"at_{k}": counts[k] for k in (1, 3, 5)},
        "transaction_function_recall": {f"at_{k}": counts[k] / EXPECTED_SAMPLES for k in (1, 3, 5)},
        "distinct_input_function_recall_counts": {f"at_{k}": distinct_counts[k] for k in (1, 3, 5)},
        "distinct_input_function_recall": {f"at_{k}": distinct_counts[k] / EXPECTED_UNIQUE_INPUTS for k in (1, 3, 5)},
        "valid_subset_transaction_recall": {f"at_{k}": counts[k] / valid_samples if valid_samples else None for k in (1, 3, 5)},
        "failure_layer_counts": dict(sorted(Counter(row["failure_layer"] for row in rows).items())),
        "usage_totals": {key: sum(int(row.get("usage", {}).get(key, 0)) for row in frozen["predictions"] if isinstance(row.get("usage"), Mapping)) for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "external_call_counts": {"rpc": 0, "source": 0, "llm": EXPECTED_UNIQUE_INPUTS},
        "samples": rows,
    }
    atomic_json(output / "evaluation_report.json", report)
    print(json.dumps({key: report[key] for key in (
        "sample_count", "unique_input_count", "physical_request_count",
        "valid_transaction_count", "valid_unique_input_count",
        "prompt_function_recall_count", "transaction_function_recall_counts",
        "transaction_function_recall", "distinct_input_function_recall_counts",
        "distinct_input_function_recall", "failure_layer_counts", "usage_totals",
    )}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate"):
        sub.add_parser(name).add_argument("--output", type=Path, default=OUTPUT)
    inference = sub.add_parser("infer")
    inference.add_argument("--output", type=Path, default=OUTPUT)
    inference.add_argument("--timeout-seconds", type=int, default=180)
    inference.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args.output)
    if args.command == "infer":
        return infer(args.output, args.timeout_seconds, args.resume)
    return evaluate(args.output)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run one frozen official-DeepSeek request for each Week-30 expanded prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_week30_adaptive_capacity_prompts as week30
from scripts import run_localization15_deepseek_official as official
from scripts import run_week14_behavior_similarity as rebuild
from scripts import run_week16_local6_openai_top5 as live
from src.cloc_v2.phase1_llm_client import OpenAICompatibleTransport
from src.cloc_v2.phase2_endpoint_saturation_guard import apply_endpoint_saturation_guard
from src.cloc_v2.phase2_localization_prompt import (
    materialize_phase2_localization_result_topk_v4,
    validate_phase2_localization_topk_v4,
)
from src.cloc_v2.source_cache import atomic_json


OUTPUT = ROOT / "data" / "prototypes" / "week31_deepseek_adaptive_capacity_20260911"
WEEK30_MANIFEST = week30.OUTPUT / "prompt_manifest.json"
WEEK27_REPORT = ROOT / "data" / "prototypes" / "week27_reliability_rerank_20260911" / "evaluation_report.json"
MAX_REQUESTS = 3


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, Any]:
    source = _load(WEEK30_MANIFEST)
    if source["sample_count"] != MAX_REQUESTS or not source["all_within_byte_limit"]:
        raise ValueError("WEEK31_SOURCE_PROMPT_MANIFEST_INVALID")
    return {
        "schema_version": "week31_deepseek_adaptive_capacity_manifest_v1",
        "mode": "FROZEN_EXPANDED_PROMPT_LIVE_LLM",
        "source_prompt_manifest_sha256": _file_sha256(WEEK30_MANIFEST),
        "provider": "DEEPSEEK_OFFICIAL",
        "provider_host": "api.deepseek.com",
        "model": official.MODEL,
        "thinking_mode": official.THINKING_MODE,
        "temperature": live.TEMPERATURE,
        "max_output_tokens": live.MAX_OUTPUT_TOKENS,
        "max_prompt_candidates": week30.MAX_PROMPT_CANDIDATES,
        "max_prompt_bytes": week30.MAX_PROMPT_BYTES,
        "max_requests_per_sample": 1,
        "max_requests_total": MAX_REQUESTS,
        "retry_enabled": False,
        "samples": [
            {
                "sample_alias": row["sample_alias"],
                "prompt_sha256": row["expanded_prompt_sha256"],
                "prompt_bytes": row["expanded_prompt_bytes"],
                "prompt_candidate_count": row["expanded_prompt_candidate_count"],
            }
            for row in source["samples"]
        ],
    }


def prepare(output: Path) -> int:
    manifest = _manifest()
    live._write_manifest_once(output / "inference_manifest.json", manifest)
    print(json.dumps({
        "selected_aliases": [row["sample_alias"] for row in manifest["samples"]],
        "prompt_hashes_frozen": len(manifest["samples"]),
        "provider": manifest["provider"],
        "model": manifest["model"],
        "max_requests_total": manifest["max_requests_total"],
        "retry_enabled": manifest["retry_enabled"],
    }, ensure_ascii=False, indent=2))
    return 0


def _expanded_preparation(
    sample: dict[str, Any],
    trees: Mapping[str, tuple[Path, str]],
    normal_records: tuple[Any, ...],
    cache_index: Any,
) -> Any:
    return rebuild._build_sample(
        sample,
        trees[sample["tx_hash"].lower()][0],
        normal_records,
        cache_index,
        rebuild.replay.DEFAULT_CACHE,
        max_prompt_candidates=week30.MAX_PROMPT_CANDIDATES,
        max_prompt_bytes=week30.MAX_PROMPT_BYTES,
    )[1]


def _apply_frozen_guard(prediction: dict[str, Any], preparation: Any) -> dict[str, Any]:
    prediction = dict(prediction)
    prediction["raw_topk"] = list(prediction.get("topk", ()))
    prediction["final_topk"] = list(prediction.get("topk", ()))
    prediction["endpoint_guard"] = {
        "applied": False,
        "reason_code": "LLM_OUTPUT_NOT_VALID",
        "controller_source_slice_ids": [],
    }
    if not prediction.get("topk_valid") or preparation.prompt is None:
        return prediction
    response = {
        "findings": [
            {
                "rank": row["rank"],
                "source_slice_id": row["source_slice_id"],
                "evidence_ids": list(row["evidence_ids"]),
            }
            for row in sorted(prediction["topk"], key=lambda row: row["rank"])
        ]
    }
    validation = validate_phase2_localization_topk_v4(response, preparation.prompt)
    if not validation.valid:
        raise ValueError("WEEK31_SAVED_TOPK_REVALIDATION_FAILED")
    guard = apply_endpoint_saturation_guard(
        validation,
        preparation.prompt,
        prediction["pre_llm_candidates"],
    )
    guarded_validation = validate_phase2_localization_topk_v4(guard["response"], preparation.prompt)
    if not guarded_validation.valid:
        raise ValueError("WEEK31_GUARDED_TOPK_VALIDATION_FAILED")
    materialized = materialize_phase2_localization_result_topk_v4(
        guarded_validation,
        preparation.prompt,
    )
    prediction["final_topk"] = live._topk_rows(preparation, materialized)
    prediction["endpoint_guard"] = {
        "applied": guard["applied"],
        "reason_code": guard["reason_code"],
        "controller_source_slice_ids": guard["controller_source_slice_ids"],
    }
    return prediction


def infer(output: Path, timeout_seconds: int, resume: bool) -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY") or ""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY_MISSING")
    expected = _manifest()
    manifest_path = output / "inference_manifest.json"
    if not manifest_path.exists() or live._canonical(_load(manifest_path)) != live._canonical(expected):
        raise ValueError("WEEK31_INFERENCE_MANIFEST_MISSING_OR_DRIFTED")

    samples, trees, normal_records, cache_index = official._context()
    samples_by_alias = {row["sample_alias"]: row for row in samples}
    completed = []
    transport = OpenAICompatibleTransport(official.DEEPSEEK_BASE_URL)
    for selected in expected["samples"]:
        alias = selected["sample_alias"]
        prediction_path = output / "predictions" / f"{alias}.json"
        if prediction_path.exists():
            if not resume:
                raise ValueError("WEEK31_PREDICTIONS_EXIST_USE_RESUME")
            completed.append(alias)
            print(f"sample={alias} status=RESUMED", flush=True)
            continue
        sample = samples_by_alias[alias]
        preparation = _expanded_preparation(sample, trees, normal_records, cache_index)
        if preparation.prompt is None or preparation.prompt_sha256 != selected["prompt_sha256"]:
            raise ValueError(f"WEEK31_PROMPT_DRIFT:{alias}")
        print(f"sample={alias} status=RUNNING", flush=True)
        prediction = live._infer_one(
            sample,
            trees[sample["tx_hash"].lower()][0],
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
            official.MODEL,
            transport,
            api_key,
            timeout_seconds,
            official.THINKING_MODE,
            max_prompt_candidates=week30.MAX_PROMPT_CANDIDATES,
            max_prompt_bytes=week30.MAX_PROMPT_BYTES,
        )
        if prediction.get("prompt_sha256") != selected["prompt_sha256"]:
            raise ValueError(f"WEEK31_INFERENCE_PROMPT_DRIFT:{alias}")
        prediction = _apply_frozen_guard(prediction, preparation)
        prediction["schema_version"] = "week31_deepseek_adaptive_capacity_prediction_v1"
        atomic_json(prediction_path, prediction)
        completed.append(alias)
        atomic_json(output / "checkpoint.json", {
            "schema_version": "week31_deepseek_adaptive_capacity_checkpoint_v1",
            "completed_aliases": completed,
            "request_count": sum(
                int(_load(output / "predictions" / f"{done}.json").get("request_count", 0))
                for done in completed
            ),
        })
        print(
            f"sample={alias} status={prediction['llm_status']} "
            f"topk={len(prediction['final_topk'])} guard={prediction['endpoint_guard']['applied']}",
            flush=True,
        )
    return 0


def _hits(topk: list[Mapping[str, Any]], truths: list[Mapping[str, Any]]) -> dict[int, bool]:
    return {
        k: any(live._function_match(candidate, truth) for candidate in topk[:k] for truth in truths)
        for k in (1, 3, 5)
    }


def evaluate(output: Path) -> int:
    manifest = _load(output / "inference_manifest.json")
    truth_by_alias = {
        row["sample_alias"]: row["truths"]
        for row in _load(live.DEFAULT_TRUTH)["labels"]
    }
    current = _load(WEEK27_REPORT)
    rows = []
    for selected in manifest["samples"]:
        alias = selected["sample_alias"]
        prediction = _load(output / "predictions" / f"{alias}.json")
        raw_topk = sorted(prediction.get("raw_topk", ()), key=lambda row: row.get("rank", 999))
        final_topk = sorted(prediction.get("final_topk", ()), key=lambda row: row.get("rank", 999))
        raw_hits = _hits(raw_topk, truth_by_alias[alias]) if prediction.get("topk_valid") else {1: False, 3: False, 5: False}
        final_hits = _hits(final_topk, truth_by_alias[alias]) if prediction.get("topk_valid") else {1: False, 3: False, 5: False}
        rows.append({
            "sample_alias": alias,
            "status": prediction["llm_status"],
            "topk_valid": prediction["topk_valid"],
            "endpoint_guard_applied": prediction["endpoint_guard"]["applied"],
            **{f"raw_hit_at_{k}": raw_hits[k] for k in (1, 3, 5)},
            **{f"final_hit_at_{k}": final_hits[k] for k in (1, 3, 5)},
            "raw_top5": [row.get("function_signature") for row in raw_topk],
            "final_top5": [row.get("function_signature") for row in final_topk],
        })

    selected_aliases = {row["sample_alias"] for row in rows}
    combined_counts = {}
    for k in (1, 3, 5):
        retained = sum(
            row[f"reranked_hit_at_{k}"]
            for row in current["samples"]
            if row["sample_alias"] not in selected_aliases
        )
        combined_counts[f"at_{k}"] = retained + sum(row[f"final_hit_at_{k}"] for row in rows)
    valid_unselected = sum(
        row["valid_top5"]
        for row in current["samples"]
        if row["sample_alias"] not in selected_aliases
    )
    valid_count = valid_unselected + sum(row["topk_valid"] for row in rows)
    report = {
        "schema_version": "week31_deepseek_adaptive_capacity_evaluation_v1",
        "scientific_generalization_result": False,
        "selected_sample_count": len(rows),
        "request_count": sum(_load(output / "predictions" / f"{row['sample_alias']}.json")["request_count"] for row in rows),
        "valid_expanded_top5_count": sum(row["topk_valid"] for row in rows),
        "expanded_prompt_hit_count": 3,
        "raw_expanded_function_recall_counts": {
            f"at_{k}": sum(row[f"raw_hit_at_{k}"] for row in rows)
            for k in (1, 3, 5)
        },
        "final_expanded_function_recall_counts": {
            f"at_{k}": sum(row[f"final_hit_at_{k}"] for row in rows)
            for k in (1, 3, 5)
        },
        "current_week27_fixed_counts": current["reranked_function_recall_counts"],
        "combined_fixed_denominator": 15,
        "combined_function_recall_counts": combined_counts,
        "combined_function_recall": {key: f"{value}/15" for key, value in combined_counts.items()},
        "combined_valid_top5_denominator": valid_count,
        "combined_valid_subset_function_recall": {
            key: f"{value}/{valid_count}" for key, value in combined_counts.items()
        },
        "target_70_percent_reached": combined_counts["at_5"] >= 11,
        "external_call_counts": {"rpc": 0, "source": 0, "llm": len(rows)},
        "samples": rows,
    }
    atomic_json(output / "evaluation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare").add_argument("--output", type=Path, default=OUTPUT)
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--output", type=Path, default=OUTPUT)
    infer_parser.add_argument("--timeout-seconds", type=int, default=180)
    infer_parser.add_argument("--resume", action="store_true")
    subparsers.add_parser("evaluate").add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args.output)
    if args.command == "infer":
        return infer(args.output, args.timeout_seconds, args.resume)
    return evaluate(args.output)


if __name__ == "__main__":
    raise SystemExit(main())

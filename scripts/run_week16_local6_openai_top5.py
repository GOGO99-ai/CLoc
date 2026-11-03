"""Blind local-trace inference and separate evaluation for six Phase-2 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_week14_behavior_similarity as rebuild
from src.cloc_v2.phase1_llm_client import OpenAICompatibleTransport, TransportError
from src.cloc_v2.phase2_localization_prompt import (
    materialize_phase2_localization_result_topk_v4,
    validate_phase2_localization_topk_v4,
)
from src.cloc_v2.source_cache import atomic_json


DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "week16_local6_openai_top5_v2_20260911"
DEFAULT_TRUTH = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "evaluation_ground_truth.json"
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_samples() -> tuple[list[dict[str, Any]], dict[str, Path]]:
    manifest = rebuild.replay._load(rebuild.replay.DEFAULT_MANIFEST)
    trees = {path.stem.lower(): path for path in rebuild.replay.DEFAULT_TREES.rglob("*.json")}
    samples = [sample for sample in manifest["samples"] if sample["tx_hash"].lower() in trees]
    return samples, trees


def _inference_manifest(samples: list[dict[str, Any]], model: str, base_url: str, thinking_mode: str) -> dict[str, Any]:
    return {
        "schema_version": "week16_local6_inference_manifest_v1",
        "mode": "BLIND_LOCAL_TRACE_LIVE_LLM",
        "sample_count": len(samples),
        "samples": [
            {"sample_alias": sample["sample_alias"], "order": sample["order"], "tx_hash": sample["tx_hash"]}
            for sample in samples
        ],
        "provider": "OPENAI_COMPATIBLE",
        "provider_host": urlparse(base_url).netloc,
        "model": model,
        "thinking_mode": thinking_mode,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_requests_per_sample": 1,
    }


def _write_manifest_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() and _canonical(_load(path)) != _canonical(value):
        raise ValueError("WEEK16_INFERENCE_MANIFEST_DRIFT")
    atomic_json(path, value)


def _provider_content(response: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return None, ["PROVIDER_CHOICES_INVALID"]
    if choices[0].get("finish_reason") != "stop":
        return None, ["PROVIDER_FINISH_REASON_INVALID"]
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        return None, ["PROVIDER_CONTENT_INVALID"]
    return message["content"], []


def _topk_rows(preparation: Any, materialized: Mapping[str, Any]) -> list[dict[str, Any]]:
    locations = preparation.source_locations
    rows = []
    for finding in materialized.get("findings", ()):
        location = locations[finding["source_slice_id"]]
        row = {
            "rank": finding["rank"],
            "source_slice_id": finding["source_slice_id"],
            "relative_file": location["relative_file"],
            "function_signature": location["function"],
            "slice_start_line": location["start_line"],
            "slice_end_line": location["end_line"],
            "evidence_ids": list(finding["evidence_ids"]),
            "line_range_source": finding.get("line_range_source"),
        }
        if "predicted_line_start" in finding:
            row["predicted_line_start"] = finding["predicted_line_start"]
            row["predicted_line_end"] = finding["predicted_line_end"]
        rows.append(row)
    return rows


def _infer_one(
    sample: dict[str, Any],
    tree: Path,
    normal_records: tuple[Any, ...],
    cache_index: Any,
    cache_root: Path,
    model: str,
    transport: OpenAICompatibleTransport,
    api_key: str,
    timeout_seconds: int,
    thinking_mode: str,
    *,
    max_prompt_candidates: int = 20,
    max_prompt_bytes: int | None = None,
) -> dict[str, Any]:
    _slices, preparation = rebuild._build_sample(
        sample,
        tree,
        normal_records,
        cache_index,
        cache_root,
        max_prompt_candidates=max_prompt_candidates,
        max_prompt_bytes=max_prompt_bytes,
    )
    base = {
        "schema_version": "week16_local6_prediction_v1",
        "sample_alias": sample["sample_alias"],
        "preparation_status": preparation.preparation_status,
        "prompt_sha256": preparation.prompt_sha256,
        "prompt_bytes": preparation.prompt_bytes,
        "pre_llm_candidate_count": preparation.pre_llm_candidate_count,
        "prompt_candidate_count": preparation.prompt_candidate_count,
        "pre_llm_candidates": preparation.pre_llm_candidates,
        "model": model,
        "request_count": 0,
        "llm_status": "NOT_CALLED",
        "reason_codes": [issue.code for issue in preparation.issues],
        "usage": None,
        "topk": [],
        "topk_valid": False,
    }
    if preparation.preparation_status != "READY_FOR_LLM" or preparation.prompt is None:
        return base
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": preparation.prompt.system_message},
            {"role": "user", "content": preparation.prompt.user_message},
        ],
        "parameters": {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "omit_temperature": False,
            "response_format": {"type": "json_object"},
        },
    }
    if thinking_mode != "provider_default":
        request["parameters"]["thinking"] = {"type": thinking_mode}
    base["request_sha256"] = _sha(request)
    base["request_count"] = 1
    try:
        response = transport.send(request, api_key, timeout_seconds)
    except TransportError as error:
        base["llm_status"] = "TRANSPORT_ERROR"
        base["reason_codes"] = sorted(set(base["reason_codes"] + [f"TRANSPORT_{error.kind.upper()}"]))
        return base
    content, provider_issues = _provider_content(response)
    base["provider_reported_model"] = response.get("model") if isinstance(response.get("model"), str) else None
    base["usage"] = dict(response["usage"]) if isinstance(response.get("usage"), Mapping) else None
    if content is None:
        base["llm_status"] = "INVALID_RESPONSE"
        base["reason_codes"] = sorted(set(base["reason_codes"] + provider_issues))
        return base
    try:
        document = json.loads(content)
    except (json.JSONDecodeError, UnicodeError):
        base["llm_status"] = "INVALID_RESPONSE"
        base["reason_codes"] = sorted(set(base["reason_codes"] + ["RESPONSE_JSON_INVALID"]))
        return base
    validation = validate_phase2_localization_topk_v4(document, preparation.prompt)
    if not validation.valid:
        base["llm_status"] = "INVALID_RESPONSE"
        base["reason_codes"] = sorted(set(base["reason_codes"] + list(validation.reason_codes)))
        return base
    materialized = materialize_phase2_localization_result_topk_v4(validation, preparation.prompt)
    base["llm_status"] = "VALIDATED"
    base["topk"] = _topk_rows(preparation, materialized)
    base["topk_valid"] = True
    return base


def infer(output: Path, limit: int | None, resume: bool, timeout_seconds: int, thinking_mode: str) -> int:
    api_key = os.getenv("OPENAI_API_KEY") or ""
    base_url = os.getenv("OPENAI_BASE_URL") or ""
    model = os.getenv("OPENAI_MODEL") or ""
    if not api_key or not base_url or not model:
        raise ValueError("WEEK16_OPENAI_CONFIGURATION_MISSING")
    samples, trees = _selected_samples()
    _write_manifest_once(output / "inference_manifest.json", _inference_manifest(samples, model, base_url, thinking_mode))
    selected = samples[:limit] if limit is not None else samples
    normal_records = rebuild.replay._load_normal_records(rebuild.replay.DEFAULT_NORMAL)
    cache_index = rebuild.replay.build_verified_source_cache_index(rebuild.replay.DEFAULT_CACHE, blockchain_chain_id=1, chain_key="eth")
    prediction_dir = output / "predictions"
    completed = []
    for sample in selected:
        path = prediction_dir / f"{sample['sample_alias']}.json"
        if resume and path.exists():
            completed.append(sample["sample_alias"])
            print(f"sample={sample['sample_alias']} status=RESUMED")
            continue
        print(f"sample={sample['sample_alias']} status=RUNNING", flush=True)
        prediction = _infer_one(
            sample,
            trees[sample["tx_hash"].lower()],
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
            model,
            OpenAICompatibleTransport(base_url),
            api_key,
            timeout_seconds,
            thinking_mode,
        )
        atomic_json(path, prediction)
        completed.append(sample["sample_alias"])
        print(f"sample={sample['sample_alias']} status={prediction['llm_status']} topk={len(prediction['topk'])}", flush=True)
        atomic_json(output / "checkpoint.json", {
            "schema_version": "week16_local6_checkpoint_v1",
            "completed_aliases": completed,
            "request_count": sum((_load(prediction_dir / f"{alias}.json").get("request_count", 0) for alias in completed), 0),
        })
    return 0


def _file_match(predicted: Any, truth: str) -> bool:
    if not isinstance(predicted, str):
        return False
    value = PurePosixPath(predicted.replace("\\", "/")).as_posix().lstrip("./")
    return value == truth or value.endswith("/" + truth) or truth.endswith("/" + value)


def _function_match(row: Mapping[str, Any], truth: Mapping[str, Any]) -> bool:
    return _file_match(row.get("relative_file"), str(truth["relative_file"])) and row.get("function_signature") == truth.get("function_signature")


def evaluate(output: Path, truth_path: Path) -> int:
    manifest = _load(output / "inference_manifest.json")
    truth_document = _load(truth_path)
    truth_by_alias = {row["sample_alias"]: row["truths"] for row in truth_document["labels"]}
    rows = []
    for sample in manifest["samples"]:
        alias = sample["sample_alias"]
        path = output / "predictions" / f"{alias}.json"
        if not path.exists():
            rows.append({"sample_alias": alias, "status": "NOT_RUN", "failure_layer": "LLM_INVALID", "prompt_truth_hit": False, "function_hit_at_1": False, "function_hit_at_3": False, "function_hit_at_5": False})
            continue
        prediction = _load(path)
        truths = truth_by_alias[alias]
        prompt = [candidate for candidate in prediction.get("pre_llm_candidates", ()) if candidate.get("included_in_prompt")]
        prompt_hit = any(_function_match(candidate, truth) for candidate in prompt for truth in truths)
        topk = sorted(prediction.get("topk", ()), key=lambda row: row.get("rank", 999)) if prediction.get("topk_valid") else []
        hits = {k: any(_function_match(candidate, truth) for candidate in topk[:k] for truth in truths) for k in (1, 3, 5)}
        if not prompt_hit:
            layer = "PROMPT_CANDIDATE_MISS"
        elif not prediction.get("topk_valid"):
            layer = "LLM_INVALID"
        elif not hits[5]:
            layer = "LLM_RANKING_MISS"
        else:
            layer = "HIT"
        rows.append({
            "sample_alias": alias,
            "status": prediction.get("llm_status"),
            "failure_layer": layer,
            "prompt_truth_hit": prompt_hit,
            "function_hit_at_1": hits[1],
            "function_hit_at_3": hits[3],
            "function_hit_at_5": hits[5],
            "predicted_top5": [row.get("function_signature") for row in topk[:5]],
        })
    usage = [
        _load(output / "predictions" / f"{sample['sample_alias']}.json").get("usage")
        for sample in manifest["samples"]
        if (output / "predictions" / f"{sample['sample_alias']}.json").exists()
    ]
    report = {
        "schema_version": "week16_local6_openai_top5_evaluation_v1",
        "scientific_generalization_result": False,
        "sample_count": len(rows),
        "model": manifest["model"],
        "valid_top5_count": sum(row["status"] == "VALIDATED" for row in rows),
        "prompt_function_recall_count": sum(row["prompt_truth_hit"] for row in rows),
        "function_recall_at_1_count": sum(row["function_hit_at_1"] for row in rows),
        "function_recall_at_3_count": sum(row["function_hit_at_3"] for row in rows),
        "function_recall_at_5_count": sum(row["function_hit_at_5"] for row in rows),
        "failure_layer_counts": dict(sorted({layer: sum(row["failure_layer"] == layer for row in rows) for layer in {row["failure_layer"] for row in rows}}.items())),
        "request_count": sum((_load(output / "predictions" / f"{sample['sample_alias']}.json").get("request_count", 0) for sample in manifest["samples"] if (output / "predictions" / f"{sample['sample_alias']}.json").exists()), 0),
        "usage_totals": {
            key: sum(int(item.get(key, 0)) for item in usage if isinstance(item, Mapping))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "samples": rows,
    }
    atomic_json(output / "evaluation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    infer_parser.add_argument("--limit", type=int)
    infer_parser.add_argument("--resume", action="store_true")
    infer_parser.add_argument("--timeout-seconds", type=int, default=120)
    infer_parser.add_argument("--thinking-mode", choices=("provider_default", "disabled"), default="provider_default")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    evaluate_parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    args = parser.parse_args(argv)
    if args.command == "infer":
        if args.limit is not None and not 1 <= args.limit <= 6:
            raise ValueError("WEEK16_LIMIT_INVALID")
        return infer(args.output, args.limit, args.resume, args.timeout_seconds, args.thinking_mode)
    return evaluate(args.output, args.truth)


if __name__ == "__main__":
    raise SystemExit(main())

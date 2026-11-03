"""Retry only the four transport failures from the frozen DeepSeek-15 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_localization15_deepseek_official as official
from scripts import run_week14_behavior_similarity as rebuild
from scripts import run_week16_local6_openai_top5 as live
from src.cloc_v2.phase1_llm_client import OpenAICompatibleTransport
from src.cloc_v2.source_cache import atomic_json


OUTPUT = ROOT / "data" / "prototypes" / "week26_deepseek_transport_retry_20260911"
ORIGINAL = official.OUTPUT
MAX_ADDITIONAL_ATTEMPTS = 2
EXPECTED_ALIASES = (
    "localization_v4_eth_006",
    "localization_v4_eth_007",
    "localization_v4_eth_009",
    "localization_v4_eth_010",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retry_manifest() -> dict[str, Any]:
    source_manifest = _load(ORIGINAL / "inference_manifest.json")
    source_report = _load(ORIGINAL / "evaluation_report.json")
    report_by_alias = {row["sample_alias"]: row for row in source_report["samples"]}
    selected = tuple(
        row["sample_alias"]
        for row in source_report["samples"]
        if row["status"] == "TRANSPORT_ERROR"
    )
    if selected != EXPECTED_ALIASES:
        raise ValueError(f"WEEK26_TRANSPORT_FAILURE_SET_DRIFT:{selected}")

    samples, trees, normal_records, cache_index = official._context()
    source_by_alias = {row["sample_alias"]: row for row in source_manifest["samples"]}
    sample_by_alias = {row["sample_alias"]: row for row in samples}
    rows = []
    for alias in selected:
        sample = sample_by_alias[alias]
        _slices, preparation = rebuild._build_sample(
            sample,
            trees[sample["tx_hash"].lower()][0],
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
        )
        expected_hash = source_by_alias[alias]["prompt_sha256"]
        if preparation.prompt is None or preparation.prompt_sha256 != expected_hash:
            raise ValueError(f"WEEK26_PROMPT_DRIFT:{alias}")
        rows.append({
            "sample_alias": alias,
            "order": sample["order"],
            "tx_hash": sample["tx_hash"],
            "source_status": report_by_alias[alias]["status"],
            "prompt_sha256": expected_hash,
            "prompt_bytes": preparation.prompt_bytes,
        })
    return {
        "schema_version": "week26_deepseek_transport_retry_manifest_v1",
        "mode": "FROZEN_TRANSPORT_RELIABILITY_FOLLOWUP",
        "source_experiment": str(ORIGINAL.relative_to(ROOT)).replace("\\", "/"),
        "source_manifest_sha256": _sha256(ORIGINAL / "inference_manifest.json"),
        "source_report_sha256": _sha256(ORIGINAL / "evaluation_report.json"),
        "selection_rule": "SOURCE_STATUS_EQUALS_TRANSPORT_ERROR",
        "retry_rule": "RETRY_ONCE_ONLY_IF_FIRST_ADDITIONAL_ATTEMPT_IS_TRANSPORT_ERROR",
        "stop_on_statuses": ["VALIDATED", "INVALID_RESPONSE", "NOT_CALLED"],
        "provider": "DEEPSEEK_OFFICIAL",
        "provider_host": "api.deepseek.com",
        "model": official.MODEL,
        "thinking_mode": official.THINKING_MODE,
        "temperature": live.TEMPERATURE,
        "max_output_tokens": live.MAX_OUTPUT_TOKENS,
        "max_additional_attempts_per_sample": MAX_ADDITIONAL_ATTEMPTS,
        "max_additional_requests_total": len(rows) * MAX_ADDITIONAL_ATTEMPTS,
        "samples": rows,
    }


def prepare(output: Path) -> int:
    manifest = _retry_manifest()
    live._write_manifest_once(output / "retry_manifest.json", manifest)
    print(json.dumps({
        "selected_aliases": [row["sample_alias"] for row in manifest["samples"]],
        "prompt_hashes_verified": len(manifest["samples"]),
        "max_additional_attempts_per_sample": MAX_ADDITIONAL_ATTEMPTS,
        "retry_rule": manifest["retry_rule"],
    }, ensure_ascii=False, indent=2))
    return 0


def infer(output: Path, timeout_seconds: int, resume: bool) -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY") or ""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY_MISSING")
    expected = _retry_manifest()
    manifest_path = output / "retry_manifest.json"
    if not manifest_path.exists() or live._canonical(_load(manifest_path)) != live._canonical(expected):
        raise ValueError("WEEK26_RETRY_MANIFEST_MISSING_OR_DRIFTED")

    samples, trees, normal_records, cache_index = official._context()
    sample_by_alias = {row["sample_alias"]: row for row in samples}
    completed = []
    for selected in expected["samples"]:
        alias = selected["sample_alias"]
        final_path = output / "retry_predictions" / f"{alias}.json"
        if final_path.exists():
            if not resume:
                raise ValueError("WEEK26_RETRY_PREDICTIONS_EXIST_USE_RESUME")
            completed.append(alias)
            print(f"sample={alias} status=RESUMED", flush=True)
            continue
        attempts = []
        sample = sample_by_alias[alias]
        for attempt_number in range(1, MAX_ADDITIONAL_ATTEMPTS + 1):
            print(f"sample={alias} attempt={attempt_number} status=RUNNING", flush=True)
            prediction = live._infer_one(
                sample,
                trees[sample["tx_hash"].lower()][0],
                normal_records,
                cache_index,
                rebuild.replay.DEFAULT_CACHE,
                official.MODEL,
                OpenAICompatibleTransport(official.DEEPSEEK_BASE_URL),
                api_key,
                timeout_seconds,
                official.THINKING_MODE,
            )
            if prediction["prompt_sha256"] != selected["prompt_sha256"]:
                raise ValueError(f"WEEK26_INFERENCE_PROMPT_DRIFT:{alias}")
            prediction["additional_attempt_number"] = attempt_number
            atomic_json(output / "attempts" / f"{alias}_attempt_{attempt_number}.json", prediction)
            attempts.append(prediction)
            print(f"sample={alias} attempt={attempt_number} status={prediction['llm_status']}", flush=True)
            if prediction["llm_status"] != "TRANSPORT_ERROR":
                break
        final = dict(attempts[-1])
        final["request_count"] = len(attempts)
        final["retry_audit"] = {
            "additional_request_count": len(attempts),
            "attempt_statuses": [row["llm_status"] for row in attempts],
            "stopped_after_non_transport_result": attempts[-1]["llm_status"] != "TRANSPORT_ERROR",
        }
        atomic_json(final_path, final)
        completed.append(alias)
        atomic_json(output / "checkpoint.json", {
            "schema_version": "week26_deepseek_transport_retry_checkpoint_v1",
            "completed_aliases": completed,
            "additional_request_count": sum(
                _load(output / "retry_predictions" / f"{done}.json")["request_count"]
                for done in completed
            ),
        })
    return 0


def evaluate(output: Path) -> int:
    expected = _retry_manifest()
    source_manifest = _load(ORIGINAL / "inference_manifest.json")
    combined = output / "combined"
    atomic_json(combined / "inference_manifest.json", {
        **source_manifest,
        "schema_version": "week26_deepseek_transport_retry_combined_manifest_v1",
        "mode": "ORIGINAL_RESULTS_PLUS_FROZEN_TRANSPORT_RETRY",
    })
    retried = {row["sample_alias"] for row in expected["samples"]}
    for sample in source_manifest["samples"]:
        alias = sample["sample_alias"]
        source = (
            output / "retry_predictions" / f"{alias}.json"
            if alias in retried
            else ORIGINAL / "predictions" / f"{alias}.json"
        )
        if not source.exists():
            raise ValueError(f"WEEK26_PREDICTION_MISSING:{alias}")
        atomic_json(combined / "predictions" / f"{alias}.json", _load(source))
    live.evaluate(combined, live.DEFAULT_TRUTH)
    report = _load(combined / "evaluation_report.json")
    source_report = _load(ORIGINAL / "evaluation_report.json")
    additional_requests = sum(
        _load(output / "retry_predictions" / f"{alias}.json")["request_count"]
        for alias in retried
    )
    report.update({
        "schema_version": "week26_deepseek_transport_retry_evaluation_v1",
        "fixed_denominator": 15,
        "source_official_recall_counts": {
            "at_1": source_report["function_recall_at_1_count"],
            "at_3": source_report["function_recall_at_3_count"],
            "at_5": source_report["function_recall_at_5_count"],
        },
        "reliability_followup_recall_counts": {
            "at_1": report["function_recall_at_1_count"],
            "at_3": report["function_recall_at_3_count"],
            "at_5": report["function_recall_at_5_count"],
        },
        "additional_request_count": additional_requests,
        "cumulative_request_count": source_report["request_count"] + additional_requests,
        "remaining_transport_error_count": sum(row["status"] == "TRANSPORT_ERROR" for row in report["samples"]),
        "not_called_count": sum(row["status"] == "NOT_CALLED" for row in report["samples"]),
    })
    atomic_json(output / "evaluation_report.json", report)
    print(json.dumps({key: report[key] for key in (
        "valid_top5_count",
        "source_official_recall_counts",
        "reliability_followup_recall_counts",
        "additional_request_count",
        "cumulative_request_count",
        "remaining_transport_error_count",
        "not_called_count",
    )}, ensure_ascii=False, indent=2))
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

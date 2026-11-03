"""Run the frozen 15-sample localization batch against official DeepSeek."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_localization15_offline_rebuild as offline
from scripts import run_week14_behavior_similarity as rebuild
from scripts import run_week16_local6_openai_top5 as live
from src.cloc_v2.phase1_llm_client import OpenAICompatibleTransport
from src.cloc_v2.source_cache import atomic_json


OUTPUT = ROOT / "data" / "prototypes" / "localization15_deepseek_official_20260911"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
THINKING_MODE = "disabled"
MAX_REQUESTS = 13


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _context() -> tuple[list[dict[str, Any]], dict[str, tuple[Path, str]], tuple[Any, ...], Any]:
    samples = _load(offline.MANIFEST)["samples"]
    trees = offline._tree_index()
    normal_records = rebuild.replay._load_normal_records(rebuild.replay.DEFAULT_NORMAL)
    cache_index = rebuild.replay.build_verified_source_cache_index(
        rebuild.replay.DEFAULT_CACHE,
        blockchain_chain_id=1,
        chain_key="eth",
    )
    return samples, trees, normal_records, cache_index


def _prepare_manifest() -> dict[str, Any]:
    samples, trees, normal_records, cache_index = _context()
    rows = []
    for sample in samples:
        tree_path, tree_source = trees[sample["tx_hash"].lower()]
        _slices, preparation = rebuild._build_sample(
            sample,
            tree_path,
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
        )
        rows.append({
            "sample_alias": sample["sample_alias"],
            "order": sample["order"],
            "tx_hash": sample["tx_hash"],
            "tree_source": tree_source,
            "preparation_status": preparation.preparation_status,
            "prompt_sha256": preparation.prompt_sha256,
            "prompt_bytes": preparation.prompt_bytes,
            "pre_llm_candidate_count": preparation.pre_llm_candidate_count,
            "prompt_candidate_count": preparation.prompt_candidate_count,
        })
    ready = sum(row["preparation_status"] == "READY_FOR_LLM" for row in rows)
    if len(rows) != 15 or ready != MAX_REQUESTS:
        raise ValueError(f"LOCALIZATION15_PREPARATION_DRIFT:samples={len(rows)}:ready={ready}")
    return {
        "schema_version": "localization15_deepseek_inference_manifest_v1",
        "mode": "BLIND_LOCAL_TRACE_LIVE_LLM",
        "sample_count": len(rows),
        "ready_for_llm_count": ready,
        "samples": rows,
        "provider": "DEEPSEEK_OFFICIAL",
        "provider_host": "api.deepseek.com",
        "model": MODEL,
        "thinking_mode": THINKING_MODE,
        "temperature": live.TEMPERATURE,
        "max_output_tokens": live.MAX_OUTPUT_TOKENS,
        "max_requests_per_sample": 1,
        "max_requests_total": MAX_REQUESTS,
        "retry_enabled": False,
    }


def prepare(output: Path) -> int:
    manifest = _prepare_manifest()
    live._write_manifest_once(output / "inference_manifest.json", manifest)
    print(json.dumps({
        "sample_count": manifest["sample_count"],
        "ready_for_llm_count": manifest["ready_for_llm_count"],
        "distinct_prompt_count": len({
            row["prompt_sha256"] for row in manifest["samples"] if row["prompt_sha256"]
        }),
        "model": manifest["model"],
        "thinking_mode": manifest["thinking_mode"],
        "retry_enabled": manifest["retry_enabled"],
    }, ensure_ascii=False, indent=2))
    return 0


def infer(output: Path, timeout_seconds: int, resume: bool) -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY") or ""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY_MISSING")
    expected = _prepare_manifest()
    manifest_path = output / "inference_manifest.json"
    if not manifest_path.exists():
        raise ValueError("RUN_PREPARE_BEFORE_INFER")
    if live._canonical(_load(manifest_path)) != live._canonical(expected):
        raise ValueError("LOCALIZATION15_INFERENCE_MANIFEST_DRIFT")

    samples, trees, normal_records, cache_index = _context()
    prediction_dir = output / "predictions"
    if prediction_dir.exists() and any(prediction_dir.glob("*.json")) and not resume:
        raise ValueError("PREDICTIONS_EXIST_USE_RESUME")

    completed: list[str] = []
    transport = OpenAICompatibleTransport(DEEPSEEK_BASE_URL)
    for sample in samples:
        alias = sample["sample_alias"]
        path = prediction_dir / f"{alias}.json"
        if resume and path.exists():
            completed.append(alias)
            print(f"sample={alias} status=RESUMED", flush=True)
            continue
        print(f"sample={alias} status=RUNNING", flush=True)
        prediction = live._infer_one(
            sample,
            trees[sample["tx_hash"].lower()][0],
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
            MODEL,
            transport,
            api_key,
            timeout_seconds,
            THINKING_MODE,
        )
        atomic_json(path, prediction)
        completed.append(alias)
        print(
            f"sample={alias} status={prediction['llm_status']} topk={len(prediction['topk'])}",
            flush=True,
        )
        request_count = sum(
            int(_load(prediction_dir / f"{done}.json").get("request_count", 0))
            for done in completed
        )
        if request_count > MAX_REQUESTS:
            raise ValueError("DEEPSEEK_REQUEST_LIMIT_EXCEEDED")
        atomic_json(output / "checkpoint.json", {
            "schema_version": "localization15_deepseek_checkpoint_v1",
            "completed_aliases": completed,
            "request_count": request_count,
        })
    return 0


def evaluate(output: Path) -> int:
    live.evaluate(output, live.DEFAULT_TRUTH)
    report_path = output / "evaluation_report.json"
    report = _load(report_path)
    report["schema_version"] = "localization15_deepseek_official_evaluation_v1"
    report["fixed_denominator"] = 15
    report["valid_response_rate"] = f"{report['valid_top5_count']}/15"
    report["function_recall_at_1"] = f"{report['function_recall_at_1_count']}/15"
    report["function_recall_at_3"] = f"{report['function_recall_at_3_count']}/15"
    report["function_recall_at_5"] = f"{report['function_recall_at_5_count']}/15"
    report["transport_error_count"] = sum(
        row["status"] == "TRANSPORT_ERROR" for row in report["samples"]
    )
    report["not_called_count"] = sum(row["status"] == "NOT_CALLED" for row in report["samples"])
    atomic_json(report_path, report)
    print(json.dumps({key: report[key] for key in (
        "sample_count",
        "request_count",
        "valid_response_rate",
        "function_recall_at_1",
        "function_recall_at_3",
        "function_recall_at_5",
        "transport_error_count",
        "not_called_count",
    )}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "evaluate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output", type=Path, default=OUTPUT)
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--output", type=Path, default=OUTPUT)
    infer_parser.add_argument("--timeout-seconds", type=int, default=180)
    infer_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args.output)
    if args.command == "infer":
        return infer(args.output, args.timeout_seconds, args.resume)
    return evaluate(args.output)


if __name__ == "__main__":
    raise SystemExit(main())

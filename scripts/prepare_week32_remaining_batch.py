"""Build and freeze prompts for the source-and-trace-ready Week 32 samples."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_localization15_offline_rebuild as offline
from scripts import run_week14_behavior_similarity as rebuild
from src.cloc_v2.source_cache import atomic_json


SCREEN_ROOT = ROOT / "data" / "prototypes" / "week32_remaining_dataset_screen_20260911"
TRACE_REPORT = SCREEN_ROOT / "trace_acquisition_report.json"
TRACE_LOCK = ROOT / "data" / "trace_acquisition" / "week32_remaining_dataset_20260911" / "locked_manifest.json"
NEW_TREES = ROOT / "data" / "trace_acquisition" / "week32_remaining_dataset_20260911" / "eth"
OUTPUT = ROOT / "data" / "prototypes" / "week32_remaining_batch_20260911"
BASE_CANDIDATES = 20
EXPANDED_CANDIDATES = 40
EXPANDED_BYTES = 60_000


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_index() -> dict[str, tuple[Path, str]]:
    trees = offline._tree_index()
    trees.update({path.parent.name.lower(): (path, "WEEK32_ACQUIRED_TREE") for path in NEW_TREES.glob("*/call_tree.json")})
    return trees


def run() -> dict[str, Any]:
    ready = set(_load(TRACE_REPORT)["ready_sample_aliases"])
    source_samples = {
        row["sample_alias"]: row
        for row in _load(TRACE_LOCK)["samples"]
        if row["sample_alias"] in ready
    }
    trees = _tree_index()
    normal_records = rebuild.replay._load_normal_records(rebuild.replay.DEFAULT_NORMAL)
    cache_index = rebuild.replay.build_verified_source_cache_index(
        rebuild.replay.DEFAULT_CACHE,
        blockchain_chain_id=1,
        chain_key="eth",
    )
    rows = []
    for order, alias in enumerate(sorted(ready), 1):
        locked = source_samples[alias]
        transaction = locked["transaction_hash"].lower()
        sample = {
            "blockchain_chain_id": 1,
            "order": order,
            "sample_alias": alias,
            "tx_hash": transaction,
        }
        try:
            base = rebuild._build_sample(
                sample,
                trees[transaction][0],
                normal_records,
                cache_index,
                rebuild.replay.DEFAULT_CACHE,
            )[1]
            expanded = base
            mode = "BASE_20"
            if base.pre_llm_candidate_count > BASE_CANDIDATES:
                expanded = rebuild._build_sample(
                    sample,
                    trees[transaction][0],
                    normal_records,
                    cache_index,
                    rebuild.replay.DEFAULT_CACHE,
                    max_prompt_candidates=EXPANDED_CANDIDATES,
                    max_prompt_bytes=EXPANDED_BYTES,
                )[1]
                mode = "ADAPTIVE_40"
        except (TypeError, ValueError) as error:
            reason = "CAUSAL_GRAPH_INVARIANT" if "ResultSeed" in str(error) else "PIPELINE_PREPARATION_ERROR"
            rows.append({
                **sample,
                "tree_source": trees[transaction][1],
                "prompt_mode": "NOT_BUILT",
                "preparation_status": "PREPARATION_FAILED",
                "prompt_sha256": None,
                "prompt_bytes": 0,
                "pre_llm_candidate_count": 0,
                "prompt_candidate_count": 0,
                "reason_codes": [reason],
            })
            continue
        if expanded.prompt is not None:
            atomic_json(OUTPUT / "prompts" / f"{alias}.json", expanded.prompt.to_dict())
        rows.append({
            **sample,
            "tree_source": trees[transaction][1],
            "prompt_mode": mode,
            "preparation_status": expanded.preparation_status,
            "prompt_sha256": expanded.prompt_sha256,
            "prompt_bytes": expanded.prompt_bytes,
            "pre_llm_candidate_count": expanded.pre_llm_candidate_count,
            "prompt_candidate_count": expanded.prompt_candidate_count,
            "reason_codes": [issue.code for issue in expanded.issues],
        })
    counts = Counter(row["preparation_status"] for row in rows)
    manifest = {
        "schema_version": "week32_remaining_inference_manifest_v1",
        "mode": "FROZEN_LOCAL_PROMPTS_NO_LLM",
        "selection_rule": "SOURCE_FUNCTION_FILE_LINE_EVALUABLE_AND_LOCAL_TRACE_VALID_EXCLUDING_DEV15",
        "sample_count": len(rows),
        "ready_for_llm_count": counts["READY_FOR_LLM"],
        "preparation_status_counts": dict(sorted(counts.items())),
        "adaptive_prompt_count": sum(row["prompt_mode"] == "ADAPTIVE_40" for row in rows),
        "max_prompt_candidates": {"base": BASE_CANDIDATES, "adaptive": EXPANDED_CANDIDATES},
        "max_prompt_bytes": EXPANDED_BYTES,
        "provider": "DEEPSEEK_OFFICIAL",
        "provider_host": "api.deepseek.com",
        "model": "deepseek-v4-pro",
        "thinking_mode": "disabled",
        "temperature": 0,
        "max_requests_per_sample": 1,
        "retry_enabled": False,
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": rows,
    }
    path = OUTPUT / "inference_manifest.json"
    if path.exists() and _load(path) != manifest:
        raise ValueError("WEEK32_PREPARED_MANIFEST_DRIFT")
    atomic_json(path, manifest)
    public = {
        "schema_version": "week32_remaining_preparation_report_v1",
        "sample_count": len(rows),
        "ready_for_llm_count": manifest["ready_for_llm_count"],
        "preparation_status_counts": manifest["preparation_status_counts"],
        "adaptive_prompt_count": manifest["adaptive_prompt_count"],
        "base_prompt_count": len(rows) - manifest["adaptive_prompt_count"],
        "ready_sample_aliases": [row["sample_alias"] for row in rows if row["preparation_status"] == "READY_FOR_LLM"],
        "not_ready_samples": [
            {"sample_alias": row["sample_alias"], "status": row["preparation_status"], "reason_codes": row["reason_codes"]}
            for row in rows if row["preparation_status"] != "READY_FOR_LLM"
        ],
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
    }
    atomic_json(OUTPUT / "preparation_report.json", public)
    return public


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

"""Rebuild all 15 localization prompts from local traces without calling an LLM."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_week14_behavior_similarity as rebuild
from src.cloc_v2.source_cache import atomic_json


MANIFEST = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "locked_inference_manifest.json"
TRUTH = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "evaluation_ground_truth.json"
OLD_TREES = ROOT / "data" / "cache" / "all_tree"
NEW_TREES = ROOT / "data" / "trace_acquisition" / "localization15_extension_20260911" / "eth"
OUTPUT = ROOT / "data" / "prototypes" / "localization15_offline_rebuild_20260911" / "report.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_index() -> dict[str, tuple[Path, str]]:
    index = {path.stem.lower(): (path, "EXISTING_LOCAL_TREE") for path in OLD_TREES.rglob("*.json")}
    index.update({path.parent.name.lower(): (path, "NEWLY_ACQUIRED_TREE") for path in NEW_TREES.glob("*/call_tree.json")})
    return index


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth_positions(candidates: tuple[Mapping[str, Any], ...], truths: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    pool = [index for index, row in enumerate(candidates, start=1) if rebuild._truth_match(row, truths)]
    prompt = sorted(
        int(str(row["source_slice_id"])[1:])
        for row in candidates
        if row.get("included_in_prompt") and rebuild._truth_match(row, truths)
    )
    return pool, prompt


def run() -> dict[str, Any]:
    manifest = _load(MANIFEST)
    trees = _tree_index()
    samples = manifest["samples"]
    missing = [sample["sample_alias"] for sample in samples if sample["tx_hash"].lower() not in trees]
    if missing:
        raise ValueError(f"LOCAL_TRACE_MISSING:{','.join(missing)}")

    normal_records = rebuild.replay._load_normal_records(rebuild.replay.DEFAULT_NORMAL)
    cache_index = rebuild.replay.build_verified_source_cache_index(
        rebuild.replay.DEFAULT_CACHE,
        blockchain_chain_id=1,
        chain_key="eth",
    )

    # Build every candidate set before loading evaluation truth.
    prepared = []
    for sample in samples:
        tree_path, tree_source = trees[sample["tx_hash"].lower()]
        _slices, preparation = rebuild._build_sample(
            sample,
            tree_path,
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
        )
        prepared.append((sample["sample_alias"], tree_path, tree_source, preparation))

    truth_document = _load(TRUTH)
    truth_by_alias = {row["sample_alias"]: row["truths"] for row in truth_document["labels"]}
    rows = []
    for alias, tree_path, tree_source, preparation in prepared:
        pool_positions, prompt_positions = _truth_positions(preparation.pre_llm_candidates, truth_by_alias[alias])
        rows.append({
            "sample_alias": alias,
            "tree_source": tree_source,
            "tree_sha256": _file_sha256(tree_path),
            "preparation_status": preparation.preparation_status,
            "pre_llm_candidate_count": preparation.pre_llm_candidate_count,
            "prompt_candidate_count": preparation.prompt_candidate_count,
            "prompt_sha256": preparation.prompt_sha256,
            "prompt_bytes": preparation.prompt_bytes,
            "truth_pre_llm_positions": pool_positions,
            "truth_prompt_positions": prompt_positions,
            "pre_llm_truth_hit": bool(pool_positions),
            "prompt_truth_hit": bool(prompt_positions),
            "reason_codes": [issue.code for issue in preparation.issues],
        })

    status_counts = Counter(row["preparation_status"] for row in rows)
    report = {
        "schema_version": "localization15_offline_rebuild_v1",
        "mode": "LOCAL_TRACE_ONLY_NO_LLM",
        "method_frozen": True,
        "scientific_generalization_result": False,
        "sample_count": len(rows),
        "existing_trace_count": sum(row["tree_source"] == "EXISTING_LOCAL_TREE" for row in rows),
        "new_trace_count": sum(row["tree_source"] == "NEWLY_ACQUIRED_TREE" for row in rows),
        "preparation_status_counts": dict(sorted(status_counts.items())),
        "ready_for_llm_count": status_counts.get("READY_FOR_LLM", 0),
        "pre_llm_truth_hit_count": sum(row["pre_llm_truth_hit"] for row in rows),
        "prompt_truth_hit_count": sum(row["prompt_truth_hit"] for row in rows),
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "locked_manifest_sha256": manifest["manifest_sha256"],
        "ground_truth_sha256": truth_document["ground_truth_sha256"],
        "samples": rows,
    }
    atomic_json(OUTPUT, report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "sample_count",
        "existing_trace_count",
        "new_trace_count",
        "preparation_status_counts",
        "pre_llm_truth_hit_count",
        "prompt_truth_hit_count",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

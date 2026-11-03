"""Build three byte-bounded 40-candidate prompts without making LLM calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_localization15_offline_rebuild as offline
from scripts import run_week14_behavior_similarity as rebuild
from src.cloc_v2.phase2_localization_prompt import validate_phase2_localization_topk_v4
from src.cloc_v2.source_cache import atomic_json


OUTPUT = ROOT / "data" / "prototypes" / "week30_adaptive_capacity_prompts_20260911"
SELECTION = ROOT / "data" / "prototypes" / "week29_remaining_miss_audit_20260911" / "selection_manifest.json"
MANIFEST = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "locked_inference_manifest.json"
TRUTH = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "evaluation_ground_truth.json"
MAX_PROMPT_CANDIDATES = 40
MAX_PROMPT_BYTES = 60_000


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["relative_file"]),
        str(row["function_signature"]),
        int(row["start_line"]),
        int(row["end_line"]),
    )


def _selected(preparation: Any) -> list[Mapping[str, Any]]:
    return sorted(
        (row for row in preparation.pre_llm_candidates if row.get("included_in_prompt")),
        key=lambda row: int(str(row["source_slice_id"])[1:]),
    )


def _synthetic_top5(prompt: Any) -> dict[str, Any]:
    findings = []
    for rank, source_slice_id in enumerate(prompt.citation_whitelist.source_slice_ids[:5], start=1):
        evidence_ids = prompt.citation_whitelist.evidence_by_source_slice[source_slice_id]
        findings.append({
            "rank": rank,
            "source_slice_id": source_slice_id,
            "evidence_ids": [evidence_ids[0]],
        })
    return {"findings": findings}


def run() -> dict[str, Any]:
    selected_aliases = _load(SELECTION)["triggered_aliases"]
    samples_by_alias = {
        row["sample_alias"]: row
        for row in _load(MANIFEST)["samples"]
    }
    trees = offline._tree_index()
    normal_records = rebuild.replay._load_normal_records(rebuild.replay.DEFAULT_NORMAL)
    cache_index = rebuild.replay.build_verified_source_cache_index(
        rebuild.replay.DEFAULT_CACHE,
        blockchain_chain_id=1,
        chain_key="eth",
    )

    # Build and save prompts before evaluation truth is loaded.
    blind_rows = []
    preparations: dict[str, tuple[Any, Any]] = {}
    for alias in selected_aliases:
        sample = samples_by_alias[alias]
        tree_path = trees[sample["tx_hash"].lower()][0]
        _base_slices, base = rebuild._build_sample(
            sample,
            tree_path,
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
        )
        _expanded_slices, expanded = rebuild._build_sample(
            sample,
            tree_path,
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
            max_prompt_candidates=MAX_PROMPT_CANDIDATES,
            max_prompt_bytes=MAX_PROMPT_BYTES,
        )
        if expanded.prompt is None:
            raise ValueError(f"WEEK30_PROMPT_NOT_BUILT:{alias}")
        synthetic_validation = validate_phase2_localization_topk_v4(
            _synthetic_top5(expanded.prompt),
            expanded.prompt,
        )
        base_keys = {_candidate_key(row) for row in _selected(base)}
        expanded_keys = {_candidate_key(row) for row in _selected(expanded)}
        row = {
            "sample_alias": alias,
            "base_prompt_sha256": base.prompt_sha256,
            "expanded_prompt_sha256": expanded.prompt_sha256,
            "pre_llm_candidate_count": expanded.pre_llm_candidate_count,
            "base_prompt_candidate_count": base.prompt_candidate_count,
            "expanded_prompt_candidate_count": expanded.prompt_candidate_count,
            "expanded_prompt_bytes": expanded.prompt_bytes,
            "within_byte_limit": expanded.prompt_bytes <= MAX_PROMPT_BYTES,
            "base_candidates_retained": base_keys.issubset(expanded_keys),
            "strict_synthetic_top5_validation": synthetic_validation.valid,
            "reason_codes": [issue.code for issue in expanded.issues],
        }
        blind_rows.append(row)
        preparations[alias] = (base, expanded)
        atomic_json(OUTPUT / "prompts" / f"{alias}.json", expanded.prompt.to_dict())

    blind_manifest = {
        "schema_version": "week30_adaptive_capacity_prompt_manifest_v1",
        "mode": "LOCAL_PROMPT_BUILD_NO_LLM",
        "max_prompt_candidates": MAX_PROMPT_CANDIDATES,
        "max_prompt_bytes": MAX_PROMPT_BYTES,
        "sample_count": len(blind_rows),
        "all_within_byte_limit": all(row["within_byte_limit"] for row in blind_rows),
        "all_base_candidates_retained": all(row["base_candidates_retained"] for row in blind_rows),
        "all_strict_synthetic_valid": all(row["strict_synthetic_top5_validation"] for row in blind_rows),
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": blind_rows,
    }
    atomic_json(OUTPUT / "prompt_manifest.json", blind_manifest)

    truth_by_alias = {
        row["sample_alias"]: row["truths"]
        for row in _load(TRUTH)["labels"]
    }
    evaluation_rows = []
    for alias in selected_aliases:
        base, expanded = preparations[alias]
        truths = truth_by_alias[alias]
        base_positions = [
            int(str(row["source_slice_id"])[1:])
            for row in _selected(base)
            if rebuild._truth_match(row, truths)
        ]
        expanded_positions = [
            int(str(row["source_slice_id"])[1:])
            for row in _selected(expanded)
            if rebuild._truth_match(row, truths)
        ]
        evaluation_rows.append({
            "sample_alias": alias,
            "base_truth_positions": base_positions,
            "expanded_truth_positions": expanded_positions,
            "base_truth_visible": bool(base_positions),
            "expanded_truth_visible": bool(expanded_positions),
            "newly_visible": bool(expanded_positions) and not base_positions,
        })

    acceptance = (
        blind_manifest["all_within_byte_limit"]
        and blind_manifest["all_base_candidates_retained"]
        and blind_manifest["all_strict_synthetic_valid"]
        and all(row["expanded_truth_visible"] for row in evaluation_rows)
    )
    report = {
        "schema_version": "week30_adaptive_capacity_prompt_evaluation_v1",
        "scientific_generalization_result": False,
        "sample_count": len(evaluation_rows),
        "base_truth_visible_count": sum(row["base_truth_visible"] for row in evaluation_rows),
        "expanded_truth_visible_count": sum(row["expanded_truth_visible"] for row in evaluation_rows),
        "newly_visible_aliases": [row["sample_alias"] for row in evaluation_rows if row["newly_visible"]],
        "local_acceptance_gate_passed": acceptance,
        "result_scope": "PROMPT_CANDIDATE_VISIBILITY_NOT_FUNCTION_RECALL_AT_5",
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": evaluation_rows,
    }
    atomic_json(OUTPUT / "evaluation_report.json", report)
    return {"prompt_manifest": blind_manifest, "evaluation_report": report}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))

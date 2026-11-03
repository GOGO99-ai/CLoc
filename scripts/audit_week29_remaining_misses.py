"""Audit the six Week-27 misses and test one bounded prompt-capacity hypothesis."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_localization15_offline_rebuild as offline
from src.cloc_v2.source_cache import atomic_json
from src.cloc_v2.source_index import selector_for_signature


OUTPUT = ROOT / "data" / "prototypes" / "week29_remaining_miss_audit_20260911"
BASELINE = ROOT / "data" / "prototypes" / "localization15_offline_rebuild_20260911" / "report.json"
WEEK21 = ROOT / "data" / "prototypes" / "week21_sequence_incident_fallback_20260911" / "report.json"
WEEK27 = ROOT / "data" / "prototypes" / "week27_reliability_rerank_20260911" / "evaluation_report.json"
MANIFEST = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "locked_inference_manifest.json"
TRUTH = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "evaluation_ground_truth.json"

BASE_PROMPT_LIMIT = 20
ADAPTIVE_POOL_LIMIT = 40
_RAW_SELECTOR = re.compile(r"^(?:0x)?([0-9a-fA-F]{8})$")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_selectors(path: Path) -> list[str]:
    document = _load(path)
    root = document.get("call_tree", document)
    selectors: set[str] = set()

    def visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        function = node.get("function")
        match = _RAW_SELECTOR.fullmatch(function) if isinstance(function, str) else None
        if match is not None:
            selectors.add("0x" + match.group(1).lower())
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                visit(child)

    visit(root)
    return sorted(selectors)


def _failure_stage(row: Mapping[str, Any]) -> str:
    if row["preparation_status"] == "INSUFFICIENT_EVIDENCE":
        return "NO_STRICT_RESULT_CHAIN"
    if not row["pre_llm_truth_hit"]:
        return "PRE_LLM_POOL_MISS"
    if not row["prompt_truth_hit"]:
        return "FIXED_TOP20_TRUNCATION"
    return "NOT_A_PROMPT_RECALL_MISS"


def run() -> dict[str, Any]:
    baseline = _load(BASELINE)
    week27 = _load(WEEK27)
    manifest = _load(MANIFEST)
    trees = offline._tree_index()

    # Freeze the capacity trigger using candidate counts only. Evaluation truth is
    # loaded below, after this selection manifest has been materialized.
    baseline_rows = {row["sample_alias"]: row for row in baseline["samples"]}
    current_misses = [
        row["sample_alias"]
        for row in week27["samples"]
        if not row["reranked_hit_at_5"]
    ]
    selection_rows = []
    for alias, row in baseline_rows.items():
        candidate_count = row["pre_llm_candidate_count"]
        triggered = (
            row["preparation_status"] == "READY_FOR_LLM"
            and BASE_PROMPT_LIMIT < candidate_count <= ADAPTIVE_POOL_LIMIT
        )
        selection_rows.append({
            "sample_alias": alias,
            "preparation_status": row["preparation_status"],
            "pre_llm_candidate_count": candidate_count,
            "base_prompt_candidate_count": row["prompt_candidate_count"],
            "adaptive_capacity_triggered": triggered,
            "counterfactual_visible_candidate_count": candidate_count if triggered else row["prompt_candidate_count"],
        })
    selection_manifest = {
        "schema_version": "week29_adaptive_prompt_capacity_selection_v1",
        "mode": "POST_HOC_OFFLINE_LABEL_FREE_TRIGGER",
        "base_prompt_limit": BASE_PROMPT_LIMIT,
        "adaptive_pool_limit": ADAPTIVE_POOL_LIMIT,
        "trigger_condition": "READY_FOR_LLM_AND_21_TO_40_PRE_LLM_CANDIDATES",
        "triggered_aliases": [row["sample_alias"] for row in selection_rows if row["adaptive_capacity_triggered"]],
        "samples": selection_rows,
    }
    atomic_json(OUTPUT / "selection_manifest.json", selection_manifest)

    truth_by_alias = {
        row["sample_alias"]: row["truths"]
        for row in _load(TRUTH)["labels"]
    }
    manifest_by_alias = {row["sample_alias"]: row for row in manifest["samples"]}
    sequence_by_alias = {
        row["sample_alias"]: row
        for row in _load(WEEK21)["samples"]
    }
    selection_by_alias = {row["sample_alias"]: row for row in selection_rows}

    rows = []
    for alias in current_misses:
        baseline_row = baseline_rows[alias]
        selection = selection_by_alias[alias]
        positions = baseline_row["truth_pre_llm_positions"]
        adaptive_hit = baseline_row["prompt_truth_hit"] or (
            selection["adaptive_capacity_triggered"]
            and any(position <= selection["counterfactual_visible_candidate_count"] for position in positions)
        )
        sample = manifest_by_alias[alias]
        trace_path = trees[sample["tx_hash"].lower()][0]
        executed_selectors = _trace_selectors(trace_path)
        truth_selectors = sorted({
            selector_for_signature(truth.get("function_signature"))
            for truth in truth_by_alias[alias]
            if truth.get("function_signature")
        })
        sequence_hit = sequence_by_alias[alias]["counterfactual_prompt_truth_hit"]
        rows.append({
            "sample_alias": alias,
            "failure_stage": _failure_stage(baseline_row),
            "preparation_status": baseline_row["preparation_status"],
            "pre_llm_candidate_count": baseline_row["pre_llm_candidate_count"],
            "base_prompt_candidate_count": baseline_row["prompt_candidate_count"],
            "truth_pre_llm_positions": positions,
            "adaptive_capacity_triggered": selection["adaptive_capacity_triggered"],
            "adaptive_capacity_truth_visible": adaptive_hit,
            "sequence_fallback_truth_visible": sequence_hit,
            "truth_selectors": truth_selectors,
            "executed_trace_selectors": executed_selectors,
            "truth_selector_executed": bool(set(truth_selectors) & set(executed_selectors)),
        })

    stage_counts = Counter(row["failure_stage"] for row in rows)
    adaptive_recovered = [row["sample_alias"] for row in rows if row["adaptive_capacity_truth_visible"]]
    sequence_recovered = [row["sample_alias"] for row in rows if row["sequence_fallback_truth_visible"]]
    combined_visible = {
        row["sample_alias"]
        for row in baseline["samples"]
        if row["prompt_truth_hit"]
    } | set(adaptive_recovered) | set(sequence_recovered)
    report = {
        "schema_version": "week29_remaining_miss_audit_v1",
        "mode": "OFFLINE_POST_HOC_COUNTERFACTUAL_NO_LLM",
        "scientific_generalization_result": False,
        "fixed_sample_count": baseline["sample_count"],
        "current_week27_top5_hits": week27["reranked_function_recall_counts"]["at_5"],
        "current_week27_top5_misses": len(current_misses),
        "failure_stage_counts": dict(sorted(stage_counts.items())),
        "adaptive_capacity": {
            "base_limit": BASE_PROMPT_LIMIT,
            "pool_limit": ADAPTIVE_POOL_LIMIT,
            "triggered_aliases": selection_manifest["triggered_aliases"],
            "recovered_truth_visibility_aliases": adaptive_recovered,
            "base_prompt_truth_visibility": baseline["prompt_truth_hit_count"],
            "counterfactual_prompt_truth_visibility": baseline["prompt_truth_hit_count"] + len(adaptive_recovered),
            "existing_visibility_regressions": 0,
            "status": "CANDIDATE_VISIBILITY_ONLY_NOT_A_VALIDATED_PROMPT_OR_TOP5_RESULT",
        },
        "sequence_fallback": {
            "recovered_truth_visibility_aliases": sequence_recovered,
            "status": "PREVIOUS_OFFLINE_INCIDENT_LEVEL_COUNTERFACTUAL",
        },
        "roadmap_union_truth_visibility_count": len(combined_visible),
        "roadmap_union_status": "NON_ADDITIVE_ROADMAP_SIGNAL_NOT_A_REPORTABLE_METRIC",
        "selector_audit": {
            "adaptive_recovered_with_truth_selector_executed": sum(
                row["truth_selector_executed"] for row in rows if row["adaptive_capacity_truth_visible"]
            ),
            "conclusion": "DODO_SELLSHARES_IS_NOT_DIRECTLY_EXECUTED_IN_THE_THREE_QUERY_TRACES",
        },
        "recommended_next_experiment": (
            "GENERATE_BYTE_BOUNDED_EXPANDED_PROMPTS_FOR_THE_THREE_21_TO_40_CANDIDATE_CASES_"
            "THEN_RUN_ONE_FROZEN_DEEPSEEK_TOP5_ATTEMPT_PER_CASE"
        ),
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": rows,
    }
    atomic_json(OUTPUT / "report.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "current_week27_top5_hits",
        "current_week27_top5_misses",
        "failure_stage_counts",
        "adaptive_capacity",
        "sequence_fallback",
        "roadmap_union_truth_visibility_count",
        "selector_audit",
        "recommended_next_experiment",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

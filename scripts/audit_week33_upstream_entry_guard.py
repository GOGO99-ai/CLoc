"""Offline new-19 and old-15 audit for the small upstream-entry guard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_localization15_deepseek_official as official
from scripts import run_week14_behavior_similarity as rebuild
from scripts import run_week16_local6_openai_top5 as live
from scripts import run_week31_deepseek_adaptive_capacity as week31
from scripts import run_week32_deepseek_remaining_batch as week32
from src.cloc_v2.phase2_localization_prompt import (
    materialize_phase2_localization_result_topk_v4,
    validate_phase2_localization_topk_v4,
)
from src.cloc_v2.phase2_upstream_entry_guard import apply_upstream_entry_guard
from src.cloc_v2.source_cache import atomic_json


OUTPUT = ROOT / "data" / "prototypes" / "week33_upstream_entry_guard_20260911"
NEW_ROOT = week32.OUTPUT
OLD_ROOT = ROOT / "data" / "prototypes" / "week26_deepseek_transport_retry_external_20260911" / "combined"
WEEK27 = ROOT / "data" / "prototypes" / "week27_reliability_rerank_20260911" / "validated_predictions"
WEEK31 = week31.OUTPUT / "predictions"
EXPECTED_NEW = 19
EXPECTED_OLD = 15


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hits(topk: list[Mapping[str, Any]], truths: list[Mapping[str, Any]]) -> dict[int, bool]:
    return {
        k: any(live._function_match(candidate, truth) for candidate in topk[:k] for truth in truths)
        for k in (1, 3, 5)
    }


def _apply(alias: str, preparation: Any, prediction: Mapping[str, Any], topk: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if preparation.prompt is None or not prediction.get("topk_valid") or len(topk) != 5:
        return [dict(row) for row in topk], {
            "applied": False,
            "reason_code": "LLM_OUTPUT_NOT_VALID",
            "endpoint_count": 0,
            "upstream_entry_source_slice_id": None,
        }
    response = {"findings": [
        {
            "rank": row["rank"],
            "source_slice_id": row["source_slice_id"],
            "evidence_ids": list(row["evidence_ids"]),
        }
        for row in sorted(topk, key=lambda row: row["rank"])
    ]}
    validation = validate_phase2_localization_topk_v4(response, preparation.prompt)
    if not validation.valid:
        raise ValueError(f"SAVED_TOPK_REVALIDATION_FAILED:{alias}:{validation.reason_codes}")
    guard = apply_upstream_entry_guard(
        validation,
        preparation.prompt,
        prediction["pre_llm_candidates"],
    )
    guarded = validate_phase2_localization_topk_v4(guard["response"], preparation.prompt)
    if not guarded.valid:
        raise ValueError(f"GUARDED_TOPK_REVALIDATION_FAILED:{alias}:{guarded.reason_codes}")
    materialized = materialize_phase2_localization_result_topk_v4(guarded, preparation.prompt)
    return live._topk_rows(preparation, materialized), guard


def _evaluate_row(
    alias: str,
    original: list[Mapping[str, Any]],
    guarded: list[Mapping[str, Any]],
    guard: Mapping[str, Any],
    truths: list[Mapping[str, Any]],
) -> dict[str, Any]:
    original_hits = _hits(original, truths)
    guarded_hits = _hits(guarded, truths)
    return {
        "sample_alias": alias,
        "guard_applied": guard["applied"],
        "reason_code": guard["reason_code"],
        "endpoint_count": guard["endpoint_count"],
        "original_hits": {f"at_{k}": original_hits[k] for k in (1, 3, 5)},
        "guarded_hits": {f"at_{k}": guarded_hits[k] for k in (1, 3, 5)},
        "new_top5_hit": guarded_hits[5] and not original_hits[5],
        "top5_regressed": original_hits[5] and not guarded_hits[5],
        "original_top5": [row.get("function_signature") for row in original],
        "guarded_top5": [row.get("function_signature") for row in guarded],
    }


def _new_rows() -> list[dict[str, Any]]:
    manifest = _load(NEW_ROOT / "inference_manifest.json")
    frozen = _load(NEW_ROOT / "frozen_predictions.json")
    truths = {
        row["sample_alias"]: row["truths"]
        for row in _load(NEW_ROOT / "evaluation_ground_truth.json")["labels"]
    }
    predictions = {row["input_alias"]: row for row in frozen["predictions"]}
    input_for_alias = {
        alias: group["input_alias"]
        for group in manifest["unique_inputs"]
        for alias in group["mapped_sample_aliases"]
    }
    samples, trees, normal_records, cache_index = week32._context(manifest)
    rows = []
    for alias in sorted(samples):
        prediction = predictions[input_for_alias[alias]]
        preparation = week32._preparation(samples[alias], trees, normal_records, cache_index)
        if preparation.prompt_sha256 != prediction["prompt_sha256"]:
            raise ValueError(f"NEW_PROMPT_DRIFT:{alias}")
        original = sorted(prediction["final_topk"], key=lambda row: row["rank"])
        guarded, guard = _apply(alias, preparation, prediction, original)
        rows.append(_evaluate_row(alias, original, guarded, guard, truths[alias]))
    if len(rows) != EXPECTED_NEW:
        raise ValueError(f"NEW_SAMPLE_COUNT_DRIFT:{len(rows)}")
    return rows


def _old_rows() -> list[dict[str, Any]]:
    manifest = _load(OLD_ROOT / "inference_manifest.json")
    truths = {
        row["sample_alias"]: row["truths"]
        for row in _load(live.DEFAULT_TRUTH)["labels"]
    }
    samples, trees, normal_records, cache_index = official._context()
    samples = {row["sample_alias"]: row for row in samples}
    rows = []
    for selected in manifest["samples"]:
        alias = selected["sample_alias"]
        week31_path = WEEK31 / f"{alias}.json"
        base_path = OLD_ROOT / "predictions" / f"{alias}.json"
        prediction = _load(week31_path if week31_path.exists() else base_path)
        if week31_path.exists():
            preparation = week31._expanded_preparation(samples[alias], trees, normal_records, cache_index)
            original = prediction["final_topk"]
        else:
            preparation = rebuild._build_sample(
                samples[alias],
                trees[samples[alias]["tx_hash"].lower()][0],
                normal_records,
                cache_index,
                rebuild.replay.DEFAULT_CACHE,
            )[1]
            week27_path = WEEK27 / f"{alias}.json"
            original = _load(week27_path)["materialized_top5"] if week27_path.exists() else prediction["topk"]
        if preparation.prompt_sha256 != prediction["prompt_sha256"]:
            raise ValueError(f"OLD_PROMPT_DRIFT:{alias}")
        original = sorted(original, key=lambda row: row["rank"])
        guarded, guard = _apply(alias, preparation, prediction, original)
        rows.append(_evaluate_row(alias, original, guarded, guard, truths[alias]))
    if len(rows) != EXPECTED_OLD:
        raise ValueError(f"OLD_SAMPLE_COUNT_DRIFT:{len(rows)}")
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "triggered_count": sum(row["guard_applied"] for row in rows),
        "original_recall_counts": {
            f"at_{k}": sum(row["original_hits"][f"at_{k}"] for row in rows)
            for k in (1, 3, 5)
        },
        "guarded_recall_counts": {
            f"at_{k}": sum(row["guarded_hits"][f"at_{k}"] for row in rows)
            for k in (1, 3, 5)
        },
        "new_top5_hit_aliases": [row["sample_alias"] for row in rows if row["new_top5_hit"]],
        "top5_regressed_aliases": [row["sample_alias"] for row in rows if row["top5_regressed"]],
    }


def run() -> dict[str, Any]:
    method = {
        "schema_version": "week33_upstream_entry_guard_method_v1",
        "label_free": True,
        "trigger": "VALID_TOP5_WITH_AT_LEAST_4_ASSET_RESULT_ENDPOINTS",
        "candidate": "PROMPT_INCLUDED_DIRECT_EXECUTED_DEPTH0_NON_ENDPOINT_WITH_EXACT_OR_UNIQUE_IDENTITY",
        "action": "INSERT_FIRST_ELIGIBLE_CANDIDATE_AT_RANK_3_AND_KEEP_5",
        "function_name_rules": False,
        "transaction_rules": False,
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
    }
    atomic_json(OUTPUT / "method_manifest.json", method)
    new_rows = _new_rows()
    old_rows = _old_rows()
    report = {
        "schema_version": "week33_upstream_entry_guard_audit_v1",
        "scientific_generalization_result": False,
        "method": method,
        "new19": _summary(new_rows),
        "old15_regression_check": _summary(old_rows),
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": {"new19": new_rows, "old15": old_rows},
    }
    atomic_json(OUTPUT / "evaluation_report.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "new19": result["new19"],
        "old15_regression_check": result["old15_regression_check"],
        "external_call_counts": result["external_call_counts"],
    }, ensure_ascii=False, indent=2))

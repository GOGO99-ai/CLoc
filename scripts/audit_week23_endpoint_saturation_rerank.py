"""Offline audit of a guarded rerank for endpoint-saturated LLM Top-5 output."""

from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_v2.source_cache import atomic_json


PREDICTIONS = ROOT / "data" / "prototypes" / "localization15_deepseek_official_20260911" / "predictions"
HISTORICAL6_PREDICTIONS = ROOT / "data" / "prototypes" / "week18_local6_causal_tags_20260911" / "predictions"
TRUTH = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "evaluation_ground_truth.json"
OUTPUT = ROOT / "data" / "prototypes" / "week23_endpoint_saturation_rerank_20260911"
MAX_CONTROLLER_CANDIDATES = 3
SUPPORTED_IDENTITIES = frozenset({"SELECTOR_EXACT_MATCH", "UNIQUE_NAME_MATCH"})


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _controller_candidate(candidate: Mapping[str, Any]) -> bool:
    features = candidate.get("ranking_features", {})
    return bool(
        candidate.get("included_in_prompt")
        and candidate.get("selector_match_status") in SUPPORTED_IDENTITIES
        and candidate.get("internal_expansion_depth") == 0
        and features.get("executed_call_node")
        and features.get("direct_strict_evidence")
        and features.get("call_asset_joint_support")
        and not features.get("asset_result_endpoint")
    )


def _candidate_row(candidate: Mapping[str, Any], origin: str) -> dict[str, Any]:
    return {
        "source_slice_id": candidate["source_slice_id"],
        "relative_file": candidate["relative_file"],
        "function_signature": candidate["function_signature"],
        "slice_start_line": candidate["start_line"],
        "slice_end_line": candidate["end_line"],
        "origin": origin,
    }


def _llm_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_slice_id": row["source_slice_id"],
        "relative_file": row["relative_file"],
        "function_signature": row["function_signature"],
        "slice_start_line": row["slice_start_line"],
        "slice_end_line": row["slice_end_line"],
        "origin": "LLM_TOP5",
    }


def _freeze_rerank(predictions_dir: Path = PREDICTIONS) -> dict[str, Any]:
    rows = []
    for path in sorted(predictions_dir.glob("*.json")):
        prediction = _load(path)
        candidates = list(prediction.get("pre_llm_candidates", []))
        candidate_by_slice = {row["source_slice_id"]: row for row in candidates}
        llm_top5 = sorted(prediction.get("topk", []), key=lambda row: row.get("rank", 999))
        valid = bool(prediction.get("topk_valid")) and len(llm_top5) == 5
        endpoint_flags = [
            bool(candidate_by_slice[row["source_slice_id"]]["ranking_features"].get("asset_result_endpoint"))
            for row in llm_top5
        ] if valid else []
        triggered = valid and all(endpoint_flags)
        controllers = [row for row in candidates if _controller_candidate(row)]
        original = [_llm_row(row) for row in llm_top5]
        proposed = [_candidate_row(row, "CONTROLLER_GUARD") for row in controllers[:MAX_CONTROLLER_CANDIDATES]]
        combined = proposed + original if triggered else original
        deduplicated = []
        seen: set[str] = set()
        for row in combined:
            if row["source_slice_id"] in seen:
                continue
            seen.add(row["source_slice_id"])
            deduplicated.append(row)
        reranked = deduplicated[:5]
        for rank, row in enumerate(reranked, start=1):
            row["rank"] = rank
        rows.append({
            "sample_alias": prediction["sample_alias"],
            "llm_status": prediction["llm_status"],
            "top5_valid": valid,
            "endpoint_flags": endpoint_flags,
            "endpoint_saturation_triggered": triggered,
            "controller_candidate_count": len(controllers),
            "controller_candidates_considered": [
                _candidate_row(row, "CONTROLLER_GUARD")
                for row in controllers[:MAX_CONTROLLER_CANDIDATES]
            ],
            "original_top5": original,
            "reranked_top5": reranked,
        })
    return {
        "schema_version": "week23_endpoint_saturation_rerank_predictions_v1",
        "mode": "OFFLINE_LABEL_FREE_COUNTERFACTUAL_RERANK",
        "trigger": "ALL_VALID_TOP5_CANDIDATES_ARE_ASSET_RESULT_ENDPOINTS",
        "controller_rule": (
            "PROMPT_INCLUDED_AND_HIGH_CONFIDENCE_IDENTITY_AND_EXECUTED_AND_DIRECT_STRICT_AND_"
            "CALL_ASSET_JOINT_AND_NOT_ASSET_ENDPOINT_AND_NOT_INTERNAL_EXPANSION"
        ),
        "max_controller_candidates": MAX_CONTROLLER_CANDIDATES,
        "sample_count": len(rows),
        "samples": rows,
    }


def _file_match(predicted: Any, truth: str) -> bool:
    if not isinstance(predicted, str):
        return False
    value = PurePosixPath(predicted.replace("\\", "/")).as_posix().lstrip("./")
    return value == truth or value.endswith("/" + truth) or truth.endswith("/" + value)


def _truth_hit(rows: list[dict[str, Any]], truths: list[dict[str, Any]], k: int) -> bool:
    return any(
        _file_match(candidate.get("relative_file"), str(truth["relative_file"]))
        and candidate.get("function_signature") == truth.get("function_signature")
        for candidate in rows[:k]
        for truth in truths
    )


def run() -> dict[str, Any]:
    frozen = _freeze_rerank()
    atomic_json(OUTPUT / "reranked_predictions.json", frozen)

    truth_by_alias = {
        row["sample_alias"]: row["truths"]
        for row in _load(TRUTH)["labels"]
    }
    rows = []
    for row in frozen["samples"]:
        truths = truth_by_alias[row["sample_alias"]]
        original_hits = {k: _truth_hit(row["original_top5"], truths, k) for k in (1, 3, 5)}
        reranked_hits = {k: _truth_hit(row["reranked_top5"], truths, k) for k in (1, 3, 5)}
        rows.append({
            "sample_alias": row["sample_alias"],
            "top5_valid": row["top5_valid"],
            "endpoint_saturation_triggered": row["endpoint_saturation_triggered"],
            "original_function_hit_at_1": original_hits[1],
            "original_function_hit_at_3": original_hits[3],
            "original_function_hit_at_5": original_hits[5],
            "reranked_function_hit_at_1": reranked_hits[1],
            "reranked_function_hit_at_3": reranked_hits[3],
            "reranked_function_hit_at_5": reranked_hits[5],
            "new_top5_hit": reranked_hits[5] and not original_hits[5],
            "top5_regressed": original_hits[5] and not reranked_hits[5],
        })

    historical_rows = []
    for row in _freeze_rerank(HISTORICAL6_PREDICTIONS)["samples"]:
        truths = truth_by_alias[row["sample_alias"]]
        original_hit = _truth_hit(row["original_top5"], truths, 5)
        reranked_hit = _truth_hit(row["reranked_top5"], truths, 5)
        historical_rows.append({
            "sample_alias": row["sample_alias"],
            "triggered": row["endpoint_saturation_triggered"],
            "original_top5_hit": original_hit,
            "reranked_top5_hit": reranked_hit,
        })
    report = {
        "schema_version": "week23_endpoint_saturation_rerank_report_v1",
        "scientific_generalization_result": False,
        "fixed_denominator": len(rows),
        "valid_top5_count": sum(row["top5_valid"] for row in rows),
        "triggered_count": sum(row["endpoint_saturation_triggered"] for row in rows),
        "original_function_recall_counts": {
            f"at_{k}": sum(row[f"original_function_hit_at_{k}"] for row in rows)
            for k in (1, 3, 5)
        },
        "reranked_function_recall_counts": {
            f"at_{k}": sum(row[f"reranked_function_hit_at_{k}"] for row in rows)
            for k in (1, 3, 5)
        },
        "new_top5_hit_aliases": [row["sample_alias"] for row in rows if row["new_top5_hit"]],
        "top5_regressed_aliases": [row["sample_alias"] for row in rows if row["top5_regressed"]],
        "historical6_repeatability": {
            "sample_count": len(historical_rows),
            "triggered_aliases": [row["sample_alias"] for row in historical_rows if row["triggered"]],
            "original_top5_hit_count": sum(row["original_top5_hit"] for row in historical_rows),
            "reranked_top5_hit_count": sum(row["reranked_top5_hit"] for row in historical_rows),
            "regressed_aliases": [
                row["sample_alias"]
                for row in historical_rows
                if row["original_top5_hit"] and not row["reranked_top5_hit"]
            ],
        },
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": rows,
    }
    atomic_json(OUTPUT / "report.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "fixed_denominator",
        "valid_top5_count",
        "triggered_count",
        "original_function_recall_counts",
        "reranked_function_recall_counts",
        "new_top5_hit_aliases",
        "top5_regressed_aliases",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

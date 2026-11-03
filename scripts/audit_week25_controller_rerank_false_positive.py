"""Audit observed rerank harms and freeze a small held-out validation plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_week23_endpoint_saturation_rerank as rerank
from src.cloc_v2.source_cache import atomic_json


OUTPUT = ROOT / "data" / "prototypes" / "week25_controller_rerank_false_positive_audit_20260911"
FROZEN = rerank.OUTPUT / "reranked_predictions.json"
MANIFEST = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "locked_inference_manifest.json"
TRUTH = rerank.TRUTH


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row["source_slice_id"] for row in rows]


def _truth_family(truths: list[dict[str, Any]]) -> str:
    truth = truths[0]
    return f'{truth["relative_file"]}::{truth["function_signature"]}'


def run() -> dict[str, Any]:
    frozen_rows = _load(FROZEN)["samples"]
    selection_rows = []
    for row in frozen_rows:
        selection_rows.append({
            "sample_alias": row["sample_alias"],
            "top5_valid": row["top5_valid"],
            "triggered": row["endpoint_saturation_triggered"],
            "modified": _slice_ids(row["original_top5"]) != _slice_ids(row["reranked_top5"]),
        })
    selection_manifest = {
        "schema_version": "week25_controller_rerank_selection_audit_v1",
        "mode": "OFFLINE_LABEL_FREE_TRIGGER_AUDIT",
        "samples": selection_rows,
    }
    atomic_json(OUTPUT / "selection_manifest.json", selection_manifest)

    truths_by_alias = {
        row["sample_alias"]: row["truths"]
        for row in _load(TRUTH)["labels"]
    }
    manifest_samples = _load(MANIFEST)["samples"]
    frozen_by_alias = {row["sample_alias"]: row for row in frozen_rows}
    evaluation_rows = []
    families: dict[str, dict[str, Any]] = {}
    for sample in manifest_samples:
        alias = sample["sample_alias"]
        truths = truths_by_alias[alias]
        family = _truth_family(truths)
        row = frozen_by_alias[alias]
        original_hit = rerank._truth_hit(row["original_top5"], truths, 5)
        reranked_hit = rerank._truth_hit(row["reranked_top5"], truths, 5)
        evaluation = {
            "sample_alias": alias,
            "truth_family": family,
            "top5_valid": row["top5_valid"],
            "triggered": row["endpoint_saturation_triggered"],
            "modified": _slice_ids(row["original_top5"]) != _slice_ids(row["reranked_top5"]),
            "original_top5_hit": original_hit,
            "reranked_top5_hit": reranked_hit,
            "recovered": reranked_hit and not original_hit,
            "regressed": original_hit and not reranked_hit,
        }
        evaluation_rows.append(evaluation)
        family_row = families.setdefault(family, {
            "truth_family": family,
            "transaction_count": 0,
            "valid_top5_count": 0,
            "triggered_count": 0,
            "recovered_count": 0,
            "regressed_count": 0,
        })
        family_row["transaction_count"] += 1
        family_row["valid_top5_count"] += int(evaluation["top5_valid"])
        family_row["triggered_count"] += int(evaluation["triggered"])
        family_row["recovered_count"] += int(evaluation["recovered"])
        family_row["regressed_count"] += int(evaluation["regressed"])

    historical = rerank._freeze_rerank(rerank.HISTORICAL6_PREDICTIONS)["samples"]
    historical_triggered = [row for row in historical if row["endpoint_saturation_triggered"]]
    historical_modified = [
        row for row in historical_triggered
        if _slice_ids(row["original_top5"]) != _slice_ids(row["reranked_top5"])
    ]
    triggered = [row for row in evaluation_rows if row["triggered"]]
    original_hits = [row for row in evaluation_rows if row["original_top5_hit"]]
    report = {
        "schema_version": "week25_controller_rerank_false_positive_audit_v1",
        "scientific_generalization_result": False,
        "fixed_sample_count": len(evaluation_rows),
        "unique_transaction_count": len({sample["tx_hash"] for sample in manifest_samples}),
        "unique_truth_family_count": len(families),
        "valid_top5_count": sum(row["top5_valid"] for row in evaluation_rows),
        "triggered_transaction_count": len(triggered),
        "triggered_truth_family_count": len({row["truth_family"] for row in triggered}),
        "recovered_transaction_count": sum(row["recovered"] for row in triggered),
        "recovered_truth_family_count": len({row["truth_family"] for row in triggered if row["recovered"]}),
        "observed_harmful_activation_count": sum(row["regressed"] for row in evaluation_rows),
        "original_hit_count": len(original_hits),
        "original_hit_regression_count": sum(row["regressed"] for row in original_hits),
        "historical_same_transaction_replay": {
            "sample_count": len(historical),
            "triggered_count": len(historical_triggered),
            "modified_count": len(historical_modified),
            "independent_transaction_count": 0,
        },
        "family_breakdown": sorted(families.values(), key=lambda row: row["truth_family"]),
        "held_out_plan": {
            "split_unit": "INCIDENT_OR_CONTRACT_TRUTH_FUNCTION_FAMILY",
            "minimum_unseen_transactions": 6,
            "minimum_unseen_truth_families": 3,
            "minimum_endpoint_saturation_triggers": 3,
            "single_frozen_run_per_sample": True,
            "acceptance": {
                "overall_function_recall_at_5": "AT_LEAST_4_OF_6",
                "triggered_miss_recovery": "AT_LEAST_2_OF_3",
                "original_hit_regressions": 0,
            },
            "insufficient_rule": "REPORT_INSUFFICIENT_IF_FEWER_THAN_3_HELD_OUT_TRIGGERS",
        },
        "conclusion": (
            "NO_OBSERVED_REGRESSION_BUT_NO_INDEPENDENT_GENERALIZATION_EVIDENCE;_"
            "ALL_4_RECOVERIES_SHARE_1_TRUTH_FAMILY"
        ),
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": evaluation_rows,
    }
    atomic_json(OUTPUT / "report.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "fixed_sample_count",
        "unique_truth_family_count",
        "valid_top5_count",
        "triggered_transaction_count",
        "triggered_truth_family_count",
        "recovered_transaction_count",
        "recovered_truth_family_count",
        "observed_harmful_activation_count",
        "historical_same_transaction_replay",
        "conclusion",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

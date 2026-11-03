"""Apply the already-frozen endpoint guard to the reliability-completed outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_localization15_offline_rebuild as offline
from scripts import run_week14_behavior_similarity as rebuild
from scripts import run_week16_local6_openai_top5 as live
from scripts import validate_week24_controller_rerank as week24
from src.cloc_v2.phase2_endpoint_saturation_guard import apply_endpoint_saturation_guard
from src.cloc_v2.phase2_localization_prompt import (
    materialize_phase2_localization_result_topk_v4,
    validate_phase2_localization_topk_v4,
)
from src.cloc_v2.source_cache import atomic_json


OUTPUT = ROOT / "data" / "prototypes" / "week27_reliability_rerank_20260911"
PREDICTIONS = ROOT / "data" / "prototypes" / "week26_deepseek_transport_retry_external_20260911" / "combined" / "predictions"
MANIFEST = week24.MANIFEST
TRUTH = week24.TRUTH
EXPECTED_TRIGGER_COUNT = 5


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_reranks() -> dict[str, Any]:
    samples = _load(MANIFEST)["samples"]
    by_alias = {row["sample_alias"]: row for row in samples}
    triggered = []
    for alias in by_alias:
        prediction = _load(PREDICTIONS / f"{alias}.json")
        if week24._is_triggered(prediction):
            triggered.append((alias, prediction))
    if len(triggered) != EXPECTED_TRIGGER_COUNT:
        raise ValueError(f"WEEK27_TRIGGER_DRIFT:{len(triggered)}")

    normal_records = rebuild.replay._load_normal_records(rebuild.replay.DEFAULT_NORMAL)
    cache_index = rebuild.replay.build_verified_source_cache_index(
        rebuild.replay.DEFAULT_CACHE,
        blockchain_chain_id=1,
        chain_key="eth",
    )
    trees = offline._tree_index()
    rows = []
    for alias, prediction in triggered:
        sample = by_alias[alias]
        _slices, preparation = rebuild._build_sample(
            sample,
            trees[sample["tx_hash"].lower()][0],
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
        )
        if preparation.prompt is None or preparation.prompt_sha256 != prediction["prompt_sha256"]:
            raise ValueError(f"WEEK27_PROMPT_DRIFT:{alias}")
        original_response = {
            "findings": [
                {
                    "rank": row["rank"],
                    "source_slice_id": row["source_slice_id"],
                    "evidence_ids": list(row["evidence_ids"]),
                }
                for row in sorted(prediction["topk"], key=lambda row: row["rank"])
            ]
        }
        original_validation = validate_phase2_localization_topk_v4(
            original_response,
            preparation.prompt,
        )
        if not original_validation.valid:
            raise ValueError(f"WEEK27_ORIGINAL_VALIDATION_FAILED:{alias}")
        guard_result = apply_endpoint_saturation_guard(
            original_validation,
            preparation.prompt,
            prediction["pre_llm_candidates"],
        )
        if not guard_result["applied"]:
            raise ValueError(f"WEEK27_FROZEN_GUARD_NOT_APPLIED:{alias}")
        response = guard_result["response"]
        controller_slice_ids = guard_result["controller_source_slice_ids"]
        validation = validate_phase2_localization_topk_v4(response, preparation.prompt)

        tampered = json.loads(json.dumps(response))
        tampered["findings"][0]["evidence_ids"] = ["E999"]
        tampered_validation = validate_phase2_localization_topk_v4(tampered, preparation.prompt)
        if tampered_validation.valid or "FINDING_EVIDENCE_UNKNOWN" not in tampered_validation.reason_codes:
            raise ValueError(f"WEEK27_TAMPER_NOT_REJECTED:{alias}")

        materialized = materialize_phase2_localization_result_topk_v4(validation, preparation.prompt)
        topk = live._topk_rows(preparation, materialized)
        atomic_json(OUTPUT / "validated_predictions" / f"{alias}.json", {
            "schema_version": "week27_reliability_rerank_prediction_v1",
            "sample_alias": alias,
            "source_prompt_sha256": preparation.prompt_sha256,
            "frozen_rule_source": "week23_endpoint_saturation_rerank_20260911",
            "controller_slice_ids": controller_slice_ids,
            "response": response,
            "validation": validation.to_dict(),
            "tampered_evidence_rejected": True,
            "materialized_top5": topk,
        })
        rows.append({
            "sample_alias": alias,
            "source_prompt_sha256": preparation.prompt_sha256,
            "controller_slice_ids": controller_slice_ids,
            "strict_validation_passed": True,
            "tampered_evidence_rejected": True,
        })
    return {
        "schema_version": "week27_reliability_rerank_validation_manifest_v1",
        "mode": "OFFLINE_FROZEN_RULE_REPLAY_WITHOUT_LABELS",
        "frozen_rule_source": "week23_endpoint_saturation_rerank_20260911",
        "triggered_count": len(rows),
        "validated_count": sum(row["strict_validation_passed"] for row in rows),
        "tamper_rejection_count": sum(row["tampered_evidence_rejected"] for row in rows),
        "samples": rows,
    }


def run() -> dict[str, Any]:
    validation_manifest = _validated_reranks()
    atomic_json(OUTPUT / "validation_manifest.json", validation_manifest)

    samples = _load(MANIFEST)["samples"]
    truths_by_alias = {
        row["sample_alias"]: row["truths"]
        for row in _load(TRUTH)["labels"]
    }
    validated_paths = {
        path.stem: path
        for path in (OUTPUT / "validated_predictions").glob("*.json")
    }
    rows = []
    for sample in samples:
        alias = sample["sample_alias"]
        prediction = _load(PREDICTIONS / f"{alias}.json")
        original_topk = sorted(prediction.get("topk", []), key=lambda row: row.get("rank", 999))
        reranked_topk = (
            _load(validated_paths[alias])["materialized_top5"]
            if alias in validated_paths
            else original_topk
        )
        truths = truths_by_alias[alias]
        original_hits = {
            k: any(live._function_match(candidate, truth) for candidate in original_topk[:k] for truth in truths)
            for k in (1, 3, 5)
        }
        reranked_hits = {
            k: any(live._function_match(candidate, truth) for candidate in reranked_topk[:k] for truth in truths)
            for k in (1, 3, 5)
        }
        truth = truths[0]
        rows.append({
            "sample_alias": alias,
            "truth_family": f'{truth["relative_file"]}::{truth["function_signature"]}',
            "valid_top5": bool(prediction.get("topk_valid")),
            "rerank_used": alias in validated_paths,
            **{f"original_hit_at_{k}": original_hits[k] for k in (1, 3, 5)},
            **{f"reranked_hit_at_{k}": reranked_hits[k] for k in (1, 3, 5)},
            "new_top5_hit": reranked_hits[5] and not original_hits[5],
            "top5_regressed": original_hits[5] and not reranked_hits[5],
        })

    valid_count = sum(row["valid_top5"] for row in rows)
    original_counts = {k: sum(row[f"original_hit_at_{k}"] for row in rows) for k in (1, 3, 5)}
    reranked_counts = {k: sum(row[f"reranked_hit_at_{k}"] for row in rows) for k in (1, 3, 5)}
    report = {
        "schema_version": "week27_reliability_rerank_evaluation_v1",
        "scientific_generalization_result": False,
        "fixed_denominator": len(rows),
        "valid_top5_denominator": valid_count,
        "strictly_validated_rerank_count": validation_manifest["validated_count"],
        "tamper_rejection_count": validation_manifest["tamper_rejection_count"],
        "original_function_recall_counts": {f"at_{k}": original_counts[k] for k in (1, 3, 5)},
        "reranked_function_recall_counts": {f"at_{k}": reranked_counts[k] for k in (1, 3, 5)},
        "valid_subset_function_recall": {
            f"at_{k}": f"{reranked_counts[k]}/{valid_count}"
            for k in (1, 3, 5)
        },
        "new_top5_hit_aliases": [row["sample_alias"] for row in rows if row["new_top5_hit"]],
        "top5_regressed_aliases": [row["sample_alias"] for row in rows if row["top5_regressed"]],
        "original_hit_truth_family_count": len({row["truth_family"] for row in rows if row["original_hit_at_5"]}),
        "reranked_hit_truth_family_count": len({row["truth_family"] for row in rows if row["reranked_hit_at_5"]}),
        "unique_truth_family_count": len({row["truth_family"] for row in rows}),
        "newly_available_trigger_aliases": [
            row["sample_alias"] for row in rows
            if row["rerank_used"] and row["sample_alias"] == "localization_v4_eth_009"
        ],
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": rows,
    }
    atomic_json(OUTPUT / "evaluation_report.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "fixed_denominator",
        "valid_top5_denominator",
        "strictly_validated_rerank_count",
        "original_function_recall_counts",
        "reranked_function_recall_counts",
        "valid_subset_function_recall",
        "new_top5_hit_aliases",
        "top5_regressed_aliases",
        "reranked_hit_truth_family_count",
        "unique_truth_family_count",
        "newly_available_trigger_aliases",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

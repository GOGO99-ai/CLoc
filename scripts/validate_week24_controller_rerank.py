"""Bind controller rerank candidates to prompt evidence and validate them offline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_week23_endpoint_saturation_rerank as rerank
from scripts import run_localization15_offline_rebuild as offline
from scripts import run_week14_behavior_similarity as rebuild
from scripts import run_week16_local6_openai_top5 as live
from src.cloc_v2.phase2_localization_prompt import (
    materialize_phase2_localization_result_topk_v4,
    validate_phase2_localization_topk_v4,
)
from src.cloc_v2.source_cache import atomic_json


OUTPUT = ROOT / "data" / "prototypes" / "week24_controller_rerank_validation_20260911"
MANIFEST = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "locked_inference_manifest.json"
PREDICTIONS = rerank.PREDICTIONS
TRUTH = rerank.TRUTH


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_triggered(prediction: dict[str, Any]) -> bool:
    topk = sorted(prediction.get("topk", []), key=lambda row: row.get("rank", 999))
    if not prediction.get("topk_valid") or len(topk) != 5:
        return False
    by_slice = {row["source_slice_id"]: row for row in prediction["pre_llm_candidates"]}
    return all(
        by_slice[row["source_slice_id"]]["ranking_features"].get("asset_result_endpoint")
        for row in topk
    )


def _response_document(prediction: dict[str, Any], preparation: Any) -> tuple[dict[str, Any], list[str]]:
    controllers = [
        row for row in prediction["pre_llm_candidates"]
        if rerank._controller_candidate(row)
    ][:rerank.MAX_CONTROLLER_CANDIDATES]
    original = sorted(prediction["topk"], key=lambda row: row["rank"])
    original_by_slice = {row["source_slice_id"]: row for row in original}
    ordered_slice_ids = []
    for source_slice_id in [
        *(row["source_slice_id"] for row in controllers),
        *(row["source_slice_id"] for row in original),
    ]:
        if source_slice_id not in ordered_slice_ids:
            ordered_slice_ids.append(source_slice_id)
    ordered_slice_ids = ordered_slice_ids[:5]

    evidence_scope = preparation.prompt.citation_whitelist.evidence_by_source_slice
    findings = []
    for rank, source_slice_id in enumerate(ordered_slice_ids, start=1):
        original_row = original_by_slice.get(source_slice_id)
        evidence_ids = (
            list(original_row["evidence_ids"])
            if original_row is not None
            else [evidence_scope[source_slice_id][0]]
        )
        findings.append({
            "rank": rank,
            "source_slice_id": source_slice_id,
            "evidence_ids": evidence_ids,
        })
    return {"findings": findings}, [row["source_slice_id"] for row in controllers]


def _freeze_validated_outputs() -> dict[str, Any]:
    samples = _load(MANIFEST)["samples"]
    by_alias = {sample["sample_alias"]: sample for sample in samples}
    triggered = []
    for alias in by_alias:
        prediction = _load(PREDICTIONS / f"{alias}.json")
        if _is_triggered(prediction):
            triggered.append((alias, prediction))
    if len(triggered) != 4:
        raise ValueError(f"CONTROLLER_RERANK_TRIGGER_DRIFT:{len(triggered)}")

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
            raise ValueError(f"CONTROLLER_RERANK_PROMPT_DRIFT:{alias}")
        response, controller_slice_ids = _response_document(prediction, preparation)
        validation = validate_phase2_localization_topk_v4(response, preparation.prompt)
        if not validation.valid:
            raise ValueError(f"CONTROLLER_RERANK_VALIDATION_FAILED:{alias}:{validation.reason_codes}")

        tampered = json.loads(json.dumps(response))
        tampered["findings"][0]["evidence_ids"] = ["E999"]
        tampered_validation = validate_phase2_localization_topk_v4(tampered, preparation.prompt)
        if tampered_validation.valid or "FINDING_EVIDENCE_UNKNOWN" not in tampered_validation.reason_codes:
            raise ValueError(f"CONTROLLER_RERANK_TAMPER_NOT_REJECTED:{alias}")

        materialized = materialize_phase2_localization_result_topk_v4(validation, preparation.prompt)
        topk = live._topk_rows(preparation, materialized)
        artifact = {
            "schema_version": "week24_controller_rerank_prediction_v1",
            "sample_alias": alias,
            "source_prompt_sha256": preparation.prompt_sha256,
            "trigger": "ALL_VALID_TOP5_CANDIDATES_ARE_ASSET_RESULT_ENDPOINTS",
            "controller_slice_ids": controller_slice_ids,
            "response": response,
            "validation": validation.to_dict(),
            "tampered_evidence_rejected": True,
            "tampered_reason_codes": list(tampered_validation.reason_codes),
            "materialized_top5": topk,
        }
        atomic_json(OUTPUT / "validated_predictions" / f"{alias}.json", artifact)
        rows.append({
            "sample_alias": alias,
            "source_prompt_sha256": preparation.prompt_sha256,
            "controller_slice_ids": controller_slice_ids,
            "strict_validation_passed": True,
            "tampered_evidence_rejected": True,
        })
    return {
        "schema_version": "week24_controller_rerank_validation_manifest_v1",
        "mode": "OFFLINE_LABEL_FREE_STRICT_OUTPUT_VALIDATION",
        "triggered_count": len(rows),
        "validated_count": sum(row["strict_validation_passed"] for row in rows),
        "tamper_rejection_count": sum(row["tampered_evidence_rejected"] for row in rows),
        "samples": rows,
    }


def run() -> dict[str, Any]:
    manifest = _freeze_validated_outputs()
    atomic_json(OUTPUT / "validation_manifest.json", manifest)

    samples = _load(MANIFEST)["samples"]
    truth_by_alias = {
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
        original = _load(PREDICTIONS / f"{alias}.json")
        original_topk = sorted(original.get("topk", []), key=lambda row: row.get("rank", 999))
        if alias in validated_paths:
            final_topk = _load(validated_paths[alias])["materialized_top5"]
        else:
            final_topk = original_topk
        truths = truth_by_alias[alias]
        original_hits = {
            k: any(live._function_match(candidate, truth) for candidate in original_topk[:k] for truth in truths)
            for k in (1, 3, 5)
        }
        final_hits = {
            k: any(live._function_match(candidate, truth) for candidate in final_topk[:k] for truth in truths)
            for k in (1, 3, 5)
        }
        rows.append({
            "sample_alias": alias,
            "strictly_validated_rerank_used": alias in validated_paths,
            "original_function_hit_at_5": original_hits[5],
            "validated_rerank_function_hit_at_5": final_hits[5],
            "new_top5_hit": final_hits[5] and not original_hits[5],
            "top5_regressed": original_hits[5] and not final_hits[5],
            **{f"validated_rerank_function_hit_at_{k}": final_hits[k] for k in (1, 3)},
        })
    report = {
        "schema_version": "week24_controller_rerank_validation_report_v1",
        "scientific_generalization_result": False,
        "fixed_denominator": len(rows),
        "strictly_validated_rerank_count": manifest["validated_count"],
        "tamper_rejection_count": manifest["tamper_rejection_count"],
        "validated_rerank_function_recall_counts": {
            f"at_{k}": sum(row[f"validated_rerank_function_hit_at_{k}"] for row in rows)
            for k in (1, 3, 5)
        },
        "new_top5_hit_aliases": [row["sample_alias"] for row in rows if row["new_top5_hit"]],
        "top5_regressed_aliases": [row["sample_alias"] for row in rows if row["top5_regressed"]],
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": rows,
    }
    atomic_json(OUTPUT / "report.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "fixed_denominator",
        "strictly_validated_rerank_count",
        "tamper_rejection_count",
        "validated_rerank_function_recall_counts",
        "new_top5_hit_aliases",
        "top5_regressed_aliases",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

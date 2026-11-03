"""Small, label-free guard against endpoint-saturated Top-5 rankings."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .phase2_localization_prompt import (
    Phase2LocalizationPrompt,
    Phase2LocalizationTopKV4Validation,
    validate_phase2_localization_topk_v4,
)


MAX_CONTROLLER_CANDIDATES = 3
SUPPORTED_IDENTITIES = frozenset({"SELECTOR_EXACT_MATCH", "UNIQUE_NAME_MATCH"})


def _controller_candidate(candidate: Mapping[str, Any]) -> bool:
    features = candidate.get("ranking_features", {})
    return bool(
        candidate.get("included_in_prompt")
        and candidate.get("selector_match_status") in SUPPORTED_IDENTITIES
        and candidate.get("internal_expansion_depth") == 0
        and isinstance(features, Mapping)
        and features.get("executed_call_node")
        and features.get("direct_strict_evidence")
        and features.get("call_asset_joint_support")
        and not features.get("asset_result_endpoint")
    )


def apply_endpoint_saturation_guard(
    decision: Phase2LocalizationTopKV4Validation,
    prompt: Phase2LocalizationPrompt,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prepend up to three evidenced controllers only when all five outputs are endpoints."""
    if (
        not isinstance(decision, Phase2LocalizationTopKV4Validation)
        or not decision.valid
        or decision.decision is None
        or not isinstance(prompt, Phase2LocalizationPrompt)
        or not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
    ):
        raise ValueError("ENDPOINT_SATURATION_GUARD_INPUT_INVALID")

    findings = sorted(
        (dict(row) for row in decision.decision["findings"]),
        key=lambda row: row["rank"],
    )
    original_response = {"findings": findings}
    if len(findings) != 5:
        return {
            "applied": False,
            "reason_code": "TOP5_NOT_COMPLETE",
            "controller_source_slice_ids": [],
            "response": original_response,
        }

    candidate_rows = [row for row in candidates if isinstance(row, Mapping)]
    by_slice = {
        row.get("source_slice_id"): row
        for row in candidate_rows
        if isinstance(row.get("source_slice_id"), str)
    }
    if any(row["source_slice_id"] not in by_slice for row in findings):
        return {
            "applied": False,
            "reason_code": "CANDIDATE_METADATA_INCOMPLETE",
            "controller_source_slice_ids": [],
            "response": original_response,
        }
    if not all(
        bool(by_slice[row["source_slice_id"]].get("ranking_features", {}).get("asset_result_endpoint"))
        for row in findings
    ):
        return {
            "applied": False,
            "reason_code": "TOP5_NOT_ENDPOINT_SATURATED",
            "controller_source_slice_ids": [],
            "response": original_response,
        }

    evidence_scope = prompt.citation_whitelist.evidence_by_source_slice
    controllers = [
        row for row in candidate_rows
        if _controller_candidate(row)
        and row.get("source_slice_id") in evidence_scope
        and evidence_scope[row["source_slice_id"]]
    ][:MAX_CONTROLLER_CANDIDATES]
    if not controllers:
        return {
            "applied": False,
            "reason_code": "NO_ELIGIBLE_CONTROLLER",
            "controller_source_slice_ids": [],
            "response": original_response,
        }

    original_by_slice = {row["source_slice_id"]: row for row in findings}
    ordered_slice_ids = []
    for source_slice_id in [
        *(row["source_slice_id"] for row in controllers),
        *(row["source_slice_id"] for row in findings),
    ]:
        if source_slice_id not in ordered_slice_ids:
            ordered_slice_ids.append(source_slice_id)
    guarded_findings = []
    for rank, source_slice_id in enumerate(ordered_slice_ids[:5], start=1):
        if source_slice_id in original_by_slice:
            finding = dict(original_by_slice[source_slice_id])
        else:
            finding = {
                "source_slice_id": source_slice_id,
                "evidence_ids": [evidence_scope[source_slice_id][0]],
            }
        finding["rank"] = rank
        guarded_findings.append(finding)
    response = {"findings": guarded_findings}
    validation = validate_phase2_localization_topk_v4(response, prompt)
    if not validation.valid:
        raise ValueError(f"ENDPOINT_SATURATION_GUARD_OUTPUT_INVALID:{validation.reason_codes}")
    return {
        "applied": True,
        "reason_code": "ENDPOINT_SATURATION_GUARD_APPLIED",
        "controller_source_slice_ids": [row["source_slice_id"] for row in controllers],
        "response": response,
    }

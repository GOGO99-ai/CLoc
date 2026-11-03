"""Protect one evidenced upstream entry when Top-5 is endpoint-heavy."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .phase2_localization_prompt import (
    Phase2LocalizationPrompt,
    Phase2LocalizationTopKV4Validation,
    validate_phase2_localization_topk_v4,
)


ENDPOINT_THRESHOLD = 4
INSERT_RANK = 3
SUPPORTED_IDENTITIES = frozenset({
    "SELECTOR_EXACT_MATCH",
    "UNIQUE_NAME_MATCH",
    "UNIQUE_SIGNATURE_MATCH",
})


def _upstream_entry(candidate: Mapping[str, Any]) -> bool:
    features = candidate.get("ranking_features", {})
    return bool(
        candidate.get("included_in_prompt")
        and candidate.get("selector_match_status") in SUPPORTED_IDENTITIES
        and candidate.get("internal_expansion_depth") == 0
        and isinstance(features, Mapping)
        and features.get("executed_call_node")
        and features.get("direct_strict_evidence")
        and not features.get("asset_result_endpoint")
    )


def apply_upstream_entry_guard(
    decision: Phase2LocalizationTopKV4Validation,
    prompt: Phase2LocalizationPrompt,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Insert one direct upstream entry at rank 3 when at least four results are endpoints."""
    if (
        not isinstance(decision, Phase2LocalizationTopKV4Validation)
        or not decision.valid
        or decision.decision is None
        or not isinstance(prompt, Phase2LocalizationPrompt)
        or not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
    ):
        raise ValueError("UPSTREAM_ENTRY_GUARD_INPUT_INVALID")

    findings = sorted(
        (dict(row) for row in decision.decision["findings"]),
        key=lambda row: row["rank"],
    )
    original_response = {"findings": findings}
    if len(findings) != 5:
        return _unchanged("TOP5_NOT_COMPLETE", original_response)

    candidate_rows = [row for row in candidates if isinstance(row, Mapping)]
    by_slice = {
        row.get("source_slice_id"): row
        for row in candidate_rows
        if isinstance(row.get("source_slice_id"), str)
    }
    if any(row["source_slice_id"] not in by_slice for row in findings):
        return _unchanged("CANDIDATE_METADATA_INCOMPLETE", original_response)

    endpoint_count = sum(
        bool(by_slice[row["source_slice_id"]].get("ranking_features", {}).get("asset_result_endpoint"))
        for row in findings
    )
    if endpoint_count < ENDPOINT_THRESHOLD:
        return _unchanged("TOP5_NOT_ENDPOINT_HEAVY", original_response, endpoint_count)

    evidence_scope = prompt.citation_whitelist.evidence_by_source_slice
    selected_ids = {row["source_slice_id"] for row in findings}
    upstream = next((
        row for row in candidate_rows
        if _upstream_entry(row)
        and row.get("source_slice_id") not in selected_ids
        and row.get("source_slice_id") in evidence_scope
        and evidence_scope[row["source_slice_id"]]
    ), None)
    if upstream is None:
        return _unchanged("NO_ELIGIBLE_UPSTREAM_ENTRY", original_response, endpoint_count)

    upstream_id = upstream["source_slice_id"]
    inserted = {
        "source_slice_id": upstream_id,
        "evidence_ids": [evidence_scope[upstream_id][0]],
    }
    ordered = [*findings[: INSERT_RANK - 1], inserted, *findings[INSERT_RANK - 1 :]]
    guarded_findings = []
    seen: set[str] = set()
    for row in ordered:
        if row["source_slice_id"] in seen:
            continue
        seen.add(row["source_slice_id"])
        finding = dict(row)
        finding["rank"] = len(guarded_findings) + 1
        guarded_findings.append(finding)
        if len(guarded_findings) == 5:
            break

    response = {"findings": guarded_findings}
    validation = validate_phase2_localization_topk_v4(response, prompt)
    if not validation.valid:
        raise ValueError(f"UPSTREAM_ENTRY_GUARD_OUTPUT_INVALID:{validation.reason_codes}")
    return {
        "applied": True,
        "reason_code": "UPSTREAM_ENTRY_GUARD_APPLIED",
        "endpoint_count": endpoint_count,
        "upstream_entry_source_slice_id": upstream_id,
        "response": response,
    }


def _unchanged(reason_code: str, response: dict[str, Any], endpoint_count: int = 0) -> dict[str, Any]:
    return {
        "applied": False,
        "reason_code": reason_code,
        "endpoint_count": endpoint_count,
        "upstream_entry_source_slice_id": None,
        "response": response,
    }

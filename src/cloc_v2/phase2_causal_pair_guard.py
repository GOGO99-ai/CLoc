"""Reserve bounded Top-5 coverage for an evidenced entry and its unique callee.

Internal candidates inherit their parent's runtime evidence; this module does
not claim that a source-level internal call was independently observed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .phase2_localization_prompt import (
    Phase2LocalizationPrompt,
    Phase2LocalizationTopKV4Validation,
    validate_phase2_localization_topk_v4,
)
from .phase2_upstream_entry_guard import SUPPORTED_IDENTITIES


IDENTITY_FIELDS = ("contract_alias", "relative_file", "function_signature", "start_line", "end_line")


def _supported(candidate: Mapping[str, Any]) -> bool:
    features = candidate.get("ranking_features", {})
    return bool(
        candidate.get("included_in_prompt")
        and isinstance(features, Mapping)
        and features.get("executed_call_node")
        and features.get("direct_strict_evidence")
        and features.get("call_asset_joint_support")
        and not features.get("asset_result_endpoint")
    )


def _entry(candidate: Mapping[str, Any]) -> bool:
    return bool(
        _supported(candidate)
        and candidate.get("selector_match_status") in SUPPORTED_IDENTITIES
        and candidate.get("internal_expansion_depth") == 0
    )


def _unique_child(child: Mapping[str, Any], parent: Mapping[str, Any]) -> bool:
    identity = child.get("unique_parent_function_identity")
    return bool(
        _supported(child)
        and child.get("selector_match_status") == "INTERNAL_UNIQUE_CALL_EXPANSION"
        and child.get("internal_expansion_depth") == 1
        and isinstance(identity, Mapping)
        and identity.get("direct_internal_callee_count") == 1
        and identity.get("selector_match_status") in SUPPORTED_IDENTITIES
        and all(parent.get(key) is not None and identity.get(key) == parent[key] for key in IDENTITY_FIELDS)
    )


def apply_causal_pair_guard(
    decision: Phase2LocalizationTopKV4Validation,
    prompt: Phase2LocalizationPrompt,
    candidates: Sequence[Mapping[str, Any]],
    *,
    protect_entry: bool = True,
    protect_unique_child: bool = True,
) -> dict[str, Any]:
    """Keep existing order, replacing only tail slots needed for coverage.

    Entry priority is the frozen candidate order. Child priority is the first
    eligible parent in the existing ranking (then the reserved entry). Exact
    source identities only bind parents; names never determine priority.
    """
    if (
        not isinstance(decision, Phase2LocalizationTopKV4Validation)
        or not decision.valid or decision.decision is None
        or not isinstance(prompt, Phase2LocalizationPrompt)
        or not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes))
    ):
        raise ValueError("CAUSAL_PAIR_GUARD_INPUT_INVALID")
    findings = sorted((dict(row) for row in decision.decision["findings"]), key=lambda row: row["rank"])
    original = {"findings": findings}

    def unchanged(reason: str) -> dict[str, Any]:
        return {"applied": False, "reason_code": reason, "reserved_source_slice_ids": [], "response": original}

    if len(findings) != 5:
        return unchanged("TOP5_NOT_COMPLETE")
    rows = [row for row in candidates if isinstance(row, Mapping)]
    by_slice = {row["source_slice_id"]: row for row in rows if isinstance(row.get("source_slice_id"), str)}
    if len(by_slice) != sum(isinstance(row.get("source_slice_id"), str) for row in rows):
        raise ValueError("CAUSAL_PAIR_GUARD_DUPLICATE_SLICE")
    if any(row["source_slice_id"] not in by_slice for row in findings):
        return unchanged("CANDIDATE_METADATA_INCOMPLETE")
    scope = prompt.citation_whitelist.evidence_by_source_slice
    eligible = [row for row in rows if scope.get(row.get("source_slice_id"))]
    entries = [row for row in eligible if _entry(row)]
    reserved = entries[:1] if protect_entry else []
    if protect_unique_child:
        parents = [by_slice[row["source_slice_id"]] for row in findings] + reserved
        for parent in parents:
            if not _entry(parent):
                continue
            children = [row for row in eligible if _unique_child(row, parent)]
            if len(children) == 1:
                # Reserve the parent too, so a rank-5 parent is not displaced
                # by the internal function whose evidence depends on it.
                reserved = [parent, children[0]]
                break
    if not reserved:
        return unchanged("NO_ELIGIBLE_CAUSAL_CANDIDATE")
    original_by_slice = {row["source_slice_id"]: row for row in findings}
    reserved_ids = list(dict.fromkeys(row["source_slice_id"] for row in reserved))
    ids = [row["source_slice_id"] for row in findings]
    missing = [slice_id for slice_id in reserved_ids if slice_id not in ids]
    protected = set(ids[:2]) | set(reserved_ids)
    for _ in missing:
        removed = next(slice_id for slice_id in reversed(ids) if slice_id not in protected)
        ids.remove(removed)
    ids.extend(missing)
    output = []
    for rank, slice_id in enumerate(ids, 1):
        finding = dict(original_by_slice.get(slice_id) or {
            "source_slice_id": slice_id, "evidence_ids": [scope[slice_id][0]],
        })
        finding["rank"] = rank
        output.append(finding)
    response = {"findings": output}
    validation = validate_phase2_localization_topk_v4(response, prompt)
    if not validation.valid:
        raise ValueError(f"CAUSAL_PAIR_GUARD_OUTPUT_INVALID:{validation.reason_codes}")
    return {
        "applied": response != original,
        "reason_code": "CAUSAL_PAIR_GUARD_APPLIED" if response != original else "CAUSAL_COVERAGE_ALREADY_PRESENT",
        "reserved_source_slice_ids": [row["source_slice_id"] for row in reserved],
        "response": response,
    }

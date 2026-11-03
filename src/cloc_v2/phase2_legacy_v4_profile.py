"""Frozen, reconstructed Phase-2 v4-style profile for future blind runs.

This module intentionally does not claim byte-for-byte identity with any
historical implementation.  It pins one new, reproducible route while reusing
the verified source/call-graph candidate builder under its legacy 20-item
limit.  No evaluation label is accepted by any public function here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .phase2_source_graph_fallback import (
    build_phase2_source_graph_fallback,
    serialize_phase2_source_candidate_union,
)
from .verified_source_cache_index import VerifiedSourceCacheIndex


PROFILE_ID = "CLOC_PHASE2_V4_LEGACY_STYLE_LOCKED_V1"
PROFILE_STATUS = "RECONSTRUCTED_AND_LOCKED"
PROFILE_SCHEMA_VERSION = "cloc_phase2_v4_legacy_style_profile_v1"

LEGACY_CANDIDATE_POOL_LIMIT = 20
LEGACY_PROMPT_CANDIDATE_LIMIT = 12
LEGACY_TOPK_LIMIT = 5

LEGACY_SORT_FIELDS = (
    "direct_strict_evidence",
    "selector_exact_match",
    "asset_result_endpoint",
    "call_relation_priority",
    "shortest_call_path_length",
    "internal_expansion_depth",
    "file_alias",
    "function_signature",
    "line_range",
    "stable_candidate_id",
)

ENABLED_CAPABILITIES = (
    "STRICT_CHAIN_DIRECT_SOURCE_CANDIDATES",
    "VERIFIED_SOURCE_CANDIDATES_FROM_EXECUTED_CALL_NODES",
    "RUNTIME_EFFECTIVE_PROXY_REQUESTED_ADDRESS_MAPPING",
    "DIRECTED_REVERSE_AND_WEAK_CALL_RELATIONS",
    "SELECTOR_EXACT_MATCH",
    "UNIQUE_FUNCTION_IDENTITY_RECOVERY",
    "SAFE_INTERNAL_FUNCTION_EXPANSION_MAX_DEPTH_2",
    "TRANSACTION_CALL_GRAPH_SOURCE_FALLBACK",
    "VERIFIED_SOURCE_CACHE_CONFLICT_REJECTION",
)

DISABLED_CAPABILITIES = (
    "MULTI_VIEW_CANDIDATE_RANK_FUSION",
    "PHASE2_COARSE_TO_FINE_TOPK_V5",
    "PHASE2_SOURCE_TOURNAMENT_TOPK_V6",
    "PHASE2_V4_100_CANDIDATE_POOL",
    "GRAPH_LINKED_SOURCE_FALLBACK_COEXISTS_WITH_RESOLVED_IDENTITY",
    "COARSE_SOURCE_FUNCTION_SUPPLEMENT",
    "LOCATION_OR_CAUSE_GUIDED_SELECTION",
    "HISTORICAL_HIT_GUIDED_SELECTION",
)

_COEXISTENCE_CODES = frozenset({
    "GRAPH_LINKED_SOURCE_FALLBACK_COEXISTS_WITH_RESOLVED_IDENTITY",
    "COARSE_SOURCE_FUNCTION_SUPPLEMENT_USED",
    "COARSE_SOURCE_FUNCTION_SUPPLEMENT_TRUNCATED",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class LegacyV4Profile:
    """Immutable settings; callers cannot override experimental constants."""

    profile_id: str = PROFILE_ID
    status: str = PROFILE_STATUS
    candidate_pool_limit: int = LEGACY_CANDIDATE_POOL_LIMIT
    prompt_candidate_limit: int = LEGACY_PROMPT_CANDIDATE_LIMIT
    topk_limit: int = LEGACY_TOPK_LIMIT
    max_deepseek_requests_per_sample: int = 1
    retry_enabled: bool = False
    internal_expansion_max_depth: int = 2
    sorting_mode: str = "LEGACY_SINGLE_STABLE_ORDER"
    prompt_selection_mode: str = "LEGACY_PREFIX_SELECTION"
    prompt_contract: str = "PHASE2_LOCALIZATION_TOPK_V4"
    provider_free_text_used: bool = False
    enabled_capabilities: tuple[str, ...] = ENABLED_CAPABILITIES
    disabled_capabilities: tuple[str, ...] = DISABLED_CAPABILITIES
    sort_fields: tuple[str, ...] = LEGACY_SORT_FIELDS

    def __post_init__(self) -> None:
        expected = (
            self.profile_id == PROFILE_ID
            and self.status == PROFILE_STATUS
            and self.candidate_pool_limit == 20
            and self.prompt_candidate_limit == 12
            and self.topk_limit == 5
            and self.max_deepseek_requests_per_sample == 1
            and self.retry_enabled is False
            and self.internal_expansion_max_depth == 2
            and self.sorting_mode == "LEGACY_SINGLE_STABLE_ORDER"
            and self.prompt_selection_mode == "LEGACY_PREFIX_SELECTION"
            and self.prompt_contract == "PHASE2_LOCALIZATION_TOPK_V4"
            and self.provider_free_text_used is False
            and self.enabled_capabilities == ENABLED_CAPABILITIES
            and self.disabled_capabilities == DISABLED_CAPABILITIES
            and self.sort_fields == LEGACY_SORT_FIELDS
        )
        if not expected:
            raise ValueError("LEGACY_V4_PROFILE_OVERRIDE_FORBIDDEN")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "status": self.status,
            "limits": {
                "candidate_function_pool": self.candidate_pool_limit,
                "prompt_candidates": self.prompt_candidate_limit,
                "final_topk": self.topk_limit,
                "internal_function_expansion_depth": self.internal_expansion_max_depth,
            },
            "sorting": {
                "mode": self.sorting_mode,
                "fields_in_priority_order": list(self.sort_fields),
                "multi_view_rrf_enabled": False,
            },
            "prompt_selection": {
                "mode": self.prompt_selection_mode,
                "uses_first_n_from_locked_order": True,
            },
            "enabled_capabilities": list(self.enabled_capabilities),
            "disabled_capabilities": list(self.disabled_capabilities),
            "deepseek_request_policy": {
                "max_requests_per_sample": self.max_deepseek_requests_per_sample,
                "retry_enabled": self.retry_enabled,
                "endpoint_or_model_override_allowed": False,
                "provider_contract": self.prompt_contract,
                "provider_free_text_used": self.provider_free_text_used,
            },
            "reconstruction_notice": {
                "historical_byte_identity_proven": False,
                "statement": "Cannot prove byte-for-byte identity with the historical best implementation.",
                "experimental_role": "A new reproducible route locked from 2026-08-24 onward.",
                "future_fault10_override_allowed": False,
            },
        }


LOCKED_LEGACY_V4_PROFILE = LegacyV4Profile()


def legacy_v4_profile_sha256(profile: LegacyV4Profile = LOCKED_LEGACY_V4_PROFILE) -> str:
    if profile != LOCKED_LEGACY_V4_PROFILE:
        raise ValueError("LEGACY_V4_PROFILE_OVERRIDE_FORBIDDEN")
    return hashlib.sha256(_canonical(profile.to_payload()).encode("utf-8")).hexdigest()


def serialize_legacy_v4_profile(profile: LegacyV4Profile = LOCKED_LEGACY_V4_PROFILE) -> str:
    if profile != LOCKED_LEGACY_V4_PROFILE:
        raise ValueError("LEGACY_V4_PROFILE_OVERRIDE_FORBIDDEN")
    document = profile.to_payload()
    document["configuration_sha256"] = legacy_v4_profile_sha256(profile)
    return json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _normal_path(value: Any) -> str:
    return str(value).replace("\\", "/")


def _request_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        _normal_path(row.get("file_path", "")),
        str(row.get("signature") or row.get("function_name") or ""),
        int(row.get("start_line", 0)),
        int(row.get("end_line", 0)),
    )


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        _normal_path(row.get("relative_file", "")),
        str(row.get("function_signature", "")),
        int(row.get("start_line", 0)),
        int(row.get("end_line", 0)),
    )


def _legacy_sort_key(row: Mapping[str, Any], file_aliases: Mapping[str, str]) -> tuple[Any, ...]:
    features = row.get("ranking_features") if isinstance(row.get("ranking_features"), Mapping) else {}
    selector_exact = row.get("selector_match_status") == "SELECTOR_EXACT_MATCH"
    distance = features.get("shortest_call_path_length", row.get("shortest_call_path_length"))
    if not isinstance(distance, int) or isinstance(distance, bool) or distance < 0:
        distance = 1 << 30
    depth = features.get("internal_expansion_depth", row.get("internal_expansion_depth"))
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        depth = 1 << 30
    relation = features.get("call_relation_priority", 99)
    if not isinstance(relation, int) or isinstance(relation, bool):
        relation = 99
    alias = str(row.get("candidate_alias", ""))
    alias_number = int(alias[1:]) if alias.startswith("C") and alias[1:].isdigit() else 1 << 30
    path = _normal_path(row.get("relative_file", ""))
    file_alias = file_aliases[path]
    file_alias_number = int(file_alias[1:])
    return (
        0 if bool(features.get("direct_strict_evidence")) else 1,
        0 if selector_exact else 1,
        0 if bool(features.get("asset_result_endpoint")) else 1,
        relation,
        distance,
        depth,
        file_alias_number,
        str(row.get("function_signature", "")),
        int(row.get("start_line", 0)),
        int(row.get("end_line", 0)),
        alias_number,
    )


def build_legacy_v4_candidates(
    transaction_graph: Any,
    strict_result_chains: Any,
    existing_source_slices: Any,
    cache_index: VerifiedSourceCacheIndex,
    cache_root: str | Path,
) -> dict[str, Any]:
    """Build and lock the 20-item legacy-style candidate order.

    The 20-item call also keeps the production candidate builder's resolved-
    identity coexistence supplement disabled.  Explicit assertions make any
    future semantic drift fail closed.
    """

    result = build_phase2_source_graph_fallback(
        transaction_graph,
        strict_result_chains,
        existing_source_slices,
        cache_index,
        cache_root,
        max_candidates=LOCKED_LEGACY_V4_PROFILE.candidate_pool_limit,
    )
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    union_value = metadata.get("candidate_union")
    union = json.loads(serialize_phase2_source_candidate_union(union_value))
    if union.get("max_candidates") != 20 or int(union.get("candidate_count", 0)) > 20:
        raise ValueError("LEGACY_V4_CANDIDATE_LIMIT_DRIFT")
    issue_codes = set(union.get("issue_codes", ()))
    source_names = {
        str(source)
        for row in union.get("candidates", ())
        if isinstance(row, Mapping)
        for source in row.get("candidate_sources", ())
    }
    if issue_codes & _COEXISTENCE_CODES or "COARSE_SOURCE_FUNCTION_SUPPLEMENT" in source_names:
        raise ValueError("LEGACY_V4_COEXISTENCE_FALLBACK_DRIFT")

    rows = [dict(row) for row in union.get("candidates", ())]
    paths = sorted({_normal_path(row.get("relative_file", "")) for row in rows})
    file_aliases = {path: f"F{index}" for index, path in enumerate(paths, start=1)}
    rows.sort(key=lambda row: _legacy_sort_key(row, file_aliases))

    for index, row in enumerate(rows, start=1):
        new_alias = f"C{index}"
        row["candidate_alias"] = new_alias
        row["file_alias"] = file_aliases[_normal_path(row["relative_file"])]
        row["legacy_sort_position"] = index

    requests = [dict(row) for row in result.get("requests", ()) if isinstance(row, Mapping)]
    request_by_key: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for request in requests:
        request_by_key.setdefault(_request_key(request), []).append(request)
    ordered_requests: list[dict[str, Any]] = []
    for row in rows:
        matches = request_by_key.get(_row_key(row), [])
        if not matches:
            raise ValueError("LEGACY_V4_CANDIDATE_SOURCE_MAPPING_INVALID")
        ordered_requests.append(matches.pop(0))

    union["candidates"] = rows
    union["candidate_count"] = len(rows)
    union["candidate_count_after_truncation"] = len(rows)
    union["legacy_profile_id"] = PROFILE_ID
    union["legacy_profile_sha256"] = legacy_v4_profile_sha256()
    union["legacy_sorting_mode"] = LOCKED_LEGACY_V4_PROFILE.sorting_mode
    union["legacy_prompt_candidate_ids"] = [
        row["candidate_alias"] for row in rows[:LOCKED_LEGACY_V4_PROFILE.prompt_candidate_limit]
    ]
    final_metadata = dict(metadata)
    final_metadata["candidate_union"] = union
    final_metadata["legacy_profile_id"] = PROFILE_ID
    final_metadata["legacy_profile_sha256"] = legacy_v4_profile_sha256()
    final_metadata["legacy_prompt_selection_mode"] = LOCKED_LEGACY_V4_PROFILE.prompt_selection_mode
    return {**dict(result), "requests": ordered_requests, "metadata": final_metadata}


def select_legacy_v4_prompt_candidates(candidate_source: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Select the first 12 candidates from the already locked single order."""

    metadata = candidate_source.get("metadata") if isinstance(candidate_source.get("metadata"), Mapping) else {}
    value = metadata.get("candidate_union", candidate_source)
    union = json.loads(serialize_phase2_source_candidate_union(value))
    if (
        union.get("legacy_profile_id") != PROFILE_ID
        or union.get("legacy_profile_sha256") != legacy_v4_profile_sha256()
        or union.get("legacy_sorting_mode") != LOCKED_LEGACY_V4_PROFILE.sorting_mode
        or union.get("max_candidates") != LOCKED_LEGACY_V4_PROFILE.candidate_pool_limit
    ):
        raise ValueError("LEGACY_V4_CANDIDATE_PROFILE_MISMATCH")
    rows = union.get("candidates")
    if not isinstance(rows, list) or len(rows) > 20:
        raise ValueError("LEGACY_V4_CANDIDATE_LIMIT_DRIFT")
    aliases = [row.get("candidate_alias") for row in rows if isinstance(row, Mapping)]
    if aliases != [f"C{index}" for index in range(1, len(rows) + 1)]:
        raise ValueError("LEGACY_V4_CANDIDATE_ORDER_INVALID")
    return tuple(dict(row) for row in rows[:LOCKED_LEGACY_V4_PROFILE.prompt_candidate_limit])

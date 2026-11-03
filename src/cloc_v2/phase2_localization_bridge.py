"""Deterministic bridge from existing causal/source artifacts to phase-2 prompts."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .phase2_localization_prompt import (
    EDGE_TYPES,
    MAX_SOURCE_SLICES,
    MAX_SOURCE_SLICES_TOPK_V4,
    Phase2EvidenceEdge,
    Phase2LocalizationPrompt,
    Phase2ResultChainContext,
    Phase2SourceSlice,
    Phase2TransactionContext,
    build_phase2_localization_prompt,
    upgrade_phase2_localization_prompt_to_topk_v4,
)
from .phase2_source_graph_fallback import (
    MAX_SOURCE_FUNCTION_CANDIDATES,
    MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
    serialize_phase2_source_candidate_union,
)

PREPARATION_STATUSES = frozenset({"READY_FOR_LLM", "INSUFFICIENT_SOURCE", "INSUFFICIENT_EVIDENCE", "STRUCTURAL_ERROR"})
SOURCE_RESOLUTION_MODES = frozenset({"EXACT_FUNCTION_SLICE", "VERIFIED_SOURCE_WINDOW_FALLBACK", "TRANSACTION_CALL_GRAPH_FALLBACK"})
SOURCE_SELECTION_MODES = frozenset({"DIRECT_EVIDENCE_SOURCE", "TRANSACTION_CALL_GRAPH_FALLBACK"})
MAX_RESULT_CHAINS = 20
MAX_EVIDENCE_EDGES = 80
MAX_PROMPT_BYTES_TOPK_V4 = 60_000
MIN_PROMPT_CANDIDATES_TOPK_V4 = 5
MULTI_VIEW_RRF_K = 60
MULTI_VIEW_QUOTA_PER_VIEW = 4
MULTI_VIEW_CONTRACT_SOFT_CAP_NUMERATOR = 1
MULTI_VIEW_CONTRACT_SOFT_CAP_DENOMINATOR = 2
MULTI_VIEW_NAMES = (
    "IDENTITY_EVIDENCE_VIEW",
    "CALL_CAUSAL_PROXIMITY_VIEW",
    "ASSET_RESULT_RELEVANCE_VIEW",
    "EXECUTED_SOURCE_COVERAGE_VIEW",
)
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_MACHINE_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _candidate_key(relative_file: Any, function_signature: Any, start_line: Any, end_line: Any) -> tuple[str, str, int, int] | None:
    if not isinstance(relative_file, str) or not isinstance(function_signature, str):
        return None
    if not isinstance(start_line, int) or isinstance(start_line, bool) or not isinstance(end_line, int) or isinstance(end_line, bool):
        return None
    return (relative_file.replace("\\", "/"), function_signature, start_line, end_line)


def _candidate_union_rows(metadata: Mapping[str, Any], max_candidates: int) -> tuple[dict[str, Any], ...]:
    value = metadata.get("candidate_union")
    if value is None:
        return ()
    # Reuse the formal redaction/identity guard before trusting metadata from
    # the source stage.  The bridge never rebuilds or relaxes that contract.
    document = json.loads(serialize_phase2_source_candidate_union(value))
    rows = document.get("candidates")
    document_limit = document.get("max_candidates")
    if (
        not isinstance(rows, list)
        or document.get("candidate_count") != len(rows)
        or not isinstance(document_limit, int)
        or isinstance(document_limit, bool)
        or document_limit > max_candidates
        or len(rows) > max_candidates
    ):
        raise ValueError("PHASE2_SOURCE_CANDIDATE_UNION_INVALID")
    expected_aliases = [f"C{index}" for index in range(1, len(rows) + 1)]
    if [item.get("candidate_alias") for item in rows if isinstance(item, Mapping)] != expected_aliases:
        raise ValueError("PHASE2_SOURCE_CANDIDATE_UNION_INVALID")
    required = {
        "candidate_alias", "relative_file", "function_signature", "start_line", "end_line",
        "selector_match_status", "candidate_sources", "call_relation_modes",
        "shortest_call_path_length", "internal_expansion_depth", "supporting_evidence_ids",
    }
    clean: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise ValueError("PHASE2_SOURCE_CANDIDATE_UNION_INVALID")
        key = _candidate_key(row.get("relative_file"), row.get("function_signature"), row.get("start_line"), row.get("end_line"))
        if key is None or key in seen or key[0].startswith(("/", "\\")) or ".." in key[0].split("/"):
            raise ValueError("PHASE2_SOURCE_CANDIDATE_UNION_INVALID")
        seen.add(key)
        features = row.get("ranking_features") if isinstance(row.get("ranking_features"), Mapping) else {}
        contract_alias = row.get("contract_alias")
        if not isinstance(contract_alias, str) or re.fullmatch(r"K[1-9][0-9]*", contract_alias) is None:
            contract_alias = f"K{len(clean) + 1}"
        parent = row.get("unique_parent_function_identity")
        clean_parent = None
        if parent is not None:
            if not isinstance(parent, Mapping):
                raise ValueError("PHASE2_SOURCE_CANDIDATE_UNION_INVALID")
            parent_key = _candidate_key(parent.get("relative_file"), parent.get("function_signature"), parent.get("start_line"), parent.get("end_line"))
            callee_count = parent.get("direct_internal_callee_count")
            parent_status = parent.get("selector_match_status")
            parent_contract_alias = parent.get("contract_alias")
            if (
                parent_key is None
                or parent_key[0].startswith(("/", "\\"))
                or ".." in parent_key[0].split("/")
                or parent_contract_alias != contract_alias
                or not isinstance(callee_count, int)
                or isinstance(callee_count, bool)
                or callee_count < 1
                or not isinstance(parent_status, str)
                or _MACHINE_CODE.fullmatch(parent_status) is None
            ):
                raise ValueError("PHASE2_SOURCE_CANDIDATE_UNION_INVALID")
            clean_parent = {
                "contract_alias": parent_contract_alias,
                "relative_file": parent_key[0],
                "function_signature": parent_key[1],
                "start_line": parent_key[2],
                "end_line": parent_key[3],
                "selector_match_status": parent_status,
                "direct_internal_callee_count": callee_count,
            }
        clean.append({
            "candidate_alias": row["candidate_alias"],
            "contract_alias": contract_alias,
            "relative_file": key[0],
            "function_signature": key[1],
            "visibility": row.get("visibility") if row.get("visibility") in {"public", "external", "internal", "private"} else "UNKNOWN",
            "start_line": key[2],
            "end_line": key[3],
            "selector_match_status": row["selector_match_status"],
            "candidate_sources": list(row["candidate_sources"]),
            "call_relation_modes": list(row["call_relation_modes"]),
            "shortest_call_path_length": row["shortest_call_path_length"],
            "internal_expansion_depth": row["internal_expansion_depth"],
            "unique_parent_function_identity": clean_parent,
            "supporting_evidence_ids": list(row["supporting_evidence_ids"]),
            "ranking_features": {
                "direct_strict_evidence": bool(features.get("direct_strict_evidence")),
                "selector_priority": int(features.get("selector_priority", 99)),
                "asset_result_endpoint": bool(features.get("asset_result_endpoint")),
                "asset_related": bool(features.get("asset_related")),
                "call_asset_joint_support": bool(features.get("call_asset_joint_support")),
                "call_relation_priority": int(features.get("call_relation_priority", 99)),
                "shortest_call_path_length": features.get("shortest_call_path_length") if isinstance(features.get("shortest_call_path_length"), int) and not isinstance(features.get("shortest_call_path_length"), bool) else None,
                "internal_expansion_depth": int(features.get("internal_expansion_depth", row["internal_expansion_depth"])),
                "address_match_source": features.get("address_match_source") if isinstance(features.get("address_match_source"), str) else "DIRECT_EVIDENCE_ADDRESS",
                "address_match_priority": int(features.get("address_match_priority", 99)),
                "executed_call_node": bool(features.get("executed_call_node", True)),
                "verified_source_conflict_free": bool(features.get("verified_source_conflict_free", True)),
            },
        })
    return tuple(clean)


def _candidate_stable_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    public = item["public"]
    alias = public.get("candidate_alias", "")
    alias_number = int(alias[1:]) if isinstance(alias, str) and alias.startswith("C") and alias[1:].isdigit() else 1 << 30
    return (
        alias_number,
        str(public.get("relative_file", "")),
        str(public.get("function_signature", "")),
        int(public.get("start_line", 0)),
        int(public.get("end_line", 0)),
    )


def _multi_view_candidate_selection(
    candidates: list[dict[str, Any]],
    limit: int,
    sequence_exact_candidate_keys: frozenset[tuple[str, str, int, int]] = frozenset(),
    abnormal_parent_function_keys: frozenset[tuple[str, str, str, int, int]] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic v4 prompt set without labels or prediction outcomes."""
    if not candidates:
        return [], {
            "selection_mode": "MULTI_VIEW_CANDIDATE_RANK_FUSION",
            "view_unique_contribution_counts": {name: 0 for name in MULTI_VIEW_NAMES},
            "protected_selected_count": 0,
            "exact_selector_selected_count": 0,
            "sequence_supported_candidate_count": 0,
            "sequence_support_selected_count": 0,
            "abnormal_parent_candidate_count": 0,
            "abnormal_parent_selected_count": 0,
            "contract_coverage_selected_count": 0,
            "covered_contract_count": 0,
            "quota_selected_count": 0,
            "fusion_fill_selected_count": 0,
            "contract_soft_cap_delayed_count": 0,
            "contract_soft_cap_relaxed": False,
        }

    def features(item: Mapping[str, Any]) -> Mapping[str, Any]:
        value = item["public"].get("ranking_features")
        return value if isinstance(value, Mapping) else {}

    def identity_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        value = features(item)
        fallback = "GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK" in item["public"].get("candidate_sources", ())
        return (
            0 if value.get("direct_strict_evidence") else 1,
            int(value.get("selector_priority", 99)),
            int(value.get("internal_expansion_depth", 0)),
            1 if fallback else 0,
            _candidate_stable_key(item),
        )

    def call_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        value = features(item)
        distance = value.get("shortest_call_path_length")
        return (
            int(value.get("call_relation_priority", 99)),
            distance if isinstance(distance, int) and not isinstance(distance, bool) else 1 << 30,
            0 if value.get("direct_strict_evidence") else 1,
            _candidate_stable_key(item),
        )

    def asset_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        value = features(item)
        distance = value.get("shortest_call_path_length")
        return (
            0 if value.get("asset_result_endpoint") else 1,
            0 if value.get("call_asset_joint_support") else 1,
            0 if value.get("asset_related") else 1,
            int(value.get("call_relation_priority", 99)),
            distance if isinstance(distance, int) and not isinstance(distance, bool) else 1 << 30,
            _candidate_stable_key(item),
        )

    def executed_base_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        value = features(item)
        return (
            0 if value.get("verified_source_conflict_free") else 1,
            int(value.get("address_match_priority", 99)),
            0 if value.get("executed_call_node") else 1,
            int(value.get("call_relation_priority", 99)),
            _candidate_stable_key(item),
        )

    executed_groups: dict[str, list[dict[str, Any]]] = {}
    for item in sorted(candidates, key=executed_base_key):
        executed_groups.setdefault(str(item["public"].get("contract_alias", "K0")), []).append(item)
    executed_ranking: list[dict[str, Any]] = []
    max_group = max((len(value) for value in executed_groups.values()), default=0)
    for offset in range(max_group):
        available = [value[offset] for _, value in sorted(executed_groups.items()) if offset < len(value)]
        executed_ranking.extend(sorted(available, key=executed_base_key))

    rankings = {
        "IDENTITY_EVIDENCE_VIEW": sorted(candidates, key=identity_key),
        "CALL_CAUSAL_PROXIMITY_VIEW": sorted(candidates, key=call_key),
        "ASSET_RESULT_RELEVANCE_VIEW": sorted(candidates, key=asset_key),
        "EXECUTED_SOURCE_COVERAGE_VIEW": executed_ranking,
    }
    ranks = {
        name: {item["candidate_key"]: index for index, item in enumerate(ranking, start=1)}
        for name, ranking in rankings.items()
    }
    quota_hits = {
        item["candidate_key"]: tuple(name for name in MULTI_VIEW_NAMES if ranks[name][item["candidate_key"]] <= MULTI_VIEW_QUOTA_PER_VIEW)
        for item in candidates
    }
    fusion_scores = {
        item["candidate_key"]: round(sum(1.0 / (MULTI_VIEW_RRF_K + ranks[name][item["candidate_key"]]) for name in MULTI_VIEW_NAMES), 15)
        for item in candidates
    }
    fused = sorted(candidates, key=lambda item: (-fusion_scores[item["candidate_key"]], _candidate_stable_key(item)))
    contract_cap = max(1, limit * MULTI_VIEW_CONTRACT_SOFT_CAP_NUMERATOR // MULTI_VIEW_CONTRACT_SOFT_CAP_DENOMINATOR)
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, int, int]] = set()
    contract_counts: dict[str, int] = {}
    deferred_keys: set[tuple[str, str, int, int]] = set()
    contribution_counts = {name: 0 for name in MULTI_VIEW_NAMES}

    def selector_exact(item: Mapping[str, Any]) -> bool:
        return item["public"].get("selector_match_status") == "SELECTOR_EXACT_MATCH"

    def sequence_supported(item: Mapping[str, Any]) -> bool:
        return item["candidate_key"] in sequence_exact_candidate_keys

    def abnormal_parent_supported(item: Mapping[str, Any]) -> bool:
        public = item["public"]
        parent = public.get("unique_parent_function_identity")
        if not isinstance(parent, Mapping) or public.get("internal_expansion_depth") != 1:
            return False
        parent_key = _candidate_key(
            parent.get("relative_file"),
            parent.get("function_signature"),
            parent.get("start_line"),
            parent.get("end_line"),
        )
        scoped_parent_key = (str(parent.get("contract_alias", "")), *parent_key) if parent_key is not None else None
        return parent.get("direct_internal_callee_count") == 1 and scoped_parent_key in abnormal_parent_function_keys

    def protected(item: Mapping[str, Any]) -> bool:
        # Direct strict evidence can be inherited by every function from a
        # linked source contract.  Only selector identity is function-specific
        # enough to reserve a prompt position unconditionally.
        return selector_exact(item) or sequence_supported(item) or abnormal_parent_supported(item)

    def add(item: dict[str, Any], stage: str, contributor: str | None, *, relax: bool = False) -> bool:
        key = item["candidate_key"]
        if key in selected_keys or len(selected) >= limit:
            return False
        contract = str(item["public"].get("contract_alias", "K0"))
        if not relax and not protected(item) and contract_counts.get(contract, 0) >= contract_cap:
            deferred_keys.add(key)
            return False
        selected.append(item)
        selected_keys.add(key)
        contract_counts[contract] = contract_counts.get(contract, 0) + 1
        item["selection_stage"] = stage
        if contributor is not None:
            contribution_counts[contributor] += 1
        return True

    for item in sorted((item for item in candidates if selector_exact(item)), key=identity_key):
        add(item, "PROTECTED_EVIDENCE", None, relax=True)
    exact_selector_count = len(selected)
    for item in sorted((item for item in candidates if sequence_supported(item) and not selector_exact(item)), key=identity_key):
        add(item, "SEQUENCE_EXACT_SUPPORT", None, relax=True)
    sequence_support_count = len(selected) - exact_selector_count
    before_parent_support = len(selected)
    for item in sorted(
        (item for item in candidates if abnormal_parent_supported(item) and not selector_exact(item) and not sequence_supported(item)),
        key=identity_key,
    ):
        add(item, "ABNORMAL_PARENT_UNIQUE_CALLEE", None, relax=True)
    abnormal_parent_selected_count = len(selected) - before_parent_support
    protected_count = len(selected)

    # Preserve source-contract coverage before global rank fusion.  Four prompt
    # positions remain available for view-specific evidence outside this pass.
    coverage_target = min(limit, max(1, limit - MULTI_VIEW_QUOTA_PER_VIEW))
    representatives = sorted((items[0] for items in executed_groups.values()), key=executed_base_key)
    for item in representatives:
        if len(selected) >= coverage_target:
            break
        add(item, "CONTRACT_COVERAGE", None)
    contract_coverage_count = len(selected) - protected_count

    view_start = len(selected)
    for name in MULTI_VIEW_NAMES:
        for item in rankings[name][:MULTI_VIEW_QUOTA_PER_VIEW]:
            add(item, "VIEW_QUOTA", name)
    quota_count = len(selected) - view_start
    for item in fused:
        add(item, "FUSION_FILL", None)
    target = min(limit, len(candidates))
    relaxed = False
    if len(selected) < target:
        relaxed = True
        for item in fused:
            add(item, "FUSION_FILL", None, relax=True)
    for item in candidates:
        key = item["candidate_key"]
        item["public"].update({
            "view_ranks": {name: ranks[name][key] for name in MULTI_VIEW_NAMES},
            "selected_by_views": list(quota_hits[key]),
            "fusion_score": fusion_scores[key],
            "selection_stage": item.get("selection_stage"),
            "sequence_exact_support": sequence_supported(item),
            "abnormal_parent_support": abnormal_parent_supported(item),
            "contract_soft_cap_deferred": key in deferred_keys,
        })
    return selected, {
        "selection_mode": "MULTI_VIEW_CANDIDATE_RANK_FUSION",
        "rrf_k": MULTI_VIEW_RRF_K,
        "view_quota": MULTI_VIEW_QUOTA_PER_VIEW,
        "contract_soft_cap": contract_cap,
        "view_unique_contribution_counts": contribution_counts,
        "protected_selected_count": protected_count,
        "exact_selector_selected_count": exact_selector_count,
        "sequence_supported_candidate_count": sum(sequence_supported(item) for item in candidates),
        "sequence_support_selected_count": sequence_support_count,
        "abnormal_parent_candidate_count": sum(abnormal_parent_supported(item) for item in candidates),
        "abnormal_parent_selected_count": abnormal_parent_selected_count,
        "contract_coverage_selected_count": contract_coverage_count,
        "covered_contract_count": len({str(item["public"].get("contract_alias", "K0")) for item in selected}),
        "quota_selected_count": quota_count,
        "fusion_fill_selected_count": len(selected) - protected_count - contract_coverage_count - quota_count,
        "contract_soft_cap_delayed_count": len(deferred_keys),
        "contract_soft_cap_relaxed": relaxed,
    }


def _document(value: Any, schema: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    elif callable(getattr(value, "to_dict", None)):
        result = value.to_dict()
    else:
        raise ValueError("PHASE2_BRIDGE_INPUT_INVALID")
    if result.get("schema_version") != schema:
        raise ValueError("PHASE2_BRIDGE_SCHEMA_INVALID")
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _codes(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    return tuple(sorted({item for item in values if isinstance(item, str) and _MACHINE_CODE.fullmatch(item)}))


def _redact_source(value: str) -> str:
    value = _RAW_HASH.sub("TRANSACTION_HASH_REDACTED", value)
    return _RAW_ADDRESS.sub("ADDRESS_LITERAL_REDACTED", value)


@dataclass(frozen=True)
class Phase2BridgeIssue:
    code: str
    severity: str
    count: int = 1

    def __post_init__(self) -> None:
        if _MACHINE_CODE.fullmatch(self.code) is None or self.severity not in {"AUDIT", "FATAL"}:
            raise ValueError("PHASE2_BRIDGE_ISSUE_INVALID")
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise ValueError("PHASE2_BRIDGE_ISSUE_COUNT_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "count": self.count}


@dataclass(frozen=True)
class Phase2LocalizationPreparation:
    preparation_status: str
    prompt: Phase2LocalizationPrompt | None
    issues: tuple[Phase2BridgeIssue, ...]
    strict_chain_count: int
    evidence_type_counts: Mapping[str, int]
    source_slice_count: int
    prompt_sha256: str | None
    prompt_bytes: int
    _source_locations: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, repr=False)
    source_resolution_mode: str = "EXACT_FUNCTION_SLICE"
    source_selection_mode: str = "DIRECT_EVIDENCE_SOURCE"
    retained_strict_chain_count: int = 0
    _pre_llm_candidates: tuple[Mapping[str, Any], ...] = field(default=(), repr=False)
    _candidate_selection_audit: Mapping[str, Any] = field(default_factory=dict, repr=False)

    SCHEMA_VERSION = "phase2_localization_preparation_v1"

    def __post_init__(self) -> None:
        if self.preparation_status not in PREPARATION_STATUSES:
            raise ValueError("PHASE2_PREPARATION_STATUS_INVALID")
        if self.source_resolution_mode not in SOURCE_RESOLUTION_MODES:
            raise ValueError("PHASE2_SOURCE_RESOLUTION_MODE_INVALID")
        if self.source_selection_mode not in SOURCE_SELECTION_MODES:
            raise ValueError("PHASE2_SOURCE_SELECTION_MODE_INVALID")
        if not isinstance(self.strict_chain_count, int) or self.strict_chain_count < 0:
            raise ValueError("PHASE2_STRICT_CHAIN_COUNT_INVALID")
        retained = self.strict_chain_count if self.retained_strict_chain_count == 0 else self.retained_strict_chain_count
        if not isinstance(retained, int) or retained < 0 or retained > self.strict_chain_count:
            raise ValueError("PHASE2_RETAINED_CHAIN_COUNT_INVALID")
        object.__setattr__(self, "retained_strict_chain_count", retained)
        ready = self.preparation_status == "READY_FOR_LLM"
        if ready != isinstance(self.prompt, Phase2LocalizationPrompt):
            raise ValueError("PHASE2_PREPARATION_PROMPT_INCONSISTENT")
        issues = tuple(sorted(self.issues, key=lambda item: (item.severity, item.code, item.count)))
        object.__setattr__(self, "issues", issues)
        counts = {key: int(self.evidence_type_counts.get(key, 0)) for key in sorted(EDGE_TYPES)}
        if any(value < 0 for value in counts.values()):
            raise ValueError("PHASE2_EVIDENCE_COUNTS_INVALID")
        object.__setattr__(self, "evidence_type_counts", MappingProxyType(counts))
        locations = {key: MappingProxyType(dict(value)) for key, value in sorted(self._source_locations.items())}
        object.__setattr__(self, "_source_locations", MappingProxyType(locations))
        candidate_rows = tuple(MappingProxyType(json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))) for value in self._pre_llm_candidates)
        candidate_text = _canonical([dict(value) for value in candidate_rows])
        if _RAW_ADDRESS.search(candidate_text) or _RAW_HASH.search(candidate_text) or len(candidate_rows) > MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4:
            raise ValueError("PHASE2_PRE_LLM_CANDIDATES_INVALID")
        object.__setattr__(self, "_pre_llm_candidates", candidate_rows)
        audit = json.loads(json.dumps(dict(self._candidate_selection_audit), ensure_ascii=False, sort_keys=True))
        audit_text = _canonical(audit)
        if _RAW_ADDRESS.search(audit_text) or _RAW_HASH.search(audit_text):
            raise ValueError("PHASE2_CANDIDATE_SELECTION_AUDIT_INVALID")
        object.__setattr__(self, "_candidate_selection_audit", MappingProxyType(audit))
        if ready:
            encoded = _canonical(self.prompt.to_dict()).encode("utf-8")
            if self.prompt_sha256 != hashlib.sha256(encoded).hexdigest() or self.prompt_bytes != len(encoded):
                raise ValueError("PHASE2_PROMPT_METADATA_INVALID")
        elif self.prompt_sha256 is not None or self.prompt_bytes != 0:
            raise ValueError("PHASE2_PROMPT_METADATA_FORBIDDEN")

    @property
    def source_locations(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._source_locations.items()}

    @property
    def pre_llm_candidates(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps([dict(value) for value in self._pre_llm_candidates], ensure_ascii=False, sort_keys=True))

    @property
    def pre_llm_candidate_count(self) -> int:
        return len(self._pre_llm_candidates)

    @property
    def prompt_candidate_count(self) -> int:
        return sum(bool(value.get("included_in_prompt")) for value in self._pre_llm_candidates)

    @property
    def fallback_candidate_count(self) -> int:
        return sum(
            "GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK" in value.get("candidate_sources", ())
            for value in self._pre_llm_candidates
        )

    @property
    def candidate_selection_audit(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self._candidate_selection_audit), ensure_ascii=False, sort_keys=True))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION, "preparation_status": self.preparation_status, "prompt": self.prompt.to_dict() if self.prompt is not None else None, "issues": [item.to_dict() for item in self.issues], "strict_chain_count": self.strict_chain_count, "retained_strict_chain_count": self.retained_strict_chain_count, "evidence_type_counts": dict(self.evidence_type_counts), "source_slice_count": self.source_slice_count, "source_resolution_mode": self.source_resolution_mode, "source_selection_mode": self.source_selection_mode, "pre_llm_candidate_count": self.pre_llm_candidate_count, "prompt_candidate_count": self.prompt_candidate_count, "fallback_candidate_count": self.fallback_candidate_count, "pre_llm_candidates": self.pre_llm_candidates, "candidate_selection_audit": self.candidate_selection_audit, "prompt_sha256": self.prompt_sha256, "prompt_bytes": self.prompt_bytes}


def prepare_phase2_localization(
    strict_result_chains: Any,
    causal_graph: Any,
    transaction_graph: Any,
    source_slices: Any,
    *,
    transaction_alias: str = "T1",
    max_pre_llm_candidates: int = MAX_SOURCE_FUNCTION_CANDIDATES,
    max_prompt_candidates: int = MAX_SOURCE_SLICES,
    max_prompt_bytes: int | None = None,
    prompt_contract: str = "v1",
    shortlist_candidate_ids: Sequence[str] | None = None,
    sequence_exact_candidate_keys: Sequence[tuple[str, str, int, int]] | None = None,
    abnormal_parent_function_keys: Sequence[tuple[str, str, str, int, int]] | None = None,
) -> Phase2LocalizationPreparation:
    """Translate only already-present evidence; no label or source selection hint is accepted."""
    try:
        strict = _document(strict_result_chains, "strict_result_chains_v1")
        graph = _document(transaction_graph, "transaction_graph_v1")
        source = _document(source_slices, "source_request_slices_v1")
        causal = dict(causal_graph) if isinstance(causal_graph, Mapping) else causal_graph.to_dict()
        if causal.get("schema_version") not in {"candidate_causal_graph_v1", "candidate_causal_graph_v2"}:
            raise ValueError("PHASE2_CAUSAL_GRAPH_SCHEMA_INVALID")
        if (
            not isinstance(max_pre_llm_candidates, int)
            or isinstance(max_pre_llm_candidates, bool)
            or not 1 <= max_pre_llm_candidates <= MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4
            or not isinstance(max_prompt_candidates, int)
            or isinstance(max_prompt_candidates, bool)
            or not 1 <= max_prompt_candidates <= min(max_pre_llm_candidates, MAX_SOURCE_SLICES_TOPK_V4)
            or (max_prompt_bytes is not None and (not isinstance(max_prompt_bytes, int) or isinstance(max_prompt_bytes, bool) or max_prompt_bytes < 1))
            or prompt_contract not in {"v1", "topk_v4"}
            or (shortlist_candidate_ids is not None and prompt_contract != "topk_v4")
            or (sequence_exact_candidate_keys is not None and prompt_contract != "topk_v4")
            or (abnormal_parent_function_keys is not None and prompt_contract != "topk_v4")
            or (shortlist_candidate_ids is not None and (sequence_exact_candidate_keys is not None or abnormal_parent_function_keys is not None))
        ):
            raise ValueError("PHASE2_BRIDGE_LIMIT_INVALID")
        shortlist = tuple(shortlist_candidate_ids) if shortlist_candidate_ids is not None else None
        sequence_keys: set[tuple[str, str, int, int]] = set()
        for value in sequence_exact_candidate_keys or ():
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                raise ValueError("PHASE2_SEQUENCE_CANDIDATE_KEY_INVALID")
            key = _candidate_key(*value)
            if key is None or key[0].startswith(("/", "\\")) or ".." in key[0].split("/"):
                raise ValueError("PHASE2_SEQUENCE_CANDIDATE_KEY_INVALID")
            sequence_keys.add(key)
        abnormal_parent_keys: set[tuple[str, str, str, int, int]] = set()
        for value in abnormal_parent_function_keys or ():
            if not isinstance(value, (list, tuple)) or len(value) != 5:
                raise ValueError("PHASE2_ABNORMAL_PARENT_KEY_INVALID")
            contract_alias, *identity = value
            key = _candidate_key(*identity)
            if (
                not isinstance(contract_alias, str)
                or re.fullmatch(r"K[1-9][0-9]*", contract_alias) is None
                or key is None
                or key[0].startswith(("/", "\\"))
                or ".." in key[0].split("/")
            ):
                raise ValueError("PHASE2_ABNORMAL_PARENT_KEY_INVALID")
            abnormal_parent_keys.add((contract_alias, *key))
        return _prepare(strict, causal, graph, source, transaction_alias, max_pre_llm_candidates, max_prompt_candidates, max_prompt_bytes, prompt_contract, shortlist, frozenset(sequence_keys), frozenset(abnormal_parent_keys))
    except (KeyError, TypeError, ValueError):
        issue = Phase2BridgeIssue("PHASE2_BRIDGE_STRUCTURE_INVALID", "FATAL")
        return Phase2LocalizationPreparation("STRUCTURAL_ERROR", None, (issue,), 0, {}, 0, None, 0, {})


def _prepare(
    strict: dict[str, Any],
    causal: dict[str, Any],
    graph: dict[str, Any],
    source: dict[str, Any],
    transaction_alias: str,
    max_pre_llm_candidates: int,
    max_prompt_candidates: int,
    max_prompt_bytes: int | None,
    prompt_contract: str,
    shortlist_candidate_ids: tuple[str, ...] | None,
    sequence_exact_candidate_keys: frozenset[tuple[str, str, int, int]],
    abnormal_parent_function_keys: frozenset[tuple[str, str, str, int, int]],
) -> Phase2LocalizationPreparation:
    all_chains = sorted(strict.get("chains", ()), key=lambda item: item.get("strict_chain_id", ""))
    if not all_chains:
        return Phase2LocalizationPreparation("INSUFFICIENT_EVIDENCE", None, (Phase2BridgeIssue("NO_STRICT_RESULT_CHAIN", "FATAL"),), 0, {}, 0, None, 0, {})
    source_metadata = source.get("metadata", {}) if isinstance(source.get("metadata", {}), Mapping) else {}
    candidate_union_rows = _candidate_union_rows(source_metadata, max_pre_llm_candidates)
    candidate_union_document = source_metadata.get("candidate_union") if isinstance(source_metadata.get("candidate_union"), Mapping) else {}
    candidate_union_audit = {
        "candidate_count_before_truncation": int(candidate_union_document.get("candidate_count_before_truncation", candidate_union_document.get("total_candidate_count_before_truncation", len(candidate_union_rows)))),
        "candidate_count_after_truncation": int(candidate_union_document.get("candidate_count_after_truncation", len(candidate_union_rows))),
        "candidate_limit_reached": bool(candidate_union_document.get("candidate_limit_reached", len(candidate_union_rows) == max_pre_llm_candidates)),
        "candidate_limit_exceeded": bool(candidate_union_document.get("candidate_limit_exceeded", candidate_union_document.get("truncated_candidate_count", 0))),
        "truncated_candidate_count": int(candidate_union_document.get("truncated_candidate_count", 0)),
        "candidate_source_counts_before_truncation": dict(candidate_union_document.get("candidate_source_counts_before_truncation", {})),
        "candidate_source_counts_after_truncation": dict(candidate_union_document.get("candidate_source_counts_after_truncation", {})),
    }
    candidate_union_by_key = {
        _candidate_key(item["relative_file"], item["function_signature"], item["start_line"], item["end_line"]): item
        for item in candidate_union_rows
    }
    candidate_union_order = {key: index for index, key in enumerate(candidate_union_by_key)}
    source_selection_mode = source_metadata.get("source_selection_mode", "DIRECT_EVIDENCE_SOURCE")
    if source_selection_mode not in SOURCE_SELECTION_MODES:
        raise ValueError("PHASE2_SOURCE_SELECTION_MODE_INVALID")
    fallback_associations = {
        item.get("strict_chain_id"): item
        for item in source_metadata.get("graph_fallback_associations", ())
        if isinstance(item, Mapping) and isinstance(item.get("strict_chain_id"), str)
    }

    def result_group(item: Mapping[str, Any]) -> tuple[str, ...]:
        values = item.get("result_fact_ids")
        if isinstance(values, (list, tuple)) and values:
            return tuple(sorted(str(value) for value in values))
        value = item.get("result_candidate_id")
        return (str(value),) if value is not None else (str(item.get("strict_chain_id", "")),)

    def chain_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        association = fallback_associations.get(item.get("strict_chain_id"), {})
        return (
            0 if association else 1,
            int(association.get("call_path_length", 1 << 30)) if isinstance(association, Mapping) else 1 << 30,
            0 if item.get("status") == "READY" else 1,
            str(item.get("strict_chain_id", "")),
        )

    ordered_chains = sorted(all_chains, key=chain_key)
    # First retain one chain per independent result candidate, then fill the
    # remaining capacity by the same stable priority.  This prevents a very
    # large fan-out from consuming every prompt slot.
    chains_raw: list[Mapping[str, Any]] = []
    groups: set[tuple[str, ...]] = set()
    for item in ordered_chains:
        group = result_group(item)
        if group not in groups:
            chains_raw.append(item); groups.add(group)
        if len(chains_raw) == MAX_RESULT_CHAINS:
            break
    if len(chains_raw) < MAX_RESULT_CHAINS:
        for item in ordered_chains:
            if item not in chains_raw:
                chains_raw.append(item)
            if len(chains_raw) == MAX_RESULT_CHAINS:
                break
    nodes = {item.get("node_id"): item for item in graph.get("nodes", ()) if isinstance(item, Mapping) and isinstance(item.get("node_id"), str)}
    if not nodes:
        raise ValueError("TRANSACTION_GRAPH_NODES_MISSING")
    addresses = sorted({str(item.get("callee_address")).lower() for item in nodes.values() if isinstance(item.get("callee_address"), str) and _RAW_ADDRESS.fullmatch(item.get("callee_address"))})
    address_alias = {address: f"A{index}" for index, address in enumerate(addresses, start=1)}
    chain_alias = {item["strict_chain_id"]: f"Q{index}" for index, item in enumerate(chains_raw, start=1)}
    chain_nodes = {item["strict_chain_id"]: set(item.get("node_ids", ())) for item in chains_raw}
    chain_contexts = tuple(Phase2ResultChainContext(chain_alias[item["strict_chain_id"]], f"observed {item.get('result_type') or 'result'} at the end of this strict evidence chain", _codes(item.get("uncertainties", ()))) for item in chains_raw)
    strict_edges = {item.get("edge_id"): item for item in strict.get("edges", ()) if isinstance(item, Mapping)}
    raw_evidence: list[tuple[str, str, str, str, tuple[str, ...], tuple[str, str]]] = []
    semantic_seen: set[tuple[str, str, str, str]] = set()
    for chain in chains_raw:
        cid = chain["strict_chain_id"]
        for edge_id in chain.get("edge_ids", ()):
            edge = strict_edges.get(edge_id)
            if not edge or edge.get("edge_type") not in {"CALL", "ASSET"}:
                continue
            src, dst, kind = edge.get("src_node_id"), edge.get("dst_node_id"), edge.get("edge_type")
            semantic_seen.add((cid, kind, src, dst))
            raw_evidence.append((cid, kind, src, dst, _codes(edge.get("uncertainties", ())), (src, dst)))
    unlinked = 0
    for edge in sorted(causal.get("edges", ()), key=lambda item: (item.get("edge_type", ""), item.get("edge_id", ""))):
        kind, src, dst = edge.get("edge_type"), edge.get("src"), edge.get("dst")
        if kind not in EDGE_TYPES:
            continue
        linked = [cid for cid, node_set in chain_nodes.items() if src in node_set and dst in node_set]
        if not linked:
            unlinked += 1
            continue
        uncertainties = set(_codes((edge.get("attributes") or {}).get("uncertainties", ())))
        if edge.get("evidence_level") == "HYPOTHESIS":
            uncertainties.add("HYPOTHESIS_EVIDENCE")
        for cid in linked:
            key = (cid, kind, src, dst)
            if key in semantic_seen:
                continue
            semantic_seen.add(key)
            raw_evidence.append((cid, kind, src, dst, tuple(sorted(uncertainties)), (src, dst)))
    raw_evidence.sort(key=lambda item: (chain_alias[item[0]], {"CALL": 0, "ASSET": 1, "ORDER": 2, "PARAM": 3, "STATE": 4}[item[1]], item[2], item[3]))
    evidence_rows: list[tuple[str, str, str, str, tuple[str, ...], tuple[str, str]]] = []
    missing_endpoint = 0
    descriptions = {"CALL": "observed call relation on this result chain", "ASSET": "observed asset-continuity relation on this result chain", "ORDER": "observed execution-order relation on this result chain", "PARAM": "existing parameter-flow evidence relates these calls", "STATE": "existing state evidence relates these calls"}
    for item in raw_evidence:
        cid, kind, src, dst, uncertainties, endpoints = item
        src_node, dst_node = nodes.get(src), nodes.get(dst)
        src_address = str(src_node.get("callee_address", "")).lower() if src_node else ""
        dst_address = str(dst_node.get("callee_address", "")).lower() if dst_node else ""
        if src_address not in address_alias or dst_address not in address_alias:
            missing_endpoint += 1
            continue
        evidence_rows.append((cid, kind, src_address, dst_address, uncertainties, endpoints))
    issues: list[Phase2BridgeIssue] = []
    candidate_union_issue_counts = {
        "GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK_USED": source_metadata.get("candidate_union", {}).get("fallback_candidate_count", 0),
        "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK": source_metadata.get("candidate_union", {}).get("fallback_candidate_count", 0),
        "GRAPH_LINKED_FUNCTION_FALLBACK_TRUNCATED": source_metadata.get("candidate_union", {}).get("fallback_truncated_candidate_count", 0),
        "PHASE2_SOURCE_FUNCTION_CANDIDATES_TRUNCATED": source_metadata.get("candidate_union", {}).get("truncated_candidate_count", 0),
    } if isinstance(source_metadata.get("candidate_union"), Mapping) else {}
    union_codes = set(_codes(source_metadata.get("candidate_union", {}).get("issue_codes", ()))) if isinstance(source_metadata.get("candidate_union"), Mapping) else set()
    for code, count in candidate_union_issue_counts.items():
        if code in union_codes and isinstance(count, int) and not isinstance(count, bool) and count > 0:
            issues.append(Phase2BridgeIssue(code, "AUDIT", count))
    if len(all_chains) > len(chains_raw):
        issues.append(Phase2BridgeIssue("PHASE2_RESULT_CHAINS_TRUNCATED", "AUDIT", len(all_chains) - len(chains_raw)))
    if unlinked:
        issues.append(Phase2BridgeIssue("UNLINKED_EXTENDED_EVIDENCE_EXCLUDED", "AUDIT", unlinked))
    if missing_endpoint:
        issues.append(Phase2BridgeIssue("EVIDENCE_ENDPOINT_UNRESOLVED", "AUDIT", missing_endpoint))
    if not evidence_rows:
        issues.append(Phase2BridgeIssue("NO_LINKED_CAUSAL_EVIDENCE", "FATAL"))
        return Phase2LocalizationPreparation("INSUFFICIENT_EVIDENCE", None, tuple(issues), len(all_chains), {}, 0, None, 0, {}, "EXACT_FUNCTION_SLICE", source_selection_mode, len(chains_raw))

    # Keep at least one observed evidence row for each selected strict chain,
    # then fill the remaining bounded evidence budget in stable order.
    retained_rows: list[tuple[str, str, str, str, tuple[str, ...], tuple[str, str]]] = []
    retained_chain_ids: set[str] = set()
    for row in evidence_rows:
        if row[0] not in retained_chain_ids:
            retained_rows.append(row); retained_chain_ids.add(row[0])
        if len(retained_rows) == MAX_EVIDENCE_EDGES:
            break
    if len(retained_rows) < MAX_EVIDENCE_EDGES:
        for row in evidence_rows:
            if row not in retained_rows:
                retained_rows.append(row)
            if len(retained_rows) == MAX_EVIDENCE_EDGES:
                break
    if len(evidence_rows) > len(retained_rows):
        issues.append(Phase2BridgeIssue("PHASE2_EVIDENCE_EDGES_TRUNCATED", "AUDIT", len(evidence_rows) - len(retained_rows)))
    evidence: list[Phase2EvidenceEdge] = []
    evidence_nodes: dict[str, tuple[str, str]] = {}
    for cid, kind, src_address, dst_address, uncertainties, endpoints in retained_rows:
        eid = f"E{len(evidence) + 1}"
        evidence.append(Phase2EvidenceEdge(eid, chain_alias[cid], kind, address_alias[src_address], address_alias[dst_address], descriptions[kind], uncertainties))
        evidence_nodes[eid] = endpoints

    anchor_nodes = {item.get("anchor", {}).get("node_id") for item in chains_raw if isinstance(item.get("anchor"), Mapping)}
    asset_nodes = {node for eid, endpoints in evidence_nodes.items() if evidence[int(eid[1:]) - 1].edge_type == "ASSET" for node in endpoints}
    semantic_nodes = {node for eid, endpoints in evidence_nodes.items() if evidence[int(eid[1:]) - 1].edge_type in {"ORDER", "PARAM", "STATE"} for node in endpoints}
    candidates: list[dict[str, Any]] = []
    window_fallback_used = source_metadata.get("resolution_mode") == "VERIFIED_SOURCE_WINDOW_FALLBACK"
    graph_fallback_used = source_selection_mode == "TRANSACTION_CALL_GRAPH_FALLBACK"
    for request in source.get("requests", ()):
        if not isinstance(request, Mapping) or request.get("function_match_status") not in {"RESOLVED", "WINDOW_FALLBACK"}:
            continue
        node_id = request.get("node_id")
        association = request.get("fallback_association") if isinstance(request.get("fallback_association"), Mapping) else None
        association_chain = association.get("strict_chain_id") if association else None
        if graph_fallback_used and isinstance(association_chain, str) and association_chain in chain_alias:
            allowed = tuple(
                item.evidence_id
                for item in evidence
                if item.chain_alias == chain_alias[association_chain]
            )
        else:
            allowed = tuple(sorted((eid for eid, endpoints in evidence_nodes.items() if node_id in endpoints), key=lambda value: int(value[1:])))
        if not allowed:
            continue
        file_path, start, end, text = request.get("file_path"), request.get("start_line"), request.get("end_line"), request.get("source")
        function_name = request.get("signature") or request.get("function_name")
        if not isinstance(file_path, str) or file_path.startswith(("/", "\\")) or ".." in file_path.split("/") or not isinstance(start, int) or not isinstance(end, int) or not isinstance(text, str) or not text or not isinstance(function_name, str):
            continue
        safe_name = request.get("function_name") or function_name.split("(", 1)[0]
        if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", safe_name or "") is None:
            continue
        priority = 0 if node_id in anchor_nodes else 1 if node_id in asset_nodes else 2 if node_id in semantic_nodes else 3
        if association is not None:
            # A direct strict/causal source remains preferable; graph fallback
            # is admitted only as a bounded, coarse transaction-level candidate.
            priority += 20
        if request.get("function_match_status") == "WINDOW_FALLBACK":
            priority += 10
        key = _candidate_key(file_path, function_name, start, end)
        if key is None:
            continue
        union_row = candidate_union_by_key.get(key)
        public = dict(union_row) if union_row is not None else {
            "candidate_alias": "",
            "relative_file": key[0],
            "function_signature": key[1],
            "start_line": key[2],
            "end_line": key[3],
            "selector_match_status": request.get("selector_match_status", "SOURCE_INDEX_RESOLVED"),
            "candidate_sources": list(_codes(request.get("candidate_sources", ()))) or ["LEGACY_VERIFIED_SOURCE_REQUEST"],
            "call_relation_modes": list(_codes(request.get("call_relation_kinds", ()))) or ([association.get("relation_kind")] if association and isinstance(association.get("relation_kind"), str) else ["DIRECT_SOURCE_WITHOUT_CALL_RELATION"]),
            "shortest_call_path_length": association.get("call_path_length") if association else None,
            "internal_expansion_depth": int(request.get("internal_expansion_depth", 0)),
            "supporting_evidence_ids": list(_codes(request.get("supporting_evidence_ids", ()))) or list(allowed),
        }
        candidates.append({
            "priority": priority,
            "node_id": node_id,
            "path": key[0],
            "function": safe_name,
            "start": start,
            "end": end,
            "text": _redact_source(text),
            "allowed": allowed,
            "uncertainties": _codes(request.get("uncertainties", ())),
            "signature": function_name,
            "resolution_mode": request.get("resolution_mode", "EXACT_FUNCTION_SLICE"),
            "candidate_key": key,
            "union_position": candidate_union_order.get(key),
            "public": public,
        })
    candidates.sort(key=lambda item: (
        0 if item["union_position"] is not None else 1,
        item["union_position"] if item["union_position"] is not None else item["priority"],
        item["priority"],
        str(item["node_id"]),
        item["path"],
        item["start"],
        item["end"],
    ))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for item in candidates:
        key = item["candidate_key"]
        if key not in seen:
            seen.add(key); unique.append(item)
    unique = unique[:max_pre_llm_candidates]
    if not candidate_union_rows:
        for index, item in enumerate(unique, start=1):
            item["public"]["candidate_alias"] = f"C{index}"
    if prompt_contract == "topk_v4":
        deterministic_candidates, selection_audit = _multi_view_candidate_selection(
            unique,
            max_prompt_candidates,
            sequence_exact_candidate_keys,
            abnormal_parent_function_keys,
        )
        prompt_candidates = deterministic_candidates
        if shortlist_candidate_ids is not None:
            expected = min(max_prompt_candidates, len(unique))
            candidate_by_alias = {item["public"].get("candidate_alias"): item for item in unique}
            if (
                len(shortlist_candidate_ids) != expected
                or len(set(shortlist_candidate_ids)) != len(shortlist_candidate_ids)
                or any(item not in candidate_by_alias for item in shortlist_candidate_ids)
            ):
                raise ValueError("PHASE2_COARSE_SHORTLIST_INVALID")
            prompt_candidates = [candidate_by_alias[item] for item in shortlist_candidate_ids]
            for item in unique:
                item["public"]["stage_a_rank"] = None
                item["public"]["selection_stage"] = None
            for rank, item in enumerate(prompt_candidates, start=1):
                item["public"]["stage_a_rank"] = rank
                item["public"]["selection_stage"] = "LLM_COARSE_SHORTLIST"
            selection_audit.update({
                "selection_mode": "LLM_COARSE_SHORTLIST",
                "stage_a_selected_count": len(prompt_candidates),
                "stage_a_candidate_ids": list(shortlist_candidate_ids),
            })
            issues.append(Phase2BridgeIssue("COARSE_LLM_SHORTLIST_USED", "AUDIT"))
        if unique:
            issues.append(Phase2BridgeIssue("MULTI_VIEW_CANDIDATE_RANK_FUSION_USED", "AUDIT"))
            issues.append(Phase2BridgeIssue("MULTI_VIEW_CANDIDATE_DIVERSITY_QUOTA_USED", "AUDIT"))
        if selection_audit.get("contract_soft_cap_relaxed"):
            issues.append(Phase2BridgeIssue("MULTI_VIEW_CONTRACT_SOFT_CAP_RELAXED", "AUDIT"))
        if selection_audit.get("sequence_support_selected_count"):
            issues.append(Phase2BridgeIssue("SEQUENCE_EXACT_SUPPORT_USED", "AUDIT", selection_audit["sequence_support_selected_count"]))
        if selection_audit.get("abnormal_parent_selected_count"):
            issues.append(Phase2BridgeIssue("ABNORMAL_PARENT_UNIQUE_CALLEE_USED", "AUDIT", selection_audit["abnormal_parent_selected_count"]))
    else:
        prompt_candidates = unique[:max_prompt_candidates]
        selection_audit = {
            "selection_mode": "LEGACY_PREFIX_SELECTION",
            "quota_selected_count": 0,
            "fusion_fill_selected_count": len(prompt_candidates),
            "contract_soft_cap_delayed_count": 0,
            "contract_soft_cap_relaxed": False,
            "view_unique_contribution_counts": {},
        }
        for item in unique:
            item["public"].update({
                "view_ranks": {},
                "selected_by_views": [],
                "fusion_score": None,
                "selection_stage": "LEGACY_PREFIX" if item in prompt_candidates else None,
                "contract_soft_cap_deferred": False,
            })
    selection_audit.update(candidate_union_audit)
    if len(unique) > len(prompt_candidates):
        issues.append(Phase2BridgeIssue("PHASE2_SOURCE_SLICES_TRUNCATED", "AUDIT", len(unique) - len(prompt_candidates)))
    if not prompt_candidates:
        issues.append(Phase2BridgeIssue("NO_VERIFIED_SOURCE_SLICE", "FATAL"))
        public_pool = tuple({**dict(item), "included_in_prompt": False, "source_slice_id": None} for item in candidate_union_rows)
        return Phase2LocalizationPreparation("INSUFFICIENT_SOURCE", None, tuple(issues), len(all_chains), _edge_counts(evidence), 0, None, 0, {}, "EXACT_FUNCTION_SLICE", source_selection_mode, len(chains_raw), public_pool)

    valid_prompt_candidates: list[dict[str, Any]] = []
    for item in prompt_candidates:
        try:
            Phase2SourceSlice("S1", "F1", item["function"], item["start"], item["end"], item["text"], item["allowed"], item["uncertainties"])
        except ValueError:
            issues.append(Phase2BridgeIssue("SOURCE_SLICE_CONTRACT_REJECTED", "AUDIT"))
            continue
        valid_prompt_candidates.append(item)
    if not valid_prompt_candidates:
        issues.append(Phase2BridgeIssue("NO_VERIFIED_SOURCE_SLICE", "FATAL"))
        pool_source = list(candidate_union_rows) if candidate_union_rows else [dict(item["public"]) for item in unique]
        public_pool = tuple({**row, "included_in_prompt": False, "source_slice_id": None} for row in pool_source)
        return Phase2LocalizationPreparation("INSUFFICIENT_SOURCE", None, tuple(issues), len(all_chains), _edge_counts(evidence), 0, None, 0, {}, "EXACT_FUNCTION_SLICE", source_selection_mode, len(chains_raw), public_pool)

    transaction_context = Phase2TransactionContext(transaction_alias, "transaction routed to phase two for source localization")

    def prompt_causal_tags(item: Mapping[str, Any]) -> tuple[str, ...]:
        if prompt_contract != "topk_v4":
            return ()
        public = item["public"]
        features = public.get("ranking_features")
        tags = []
        if public.get("abnormal_parent_support") is True:
            tags.append("ABNORMAL_PARENT_UNIQUE_CALLEE")
        if isinstance(features, Mapping) and features.get("asset_result_endpoint") is True:
            tags.append("ASSET_RESULT_ENDPOINT")
        path_length = public.get("shortest_call_path_length")
        if isinstance(path_length, int) and not isinstance(path_length, bool) and path_length > 0:
            tags.append("UPSTREAM_OF_RESULT_CHAIN")
        return tuple(tags)

    def build_prompt_selection(selected: list[dict[str, Any]]) -> tuple[Phase2LocalizationPrompt, list[Phase2SourceSlice], dict[str, dict[str, Any]], dict[tuple[str, str, int, int], str], int]:
        files = {path: f"F{index}" for index, path in enumerate(sorted({item["path"] for item in selected}), start=1)}
        built_slices: list[Phase2SourceSlice] = []
        built_locations: dict[str, dict[str, Any]] = {}
        built_selected: dict[tuple[str, str, int, int], str] = {}
        for index, item in enumerate(selected, start=1):
            path, function, start, end = item["path"], item["function"], item["start"], item["end"]
            sid = f"S{index}"
            built_slices.append(Phase2SourceSlice(sid, files[path], function, start, end, item["text"], item["allowed"], item["uncertainties"], prompt_causal_tags(item)))
            built_selected[item["candidate_key"]] = sid
            built_locations[sid] = {"relative_file": path, "function": item["signature"], "start_line": start, "end_line": end, "resolution_mode": item["resolution_mode"]}
        built_prompt = build_phase2_localization_prompt(
            transaction_context,
            chain_contexts,
            tuple(evidence),
            tuple(built_slices),
            max_source_slices=max_prompt_candidates,
        )
        if prompt_contract == "topk_v4":
            built_prompt = upgrade_phase2_localization_prompt_to_topk_v4(built_prompt)
        built_bytes = len(_canonical(built_prompt.to_dict()).encode("utf-8"))
        return built_prompt, built_slices, built_locations, built_selected, built_bytes

    active_prompt_candidates = list(valid_prompt_candidates)
    prompt_count = len(active_prompt_candidates)
    prompt, slices, locations, selected_slices, prompt_bytes = build_prompt_selection(active_prompt_candidates)
    original_prompt_count = prompt_count
    minimum_prompt_count = min(MIN_PROMPT_CANDIDATES_TOPK_V4, prompt_count)
    while max_prompt_bytes is not None and prompt_bytes > max_prompt_bytes and prompt_count > minimum_prompt_count:
        removable_index: int | None = None
        if prompt_contract == "topk_v4":
            view_counts = {
                name: sum(name in item["public"].get("selected_by_views", ()) for item in active_prompt_candidates)
                for name in MULTI_VIEW_NAMES
            }
            for index in range(len(active_prompt_candidates) - 1, -1, -1):
                views = active_prompt_candidates[index]["public"].get("selected_by_views", ())
                if all(view_counts.get(name, 0) > 1 for name in views):
                    removable_index = index
                    break
        if removable_index is None:
            removable_index = len(active_prompt_candidates) - 1
        removed = active_prompt_candidates.pop(removable_index)
        removed["public"]["selection_stage"] = "BYTE_LIMIT_REMOVED"
        prompt_count = len(active_prompt_candidates)
        prompt, slices, locations, selected_slices, prompt_bytes = build_prompt_selection(active_prompt_candidates)
    if prompt_count < original_prompt_count:
        issues.append(Phase2BridgeIssue("PHASE2_PROMPT_BYTE_LIMIT_CANDIDATES_REDUCED", "AUDIT", original_prompt_count - prompt_count))
    if max_prompt_bytes is not None and prompt_bytes > max_prompt_bytes:
        issues.append(Phase2BridgeIssue("PHASE2_PROMPT_BYTE_LIMIT_EXCEEDED", "FATAL", prompt_bytes - max_prompt_bytes))
        pool_source = list(candidate_union_rows) if candidate_union_rows else [dict(item["public"]) for item in unique]
        public_pool = tuple({**row, "included_in_prompt": False, "source_slice_id": None} for row in pool_source)
        return Phase2LocalizationPreparation("STRUCTURAL_ERROR", None, tuple(issues), len(all_chains), _edge_counts(evidence), 0, None, 0, {}, "EXACT_FUNCTION_SLICE", source_selection_mode, len(chains_raw), public_pool, selection_audit)

    annotated_public = {item["candidate_key"]: dict(item["public"]) for item in unique}
    if candidate_union_rows:
        pool_source = [
            annotated_public.get(
                _candidate_key(row["relative_file"], row["function_signature"], row["start_line"], row["end_line"]),
                {**dict(row), "view_ranks": {}, "selected_by_views": [], "fusion_score": None, "selection_stage": None, "contract_soft_cap_deferred": False},
            )
            for row in candidate_union_rows
        ]
    else:
        pool_source = [dict(item["public"]) for item in unique]
    public_pool = tuple({
        **row,
        "included_in_prompt": _candidate_key(row["relative_file"], row["function_signature"], row["start_line"], row["end_line"]) in selected_slices,
        "source_slice_id": selected_slices.get(_candidate_key(row["relative_file"], row["function_signature"], row["start_line"], row["end_line"])),
    } for row in pool_source)
    if window_fallback_used:
        issues.append(Phase2BridgeIssue("SOURCE_WINDOW_FALLBACK_USED", "AUDIT", len(slices)))
    if graph_fallback_used:
        issues.append(Phase2BridgeIssue("TRANSACTION_CALL_GRAPH_SOURCE_FALLBACK_USED", "AUDIT", len(slices)))
        issues.append(Phase2BridgeIssue("SOURCE_TO_RESULT_CALL_PATH_COARSE", "AUDIT", len(slices)))
    encoded = _canonical(prompt.to_dict()).encode("utf-8")
    selection_audit.update({
        "full_candidate_count": len(pool_source),
        "prompt_candidate_count": len(slices),
        "fallback_candidate_count": sum(
            "GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK" in row.get("candidate_sources", ())
            for row in pool_source
        ),
        "prompt_bytes": len(encoded),
        "prompt_byte_limit_candidates_reduced": prompt_count < original_prompt_count,
        "prompt_byte_limit_candidates_reduced_count": original_prompt_count - prompt_count,
    })
    resolution_mode = "TRANSACTION_CALL_GRAPH_FALLBACK" if graph_fallback_used else "VERIFIED_SOURCE_WINDOW_FALLBACK" if window_fallback_used else "EXACT_FUNCTION_SLICE"
    return Phase2LocalizationPreparation("READY_FOR_LLM", prompt, tuple(issues), len(all_chains), _edge_counts(evidence), len(slices), hashlib.sha256(encoded).hexdigest(), len(encoded), locations, resolution_mode, source_selection_mode, len(chains_raw), public_pool, selection_audit)


def _edge_counts(edges: list[Phase2EvidenceEdge]) -> dict[str, int]:
    return {kind: sum(item.edge_type == kind for item in edges) for kind in sorted(EDGE_TYPES)}


def serialize_phase2_localization_preparation(preparation: Phase2LocalizationPreparation) -> str:
    if not isinstance(preparation, Phase2LocalizationPreparation):
        raise ValueError("PHASE2_PREPARATION_INVALID")
    return json.dumps(preparation.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

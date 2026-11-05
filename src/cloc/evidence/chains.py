"""Extract bounded CALL/ASSET paths ending at asset outcomes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Mapping

from cloc.source.identity import (
    FunctionIdentityResolution,
    OfflineFunctionCandidate,
    resolve_function_identity,
)
from cloc.evidence.models import NormalizedCallNode, TransactionGraph
from cloc.evidence.outflow_scoring import scoring_context_id


_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_TRANSFER_SELECTORS = {"0xa9059cbb", "0x23b872dd"}
_EDGE_ORDER = {"CALL": 0, "ASSET": 1}


@dataclass(frozen=True)
class StrictEvidenceEdge:
    """One allowed runtime relation in a strict result chain."""

    edge_id: str
    edge_type: str
    src_node_id: str
    dst_node_id: str
    source_edge_id: str
    src_fact_id: str | None = None
    dst_fact_id: str | None = None
    asset_type: str | None = None
    asset_address: str | None = None
    via_address: str | None = None
    allocated_amount_raw: str | None = None
    match_type: str | None = None
    evidence_level: str | None = None
    uncertainties: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "source_edge_id": self.source_edge_id,
            "src_fact_id": self.src_fact_id,
            "dst_fact_id": self.dst_fact_id,
            "asset_type": self.asset_type,
            "asset_address": self.asset_address,
            "via_address": self.via_address,
            "allocated_amount_raw": self.allocated_amount_raw,
            "match_type": self.match_type,
            "evidence_level": self.evidence_level,
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class CriticalUncertainty:
    """A malformed ASSET input that must not disappear silently."""

    code: str
    source_edge_id: str | None
    related_node_ids: tuple[str, ...] = ()
    related_fact_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "source_edge_id": self.source_edge_id,
            "related_node_ids": list(self.related_node_ids),
            "related_fact_ids": list(self.related_fact_ids),
        }


@dataclass(frozen=True)
class MainAnchorCandidate:
    """A trace-only anchor candidate; implementation resolution is deferred."""

    blockchain_chain_id: int | str | None
    node_id: str
    runtime_address: str
    effective_address: str
    address_resolution: str
    selector: str
    function_identity_source: str
    candidate_signatures: tuple[str, ...]
    candidate_selectors: tuple[str, ...]
    candidate_evidence_sources: tuple[str, ...]
    function_identity_uncertainties: tuple[str, ...]
    execution_status: str
    execution_status_source: str
    execution_success_evidence: bool
    execution_status_uncertainties: tuple[str, ...]
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockchain_chain_id": self.blockchain_chain_id,
            "node_id": self.node_id,
            "runtime_address": self.runtime_address,
            "effective_address": self.effective_address,
            "address_resolution": self.address_resolution,
            "selector": self.selector,
            "function_identity_source": self.function_identity_source,
            "candidate_signatures": list(self.candidate_signatures),
            "candidate_selectors": list(self.candidate_selectors),
            "candidate_evidence_sources": list(self.candidate_evidence_sources),
            "function_identity_uncertainties": list(self.function_identity_uncertainties),
            "execution_status": self.execution_status,
            "execution_status_source": self.execution_status_source,
            "execution_success_evidence": self.execution_success_evidence,
            "execution_status_uncertainties": list(self.execution_status_uncertainties),
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class StrictResultChain:
    """A single forward-ordered path ending at one result candidate."""

    strict_chain_id: str
    result_id: str
    result_node_id: str
    result_fact_ids: tuple[str, ...]
    result_type: str | None
    asset_type: str | None
    asset_address: str | None
    from_address: str | None
    to_address: str | None
    amount_raw: str | None
    result_uncertainties: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    anchor: MainAnchorCandidate | None
    status: str
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strict_chain_id": self.strict_chain_id,
            "result_id": self.result_id,
            "result_node_id": self.result_node_id,
            "result_fact_ids": list(self.result_fact_ids),
            "result_type": self.result_type,
            "asset_type": self.asset_type,
            "asset_address": self.asset_address,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "amount_raw": self.amount_raw,
            "result_uncertainties": list(self.result_uncertainties),
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "anchor": self.anchor.to_dict() if self.anchor is not None else None,
            "status": self.status,
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class ResultChainStatus:
    """Coverage and uncertainty status for exactly one result candidate."""

    result_id: str
    result_node_id: str | None
    strict_chain_ids: tuple[str, ...]
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "result_node_id": self.result_node_id,
            "strict_chain_ids": list(self.strict_chain_ids),
            "status": self.status,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class StrictResultChainsResult:
    """Serializable document for the strict first-stage representation."""

    blockchain_chain_id: int | str | None
    edges: tuple[StrictEvidenceEdge, ...]
    chains: tuple[StrictResultChain, ...]
    result_statuses: tuple[ResultChainStatus, ...]
    critical_uncertainties: tuple[CriticalUncertainty, ...]
    coverage: Mapping[str, Any]
    result_scoring_context: str = "legacy_independent_weights"

    SCHEMA_VERSION = "strict_result_chains_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "blockchain_chain_id": self.blockchain_chain_id,
            "edges": [edge.to_dict() for edge in self.edges],
            "chains": [chain.to_dict() for chain in self.chains],
            "result_statuses": [status.to_dict() for status in self.result_statuses],
            "critical_uncertainties": [item.to_dict() for item in self.critical_uncertainties],
            "coverage": dict(self.coverage),
            **({"result_scoring_context": self.result_scoring_context} if self.result_scoring_context != "legacy_independent_weights" else {}),
        }


@dataclass(frozen=True)
class _Path:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    uncertainties: tuple[str, ...]


@dataclass(frozen=True)
class _OfflineAddressResolution:
    valid: Mapping[str, str]
    invalid_values: Mapping[str, Any]
    uncertainties: tuple[str, ...]


def build_strict_result_chains(
    transaction_graph: TransactionGraph,
    asset_continuity: Mapping[str, Any],
    result_candidates: Mapping[str, Any],
    *,
    blockchain_chain_id: int | str | None = None,
    offline_function_candidates: Iterable[OfflineFunctionCandidate] = (),
    offline_effective_address_map: Mapping[str, str] | None = None,
) -> StrictResultChainsResult:
    """Extract every deterministic CALL/ASSET path ending at each result node.

    ``blockchain_chain_id`` is intentionally explicit.  Existing local chain
    labels such as ``chain_c001`` are never accepted or inferred as a network
    chain identifier.
    """
    _validate_inputs(transaction_graph, asset_continuity, result_candidates)
    nodes = {node.node_id: node for node in transaction_graph.nodes}
    edges, critical_uncertainties = _build_edges(transaction_graph, asset_continuity, nodes)
    incoming = _incoming_edges(edges, nodes)
    chain_id_reasons = _blockchain_chain_id_reasons(blockchain_chain_id)
    function_candidates, candidate_input_reasons = _function_candidates(offline_function_candidates)
    effective_addresses = _normalize_effective_address_map(offline_effective_address_map)
    resolver_input_reasons = tuple(
        _sorted_unique((*candidate_input_reasons, *effective_addresses.uncertainties))
    )

    all_chains: list[StrictResultChain] = []
    statuses: list[ResultChainStatus] = []
    next_chain_number = 1
    candidates = sorted(
        result_candidates.get("result_candidates", ()),
        key=lambda item: (str(item.get("result_id", "")), str(item.get("node_id", ""))),
    )

    for candidate in candidates:
        result_id = str(candidate.get("result_id", ""))
        result_node_id = candidate.get("node_id")
        result_evidence, result_reasons = _result_evidence(candidate)
        if not result_id:
            raise ValueError("Result candidate is missing result_id")
        if not isinstance(result_node_id, str) or result_node_id not in nodes:
            statuses.append(
                ResultChainStatus(
                    result_id=result_id,
                    result_node_id=result_node_id if isinstance(result_node_id, str) else None,
                    strict_chain_ids=(),
                    status="UNCERTAIN",
                    reasons=tuple(
                        _sorted_unique(("RESULT_NODE_MISSING", *result_reasons, *chain_id_reasons))
                    ),
                )
            )
            continue

        paths = _deduplicate_paths(_enumerate_paths(result_node_id, incoming, nodes))
        if not paths:
            statuses.append(
                ResultChainStatus(
                    result_id=result_id,
                    result_node_id=result_node_id,
                    strict_chain_ids=(),
                    status="UNCERTAIN",
                    reasons=tuple(
                        _sorted_unique(("NO_USABLE_STRICT_PATH", *result_reasons, *chain_id_reasons))
                    ),
                )
            )
            continue

        candidate_chains: list[StrictResultChain] = []
        fact_ids = tuple(sorted(str(item) for item in candidate.get("source_fact_ids", ()) if isinstance(item, str)))
        critical_reasons = _related_critical_reasons(
            critical_uncertainties,
            result_node_id,
            fact_ids,
            paths,
        )
        for path in paths:
            anchor, anchor_search_reasons = _find_anchor(
                path.node_ids,
                nodes,
                blockchain_chain_id,
                function_candidates,
                effective_addresses,
                resolver_input_reasons,
            )
            reasons = [
                *path.uncertainties,
                *result_reasons,
                *critical_reasons,
                *anchor_search_reasons,
                *chain_id_reasons,
            ]
            if anchor is None:
                reasons.append("ANCHOR_NOT_FOUND")
            else:
                reasons.extend(anchor.uncertainties)
            chain = StrictResultChain(
                strict_chain_id=f"strict_result_chain_{next_chain_number:03d}",
                result_id=result_id,
                result_node_id=result_node_id,
                result_fact_ids=fact_ids,
                result_type=result_evidence["result_type"],
                asset_type=result_evidence["asset_type"],
                asset_address=result_evidence["asset_address"],
                from_address=result_evidence["from_address"],
                to_address=result_evidence["to_address"],
                amount_raw=result_evidence["amount_raw"],
                result_uncertainties=result_evidence["uncertainties"],
                node_ids=path.node_ids,
                edge_ids=path.edge_ids,
                anchor=anchor,
                status="UNCERTAIN" if reasons else "READY",
                uncertainties=tuple(_sorted_unique(reasons)),
            )
            next_chain_number += 1
            candidate_chains.append(chain)
            all_chains.append(chain)

        candidate_reasons = _sorted_unique(
            reason for chain in candidate_chains for reason in chain.uncertainties
        )
        statuses.append(
            ResultChainStatus(
                result_id=result_id,
                result_node_id=result_node_id,
                strict_chain_ids=tuple(chain.strict_chain_id for chain in candidate_chains),
                status="UNCERTAIN" if candidate_reasons else "READY",
                reasons=tuple(candidate_reasons),
            )
        )

    coverage = _coverage(statuses, critical_uncertainties)
    return StrictResultChainsResult(
        blockchain_chain_id=blockchain_chain_id,
        edges=tuple(edges),
        chains=tuple(all_chains),
        result_statuses=tuple(statuses),
        critical_uncertainties=tuple(critical_uncertainties),
        coverage=coverage,
        result_scoring_context=(scoring_context_id(result_candidates["detection_config"]) if "detection_config" in result_candidates else "legacy_independent_weights"),
    )


def serialize_strict_result_chains(result: StrictResultChainsResult) -> str:
    """Return stable JSON without creating an output artifact."""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def _validate_inputs(
    transaction_graph: TransactionGraph,
    asset_continuity: Mapping[str, Any],
    result_candidates: Mapping[str, Any],
) -> None:
    if not isinstance(transaction_graph, TransactionGraph):
        raise ValueError("transaction_graph must be a TransactionGraph")
    if asset_continuity.get("schema_version") != "asset_continuity_v1":
        raise ValueError("asset_continuity must be asset_continuity_v1")
    if result_candidates.get("schema_version") != "result_candidates_v1":
        raise ValueError("result_candidates must be result_candidates_v1")
    seen: set[str] = set()
    for candidate in result_candidates.get("result_candidates", ()):
        result_id = candidate.get("result_id") if isinstance(candidate, Mapping) else None
        if not isinstance(result_id, str) or not result_id or result_id in seen:
            raise ValueError("Result candidates require unique non-empty result_id values")
        seen.add(result_id)


def _build_edges(
    transaction_graph: TransactionGraph,
    asset_continuity: Mapping[str, Any],
    nodes: Mapping[str, NormalizedCallNode],
) -> tuple[list[StrictEvidenceEdge], list[CriticalUncertainty]]:
    edges: list[StrictEvidenceEdge] = []
    critical: list[CriticalUncertainty] = []
    for edge in transaction_graph.call_edges:
        edges.append(
            StrictEvidenceEdge(
                edge_id=f"strict_call_{edge.edge_id}",
                edge_type="CALL",
                src_node_id=edge.src,
                dst_node_id=edge.dst,
                source_edge_id=edge.edge_id,
                evidence_level=edge.evidence_level,
            )
        )
    for index, raw in enumerate(asset_continuity.get("asset_edges", ())):
        if not isinstance(raw, Mapping):
            critical.append(
                CriticalUncertainty(
                    code="ASSET_EDGE_NOT_OBJECT",
                    source_edge_id=None,
                )
            )
            continue
        source = raw.get("src_node_id")
        target = raw.get("dst_node_id")
        source_edge_id = raw.get("edge_id")
        related_nodes = tuple(
            sorted(item for item in (source, target) if isinstance(item, str) and item)
        )
        related_facts = tuple(
            sorted(
                item
                for item in (raw.get("src_fact_id"), raw.get("dst_fact_id"))
                if isinstance(item, str) and item
            )
        )
        critical_source_id = source_edge_id if isinstance(source_edge_id, str) and source_edge_id else None
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
            critical.append(
                CriticalUncertainty(
                    code="ASSET_EDGE_ENDPOINT_MISSING",
                    source_edge_id=critical_source_id,
                    related_node_ids=related_nodes,
                    related_fact_ids=related_facts,
                )
            )
            continue
        if source not in nodes or target not in nodes:
            critical.append(
                CriticalUncertainty(
                    code="ASSET_EDGE_ENDPOINT_UNKNOWN",
                    source_edge_id=critical_source_id,
                    related_node_ids=related_nodes,
                    related_fact_ids=related_facts,
                )
            )
            continue
        if not isinstance(source_edge_id, str) or not source_edge_id:
            critical.append(
                CriticalUncertainty(
                    code="ASSET_EDGE_ID_MISSING",
                    source_edge_id=None,
                    related_node_ids=related_nodes,
                    related_fact_ids=related_facts,
                )
            )
            continue

        metadata, metadata_codes = _asset_edge_metadata(raw)
        critical.extend(
            CriticalUncertainty(
                code=code,
                source_edge_id=source_edge_id,
                related_node_ids=related_nodes,
                related_fact_ids=related_facts,
            )
            for code in metadata_codes
        )
        uncertainties = tuple(
            str(item) for item in raw.get("uncertainties", ()) if isinstance(item, str)
        )
        edges.append(
            StrictEvidenceEdge(
                edge_id=f"strict_asset_{source_edge_id}",
                edge_type="ASSET",
                src_node_id=source,
                dst_node_id=target,
                source_edge_id=source_edge_id,
                src_fact_id=metadata["src_fact_id"],
                dst_fact_id=metadata["dst_fact_id"],
                asset_type=metadata["asset_type"],
                asset_address=metadata["asset_address"],
                via_address=metadata["via_address"],
                allocated_amount_raw=metadata["allocated_amount_raw"],
                match_type=metadata["match_type"],
                evidence_level=metadata["evidence_level"],
                uncertainties=uncertainties,
            )
        )
    sorted_edges = sorted(
        edges,
        key=lambda edge: (
            nodes[edge.src_node_id].trace_index,
            nodes[edge.dst_node_id].trace_index,
            _EDGE_ORDER[edge.edge_type],
            edge.source_edge_id,
            edge.edge_id,
        ),
    )
    sorted_critical = sorted(
        set(critical),
        key=lambda item: (
            item.code,
            item.source_edge_id or "",
            item.related_node_ids,
            item.related_fact_ids,
        ),
    )
    return sorted_edges, sorted_critical


def _asset_edge_metadata(raw: Mapping[str, Any]) -> tuple[dict[str, str | None], tuple[str, ...]]:
    values: dict[str, str | None] = {}
    codes: list[str] = []
    for field in (
        "src_fact_id",
        "dst_fact_id",
        "asset_type",
        "allocated_amount_raw",
        "match_type",
        "evidence_level",
    ):
        value = raw.get(field)
        if isinstance(value, str) and value:
            values[field] = value
        else:
            values[field] = None
            codes.append(f"ASSET_EDGE_{field.upper()}_MISSING")

    asset_type = values["asset_type"]
    raw_asset_address = raw.get("asset_address")
    asset_address = _normalize_address(raw_asset_address)
    if asset_address is None and asset_type != "NATIVE":
        codes.append("ASSET_EDGE_ASSET_ADDRESS_INVALID")
    values["asset_address"] = asset_address

    via_address = _normalize_address(raw.get("via_address"))
    if via_address is None:
        codes.append("ASSET_EDGE_VIA_ADDRESS_INVALID")
    values["via_address"] = via_address
    return values, tuple(_sorted_unique(codes))


def _incoming_edges(
    edges: list[StrictEvidenceEdge],
    nodes: Mapping[str, NormalizedCallNode],
) -> dict[str, tuple[StrictEvidenceEdge, ...]]:
    incoming: dict[str, list[StrictEvidenceEdge]] = {node_id: [] for node_id in nodes}
    for edge in edges:
        incoming[edge.dst_node_id].append(edge)
    return {
        node_id: tuple(
            sorted(
                candidates,
                key=lambda edge: (
                    nodes[edge.src_node_id].trace_index,
                    _EDGE_ORDER[edge.edge_type],
                    edge.source_edge_id,
                    edge.edge_id,
                ),
            )
        )
        for node_id, candidates in incoming.items()
    }


def _enumerate_paths(
    result_node_id: str,
    incoming: Mapping[str, tuple[StrictEvidenceEdge, ...]],
    nodes: Mapping[str, NormalizedCallNode],
) -> list[_Path]:
    def visit(
        current_node_id: str,
        reverse_node_ids: tuple[str, ...],
        reverse_edge_ids: tuple[str, ...],
        visited: frozenset[str],
        reasons: tuple[str, ...],
    ) -> list[_Path]:
        candidates = incoming.get(current_node_id, ())
        cycle_reasons = tuple(
            f"CYCLE_EDGE_SKIPPED:{edge.source_edge_id}"
            for edge in candidates
            if edge.src_node_id in visited
        )
        available = tuple(edge for edge in candidates if edge.src_node_id not in visited)
        next_reasons = tuple(_sorted_unique((*reasons, *cycle_reasons)))
        if not available:
            return [
                _Path(
                    node_ids=tuple(reversed(reverse_node_ids)),
                    edge_ids=tuple(reversed(reverse_edge_ids)),
                    uncertainties=next_reasons,
                )
            ]

        paths: list[_Path] = []
        for edge in available:
            edge_reasons = tuple(_sorted_unique((*next_reasons, *edge.uncertainties)))
            paths.extend(
                visit(
                    edge.src_node_id,
                    reverse_node_ids + (edge.src_node_id,),
                    reverse_edge_ids + (edge.edge_id,),
                    visited | {edge.src_node_id},
                    edge_reasons,
                )
            )
        return paths

    paths = visit(result_node_id, (result_node_id,), (), frozenset({result_node_id}), ())
    return sorted(
        paths,
        key=lambda path: (
            tuple(nodes[node_id].trace_index for node_id in path.node_ids),
            path.edge_ids,
        ),
    )


def _deduplicate_paths(paths: list[_Path]) -> list[_Path]:
    deduplicated: dict[tuple[tuple[str, ...], tuple[str, ...]], _Path] = {}
    for path in paths:
        key = path.node_ids, path.edge_ids
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = path
        else:
            deduplicated[key] = _Path(
                node_ids=path.node_ids,
                edge_ids=path.edge_ids,
                uncertainties=tuple(_sorted_unique((*existing.uncertainties, *path.uncertainties))),
            )
    return list(deduplicated.values())


def _result_evidence(candidate: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    reasons: list[str] = []
    values: dict[str, Any] = {}
    for field in ("result_type", "asset_type", "amount_raw"):
        value = candidate.get(field)
        if isinstance(value, str) and value:
            values[field] = value
        else:
            values[field] = None
            reasons.append(f"RESULT_{field.upper()}_MISSING")

    asset_address = _normalize_address(candidate.get("asset_address"))
    if asset_address is None and values["asset_type"] != "NATIVE":
        reasons.append("RESULT_ASSET_ADDRESS_INVALID")
    values["asset_address"] = asset_address

    for source_field, output_field in (("from_address", "from_address"), ("to_address", "to_address")):
        normalized = _normalize_address(candidate.get(source_field))
        if normalized is None:
            reasons.append(f"RESULT_{source_field.upper()}_INVALID")
        values[output_field] = normalized

    original_uncertainties = tuple(
        _sorted_unique(
            item for item in candidate.get("uncertainties", ()) if isinstance(item, str) and item
        )
    )
    values["uncertainties"] = original_uncertainties
    reasons.extend(original_uncertainties)
    return values, tuple(_sorted_unique(reasons))


def _normalize_address(value: Any) -> str | None:
    if isinstance(value, Mapping):
        normalized = value.get("normalized")
        original = value.get("original")
        value = normalized if isinstance(normalized, str) and normalized else original
    if not isinstance(value, str) or not _ADDRESS.fullmatch(value):
        return None
    return value.lower()


def _related_critical_reasons(
    critical_uncertainties: list[CriticalUncertainty],
    result_node_id: str,
    fact_ids: tuple[str, ...],
    paths: list[_Path],
) -> tuple[str, ...]:
    path_node_ids = {node_id for path in paths for node_id in path.node_ids}
    candidate_fact_ids = set(fact_ids)
    reasons = []
    for item in critical_uncertainties:
        has_context = bool(item.related_node_ids or item.related_fact_ids)
        is_related = (
            not has_context
            or result_node_id in item.related_node_ids
            or bool(path_node_ids.intersection(item.related_node_ids))
            or bool(candidate_fact_ids.intersection(item.related_fact_ids))
        )
        if is_related:
            reasons.append(item.code)
    return tuple(_sorted_unique(reasons))


def _function_candidates(
    values: Iterable[OfflineFunctionCandidate],
) -> tuple[tuple[OfflineFunctionCandidate, ...], tuple[str, ...]]:
    try:
        supplied = tuple(values)
    except TypeError:
        return (), ("OFFLINE_FUNCTION_CANDIDATES_INVALID",)
    valid = tuple(item for item in supplied if isinstance(item, OfflineFunctionCandidate))
    reasons = () if len(valid) == len(supplied) else ("OFFLINE_FUNCTION_CANDIDATE_INVALID",)
    return valid, reasons


def _normalize_effective_address_map(
    values: Mapping[str, str] | None,
) -> _OfflineAddressResolution:
    if values is None:
        return _OfflineAddressResolution(valid={}, invalid_values={}, uncertainties=())
    if not isinstance(values, Mapping):
        return _OfflineAddressResolution(
            valid={},
            invalid_values={},
            uncertainties=("OFFLINE_EFFECTIVE_ADDRESS_MAP_INVALID",),
        )

    valid: dict[str, str] = {}
    invalid_values: dict[str, Any] = {}
    reasons: list[str] = []
    for raw_runtime, raw_effective in values.items():
        runtime = _normalize_hex_address(raw_runtime)
        effective = _normalize_hex_address(raw_effective)
        if runtime is None:
            reasons.append("OFFLINE_EFFECTIVE_ADDRESS_MAP_RUNTIME_ADDRESS_INVALID")
            continue
        if effective is None:
            valid.pop(runtime, None)
            invalid_values[runtime] = raw_effective
            reasons.append("OFFLINE_EFFECTIVE_ADDRESS_MAP_EFFECTIVE_ADDRESS_INVALID")
            continue
        if runtime in invalid_values:
            continue
        existing = valid.get(runtime)
        if existing is not None and existing != effective:
            valid.pop(runtime, None)
            invalid_values[runtime] = None
            reasons.append("OFFLINE_EFFECTIVE_ADDRESS_MAP_CONFLICT")
            continue
        valid[runtime] = effective
    return _OfflineAddressResolution(
        valid=valid,
        invalid_values=invalid_values,
        uncertainties=tuple(_sorted_unique(reasons)),
    )


def _effective_address_input(
    runtime_address: Any,
    addresses: _OfflineAddressResolution,
) -> tuple[Any, str]:
    runtime = _normalize_hex_address(runtime_address)
    if runtime is not None and runtime in addresses.invalid_values:
        return "", "OFFLINE_EFFECTIVE_ADDRESS_MAP_INVALID"
    if runtime is not None and runtime in addresses.valid:
        return addresses.valid[runtime], "OFFLINE_EFFECTIVE_ADDRESS_MAP"
    return None, "RUNTIME_ADDRESS_FALLBACK"


def _find_anchor(
    node_ids: tuple[str, ...],
    nodes: Mapping[str, NormalizedCallNode],
    blockchain_chain_id: int | str | None,
    offline_function_candidates: tuple[OfflineFunctionCandidate, ...],
    effective_addresses: _OfflineAddressResolution,
    resolver_input_reasons: tuple[str, ...],
) -> tuple[MainAnchorCandidate | None, tuple[str, ...]]:
    reasons: list[str] = list(resolver_input_reasons)
    fallback_anchor: MainAnchorCandidate | None = None
    for node_id in reversed(node_ids):
        node = nodes[node_id]
        observed_standard_asset_call = _is_standard_asset_call(node)
        effective_input, address_resolution = _effective_address_input(
            node.callee_address,
            effective_addresses,
        )
        identity = resolve_function_identity(
            blockchain_chain_id=_normalized_blockchain_chain_id(blockchain_chain_id),
            runtime_address=node.callee_address,
            effective_address=effective_input,
            trace_selector=node.function_selector,
            function_name=node.function_name,
            offline_candidates=offline_function_candidates,
        )
        if observed_standard_asset_call:
            continue
        if identity.selector in _TRANSFER_SELECTORS:
            reasons.extend(identity.uncertainties)
            continue
        status_reasons = _anchor_status_reasons(node)
        if identity.selector is None:
            reasons.extend(identity.uncertainties)
            reasons.append("CLOSER_NON_ASSET_CALL_SELECTOR_MISSING")
            reasons.extend(status_reasons)
            continue
        anchor = _anchor_from_identity(node_id, identity, address_resolution, node)
        if anchor is None:
            reasons.extend(identity.uncertainties)
            reasons.extend(status_reasons)
            continue
        if identity.usable_for_anchor and _has_explicit_success_evidence(node):
            return anchor, tuple(_sorted_unique(reasons))
        reasons.extend(identity.uncertainties)
        reasons.extend(status_reasons)
        if fallback_anchor is None:
            fallback_anchor = anchor
    return fallback_anchor, tuple(_sorted_unique(reasons))


def _anchor_from_identity(
    node_id: str,
    identity: FunctionIdentityResolution,
    address_resolution: str,
    node: NormalizedCallNode,
) -> MainAnchorCandidate | None:
    if (
        identity.selector is None
        or identity.runtime_address is None
        or identity.effective_address is None
    ):
        return None
    identity_reasons = tuple(_sorted_unique(identity.uncertainties))
    status_reasons = _anchor_status_reasons(node)
    return MainAnchorCandidate(
        blockchain_chain_id=identity.blockchain_chain_id,
        node_id=node_id,
        runtime_address=identity.runtime_address,
        effective_address=identity.effective_address,
        address_resolution=address_resolution,
        selector=identity.selector,
        function_identity_source=identity.resolution_source,
        candidate_signatures=identity.candidate_signatures,
        candidate_selectors=identity.candidate_selectors,
        candidate_evidence_sources=identity.candidate_evidence_sources,
        function_identity_uncertainties=identity_reasons,
        execution_status=node.execution_status,
        execution_status_source=node.execution_status_source,
        execution_success_evidence=node.execution_success_evidence,
        execution_status_uncertainties=tuple(_sorted_unique(node.execution_status_uncertainties)),
        uncertainties=tuple(
            _sorted_unique((*identity_reasons, *status_reasons))
        ),
    )


def _has_explicit_success_evidence(node: NormalizedCallNode) -> bool:
    """Accept only a node's own complete callTracer SUCCESS evidence."""
    return (
        node.execution_status == "SUCCESS"
        and node.execution_success_evidence is True
        and not node.execution_status_uncertainties
    )


def _anchor_status_reasons(node: NormalizedCallNode) -> tuple[str, ...]:
    """Return conservative, machine-readable reasons for a non-anchor status."""
    reasons = list(node.execution_status_uncertainties)
    if _has_explicit_success_evidence(node):
        return tuple(_sorted_unique(reasons))
    if node.execution_status == "FAILED":
        reasons.append("CLOSER_NON_ASSET_CALL_FAILED")
    else:
        reasons.append("CLOSER_NON_ASSET_CALL_STATUS_UNKNOWN")
    return tuple(_sorted_unique(reasons))


def _normalized_blockchain_chain_id(blockchain_chain_id: int | str | None) -> int | None:
    if isinstance(blockchain_chain_id, bool):
        return None
    if isinstance(blockchain_chain_id, int):
        return blockchain_chain_id if blockchain_chain_id > 0 else None
    if isinstance(blockchain_chain_id, str):
        value = blockchain_chain_id.strip()
        if value.isdecimal() and int(value) > 0:
            return int(value)
    return None


def _blockchain_chain_id_reasons(blockchain_chain_id: int | str | None) -> tuple[str, ...]:
    if isinstance(blockchain_chain_id, bool):
        return ("BLOCKCHAIN_CHAIN_ID_MISSING",)
    if isinstance(blockchain_chain_id, int):
        return () if blockchain_chain_id > 0 else ("BLOCKCHAIN_CHAIN_ID_MISSING",)
    if isinstance(blockchain_chain_id, str):
        value = blockchain_chain_id.strip()
        if value.isdecimal() and int(value) > 0:
            return ()
    return ("BLOCKCHAIN_CHAIN_ID_MISSING",)


def _is_standard_asset_call(node: NormalizedCallNode) -> bool:
    selector = (node.function_selector or "").lower()
    if selector in _TRANSFER_SELECTORS:
        return True
    name = (node.function_name or "").split("(", 1)[0].strip()
    return name in {"transfer", "transferFrom"}


def _normalize_hex_address(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and _ADDRESS.fullmatch(value) else None


def _coverage(
    statuses: list[ResultChainStatus],
    critical_uncertainties: list[CriticalUncertainty],
) -> dict[str, Any]:
    total = len(statuses)
    formed = sum(bool(status.strict_chain_ids) for status in statuses)
    uncertain = sum(status.status == "UNCERTAIN" for status in statuses)
    rate = Decimal(formed) / Decimal(total) if total else Decimal(1)
    return {
        "result_candidate_count": total,
        "chains_formed_result_candidate_count": formed,
        "uncertain_result_candidate_count": uncertain,
        "critical_missing_count": len(critical_uncertainties),
        "coverage_rate": f"{rate:.4f}",
    }


def _sorted_unique(values: Any) -> list[str]:
    return sorted({str(value) for value in values})


__all__ = [
    "CriticalUncertainty",
    "MainAnchorCandidate",
    "ResultChainStatus",
    "StrictEvidenceEdge",
    "StrictResultChain",
    "StrictResultChainsResult",
    "build_strict_result_chains",
    "serialize_strict_result_chains",
]

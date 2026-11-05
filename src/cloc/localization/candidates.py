"""Resolve source candidates from execution and verified cache bindings."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import re
from typing import Any, Mapping

from cloc.source.cache import norm_address
from cloc.source.index import build_source_index, resolve_unique_internal_function_calls
from cloc.source.slices import build_source_request_slices
from cloc.source.windows import build_verified_source_window_fallback
from cloc.source.verified_cache import VerifiedSourceCacheIndex, lookup_verified_source_cache

MODE = "TRANSACTION_CALL_GRAPH_FALLBACK"
MAX_RESULT_CHAINS = 20
MAX_SOURCE_SLICES = 12
MAX_SOURCE_FUNCTION_CANDIDATES = 20
MAX_SOURCE_FUNCTION_CANDIDATES_RANKING = 100
MAX_INTERNAL_EXPANSION_DEPTH = 2

_MATCH_SOURCES = {
    "effective_address": "EFFECTIVE_ADDRESS_MATCH",
    "runtime_address": "RUNTIME_ADDRESS_MATCH",
    "proxy_implementation": "PROXY_IMPLEMENTATION_MATCH",
    "requested_address": "DIRECT_EVIDENCE_ADDRESS",
}
_ADDRESS_MATCH_PRIORITY = {
    "EFFECTIVE_ADDRESS_MATCH": 0,
    "PROXY_IMPLEMENTATION_MATCH": 1,
    "RUNTIME_ADDRESS_MATCH": 2,
    "DIRECT_EVIDENCE_ADDRESS": 3,
    "TRANSACTION_CALL_GRAPH_FALLBACK": 4,
}
_RELATION_PRIORITY = {
    "STRICT_CHAIN_NODE": 0,
    "DIRECTED_TO_RESULT": 1,
    "DIRECTED_FROM_STRICT_CHAIN": 2,
    "UNDIRECTED_CALL_CORRIDOR": 3,
    "DIRECT_SOURCE_WITHOUT_CALL_RELATION": 4,
}
_SELECTOR_PRIORITY = {
    "SELECTOR_EXACT_MATCH": 0,
    "UNIQUE_SIGNATURE_MATCH": 1,
    "UNIQUE_NAME_MATCH": 2,
    "INTERNAL_UNIQUE_CALL_EXPANSION": 3,
    "SOURCE_INDEX_RESOLVED": 4,
    "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK": 5,
}
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_PRIVATE_SERIALIZATION_KEYS = frozenset({"source", "source_address", "requested_address", "tx_hash", "transaction_hash"})


def _document(value: Any, schema: str) -> dict[str, Any]:
    result = dict(value) if isinstance(value, Mapping) else value.to_dict()
    if not isinstance(result, dict) or result.get("schema_version") != schema:
        raise ValueError("SOURCE_GRAPH_FALLBACK_INPUT_INVALID")
    return result


def _path(
    adjacency: Mapping[str, tuple[tuple[str, str], ...]],
    source: str,
    result: str,
) -> tuple[str, ...] | None:
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, ())])
    seen = {source}
    while queue:
        node, edges = queue.popleft()
        if node == result:
            return edges
        for target, edge_id in adjacency.get(node, ()):
            if target not in seen:
                seen.add(target)
                queue.append((target, edges + (edge_id,)))
    return None


def _nearest_path(
    adjacency: Mapping[str, tuple[tuple[str, str], ...]],
    sources: tuple[str, ...],
    target: str,
) -> tuple[str, tuple[str, ...]] | None:
    paths = []
    for source in sources:
        path = _path(adjacency, source, target)
        if path is not None:
            paths.append((len(path), source, path))
    if not paths:
        return None
    _, source, path = min(paths, key=lambda item: (item[0], item[1], item[2]))
    return source, path


def _safe_address(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return norm_address(value)
    except ValueError:
        return None


def _stable_codes(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return []
    return sorted({item for item in values if isinstance(item, str) and item})


def _graph_context(graph: dict[str, Any], strict: dict[str, Any]) -> dict[str, Any]:
    nodes = {
        item.get("node_id"): item
        for item in graph.get("nodes", ())
        if isinstance(item, Mapping) and isinstance(item.get("node_id"), str)
    }
    calls = [
        item
        for item in graph.get("call_edges", ())
        if isinstance(item, Mapping)
        and item.get("edge_type", "CALL") == "CALL"
        and isinstance(item.get("edge_id"), str)
        and item.get("src") in nodes
        and item.get("dst") in nodes
    ]
    forward_lists: dict[str, list[tuple[str, str]]] = {}
    undirected_lists: dict[str, list[tuple[str, str]]] = {}
    for edge in calls:
        forward_lists.setdefault(edge["src"], []).append((edge["dst"], edge["edge_id"]))
        undirected_lists.setdefault(edge["src"], []).append((edge["dst"], edge["edge_id"]))
        undirected_lists.setdefault(edge["dst"], []).append((edge["src"], edge["edge_id"]))
    forward = {key: tuple(sorted(value)) for key, value in forward_lists.items()}
    undirected = {key: tuple(sorted(value)) for key, value in undirected_lists.items()}
    chain_rows = [
        item
        for item in strict.get("chains", ())
        if isinstance(item, Mapping)
        and isinstance(item.get("strict_chain_id"), str)
        and isinstance(item.get("node_ids"), list)
        and item["node_ids"]
    ]
    chain_rows.sort(key=lambda item: item["strict_chain_id"])
    chain_nodes = {
        item["strict_chain_id"]: tuple(node_id for node_id in item["node_ids"] if node_id in nodes)
        for item in chain_rows
    }
    chain_rows = [item for item in chain_rows if chain_nodes[item["strict_chain_id"]]]
    membership_count = {
        node_id: sum(node_id in chain_nodes[item["strict_chain_id"]] for item in chain_rows)
        for node_id in nodes
    }
    result_nodes = {
        item.get("result_node_id") or chain_nodes[item["strict_chain_id"]][-1]
        for item in chain_rows
    }
    evidence_by_chain = {
        item["strict_chain_id"]: tuple(sorted({edge for edge in item.get("edge_ids", ()) if isinstance(edge, str)}))
        for item in chain_rows
    }
    strict_edges = {
        item.get("edge_id"): item
        for item in strict.get("edges", ())
        if isinstance(item, Mapping) and isinstance(item.get("edge_id"), str)
    }
    asset_nodes: set[str] = set()
    call_asset_chain_ids: set[str] = set()
    for chain in chain_rows:
        kinds: set[str] = set()
        for edge_id in evidence_by_chain[chain["strict_chain_id"]]:
            edge = strict_edges.get(edge_id)
            if not isinstance(edge, Mapping):
                continue
            kind = edge.get("edge_type")
            if kind in {"CALL", "ASSET"}:
                kinds.add(kind)
            if kind == "ASSET":
                for field in ("src_node_id", "dst_node_id"):
                    if edge.get(field) in nodes:
                        asset_nodes.add(edge[field])
        if {"CALL", "ASSET"}.issubset(kinds):
            call_asset_chain_ids.add(chain["strict_chain_id"])
    return {
        "nodes": nodes,
        "calls": calls,
        "forward": forward,
        "undirected": undirected,
        "chain_rows": chain_rows,
        "chain_nodes": chain_nodes,
        "membership_count": membership_count,
        "result_nodes": result_nodes,
        "evidence_by_chain": evidence_by_chain,
        "asset_nodes": asset_nodes,
        "call_asset_chain_ids": call_asset_chain_ids,
    }


def _relation_for_node(node_id: str, context: Mapping[str, Any]) -> dict[str, Any] | None:
    relations = []
    for chain in context["chain_rows"]:
        chain_id = chain["strict_chain_id"]
        strict_nodes = context["chain_nodes"][chain_id]
        result_node = strict_nodes[-1]
        if node_id in strict_nodes:
            relations.append((0, 0, chain_id, (), "STRICT_CHAIN_NODE", "IDENTITY"))
            continue
        path_to_result = _path(context["forward"], node_id, result_node)
        if path_to_result:
            relations.append((1, len(path_to_result), chain_id, path_to_result, "DIRECTED_TO_RESULT", "FORWARD"))
            continue
        from_strict = _nearest_path(context["forward"], strict_nodes, node_id)
        if from_strict is not None and from_strict[1]:
            relations.append((2, len(from_strict[1]), chain_id, from_strict[1], "DIRECTED_FROM_STRICT_CHAIN", "REVERSE"))
            continue
        corridor = _nearest_path(context["undirected"], strict_nodes, node_id)
        if corridor is not None and corridor[1]:
            relations.append((3, len(corridor[1]), chain_id, corridor[1], "UNDIRECTED_CALL_CORRIDOR", "UNDIRECTED"))
    if not relations:
        return None
    tier, distance, chain_id, path, relation_kind, directionality = min(
        relations,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    return {
        "tier": tier,
        "distance": distance,
        "strict_chain_id": chain_id,
        "call_edge_ids": path,
        "relation_kind": relation_kind,
        "path_directionality": directionality,
    }


def _graph_cache_candidates(
    context: Mapping[str, Any],
    cache_index: VerifiedSourceCacheIndex,
) -> tuple[list[dict[str, Any]], set[str]]:
    candidates = []
    issues: set[str] = set()
    for node_id, node in sorted(context["nodes"].items()):
        address = node.get("callee_address")
        if not isinstance(address, str):
            continue
        proxy_values = node.get("proxy_implementation_addresses", ())
        proxy_addresses = tuple(item for item in proxy_values if isinstance(item, str)) if isinstance(proxy_values, (list, tuple)) else ()
        lookup = lookup_verified_source_cache(
            cache_index,
            runtime_address=address,
            effective_address=node.get("effective_address"),
            proxy_implementation_addresses=proxy_addresses,
            requested_address=address,
        )
        if lookup.status in {"VERIFIED_SOURCE_CACHE_CONFLICT", "VERIFIED_SOURCE_CACHE_INVALID"}:
            issues.add(lookup.status)
            continue
        if lookup.status != "VERIFIED_SOURCE_CACHE_HIT" or lookup.binding is None:
            continue
        relation = _relation_for_node(node_id, context)
        if relation is None:
            continue
        candidates.append({
            "node_id": node_id,
            "node": node,
            "lookup": lookup,
            "relation": relation,
            "membership_count": context["membership_count"][node_id],
        })
    candidates.sort(key=lambda item: (
        item["relation"]["tier"],
        item["relation"]["distance"],
        -item["membership_count"],
        item["node"].get("trace_index", 0),
        item["node_id"],
        item["lookup"].binding.source_sha256,
        item["lookup"].binding.artifact_relative_path,
    ))
    return candidates, issues


def _trusted_direct_requests(
    existing: Mapping[str, Any],
    context: Mapping[str, Any],
    cache_index: VerifiedSourceCacheIndex,
) -> tuple[list[dict[str, Any]], set[str]]:
    requests = []
    issues: set[str] = set()
    for raw in existing.get("requests", ()):
        if not isinstance(raw, Mapping):
            continue
        node = context["nodes"].get(raw.get("node_id"), {})
        runtime = node.get("callee_address") if isinstance(node, Mapping) else raw.get("requested_address")
        proxy_values = node.get("proxy_implementation_addresses", ()) if isinstance(node, Mapping) else ()
        proxy_addresses = tuple(item for item in proxy_values if isinstance(item, str)) if isinstance(proxy_values, (list, tuple)) else ()
        lookup = lookup_verified_source_cache(
            cache_index,
            runtime_address=runtime,
            effective_address=node.get("effective_address") if isinstance(node, Mapping) else raw.get("source_address"),
            proxy_implementation_addresses=proxy_addresses,
            requested_address=raw.get("requested_address") or runtime,
        )
        if lookup.status in {"VERIFIED_SOURCE_CACHE_CONFLICT", "VERIFIED_SOURCE_CACHE_INVALID"}:
            issues.add(lookup.status)
            continue
        requests.append(dict(raw))
    return requests, issues


def _source_index_for_union(
    cache_index: VerifiedSourceCacheIndex,
    cache_root: str | Path,
    graph_candidates: list[dict[str, Any]],
    existing: Mapping[str, Any],
) -> dict[str, Any]:
    addresses = []
    for item in graph_candidates:
        runtime = _safe_address(item["node"].get("callee_address"))
        if runtime:
            addresses.append(runtime)
        addresses.append(item["lookup"].binding.artifact_address)
    for request in existing.get("requests", ()):
        if not isinstance(request, Mapping):
            continue
        for field in ("source_address", "requested_address"):
            address = _safe_address(request.get(field))
            if address:
                addresses.append(address)
    acquisition = {
        "schema_version": "source_acquisition_v1",
        "chain": cache_index.chain_key,
        "addresses": [{"requested_address": address} for address in dict.fromkeys(addresses)],
    }
    return build_source_index(acquisition, cache_root)


def _graph_source_slices(
    graph: dict[str, Any],
    index: dict[str, Any],
    cache_root: str | Path,
    graph_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not graph_candidates:
        return {"schema_version": "source_request_slices_v1", "chain": index.get("chain"), "requests": [], "metadata": {"resolution_mode": "NONE"}}
    selected_by_node = {item["node_id"]: item for item in graph_candidates}
    source_nodes = []
    for node in graph.get("nodes", ()):
        copy = dict(node)
        item = selected_by_node.get(copy.get("node_id"))
        if item is not None:
            copy["callee_address"] = item["lookup"].binding.artifact_address
        source_nodes.append(copy)
    source_graph = {**graph, "nodes": source_nodes}
    requested = [
        {"chain_id": "Q", "node_id": item["node_id"], "focus": "ranking_localization", "priority": item["relation"]["tier"]}
        for item in graph_candidates
    ]
    exact = build_source_request_slices(index, source_graph, cache_root, requested)
    slices = build_verified_source_window_fallback(index, source_graph, exact, cache_root)
    requests = []
    for request in slices.get("requests", ()):
        item = selected_by_node.get(request.get("node_id")) if isinstance(request, Mapping) else None
        if item is None:
            continue
        relation = item["relation"]
        enriched = dict(request)
        enriched["fallback_association"] = {
            "relation_kind": relation["relation_kind"],
            "strict_chain_id": relation["strict_chain_id"],
            "call_edge_ids": list(relation["call_edge_ids"]),
            "call_path_length": relation["distance"],
            "path_directionality": relation["path_directionality"],
            "address_match_source": _MATCH_SOURCES.get(item["lookup"].lookup_kind, "TRANSACTION_CALL_GRAPH_FALLBACK"),
        }
        enriched["candidate_sources"] = ["TRANSACTION_CALL_GRAPH_EXECUTED_FUNCTION"]
        enriched["supporting_evidence_ids"] = list(relation["call_edge_ids"])
        uncertainties = set(_stable_codes(enriched.get("uncertainties", ())))
        uncertainties.update({"SOURCE_TO_RESULT_CALL_PATH_COARSE", "TRANSACTION_CALL_GRAPH_SOURCE_FALLBACK_USED"})
        if relation["relation_kind"] == "UNDIRECTED_CALL_CORRIDOR":
            uncertainties.update({"UNDIRECTED_CALL_CORRIDOR_USED", "WEAK_CALL_COMPONENT_CANDIDATE_USED"})
        enriched["uncertainties"] = sorted(uncertainties)
        requests.append(enriched)
    return {**dict(slices), "requests": requests}


def _selector_status(request: Mapping[str, Any], nodes: Mapping[str, Mapping[str, Any]]) -> str:
    if request.get("selector_match_status") == "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK":
        return "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK"
    if request.get("internal_expansion_depth", 0):
        return "INTERNAL_UNIQUE_CALL_EXPANSION"
    node = nodes.get(request.get("node_id"), {})
    observed = node.get("function_selector") if isinstance(node, Mapping) else None
    resolved = request.get("selector")
    if isinstance(observed, str) and isinstance(resolved, str) and observed.lower() == resolved.lower() and observed.upper() != "UNKNOWN":
        return "SELECTOR_EXACT_MATCH"
    uncertainties = set(_stable_codes(request.get("uncertainties", ())))
    if "UNIQUE_FUNCTION_NAME_FALLBACK" in uncertainties:
        return "UNIQUE_NAME_MATCH"
    if isinstance(node.get("function_name"), str) and "(" in node.get("function_name", ""):
        return "UNIQUE_SIGNATURE_MATCH"
    return "SOURCE_INDEX_RESOLVED"


def _direct_annotations(
    request: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    node_id = request.get("node_id")
    chains = [
        item["strict_chain_id"]
        for item in context["chain_rows"]
        if node_id in context["chain_nodes"][item["strict_chain_id"]]
    ]
    evidence = set(_stable_codes(request.get("evidence_ids", ())))
    evidence.update(_stable_codes(request.get("supporting_evidence_ids", ())))
    for chain_id in chains:
        evidence.update(context["evidence_by_chain"].get(chain_id, ()))
    depth = int(request.get("internal_expansion_depth", 0))
    sources = set(_stable_codes(request.get("candidate_sources", ())))
    sources.add("DIRECT_STRICT_CHAIN_SOURCE" if chains else "DIRECT_EXTENDED_EVIDENCE_SOURCE")
    if depth:
        sources.add("INTERNAL_FUNCTION_EXPANSION")
    return {
        "candidate_sources": sorted(sources),
        "supporting_evidence_ids": sorted(evidence),
        "direct_strict_evidence": bool(chains),
        "asset_result_endpoint": node_id in context["result_nodes"],
        "asset_related": node_id in context["asset_nodes"],
        "call_asset_joint_support": any(chain_id in context["call_asset_chain_ids"] for chain_id in chains),
        "call_relation_kinds": ["STRICT_CHAIN_NODE"] if chains else ["DIRECT_SOURCE_WITHOUT_CALL_RELATION"],
        "shortest_call_path_length": 0 if chains else None,
        "internal_expansion_depth": depth,
    }


def _annotate_request(
    request: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    enriched = dict(request)
    association = request.get("fallback_association") if isinstance(request.get("fallback_association"), Mapping) else None
    if association is None:
        annotations = _direct_annotations(request, context)
    else:
        evidence = set(_stable_codes(request.get("supporting_evidence_ids", ())))
        evidence.update(_stable_codes(association.get("call_edge_ids", ())))
        evidence.update(context["evidence_by_chain"].get(association.get("strict_chain_id"), ()))
        annotations = {
            "candidate_sources": list(request.get("candidate_sources", ())) or ["TRANSACTION_CALL_GRAPH_EXECUTED_FUNCTION"],
            "supporting_evidence_ids": sorted(evidence),
            "direct_strict_evidence": association.get("relation_kind") == "STRICT_CHAIN_NODE",
            "asset_result_endpoint": request.get("node_id") in context["result_nodes"],
            "asset_related": request.get("node_id") in context["asset_nodes"],
            "call_asset_joint_support": association.get("strict_chain_id") in context["call_asset_chain_ids"],
            "call_relation_kinds": [association.get("relation_kind")],
            "shortest_call_path_length": association.get("call_path_length"),
            "internal_expansion_depth": int(request.get("internal_expansion_depth", 0)),
        }
    enriched.update(annotations)
    enriched["selector_match_status"] = _selector_status(enriched, context["nodes"])
    enriched["executed_call_node"] = enriched.get("node_id") in context["nodes"]
    enriched["verified_source_conflict_free"] = True
    association = enriched.get("fallback_association") if isinstance(enriched.get("fallback_association"), Mapping) else None
    enriched["address_match_source"] = (
        association.get("address_match_source")
        if association and isinstance(association.get("address_match_source"), str)
        else "DIRECT_EVIDENCE_ADDRESS"
    )
    return enriched


def _index_function(index: Mapping[str, Any], request: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    address = _safe_address(request.get("source_address"))
    if address is None:
        return None
    matches = []
    for record in index.get("addresses", ()):
        if not isinstance(record, Mapping) or record.get("address") != address:
            continue
        for function in record.get("functions", ()):
            if not isinstance(function, Mapping):
                continue
            if (
                function.get("file_path") == request.get("file_path")
                and (function.get("canonical_signature") or function.get("function_name")) == (request.get("signature") or request.get("function_name"))
                and function.get("signature_start_line") == request.get("start_line")
                and function.get("end_line") == request.get("end_line")
            ):
                matches.append(function)
    if len(matches) != 1:
        return None
    return address, matches[0]


def _function_source(index: Mapping[str, Any], cache_root: str | Path, address: str, function: Mapping[str, Any]) -> str | None:
    relative = Path(str(function.get("file_path", "")))
    root = Path(cache_root).resolve()
    path = (root / str(index.get("chain")) / address / relative).resolve()
    try:
        path.relative_to(root)
        lines = path.read_text(encoding="utf-8").splitlines()
    except (ValueError, OSError, UnicodeError):
        return None
    start, end = function.get("signature_start_line"), function.get("end_line")
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or start > end or end > len(lines):
        return None
    return "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, end + 1))


def _expand_internal_functions(
    requests: list[dict[str, Any]],
    index: Mapping[str, Any],
    cache_root: str | Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    expanded = list(requests)
    issues: set[str] = set()
    queue = []
    for request in requests:
        matched = _index_function(index, request)
        if matched is not None:
            queue.append((request, matched[0], matched[1], 0, frozenset({matched[1].get("function_id")})))
    while queue:
        parent, address, function, depth, lineage = queue.pop(0)
        if depth >= MAX_INTERNAL_EXPANSION_DEPTH:
            continue
        resolution = resolve_unique_internal_function_calls(index, cache_root, address, function.get("function_id"))
        issues.update(_stable_codes(resolution.get("issue_codes", ())))
        callees = tuple(item for item in resolution.get("calls", ()) if isinstance(item, Mapping))
        parent_identity = {
            "relative_file": str(function.get("file_path", "")).replace("\\", "/"),
            "function_signature": function.get("canonical_signature") or function.get("function_name"),
            "start_line": function.get("signature_start_line"),
            "end_line": function.get("end_line"),
            "selector_match_status": parent.get("selector_match_status"),
            "direct_internal_callee_count": len(callees),
        }
        for callee in callees:
            function_id = callee.get("function_id")
            if not isinstance(function_id, str) or function_id in lineage:
                continue
            source = _function_source(index, cache_root, address, callee)
            if source is None:
                issues.add("INTERNAL_CALL_SOURCE_UNAVAILABLE")
                continue
            child = {
                **dict(parent),
                "status": "RESOLVED",
                "function_match_status": "RESOLVED",
                "source_address": address,
                "file_path": callee.get("file_path"),
                "contract_name": callee.get("contract_name"),
                "function_name": callee.get("function_name"),
                "signature": callee.get("canonical_signature") or callee.get("function_name"),
                "selector": callee.get("selector"),
                "start_line": callee.get("signature_start_line"),
                "end_line": callee.get("end_line"),
                "source": source,
                "candidate_functions": [],
                "internal_expansion_depth": depth + 1,
                "unique_parent_function_identity": parent_identity,
                "selector_match_status": "INTERNAL_UNIQUE_CALL_EXPANSION",
                "candidate_sources": sorted(set(parent.get("candidate_sources", ())) | {"INTERNAL_FUNCTION_EXPANSION"}),
                "uncertainties": sorted(set(_stable_codes(parent.get("uncertainties", ()))) | {"INTERNAL_FUNCTION_EXPANSION_USED"}),
            }
            expanded.append(child)
            issues.add("INTERNAL_FUNCTION_EXPANSION_USED")
            queue.append((child, address, callee, depth + 1, lineage | {function_id}))
    return expanded, issues


def _graph_linked_public_external_fallbacks(
    index: Mapping[str, Any],
    cache_root: str | Path,
    graph_candidates: list[dict[str, Any]],
    resolved_requests: list[dict[str, Any]],
    *,
    allow_resolved_identity_coexistence: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Admit parsed public/external functions only for unresolved graph nodes.

    The graph/cache binding and weak-component relation have already passed the
    formal checks in ``_graph_cache_candidates``.  This fallback never infers a
    selector or bare function name: every candidate is an actual, fully indexed
    Solidity function with a verified source body.
    """
    resolved_nodes = {
        item.get("node_id")
        for item in resolved_requests
        if isinstance(item, Mapping) and item.get("function_match_status") == "RESOLVED"
    }
    requests: list[dict[str, Any]] = []
    issues: set[str] = set()
    coexistence_used = False
    for item in graph_candidates:
        node_id = item["node_id"]
        coexists_with_resolved_identity = node_id in resolved_nodes
        if coexists_with_resolved_identity and not allow_resolved_identity_coexistence:
            continue
        address = _safe_address(item["lookup"].binding.artifact_address)
        if address is None:
            continue
        records = [
            record
            for record in index.get("addresses", ())
            if isinstance(record, Mapping)
            and record.get("address") == address
            and record.get("status") == "verified"
            and record.get("verified") is True
        ]
        if len(records) != 1:
            continue
        relation = item["relation"]
        for function in records[0].get("functions", ()):
            if not isinstance(function, Mapping):
                continue
            if function.get("function_kind") != "FUNCTION" or function.get("visibility") not in {"public", "external"}:
                continue
            signature = function.get("canonical_signature")
            if not isinstance(signature, str) or not signature:
                continue
            if not isinstance(function.get("body_start_line"), int):
                continue
            source = _function_source(index, cache_root, address, function)
            if source is None:
                continue
            candidate_sources = ["GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK"]
            uncertainties = [
                "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK",
                "GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK_USED",
                "SOURCE_TO_RESULT_CALL_PATH_COARSE",
                "TRANSACTION_CALL_GRAPH_SOURCE_FALLBACK_USED",
            ]
            if coexists_with_resolved_identity:
                coexistence_used = True
                candidate_sources.append("COARSE_SOURCE_FUNCTION_SUPPLEMENT")
                uncertainties.extend([
                    "COARSE_SOURCE_FUNCTION_SUPPLEMENT_USED",
                    "GRAPH_LINKED_SOURCE_FALLBACK_COEXISTS_WITH_RESOLVED_IDENTITY",
                ])
            requests.append({
                "chain_id": "Q",
                "node_id": node_id,
                "focus": "ranking_localization",
                "priority": relation["tier"],
                "requested_address": item["node"].get("callee_address"),
                "source_address": address,
                "status": "RESOLVED",
                "function_match_status": "RESOLVED",
                "file_path": function.get("file_path"),
                "contract_name": function.get("contract_name"),
                "function_name": function.get("function_name"),
                "signature": signature,
                "selector": function.get("selector"),
                "visibility": function.get("visibility"),
                "start_line": function.get("signature_start_line"),
                "end_line": function.get("end_line"),
                "source": source,
                "candidate_functions": [],
                "internal_expansion_depth": 0,
                "selector_match_status": "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK",
                "candidate_sources": candidate_sources,
                "supporting_evidence_ids": list(relation["call_edge_ids"]),
                "resolution_mode": "EXACT_FUNCTION_SLICE",
                "fallback_association": {
                    "relation_kind": relation["relation_kind"],
                    "strict_chain_id": relation["strict_chain_id"],
                    "call_edge_ids": list(relation["call_edge_ids"]),
                    "call_path_length": relation["distance"],
                    "path_directionality": relation["path_directionality"],
                    "address_match_source": _MATCH_SOURCES.get(item["lookup"].lookup_kind, "TRANSACTION_CALL_GRAPH_FALLBACK"),
                },
                "uncertainties": uncertainties,
            })
    if requests:
        issues.update({
            "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK",
            "GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK_USED",
        })
    if coexistence_used:
        issues.update({
            "COARSE_SOURCE_FUNCTION_SUPPLEMENT_USED",
            "GRAPH_LINKED_SOURCE_FALLBACK_COEXISTS_WITH_RESOLVED_IDENTITY",
        })
    return requests, issues


def _merge_exact_candidates(
    requests: list[dict[str, Any]],
    context: Mapping[str, Any],
    max_candidates: int,
) -> tuple[list[dict[str, Any]], int, int, int, dict[str, int], dict[str, int], int, int]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in requests:
        if raw.get("function_match_status") != "RESOLVED":
            continue
        request = _annotate_request(raw, context)
        address = _safe_address(request.get("source_address")) or _safe_address(context["nodes"].get(request.get("node_id"), {}).get("effective_address")) or _safe_address(context["nodes"].get(request.get("node_id"), {}).get("callee_address"))
        key = (
            address,
            request.get("file_path"),
            request.get("signature") or request.get("function_name"),
            request.get("start_line"),
            request.get("end_line"),
        )
        if not all((address, isinstance(key[1], str), isinstance(key[2], str), isinstance(key[3], int), isinstance(key[4], int))):
            continue
        current = grouped.get(key)
        if current is None:
            request["candidate_sources"] = sorted(set(request["candidate_sources"]))
            request["supporting_evidence_ids"] = sorted(set(request["supporting_evidence_ids"]))
            request["call_relation_kinds"] = sorted(set(request["call_relation_kinds"]), key=lambda value: (_RELATION_PRIORITY.get(value, 99), value))
            grouped[key] = request
            continue
        _merge_candidate_evidence(current, request)
    # Function localization identifies source ranges, not deployments.  Merge
    # cloned deployments only when the exact source slice also agrees.
    cloned: dict[tuple[Any, ...], dict[str, Any]] = {}
    for request in grouped.values():
        source = request.get("source")
        address = _safe_address(request.get("source_address")) or _safe_address(request.get("requested_address"))
        key = (
            "EXACT_SOURCE" if isinstance(source, str) else address,
            request.get("file_path"),
            request.get("signature") or request.get("function_name"),
            request.get("start_line"),
            request.get("end_line"),
            source if isinstance(source, str) else None,
            tuple(request.get("call_relation_kinds", ())),
        )
        current = cloned.get(key)
        if current is None:
            cloned[key] = request
        else:
            _merge_candidate_evidence(current, request)
    candidates = list(cloned.values())

    candidates.sort(key=lambda item: (
        0 if item["direct_strict_evidence"] else 1,
        _SELECTOR_PRIORITY.get(item["selector_match_status"], 99),
        0 if item["asset_result_endpoint"] else 1,
        min((_RELATION_PRIORITY.get(value, 99) for value in item["call_relation_kinds"]), default=99),
        item["shortest_call_path_length"] if isinstance(item["shortest_call_path_length"], int) else 1 << 30,
        item["internal_expansion_depth"],
        str(item.get("file_path", "")),
        str(item.get("signature") or item.get("function_name") or ""),
        item.get("start_line", 0),
        item.get("end_line", 0),
    ))
    fallback_before = sum(
        "GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK" in item.get("candidate_sources", ())
        for item in candidates
    )
    supplement_before = sum(
        "COARSE_SOURCE_FUNCTION_SUPPLEMENT" in item.get("candidate_sources", ())
        for item in candidates
    )
    source_counts_before: dict[str, int] = {}
    for item in candidates:
        for source in item.get("candidate_sources", ()):
            source_counts_before[source] = source_counts_before.get(source, 0) + 1
    selected = candidates[:max_candidates]
    fallback_selected = sum(
        "GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK" in item.get("candidate_sources", ())
        for item in selected
    )
    supplement_selected = sum(
        "COARSE_SOURCE_FUNCTION_SUPPLEMENT" in item.get("candidate_sources", ())
        for item in selected
    )
    source_counts_after: dict[str, int] = {}
    for item in selected:
        for source in item.get("candidate_sources", ()):
            source_counts_after[source] = source_counts_after.get(source, 0) + 1
    truncated = max(0, len(candidates) - max_candidates)
    return (
        selected,
        truncated,
        fallback_before,
        max(0, fallback_before - fallback_selected),
        dict(sorted(source_counts_before.items())),
        dict(sorted(source_counts_after.items())),
        supplement_before,
        max(0, supplement_before - supplement_selected),
    )


def _merge_candidate_evidence(current: dict[str, Any], request: Mapping[str, Any]) -> None:
    current["candidate_sources"] = sorted(set(current["candidate_sources"]) | set(request["candidate_sources"]))
    current["supporting_evidence_ids"] = sorted(set(current["supporting_evidence_ids"]) | set(request["supporting_evidence_ids"]))
    current["call_relation_kinds"] = sorted(set(current["call_relation_kinds"]) | set(request["call_relation_kinds"]), key=lambda value: (_RELATION_PRIORITY.get(value, 99), value))
    current["direct_strict_evidence"] = current["direct_strict_evidence"] or request["direct_strict_evidence"]
    current["asset_result_endpoint"] = current["asset_result_endpoint"] or request["asset_result_endpoint"]
    current["asset_related"] = current.get("asset_related", False) or request.get("asset_related", False)
    current["call_asset_joint_support"] = current.get("call_asset_joint_support", False) or request.get("call_asset_joint_support", False)
    current["executed_call_node"] = current.get("executed_call_node", False) or request.get("executed_call_node", False)
    current["verified_source_conflict_free"] = current.get("verified_source_conflict_free", False) or request.get("verified_source_conflict_free", False)
    if _ADDRESS_MATCH_PRIORITY.get(request.get("address_match_source"), 99) < _ADDRESS_MATCH_PRIORITY.get(current.get("address_match_source"), 99):
        current["address_match_source"] = request.get("address_match_source")
    distances = [value for value in (current.get("shortest_call_path_length"), request.get("shortest_call_path_length")) if isinstance(value, int)]
    current["shortest_call_path_length"] = min(distances) if distances else None
    current["internal_expansion_depth"] = min(current["internal_expansion_depth"], request["internal_expansion_depth"])
    if current.get("unique_parent_function_identity") != request.get("unique_parent_function_identity"):
        current["unique_parent_function_identity"] = None
    if _SELECTOR_PRIORITY.get(request["selector_match_status"], 99) < _SELECTOR_PRIORITY.get(current["selector_match_status"], 99):
        current["selector_match_status"] = request["selector_match_status"]
    current["uncertainties"] = sorted(set(_stable_codes(current.get("uncertainties", ()))) | set(_stable_codes(request.get("uncertainties", ()))))
    if current.get("fallback_association") is not None and request.get("fallback_association") is None:
        for field in ("node_id", "requested_address", "priority", "focus"):
            current[field] = request.get(field)
        current.pop("fallback_association", None)


def _candidate_union_document(
    candidates: list[dict[str, Any]],
    issue_codes: set[str],
    truncated: int,
    max_candidates: int,
    fallback_before: int,
    fallback_truncated: int,
    source_counts_before: Mapping[str, int],
    source_counts_after: Mapping[str, int],
    supplement_before: int,
    supplement_truncated: int,
) -> dict[str, Any]:
    if truncated:
        issue_codes.add("RANKING_SOURCE_FUNCTION_CANDIDATES_TRUNCATED")
        if max_candidates == MAX_SOURCE_FUNCTION_CANDIDATES_RANKING:
            issue_codes.add("RANKING_V4_SOURCE_FUNCTION_CANDIDATES_TRUNCATED_AT_100")
    if fallback_truncated:
        issue_codes.add("GRAPH_LINKED_FUNCTION_FALLBACK_TRUNCATED")
    if supplement_truncated:
        issue_codes.add("COARSE_SOURCE_FUNCTION_SUPPLEMENT_TRUNCATED")
    addresses = sorted({
        _safe_address(item.get("source_address")) or _safe_address(item.get("requested_address"))
        for item in candidates
        if _safe_address(item.get("source_address")) or _safe_address(item.get("requested_address"))
    })
    contract_aliases = {address: f"K{index}" for index, address in enumerate(addresses, start=1)}
    rows = []
    for position, item in enumerate(candidates, start=1):
        relation_priority = min((_RELATION_PRIORITY.get(value, 99) for value in item["call_relation_kinds"]), default=99)
        address = _safe_address(item.get("source_address")) or _safe_address(item.get("requested_address"))
        contract_alias = contract_aliases.get(address, f"K{position}")
        address_match_source = item.get("address_match_source", "DIRECT_EVIDENCE_ADDRESS")
        parent_identity = item.get("unique_parent_function_identity")
        if isinstance(parent_identity, Mapping):
            parent_identity = {**parent_identity, "contract_alias": contract_alias}
        rows.append({
            "candidate_alias": f"C{position}",
            "contract_alias": contract_alias,
            "relative_file": str(item["file_path"]).replace("\\", "/"),
            "function_signature": item.get("signature") or item.get("function_name"),
            "visibility": item.get("visibility") if item.get("visibility") in {"public", "external", "internal", "private"} else "UNKNOWN",
            "start_line": item["start_line"],
            "end_line": item["end_line"],
            "selector_match_status": item["selector_match_status"],
            "candidate_sources": list(item["candidate_sources"]),
            "call_relation_modes": list(item["call_relation_kinds"]),
            "shortest_call_path_length": item["shortest_call_path_length"],
            "internal_expansion_depth": item["internal_expansion_depth"],
            "unique_parent_function_identity": parent_identity,
            "supporting_evidence_ids": list(item["supporting_evidence_ids"]),
            "ranking_features": {
                "direct_strict_evidence": item["direct_strict_evidence"],
                "selector_priority": _SELECTOR_PRIORITY.get(item["selector_match_status"], 99),
                "asset_result_endpoint": item["asset_result_endpoint"],
                "asset_related": bool(item.get("asset_related")),
                "call_asset_joint_support": bool(item.get("call_asset_joint_support")),
                "call_relation_priority": relation_priority,
                "shortest_call_path_length": item["shortest_call_path_length"],
                "internal_expansion_depth": item["internal_expansion_depth"],
                "address_match_source": address_match_source,
                "address_match_priority": _ADDRESS_MATCH_PRIORITY.get(address_match_source, 99),
                "executed_call_node": bool(item.get("executed_call_node")),
                "verified_source_conflict_free": bool(item.get("verified_source_conflict_free")),
            },
        })
    return {
        "schema_version": "ranking_source_candidate_union_v1",
        "candidate_count": len(rows),
        "max_candidates": max_candidates,
        "total_candidate_count_before_truncation": len(rows) + truncated,
        "candidate_count_before_truncation": len(rows) + truncated,
        "candidate_count_after_truncation": len(rows),
        "candidate_limit_reached": len(rows) == max_candidates,
        "candidate_limit_exceeded": truncated > 0,
        "fallback_candidate_count": sum("GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK" in item["candidate_sources"] for item in candidates),
        "fallback_candidate_count_before_truncation": fallback_before,
        "fallback_truncated_candidate_count": fallback_truncated,
        "coarse_source_function_supplement_count_before_truncation": supplement_before,
        "coarse_source_function_supplement_count_after_truncation": supplement_before - supplement_truncated,
        "coarse_source_function_supplement_truncated_count": supplement_truncated,
        "candidate_source_counts_before_truncation": dict(sorted(source_counts_before.items())),
        "candidate_source_counts_after_truncation": dict(sorted(source_counts_after.items())),
        "truncated_candidate_count": truncated,
        "issue_codes": sorted(issue_codes),
        "candidates": rows,
    }


def _build_union(
    transaction_graph: Any,
    strict_result_chains: Any,
    existing_source_slices: Any,
    cache_index: VerifiedSourceCacheIndex,
    cache_root: str | Path,
    *,
    max_candidates: int = MAX_SOURCE_FUNCTION_CANDIDATES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = _document(transaction_graph, "transaction_graph_v1")
    strict = _document(strict_result_chains, "strict_result_chains_v1")
    existing = _document(existing_source_slices, "source_request_slices_v1")
    if not isinstance(cache_index, VerifiedSourceCacheIndex):
        raise ValueError("SOURCE_GRAPH_FALLBACK_CACHE_INDEX_INVALID")
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or not 1 <= max_candidates <= MAX_SOURCE_FUNCTION_CANDIDATES_RANKING:
        raise ValueError("SOURCE_GRAPH_FALLBACK_CANDIDATE_LIMIT_INVALID")
    context = _graph_context(graph, strict)
    graph_candidates, cache_issues = _graph_cache_candidates(context, cache_index)
    index = _source_index_for_union(cache_index, cache_root, graph_candidates, existing)
    graph_slices = _graph_source_slices(graph, index, cache_root, graph_candidates)
    direct_requests, direct_cache_issues = _trusted_direct_requests(existing, context, cache_index)
    graph_requests = [dict(item) for item in graph_slices.get("requests", ()) if isinstance(item, Mapping)]
    annotated_roots = [_annotate_request(item, context) for item in direct_requests + graph_requests if item.get("function_match_status") == "RESOLVED"]
    expanded, expansion_issues = _expand_internal_functions(annotated_roots, index, cache_root)
    coarse_fallbacks, coarse_fallback_issues = _graph_linked_public_external_fallbacks(
        index,
        cache_root,
        graph_candidates,
        direct_requests + graph_requests,
        allow_resolved_identity_coexistence=max_candidates > MAX_SOURCE_FUNCTION_CANDIDATES,
    )
    (
        candidates,
        truncated,
        fallback_before,
        fallback_truncated,
        source_counts_before,
        source_counts_after,
        supplement_before,
        supplement_truncated,
    ) = _merge_exact_candidates(expanded + coarse_fallbacks, context, max_candidates)
    issue_codes = set(expansion_issues) | set(coarse_fallback_issues) | cache_issues | direct_cache_issues
    if any("UNDIRECTED_CALL_CORRIDOR" in item.get("call_relation_kinds", ()) for item in candidates):
        issue_codes.update({"UNDIRECTED_CALL_CORRIDOR_USED", "WEAK_CALL_COMPONENT_CANDIDATE_USED"})
    union = _candidate_union_document(
        candidates,
        issue_codes,
        truncated,
        max_candidates,
        fallback_before,
        fallback_truncated,
        source_counts_before,
        source_counts_after,
        supplement_before,
        supplement_truncated,
    )
    if candidates:
        final_requests = candidates
    else:
        window_rows = [item for item in direct_requests + graph_requests if item.get("function_match_status") == "WINDOW_FALLBACK"]
        window_rows.sort(key=lambda item: (0 if item in direct_requests else 1, item.get("file_path", ""), item.get("start_line", 0), item.get("end_line", 0), item.get("node_id", "")))
        final_requests = window_rows[:MAX_SOURCE_SLICES]
    graph_associations = [
        item["fallback_association"] | {"node_id": item.get("node_id")}
        for item in final_requests
        if isinstance(item.get("fallback_association"), Mapping)
    ]
    metadata = dict(existing.get("metadata", {}))
    metadata.update({
        "request_count": len(final_requests),
        "resolved_count": sum(item.get("function_match_status") == "RESOLVED" for item in final_requests),
        "source_selection_mode": MODE if graph_requests else metadata.get("source_selection_mode", "DIRECT_EVIDENCE_SOURCE"),
        "graph_fallback_associations": graph_associations,
        "fallback_issue_codes": sorted(set(_stable_codes(metadata.get("fallback_issue_codes", ()))) | set(union["issue_codes"]) | ({"TRANSACTION_CALL_GRAPH_SOURCE_FALLBACK_USED", "SOURCE_TO_RESULT_CALL_PATH_COARSE"} if graph_requests else set())),
        "candidate_union": union,
    })
    if candidates:
        metadata["resolution_mode"] = "EXACT_FUNCTION_SLICE"
    result = {
        "schema_version": "source_request_slices_v1",
        "chain": existing.get("chain") or cache_index.chain_key,
        "source_index_sha256": graph_slices.get("source_index_sha256") or existing.get("source_index_sha256"),
        "transaction_graph_sha256": graph_slices.get("transaction_graph_sha256") or existing.get("transaction_graph_sha256"),
        "review_ranked_chains_sha256": existing.get("review_ranked_chains_sha256"),
        "requests": final_requests,
        "metadata": metadata,
    }
    return result, union


def build_ranking_source_candidate_union(
    transaction_graph: Any,
    strict_result_chains: Any,
    existing_source_slices: Any,
    cache_index: VerifiedSourceCacheIndex,
    cache_root: str | Path,
    *,
    max_candidates: int = MAX_SOURCE_FUNCTION_CANDIDATES,
) -> dict[str, Any]:
    """Build a redacted, stable pool of verified functions under an explicit cap."""
    return _build_union(transaction_graph, strict_result_chains, existing_source_slices, cache_index, cache_root, max_candidates=max_candidates)[1]


def serialize_ranking_source_candidate_union(candidate_union: Mapping[str, Any]) -> str:
    if not isinstance(candidate_union, Mapping) or candidate_union.get("schema_version") != "ranking_source_candidate_union_v1":
        raise ValueError("RANKING_SOURCE_CANDIDATE_UNION_INVALID")
    stack = [candidate_union]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            if any(key in _PRIVATE_SERIALIZATION_KEYS for key in value):
                raise ValueError("RANKING_SOURCE_CANDIDATE_UNION_PRIVATE_FIELD")
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    serialized = json.dumps(dict(candidate_union), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if _RAW_ADDRESS.search(serialized) or _RAW_HASH.search(serialized):
        raise ValueError("RANKING_SOURCE_CANDIDATE_UNION_RAW_IDENTITY")
    return serialized


def build_ranking_source_graph_fallback(
    transaction_graph: Any,
    strict_result_chains: Any,
    existing_source_slices: Any,
    cache_index: VerifiedSourceCacheIndex,
    cache_root: str | Path,
    *,
    max_candidates: int = MAX_SOURCE_FUNCTION_CANDIDATES,
) -> dict[str, Any]:
    """Return the direct-plus-CALL-graph verified source candidate union."""
    return _build_union(transaction_graph, strict_result_chains, existing_source_slices, cache_index, cache_root, max_candidates=max_candidates)[0]

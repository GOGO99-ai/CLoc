"""Small, JSON-only normal-transaction reference library prototype.

This module intentionally consumes only a platform, faultless transaction hashes,
and observed traces.  It does not read fault causes, locations, or disclosures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.runtime_semantics import build_runtime_annotations
from src.cloc_v2.transaction_graph import build_transaction_graph


TRANSFER_SELECTORS = {"0xa9059cbb", "0x23b872dd"}
SELECTOR_FOR_FUNCTION = {"transfer": "0xa9059cbb", "transferfrom": "0x23b872dd"}
POSITIVE_VALUE_RE = re.compile(r"\{value:\s*(\d+)\}", re.IGNORECASE)


def flatten_transaction_hashes(value: Any) -> list[str]:
    """Recursively flatten nested hash lists, preserving first-seen order."""
    flattened: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            normalized = item.lower()
            if normalized not in flattened:
                flattened.append(normalized)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return flattened


def load_faultless_candidates(case_path: str | Path) -> tuple[str, list[str]]:
    """Read only the platform and faultless hashes from a dataset case."""
    payload = json.loads(Path(case_path).read_text(encoding="utf-8"))
    return str(payload.get("platform", "")), flatten_transaction_hashes(payload.get("faultless", {}).get("transaction_hash", []))


def locate_cached_call_tree(cache_root: str | Path, tx_hash: str) -> Path | None:
    """Find a parsed trace by hash without using its parent directory as data."""
    matches = sorted(Path(cache_root).rglob(f"{tx_hash.lower()}.json"))
    return matches[0] if matches else None


def load_cached_or_replay_call_tree(
    tx_hash: str,
    *,
    parsed_cache_root: str | Path,
    replayer: Any,
    trace_parser: Any,
) -> tuple[dict[str, Any], bool, str]:
    """Prefer parsed cache; otherwise use the repository's existing replayer."""
    cached_path = locate_cached_call_tree(parsed_cache_root, tx_hash)
    if cached_path is not None:
        payload = json.loads(cached_path.read_text(encoding="utf-8"))
        return _unwrap_call_tree(payload), True, f"cache:{cached_path.name}"

    raw_cache_path = Path(replayer.cache_dir) / f"{replayer.chain}_{tx_hash.lower()}.txt"
    if raw_cache_path.exists():
        parsed = trace_parser.parse(raw_cache_path.read_text(encoding="utf-8"))
        return _unwrap_call_tree(parsed), True, f"cache:{raw_cache_path.name}"

    trace_text = replayer.run(tx_hash)
    parsed = trace_parser.parse(trace_text)
    return _unwrap_call_tree(parsed), False, "replayer"


def build_reference_record(reference_id: str, tx_hash: str, platform: str, call_tree: dict[str, Any], *, cache_used: bool, trace_path: str) -> dict[str, Any]:
    """Create the minimal CALL/ASSET evidence graph and explainable chains."""
    graph = build_transaction_graph(call_tree)
    annotations = build_runtime_annotations(graph)
    ledger = build_observed_asset_ledger(graph, annotations)
    raw_nodes_by_path = _raw_nodes_by_path(call_tree)

    nodes = [_simple_node(node) for node in graph.nodes]
    call_edges = [{"edge_type": "CALL", "src": edge.src, "dst": edge.dst} for edge in graph.call_edges]
    asset_edges = [
        _simple_asset_edge(fact)
        for fact in ledger.transfer_facts
        if _is_complete_asset_fact(fact)
        and _raw_node_succeeded(raw_nodes_by_path.get(fact.trace_path, {}))
    ]
    asset_edges.extend(_selector_fallback_asset_edges(graph.nodes, call_tree, {fact.node_id for fact in ledger.transfer_facts}))
    reference_chains = _reference_chains(nodes, asset_edges)
    addresses = sorted({node["address"].lower() for node in nodes if isinstance(node["address"], str)})
    functions = _function_identities(nodes)
    root = nodes[0]

    return {
        "reference_id": reference_id,
        "tx_hash": tx_hash.lower(),
        "platform": platform,
        "trace_source": {"cache_used": cache_used, "trace_path": trace_path},
        "transaction_identity": {
            "entry_address": root["address"],
            "entry_selector": _selector(root["function_or_selector"]),
            "involved_addresses": addresses,
        },
        "evidence_graph": {"nodes": nodes, "call_edges": call_edges, "asset_edges": asset_edges},
        "reference_chains": reference_chains,
        "retrieval_features": {
            "addresses": addresses,
            "functions_or_selectors": functions,
            "node_count": len(nodes),
            "call_edge_count": len(call_edges),
            "asset_edge_count": len(asset_edges),
        },
    }


def build_query_features(call_tree: dict[str, Any]) -> dict[str, Any]:
    """Use the identical observed-trace conversion for a retrieval query."""
    record = build_reference_record("query", "query", "Ethereum", call_tree, cache_used=True, trace_path="query")
    return {"transaction_identity": record["transaction_identity"], "retrieval_features": record["retrieval_features"]}


def retrieve_similar_references(query: dict[str, Any], library: Iterable[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Apply the requested deterministic lexicographic ranking rule."""
    query_identity, query_features = query["transaction_identity"], query["retrieval_features"]
    query_addresses = set(query_features["addresses"])
    query_functions = set(query_features["functions_or_selectors"])
    results: list[dict[str, Any]] = []
    for reference in library:
        features = reference["retrieval_features"]
        shared_addresses = sorted(query_addresses & set(features["addresses"]))
        shared_functions = sorted(query_functions & set(features["functions_or_selectors"]))
        same_entry_selector = query_identity["entry_selector"] == reference["transaction_identity"]["entry_selector"]
        results.append(
            {
                "reference_id": reference["reference_id"],
                "same_entry_selector": same_entry_selector,
                "shared_address_count": len(shared_addresses),
                "shared_function_count": len(shared_functions),
                "node_count_difference": abs(query_features["node_count"] - features["node_count"]),
                "shared_addresses": shared_addresses,
                "shared_functions": shared_functions,
                "asset_edge_count": features["asset_edge_count"],
                "reference_chain_fallback": any("fallback_reason" in chain for chain in reference["reference_chains"]),
            }
        )
    results.sort(key=lambda item: (-int(item["same_entry_selector"]), -item["shared_address_count"], -item["shared_function_count"], item["node_count_difference"], item["reference_id"]))
    for rank, result in enumerate(results[:limit], start=1):
        result["rank"] = rank
    return results[:limit]


def _unwrap_call_tree(payload: dict[str, Any]) -> dict[str, Any]:
    tree = payload.get("call_tree", payload)
    if hasattr(tree, "to_dict"):
        tree = tree.to_dict()
    if not isinstance(tree, dict):
        raise ValueError("trace cache did not contain a call tree object")
    return tree


def _simple_node(node: Any) -> dict[str, Any]:
    identifier = node.function_selector or node.function_name
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "trace_index": node.trace_index,
        "address": node.callee_address,
        "function_or_selector": identifier,
        "call_type": node.call_type,
        "value": node.value,
        "status": "OBSERVED",
    }


def _simple_asset_edge(fact: Any) -> dict[str, Any]:
    return {
        "edge_id": fact.fact_id,
        "edge_type": "ASSET",
        "node_id": fact.node_id,
        "trace_index": fact.trace_index,
        "src": fact.from_address["normalized"] if fact.from_address else None,
        "dst": fact.to_address["normalized"] if fact.to_address else None,
        "asset_type": fact.asset_type,
        "asset_address": fact.asset_address["normalized"] if fact.asset_address else None,
        "amount_raw": fact.amount_raw,
        "transfer_kind": fact.transfer_kind,
        "status": fact.status,
    }


def _is_complete_asset_fact(fact: Any) -> bool:
    """Keep only fully observed ledger facts in the legacy reference graph."""
    if fact.status != "COMPLETE":
        return False
    if _normalized_fact_address(fact.from_address) is None or _normalized_fact_address(fact.to_address) is None:
        return False
    if not isinstance(fact.amount_raw, str) or re.fullmatch(r"(?:0|[1-9][0-9]*)", fact.amount_raw) is None:
        return False
    if fact.asset_type == "TOKEN":
        return _normalized_fact_address(fact.asset_address) is not None
    return fact.asset_type == "NATIVE"


def _normalized_fact_address(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    return _address(value.get("normalized"))


def _selector_fallback_asset_edges(graph_nodes: Iterable[Any], call_tree: dict[str, Any], existing_node_ids: set[str]) -> list[dict[str, Any]]:
    """Decode only standard ERC-20 selectors and explicit positive native value."""
    raw_by_path = _raw_nodes_by_path(call_tree)
    edges: list[dict[str, Any]] = []
    for node in graph_nodes:
        if node.node_id in existing_node_ids:
            continue
        raw_node = raw_by_path.get(node.trace_path, {})
        if not _raw_node_succeeded(raw_node) or (node.call_type or "").upper() != "CALL":
            continue
        selector = node.function_selector
        if selector == "0xa9059cbb":
            decoded = _decode_erc20_calldata(node.params_raw, selector, 2)
            if decoded is not None:
                recipient, amount = decoded
                edge = _asset_edge_from_selector(node, "TRANSFER", _node_caller(node, raw_node), recipient, amount)
                if edge is not None:
                    edges.append(edge)
        elif selector == "0x23b872dd":
            decoded = _decode_erc20_calldata(node.params_raw, selector, 3)
            if decoded is not None:
                sender, recipient, amount = decoded
                edge = _asset_edge_from_selector(node, "TRANSFER_FROM", sender, recipient, amount)
                if edge is not None:
                    edges.append(edge)
        else:
            value = _positive_value(raw_node.get("value")) or _positive_value_from_function(raw_node.get("function"))
            sender = _node_caller(node, raw_node)
            if value is not None and sender is not None and _address(node.callee_address) is not None:
                edges.append(_new_selector_edge(node, "NATIVE_VALUE", "NATIVE", None, sender, _address(node.callee_address), value))
    return edges


def _asset_edge_from_selector(node: Any, transfer_kind: str, sender: str | None, recipient: str, amount: str) -> dict[str, Any] | None:
    asset_address = _address(node.callee_address)
    if asset_address is None or sender is None:
        return None
    return _new_selector_edge(node, transfer_kind, "TOKEN", asset_address, sender, recipient, amount)


def _new_selector_edge(node: Any, transfer_kind: str, asset_type: str, asset_address: str | None, sender: str, recipient: str, amount: str) -> dict[str, Any]:
    return {
        "edge_id": f"asset_selector_{node.node_id}",
        "edge_type": "ASSET",
        "node_id": node.node_id,
        "trace_index": node.trace_index,
        "src": sender,
        "dst": recipient,
        "asset_type": asset_type,
        "asset_address": asset_address,
        "amount_raw": amount,
        "transfer_kind": transfer_kind,
        "status": "OBSERVED",
        "reason": "standard selector/calldata fallback",
    }


def _raw_nodes_by_path(call_tree: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    def visit(node: dict[str, Any], path: str) -> None:
        result[path] = node
        for index, child in enumerate(node.get("children", []) or []):
            if isinstance(child, dict):
                visit(child, f"{path}/{index}")

    visit(call_tree, "0")
    return result


def _raw_node_succeeded(raw_node: dict[str, Any]) -> bool:
    status = str(raw_node.get("status", "")).strip().upper()
    return not raw_node.get("error") and status not in {"FAILED", "ERROR", "REVERTED", "REVERT"}


def _decode_erc20_calldata(params_raw: Any, selector: str, argument_count: int) -> tuple[str, ...] | None:
    if not isinstance(params_raw, str):
        return None
    calldata = params_raw.strip().lower()
    normalized_selector = selector.lower() if isinstance(selector, str) else ""
    expected_length = 10 + argument_count * 64
    if (
        re.fullmatch(r"0x[0-9a-f]{8}", normalized_selector) is None
        or len(calldata) != expected_length
        or re.fullmatch(r"0x[0-9a-f]+", calldata) is None
        or calldata[:10] != normalized_selector
    ):
        return None
    words = [calldata[10 + index * 64:10 + (index + 1) * 64] for index in range(argument_count)]
    if any(len(word) != 64 for word in words):
        return None
    if any(word[:24] != "0" * 24 for word in words[:-1]):
        return None
    addresses = [f"0x{word[-40:]}" for word in words[:-1]]
    if any(_address(item) is None for item in addresses):
        return None
    try:
        amount = str(int(words[-1], 16))
    except ValueError:
        return None
    return (*addresses, amount)


def _node_caller(node: Any, raw_node: dict[str, Any]) -> str | None:
    return _address(raw_node.get("from")) or _address(node.caller_address)


def _address(value: Any) -> str | None:
    text = str(value or "").lower()
    return text if re.fullmatch(r"0x[0-9a-f]{40}", text) else None


def _positive_value(value: Any) -> str | None:
    if isinstance(value, int):
        return str(value) if value > 0 else None
    text = str(value or "").strip()
    try:
        parsed = int(text, 16) if text.lower().startswith("0x") else int(text)
    except ValueError:
        return None
    return str(parsed) if parsed > 0 else None


def _positive_value_from_function(value: Any) -> str | None:
    match = POSITIVE_VALUE_RE.search(str(value or ""))
    return match.group(1) if match and int(match.group(1)) > 0 else None


def _function_identities(nodes: Iterable[dict[str, Any]]) -> list[str]:
    identities: set[str] = set()
    for node in nodes:
        identity = node["function_or_selector"]
        if identity:
            identities.add(identity)
            standard_selector = SELECTOR_FOR_FUNCTION.get(identity.lower())
            if standard_selector is not None:
                identities.add(standard_selector)
    return sorted(identities)


def _reference_chains(nodes: list[dict[str, Any]], asset_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {node["node_id"]: node for node in nodes}
    chains: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for edge in asset_edges:
        node_ids = _path_to_root(edge["node_id"], by_id)
        signature = tuple(node_ids)
        if signature not in seen:
            seen.add(signature)
            chains.append({"node_ids": node_ids, "result_asset_edge_id": edge["edge_id"]})
        if len(chains) == 3:
            return chains
    if chains:
        return chains

    leaf = max(nodes, key=lambda node: (node["trace_index"], node["node_id"]))
    return [{"node_ids": _path_to_root(leaf["node_id"], by_id), "fallback_reason": "NO_ASSET_EDGE"}]


def _path_to_root(node_id: str, by_id: dict[str, dict[str, Any]]) -> list[str]:
    path: list[str] = []
    current = by_id[node_id]
    while True:
        path.append(current["node_id"])
        parent_id = current["parent_id"]
        if parent_id is None:
            return list(reversed(path))
        current = by_id[parent_id]


def _selector(identifier: str | None) -> str | None:
    return identifier.lower() if identifier and identifier.lower() in TRANSFER_SELECTORS else (identifier.lower() if identifier and identifier.lower().startswith("0x") and len(identifier) == 10 else None)

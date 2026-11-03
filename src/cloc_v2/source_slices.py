"""Offline source-request resolution and real-line Solidity slicing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .source_cache import atomic_json, norm_address
from .source_index import canonical_sha256


def _index_addresses(index: dict) -> dict[str, dict]: return {item["address"]: item for item in index.get("addresses", [])}
def _candidate(function: dict) -> dict: return {key: function.get(key) for key in ("function_id", "file_path", "contract_name", "function_kind", "function_name", "canonical_signature", "selector", "signature_start_line", "end_line")}

def _unique_implemented_function(hits: list[dict]) -> list[dict]:
    signatures = {item.get("canonical_signature") for item in hits}
    if len(hits) < 2 or len(signatures) != 1 or None in signatures:
        return []
    implemented = [item for item in hits if isinstance(item.get("body_start_line"), int)]
    return implemented if len(implemented) == 1 else []

def _match(functions: list[dict], function_name: str | None, selector: str | None) -> tuple[str, list[dict], list[str]]:
    if selector and selector != "UNKNOWN":
        hits = [item for item in functions if item.get("selector", "").lower() == selector.lower()]
        if len(hits) == 1: return "RESOLVED", hits, []
        implemented = _unique_implemented_function(hits)
        if implemented: return "RESOLVED", implemented, ["UNIQUE_IMPLEMENTATION_BODY_MATCH"]
        if len(hits) > 1: return "AMBIGUOUS", hits, ["MULTIPLE_SELECTOR_MATCHES"]
    signature = function_name if function_name and "(" in function_name else None
    if signature:
        hits = [item for item in functions if item.get("canonical_signature") == signature]
        if len(hits) == 1: return "RESOLVED", hits, []
        implemented = _unique_implemented_function(hits)
        if implemented: return "RESOLVED", implemented, ["UNIQUE_IMPLEMENTATION_BODY_MATCH"]
        if len(hits) > 1: return "AMBIGUOUS", hits, ["MULTIPLE_SIGNATURE_MATCHES"]
    special = {"fallback": "FALLBACK", "receive": "RECEIVE", "constructor": "CONSTRUCTOR"}.get((function_name or "").lower())
    if special:
        hits = [item for item in functions if item.get("function_kind") == special]
        if len(hits) == 1: return "RESOLVED", hits, []
        if len(hits) > 1: return "AMBIGUOUS", hits, ["MULTIPLE_SPECIAL_FUNCTIONS"]
    if function_name:
        hits = [item for item in functions if item.get("function_name") == function_name]
        if len(hits) == 1: return "RESOLVED", hits, ["UNIQUE_FUNCTION_NAME_FALLBACK"]
        implemented = _unique_implemented_function(hits)
        if implemented: return "RESOLVED", implemented, ["UNIQUE_FUNCTION_NAME_FALLBACK", "UNIQUE_IMPLEMENTATION_BODY_MATCH"]
        if len(hits) > 1: return "AMBIGUOUS", hits, ["OVERLOAD_OR_CROSS_FILE_AMBIGUITY"]
    return "FUNCTION_NOT_FOUND", [], []

def _resolve(index_by_address: dict[str, dict], requested_address: str, function_name: str | None, selector: str | None) -> tuple[str, str | None, list[str], list[dict], list[str]]:
    address = norm_address(requested_address); visited: list[str] = []
    while address and address not in visited:
        visited.append(address); record = index_by_address.get(address)
        if not record: return "PROXY_UNRESOLVED" if len(visited) > 1 else "SOURCE_MISSING", None, visited, [], ["ADDRESS_NOT_INDEXED"]
        if record.get("status") != "verified": return "SOURCE_MISSING", None, visited, [], record.get("uncertainties", [])
        status, hits, uncertainty = _match(record.get("functions", []), function_name, selector)
        if status != "FUNCTION_NOT_FOUND": return status, address, visited, hits, uncertainty
        if any(item.get("parse_status") == "PARSE_FAILED" for item in record.get("files", [])):
            return "PARSE_FAILED", address, visited, [], ["SOURCE_FILE_PARSE_FAILED"]
        if not record.get("files"):
            return "SOURCE_MISSING", address, visited, [], record.get("uncertainties", []) + ["NO_INDEXED_SOURCE_FILES"]
        proxy = record.get("proxy", {}); implementation = proxy.get("implementation_address")
        if not implementation: return status, address, visited, [], uncertainty
        try: address = norm_address(implementation)
        except ValueError: return "PROXY_UNRESOLVED", None, visited, [], ["INVALID_IMPLEMENTATION_ADDRESS"]
    return "PROXY_UNRESOLVED", None, visited, [], ["PROXY_CYCLE"]

def _line_source(cache_root: Path, chain: str, address: str, function: dict) -> str:
    path = cache_root / chain / address / function["file_path"]
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = function["signature_start_line"], function["end_line"]
    return "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, end + 1))

def requests_from_ranked(ranked: dict) -> list[dict]:
    if ranked.get("schema_version") != "phase1_ranked_chains_v1" or ranked.get("selection_audit", {}).get("all_chains_scored_exactly_once") is not True: raise ValueError("ranked input must be a complete formal phase1_ranked_chains_v1")
    requests = []
    for chain in ranked.get("selected_topk", []):
        for request in chain.get("source_requests", []):
            item = {"chain_id": chain.get("chain_id"), **request}
            if item not in requests: requests.append(item)
    return sorted(requests, key=lambda item: (item["chain_id"], item["node_id"], item.get("priority", 9), item.get("focus", "")))

def build_source_request_slices(index: dict, transaction_graph: dict, cache_root: str | Path, requests: list[dict], ranked: dict | None = None) -> dict:
    if index.get("schema_version") != "source_index_v1" or transaction_graph.get("schema_version") != "transaction_graph_v1": raise ValueError("invalid index or transaction graph schema")
    nodes = {node["node_id"]: node for node in transaction_graph.get("nodes", [])}; indexed = _index_addresses(index); slices = []
    for request in requests:
        node_id = request.get("node_id"); node = nodes.get(node_id)
        base = {"chain_id": request.get("chain_id"), "node_id": node_id, "focus": request.get("focus"), "priority": request.get("priority"), "requested_address": None, "source_address": None, "status": "SOURCE_MISSING", "function_match_status": "SOURCE_MISSING", "file_path": None, "contract_name": None, "function_name": None, "signature": None, "selector": None, "start_line": None, "end_line": None, "source": None, "candidate_functions": [], "proxy_resolution_path": [], "uncertainties": []}
        if not node or not node.get("callee_address"):
            base["uncertainties"].append("NODE_OR_CALLEE_ADDRESS_MISSING"); slices.append(base); continue
        base["requested_address"] = norm_address(node["callee_address"]); function_name = node.get("function_name") or node.get("function_selector") or None; selector = node.get("function_selector") or None
        status, source_address, path, candidates, uncertainties = _resolve(indexed, base["requested_address"], function_name, selector)
        base.update({"status": status, "function_match_status": status, "source_address": source_address, "proxy_resolution_path": path, "candidate_functions": [_candidate(item) for item in candidates], "uncertainties": uncertainties})
        if status == "RESOLVED":
            function = candidates[0]; base.update({"file_path": function["file_path"], "contract_name": function["contract_name"], "function_name": function["function_name"], "signature": function["canonical_signature"], "selector": function["selector"], "start_line": function["signature_start_line"], "end_line": function["end_line"]})
            try:
                base["source"] = _line_source(Path(cache_root), index["chain"], source_address, function)
                if function["end_line"] - function["signature_start_line"] + 1 > 200: base["uncertainties"].append("LONG_FUNCTION")
            except OSError:
                base["status"] = base["function_match_status"] = "SOURCE_MISSING"; base["uncertainties"].append("INDEXED_FILE_UNAVAILABLE")
        slices.append(base)
    return {"schema_version": "source_request_slices_v1", "chain": index["chain"], "source_index_sha256": canonical_sha256(index), "transaction_graph_sha256": canonical_sha256(transaction_graph), "phase1_ranked_chains_sha256": canonical_sha256(ranked) if ranked is not None else None, "requests": slices, "metadata": {"request_count": len(slices), "resolved_count": sum(item["function_match_status"] == "RESOLVED" for item in slices)}}

def write_source_request_slices(payload: dict, path: str | Path) -> None: atomic_json(path, payload)

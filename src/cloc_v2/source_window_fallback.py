"""Deterministic, offline fallback windows for already verified Solidity source."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .source_index import canonical_sha256, parse_solidity_source

WINDOW_LINES = 120
WINDOW_OVERLAP_LINES = 20
MAX_FILES = 3
MAX_WINDOWS = 12
MODE = "VERIFIED_SOURCE_WINDOW_FALLBACK"
_UNCERTAINTIES = ("EVIDENCE_TO_WINDOW_ASSOCIATION_COARSE", "FUNCTION_IDENTITY_UNRESOLVED", "SOURCE_WINDOW_FALLBACK_USED")


def _norm(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and value.startswith("0x") and len(value) == 42 else None


def _windows(line_count: int) -> list[tuple[int, int]]:
    if line_count <= 0:
        return []
    result: list[tuple[int, int]] = []; start = 1
    while True:
        end = min(start + WINDOW_LINES - 1, line_count); result.append((start, end))
        if end == line_count: return result
        start = end - WINDOW_OVERLAP_LINES + 1


def _window_function(parsed: Mapping[str, Any], start: int, end: int) -> tuple[str, str] | None:
    candidates = []
    for function in parsed.get("functions", ()):
        name, signature = function.get("function_name"), function.get("canonical_signature")
        first, last = function.get("signature_start_line"), function.get("end_line")
        if isinstance(name, str) and isinstance(first, int) and isinstance(last, int) and first <= end and last >= start:
            candidates.append((first, last, name, signature if isinstance(signature, str) else name))
    if not candidates:
        return None
    _, _, name, signature = sorted(candidates)[0]
    return name, signature


def build_verified_source_window_fallback(index: Mapping[str, Any], transaction_graph: Mapping[str, Any], exact_slices: Mapping[str, Any], cache_root: str | Path) -> dict[str, Any]:
    """Return exact slices unchanged, or verified-source windows when none resolved.

    The returned payload keeps the existing ``source_request_slices_v1`` shape so
    that the bridge remains the sole evidence-to-prompt translator.
    """
    if not isinstance(index, Mapping) or index.get("schema_version") != "source_index_v1" or not isinstance(transaction_graph, Mapping) or transaction_graph.get("schema_version") != "transaction_graph_v1" or not isinstance(exact_slices, Mapping) or exact_slices.get("schema_version") != "source_request_slices_v1":
        raise ValueError("SOURCE_WINDOW_FALLBACK_INPUT_INVALID")
    exact = list(exact_slices.get("requests", ()))
    if any(isinstance(item, Mapping) and item.get("function_match_status") == "RESOLVED" for item in exact):
        payload = dict(exact_slices); payload["requests"] = exact; payload["metadata"] = {**dict(exact_slices.get("metadata", {})), "resolution_mode": "EXACT_FUNCTION_SLICE", "window_count": 0, "window_file_count": 0}
        return payload
    nodes = {item.get("node_id"): item for item in transaction_graph.get("nodes", ()) if isinstance(item, Mapping) and isinstance(item.get("node_id"), str)}
    node_by_address: dict[str, list[str]] = {}
    for node_id, node in nodes.items():
        address = _norm(node.get("callee_address"))
        if address: node_by_address.setdefault(address, []).append(node_id)
    requested: set[str] = set()
    for request in exact:
        if isinstance(request, Mapping):
            address = _norm(request.get("requested_address"))
            if address: requested.add(address)
            node = nodes.get(request.get("node_id")); address = _norm(node.get("callee_address")) if node else None
            if address: requested.add(address)
    selected: list[tuple[int, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for address_record in index.get("addresses", ()):
        if not isinstance(address_record, Mapping) or address_record.get("status") != "verified" or not address_record.get("verified"):
            continue
        address = _norm(address_record.get("address"))
        path = tuple(_norm(item) for item in address_record.get("proxy_resolution_path", ()) if _norm(item))
        if not address or not (address in requested or requested.intersection(path)):
            continue
        implementation = address not in requested and bool(requested.intersection(path))
        direct = address in requested
        for file_record in address_record.get("files", ()):
            if not isinstance(file_record, Mapping) or not isinstance(file_record.get("file_path"), str) or not file_record["file_path"].lower().endswith(".sol"):
                continue
            names = {item.get("function_name") for item in file_record.get("functions", ()) if isinstance(item, Mapping)}
            runtime_names = {str(nodes[node_id].get("function_name")) for node_id in node_by_address.get(address, ()) if isinstance(nodes[node_id].get("function_name"), str)}
            priority = 0 if implementation else 1 if direct else 2 if names.intersection(runtime_names) else 3
            selected.append((priority, address, address_record, file_record))
    selected.sort(key=lambda item: (item[0], item[1], str(item[3]["file_path"]).replace("\\", "/")))
    unique_files: list[tuple[int, str, Mapping[str, Any], Mapping[str, Any]]] = []; seen: set[tuple[str, str]] = set()
    for item in selected:
        key = (item[1], str(item[3]["file_path"]).replace("\\", "/"))
        if key not in seen:
            seen.add(key); unique_files.append(item)
        if len(unique_files) == MAX_FILES: break
    requests: list[dict[str, Any]] = []; root = Path(cache_root); chain = index.get("chain")
    for _, address, record, file_record in unique_files:
        path = str(file_record["file_path"]).replace("\\", "/")
        file_path = root / str(chain) / address / path
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        expected = file_record.get("file_sha256")
        if isinstance(expected, str) and hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
            continue
        parsed = parse_solidity_source(text)
        if parsed.get("parse_status") != "PARSED":
            continue
        node_ids = sorted(node_by_address.get(address, ()))
        if not node_ids:
            for original in exact:
                if isinstance(original, Mapping) and isinstance(original.get("node_id"), str):
                    node_ids.append(original["node_id"])
                    break
        if not node_ids:
            continue
        lines = text.splitlines()
        for start, end in _windows(len(lines)):
            function = _window_function(parsed, start, end)
            if function is None:
                continue
            name, signature = function
            requests.append({"chain_id": "Q", "node_id": node_ids[0], "focus": "phase2_localization", "priority": 9, "requested_address": node_by_address and next((candidate for candidate, ids in node_by_address.items() if node_ids[0] in ids), None), "source_address": address, "status": "WINDOW_FALLBACK", "function_match_status": "WINDOW_FALLBACK", "file_path": path, "contract_name": None, "function_name": name, "signature": signature, "selector": None, "start_line": start, "end_line": end, "source": "\n".join(lines[start - 1:end]), "candidate_functions": [], "proxy_resolution_path": list(record.get("proxy_resolution_path", ())), "uncertainties": list(_UNCERTAINTIES), "resolution_mode": MODE})
            if len(requests) == MAX_WINDOWS: break
        if len(requests) == MAX_WINDOWS: break
    requests.sort(key=lambda item: (str(item["file_path"]), item["start_line"], item["end_line"], item["node_id"]))
    return {"schema_version": "source_request_slices_v1", "chain": chain, "source_index_sha256": canonical_sha256(index), "transaction_graph_sha256": canonical_sha256(transaction_graph), "phase1_ranked_chains_sha256": None, "requests": requests, "metadata": {"request_count": len(requests), "resolved_count": 0, "resolution_mode": MODE if requests else "NONE", "window_count": len(requests), "window_file_count": len({item["file_path"] for item in requests})}}

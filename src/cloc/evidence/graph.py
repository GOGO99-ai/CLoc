"""Normalize execution records into a transaction graph."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from cloc.evidence.models import NormalizedCallNode, TransactionGraph, TransactionGraphEdge


FOUR_BYTE_SELECTOR_RE = re.compile(r"^(?:0x)?([0-9a-fA-F]{8})$")


def build_transaction_graph(call_tree: dict[str, Any]) -> TransactionGraph:
    """Build a transaction graph without modifying or otherwise consuming input.

    ``call_tree`` must be the root call node, rather than a wrapper object such
    as ``{"call_tree": ...}``.  File-format handling belongs in
    :func:`load_call_tree`, keeping this function pure and directly testable.
    """
    _validate_call_tree(call_tree)
    source_tree_sha256 = _canonical_tree_sha256(call_tree)
    nodes: list[NormalizedCallNode] = []
    call_edges: list[TransactionGraphEdge] = []
    order_edges: list[TransactionGraphEdge] = []

    def visit(
        raw_node: dict[str, Any],
        *,
        parent: NormalizedCallNode | None,
        trace_path: str,
        depth: int,
        sibling_order: int,
    ) -> NormalizedCallNode:
        trace_index = len(nodes) + 1
        node = _normalize_node(
            raw_node,
            node_id=f"n{trace_index:03d}",
            trace_index=trace_index,
            trace_path=trace_path,
            depth=depth,
            parent=parent,
            sibling_order=sibling_order,
        )
        nodes.append(node)

        if parent is not None:
            call_edges.append(
                TransactionGraphEdge(
                    edge_id=f"call_e{len(call_edges) + 1:03d}",
                    edge_type="CALL",
                    src=parent.node_id,
                    dst=node.node_id,
                    evidence_level="OBSERVED",
                    reason="direct parent-child relation in call tree",
                    trace_refs=(parent.trace_path, node.trace_path),
                )
            )

        child_nodes: list[NormalizedCallNode] = []
        for child_index, child in enumerate(raw_node.get("children", []) or []):
            if not isinstance(child, dict):
                raise ValueError(f"Invalid child at trace path {trace_path}/{child_index}: expected object")
            child_nodes.append(
                visit(
                    child,
                    parent=node,
                    trace_path=f"{trace_path}/{child_index}",
                    depth=depth + 1,
                    sibling_order=child_index,
                )
            )

        for previous, following in zip(child_nodes, child_nodes[1:]):
            order_edges.append(
                TransactionGraphEdge(
                    edge_id=f"order_e{len(order_edges) + 1:03d}",
                    edge_type="ORDER",
                    src=previous.node_id,
                    dst=following.node_id,
                    evidence_level="OBSERVED",
                    reason="adjacent sibling calls under the same parent",
                    trace_refs=(previous.trace_path, following.trace_path),
                )
            )
        return node

    root = visit(call_tree, parent=None, trace_path="0", depth=0, sibling_order=0)
    node_order = {node.node_id: node.trace_index for node in nodes}
    edge_key = lambda edge: (node_order[edge.src], node_order[edge.dst], edge.edge_type)
    return TransactionGraph(
        root_id=root.node_id,
        source_tree_sha256=source_tree_sha256,
        nodes=tuple(nodes),
        call_edges=tuple(sorted(call_edges, key=edge_key)),
        order_edges=tuple(sorted(order_edges, key=edge_key)),
    )


def load_call_tree(path: str | Path) -> dict[str, Any]:
    """Read a direct call tree or the legacy ``{"call_tree": ...}`` wrapper."""
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Input call tree does not exist: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input is not valid JSON: {input_path} ({exc.msg})") from exc

    if isinstance(payload, dict) and "call_tree" in payload:
        payload = payload["call_tree"]
    _validate_call_tree(payload)
    return payload


def serialize_transaction_graph(graph: TransactionGraph) -> str:
    """Return a stable JSON representation without any time-dependent fields."""
    payload = graph.to_dict()
    if _is_legacy_status_graph(graph):
        for node in payload["nodes"]:
            for field in (
                "gas_used",
                "execution_status",
                "execution_status_source",
                "execution_status_uncertainties",
                "execution_success_evidence",
            ):
                node.pop(field)
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def write_transaction_graph(graph: TransactionGraph, output_path: str | Path) -> Path:
    """Serialize an already-built graph to an explicitly selected path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_transaction_graph(graph), encoding="utf-8")
    return path


def _normalize_node(
    raw_node: dict[str, Any],
    *,
    node_id: str,
    trace_index: int,
    trace_path: str,
    depth: int,
    parent: NormalizedCallNode | None,
    sibling_order: int,
) -> NormalizedCallNode:
    function_name, function_selector = _split_function(raw_node.get("function"))
    caller_address, caller_source = _caller_for(raw_node, parent)
    (
        execution_status,
        execution_status_source,
        execution_status_uncertainties,
        execution_success_evidence,
    ) = _normalize_call_status(raw_node)
    return NormalizedCallNode(
        node_id=node_id,
        trace_index=trace_index,
        trace_path=trace_path,
        depth=depth,
        parent_id=parent.node_id if parent is not None else None,
        sibling_order=sibling_order,
        caller_address=caller_address,
        caller_source=caller_source,
        callee_address=_nullable_value(raw_node.get("address")),
        node_type=_nullable_value(raw_node.get("type")),
        function_name=function_name,
        function_selector=function_selector,
        call_type=_nullable_value(raw_node.get("call_type")),
        params_raw=raw_node.get("params"),
        decoded_params=raw_node.get("decoded_params"),
        value=raw_node.get("value"),
        gas=raw_node.get("gas"),
        gas_used=raw_node.get("gas_used", raw_node.get("gasUsed")),
        execution_status=execution_status,
        execution_status_source=execution_status_source,
        execution_status_uncertainties=execution_status_uncertainties,
        execution_success_evidence=execution_success_evidence,
    )


def _caller_for(raw_node: dict[str, Any], parent: NormalizedCallNode | None) -> tuple[str | None, str]:
    if parent is None:
        return None, "UNKNOWN"

    explicit_from = raw_node.get("from")
    if explicit_from is None:
        explicit_from = raw_node.get("from_address")
    if explicit_from is not None:
        return explicit_from, "EXPLICIT_FROM"

    if parent.callee_address is not None:
        return parent.callee_address, "DERIVED_FROM_PARENT"
    return None, "UNKNOWN"


def _split_function(raw_function: Any) -> tuple[str | None, str | None]:
    if raw_function is None:
        return None, None
    function = str(raw_function).strip()
    if not function:
        return None, None

    # The current trace formatter can append ``{value: ...}`` to a selector or
    # function name.  It is not part of the callable identifier.
    identifier = function.split("{", 1)[0].strip()
    selector_match = FOUR_BYTE_SELECTOR_RE.fullmatch(identifier)
    if selector_match:
        return None, f"0x{selector_match.group(1).lower()}"
    return identifier or None, None


def _nullable_value(value: Any) -> Any:
    return value if value is not None else None


def _normalize_call_status(raw_node: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...], bool]:
    if "call_status" not in raw_node:
        return (
            "UNKNOWN",
            "NOT_REPORTED_BY_CALL_TREE",
            ("CALL_SUCCESS_STATUS_NOT_REPORTED",),
            False,
        )
    value = raw_node.get("call_status")
    if not isinstance(value, Mapping):
        return _invalid_call_status("CALL_STATUS_STRUCTURE_INVALID")

    status = value.get("status")
    source = value.get("decision_source")
    output_present = value.get("output_present")
    output_valid_hex = value.get("output_valid_hex")
    error_evidence = value.get("error_failure_evidence")
    revert_evidence = value.get("revert_reason_failure_evidence")
    success_evidence = value.get("usable_as_success_evidence")
    uncertainties = value.get("uncertainties")
    if (
        status not in {"SUCCESS", "FAILED", "UNKNOWN"}
        or not isinstance(source, str)
        or not source
        or not all(
            isinstance(item, bool)
            for item in (
                output_present,
                output_valid_hex,
                error_evidence,
                revert_evidence,
                success_evidence,
            )
        )
        or not isinstance(uncertainties, (list, tuple))
        or any(not isinstance(item, str) or not item for item in uncertainties)
    ):
        return _invalid_call_status("CALL_STATUS_STRUCTURE_INVALID")

    normalized_uncertainties = tuple(sorted(set(uncertainties)))
    if status == "SUCCESS":
        if (
            source != "CALLTRACER_OUTPUT"
            or not output_present
            or not output_valid_hex
            or error_evidence
            or revert_evidence
            or not success_evidence
            or normalized_uncertainties
        ):
            return _invalid_call_status("CALL_STATUS_SUCCESS_EVIDENCE_INVALID")
    elif status == "FAILED":
        expected_source = "CALLTRACER_ERROR" if error_evidence else "CALLTRACER_REVERT_REASON"
        if (
            not (error_evidence or revert_evidence)
            or source != expected_source
            or success_evidence
            or normalized_uncertainties
        ):
            return _invalid_call_status("CALL_STATUS_FAILURE_EVIDENCE_INVALID")
    elif source != "UNRESOLVED" or success_evidence or error_evidence or revert_evidence:
        return _invalid_call_status("CALL_STATUS_UNKNOWN_EVIDENCE_INVALID")

    return status, source, normalized_uncertainties, success_evidence


def _invalid_call_status(reason: str) -> tuple[str, str, tuple[str, ...], bool]:
    return "UNKNOWN", "INVALID_CALL_STATUS", (reason,), False


def _is_legacy_status_graph(graph: TransactionGraph) -> bool:
    return all(
        node.execution_status == "UNKNOWN"
        and node.execution_status_source == "NOT_REPORTED_BY_CALL_TREE"
        and node.execution_status_uncertainties == ("CALL_SUCCESS_STATUS_NOT_REPORTED",)
        and not node.execution_success_evidence
        for node in graph.nodes
    )


def _canonical_tree_sha256(call_tree: dict[str, Any]) -> str:
    try:
        canonical = json.dumps(call_tree, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Call tree must contain JSON-serializable values") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_call_tree(call_tree: Any) -> None:
    if not isinstance(call_tree, dict):
        raise ValueError("Call tree root must be a JSON object")
    children = call_tree.get("children", [])
    if children is None:
        return
    if not isinstance(children, list):
        raise ValueError("Call tree field 'children' must be a list when present")

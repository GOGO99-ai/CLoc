"""Immutable-shaped data models for the phase-1 transaction graph.

These models intentionally contain observable call-tree facts only.  They do
not model causal, source-code, or vulnerability information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedCallNode:
    node_id: str
    trace_index: int
    trace_path: str
    depth: int
    parent_id: str | None
    sibling_order: int
    caller_address: str | None
    caller_source: str
    callee_address: str | None
    node_type: str | None
    function_name: str | None
    function_selector: str | None
    call_type: str | None
    params_raw: Any
    decoded_params: Any
    value: Any
    gas: Any
    gas_used: Any = None
    execution_status: str = "UNKNOWN"
    execution_status_source: str = "NOT_REPORTED_BY_CALL_TREE"
    execution_status_uncertainties: tuple[str, ...] = ("CALL_SUCCESS_STATUS_NOT_REPORTED",)
    execution_success_evidence: bool = False
    contract_role: str = "UNKNOWN"
    semantic_flags: tuple[str, ...] = ()
    source_ref: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "trace_index": self.trace_index,
            "trace_path": self.trace_path,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "sibling_order": self.sibling_order,
            "caller_address": self.caller_address,
            "caller_source": self.caller_source,
            "callee_address": self.callee_address,
            "node_type": self.node_type,
            "function_name": self.function_name,
            "function_selector": self.function_selector,
            "call_type": self.call_type,
            "params_raw": self.params_raw,
            "decoded_params": self.decoded_params,
            "value": self.value,
            "gas": self.gas,
            "gas_used": self.gas_used,
            "execution_status": self.execution_status,
            "execution_status_source": self.execution_status_source,
            "execution_status_uncertainties": list(self.execution_status_uncertainties),
            "execution_success_evidence": self.execution_success_evidence,
            "contract_role": self.contract_role,
            "semantic_flags": list(self.semantic_flags),
            "source_ref": self.source_ref,
            "raw_evidence": {"trace_path": self.trace_path},
        }


@dataclass(frozen=True)
class TransactionGraphEdge:
    edge_id: str
    edge_type: str
    src: str
    dst: str
    evidence_level: str
    reason: str
    trace_refs: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "src": self.src,
            "dst": self.dst,
            "evidence_level": self.evidence_level,
            "reason": self.reason,
            "trace_refs": list(self.trace_refs),
        }


@dataclass(frozen=True)
class TransactionGraph:
    root_id: str
    source_tree_sha256: str
    nodes: tuple[NormalizedCallNode, ...]
    call_edges: tuple[TransactionGraphEdge, ...]
    order_edges: tuple[TransactionGraphEdge, ...]

    SCHEMA_VERSION = "transaction_graph_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "root_id": self.root_id,
            "source_tree_sha256": self.source_tree_sha256,
            "nodes": [node.to_dict() for node in self.nodes],
            "call_edges": [edge.to_dict() for edge in self.call_edges],
            "order_edges": [edge.to_dict() for edge in self.order_edges],
            "metadata": {
                "node_count": len(self.nodes),
                "call_edge_count": len(self.call_edges),
                "order_edge_count": len(self.order_edges),
            },
        }

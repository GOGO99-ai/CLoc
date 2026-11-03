"""Immutable records for the phase-2D-1 candidate causal graph skeleton."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class NodeRef:
    node_id: str; trace_index: int; trace_path: str; parent_id: str | None
    def to_dict(self) -> dict[str, Any]: return {"node_id": self.node_id, "trace_index": self.trace_index, "trace_path": self.trace_path, "parent_id": self.parent_id}

@dataclass(frozen=True)
class EvidenceEdge:
    edge_id: str; edge_type: str; src: str; dst: str; evidence_level: str; reason: str; trace_refs: tuple[str, ...]; source_stage: str; source_refs: tuple[str, ...]; attributes: dict[str, Any]
    def to_dict(self) -> dict[str, Any]: return {"edge_id": self.edge_id, "edge_type": self.edge_type, "src": self.src, "dst": self.dst, "evidence_level": self.evidence_level, "reason": self.reason, "trace_refs": list(self.trace_refs), "source_stage": self.source_stage, "source_refs": list(self.source_refs), "attributes": self.attributes}

@dataclass(frozen=True)
class ResultSeed:
    result_id: str; node_id: str; result_type: str; confidence: str; evidence_level: str; selection_reason: str
    def to_dict(self) -> dict[str, Any]: return {"result_id": self.result_id, "node_id": self.node_id, "result_type": self.result_type, "confidence": self.confidence, "evidence_level": self.evidence_level, "selection_reason": self.selection_reason}

@dataclass(frozen=True)
class CandidateCausalGraph:
    transaction_graph_sha256: str; runtime_annotations_sha256: str; observed_asset_ledger_sha256: str; asset_continuity_sha256: str; result_candidates_sha256: str
    nodes: tuple[NodeRef, ...]; edges: tuple[EvidenceEdge, ...]; result_seeds: tuple[ResultSeed, ...]; root_results_without_edge: tuple[str, ...]
    SCHEMA_VERSION = "candidate_causal_graph_v1"
    def to_dict(self) -> dict[str, Any]:
        counts = {kind: sum(edge.edge_type == kind for edge in self.edges) for kind in ("CALL", "ORDER", "ASSET", "RESULT")}
        return {"schema_version": self.SCHEMA_VERSION, "transaction_graph_sha256": self.transaction_graph_sha256, "runtime_annotations_sha256": self.runtime_annotations_sha256, "observed_asset_ledger_sha256": self.observed_asset_ledger_sha256, "asset_continuity_sha256": self.asset_continuity_sha256, "result_candidates_sha256": self.result_candidates_sha256, "nodes": [x.to_dict() for x in self.nodes], "edges": [x.to_dict() for x in self.edges], "result_seeds": [x.to_dict() for x in self.result_seeds], "root_results_without_edge": list(self.root_results_without_edge), "edge_type_support": {"CALL":"SUPPORTED", "ORDER":"SUPPORTED", "ASSET":"SUPPORTED", "RESULT":"SUPPORTED", "PARAM":"NOT_IMPLEMENTED", "STATE":"NOT_IMPLEMENTED"}, "metadata": {"node_count":len(self.nodes), "call_edge_count":counts["CALL"], "order_edge_count":counts["ORDER"], "asset_edge_count":counts["ASSET"], "result_edge_count":counts["RESULT"], "result_seed_count":len(self.result_seeds), "dangling_edge_count":0}}

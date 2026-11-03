"""Models for deterministic candidate causal-chain assembly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChainAssemblyConfig:
    max_branches_per_chain: int = 4
    max_branch_depth: int = 3
    max_total_nodes_per_chain: int = 32
    max_hypothesis_branches: int = 2
    max_edges_per_state_object_per_chain: int = 3
    max_branch_candidates_per_chain: int = 100

    def to_dict(self) -> dict[str, int]:
        return {
            "max_branches_per_chain": self.max_branches_per_chain,
            "max_branch_depth": self.max_branch_depth,
            "max_total_nodes_per_chain": self.max_total_nodes_per_chain,
            "max_hypothesis_branches": self.max_hypothesis_branches,
            "max_edges_per_state_object_per_chain": self.max_edges_per_state_object_per_chain,
            "max_branch_candidates_per_chain": self.max_branch_candidates_per_chain,
        }


@dataclass(frozen=True)
class EvidenceBranch:
    branch_id: str
    attachment_node_id: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    semantic_edge_id: str
    semantic_edge_type: str
    evidence_level: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "attachment_node_id": self.attachment_node_id,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "semantic_edge_id": self.semantic_edge_id,
            "semantic_edge_type": self.semantic_edge_type,
            "evidence_level": self.evidence_level,
        }


@dataclass(frozen=True)
class CandidateCausalChain:
    chain_id: str
    result_id: str
    result_node_id: str
    semantic_views: tuple[str, ...]
    backbone_node_ids: tuple[str, ...]
    backbone_edge_ids: tuple[str, ...]
    supporting_edge_ids: tuple[str, ...]
    stop_reason: str
    branches: tuple[EvidenceBranch, ...]
    all_node_ids: tuple[str, ...]
    all_edge_ids: tuple[str, ...]
    structural_features: dict[str, Any]
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "result_id": self.result_id,
            "result_node_id": self.result_node_id,
            "semantic_views": list(self.semantic_views),
            "backbone": {
                "node_ids": list(self.backbone_node_ids),
                "edge_ids": list(self.backbone_edge_ids),
                "supporting_edge_ids": list(self.supporting_edge_ids),
                "stop_reason": self.stop_reason,
            },
            "branches": [branch.to_dict() for branch in self.branches],
            "all_node_ids": list(self.all_node_ids),
            "all_edge_ids": list(self.all_edge_ids),
            "structural_features": self.structural_features,
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class CandidateCausalChainsResult:
    candidate_causal_graph_sha256: str
    causal_backbones_sha256: str
    assembly_config: ChainAssemblyConfig
    chains: tuple[CandidateCausalChain, ...]
    result_coverage: dict[str, Any]
    assembly_audit: dict[str, Any]

    SCHEMA_VERSION = "candidate_causal_chains_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "candidate_causal_graph_sha256": self.candidate_causal_graph_sha256,
            "causal_backbones_sha256": self.causal_backbones_sha256,
            "assembly_config": self.assembly_config.to_dict(),
            "chains": [chain.to_dict() for chain in self.chains],
            "result_coverage": self.result_coverage,
            "assembly_audit": self.assembly_audit,
        }

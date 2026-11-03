"""Phase-2C-1 strong-association records, separate from observed facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssetContinuityEdge:
    edge_id: str
    src_node_id: str
    dst_node_id: str
    src_fact_id: str
    dst_fact_id: str
    asset_type: str
    asset_address: dict[str, str] | None
    via_address: dict[str, str]
    allocated_amount_raw: str
    match_type: str
    evidence_level: str
    reason: str
    trace_refs: tuple[str, ...]
    matching_basis: tuple[str, ...]
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_type": "ASSET",
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "src_fact_id": self.src_fact_id,
            "dst_fact_id": self.dst_fact_id,
            "asset_type": self.asset_type,
            "asset_address": self.asset_address,
            "via_address": self.via_address,
            "allocated_amount_raw": self.allocated_amount_raw,
            "match_type": self.match_type,
            "evidence_level": self.evidence_level,
            "reason": self.reason,
            "trace_refs": list(self.trace_refs),
            "matching_basis": list(self.matching_basis),
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class AmountObservation:
    fact_id: str
    node_id: str
    trace_index: int
    trace_path: str
    asset_type: str
    asset_address: dict[str, str] | None
    holder_address: dict[str, str]
    amount_raw: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "node_id": self.node_id,
            "trace_index": self.trace_index,
            "trace_path": self.trace_path,
            "asset_type": self.asset_type,
            "asset_address": self.asset_address,
            "holder_address": self.holder_address,
            "amount_raw": self.amount_raw,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SelfTransferObservation:
    fact_id: str
    node_id: str
    trace_index: int
    asset_type: str
    asset_address: dict[str, str] | None
    holder_address: dict[str, str]
    amount_raw: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "node_id": self.node_id,
            "trace_index": self.trace_index,
            "asset_type": self.asset_type,
            "asset_address": self.asset_address,
            "holder_address": self.holder_address,
            "amount_raw": self.amount_raw,
            "reason": "self transfer is retained as an observation but excluded from continuity matching",
        }


@dataclass(frozen=True)
class FinancingEpisode:
    episode_id: str
    financing_type: str
    initiation_node_id: str
    callback_node_id: str
    provider_address: dict[str, str] | None
    receiver_address: dict[str, str] | None
    asset_address: dict[str, str] | None
    funding_fact_ids: tuple[str, ...]
    repayment_fact_ids: tuple[str, ...]
    principal_raw: str
    repayment_raw: str
    status: str
    evidence_level: str
    reason: str
    trace_refs: tuple[str, ...]
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "financing_type": self.financing_type,
            "initiation_node_id": self.initiation_node_id,
            "callback_node_id": self.callback_node_id,
            "provider_address": self.provider_address,
            "receiver_address": self.receiver_address,
            "asset_address": self.asset_address,
            "funding_fact_ids": list(self.funding_fact_ids),
            "repayment_fact_ids": list(self.repayment_fact_ids),
            "principal_raw": self.principal_raw,
            "repayment_raw": self.repayment_raw,
            "status": self.status,
            "evidence_level": self.evidence_level,
            "reason": self.reason,
            "trace_refs": list(self.trace_refs),
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class AssetContinuityResult:
    transaction_graph_sha256: str
    runtime_annotations_sha256: str
    observed_asset_ledger_sha256: str
    asset_edges: tuple[AssetContinuityEdge, ...]
    financing_episodes: tuple[FinancingEpisode, ...]
    unmatched_outflows: tuple[AmountObservation, ...]
    remaining_inflows: tuple[AmountObservation, ...]
    self_transfers: tuple[SelfTransferObservation, ...]

    SCHEMA_VERSION = "asset_continuity_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "transaction_graph_sha256": self.transaction_graph_sha256,
            "runtime_annotations_sha256": self.runtime_annotations_sha256,
            "observed_asset_ledger_sha256": self.observed_asset_ledger_sha256,
            "asset_edges": [item.to_dict() for item in self.asset_edges],
            "financing_episodes": [item.to_dict() for item in self.financing_episodes],
            "unmatched_outflows": [item.to_dict() for item in self.unmatched_outflows],
            "remaining_inflows": [item.to_dict() for item in self.remaining_inflows],
            "self_transfers": [item.to_dict() for item in self.self_transfers],
            "metadata": {
                "asset_edge_count": len(self.asset_edges),
                "exact_edge_count": sum(item.match_type == "EXACT" for item in self.asset_edges),
                "partial_edge_count": sum(item.match_type == "PARTIAL" for item in self.asset_edges),
                "merged_edge_count": sum(item.match_type == "MERGED" for item in self.asset_edges),
                "financing_episode_count": len(self.financing_episodes),
                "complete_financing_count": sum(item.status == "COMPLETE" for item in self.financing_episodes),
                "unmatched_outflow_count": len(self.unmatched_outflows),
                "remaining_inflow_count": len(self.remaining_inflows),
                "self_transfer_count": len(self.self_transfers),
            },
        }

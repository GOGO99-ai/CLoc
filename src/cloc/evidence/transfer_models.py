"""Typed transfer observations and uncertainty records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssetTransferFact:
    fact_id: str
    node_id: str
    trace_index: int
    trace_path: str
    transfer_kind: str
    asset_type: str
    asset_address: dict[str, str] | None
    operator_address: dict[str, str] | None
    from_address: dict[str, str] | None
    to_address: dict[str, str] | None
    amount_raw: str | None
    status: str
    evidence_level: str
    reason: str
    trace_refs: tuple[str, ...]
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "node_id": self.node_id,
            "trace_index": self.trace_index,
            "trace_path": self.trace_path,
            "transfer_kind": self.transfer_kind,
            "asset_type": self.asset_type,
            "asset_address": self.asset_address,
            "operator_address": self.operator_address,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "amount_raw": self.amount_raw,
            "status": self.status,
            "evidence_level": self.evidence_level,
            "reason": self.reason,
            "trace_refs": list(self.trace_refs),
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class AddressFlowSummary:
    asset_type: str
    asset_address: dict[str, str] | None
    address: dict[str, str]
    incoming_transfer_count: int
    outgoing_transfer_count: int
    incoming_amount_raw: str
    outgoing_amount_raw: str
    observed_delta_raw: str
    completeness: str = "PARTIAL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "asset_address": self.asset_address,
            "address": self.address,
            "incoming_transfer_count": self.incoming_transfer_count,
            "outgoing_transfer_count": self.outgoing_transfer_count,
            "incoming_amount_raw": self.incoming_amount_raw,
            "outgoing_amount_raw": self.outgoing_amount_raw,
            "observed_delta_raw": self.observed_delta_raw,
            "completeness": self.completeness,
        }


@dataclass(frozen=True)
class ObservedAssetLedger:
    transaction_graph_sha256: str
    runtime_annotations_sha256: str
    transfer_facts: tuple[AssetTransferFact, ...]
    address_flow_summaries: tuple[AddressFlowSummary, ...]

    SCHEMA_VERSION = "observed_asset_ledger_v1"

    def to_dict(self) -> dict[str, Any]:
        complete_facts = [fact for fact in self.transfer_facts if fact.status == "COMPLETE"]
        asset_keys = {
            (fact.asset_type, fact.asset_address["normalized"] if fact.asset_address is not None else "NATIVE")
            for fact in self.transfer_facts
            if fact.asset_type == "NATIVE" or fact.asset_address is not None
        }
        participants = {
            address["normalized"]
            for fact in complete_facts
            for address in (fact.from_address, fact.to_address)
            if address is not None
        }
        return {
            "schema_version": self.SCHEMA_VERSION,
            "transaction_graph_sha256": self.transaction_graph_sha256,
            "runtime_annotations_sha256": self.runtime_annotations_sha256,
            "transfer_facts": [fact.to_dict() for fact in self.transfer_facts],
            "address_flow_summaries": [summary.to_dict() for summary in self.address_flow_summaries],
            "metadata": {
                "complete_fact_count": len(complete_facts),
                "incomplete_fact_count": len(self.transfer_facts) - len(complete_facts),
                "token_transfer_fact_count": sum(fact.asset_type == "TOKEN" for fact in self.transfer_facts),
                "native_transfer_fact_count": sum(fact.asset_type == "NATIVE" for fact in self.transfer_facts),
                "unique_asset_count": len(asset_keys),
                "unique_participant_count": len(participants),
            },
        }

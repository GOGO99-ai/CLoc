"""Outcome candidates and their selection settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResultSignal:
    signal_type: str
    value: str
    reason: str
    trace_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "value": self.value,
            "reason": self.reason,
            "trace_refs": list(self.trace_refs),
        }


@dataclass(frozen=True)
class ResultCandidate:
    result_id: str
    node_id: str
    trace_index: int
    result_type: str
    confidence: str
    evidence_level: str
    asset_type: str
    asset_address: dict[str, str] | None
    from_address: dict[str, str]
    to_address: dict[str, str]
    amount_raw: str
    source_fact_ids: tuple[str, ...]
    signals: tuple[ResultSignal, ...]
    negative_evidence: tuple[str, ...]
    selection_reason: str
    uncertainties: tuple[str, ...]
    outflow_score_audit: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "node_id": self.node_id,
            "result_type": self.result_type,
            "confidence": self.confidence,
            "evidence_level": self.evidence_level,
            "asset_type": self.asset_type,
            "asset_address": self.asset_address,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "amount_raw": self.amount_raw,
            "source_fact_ids": list(self.source_fact_ids),
            "signals": [signal.to_dict() for signal in self.signals],
            "negative_evidence": list(self.negative_evidence),
            "selection_reason": self.selection_reason,
            "uncertainties": list(self.uncertainties),
            **({"outflow_score_audit": self.outflow_score_audit} if self.outflow_score_audit is not None else {}),
        }


@dataclass(frozen=True)
class ResultDetectionConfig:
    outflow_lambda: str = "0.65"
    profit_complete_financing_weight: str = "0.30"
    profit_after_last_repayment_weight: str = "0.25"
    profit_same_financing_asset_weight: str = "0.20"
    profit_terminal_retention_weight: str = "0.20"
    profit_root_receiver_weight: str = "0.05"
    outflow_threshold: str = "0.5000"
    max_candidates_per_asset: int = 5
    ensure_top1_per_asset: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "outflow_scoring_version": "grouped_mean_v1",
            "outflow_feature_semantics_version": "runtime_signals_v1",
            "outflow_decimal_precision": 50,
            "outflow_rounding": "ROUND_HALF_UP",
            "outflow_lambda": str(self.outflow_lambda),
            "profit_weights": {
                "complete_financing": self.profit_complete_financing_weight,
                "after_last_repayment": self.profit_after_last_repayment_weight,
                "same_financing_asset": self.profit_same_financing_asset_weight,
                "terminal_retention": self.profit_terminal_retention_weight,
                "root_receiver": self.profit_root_receiver_weight,
            },
            "outflow_threshold": self.outflow_threshold,
            "max_candidates_per_asset": self.max_candidates_per_asset,
            "ensure_top1_per_asset": self.ensure_top1_per_asset,
        }


@dataclass(frozen=True)
class ResultDetectionResult:
    transaction_graph_sha256: str
    runtime_annotations_sha256: str
    observed_asset_ledger_sha256: str
    asset_continuity_sha256: str
    detection_config: ResultDetectionConfig
    result_type_coverage: dict[str, dict[str, str]]
    result_candidates: tuple[ResultCandidate, ...]

    SCHEMA_VERSION = "result_candidates_v1"

    def to_dict(self) -> dict[str, Any]:
        counts = {result_type: sum(item.result_type == result_type for item in self.result_candidates) for result_type in ("R_OUTFLOW", "R_PROFIT", "R_STATE", "R_FAILURE")}
        return {
            "schema_version": self.SCHEMA_VERSION,
            "transaction_graph_sha256": self.transaction_graph_sha256,
            "runtime_annotations_sha256": self.runtime_annotations_sha256,
            "observed_asset_ledger_sha256": self.observed_asset_ledger_sha256,
            "asset_continuity_sha256": self.asset_continuity_sha256,
            "detection_config": self.detection_config.to_dict(),
            "result_type_coverage": self.result_type_coverage,
            "result_candidates": [item.to_dict() for item in self.result_candidates],
            "metadata": {
                "r_outflow_count": counts["R_OUTFLOW"],
                "r_profit_count": counts["R_PROFIT"],
                "r_state_count": counts["R_STATE"],
                "r_failure_count": counts["R_FAILURE"],
                "top1_fallback_count": sum(item.selection_reason == "TOP1_FALLBACK" for item in self.result_candidates),
                "assets_with_candidates": len({_asset_key(item.asset_type, item.asset_address) for item in self.result_candidates}),
            },
        }


def _asset_key(asset_type: str, asset_address: dict[str, str] | None) -> tuple[str, str]:
    return asset_type, asset_address["normalized"] if asset_address is not None else "NATIVE"

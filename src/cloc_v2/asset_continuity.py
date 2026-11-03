"""FIFO candidate ASSET continuity, without source-level causal claims."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_continuity_models import AmountObservation, AssetContinuityEdge, AssetContinuityResult, SelfTransferObservation
from .asset_ledger_models import AddressFlowSummary, AssetTransferFact, ObservedAssetLedger
from .financing_episodes import detect_financing_episodes
from .graph_models import TransactionGraph
from .observed_asset_ledger import load_runtime_annotations, runtime_annotations_sha256
from .runtime_annotation_models import RuntimeAnnotations
from .runtime_semantics import load_transaction_graph, transaction_graph_sha256


@dataclass
class _InflowBatch:
    fact: AssetTransferFact
    remaining: int


def build_asset_continuity(
    transaction_graph: TransactionGraph,
    runtime_annotations: RuntimeAnnotations,
    observed_asset_ledger: ObservedAssetLedger,
) -> AssetContinuityResult:
    graph_hash = transaction_graph_sha256(transaction_graph)
    annotations_hash = runtime_annotations_sha256(runtime_annotations)
    ledger_hash = observed_asset_ledger_sha256(observed_asset_ledger)
    if runtime_annotations.transaction_graph_sha256 != graph_hash:
        raise ValueError("Runtime annotations are not bound to the supplied transaction graph")
    if observed_asset_ledger.transaction_graph_sha256 != graph_hash:
        raise ValueError("Observed asset ledger is not bound to the supplied transaction graph")
    if observed_asset_ledger.runtime_annotations_sha256 != annotations_hash:
        raise ValueError("Observed asset ledger is not bound to the supplied runtime annotations")

    edges, unmatched, remaining, self_transfers = _build_fifo_continuity(observed_asset_ledger.transfer_facts)
    episodes = detect_financing_episodes(transaction_graph, runtime_annotations, observed_asset_ledger)
    return AssetContinuityResult(
        transaction_graph_sha256=graph_hash,
        runtime_annotations_sha256=annotations_hash,
        observed_asset_ledger_sha256=ledger_hash,
        asset_edges=edges,
        financing_episodes=tuple(episodes),
        unmatched_outflows=unmatched,
        remaining_inflows=remaining,
        self_transfers=self_transfers,
    )


def load_observed_asset_ledger(path: str | Path) -> ObservedAssetLedger:
    ledger_path = Path(path)
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Observed asset ledger does not exist: {ledger_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Observed asset ledger is not valid JSON: {ledger_path} ({exc.msg})") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != ObservedAssetLedger.SCHEMA_VERSION:
        raise ValueError("Input schema_version must be observed_asset_ledger_v1")
    facts = tuple(_fact_from_dict(item) for item in payload.get("transfer_facts", []))
    summaries = tuple(_summary_from_dict(item) for item in payload.get("address_flow_summaries", []))
    return ObservedAssetLedger(
        transaction_graph_sha256=payload["transaction_graph_sha256"],
        runtime_annotations_sha256=payload["runtime_annotations_sha256"],
        transfer_facts=facts,
        address_flow_summaries=summaries,
    )


def serialize_asset_continuity(result: AssetContinuityResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"


def write_asset_continuity(result: AssetContinuityResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_asset_continuity(result), encoding="utf-8")
    return path


def observed_asset_ledger_sha256(ledger: ObservedAssetLedger) -> str:
    return hashlib.sha256(json.dumps(ledger.to_dict(), indent=2, ensure_ascii=False).encode("utf-8") + b"\n").hexdigest()


def _build_fifo_continuity(facts: tuple[AssetTransferFact, ...]) -> tuple[tuple[AssetContinuityEdge, ...], tuple[AmountObservation, ...], tuple[AmountObservation, ...], tuple[SelfTransferObservation, ...]]:
    queues: dict[tuple[str, str, str], list[_InflowBatch]] = {}
    raw_edges: list[AssetContinuityEdge] = []
    unmatched: list[AmountObservation] = []
    self_transfers: list[SelfTransferObservation] = []
    for fact in sorted(facts, key=lambda item: (item.trace_index, item.fact_id)):
        if fact.status != "COMPLETE" or fact.amount_raw is None or fact.from_address is None or fact.to_address is None:
            continue
        if fact.from_address["normalized"] == fact.to_address["normalized"]:
            self_transfers.append(_self_transfer(fact))
            continue
        amount = int(fact.amount_raw)
        out_key = _holder_key(fact, fact.from_address)
        allocations: list[tuple[_InflowBatch, int]] = []
        remaining_outflow = amount
        for batch in queues.get(out_key, []):
            if remaining_outflow == 0:
                break
            allocated = min(batch.remaining, remaining_outflow)
            if allocated:
                allocations.append((batch, allocated))
                batch.remaining -= allocated
                remaining_outflow -= allocated
        if allocations:
            raw_edges.extend(_edges_for_allocations(fact, allocations, amount))
        if remaining_outflow:
            unmatched.append(_amount_observation(fact, fact.from_address, remaining_outflow, "outflow amount has no prior observed inflow for this asset and holder"))
        in_key = _holder_key(fact, fact.to_address)
        queues.setdefault(in_key, []).append(_InflowBatch(fact=fact, remaining=amount))

    remaining: list[AmountObservation] = []
    for batches in queues.values():
        for batch in batches:
            if batch.remaining:
                remaining.append(_amount_observation(batch.fact, batch.fact.to_address, batch.remaining, "inflow amount has no later observed outflow use for this asset and holder"))
    ordered_edges = sorted(raw_edges, key=lambda item: (item.src_node_id, item.dst_node_id, item.src_fact_id, item.dst_fact_id))
    edges = tuple(_with_edge_id(item, index) for index, item in enumerate(ordered_edges, start=1))
    return edges, tuple(sorted(unmatched, key=lambda item: (item.trace_index, item.fact_id))), tuple(sorted(remaining, key=lambda item: (item.trace_index, item.fact_id))), tuple(sorted(self_transfers, key=lambda item: (item.trace_index, item.fact_id)))


def _edges_for_allocations(outflow: AssetTransferFact, allocations: list[tuple[_InflowBatch, int]], outflow_amount: int) -> list[AssetContinuityEdge]:
    result: list[AssetContinuityEdge] = []
    for batch, amount in allocations:
        if len(allocations) > 1:
            match_type = "MERGED"
        elif amount == outflow_amount:
            match_type = "EXACT"
        else:
            match_type = "PARTIAL"
        result.append(
            AssetContinuityEdge(
                edge_id="",
                src_node_id=batch.fact.node_id,
                dst_node_id=outflow.node_id,
                src_fact_id=batch.fact.fact_id,
                dst_fact_id=outflow.fact_id,
                asset_type=outflow.asset_type,
                asset_address=outflow.asset_address,
                via_address=outflow.from_address,
                allocated_amount_raw=str(amount),
                match_type=match_type,
                evidence_level="STRONG_ASSOCIATION",
                reason="same asset was received and later transferred by the same address",
                trace_refs=(batch.fact.trace_path, outflow.trace_path),
                matching_basis=("SAME_ASSET", "RECEIVER_EQUALS_LATER_SENDER", "TEMPORAL_PRECEDENCE", "AVAILABLE_OBSERVED_AMOUNT"),
                uncertainties=("observed transfer calls may not equal actual balance changes",),
            )
        )
    return result


def _with_edge_id(edge: AssetContinuityEdge, index: int) -> AssetContinuityEdge:
    return AssetContinuityEdge(edge_id=f"asset_e{index:03d}", src_node_id=edge.src_node_id, dst_node_id=edge.dst_node_id, src_fact_id=edge.src_fact_id, dst_fact_id=edge.dst_fact_id, asset_type=edge.asset_type, asset_address=edge.asset_address, via_address=edge.via_address, allocated_amount_raw=edge.allocated_amount_raw, match_type=edge.match_type, evidence_level=edge.evidence_level, reason=edge.reason, trace_refs=edge.trace_refs, matching_basis=edge.matching_basis, uncertainties=edge.uncertainties)


def _holder_key(fact: AssetTransferFact, holder: dict[str, str]) -> tuple[str, str, str]:
    return fact.asset_type, fact.asset_address["normalized"] if fact.asset_address is not None else "NATIVE", holder["normalized"]


def _amount_observation(fact: AssetTransferFact, holder: dict[str, str], amount: int, reason: str) -> AmountObservation:
    return AmountObservation(fact_id=fact.fact_id, node_id=fact.node_id, trace_index=fact.trace_index, trace_path=fact.trace_path, asset_type=fact.asset_type, asset_address=fact.asset_address, holder_address=holder, amount_raw=str(amount), reason=reason)


def _self_transfer(fact: AssetTransferFact) -> SelfTransferObservation:
    return SelfTransferObservation(fact_id=fact.fact_id, node_id=fact.node_id, trace_index=fact.trace_index, asset_type=fact.asset_type, asset_address=fact.asset_address, holder_address=fact.from_address, amount_raw=fact.amount_raw or "0")


def _fact_from_dict(item: dict[str, Any]) -> AssetTransferFact:
    return AssetTransferFact(fact_id=item["fact_id"], node_id=item["node_id"], trace_index=item["trace_index"], trace_path=item["trace_path"], transfer_kind=item["transfer_kind"], asset_type=item["asset_type"], asset_address=item.get("asset_address"), operator_address=item.get("operator_address"), from_address=item.get("from_address"), to_address=item.get("to_address"), amount_raw=item.get("amount_raw"), status=item["status"], evidence_level=item["evidence_level"], reason=item["reason"], trace_refs=tuple(item.get("trace_refs", [])), uncertainties=tuple(item.get("uncertainties", [])))


def _summary_from_dict(item: dict[str, Any]) -> AddressFlowSummary:
    return AddressFlowSummary(asset_type=item["asset_type"], asset_address=item.get("asset_address"), address=item["address"], incoming_transfer_count=item["incoming_transfer_count"], outgoing_transfer_count=item["outgoing_transfer_count"], incoming_amount_raw=item["incoming_amount_raw"], outgoing_amount_raw=item["outgoing_amount_raw"], observed_delta_raw=item["observed_delta_raw"], completeness=item.get("completeness", "PARTIAL"))

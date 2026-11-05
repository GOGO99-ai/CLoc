"""Select observable asset outcomes for backward tracing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Any, Mapping

from cloc.evidence.continuity import load_observed_asset_ledger, observed_asset_ledger_sha256
from cloc.evidence.transfer_models import AssetTransferFact, ObservedAssetLedger
from cloc.evidence.models import TransactionGraph
from cloc.evidence.ledger import load_runtime_annotations, runtime_annotations_sha256
from cloc.evidence.outcome_models import ResultCandidate, ResultDetectionConfig, ResultDetectionResult, ResultSignal
from cloc.evidence.annotations import RuntimeAnnotations
from cloc.evidence.semantics import load_transaction_graph, transaction_graph_sha256
from cloc.evidence.outflow_scoring import SCORING_CONTEXT, grouped_outflow_score, unit_decimal


DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")
FOUR_PLACES = Decimal("0.0001")
RESULT_TYPE_COVERAGE = {
    "R_OUTFLOW": {"status": "SUPPORTED", "reason": "complete asset transfer facts are available"},
    "R_PROFIT": {"status": "PARTIALLY_SUPPORTED", "reason": "requires complete financing episodes"},
    "R_STATE": {"status": "UNSUPPORTED", "reason": "no storage diff or explicit state-change trace evidence"},
    "R_FAILURE": {"status": "UNSUPPORTED", "reason": "trace parser does not preserve revert or error evidence"},
}


def detect_result_candidates(
    transaction_graph: TransactionGraph,
    runtime_annotations: RuntimeAnnotations,
    observed_asset_ledger: ObservedAssetLedger,
    asset_continuity: Mapping[str, Any],
    config: ResultDetectionConfig | None = None,
) -> ResultDetectionResult:
    """Return deterministic R_OUTFLOW/R_PROFIT candidates from supplied facts."""
    effective_config = config or ResultDetectionConfig()
    _validate_config(effective_config)
    graph_hash, annotations_hash, ledger_hash, continuity_hash = _validate_bindings(
        transaction_graph, runtime_annotations, observed_asset_ledger, asset_continuity
    )
    nodes = {node.node_id: node for node in transaction_graph.nodes}
    root = nodes[transaction_graph.root_id]
    facts = tuple(observed_asset_ledger.transfer_facts)
    fact_by_id = {fact.fact_id: fact for fact in facts}
    remaining_by_fact = _remaining_amounts(asset_continuity)
    episodes = tuple(asset_continuity.get("financing_episodes", ()))
    funding_fact_ids = {fact_id for episode in episodes for fact_id in episode.get("funding_fact_ids", ())}
    complete_repayment_fact_ids = {
        fact_id
        for episode in episodes
        if episode.get("status") == "COMPLETE"
        for fact_id in episode.get("repayment_fact_ids", ())
    }
    all_repayment_fact_ids = {fact_id for episode in episodes for fact_id in episode.get("repayment_fact_ids", ())}
    self_transfer_fact_ids = {item.get("fact_id") for item in asset_continuity.get("self_transfers", ())}
    eligible = [
        fact
        for fact in facts
        if _eligible_outflow_fact(fact, funding_fact_ids, complete_repayment_fact_ids, self_transfer_fact_ids)
    ]
    source_callees = {node.callee_address.lower() for node in transaction_graph.nodes if isinstance(node.callee_address, str)}
    node_actions = {item.node_id: set(item.action_types) for item in runtime_annotations.node_annotations}
    swap_trace_paths = tuple(
        nodes[node_id].trace_path
        for node_id, actions in node_actions.items()
        if "SWAP" in actions and node_id in nodes
    )
    unmatched_assets = {_asset_key_from_observation(item) for item in asset_continuity.get("unmatched_outflows", ())}
    outflows = _select_outflows(
        eligible,
        transaction_graph,
        root.callee_address,
        source_callees,
        remaining_by_fact,
        node_actions,
        swap_trace_paths,
        facts,
        episodes,
        unmatched_assets,
        effective_config,
    )
    profits = _detect_profits(
        eligible,
        facts,
        episodes,
        all_repayment_fact_ids,
        funding_fact_ids,
        root.callee_address,
        remaining_by_fact,
        node_actions,
        swap_trace_paths,
        unmatched_assets,
        transaction_graph,
        effective_config,
    )
    ordered = sorted(_unique_node_types(outflows + profits), key=lambda item: (item.result_type, -Decimal(item.confidence), item.trace_index, item.node_id))
    numbered = tuple(_with_result_id(candidate, index) for index, candidate in enumerate(ordered, start=1))
    return ResultDetectionResult(
        transaction_graph_sha256=graph_hash,
        runtime_annotations_sha256=annotations_hash,
        observed_asset_ledger_sha256=ledger_hash,
        asset_continuity_sha256=continuity_hash,
        detection_config=effective_config,
        result_type_coverage={key: dict(value) for key, value in RESULT_TYPE_COVERAGE.items()},
        result_candidates=numbered,
    )


def load_asset_continuity(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Asset continuity artifact does not exist: {artifact_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Asset continuity artifact is not valid JSON: {artifact_path} ({exc.msg})") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "asset_continuity_v1":
        raise ValueError("Input schema_version must be asset_continuity_v1")
    required = ("transaction_graph_sha256", "runtime_annotations_sha256", "observed_asset_ledger_sha256", "asset_edges", "financing_episodes", "remaining_inflows")
    if any(key not in payload for key in required):
        raise ValueError("Invalid asset_continuity_v1 structure")
    return payload


def asset_continuity_sha256(asset_continuity: Mapping[str, Any]) -> str:
    """Hash normalized JSON content, independent of source-file whitespace."""
    normalized = json.dumps(asset_continuity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def serialize_result_detection(result: ResultDetectionResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def write_result_detection(result: ResultDetectionResult, output_path: str | Path) -> Path:
    """Atomically publish the independent artifact after detection has succeeded."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_result_detection(result)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(payload)
        os.replace(handle.name, target)
    except Exception:
        try:
            Path(handle.name).unlink(missing_ok=True)
        finally:
            raise
    return target


def _validate_bindings(
    transaction_graph: TransactionGraph,
    runtime_annotations: RuntimeAnnotations,
    observed_asset_ledger: ObservedAssetLedger,
    asset_continuity: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    if not isinstance(asset_continuity, Mapping) or asset_continuity.get("schema_version") != "asset_continuity_v1":
        raise ValueError("Input schema_version must be asset_continuity_v1")
    graph_hash = transaction_graph_sha256(transaction_graph)
    annotations_hash = runtime_annotations_sha256(runtime_annotations)
    ledger_hash = observed_asset_ledger_sha256(observed_asset_ledger)
    if runtime_annotations.transaction_graph_sha256 != graph_hash:
        raise ValueError("Runtime annotations are not bound to the supplied transaction graph")
    if observed_asset_ledger.transaction_graph_sha256 != graph_hash:
        raise ValueError("Observed asset ledger is not bound to the supplied transaction graph")
    if observed_asset_ledger.runtime_annotations_sha256 != annotations_hash:
        raise ValueError("Observed asset ledger is not bound to the supplied runtime annotations")
    if asset_continuity.get("transaction_graph_sha256") != graph_hash:
        raise ValueError("Asset continuity is not bound to the supplied transaction graph")
    if asset_continuity.get("runtime_annotations_sha256") != annotations_hash:
        raise ValueError("Asset continuity is not bound to the supplied runtime annotations")
    if asset_continuity.get("observed_asset_ledger_sha256") != ledger_hash:
        raise ValueError("Asset continuity is not bound to the supplied observed asset ledger")
    return graph_hash, annotations_hash, ledger_hash, asset_continuity_sha256(asset_continuity)


def _validate_config(config: ResultDetectionConfig) -> None:
    weights = (
        config.profit_complete_financing_weight,
        config.profit_after_last_repayment_weight,
        config.profit_same_financing_asset_weight,
        config.profit_terminal_retention_weight,
        config.profit_root_receiver_weight,
    )
    unit_decimal(config.outflow_lambda, "outflow_lambda")
    parsed = [unit_decimal(item, "profit weight") for item in weights]
    if sum(parsed) != DECIMAL_ONE:
        raise ValueError("Detection weights must be non-negative and each formula must sum to 1")
    unit_decimal(config.outflow_threshold, "outflow_threshold")
    if config.max_candidates_per_asset < 1:
        raise ValueError("max_candidates_per_asset must be positive")


def _select_outflows(*args, **kwargs) -> list[ResultCandidate]:
    # Fix feature division and grouped arithmetic precision without changing
    # the independent R_PROFIT calculation's existing numerical contract.
    with localcontext(SCORING_CONTEXT):
        return _select_outflows_at_precision(*args, **kwargs)


def _select_outflows_at_precision(
    facts: list[AssetTransferFact],
    transaction_graph: TransactionGraph,
    root_address: str | None,
    source_callees: set[str],
    remaining_by_fact: dict[str, int],
    node_actions: dict[str, set[str]],
    swap_trace_paths: tuple[str, ...],
    all_facts: tuple[AssetTransferFact, ...],
    episodes: tuple[Mapping[str, Any], ...],
    unmatched_assets: set[tuple[str, str]],
    config: ResultDetectionConfig,
) -> list[ResultCandidate]:
    grouped: dict[tuple[str, str], list[AssetTransferFact]] = defaultdict(list)
    for fact in facts:
        grouped[_asset_key(fact)].append(fact)
    selected: list[ResultCandidate] = []
    threshold = Decimal(config.outflow_threshold)
    for asset_key in sorted(grouped):
        group = grouped[asset_key]
        magnitudes = _magnitude_scores(group)
        scored = [
            _outflow_candidate(
                fact,
                transaction_graph,
                root_address,
                source_callees,
                remaining_by_fact,
                magnitudes[fact.fact_id],
                node_actions,
                swap_trace_paths,
                all_facts,
                episodes,
                unmatched_assets,
                config,
            )
            for fact in group
        ]
        ordered = sorted(scored, key=lambda item: (-Decimal(item.confidence), item.trace_index, item.node_id, item.source_fact_ids))
        kept = [item for item in ordered if Decimal(item.confidence) >= threshold][: config.max_candidates_per_asset]
        if not kept and config.ensure_top1_per_asset and ordered:
            kept = [_with_selection_reason(ordered[0], "TOP1_FALLBACK")]
        selected.extend(kept)
    return selected


def _outflow_candidate(
    fact: AssetTransferFact,
    transaction_graph: TransactionGraph,
    root_address: str | None,
    source_callees: set[str],
    remaining_by_fact: dict[str, int],
    magnitude: Decimal,
    node_actions: dict[str, set[str]],
    swap_trace_paths: tuple[str, ...],
    all_facts: tuple[AssetTransferFact, ...],
    episodes: tuple[Mapping[str, Any], ...],
    unmatched_assets: set[tuple[str, str]],
    config: ResultDetectionConfig,
) -> ResultCandidate:
    retention = _retention(fact, remaining_by_fact)
    late = Decimal(fact.trace_index) / Decimal(len(transaction_graph.nodes))
    root_receiver = _same_address(fact.to_address, root_address)
    source_activity = fact.from_address["normalized"] in source_callees
    confidence, score_audit = grouped_outflow_score(
        (magnitude, retention, late, Decimal(int(root_receiver)), Decimal(int(source_activity))),
        config.outflow_lambda,
    )
    signals = (
        _signal("MAGNITUDE_RANK", magnitude, "amount rank within the same asset only", fact.trace_path),
        _signal("TERMINAL_RETENTION", retention, "received amount is not fully consumed by later observed outflows", fact.trace_path),
        _signal("LATE_EXECUTION", late, "relative execution position only", fact.trace_path),
        _signal("ROOT_RECEIVER", Decimal(int(root_receiver)), "receiver equals the transaction graph root callee", fact.trace_path),
        _signal("SOURCE_ACTIVITY", Decimal(int(source_activity)), "source address appears as a transaction graph callee", fact.trace_path),
    )
    negative, uncertainties = _context_notes(fact, retention, node_actions, swap_trace_paths, all_facts, episodes, unmatched_assets)
    return ResultCandidate(
        result_id="",
        node_id=fact.node_id,
        trace_index=fact.trace_index,
        result_type="R_OUTFLOW",
        confidence=_decimal_text(confidence),
        evidence_level="STRONG_ASSOCIATION",
        asset_type=fact.asset_type,
        asset_address=fact.asset_address,
        from_address=fact.from_address or {},
        to_address=fact.to_address or {},
        amount_raw=fact.amount_raw or "0",
        source_fact_ids=(fact.fact_id,),
        signals=signals,
        negative_evidence=negative,
        selection_reason="SCORE_THRESHOLD",
        uncertainties=uncertainties,
        outflow_score_audit=score_audit,
    )


def _detect_profits(
    facts: list[AssetTransferFact],
    all_facts: tuple[AssetTransferFact, ...],
    episodes: tuple[Mapping[str, Any], ...],
    all_repayment_fact_ids: set[str],
    funding_fact_ids: set[str],
    root_address: str | None,
    remaining_by_fact: dict[str, int],
    node_actions: dict[str, set[str]],
    swap_trace_paths: tuple[str, ...],
    unmatched_assets: set[tuple[str, str]],
    transaction_graph: TransactionGraph,
    config: ResultDetectionConfig,
) -> list[ResultCandidate]:
    fact_by_id = {fact.fact_id: fact for fact in all_facts}
    episodes_by_asset: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        funding_ids = episode.get("funding_fact_ids", ())
        first_funding = fact_by_id.get(funding_ids[0]) if funding_ids else None
        if first_funding is not None:
            declared_address = episode.get("asset_address")
            declared_key = declared_address.get("normalized") if isinstance(declared_address, Mapping) else "NATIVE"
            episodes_by_asset[(first_funding.asset_type, declared_key)].append(episode)
    candidates: list[ResultCandidate] = []
    for fact in facts:
        if fact.fact_id in funding_fact_ids or fact.fact_id in all_repayment_fact_ids:
            continue
        retention = _retention(fact, remaining_by_fact)
        relevant = episodes_by_asset.get(_asset_key(fact), [])
        if not relevant or retention <= DECIMAL_ZERO or any(episode.get("status") != "COMPLETE" for episode in relevant):
            continue
        repayment_indexes = [fact_by_id[fact_id].trace_index for episode in relevant for fact_id in episode.get("repayment_fact_ids", ()) if fact_id in fact_by_id]
        if not repayment_indexes or fact.trace_index <= max(repayment_indexes):
            continue
        root_receiver = _same_address(fact.to_address, root_address)
        confidence = (
            Decimal(config.profit_complete_financing_weight)
            + Decimal(config.profit_after_last_repayment_weight)
            + Decimal(config.profit_same_financing_asset_weight)
            + Decimal(config.profit_terminal_retention_weight) * retention
            + Decimal(config.profit_root_receiver_weight) * Decimal(int(root_receiver))
        )
        signals = (
            _signal("COMPLETE_FINANCING", DECIMAL_ONE, "all relevant financing episodes are complete", fact.trace_path),
            _signal("AFTER_LAST_REPAYMENT", DECIMAL_ONE, "transfer occurs after all relevant observed repayments", fact.trace_path),
            _signal("SAME_FINANCING_ASSET", DECIMAL_ONE, "transfer asset matches the complete financing asset", fact.trace_path),
            _signal("TERMINAL_RETENTION", retention, "received amount is not fully consumed by later observed outflows", fact.trace_path),
            _signal("ROOT_RECEIVER", Decimal(int(root_receiver)), "receiver equals the transaction graph root callee", fact.trace_path),
        )
        negative, uncertainties = _context_notes(fact, retention, node_actions, swap_trace_paths, all_facts, episodes, unmatched_assets)
        candidates.append(
            ResultCandidate(
                result_id="",
                node_id=fact.node_id,
                trace_index=fact.trace_index,
                result_type="R_PROFIT",
                confidence=_decimal_text(confidence),
                evidence_level="STRONG_ASSOCIATION",
                asset_type=fact.asset_type,
                asset_address=fact.asset_address,
                from_address=fact.from_address or {},
                to_address=fact.to_address or {},
                amount_raw=fact.amount_raw or "0",
                source_fact_ids=(fact.fact_id,),
                signals=signals,
                negative_evidence=negative,
                selection_reason="STRICT_FINANCING_EVIDENCE",
                uncertainties=uncertainties,
            )
        )
    return candidates


def _eligible_outflow_fact(fact: AssetTransferFact, funding_fact_ids: set[str], complete_repayment_fact_ids: set[str], self_transfer_fact_ids: set[str | None]) -> bool:
    return (
        fact.status == "COMPLETE"
        and fact.amount_raw is not None
        and int(fact.amount_raw) > 0
        and fact.from_address is not None
        and fact.to_address is not None
        and fact.from_address["normalized"] != fact.to_address["normalized"]
        and fact.fact_id not in self_transfer_fact_ids
        and fact.fact_id not in funding_fact_ids
        and fact.fact_id not in complete_repayment_fact_ids
    )


def _magnitude_scores(facts: list[AssetTransferFact]) -> dict[str, Decimal]:
    amounts = [int(fact.amount_raw or "0") for fact in facts]
    minimum, maximum = min(amounts), max(amounts)
    if minimum == maximum:
        return {fact.fact_id: DECIMAL_ONE for fact in facts}
    span = Decimal(maximum - minimum)
    return {fact.fact_id: Decimal(int(fact.amount_raw or "0") - minimum) / span for fact in facts}


def _remaining_amounts(asset_continuity: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for item in asset_continuity.get("remaining_inflows", ()):
        fact_id, amount = item.get("fact_id"), item.get("amount_raw")
        if isinstance(fact_id, str) and isinstance(amount, str) and amount.isdigit():
            result[fact_id] += int(amount)
    return result


def _retention(fact: AssetTransferFact, remaining_by_fact: Mapping[str, int]) -> Decimal:
    amount = int(fact.amount_raw or "0")
    if amount <= 0:
        return DECIMAL_ZERO
    retained = max(0, min(amount, remaining_by_fact.get(fact.fact_id, 0)))
    return Decimal(retained) / Decimal(amount)


def _context_notes(
    fact: AssetTransferFact,
    retention: Decimal,
    node_actions: Mapping[str, set[str]],
    swap_trace_paths: tuple[str, ...],
    all_facts: tuple[AssetTransferFact, ...],
    episodes: tuple[Mapping[str, Any], ...],
    unmatched_assets: set[tuple[str, str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    negative: list[str] = []
    if retention < Decimal("0.5"):
        negative.append("MOSTLY_CONSUMED_BY_LATER_OBSERVED_OUTFLOWS")
    if "SWAP" in node_actions.get(fact.node_id, set()) or any(fact.trace_path.startswith(path + "/") for path in swap_trace_paths):
        negative.append("WITHIN_SWAP_SUBTREE")
    if any(
        later.trace_index > fact.trace_index
        and _asset_key(later) == _asset_key(fact)
        and later.from_address is not None
        and fact.to_address is not None
        and later.from_address["normalized"] == fact.to_address["normalized"]
        for later in all_facts
    ):
        negative.append("RECEIVER_LATER_SAME_ASSET_TRANSFER")
    if _asset_key(fact) in unmatched_assets:
        negative.append("ASSET_HAS_UNMATCHED_OBSERVED_OUTFLOWS")
    if any(_episode_asset_key(episode) == _asset_key(fact) and episode.get("status") in {"FUNDING_ONLY", "AMBIGUOUS"} for episode in episodes):
        negative.append("INCOMPLETE_FINANCING_EPISODE_PRESENT")
    # Candidate uncertainty is evidence-specific.  Ledger coverage limits,
    # token decimals, and real balance changes describe the method boundary;
    # they are not missing fields on an otherwise complete transfer fact.
    uncertainties = list(fact.uncertainties)
    if any("derived" in item.lower() for item in fact.uncertainties):
        negative.append("DERIVED_ADDRESS_CONTEXT")
    return tuple(negative), tuple(uncertainties)


def _episode_asset_key(episode: Mapping[str, Any]) -> tuple[str, str]:
    address = episode.get("asset_address")
    return "TOKEN", address.get("normalized") if isinstance(address, Mapping) else "NATIVE"


def _signal(signal_type: str, value: Decimal, reason: str, trace_path: str) -> ResultSignal:
    return ResultSignal(signal_type=signal_type, value=_decimal_text(value), reason=reason, trace_refs=(trace_path,))


def _decimal_text(value: Decimal) -> str:
    bounded = min(DECIMAL_ONE, max(DECIMAL_ZERO, value))
    return format(bounded.quantize(FOUR_PLACES, rounding=ROUND_HALF_UP), "f")


def _same_address(address: dict[str, str] | None, raw_address: str | None) -> bool:
    return address is not None and isinstance(raw_address, str) and address["normalized"] == raw_address.lower()


def _asset_key(fact: AssetTransferFact) -> tuple[str, str]:
    return fact.asset_type, fact.asset_address["normalized"] if fact.asset_address is not None else "NATIVE"


def _asset_key_from_observation(item: Mapping[str, Any]) -> tuple[str, str]:
    address = item.get("asset_address")
    return item.get("asset_type", ""), address.get("normalized") if isinstance(address, Mapping) else "NATIVE"


def _with_selection_reason(candidate: ResultCandidate, reason: str) -> ResultCandidate:
    return ResultCandidate(**{**candidate.__dict__, "selection_reason": reason})


def _with_result_id(candidate: ResultCandidate, index: int) -> ResultCandidate:
    return ResultCandidate(**{**candidate.__dict__, "result_id": f"result_r{index:03d}"})


def _unique_node_types(candidates: list[ResultCandidate]) -> list[ResultCandidate]:
    """Keep one deterministic best candidate for each node and result type."""
    ordered = sorted(candidates, key=lambda item: (item.result_type, -Decimal(item.confidence), item.trace_index, item.node_id, item.source_fact_ids))
    selected: list[ResultCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in ordered:
        key = candidate.node_id, candidate.result_type
        if key not in seen:
            seen.add(key)
            selected.append(candidate)
    return selected


__all__ = [
    "asset_continuity_sha256",
    "detect_result_candidates",
    "load_asset_continuity",
    "load_observed_asset_ledger",
    "load_runtime_annotations",
    "load_transaction_graph",
    "serialize_result_detection",
    "write_result_detection",
]

"""Collect observed native and token transfers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from cloc.evidence.transfer_models import AddressFlowSummary, AssetTransferFact, ObservedAssetLedger
from cloc.evidence.models import TransactionGraph
from cloc.evidence.annotations import AnnotationEvidence, NodeRuntimeAnnotation, RuntimeAnnotations
from cloc.evidence.semantics import load_transaction_graph, serialize_runtime_annotations, transaction_graph_sha256


DECIMAL_INTEGER_RE = re.compile(r"^\d+$")
HEX_INTEGER_RE = re.compile(r"^0x[0-9a-fA-F]+$")
ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def build_observed_asset_ledger(
    transaction_graph: TransactionGraph,
    runtime_annotations: RuntimeAnnotations,
) -> ObservedAssetLedger:
    """Return observed transfer facts, never inferred continuity or causality."""
    graph_hash = transaction_graph_sha256(transaction_graph)
    if runtime_annotations.transaction_graph_sha256 != graph_hash:
        raise ValueError(
            "Runtime annotation transaction_graph_sha256 does not match the supplied transaction_graph_v1"
        )

    annotations_by_id = {annotation.node_id: annotation for annotation in runtime_annotations.node_annotations}
    raw_facts: list[AssetTransferFact] = []
    for node in transaction_graph.nodes:
        annotation = annotations_by_id.get(node.node_id)
        if annotation is not None:
            token_fact = _token_fact_for_node(node, annotation)
            if token_fact is not None:
                raw_facts.append(token_fact)
        native_fact = _native_value_fact_for_node(node)
        if native_fact is not None:
            raw_facts.append(native_fact)

    _validate_no_conflicting_token_facts(raw_facts)
    ordered = sorted(raw_facts, key=lambda fact: (fact.trace_index, fact.node_id, _transfer_kind_order(fact.transfer_kind)))
    facts = tuple(
        AssetTransferFact(
            fact_id=f"asset_f{index:03d}",
            node_id=fact.node_id,
            trace_index=fact.trace_index,
            trace_path=fact.trace_path,
            transfer_kind=fact.transfer_kind,
            asset_type=fact.asset_type,
            asset_address=fact.asset_address,
            operator_address=fact.operator_address,
            from_address=fact.from_address,
            to_address=fact.to_address,
            amount_raw=fact.amount_raw,
            status=fact.status,
            evidence_level=fact.evidence_level,
            reason=fact.reason,
            trace_refs=fact.trace_refs,
            uncertainties=fact.uncertainties,
        )
        for index, fact in enumerate(ordered, start=1)
    )
    return ObservedAssetLedger(
        transaction_graph_sha256=graph_hash,
        runtime_annotations_sha256=runtime_annotations_sha256(runtime_annotations),
        transfer_facts=facts,
        address_flow_summaries=_build_address_flow_summaries(facts),
    )


def load_runtime_annotations(path: str | Path) -> RuntimeAnnotations:
    """Load a valid standalone runtime_annotations_v1 artifact."""
    annotation_path = Path(path)
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Runtime annotations do not exist: {annotation_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Runtime annotations are not valid JSON: {annotation_path} ({exc.msg})") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != RuntimeAnnotations.SCHEMA_VERSION:
        raise ValueError("Input schema_version must be runtime_annotations_v1")
    try:
        annotations = tuple(_annotation_from_dict(item) for item in payload["node_annotations"])
        profiles = tuple(_profile_from_dict(item) for item in payload.get("address_profiles", []))
        return RuntimeAnnotations(
            transaction_graph_sha256=payload["transaction_graph_sha256"],
            node_annotations=annotations,
            address_profiles=profiles,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid runtime_annotations_v1 structure: {exc}") from exc


def serialize_observed_asset_ledger(ledger: ObservedAssetLedger) -> str:
    return json.dumps(ledger.to_dict(), indent=2, ensure_ascii=False) + "\n"


def write_observed_asset_ledger(ledger: ObservedAssetLedger, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_observed_asset_ledger(ledger), encoding="utf-8")
    return path


def runtime_annotations_sha256(runtime_annotations: RuntimeAnnotations) -> str:
    return hashlib.sha256(serialize_runtime_annotations(runtime_annotations).encode("utf-8")).hexdigest()


def _token_fact_for_node(node: Any, annotation: NodeRuntimeAnnotation) -> AssetTransferFact | None:
    action_types = set(annotation.action_types)
    if "TOKEN_TRANSFER_FROM" in action_types:
        transfer_kind = "TRANSFER_FROM"
    elif "TOKEN_TRANSFER" in action_types:
        transfer_kind = "TRANSFER"
    else:
        return None

    operands = annotation.observed_operands
    asset_address = _address_or_none(operands.get("token_address")) or _address_or_none(node.callee_address)
    operator_address = _address_or_none(node.caller_address)
    if transfer_kind == "TRANSFER_FROM":
        from_address = _address_or_none(operands.get("from_address"))
        reason = "standard ERC-20 transferFrom call"
    else:
        from_address = _address_or_none(node.caller_address)
        reason = "standard ERC-20 transfer call"
    to_address = _address_or_none(operands.get("to_address"))
    amount_raw = _decimal_amount_or_none(operands.get("amount_raw"))

    uncertainties = _transfer_uncertainties(
        node=node,
        transfer_kind=transfer_kind,
        asset_address=asset_address,
        operator_address=operator_address,
        from_address=from_address,
        to_address=to_address,
        amount_raw=amount_raw,
    )
    status = "COMPLETE" if all((asset_address, from_address, to_address, amount_raw)) else "INCOMPLETE"
    return AssetTransferFact(
        fact_id="",
        node_id=node.node_id,
        trace_index=node.trace_index,
        trace_path=node.trace_path,
        transfer_kind=transfer_kind,
        asset_type="TOKEN",
        asset_address=asset_address,
        operator_address=operator_address,
        from_address=from_address,
        to_address=to_address,
        amount_raw=amount_raw,
        status=status,
        evidence_level="OBSERVED",
        reason=reason,
        trace_refs=(node.trace_path,),
        uncertainties=tuple(uncertainties),
    )


def _native_value_fact_for_node(node: Any) -> AssetTransferFact | None:
    amount_raw = _explicit_positive_value_or_none(node.value)
    if amount_raw is None:
        return None
    operator_address = _address_or_none(node.caller_address)
    from_address = _address_or_none(node.caller_address)
    to_address = _address_or_none(node.callee_address)
    uncertainties: list[str] = []
    if node.caller_source == "DERIVED_FROM_PARENT":
        uncertainties.append("from_address is derived from parent call context")
    if operator_address is None:
        uncertainties.append("operator_address is unavailable from transaction graph")
    if from_address is None:
        uncertainties.append("from_address is unavailable from transaction graph")
    if to_address is None:
        uncertainties.append("to_address is unavailable from transaction graph")
    status = "COMPLETE" if all((from_address, to_address, amount_raw)) else "INCOMPLETE"
    return AssetTransferFact(
        fact_id="",
        node_id=node.node_id,
        trace_index=node.trace_index,
        trace_path=node.trace_path,
        transfer_kind="NATIVE_VALUE",
        asset_type="NATIVE",
        asset_address=None,
        operator_address=operator_address,
        from_address=from_address,
        to_address=to_address,
        amount_raw=amount_raw,
        status=status,
        evidence_level="OBSERVED",
        reason="explicit positive value field on call node",
        trace_refs=(node.trace_path,),
        uncertainties=tuple(uncertainties),
    )


def _transfer_uncertainties(
    *,
    node: Any,
    transfer_kind: str,
    asset_address: dict[str, str] | None,
    operator_address: dict[str, str] | None,
    from_address: dict[str, str] | None,
    to_address: dict[str, str] | None,
    amount_raw: str | None,
) -> list[str]:
    uncertainties: list[str] = []
    if node.caller_source == "DERIVED_FROM_PARENT":
        if transfer_kind == "TRANSFER":
            uncertainties.append("from_address is derived from parent call context")
        else:
            uncertainties.append("operator_address is derived from parent call context")
    for field, value in (
        ("asset_address", asset_address),
        ("operator_address", operator_address),
        ("from_address", from_address),
        ("to_address", to_address),
        ("amount_raw", amount_raw),
    ):
        if value is None:
            uncertainties.append(f"{field} is unavailable from observed trace fields")
    return uncertainties


def _build_address_flow_summaries(facts: tuple[AssetTransferFact, ...]) -> tuple[AddressFlowSummary, ...]:
    totals: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fact in facts:
        if fact.status != "COMPLETE" or fact.amount_raw is None or fact.from_address is None or fact.to_address is None:
            continue
        asset_key = _asset_key(fact.asset_type, fact.asset_address)
        amount = int(fact.amount_raw)
        _add_flow(totals, asset_key, fact.asset_type, fact.asset_address, fact.from_address, amount, outgoing=True)
        _add_flow(totals, asset_key, fact.asset_type, fact.asset_address, fact.to_address, amount, outgoing=False)

    summaries: list[AddressFlowSummary] = []
    for _, entry in totals.items():
        summaries.append(
            AddressFlowSummary(
                asset_type=entry["asset_type"],
                asset_address=entry["asset_address"],
                address=entry["address"],
                incoming_transfer_count=entry["incoming_count"],
                outgoing_transfer_count=entry["outgoing_count"],
                incoming_amount_raw=str(entry["incoming_amount"]),
                outgoing_amount_raw=str(entry["outgoing_amount"]),
                observed_delta_raw=str(entry["incoming_amount"] - entry["outgoing_amount"]),
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda item: (
                _asset_key(item.asset_type, item.asset_address),
                item.address["normalized"],
            ),
        )
    )


def _add_flow(
    totals: dict[tuple[str, str, str], dict[str, Any]],
    asset_key: tuple[str, str],
    asset_type: str,
    asset_address: dict[str, str] | None,
    address: dict[str, str],
    amount: int,
    *,
    outgoing: bool,
) -> None:
    key = (*asset_key, address["normalized"])
    entry = totals.setdefault(
        key,
        {
            "asset_type": asset_type,
            "asset_address": asset_address,
            "address": address,
            "incoming_count": 0,
            "outgoing_count": 0,
            "incoming_amount": 0,
            "outgoing_amount": 0,
        },
    )
    if outgoing:
        entry["outgoing_count"] += 1
        entry["outgoing_amount"] += amount
    else:
        entry["incoming_count"] += 1
        entry["incoming_amount"] += amount


def _validate_no_conflicting_token_facts(facts: list[AssetTransferFact]) -> None:
    token_counts: dict[str, int] = {}
    keys: set[tuple[str, str]] = set()
    for fact in facts:
        if fact.asset_type != "TOKEN":
            continue
        token_counts[fact.node_id] = token_counts.get(fact.node_id, 0) + 1
        key = (fact.node_id, fact.transfer_kind)
        if key in keys:
            raise ValueError(f"Conflicting duplicate token fact for node {fact.node_id} and kind {fact.transfer_kind}")
        keys.add(key)
    conflicts = sorted(node_id for node_id, count in token_counts.items() if count > 1)
    if conflicts:
        raise ValueError(f"Conflicting token facts: nodes have more than one token transfer fact: {', '.join(conflicts)}")


def _transfer_kind_order(transfer_kind: str) -> int:
    return {"TRANSFER_FROM": 0, "TRANSFER": 1, "NATIVE_VALUE": 2}.get(transfer_kind, 99)


def _address_or_none(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict):
        original = value.get("original")
        normalized = value.get("normalized")
        if isinstance(original, str) and isinstance(normalized, str):
            return {"original": original, "normalized": normalized}
        return None
    if isinstance(value, str) and ADDRESS_RE.fullmatch(value):
        return {"original": value, "normalized": value.lower()}
    return None


def _decimal_amount_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and DECIMAL_INTEGER_RE.fullmatch(value) else None


def _explicit_positive_value_or_none(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value) if value > 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if DECIMAL_INTEGER_RE.fullmatch(text):
        return text if int(text) > 0 else None
    if HEX_INTEGER_RE.fullmatch(text):
        parsed = int(text, 16)
        return str(parsed) if parsed > 0 else None
    return None


def _asset_key(asset_type: str, asset_address: dict[str, str] | None) -> tuple[str, str]:
    if asset_type == "NATIVE":
        return "NATIVE", "NATIVE"
    return "TOKEN", asset_address["normalized"] if asset_address is not None else "UNKNOWN"


def _annotation_from_dict(payload: dict[str, Any]) -> NodeRuntimeAnnotation:
    evidence = tuple(
        AnnotationEvidence(
            action_type=item["action_type"],
            evidence_level=item["evidence_level"],
            rule_id=item["rule_id"],
            reason=item["reason"],
            trace_refs=tuple(item["trace_refs"]),
        )
        for item in payload.get("annotation_evidence", [])
    )
    return NodeRuntimeAnnotation(
        node_id=payload["node_id"],
        action_types=tuple(payload.get("action_types", [])),
        observed_operands=payload.get("observed_operands", {}),
        annotation_evidence=evidence,
        uncertainties=tuple(payload.get("uncertainties", [])),
    )


def _profile_from_dict(payload: dict[str, Any]) -> Any:
    from cloc.evidence.annotations import AddressProfile

    return AddressProfile(
        address=payload["address"],
        normalized_address=payload["normalized_address"],
        observed_functions=tuple(payload.get("observed_functions", [])),
        observed_capabilities=tuple(payload.get("observed_capabilities", [])),
        evidence_node_ids=tuple(payload.get("evidence_node_ids", [])),
    )

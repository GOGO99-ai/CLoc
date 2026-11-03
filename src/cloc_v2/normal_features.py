"""Pure translation from runtime evidence into normal-library chain features.

The extractor consumes only caller-supplied graph, strict-chain, ledger, and
offline identity evidence.  It does not read files, configuration, labels,
normal baselines, or external services.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .asset_ledger_models import AssetTransferFact, ObservedAssetLedger
from .function_identity import OfflineFunctionCandidate, resolve_function_identity
from .graph_models import NormalizedCallNode, TransactionGraph
from .normal_distance import AssetBehavior, CallStep, ChainFeatureRecord, MainAnchorKey
from .runtime_semantics import transaction_graph_sha256
from .strict_result_chains import MainAnchorCandidate, StrictResultChain, StrictResultChainsResult


_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_SELECTOR = re.compile(r"^0x[0-9a-fA-F]{8}$")
_ASSET_RESULT_TYPES = {"R_OUTFLOW", "R_PROFIT"}


@dataclass(frozen=True)
class FeatureExtractionIssue:
    """One machine-readable issue attached to a precise evidence object."""

    code: str
    source_kind: str
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class ChainFeatureExtractionStatus:
    """Extraction outcome for exactly one strict chain."""

    strict_chain_id: str
    record_id: str
    status: str
    record_created: bool
    reasons: tuple[str, ...]
    issues: tuple[FeatureExtractionIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strict_chain_id": self.strict_chain_id,
            "record_id": self.record_id,
            "status": self.status,
            "record_created": self.record_created,
            "reasons": list(self.reasons),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class NormalFeatureExtractionResult:
    """Stable, immutable feature records plus complete per-chain audit status."""

    transaction_record_id: str
    records: tuple[ChainFeatureRecord, ...]
    chain_statuses: tuple[ChainFeatureExtractionStatus, ...]
    input_chain_count: int
    record_count: int
    clean_record_count: int
    uncertain_record_count: int
    rejected_chain_count: int

    SCHEMA_VERSION = "normal_features_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "transaction_record_id": self.transaction_record_id,
            "records": [record.to_dict() for record in self.records],
            "chain_statuses": [status.to_dict() for status in self.chain_statuses],
            "metadata": {
                "input_chain_count": self.input_chain_count,
                "record_count": self.record_count,
                "clean_record_count": self.clean_record_count,
                "uncertain_record_count": self.uncertain_record_count,
                "rejected_chain_count": self.rejected_chain_count,
            },
        }


@dataclass(frozen=True)
class _EffectiveAddressMap:
    valid: Mapping[str, str]
    invalid_runtime_addresses: frozenset[str]
    global_issues: tuple[FeatureExtractionIssue, ...]


def extract_normal_features(
    transaction_graph: TransactionGraph,
    strict_result_chains: StrictResultChainsResult,
    observed_asset_ledger: ObservedAssetLedger,
    *,
    transaction_record_id: str,
    offline_function_candidates: Iterable[OfflineFunctionCandidate] = (),
    offline_effective_address_map: Mapping[str, str] | None = None,
) -> NormalFeatureExtractionResult:
    """Translate strict chains into prepared normal-distance feature records."""
    _validate_inputs(
        transaction_graph,
        strict_result_chains,
        observed_asset_ledger,
        transaction_record_id,
    )
    if observed_asset_ledger.transaction_graph_sha256 != transaction_graph_sha256(transaction_graph):
        raise ValueError("ObservedAssetLedger transaction_graph_sha256 does not match TransactionGraph")

    candidates = tuple(offline_function_candidates)
    if any(not isinstance(item, OfflineFunctionCandidate) for item in candidates):
        raise ValueError("offline_function_candidates must contain OfflineFunctionCandidate values")
    effective_addresses = _normalize_effective_address_map(offline_effective_address_map)
    nodes = _unique_by_id(transaction_graph.nodes, "node_id", "TransactionGraph node_id")
    edges = _unique_by_id(strict_result_chains.edges, "edge_id", "strict edge_id")
    facts = _unique_by_id(observed_asset_ledger.transfer_facts, "fact_id", "asset fact_id")
    chains = tuple(sorted(strict_result_chains.chains, key=lambda item: item.strict_chain_id))
    if len({chain.strict_chain_id for chain in chains}) != len(chains):
        raise ValueError("strict_chain_id values must be unique")

    records: list[ChainFeatureRecord] = []
    statuses: list[ChainFeatureExtractionStatus] = []
    for chain in chains:
        record_id = f"{transaction_record_id}:{chain.strict_chain_id}"
        anchor_key, anchor_issues, business_addresses = _main_anchor(
            chain.anchor,
            effective_addresses,
            nodes,
        )
        issues: list[FeatureExtractionIssue] = [*effective_addresses.global_issues, *anchor_issues]
        issues.extend(
            FeatureExtractionIssue(reason, "STRICT_CHAIN", chain.strict_chain_id)
            for reason in chain.uncertainties
        )
        issues.extend(
            FeatureExtractionIssue(reason, "STRICT_RESULT", chain.result_id)
            for reason in chain.result_uncertainties
        )
        if chain.status != "READY" and not chain.uncertainties:
            issues.append(FeatureExtractionIssue("STRICT_CHAIN_NOT_READY", "STRICT_CHAIN", chain.strict_chain_id))

        if anchor_key is None:
            frozen_issues = _ordered_issues(issues)
            statuses.append(
                ChainFeatureExtractionStatus(
                    strict_chain_id=chain.strict_chain_id,
                    record_id=record_id,
                    status="REJECTED",
                    record_created=False,
                    reasons=_issue_codes(frozen_issues),
                    issues=frozen_issues,
                )
            )
            continue

        call_steps, call_issues = _call_features(
            chain,
            nodes,
            anchor_key.blockchain_chain_id,
            candidates,
            effective_addresses,
        )
        asset_behaviors, asset_issues = _asset_features(
            chain,
            edges,
            facts,
            business_addresses,
        )
        issues.extend(call_issues)
        issues.extend(asset_issues)
        frozen_issues = _ordered_issues(issues)
        record = ChainFeatureRecord(
            record_id=record_id,
            main_anchor=anchor_key,
            call_steps=call_steps,
            asset_behaviors=asset_behaviors,
            uncertainties=_issue_codes(frozen_issues),
            result_scoring_context=strict_result_chains.result_scoring_context,
        )
        records.append(record)
        statuses.append(
            ChainFeatureExtractionStatus(
                strict_chain_id=chain.strict_chain_id,
                record_id=record_id,
                status="EXTRACTED_UNCERTAIN" if record.uncertainties else "EXTRACTED_CLEAN",
                record_created=True,
                reasons=record.uncertainties,
                issues=frozen_issues,
            )
        )

    frozen_records = tuple(sorted(records, key=lambda item: item.record_id))
    frozen_statuses = tuple(sorted(statuses, key=lambda item: item.strict_chain_id))
    clean = sum(not record.uncertainties for record in frozen_records)
    return NormalFeatureExtractionResult(
        transaction_record_id=transaction_record_id,
        records=frozen_records,
        chain_statuses=frozen_statuses,
        input_chain_count=len(chains),
        record_count=len(frozen_records),
        clean_record_count=clean,
        uncertain_record_count=len(frozen_records) - clean,
        rejected_chain_count=sum(not status.record_created for status in frozen_statuses),
    )


def serialize_normal_features(result: NormalFeatureExtractionResult) -> str:
    """Return stable JSON without creating a runtime artifact."""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def _validate_inputs(
    graph: Any,
    strict: Any,
    ledger: Any,
    transaction_record_id: Any,
) -> None:
    if not isinstance(graph, TransactionGraph):
        raise ValueError("transaction_graph must be a TransactionGraph")
    if not isinstance(strict, StrictResultChainsResult):
        raise ValueError("strict_result_chains must be a StrictResultChainsResult")
    if not isinstance(ledger, ObservedAssetLedger):
        raise ValueError("observed_asset_ledger must be an ObservedAssetLedger")
    if not isinstance(transaction_record_id, str) or not transaction_record_id:
        raise ValueError("transaction_record_id must be a non-empty string")


def _main_anchor(
    anchor: MainAnchorCandidate | None,
    addresses: _EffectiveAddressMap,
    nodes: Mapping[str, NormalizedCallNode],
) -> tuple[MainAnchorKey | None, tuple[FeatureExtractionIssue, ...], frozenset[str]]:
    if anchor is None:
        issue = FeatureExtractionIssue("MAIN_ANCHOR_MISSING", "MAIN_ANCHOR")
        return None, (issue,), frozenset()
    issues: list[FeatureExtractionIssue] = []
    chain_id = anchor.blockchain_chain_id
    effective = _normalize_address(anchor.effective_address)
    selector = _normalize_selector(anchor.selector)
    runtime = _normalize_address(anchor.runtime_address)
    if not isinstance(chain_id, int) or isinstance(chain_id, bool) or chain_id <= 0:
        issues.append(FeatureExtractionIssue("MAIN_ANCHOR_BLOCKCHAIN_CHAIN_ID_INVALID", "MAIN_ANCHOR", anchor.node_id))
    if effective is None:
        issues.append(FeatureExtractionIssue("MAIN_ANCHOR_EFFECTIVE_ADDRESS_INVALID", "MAIN_ANCHOR", anchor.node_id))
    if selector is None:
        issues.append(FeatureExtractionIssue("MAIN_ANCHOR_SELECTOR_INVALID", "MAIN_ANCHOR", anchor.node_id))
    if runtime is None:
        issues.append(FeatureExtractionIssue("MAIN_ANCHOR_RUNTIME_ADDRESS_INVALID", "MAIN_ANCHOR", anchor.node_id))
    if runtime is not None and runtime in addresses.invalid_runtime_addresses:
        issues.append(FeatureExtractionIssue("OFFLINE_EFFECTIVE_ADDRESS_MAP_INVALID", "MAIN_ANCHOR", anchor.node_id))
    if runtime is not None and runtime in addresses.valid and effective is not None:
        if addresses.valid[runtime] != effective:
            issues.append(FeatureExtractionIssue("MAIN_ANCHOR_EFFECTIVE_ADDRESS_MAP_MISMATCH", "MAIN_ANCHOR", anchor.node_id))
    business_addresses: frozenset[str] = frozenset()
    anchor_node = nodes.get(anchor.node_id)
    if anchor_node is None:
        issues.append(FeatureExtractionIssue("MAIN_ANCHOR_NODE_NOT_FOUND", "MAIN_ANCHOR", anchor.node_id))
    else:
        node_callee = _normalize_address(anchor_node.callee_address)
        if node_callee is None:
            issues.append(FeatureExtractionIssue("MAIN_ANCHOR_NODE_CALLEE_ADDRESS_INVALID", "MAIN_ANCHOR", anchor.node_id))
        elif runtime is not None and node_callee != runtime:
            issues.append(FeatureExtractionIssue("MAIN_ANCHOR_RUNTIME_NODE_CALLEE_MISMATCH", "MAIN_ANCHOR", anchor.node_id))

        call_type = _normalized_call_type(anchor_node)
        if call_type in {"DELEGATECALL", "CALLCODE"}:
            context_address = _normalize_address(anchor_node.caller_address)
            if context_address is None:
                issues.append(
                    FeatureExtractionIssue(
                        "MAIN_ANCHOR_DELEGATE_CONTEXT_CALLER_ADDRESS_INVALID",
                        "MAIN_ANCHOR",
                        anchor.node_id,
                    )
                )
            else:
                business_addresses = frozenset({context_address})
        elif node_callee is not None:
            business_addresses = frozenset({node_callee})
    issues.extend(
        FeatureExtractionIssue(reason, "MAIN_ANCHOR", anchor.node_id)
        for reason in anchor.uncertainties
    )
    issues.extend(
        FeatureExtractionIssue(reason, "MAIN_ANCHOR", anchor.node_id)
        for reason in (*anchor.function_identity_uncertainties, *anchor.execution_status_uncertainties)
    )
    if (
        anchor.execution_status != "SUCCESS"
        or anchor.execution_success_evidence is not True
        or anchor.execution_status_uncertainties
    ):
        issues.append(FeatureExtractionIssue("MAIN_ANCHOR_SUCCESS_EVIDENCE_INVALID", "MAIN_ANCHOR", anchor.node_id))
    if not isinstance(chain_id, int) or isinstance(chain_id, bool) or chain_id <= 0 or effective is None or selector is None:
        return None, _ordered_issues(issues), business_addresses
    return (
        MainAnchorKey(chain_id, effective, selector),
        _ordered_issues(issues),
        business_addresses,
    )


def _normalized_call_type(node: NormalizedCallNode) -> str | None:
    value = node.call_type if isinstance(node.call_type, str) and node.call_type.strip() else node.node_type
    return value.strip().upper() if isinstance(value, str) and value.strip() else None


def _call_features(
    chain: StrictResultChain,
    nodes: Mapping[str, NormalizedCallNode],
    blockchain_chain_id: int,
    candidates: tuple[OfflineFunctionCandidate, ...],
    addresses: _EffectiveAddressMap,
) -> tuple[tuple[CallStep, ...], tuple[FeatureExtractionIssue, ...]]:
    steps: list[CallStep] = []
    issues: list[FeatureExtractionIssue] = []
    for node_id in chain.node_ids:
        node = nodes.get(node_id)
        if node is None:
            issues.append(FeatureExtractionIssue("TRANSACTION_GRAPH_NODE_NOT_FOUND", "CALL_NODE", node_id))
            continue
        runtime = _normalize_address(node.callee_address)
        if runtime is not None and runtime in addresses.invalid_runtime_addresses:
            effective_input: Any = ""
            issues.append(FeatureExtractionIssue("OFFLINE_EFFECTIVE_ADDRESS_MAP_INVALID", "CALL_NODE", node_id))
        elif runtime is not None and runtime in addresses.valid:
            effective_input = addresses.valid[runtime]
        else:
            effective_input = None
        identity = resolve_function_identity(
            blockchain_chain_id=blockchain_chain_id,
            runtime_address=node.callee_address,
            effective_address=effective_input,
            trace_selector=node.function_selector,
            function_name=node.function_name,
            offline_candidates=candidates,
        )
        if identity.usable_for_anchor and identity.effective_address is not None and identity.selector is not None:
            steps.append(CallStep(identity.effective_address, identity.selector))
            continue
        issues.append(FeatureExtractionIssue("CALL_FUNCTION_IDENTITY_UNRESOLVED", "CALL_NODE", node_id))
        issues.extend(
            FeatureExtractionIssue(reason, "CALL_NODE", node_id)
            for reason in identity.uncertainties
        )
    return tuple(steps), _ordered_issues(issues)


def _asset_features(
    chain: StrictResultChain,
    edges: Mapping[str, Any],
    facts: Mapping[str, AssetTransferFact],
    business_addresses: frozenset[str],
) -> tuple[frozenset[AssetBehavior], tuple[FeatureExtractionIssue, ...]]:
    fact_ids: set[str] = set()
    issues: list[FeatureExtractionIssue] = []
    for fact_id in chain.result_fact_ids:
        if isinstance(fact_id, str) and fact_id:
            fact_ids.add(fact_id)
        else:
            issues.append(FeatureExtractionIssue("RESULT_FACT_ID_INVALID", "STRICT_CHAIN", chain.strict_chain_id))
    for edge_id in chain.edge_ids:
        edge = edges.get(edge_id)
        if edge is None:
            issues.append(FeatureExtractionIssue("STRICT_EDGE_NOT_FOUND", "STRICT_EDGE", edge_id))
            continue
        if edge.edge_type != "ASSET":
            continue
        for fact_id in (edge.src_fact_id, edge.dst_fact_id):
            if isinstance(fact_id, str) and fact_id:
                fact_ids.add(fact_id)
            else:
                issues.append(FeatureExtractionIssue("ASSET_EDGE_FACT_ID_MISSING", "STRICT_EDGE", edge_id))

    if not fact_ids and chain.result_type in _ASSET_RESULT_TYPES:
        issues.append(FeatureExtractionIssue("EXPECTED_ASSET_FACT_MISSING", "STRICT_CHAIN", chain.strict_chain_id))

    behaviors: set[AssetBehavior] = set()
    for fact_id in sorted(fact_ids):
        fact = facts.get(fact_id)
        if fact is None:
            issues.append(FeatureExtractionIssue("ASSET_FACT_NOT_FOUND", "ASSET_FACT", fact_id))
            continue
        if fact.status != "COMPLETE":
            issues.append(FeatureExtractionIssue("ASSET_FACT_INCOMPLETE", "ASSET_FACT", fact_id))
            if fact.uncertainties:
                issues.append(FeatureExtractionIssue("ASSET_FACT_HAS_UNCERTAINTIES", "ASSET_FACT", fact_id))
            continue
        if fact.uncertainties:
            issues.append(FeatureExtractionIssue("ASSET_FACT_HAS_UNCERTAINTIES", "ASSET_FACT", fact_id))
        from_address = _normalized_address_field(fact.from_address)
        to_address = _normalized_address_field(fact.to_address)
        if from_address is None or to_address is None:
            issues.append(FeatureExtractionIssue("ASSET_FROM_TO_ADDRESS_UNRESOLVED", "ASSET_FACT", fact_id))
            continue
        if fact.asset_type == "NATIVE":
            asset_identifier = "NATIVE"
        elif fact.asset_type == "TOKEN":
            asset_identifier = _normalized_address_field(fact.asset_address)
            if asset_identifier is None:
                issues.append(FeatureExtractionIssue("ASSET_ADDRESS_MISSING_OR_INVALID", "ASSET_FACT", fact_id))
                continue
        else:
            issues.append(FeatureExtractionIssue("ASSET_TYPE_UNSUPPORTED", "ASSET_FACT", fact_id))
            continue

        from_business = from_address in business_addresses
        to_business = to_address in business_addresses
        if not from_business and not to_business:
            issues.append(
                FeatureExtractionIssue(
                    "ASSET_DIRECTION_RELATIVE_TO_ANCHOR_UNRESOLVED",
                    "ASSET_FACT",
                    fact_id,
                )
            )
            continue
        if from_business:
            behaviors.add(AssetBehavior(asset_identifier, "OUTFLOW"))
        if to_business:
            behaviors.add(AssetBehavior(asset_identifier, "INFLOW"))
        if from_business and to_business:
            issues.append(FeatureExtractionIssue("ASSET_SELF_TRANSFER_RELATIVE_TO_ANCHOR", "ASSET_FACT", fact_id))
    return frozenset(behaviors), _ordered_issues(issues)


def _normalize_effective_address_map(value: Mapping[str, str] | None) -> _EffectiveAddressMap:
    if value is None:
        return _EffectiveAddressMap({}, frozenset(), ())
    if not isinstance(value, Mapping):
        raise ValueError("offline_effective_address_map must be a mapping")
    valid: dict[str, str] = {}
    invalid_runtime: set[str] = set()
    issues: list[FeatureExtractionIssue] = []
    for raw_runtime, raw_effective in value.items():
        runtime = _normalize_address(raw_runtime)
        effective = _normalize_address(raw_effective)
        if runtime is None:
            issues.append(
                FeatureExtractionIssue(
                    "OFFLINE_EFFECTIVE_ADDRESS_MAP_RUNTIME_ADDRESS_INVALID",
                    "OFFLINE_EFFECTIVE_ADDRESS_MAP",
                )
            )
            continue
        if effective is None:
            invalid_runtime.add(runtime)
            valid.pop(runtime, None)
            issues.append(
                FeatureExtractionIssue(
                    "OFFLINE_EFFECTIVE_ADDRESS_MAP_EFFECTIVE_ADDRESS_INVALID",
                    "OFFLINE_EFFECTIVE_ADDRESS_MAP",
                )
            )
            continue
        if runtime in invalid_runtime:
            continue
        previous = valid.get(runtime)
        if previous is not None and previous != effective:
            valid.pop(runtime, None)
            invalid_runtime.add(runtime)
            issues.append(
                FeatureExtractionIssue(
                    "OFFLINE_EFFECTIVE_ADDRESS_MAP_CONFLICT",
                    "OFFLINE_EFFECTIVE_ADDRESS_MAP",
                )
            )
            continue
        valid[runtime] = effective
    return _EffectiveAddressMap(
        valid=valid,
        invalid_runtime_addresses=frozenset(invalid_runtime),
        global_issues=_ordered_issues(issues),
    )


def _unique_by_id(items: Iterable[Any], field: str, description: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        identifier = getattr(item, field, None)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{description} must be a non-empty string")
        if identifier in result:
            raise ValueError(f"Duplicate {description}: {identifier}")
        result[identifier] = item
    return result


def _normalize_address(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and _ADDRESS.fullmatch(value) else None


def _normalize_selector(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and _SELECTOR.fullmatch(value) else None


def _normalized_address_field(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return _normalize_address(value.get("normalized"))


def _ordered_issues(issues: Iterable[FeatureExtractionIssue]) -> tuple[FeatureExtractionIssue, ...]:
    return tuple(
        sorted(
            set(issues),
            key=lambda item: (item.code, item.source_kind, item.source_id or ""),
        )
    )


def _issue_codes(issues: Iterable[FeatureExtractionIssue]) -> tuple[str, ...]:
    return tuple(sorted({issue.code for issue in issues}))


__all__ = [
    "ChainFeatureExtractionStatus",
    "FeatureExtractionIssue",
    "NormalFeatureExtractionResult",
    "extract_normal_features",
    "serialize_normal_features",
]

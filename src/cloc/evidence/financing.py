"""Identify financing and repayment episodes in asset flows."""

from __future__ import annotations

from typing import Any

from cloc.evidence.continuity_models import FinancingEpisode
from cloc.evidence.transfer_models import AssetTransferFact, ObservedAssetLedger
from cloc.evidence.models import TransactionGraph
from cloc.evidence.annotations import RuntimeAnnotations


def detect_financing_episodes(
    transaction_graph: TransactionGraph,
    runtime_annotations: RuntimeAnnotations,
    observed_asset_ledger: ObservedAssetLedger,
) -> list[FinancingEpisode]:
    """Return only explicit FLASH_LOAN/SWAP scopes that contain a CALLBACK."""
    annotations = {item.node_id: item for item in runtime_annotations.node_annotations}
    nodes = {item.node_id: item for item in transaction_graph.nodes}
    complete_facts = [item for item in observed_asset_ledger.transfer_facts if item.status == "COMPLETE"]
    candidates: list[FinancingEpisode] = []

    for callback in transaction_graph.nodes:
        annotation = annotations.get(callback.node_id)
        if annotation is None or "CALLBACK" not in annotation.action_types:
            continue
        initiation = _nearest_financing_ancestor(callback, nodes, annotations)
        if initiation is None:
            continue
        initiation_annotation = annotations[initiation.node_id]
        financing_type = "EXPLICIT_FLASH_LOAN" if "FLASH_LOAN" in initiation_annotation.action_types else "CALLBACK_SWAP_CANDIDATE"
        provider = _address(initiation.callee_address)
        receiver = _address(callback.callee_address)
        if provider is None or receiver is None:
            continue
        scoped = [fact for fact in complete_facts if _in_subtree(fact.trace_path, initiation.trace_path)]
        fundings = [
            fact
            for fact in scoped
            if fact.trace_index < callback.trace_index
            and _same_address(fact.from_address, provider)
            and _same_address(fact.to_address, receiver)
        ]
        for asset_key, funding_group in _group_by_asset(fundings).items():
            repayments = [
                fact
                for fact in scoped
                if fact.trace_index > max(item.trace_index for item in funding_group)
                and _asset_key_for_fact(fact) == asset_key
                and _same_address(fact.from_address, receiver)
                and _same_address(fact.to_address, provider)
            ]
            third_party_returns = [
                fact
                for fact in scoped
                if fact.trace_index > max(item.trace_index for item in funding_group)
                and _asset_key_for_fact(fact) == asset_key
                and not _same_address(fact.from_address, receiver)
                and _same_address(fact.to_address, provider)
            ]
            principal = sum(int(item.amount_raw or "0") for item in funding_group)
            repayment = sum(int(item.amount_raw or "0") for item in repayments)
            status, uncertainties = _episode_status(principal, repayment, repayments, third_party_returns)
            asset_address = funding_group[0].asset_address
            candidates.append(
                FinancingEpisode(
                    episode_id="",
                    financing_type=financing_type,
                    initiation_node_id=initiation.node_id,
                    callback_node_id=callback.node_id,
                    provider_address=provider,
                    receiver_address=receiver,
                    asset_address=asset_address,
                    funding_fact_ids=tuple(item.fact_id for item in funding_group),
                    repayment_fact_ids=tuple(item.fact_id for item in repayments),
                    principal_raw=str(principal),
                    repayment_raw=str(repayment),
                    status=status,
                    evidence_level="STRONG_ASSOCIATION",
                    reason="funding transfer, callback and later return transfer occur in the same financing scope",
                    trace_refs=(initiation.trace_path, callback.trace_path),
                    uncertainties=tuple(uncertainties),
                )
            )

    ordered = sorted(candidates, key=lambda item: (nodes[item.initiation_node_id].trace_index, nodes[item.callback_node_id].trace_index, _asset_sort_key(item.asset_address)))
    return [
        FinancingEpisode(
            episode_id=f"financing_e{index:03d}",
            financing_type=item.financing_type,
            initiation_node_id=item.initiation_node_id,
            callback_node_id=item.callback_node_id,
            provider_address=item.provider_address,
            receiver_address=item.receiver_address,
            asset_address=item.asset_address,
            funding_fact_ids=item.funding_fact_ids,
            repayment_fact_ids=item.repayment_fact_ids,
            principal_raw=item.principal_raw,
            repayment_raw=item.repayment_raw,
            status=item.status,
            evidence_level=item.evidence_level,
            reason=item.reason,
            trace_refs=item.trace_refs,
            uncertainties=item.uncertainties,
        )
        for index, item in enumerate(ordered, start=1)
    ]


def _nearest_financing_ancestor(node: Any, nodes: dict[str, Any], annotations: dict[str, Any]) -> Any | None:
    cursor = nodes.get(node.parent_id) if node.parent_id else None
    while cursor is not None:
        actions = set(annotations.get(cursor.node_id).action_types) if cursor.node_id in annotations else set()
        if "FLASH_LOAN" in actions or "SWAP" in actions:
            return cursor
        cursor = nodes.get(cursor.parent_id) if cursor.parent_id else None
    return None


def _episode_status(principal: int, repayment: int, repayments: list[AssetTransferFact], third_party_returns: list[AssetTransferFact]) -> tuple[str, list[str]]:
    if repayments and repayment >= principal:
        return "COMPLETE", []
    if repayments:
        return "AMBIGUOUS", ["observed return amount is smaller than observed funding amount"]
    if third_party_returns:
        return "REPAYMENT_UNRESOLVED", ["a later return to provider is observed from an address other than the callback receiver"]
    return "FUNDING_ONLY", ["no qualifying receiver-to-provider return transfer is observed in the financing scope"]


def _group_by_asset(facts: list[AssetTransferFact]) -> dict[tuple[str, str], list[AssetTransferFact]]:
    result: dict[tuple[str, str], list[AssetTransferFact]] = {}
    for fact in facts:
        result.setdefault(_asset_key_for_fact(fact), []).append(fact)
    return result


def _asset_key_for_fact(fact: AssetTransferFact) -> tuple[str, str]:
    return fact.asset_type, fact.asset_address["normalized"] if fact.asset_address is not None else "NATIVE"


def _in_subtree(trace_path: str, root_path: str) -> bool:
    return trace_path == root_path or trace_path.startswith(root_path + "/")


def _address(value: Any) -> dict[str, str] | None:
    if isinstance(value, str) and value.startswith("0x") and len(value) == 42:
        return {"original": value, "normalized": value.lower()}
    return None


def _same_address(left: dict[str, str] | None, right: dict[str, str] | None) -> bool:
    return left is not None and right is not None and left["normalized"] == right["normalized"]


def _asset_sort_key(asset_address: dict[str, str] | None) -> str:
    return asset_address["normalized"] if asset_address is not None else "NATIVE"

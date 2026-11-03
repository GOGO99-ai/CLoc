from __future__ import annotations

import unittest
from dataclasses import replace

from src.cloc_v2.asset_ledger_models import AssetTransferFact, ObservedAssetLedger
from src.cloc_v2.function_identity import OfflineFunctionCandidate
from src.cloc_v2.normal_baseline import build_normal_baseline
from src.cloc_v2.normal_distance import AssetBehavior, CallStep, MainAnchorKey
from src.cloc_v2.normal_features import extract_normal_features, serialize_normal_features
from src.cloc_v2.runtime_semantics import transaction_graph_sha256
from src.cloc_v2.source_index import selector_for_signature
from src.cloc_v2.strict_result_chains import (
    MainAnchorCandidate,
    StrictEvidenceEdge,
    StrictResultChain,
    StrictResultChainsResult,
)
from src.cloc_v2.transaction_graph import build_transaction_graph


def addr(character: str) -> str:
    return "0x" + character * 40


def node(
    address: str,
    function: str,
    children: list[dict] | None = None,
    *,
    call_type: str = "call",
    caller_address: str | None = None,
) -> dict:
    value = {
        "type": call_type,
        "call_type": call_type,
        "address": address,
        "function": function,
        "children": children or [],
    }
    if caller_address is not None:
        value["from"] = caller_address
    return value


def graph(tree: dict):
    return build_transaction_graph(tree)


def anchor(
    *,
    node_id: str = "n001",
    runtime_address: str = addr("a"),
    effective_address: str | None = None,
    selector: str = "0x11111111",
) -> MainAnchorCandidate:
    effective = effective_address or runtime_address
    return MainAnchorCandidate(
        blockchain_chain_id=1,
        node_id=node_id,
        runtime_address=runtime_address,
        effective_address=effective,
        address_resolution="RUNTIME_ADDRESS_FALLBACK" if effective == runtime_address else "OFFLINE_EFFECTIVE_ADDRESS_MAP",
        selector=selector,
        function_identity_source="TRACE_NATIVE",
        candidate_signatures=(),
        candidate_selectors=(),
        candidate_evidence_sources=(),
        function_identity_uncertainties=(),
        execution_status="SUCCESS",
        execution_status_source="CALLTRACER_OUTPUT",
        execution_success_evidence=True,
        execution_status_uncertainties=(),
        uncertainties=(),
    )


def strict_chain(
    strict_chain_id: str = "strict_result_chain_001",
    *,
    node_ids: tuple[str, ...] = ("n001",),
    edge_ids: tuple[str, ...] = (),
    result_fact_ids: tuple[str, ...] = (),
    result_type: str | None = None,
    main_anchor: MainAnchorCandidate | None = None,
    status: str = "READY",
    uncertainties: tuple[str, ...] = (),
) -> StrictResultChain:
    return StrictResultChain(
        strict_chain_id=strict_chain_id,
        result_id=f"result_{strict_chain_id}",
        result_node_id=node_ids[-1],
        result_fact_ids=result_fact_ids,
        result_type=result_type,
        asset_type=None,
        asset_address=None,
        from_address=None,
        to_address=None,
        amount_raw=None,
        result_uncertainties=(),
        node_ids=node_ids,
        edge_ids=edge_ids,
        anchor=main_anchor if main_anchor is not None else anchor(),
        status=status,
        uncertainties=uncertainties,
    )


def strict_result(*chains: StrictResultChain, edges: tuple[StrictEvidenceEdge, ...] = ()) -> StrictResultChainsResult:
    return StrictResultChainsResult(
        blockchain_chain_id=1,
        edges=edges,
        chains=tuple(chains),
        result_statuses=(),
        critical_uncertainties=(),
        coverage={},
    )


def fact(
    fact_id: str,
    *,
    asset_type: str = "TOKEN",
    asset_address: str | None = None,
    from_address: str | None = None,
    to_address: str | None = None,
    status: str = "COMPLETE",
    amount_raw: str | None = "999999999999999999",
    uncertainties: tuple[str, ...] = (),
) -> AssetTransferFact:
    def wrapped(value: str | None):
        return None if value is None else {"original": value, "normalized": value.lower()}

    return AssetTransferFact(
        fact_id=fact_id,
        node_id="n001",
        trace_index=1,
        trace_path="0",
        transfer_kind="NATIVE_VALUE" if asset_type == "NATIVE" else "TRANSFER",
        asset_type=asset_type,
        asset_address=wrapped(asset_address),
        operator_address=wrapped(from_address),
        from_address=wrapped(from_address),
        to_address=wrapped(to_address),
        amount_raw=amount_raw,
        status=status,
        evidence_level="OBSERVED",
        reason="test evidence",
        trace_refs=("0",),
        uncertainties=uncertainties,
    )


def ledger(transaction_graph, *facts: AssetTransferFact) -> ObservedAssetLedger:
    return ObservedAssetLedger(
        transaction_graph_sha256=transaction_graph_sha256(transaction_graph),
        runtime_annotations_sha256="runtime-annotations",
        transfer_facts=tuple(facts),
        address_flow_summaries=(),
    )


def extract(transaction_graph, strict, observed, **kwargs):
    return extract_normal_features(
        transaction_graph,
        strict,
        observed,
        transaction_record_id="tx-normal-001",
        **kwargs,
    )


class NormalFeatureTests(unittest.TestCase):
    def test_valid_main_anchor_is_converted_without_reselection(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        result = extract(transaction_graph, strict_result(strict_chain()), ledger(transaction_graph))
        self.assertEqual(result.records[0].main_anchor, MainAnchorKey(1, addr("a"), "0x11111111"))
        self.assertEqual(result.records[0].record_id, "tx-normal-001:strict_result_chain_001")

    def test_call_order_and_repeated_steps_are_preserved(self) -> None:
        transaction_graph = graph(
            node(addr("a"), "0x11111111", [node(addr("b"), "0x22222222", [node(addr("a"), "0x11111111")])])
        )
        chain = strict_chain(node_ids=("n001", "n002", "n003"))
        result = extract(transaction_graph, strict_result(chain), ledger(transaction_graph))
        self.assertEqual(
            result.records[0].call_steps,
            (
                CallStep(addr("a"), "0x11111111"),
                CallStep(addr("b"), "0x22222222"),
                CallStep(addr("a"), "0x11111111"),
            ),
        )

    def test_native_selector_has_priority(self) -> None:
        transaction_graph = graph(node(addr("a"), "0xabcdef01"))
        chain = strict_chain(main_anchor=anchor(selector="0xabcdef01"))
        candidate = OfflineFunctionCandidate(1, addr("a"), "anything", "anything(uint256)")
        result = extract(
            transaction_graph,
            strict_result(chain),
            ledger(transaction_graph),
            offline_function_candidates=(candidate,),
        )
        self.assertEqual(result.records[0].call_steps[0].selector, "0xabcdef01")

    def test_function_name_is_recovered_by_unique_offline_abi(self) -> None:
        transaction_graph = graph(node(addr("a"), "borrow"))
        candidate = OfflineFunctionCandidate(1, addr("a"), "borrow", "borrow(uint256)", evidence_source="local-abi")
        result = extract(
            transaction_graph,
            strict_result(strict_chain()),
            ledger(transaction_graph),
            offline_function_candidates=(candidate,),
        )
        self.assertEqual(result.records[0].call_steps[0].selector, selector_for_signature("borrow(uint256)"))
        self.assertEqual(result.records[0].uncertainties, ())

    def test_ambiguous_function_identity_becomes_uncertain(self) -> None:
        transaction_graph = graph(node(addr("a"), "borrow"))
        candidates = (
            OfflineFunctionCandidate(1, addr("a"), "borrow", "borrow(uint256)"),
            OfflineFunctionCandidate(1, addr("a"), "borrow", "borrow(address)"),
        )
        result = extract(
            transaction_graph,
            strict_result(strict_chain()),
            ledger(transaction_graph),
            offline_function_candidates=candidates,
        )
        self.assertEqual(result.records[0].call_steps, ())
        self.assertIn("SELECTOR_RECOVERY_AMBIGUOUS", result.records[0].uncertainties)
        self.assertEqual(result.chain_statuses[0].status, "EXTRACTED_UNCERTAIN")

    def test_effective_address_map_is_used_for_call_steps(self) -> None:
        implementation = addr("e")
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        chain = strict_chain(main_anchor=anchor(effective_address=implementation))
        result = extract(
            transaction_graph,
            strict_result(chain),
            ledger(transaction_graph),
            offline_effective_address_map={addr("a"): implementation},
        )
        self.assertEqual(result.records[0].call_steps[0].effective_address, implementation)

    def test_outflow_is_extracted_from_asset_edge_fact_ids(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        asset_edge = StrictEvidenceEdge(
            "strict_asset_e001", "ASSET", "n001", "n001", "asset_e001",
            src_fact_id="asset_f001", dst_fact_id="asset_f001",
        )
        chain = strict_chain(edge_ids=(asset_edge.edge_id,), result_type="R_OUTFLOW")
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=addr("d"), from_address=addr("a"), to_address=addr("b")),
        )
        result = extract(transaction_graph, strict_result(chain, edges=(asset_edge,)), observed)
        self.assertIn(AssetBehavior(addr("d"), "OUTFLOW"), result.records[0].asset_behaviors)

    def test_inflow_is_extracted_relative_to_anchor(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        chain = strict_chain(result_fact_ids=("asset_f001",), result_type="R_PROFIT")
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=addr("d"), from_address=addr("b"), to_address=addr("a")),
        )
        result = extract(transaction_graph, strict_result(chain), observed)
        self.assertEqual(result.records[0].asset_behaviors, frozenset({AssetBehavior(addr("d"), "INFLOW")}))

    def test_delegatecall_uses_caller_proxy_for_outflow_and_can_be_clean(self) -> None:
        proxy, implementation, receiver, token = addr("a"), addr("e"), addr("b"), addr("d")
        transaction_graph = graph(
            node(
                proxy,
                "0xaaaaaaaa",
                [node(implementation, "0x11111111", call_type="DELEGATECALL")],
            )
        )
        chain = strict_chain(
            node_ids=("n001", "n002"),
            result_fact_ids=("asset_f001",),
            result_type="R_OUTFLOW",
            main_anchor=anchor(
                node_id="n002",
                runtime_address=implementation,
                effective_address=implementation,
            ),
        )
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=token, from_address=proxy, to_address=receiver),
        )
        result = extract(transaction_graph, strict_result(chain), observed)
        self.assertEqual(result.records[0].main_anchor.effective_address, implementation)
        self.assertEqual(result.records[0].asset_behaviors, frozenset({AssetBehavior(token, "OUTFLOW")}))
        self.assertEqual(result.chain_statuses[0].status, "EXTRACTED_CLEAN")
        self.assertEqual((result.clean_record_count, result.uncertain_record_count), (1, 0))

    def test_delegatecall_uses_caller_proxy_for_inflow(self) -> None:
        proxy, implementation, sender, token = addr("a"), addr("e"), addr("b"), addr("d")
        transaction_graph = graph(
            node(
                proxy,
                "0xaaaaaaaa",
                [node(implementation, "0x11111111", call_type="delegatecall")],
            )
        )
        chain = strict_chain(
            node_ids=("n001", "n002"),
            result_fact_ids=("asset_f001",),
            result_type="R_PROFIT",
            main_anchor=anchor(node_id="n002", runtime_address=implementation),
        )
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=token, from_address=sender, to_address=proxy),
        )
        result = extract(transaction_graph, strict_result(chain), observed)
        self.assertEqual(result.records[0].asset_behaviors, frozenset({AssetBehavior(token, "INFLOW")}))

    def test_delegatecall_does_not_use_implementation_as_asset_context(self) -> None:
        proxy, implementation, receiver, token = addr("a"), addr("e"), addr("b"), addr("d")
        transaction_graph = graph(
            node(
                proxy,
                "0xaaaaaaaa",
                [node(implementation, "0x11111111", call_type="DELEGATECALL")],
            )
        )
        chain = strict_chain(
            node_ids=("n001", "n002"),
            result_fact_ids=("asset_f001",),
            result_type="R_OUTFLOW",
            main_anchor=anchor(
                node_id="n002",
                runtime_address=implementation,
                effective_address=implementation,
            ),
        )
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=token, from_address=implementation, to_address=receiver),
        )
        result = extract(transaction_graph, strict_result(chain), observed)
        self.assertEqual(result.records[0].asset_behaviors, frozenset())
        self.assertIn("ASSET_DIRECTION_RELATIVE_TO_ANCHOR_UNRESOLVED", result.records[0].uncertainties)

    def test_ordinary_call_uses_callee_as_asset_context(self) -> None:
        business, receiver, token = addr("a"), addr("b"), addr("d")
        transaction_graph = graph(node(business, "0x11111111", call_type="CALL"))
        chain = strict_chain(
            result_fact_ids=("asset_f001",),
            result_type="R_OUTFLOW",
        )
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=token, from_address=business, to_address=receiver),
        )
        result = extract(transaction_graph, strict_result(chain), observed)
        self.assertEqual(result.records[0].asset_behaviors, frozenset({AssetBehavior(token, "OUTFLOW")}))

    def test_delegatecall_external_endpoints_remain_direction_unresolved(self) -> None:
        proxy, implementation, sender, receiver, token = addr("a"), addr("e"), addr("b"), addr("c"), addr("d")
        transaction_graph = graph(
            node(
                proxy,
                "0xaaaaaaaa",
                [node(implementation, "0x11111111", call_type="CALLCODE")],
            )
        )
        chain = strict_chain(
            node_ids=("n001", "n002"),
            result_fact_ids=("asset_f001",),
            result_type="R_OUTFLOW",
            main_anchor=anchor(node_id="n002", runtime_address=implementation),
        )
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=token, from_address=sender, to_address=receiver),
        )
        result = extract(transaction_graph, strict_result(chain), observed)
        self.assertEqual(result.records[0].asset_behaviors, frozenset())
        self.assertIn("ASSET_DIRECTION_RELATIVE_TO_ANCHOR_UNRESOLVED", result.records[0].uncertainties)

    def test_delegatecall_missing_caller_remains_uncertain(self) -> None:
        implementation, receiver, token = addr("e"), addr("b"), addr("d")
        transaction_graph = graph(node(implementation, "0x11111111", call_type="DELEGATECALL"))
        chain = strict_chain(
            result_fact_ids=("asset_f001",),
            result_type="R_OUTFLOW",
            main_anchor=anchor(runtime_address=implementation),
        )
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=token, from_address=implementation, to_address=receiver),
        )
        result = extract(transaction_graph, strict_result(chain), observed)
        self.assertIn(
            "MAIN_ANCHOR_DELEGATE_CONTEXT_CALLER_ADDRESS_INVALID",
            result.records[0].uncertainties,
        )

    def test_native_asset_uses_native_identifier(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        chain = strict_chain(result_fact_ids=("asset_f001",), result_type="R_OUTFLOW")
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_type="NATIVE", from_address=addr("a"), to_address=addr("b")),
        )
        result = extract(transaction_graph, strict_result(chain), observed)
        self.assertIn(AssetBehavior("NATIVE", "OUTFLOW"), result.records[0].asset_behaviors)

    def test_asset_amount_does_not_enter_features(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        chain = strict_chain(result_fact_ids=("asset_f001",), result_type="R_OUTFLOW")
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=addr("d"), from_address=addr("a"), to_address=addr("b"), amount_raw="9876543210123456789"),
        )
        rendered = serialize_normal_features(extract(transaction_graph, strict_result(chain), observed))
        self.assertNotIn("9876543210123456789", rendered)
        self.assertNotIn("amount", rendered.lower())

    def test_missing_and_incomplete_facts_propagate_uncertainty(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        missing = extract(
            transaction_graph,
            strict_result(strict_chain(result_fact_ids=("asset_missing",), result_type="R_OUTFLOW")),
            ledger(transaction_graph),
        )
        incomplete_fact = fact(
            "asset_f001",
            asset_address=addr("d"),
            from_address=addr("a"),
            to_address=None,
            status="INCOMPLETE",
        )
        incomplete = extract(
            transaction_graph,
            strict_result(strict_chain(result_fact_ids=("asset_f001",), result_type="R_OUTFLOW")),
            ledger(transaction_graph, incomplete_fact),
        )
        self.assertIn("ASSET_FACT_NOT_FOUND", missing.records[0].uncertainties)
        self.assertIn("ASSET_FACT_INCOMPLETE", incomplete.records[0].uncertainties)

    def test_direction_unresolved_does_not_guess_behavior(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        chain = strict_chain(result_fact_ids=("asset_f001",), result_type="R_OUTFLOW")
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=addr("d"), from_address=addr("b"), to_address=addr("c")),
        )
        result = extract(transaction_graph, strict_result(chain), observed)
        self.assertEqual(result.records[0].asset_behaviors, frozenset())
        self.assertIn("ASSET_DIRECTION_RELATIVE_TO_ANCHOR_UNRESOLVED", result.records[0].uncertainties)

    def test_non_ready_strict_chain_cannot_create_clean_record(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        chain = strict_chain(status="UNCERTAIN", uncertainties=("CALL_SUCCESS_STATUS_NOT_REPORTED",))
        result = extract(transaction_graph, strict_result(chain), ledger(transaction_graph))
        self.assertEqual(result.uncertain_record_count, 1)
        self.assertIn("CALL_SUCCESS_STATUS_NOT_REPORTED", result.records[0].uncertainties)

    def test_ledger_graph_hash_mismatch_is_rejected(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        mismatched = replace(ledger(transaction_graph), transaction_graph_sha256="different")
        with self.assertRaisesRegex(ValueError, "transaction_graph_sha256"):
            extract(transaction_graph, strict_result(strict_chain()), mismatched)

    def test_input_order_does_not_change_stable_serialization(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        first_chain = strict_chain("strict_result_chain_001")
        second_chain = strict_chain("strict_result_chain_002")
        observed = ledger(transaction_graph)
        first = extract(transaction_graph, strict_result(second_chain, first_chain), observed)
        second = extract(transaction_graph, strict_result(first_chain, second_chain), observed)
        self.assertEqual(first, second)
        self.assertEqual(serialize_normal_features(first), serialize_normal_features(second))

    def test_output_contains_no_forbidden_fields(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        rendered = serialize_normal_features(
            extract(transaction_graph, strict_result(strict_chain()), ledger(transaction_graph))
        ).lower()
        for forbidden in ("label", "cause", "location", "pair", "ground_truth"):
            self.assertNotIn(forbidden, rendered)

    def test_clean_extracted_record_can_build_normal_baseline(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        extracted = extract(transaction_graph, strict_result(strict_chain()), ledger(transaction_graph))
        self.assertEqual((extracted.clean_record_count, extracted.uncertain_record_count), (1, 0))
        baseline = build_normal_baseline(extracted.records, min_group_size=1)
        self.assertEqual((baseline.record_count, baseline.group_count), (1, 1))

    def test_missing_main_anchor_rejects_record_instead_of_reselecting(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        chain = replace(strict_chain(), anchor=None)
        extracted = extract(transaction_graph, strict_result(chain), ledger(transaction_graph))
        self.assertEqual(extracted.records, ())
        self.assertEqual(extracted.chain_statuses[0].status, "REJECTED")
        self.assertIn("MAIN_ANCHOR_MISSING", extracted.chain_statuses[0].reasons)

    def test_self_transfer_keeps_both_directions_and_marks_uncertain(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        chain = strict_chain(result_fact_ids=("asset_f001",), result_type="R_OUTFLOW")
        observed = ledger(
            transaction_graph,
            fact("asset_f001", asset_address=addr("d"), from_address=addr("a"), to_address=addr("a")),
        )
        extracted = extract(transaction_graph, strict_result(chain), observed)
        self.assertEqual(
            extracted.records[0].asset_behaviors,
            frozenset({AssetBehavior(addr("d"), "INFLOW"), AssetBehavior(addr("d"), "OUTFLOW")}),
        )
        self.assertIn("ASSET_SELF_TRANSFER_RELATIVE_TO_ANCHOR", extracted.records[0].uncertainties)

    def test_asset_result_without_fact_references_is_uncertain(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        chain = strict_chain(result_type="R_PROFIT")
        extracted = extract(transaction_graph, strict_result(chain), ledger(transaction_graph))
        self.assertIn("EXPECTED_ASSET_FACT_MISSING", extracted.records[0].uncertainties)

    def test_invalid_effective_address_map_cannot_be_silently_ignored(self) -> None:
        transaction_graph = graph(node(addr("a"), "0x11111111"))
        extracted = extract(
            transaction_graph,
            strict_result(strict_chain()),
            ledger(transaction_graph),
            offline_effective_address_map={addr("f"): "invalid"},
        )
        self.assertIn(
            "OFFLINE_EFFECTIVE_ADDRESS_MAP_EFFECTIVE_ADDRESS_INVALID",
            extracted.records[0].uncertainties,
        )


if __name__ == "__main__":
    unittest.main()

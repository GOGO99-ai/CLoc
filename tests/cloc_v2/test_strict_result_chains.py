from __future__ import annotations

import copy
import unittest

from src.cloc_v2.function_identity import OfflineFunctionCandidate
from src.cloc_v2.call_status import resolve_call_status
from src.cloc_v2.strict_result_chains import (
    build_strict_result_chains,
    serialize_strict_result_chains,
)
from src.cloc_v2.transaction_graph import build_transaction_graph


def addr(character: str) -> str:
    return "0x" + character * 40


def graph(tree: dict):
    return build_transaction_graph(tree)


def call_status(frame: dict) -> dict:
    """Construct only the safe status contract a call tree may carry."""
    return resolve_call_status(frame).to_dict()


def call_node(address: str, function: str, status: dict, children: list[dict] | None = None) -> dict:
    return {
        "type": "call",
        "address": address,
        "function": function,
        "call_status": status,
        "children": children or [],
    }


def continuity(*edges: dict) -> dict:
    return {"schema_version": "asset_continuity_v1", "asset_edges": list(edges)}


def asset(edge_id: str, source: str, target: str, *, uncertainties: list[str] | None = None) -> dict:
    return {
        "edge_id": edge_id,
        "src_node_id": source,
        "dst_node_id": target,
        "src_fact_id": f"{edge_id}_source",
        "dst_fact_id": f"{edge_id}_target",
        "asset_type": "TOKEN",
        "asset_address": {"original": addr("D"), "normalized": addr("d")},
        "via_address": {"original": addr("E"), "normalized": addr("e")},
        "allocated_amount_raw": "123",
        "match_type": "EXACT",
        "evidence_level": "STRONG_ASSOCIATION",
        "uncertainties": uncertainties or [],
    }


def results(*items: dict) -> dict:
    return {"schema_version": "result_candidates_v1", "result_candidates": list(items)}


def result(
    result_id: str,
    node_id: str,
    *fact_ids: str,
    uncertainties: list[str] | None = None,
) -> dict:
    return {
        "result_id": result_id,
        "node_id": node_id,
        "result_type": "R_OUTFLOW",
        "asset_type": "TOKEN",
        "asset_address": {"original": addr("D"), "normalized": addr("d")},
        "from_address": {"original": addr("A"), "normalized": addr("a")},
        "to_address": {"original": addr("B"), "normalized": addr("b")},
        "amount_raw": "123",
        "source_fact_ids": list(fact_ids),
        "uncertainties": uncertainties or [],
    }


def offline_candidate(
    function_name: str,
    signature: str,
    *,
    effective_address: str | None = None,
    evidence_source: str = "offline_abi",
) -> OfflineFunctionCandidate:
    return OfflineFunctionCandidate(
        blockchain_chain_id=1,
        effective_address=effective_address or addr("b"),
        function_name=function_name,
        signature=signature,
        evidence_source=evidence_source,
    )


def business_result_graph(function: str) -> dict:
    return {
        "type": "call",
        "address": addr("a"),
        "function": "0x11111111",
        "children": [
            {
                "type": "call",
                "address": addr("b"),
                "function": function,
                "children": [
                    {"type": "call", "address": addr("c"), "function": "transfer", "children": []}
                ],
            }
        ],
    }


class StrictResultChainTests(unittest.TestCase):
    def test_simple_call_chain_and_runtime_anchor(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [
                    {
                        "type": "call",
                        "address": addr("b"),
                        "function": "transfer",
                        "children": [],
                    }
                ],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(),
            results(result("result_001", "n002", "fact_001")),
            blockchain_chain_id=1,
        )
        chain = built.chains[0]
        self.assertEqual((chain.node_ids, chain.edge_ids), (("n001", "n002"), ("strict_call_call_e001",)))
        self.assertEqual((chain.anchor.node_id, chain.anchor.blockchain_chain_id), ("n001", 1))
        self.assertEqual(chain.anchor.address_resolution, "RUNTIME_ADDRESS_FALLBACK")
        self.assertEqual(chain.status, "UNCERTAIN")
        self.assertIn("CALL_SUCCESS_STATUS_NOT_REPORTED", chain.uncertainties)
        self.assertNotIn("BLOCKCHAIN_CHAIN_ID_MISSING", chain.uncertainties)
        self.assertEqual(built.coverage["coverage_rate"], "1.0000")

    def test_call_and_asset_paths_contain_only_allowed_edge_types(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [
                    {"type": "call", "address": addr("b"), "function": "0x22222222", "children": []},
                    {
                        "type": "call",
                        "address": addr("c"),
                        "function": "0x33333333",
                        "children": [],
                    },
                ],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(asset("asset_e001", "n002", "n003")),
            results(result("result_001", "n003")),
        )
        by_id = {edge.edge_id: edge.edge_type for edge in built.edges}
        self.assertEqual(len(built.chains), 2)
        self.assertTrue(all(by_id[edge_id] in {"CALL", "ASSET"} for chain in built.chains for edge_id in chain.edge_ids))
        self.assertIn("strict_asset_asset_e001", {edge_id for chain in built.chains for edge_id in chain.edge_ids})

    def test_branching_preserves_multiple_paths(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [
                    {"type": "call", "address": addr("b"), "function": "0x22222222", "children": []},
                    {"type": "call", "address": addr("c"), "function": "0x33333333", "children": []},
                ],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(asset("asset_e001", "n002", "n003")),
            results(result("result_001", "n003")),
        )
        self.assertEqual([chain.node_ids for chain in built.chains], [("n001", "n002", "n003"), ("n001", "n003")])
        self.assertEqual(built.result_statuses[0].strict_chain_ids, ("strict_result_chain_001", "strict_result_chain_002"))

    def test_completely_duplicate_paths_are_deduplicated(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [{"type": "call", "address": addr("b"), "function": "0x22222222", "children": []}],
            }
        )
        duplicate = asset("asset_e001", "n001", "n002")
        built = build_strict_result_chains(
            transaction_graph,
            continuity(duplicate, copy.deepcopy(duplicate)),
            results(result("result_001", "n002")),
        )
        asset_paths = [
            chain for chain in built.chains if chain.edge_ids == ("strict_asset_asset_e001",)
        ]
        self.assertEqual(len(asset_paths), 1)
        self.assertEqual(len(built.chains), 2)  # The independent CALL path remains.

    def test_different_result_candidates_are_never_merged(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [{"type": "call", "address": addr("b"), "function": "0x22222222", "children": []}],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(),
            results(result("result_001", "n002"), result("result_002", "n002")),
        )
        self.assertEqual([chain.result_id for chain in built.chains], ["result_001", "result_002"])
        self.assertEqual(len(built.result_statuses), 2)

    def test_standard_transfer_calls_are_skipped_as_anchors(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [
                    {
                        "type": "call",
                        "address": addr("b"),
                        "function": "transfer",
                        "children": [
                            {
                                "type": "call",
                                "address": addr("c"),
                                "function": "0x23b872dd",
                                "children": [],
                            }
                        ],
                    }
                ],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(),
            results(result("result_001", "n003")),
        )
        self.assertEqual(built.chains[0].anchor.node_id, "n001")

    def test_missing_selector_marks_result_uncertain(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "main",
                "children": [{"type": "call", "address": addr("b"), "function": "transfer", "children": []}],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(),
            results(result("result_001", "n002")),
        )
        self.assertEqual(built.chains[0].status, "UNCERTAIN")
        self.assertIn("ANCHOR_NOT_FOUND", built.result_statuses[0].reasons)

    def test_cycle_protection_and_output_are_deterministic(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [{"type": "call", "address": addr("b"), "function": "0x22222222", "children": []}],
            }
        )
        cycle = continuity(asset("asset_e001", "n002", "n001"))
        candidate_document = results(result("result_001", "n002"))
        first = build_strict_result_chains(transaction_graph, cycle, candidate_document)
        second = build_strict_result_chains(transaction_graph, cycle, candidate_document)
        self.assertEqual(serialize_strict_result_chains(first), serialize_strict_result_chains(second))
        self.assertTrue(any(reason.startswith("CYCLE_EDGE_SKIPPED") for reason in first.chains[0].uncertainties))

    def test_coverage_counts_missing_result_endpoint(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [{"type": "call", "address": addr("b"), "function": "0x22222222", "children": []}],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(),
            results(result("result_001", "n002"), result("result_002", "missing")),
        )
        self.assertEqual(
            built.coverage,
            {
                "result_candidate_count": 2,
                "chains_formed_result_candidate_count": 1,
                "uncertain_result_candidate_count": 2,
                "critical_missing_count": 0,
                "coverage_rate": "0.5000",
            },
        )

    def test_asset_metadata_is_preserved_and_serialization_is_stable(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [{"type": "call", "address": addr("b"), "function": "0x22222222", "children": []}],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(asset("asset_e001", "n001", "n002")),
            results(result("result_001", "n002", "asset_e001_target")),
        )
        asset_edge = next(edge for edge in built.edges if edge.edge_type == "ASSET")
        self.assertEqual(
            (
                asset_edge.src_fact_id,
                asset_edge.dst_fact_id,
                asset_edge.asset_type,
                asset_edge.asset_address,
                asset_edge.via_address,
                asset_edge.allocated_amount_raw,
                asset_edge.match_type,
                asset_edge.evidence_level,
            ),
            (
                "asset_e001_source",
                "asset_e001_target",
                "TOKEN",
                addr("d"),
                addr("e"),
                "123",
                "EXACT",
                "STRONG_ASSOCIATION",
            ),
        )
        call_edge = next(edge for edge in built.edges if edge.edge_type == "CALL")
        self.assertIsNone(call_edge.asset_address)
        self.assertEqual(serialize_strict_result_chains(built), serialize_strict_result_chains(built))

    def test_result_metadata_and_uncertainty_are_propagated(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(),
            results(result("result_001", "n001", "fact_001", uncertainties=["SOURCE_AMOUNT_UNCERTAIN"])),
        )
        chain = built.chains[0]
        self.assertEqual(
            (chain.result_type, chain.asset_type, chain.asset_address, chain.from_address, chain.to_address, chain.amount_raw),
            ("R_OUTFLOW", "TOKEN", addr("d"), addr("a"), addr("b"), "123"),
        )
        self.assertEqual(chain.result_uncertainties, ("SOURCE_AMOUNT_UNCERTAIN",))
        self.assertIn("SOURCE_AMOUNT_UNCERTAIN", chain.uncertainties)
        self.assertIn("SOURCE_AMOUNT_UNCERTAIN", built.result_statuses[0].reasons)

    def test_malformed_asset_edge_becomes_related_critical_uncertainty(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [{"type": "call", "address": addr("b"), "function": "0x22222222", "children": []}],
            }
        )
        malformed = asset("asset_e001", "unknown", "n002")
        built = build_strict_result_chains(
            transaction_graph,
            continuity(malformed),
            results(result("result_001", "n002", "asset_e001_target")),
        )
        self.assertEqual(built.critical_uncertainties[0].code, "ASSET_EDGE_ENDPOINT_UNKNOWN")
        self.assertEqual(built.coverage["critical_missing_count"], 1)
        self.assertEqual(built.result_statuses[0].status, "UNCERTAIN")
        self.assertIn("ASSET_EDGE_ENDPOINT_UNKNOWN", built.result_statuses[0].reasons)

    def test_closer_business_call_without_selector_blocks_farther_anchor(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [
                    {
                        "type": "call",
                        "address": addr("b"),
                        "function": "borrow",
                        "children": [
                            {"type": "call", "address": addr("c"), "function": "transfer", "children": []}
                        ],
                    }
                ],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(),
            results(result("result_001", "n003")),
        )
        chain = built.chains[0]
        self.assertEqual(chain.anchor.node_id, "n001")
        self.assertEqual(chain.status, "UNCERTAIN")
        self.assertIn("CLOSER_NON_ASSET_CALL_SELECTOR_MISSING", chain.uncertainties)

    def test_unusable_blockchain_chain_id_marks_chain_and_result_uncertain(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [],
            }
        )
        for unusable_chain_id in (None, "", "ethereum", 0, -1, False):
            with self.subTest(blockchain_chain_id=unusable_chain_id):
                built = build_strict_result_chains(
                    transaction_graph,
                    continuity(),
                    results(result("result_001", "n001")),
                    blockchain_chain_id=unusable_chain_id,
                )
                self.assertEqual(built.chains[0].status, "UNCERTAIN")
                self.assertIn("BLOCKCHAIN_CHAIN_ID_MISSING", built.chains[0].uncertainties)
                self.assertEqual(built.result_statuses[0].status, "UNCERTAIN")
                self.assertIn("BLOCKCHAIN_CHAIN_ID_MISSING", built.result_statuses[0].reasons)

    def test_missing_function_and_selector_still_blocks_farther_anchor(self) -> None:
        transaction_graph = graph(
            {
                "type": "call",
                "address": addr("a"),
                "function": "0x11111111",
                "children": [
                    {
                        "type": "call",
                        "address": addr("b"),
                        "children": [
                            {"type": "call", "address": addr("c"), "function": "transfer", "children": []}
                        ],
                    }
                ],
            }
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(),
            results(result("result_001", "n003")),
            blockchain_chain_id=1,
        )
        chain = built.chains[0]
        self.assertEqual(chain.anchor.node_id, "n001")
        self.assertIn("CLOSER_NON_ASSET_CALL_SELECTOR_MISSING", chain.uncertainties)

    def test_native_selector_keeps_original_anchor_behavior_after_integration(self) -> None:
        transaction_graph = graph(
            {"type": "call", "address": addr("a"), "function": "0x11111111", "children": []}
        )
        built = build_strict_result_chains(
            transaction_graph,
            continuity(),
            results(result("result_001", "n001")),
            blockchain_chain_id=1,
        )
        anchor = built.chains[0].anchor
        self.assertEqual(anchor.selector, "0x11111111")
        self.assertEqual(anchor.function_identity_source, "TRACE_NATIVE")
        self.assertEqual(anchor.address_resolution, "RUNTIME_ADDRESS_FALLBACK")
        self.assertEqual(anchor.candidate_signatures, ())
        self.assertIn("CALL_SUCCESS_STATUS_NOT_REPORTED", built.chains[0].uncertainties)

    def test_unique_offline_abi_recovers_function_name_anchor(self) -> None:
        built = build_strict_result_chains(
            graph(business_result_graph("borrow")),
            continuity(),
            results(result("result_001", "n003")),
            blockchain_chain_id=1,
            offline_function_candidates=(
                offline_candidate("borrow", "borrow(uint256)", evidence_source="local_index"),
            ),
        )
        anchor = built.chains[0].anchor
        self.assertEqual(anchor.node_id, "n002")
        self.assertEqual(anchor.function_identity_source, "OFFLINE_ABI_UNIQUE")
        self.assertEqual(anchor.candidate_signatures, ("borrow(uint256)",))
        self.assertEqual(anchor.candidate_selectors, (anchor.selector,))
        self.assertEqual(anchor.candidate_evidence_sources, ("local_index",))
        self.assertEqual(built.chains[0].status, "UNCERTAIN")

    def test_offline_overload_ambiguity_propagates_and_uses_farther_anchor(self) -> None:
        built = build_strict_result_chains(
            graph(business_result_graph("borrow")),
            continuity(),
            results(result("result_001", "n003")),
            blockchain_chain_id=1,
            offline_function_candidates=(
                offline_candidate("borrow", "borrow(uint256)"),
                offline_candidate("borrow", "borrow(address)"),
            ),
        )
        self.assertEqual(built.chains[0].anchor.node_id, "n001")
        self.assertIn("SELECTOR_RECOVERY_AMBIGUOUS", built.chains[0].uncertainties)
        self.assertIn("SELECTOR_RECOVERY_AMBIGUOUS", built.result_statuses[0].reasons)

    def test_missing_offline_source_propagates_and_uses_farther_anchor(self) -> None:
        built = build_strict_result_chains(
            graph(business_result_graph("borrow")),
            continuity(),
            results(result("result_001", "n003")),
            blockchain_chain_id=1,
        )
        self.assertEqual(built.chains[0].anchor.node_id, "n001")
        self.assertIn("SELECTOR_RECOVERY_SOURCE_MISSING", built.chains[0].uncertainties)
        self.assertIn("CLOSER_NON_ASSET_CALL_SELECTOR_MISSING", built.chains[0].uncertainties)

    def test_offline_effective_address_map_selects_candidate_and_anchor_address(self) -> None:
        implementation = addr("d")
        built = build_strict_result_chains(
            graph(business_result_graph("borrow")),
            continuity(),
            results(result("result_001", "n003")),
            blockchain_chain_id=1,
            offline_function_candidates=(
                offline_candidate("borrow", "borrow(uint256)", effective_address=implementation),
            ),
            offline_effective_address_map={addr("b"): implementation},
        )
        anchor = built.chains[0].anchor
        self.assertEqual(anchor.node_id, "n002")
        self.assertEqual(anchor.runtime_address, addr("b"))
        self.assertEqual(anchor.effective_address, implementation)
        self.assertEqual(anchor.address_resolution, "OFFLINE_EFFECTIVE_ADDRESS_MAP")

    def test_invalid_effective_address_map_blocks_ready(self) -> None:
        built = build_strict_result_chains(
            graph(business_result_graph("borrow")),
            continuity(),
            results(result("result_001", "n003")),
            blockchain_chain_id=1,
            offline_function_candidates=(offline_candidate("borrow", "borrow(uint256)"),),
            offline_effective_address_map={addr("b"): "invalid-address"},
        )
        chain = built.chains[0]
        self.assertEqual(chain.anchor.node_id, "n001")
        self.assertEqual(chain.status, "UNCERTAIN")
        self.assertIn(
            "OFFLINE_EFFECTIVE_ADDRESS_MAP_EFFECTIVE_ADDRESS_INVALID",
            chain.uncertainties,
        )

    def test_offline_candidate_for_standard_transfer_is_still_skipped(self) -> None:
        built = build_strict_result_chains(
            graph(business_result_graph("transfer")),
            continuity(),
            results(result("result_001", "n003")),
            blockchain_chain_id=1,
            offline_function_candidates=(
                offline_candidate("transfer", "transfer(address,uint256)"),
            ),
        )
        self.assertEqual(built.chains[0].anchor.node_id, "n001")
        self.assertEqual(built.chains[0].anchor.function_identity_source, "TRACE_NATIVE")

    def test_offline_candidate_order_does_not_change_strict_serialization(self) -> None:
        first_candidate = offline_candidate("borrow", "borrow(uint256)", evidence_source="b")
        second_candidate = offline_candidate("borrow", "borrow(address)", evidence_source="a")
        inputs = (
            graph(business_result_graph("borrow")),
            continuity(),
            results(result("result_001", "n003")),
        )
        first = build_strict_result_chains(
            *inputs,
            blockchain_chain_id=1,
            offline_function_candidates=(first_candidate, second_candidate),
        )
        second = build_strict_result_chains(
            *inputs,
            blockchain_chain_id=1,
            offline_function_candidates=(second_candidate, first_candidate),
        )
        self.assertEqual(serialize_strict_result_chains(first), serialize_strict_result_chains(second))

    def test_explicit_success_nearest_business_call_is_ready(self) -> None:
        built = build_strict_result_chains(
            graph(call_node(addr("a"), "0x11111111", call_status({"output": "0x01"}))),
            continuity(),
            results(result("result_001", "n001")),
            blockchain_chain_id=1,
        )
        chain = built.chains[0]
        self.assertEqual(chain.status, "READY")
        self.assertEqual(chain.anchor.node_id, "n001")
        self.assertEqual(chain.anchor.execution_status, "SUCCESS")
        self.assertTrue(chain.anchor.execution_success_evidence)
        self.assertEqual(chain.anchor.execution_status_uncertainties, ())

    def test_zero_length_output_success_is_usable_for_anchor(self) -> None:
        built = build_strict_result_chains(
            graph(call_node(addr("a"), "0x11111111", call_status({"output": "0x"}))),
            continuity(),
            results(result("result_001", "n001")),
            blockchain_chain_id=1,
        )
        self.assertEqual(built.chains[0].status, "READY")
        self.assertEqual(built.chains[0].anchor.execution_status_source, "CALLTRACER_OUTPUT")

    def test_unknown_call_is_not_a_formal_anchor(self) -> None:
        built = build_strict_result_chains(
            graph(call_node(addr("a"), "0x11111111", call_status({}))),
            continuity(),
            results(result("result_001", "n001")),
            blockchain_chain_id=1,
        )
        chain = built.chains[0]
        self.assertEqual(chain.status, "UNCERTAIN")
        self.assertEqual(chain.anchor.execution_status, "UNKNOWN")
        self.assertIn("CLOSER_NON_ASSET_CALL_STATUS_UNKNOWN", chain.uncertainties)

    def test_failed_call_is_not_a_formal_anchor(self) -> None:
        built = build_strict_result_chains(
            graph(call_node(addr("a"), "0x11111111", call_status({"error": "reverted"}))),
            continuity(),
            results(result("result_001", "n001")),
            blockchain_chain_id=1,
        )
        chain = built.chains[0]
        self.assertEqual(chain.status, "UNCERTAIN")
        self.assertEqual(chain.anchor.execution_status, "FAILED")
        self.assertIn("CLOSER_NON_ASSET_CALL_FAILED", chain.uncertainties)

    def test_closer_failed_call_uses_farther_success_but_remains_uncertain(self) -> None:
        built = build_strict_result_chains(
            graph(
                call_node(
                    addr("a"),
                    "0x11111111",
                    call_status({"output": "0x"}),
                    [call_node(addr("b"), "0x22222222", call_status({"error": "reverted"}))],
                )
            ),
            continuity(),
            results(result("result_001", "n002")),
            blockchain_chain_id=1,
        )
        chain = built.chains[0]
        self.assertEqual(chain.anchor.node_id, "n001")
        self.assertEqual(chain.anchor.execution_status, "SUCCESS")
        self.assertEqual(chain.status, "UNCERTAIN")
        self.assertIn("CLOSER_NON_ASSET_CALL_FAILED", chain.uncertainties)

    def test_closer_unknown_call_uses_farther_success_but_remains_uncertain(self) -> None:
        built = build_strict_result_chains(
            graph(
                call_node(
                    addr("a"),
                    "0x11111111",
                    call_status({"output": "0x"}),
                    [call_node(addr("b"), "0x22222222", call_status({}))],
                )
            ),
            continuity(),
            results(result("result_001", "n002")),
            blockchain_chain_id=1,
        )
        chain = built.chains[0]
        self.assertEqual(chain.anchor.node_id, "n001")
        self.assertEqual(chain.status, "UNCERTAIN")
        self.assertIn("CLOSER_NON_ASSET_CALL_STATUS_UNKNOWN", chain.uncertainties)

    def test_successful_standard_transfer_is_still_skipped(self) -> None:
        built = build_strict_result_chains(
            graph(
                call_node(
                    addr("a"),
                    "0x11111111",
                    call_status({"output": "0x"}),
                    [call_node(addr("b"), "0xa9059cbb", call_status({"output": "0x"}))],
                )
            ),
            continuity(),
            results(result("result_001", "n002")),
            blockchain_chain_id=1,
        )
        self.assertEqual(built.chains[0].anchor.node_id, "n001")
        self.assertEqual(built.chains[0].status, "READY")

    def test_legacy_call_tree_remains_uncertain_for_unreported_status(self) -> None:
        built = build_strict_result_chains(
            graph({"type": "call", "address": addr("a"), "function": "0x11111111", "children": []}),
            continuity(),
            results(result("result_001", "n001")),
            blockchain_chain_id=1,
        )
        self.assertEqual(built.chains[0].status, "UNCERTAIN")
        self.assertIn("CALL_SUCCESS_STATUS_NOT_REPORTED", built.chains[0].uncertainties)

    def test_offline_abi_identity_with_success_status_is_ready(self) -> None:
        built = build_strict_result_chains(
            graph(call_node(addr("b"), "borrow", call_status({"output": "0x"}))),
            continuity(),
            results(result("result_001", "n001")),
            blockchain_chain_id=1,
            offline_function_candidates=(offline_candidate("borrow", "borrow(uint256)"),),
        )
        self.assertEqual(built.chains[0].anchor.function_identity_source, "OFFLINE_ABI_UNIQUE")
        self.assertEqual(built.chains[0].status, "READY")

    def test_anchor_status_evidence_serialization_is_stable(self) -> None:
        inputs = (
            graph(call_node(addr("a"), "0x11111111", call_status({"output": "0x"}))),
            continuity(),
            results(result("result_001", "n001")),
        )
        first = build_strict_result_chains(*inputs, blockchain_chain_id=1)
        second = build_strict_result_chains(*inputs, blockchain_chain_id=1)
        anchor = first.chains[0].anchor.to_dict()
        self.assertEqual(anchor["execution_status"], "SUCCESS")
        self.assertEqual(anchor["execution_status_source"], "CALLTRACER_OUTPUT")
        self.assertTrue(anchor["execution_success_evidence"])
        self.assertEqual(anchor["execution_status_uncertainties"], [])
        self.assertEqual(serialize_strict_result_chains(first), serialize_strict_result_chains(second))


if __name__ == "__main__":
    unittest.main()

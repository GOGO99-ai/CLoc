from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from src.cloc_v2.asset_continuity import build_asset_continuity, serialize_asset_continuity
from src.cloc_v2.asset_ledger_models import AssetTransferFact
from src.cloc_v2.observed_asset_ledger import runtime_annotations_sha256
from src.cloc_v2.asset_ledger_models import ObservedAssetLedger
from src.cloc_v2.runtime_semantics import build_runtime_annotations, transaction_graph_sha256
from src.cloc_v2.transaction_graph import build_transaction_graph


def address(char: str) -> dict[str, str]:
    original = "0x" + char * 40
    return {"original": original, "normalized": original.lower()}


def fact(index: int, sender: dict[str, str] | None, receiver: dict[str, str] | None, amount: str, *, token: str = "t", status: str = "COMPLETE") -> AssetTransferFact:
    return AssetTransferFact(
        fact_id=f"asset_f{index:03d}", node_id=f"n{index:03d}", trace_index=index, trace_path=f"0/{index}", transfer_kind="TRANSFER", asset_type="TOKEN", asset_address=address(token), operator_address=sender, from_address=sender, to_address=receiver, amount_raw=amount, status=status, evidence_level="OBSERVED", reason="test", trace_refs=(f"0/{index}",), uncertainties=(),
    )


def context(facts: tuple[AssetTransferFact, ...]) -> tuple[object, object, ObservedAssetLedger]:
    graph = build_transaction_graph({"type": "call", "address": address("r")["original"], "function": "root", "children": []})
    annotations = build_runtime_annotations(graph)
    ledger = ObservedAssetLedger(transaction_graph_sha256=transaction_graph_sha256(graph), runtime_annotations_sha256=runtime_annotations_sha256(annotations), transfer_facts=facts, address_flow_summaries=())
    return graph, annotations, ledger


class AssetContinuityTests(unittest.TestCase):
    def test_exact_fifo_and_amount_conservation(self) -> None:
        a, b, c = address("a"), address("b"), address("c")
        graph, annotations, ledger = context((fact(1, a, b, "100"), fact(2, b, c, "100")))
        result = build_asset_continuity(graph, annotations, ledger)
        self.assertEqual(len(result.asset_edges), 1)
        edge = result.asset_edges[0]
        self.assertEqual((edge.src_fact_id, edge.dst_fact_id, edge.allocated_amount_raw, edge.match_type), ("asset_f001", "asset_f002", "100", "EXACT"))
        self.assertEqual(edge.evidence_level, "STRONG_ASSOCIATION")

    def test_split_and_merged_fifo_allocations(self) -> None:
        a, b, c, d, e = address("a"), address("b"), address("c"), address("d"), address("e")
        graph, annotations, ledger = context((fact(1, a, b, "100"), fact(2, b, c, "40"), fact(3, b, d, "60"), fact(4, a, b, "30"), fact(5, a, b, "70"), fact(6, b, e, "100")))
        result = build_asset_continuity(graph, annotations, ledger)
        self.assertEqual([edge.allocated_amount_raw for edge in result.asset_edges], ["40", "60", "30", "70"])
        self.assertEqual([edge.dst_fact_id for edge in result.asset_edges[-2:]], ["asset_f006", "asset_f006"])
        self.assertTrue(all(edge.match_type == "MERGED" for edge in result.asset_edges[-2:]))

    def test_unmatched_future_and_different_asset_holder_do_not_match(self) -> None:
        a, b, c, d = address("a"), address("b"), address("c"), address("d")
        graph, annotations, ledger = context((fact(1, b, c, "50"), fact(2, a, b, "50"), fact(3, a, d, "50", token="u")))
        result = build_asset_continuity(graph, annotations, ledger)
        self.assertEqual(result.asset_edges, ())
        self.assertEqual(len(result.unmatched_outflows), 3)
        self.assertEqual(len(result.remaining_inflows), 3)

    def test_incomplete_and_self_transfer_do_not_form_edges(self) -> None:
        a, b = address("a"), address("b")
        incomplete = fact(1, a, b, "10", status="INCOMPLETE")
        self_transfer = fact(2, a, a, "20")
        graph, annotations, ledger = context((incomplete, self_transfer))
        result = build_asset_continuity(graph, annotations, ledger)
        self.assertEqual(result.asset_edges, ())
        self.assertEqual(len(result.self_transfers), 1)
        self.assertEqual(result.self_transfers[0].fact_id, "asset_f002")

    def test_transfer_from_operator_is_not_used_as_holder(self) -> None:
        owner, holder, recipient, spender = address("a"), address("b"), address("c"), address("d")
        incoming = fact(1, owner, holder, "10")
        outgoing = replace(fact(2, holder, recipient, "10"), transfer_kind="TRANSFER_FROM", operator_address=spender)
        graph, annotations, ledger = context((incoming, outgoing))
        result = build_asset_continuity(graph, annotations, ledger)
        self.assertEqual(result.asset_edges[0].via_address["normalized"], holder["normalized"])

    def test_determinism_input_immutability_and_hash_mismatch(self) -> None:
        a, b, c = address("a"), address("b"), address("c")
        graph, annotations, ledger = context((fact(1, a, b, "999999999999999999999999999999"), fact(2, b, c, "999999999999999999999999999999")))
        graph_before, annotations_before, ledger_before = copy.deepcopy(graph.to_dict()), copy.deepcopy(annotations.to_dict()), copy.deepcopy(ledger.to_dict())
        first, second = build_asset_continuity(graph, annotations, ledger), build_asset_continuity(graph, annotations, ledger)
        self.assertEqual(serialize_asset_continuity(first).encode(), serialize_asset_continuity(second).encode())
        self.assertEqual(graph.to_dict(), graph_before)
        self.assertEqual(annotations.to_dict(), annotations_before)
        self.assertEqual(ledger.to_dict(), ledger_before)
        with self.assertRaisesRegex(ValueError, "Observed asset ledger"):
            build_asset_continuity(graph, annotations, replace(ledger, transaction_graph_sha256="bad"))

    def test_output_has_only_asset_edges_and_no_roles(self) -> None:
        a, b, c = address("a"), address("b"), address("c")
        graph, annotations, ledger = context((fact(1, a, b, "1"), fact(2, b, c, "1")))
        text = serialize_asset_continuity(build_asset_continuity(graph, annotations, ledger)).upper()
        for forbidden in ("ATTACKER", "VICTIM", "ROOT_CAUSE", '"RESULT"', '"PARAM"', '"STATE"'):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.cloc_v2.observed_asset_ledger import (
    build_observed_asset_ledger,
    load_runtime_annotations,
    runtime_annotations_sha256,
    serialize_observed_asset_ledger,
    write_observed_asset_ledger,
)
from src.cloc_v2.runtime_semantics import build_runtime_annotations, load_transaction_graph
from src.cloc_v2.transaction_graph import build_transaction_graph, write_transaction_graph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "runs"
    / "eth"
    / "0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299"
    / "transaction_graph_v1.json"
)
REAL_ANNOTATIONS_PATH = REAL_GRAPH_PATH.with_name("runtime_annotations_v1.json")


def address(character: str) -> str:
    return "0x" + character * 40


def build_graph(children: list[dict], *, root_address: str | None = None) -> object:
    return build_transaction_graph(
        {
            "type": "call",
            "address": root_address or address("a"),
            "function": "root",
            "children": children,
        }
    )


def build_ledger(children: list[dict], *, root_address: str | None = None) -> object:
    graph = build_graph(children, root_address=root_address)
    annotations = build_runtime_annotations(graph)
    return build_observed_asset_ledger(graph, annotations)


class ObservedAssetLedgerTests(unittest.TestCase):
    def test_transfer_generates_one_complete_observed_fact(self) -> None:
        ledger = build_ledger(
            [{"type": "call", "address": address("b"), "function": "transfer", "params": f"{address('c')}, 123456789012345678901234567890", "children": []}]
        )
        fact = ledger.transfer_facts[0]
        self.assertEqual(fact.fact_id, "asset_f001")
        self.assertEqual(fact.transfer_kind, "TRANSFER")
        self.assertEqual(fact.asset_type, "TOKEN")
        self.assertEqual(fact.asset_address["normalized"], address("b"))
        self.assertEqual(fact.operator_address["normalized"], address("a"))
        self.assertEqual(fact.from_address["normalized"], address("a"))
        self.assertEqual(fact.to_address["normalized"], address("c"))
        self.assertEqual(fact.amount_raw, "123456789012345678901234567890")
        self.assertEqual(fact.status, "COMPLETE")
        self.assertIn("from_address is derived from parent call context", fact.uncertainties)

    def test_transfer_from_dual_action_annotation_generates_exactly_one_fact(self) -> None:
        ledger = build_ledger(
            [{"type": "call", "address": address("d"), "function": "transferFrom", "params": f"{address('b')}, {address('c')}, 5", "children": []}]
        )
        self.assertEqual(len(ledger.transfer_facts), 1)
        fact = ledger.transfer_facts[0]
        self.assertEqual(fact.transfer_kind, "TRANSFER_FROM")
        self.assertEqual(fact.operator_address["normalized"], address("a"))
        self.assertEqual(fact.from_address["normalized"], address("b"))
        self.assertEqual(fact.to_address["normalized"], address("c"))

    def test_safe_transfer_from_generates_one_complete_observed_fact(self) -> None:
        ledger = build_ledger(
            [{"type": "call", "address": address("d"), "function": "safeTransferFrom", "params": f"{address('b')}, {address('c')}, 5110, 0x", "children": []}]
        )
        self.assertEqual(len(ledger.transfer_facts), 1)
        fact = ledger.transfer_facts[0]
        self.assertEqual(fact.transfer_kind, "TRANSFER_FROM")
        self.assertEqual(fact.from_address["normalized"], address("b"))
        self.assertEqual(fact.to_address["normalized"], address("c"))
        self.assertEqual(fact.amount_raw, "5110")
        self.assertEqual(fact.status, "COMPLETE")

    def test_explicit_native_value_is_a_separate_observed_fact(self) -> None:
        ledger = build_ledger(
            [{"type": "call", "address": address("b"), "function": "receive", "params": "", "value": "0x2a", "children": []}]
        )
        fact = ledger.transfer_facts[0]
        self.assertEqual(fact.transfer_kind, "NATIVE_VALUE")
        self.assertEqual(fact.asset_type, "NATIVE")
        self.assertIsNone(fact.asset_address)
        self.assertEqual(fact.amount_raw, "42")
        self.assertEqual(fact.status, "COMPLETE")

    def test_empty_or_function_text_value_does_not_create_native_fact(self) -> None:
        empty_value = build_ledger(
            [{"type": "call", "address": address("b"), "function": "receive", "params": "", "value": "0", "children": []}]
        )
        embedded_value = build_ledger(
            [{"type": "call", "address": address("b"), "function": "deposit{value: 99}", "params": "", "value": None, "children": []}]
        )
        self.assertEqual(empty_value.transfer_facts, ())
        self.assertEqual(embedded_value.transfer_facts, ())

    def test_missing_transfer_fields_remain_incomplete_and_excluded_from_flows(self) -> None:
        ledger = build_ledger(
            [{"type": "call", "address": address("b"), "function": "transfer", "params": f"{address('c')}, not-a-number", "children": []}]
        )
        fact = ledger.transfer_facts[0]
        self.assertEqual(fact.status, "INCOMPLETE")
        self.assertIsNone(fact.amount_raw)
        self.assertEqual(ledger.address_flow_summaries, ())

    def test_missing_asset_address_is_incomplete(self) -> None:
        ledger = build_ledger(
            [{"type": "call", "address": "not-an-address", "function": "transfer", "params": f"{address('c')}, 7", "children": []}]
        )
        fact = ledger.transfer_facts[0]
        self.assertEqual(fact.status, "INCOMPLETE")
        self.assertIsNone(fact.asset_address)
        self.assertIn("asset_address is unavailable from observed trace fields", fact.uncertainties)

    def test_flow_summaries_keep_tokens_separate_and_can_be_negative(self) -> None:
        ledger = build_ledger(
            [
                {"type": "call", "address": address("b"), "function": "transfer", "params": f"{address('c')}, 10", "children": []},
                {"type": "call", "address": address("d"), "function": "transfer", "params": f"{address('c')}, 20", "children": []},
            ]
        )
        summaries = [summary.to_dict() for summary in ledger.address_flow_summaries]
        root_token_b = next(item for item in summaries if item["asset_address"]["normalized"] == address("b") and item["address"]["normalized"] == address("a"))
        recipient_token_d = next(item for item in summaries if item["asset_address"]["normalized"] == address("d") and item["address"]["normalized"] == address("c"))
        self.assertEqual(root_token_b["observed_delta_raw"], "-10")
        self.assertEqual(recipient_token_d["incoming_amount_raw"], "20")
        self.assertEqual(root_token_b["completeness"], "PARTIAL")
        self.assertEqual(len({item["asset_address"]["normalized"] for item in summaries}), 2)

    def test_fact_ids_and_serialization_are_stable_and_inputs_are_unchanged(self) -> None:
        graph = build_graph(
            [
                {"type": "call", "address": address("b"), "function": "transfer", "params": f"{address('c')}, 1", "children": []},
                {"type": "call", "address": address("d"), "function": "transferFrom", "params": f"{address('a')}, {address('c')}, 2", "children": []},
            ]
        )
        annotations = build_runtime_annotations(graph)
        graph_before = copy.deepcopy(graph.to_dict())
        annotations_before = copy.deepcopy(annotations.to_dict())
        first = build_observed_asset_ledger(graph, annotations)
        second = build_observed_asset_ledger(graph, annotations)
        self.assertEqual([fact.fact_id for fact in first.transfer_facts], ["asset_f001", "asset_f002"])
        self.assertEqual(serialize_observed_asset_ledger(first).encode("utf-8"), serialize_observed_asset_ledger(second).encode("utf-8"))
        self.assertEqual(graph.to_dict(), graph_before)
        self.assertEqual(annotations.to_dict(), annotations_before)

    def test_hash_mismatch_fails_before_ledger_output(self) -> None:
        graph = build_graph([])
        annotations = build_runtime_annotations(graph)
        mismatched = replace(annotations, transaction_graph_sha256="not-the-graph-hash")
        with self.assertRaisesRegex(ValueError, "transaction_graph_sha256"):
            build_observed_asset_ledger(graph, mismatched)

    def test_output_contains_no_causal_edges_or_forbidden_roles(self) -> None:
        ledger = build_ledger(
            [{"type": "call", "address": address("b"), "function": "transfer", "params": f"{address('c')}, 1", "children": []}]
        )
        payload = ledger.to_dict()
        self.assertNotIn("edges", payload)
        rendered = json.dumps(payload).upper()
        for forbidden in ("ATTACKER", "VICTIM", "ROOT_CAUSE", "VULNERABLE", "MALICIOUS", '"RESULT"', '"PARAM"', '"STATE"'):
            self.assertNotIn(forbidden, rendered)

    def test_real_graph_and_annotations_are_read_only_compatible_when_available(self) -> None:
        if not REAL_GRAPH_PATH.exists() or not REAL_ANNOTATIONS_PATH.exists():
            self.skipTest("Real graph or runtime annotations are unavailable in this checkout")
        graph_before = REAL_GRAPH_PATH.read_bytes()
        annotations_before = REAL_ANNOTATIONS_PATH.read_bytes()
        graph = load_transaction_graph(REAL_GRAPH_PATH)
        annotations = load_runtime_annotations(REAL_ANNOTATIONS_PATH)
        ledger = build_observed_asset_ledger(graph, annotations)
        self.assertTrue(ledger.transaction_graph_sha256)
        self.assertEqual(ledger.runtime_annotations_sha256, runtime_annotations_sha256(annotations))
        self.assertEqual(REAL_GRAPH_PATH.read_bytes(), graph_before)
        self.assertEqual(REAL_ANNOTATIONS_PATH.read_bytes(), annotations_before)

    def test_round_trip_writer_keeps_inputs_separate(self) -> None:
        graph = build_graph([{"type": "call", "address": address("b"), "function": "transfer", "params": f"{address('c')}, 1", "children": []}])
        annotations = build_runtime_annotations(graph)
        ledger = build_observed_asset_ledger(graph, annotations)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "observed_asset_ledger_v1.json"
            write_observed_asset_ledger(ledger, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), serialize_observed_asset_ledger(ledger))


if __name__ == "__main__":
    unittest.main()

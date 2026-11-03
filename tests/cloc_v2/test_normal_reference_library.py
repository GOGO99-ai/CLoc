from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from src.cloc_v2.asset_ledger_models import AssetTransferFact
from src.cloc_v2.normal_reference_library import _unwrap_call_tree, build_reference_record, flatten_transaction_hashes, load_faultless_candidates, retrieve_similar_references


def addr(character: str) -> str:
    return "0x" + character * 40


class NormalReferenceLibraryTests(unittest.TestCase):
    def test_nested_hashes_are_flattened_and_deduplicated(self) -> None:
        self.assertEqual(flatten_transaction_hashes([["0xAA", ["0xbb", "0xaa"]], "0xCC"]), ["0xaa", "0xbb", "0xcc"])

    def test_candidate_loading_ignores_ground_truth_fields(self) -> None:
        payload = {"platform": "Ethereum", "cause": "ignored", "fault": {"location": ["ignored"], "detection": {"link": "ignored"}}, "faultless": {"transaction_hash": [["0xAA"], "0xaa"]}}
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_path = Path(temporary_directory) / "case.json"
            case_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_faultless_candidates(case_path), ("Ethereum", ["0xaa"]))

    def test_trace_node_payload_is_converted_to_a_call_tree_dict(self) -> None:
        class TraceNodeLike:
            def to_dict(self) -> dict:
                return {"type": "call", "address": addr("a"), "function": "entry", "children": []}

        tree = _unwrap_call_tree({"call_tree": TraceNodeLike()})
        self.assertEqual(tree["address"], addr("a"))

    def test_selector_transfer_generates_asset_edge(self) -> None:
        recipient = addr("c")
        calldata = "0xa9059cbb" + "0" * 24 + recipient[2:] + f"{42:064x}"
        record = build_reference_record("normal_001", "0x1", "Ethereum", {"type": "call", "address": addr("a"), "function": "entry", "children": [{"type": "call", "address": addr("b"), "function": "0xa9059cbb", "params": calldata, "call_type": "CALL", "children": []}]}, cache_used=True, trace_path="cache:trace.txt")
        edge = record["evidence_graph"]["asset_edges"][0]
        self.assertEqual((edge["transfer_kind"], edge["src"], edge["dst"], edge["amount_raw"]), ("TRANSFER", addr("a"), recipient, "42"))
        self.assertIn("0xa9059cbb", record["retrieval_features"]["functions_or_selectors"])

    def test_selector_transfer_from_generates_asset_edge(self) -> None:
        sender, recipient = addr("c"), addr("d")
        calldata = "0x23b872dd" + "0" * 24 + sender[2:] + "0" * 24 + recipient[2:] + f"{7:064x}"
        record = build_reference_record("normal_001", "0x1", "Ethereum", {"type": "call", "address": addr("a"), "function": "entry", "children": [{"type": "call", "address": addr("b"), "function": "0x23b872dd", "params": calldata, "call_type": "CALL", "children": []}]}, cache_used=True, trace_path="cache:trace.txt")
        edge = record["evidence_graph"]["asset_edges"][0]
        self.assertEqual((edge["transfer_kind"], edge["src"], edge["dst"], edge["amount_raw"]), ("TRANSFER_FROM", sender, recipient, "7"))

    def test_native_value_short_calldata_and_revert_handling(self) -> None:
        tree = {"type": "call", "address": addr("a"), "function": "entry", "children": [
            {"type": "call", "address": addr("b"), "function": "0x12345678{value: 9}", "params": "0x12345678", "call_type": "CALL", "children": []},
            {"type": "call", "address": addr("b"), "function": "0xa9059cbb", "params": "0xa9059cbb00", "call_type": "CALL", "children": []},
            {"type": "call", "address": addr("b"), "function": "0xa9059cbb", "params": "0xa9059cbb" + "0" * 128, "call_type": "CALL", "status": "REVERTED", "children": []},
        ]}
        record = build_reference_record("normal_001", "0x1", "Ethereum", tree, cache_used=True, trace_path="cache:trace.txt")
        edges = record["evidence_graph"]["asset_edges"]
        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0]["transfer_kind"], edges[0]["amount_raw"]), ("NATIVE_VALUE", "9"))

    def test_malformed_selector_calldata_never_creates_token_asset_edge(self) -> None:
        recipient = addr("c")
        amount = f"{42:064x}"
        valid = "0xa9059cbb" + "0" * 24 + recipient[2:] + amount
        malformed = {
            "short": valid[:-2],
            "long": valid + "00",
            "noncanonical_address_padding": "0xa9059cbb" + "1" * 24 + recipient[2:] + amount,
        }
        for name, calldata in malformed.items():
            with self.subTest(name=name):
                tree = {
                    "type": "call",
                    "address": addr("a"),
                    "function": "entry",
                    "children": [{
                        "type": "call",
                        "address": addr("b"),
                        "function": "0xa9059cbb",
                        "params": calldata,
                        "call_type": "CALL",
                        "children": [],
                    }],
                }
                record = build_reference_record(
                    "normal_001", "0x1", "Ethereum", tree,
                    cache_used=True, trace_path="cache:trace.txt",
                )
                self.assertEqual(record["evidence_graph"]["asset_edges"], [])

    def test_incomplete_ledger_fact_does_not_create_asset_edge(self) -> None:
        incomplete = AssetTransferFact(
            fact_id="asset_f001",
            node_id="n001",
            trace_index=1,
            trace_path="0",
            transfer_kind="TRANSFER",
            asset_type="TOKEN",
            asset_address={"original": addr("b"), "normalized": addr("b")},
            operator_address={"original": addr("a"), "normalized": addr("a")},
            from_address={"original": addr("a"), "normalized": addr("a")},
            to_address=None,
            amount_raw="1",
            status="INCOMPLETE",
            evidence_level="OBSERVED",
            reason="test incomplete fact",
            trace_refs=("0",),
            uncertainties=("TO_ADDRESS_MISSING",),
        )
        tree = {"type": "call", "address": addr("a"), "function": "entry", "children": []}
        with patch(
            "src.cloc_v2.normal_reference_library.build_observed_asset_ledger",
            return_value=type("Ledger", (), {"transfer_facts": (incomplete,)})(),
        ):
            record = build_reference_record(
                "normal_001", "0x1", "Ethereum", tree,
                cache_used=True, trace_path="cache:trace.txt",
            )
        self.assertEqual(record["evidence_graph"]["asset_edges"], [])

    def test_explicit_revert_filters_otherwise_valid_transfer(self) -> None:
        recipient = addr("c")
        calldata = "0xa9059cbb" + "0" * 24 + recipient[2:] + f"{42:064x}"
        tree = {
            "type": "call",
            "address": addr("a"),
            "function": "entry",
            "children": [{
                "type": "call",
                "address": addr("b"),
                "function": "0xa9059cbb",
                "params": calldata,
                "call_type": "CALL",
                "status": "REVERTED",
                "children": [],
            }],
        }
        record = build_reference_record(
            "normal_001", "0x1", "Ethereum", tree,
            cache_used=True, trace_path="cache:trace.txt",
        )
        self.assertEqual(record["evidence_graph"]["asset_edges"], [])

    def test_decoded_transfer_adds_its_standard_selector_to_features(self) -> None:
        record = build_reference_record("normal_001", "0x1", "Ethereum", {"type": "call", "address": addr("a"), "function": "transfer", "params": f"{addr('b')}, 1", "children": []}, cache_used=True, trace_path="cache:trace.txt")
        self.assertIn("transfer", record["retrieval_features"]["functions_or_selectors"])
        self.assertIn("0xa9059cbb", record["retrieval_features"]["functions_or_selectors"])

    def test_graph_only_has_call_and_asset_edges_and_limits_chains(self) -> None:
        tree = {"type": "call", "address": addr("a"), "function": "entry", "children": [
            {"type": "call", "address": addr("b"), "function": "transfer", "params": f"{addr('c')}, 1", "children": []},
            {"type": "call", "address": addr("b"), "function": "transferFrom", "params": f"{addr('c')}, {addr('d')}, 2", "children": []},
        ]}
        record = build_reference_record("normal_001", "0x1", "Ethereum", tree, cache_used=True, trace_path="cache:trace.json")
        graph = record["evidence_graph"]
        self.assertTrue(all(edge["edge_type"] == "CALL" for edge in graph["call_edges"]))
        self.assertTrue(all(edge["edge_type"] == "ASSET" for edge in graph["asset_edges"]))
        self.assertLessEqual(len(record["reference_chains"]), 3)

    def test_retrieval_order_is_deterministic(self) -> None:
        query = {"transaction_identity": {"entry_selector": "0x11111111"}, "retrieval_features": {"addresses": [addr("a")], "functions_or_selectors": ["f"], "node_count": 4}}
        library = [
            {"reference_id": "b", "transaction_identity": {"entry_selector": None}, "retrieval_features": {"addresses": [addr("a")], "functions_or_selectors": ["f"], "node_count": 4, "asset_edge_count": 0}, "reference_chains": [{"fallback_reason": "NO_ASSET_EDGE"}]},
            {"reference_id": "a", "transaction_identity": {"entry_selector": "0x11111111"}, "retrieval_features": {"addresses": [], "functions_or_selectors": [], "node_count": 100, "asset_edge_count": 2}, "reference_chains": [{"result_asset_edge_id": "asset_001"}]},
        ]
        results = retrieve_similar_references(query, library)
        self.assertEqual([item["reference_id"] for item in results], ["a", "b"])
        self.assertEqual([item["rank"] for item in results], [1, 2])
        self.assertEqual((results[0]["asset_edge_count"], results[0]["reference_chain_fallback"]), (2, False))

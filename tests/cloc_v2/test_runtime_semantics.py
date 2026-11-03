from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.cloc_v2.calltracer_adapter import adapt_calltracer_call_tree
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.runtime_semantics import (
    ALLOWED_ACTION_TYPES,
    build_runtime_annotations,
    load_transaction_graph,
    serialize_runtime_annotations,
    split_top_level_arguments,
    write_runtime_annotations,
)
from src.cloc_v2.transaction_graph import build_transaction_graph, serialize_transaction_graph, write_transaction_graph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "runs"
    / "eth"
    / "0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299"
    / "transaction_graph_v1.json"
)


def address(character: str) -> str:
    return "0x" + character * 40


def make_graph(children: list[dict]) -> object:
    return build_transaction_graph({"type": "call", "address": address("a"), "function": "root", "children": children})


def annotation_for(result: object, node_id: str) -> dict:
    return next(item.to_dict() for item in result.node_annotations if item.node_id == node_id)


def abi_address_word(value: str, *, padding: str = "0" * 24) -> str:
    return padding + value[2:].lower()


def abi_uint256_word(value: int) -> str:
    return f"{value:064x}"


def transfer_calldata(to_address: str, amount: int) -> str:
    return "0xa9059cbb" + abi_address_word(to_address) + abi_uint256_word(amount)


def transfer_from_calldata(from_address: str, to_address: str, amount: int) -> str:
    return (
        "0x23b872dd"
        + abi_address_word(from_address)
        + abi_address_word(to_address)
        + abi_uint256_word(amount)
    )


class RuntimeSemanticsTests(unittest.TestCase):
    def test_transfer_selector_decodes_exact_calltracer_calldata(self) -> None:
        recipient = address("C")
        graph = make_graph(
            [
                {
                    "type": "call",
                    "from": address("a"),
                    "address": address("b"),
                    "function": "0xa9059cbb",
                    "params": transfer_calldata(recipient, 2**255 + 7),
                    "children": [],
                }
            ]
        )
        annotation = annotation_for(build_runtime_annotations(graph), "n002")
        self.assertEqual(annotation["action_types"], ["TOKEN_TRANSFER"])
        self.assertEqual(annotation["observed_operands"]["to_address"]["normalized"], recipient.lower())
        self.assertEqual(annotation["observed_operands"]["amount_raw"], str(2**255 + 7))
        self.assertEqual(annotation["annotation_evidence"][0]["rule_id"], "SELECTOR_EXACT_ERC20_TRANSFER")

    def test_transfer_from_selector_decodes_exact_calltracer_calldata(self) -> None:
        sender, recipient = address("B"), address("C")
        graph = make_graph(
            [
                {
                    "type": "call",
                    "from": address("a"),
                    "address": address("d"),
                    "function": "0x23b872dd",
                    "params": transfer_from_calldata(sender, recipient, 200000000),
                    "children": [],
                }
            ]
        )
        annotation = annotation_for(build_runtime_annotations(graph), "n002")
        operands = annotation["observed_operands"]
        self.assertEqual(annotation["action_types"], ["TOKEN_TRANSFER", "TOKEN_TRANSFER_FROM"])
        self.assertEqual(operands["from_address"]["normalized"], sender.lower())
        self.assertEqual(operands["to_address"]["normalized"], recipient.lower())
        self.assertEqual(operands["amount_raw"], "200000000")
        self.assertEqual(annotation["annotation_evidence"][0]["rule_id"], "SELECTOR_EXACT_ERC20_TRANSFER_FROM")

    def test_calltracer_adapter_to_asset_ledger_produces_complete_token_fact(self) -> None:
        recipient = address("c")
        raw_frame = {
            "type": "CALL",
            "from": address("f"),
            "to": address("a"),
            "input": "0x11111111",
            "output": "0x",
            "calls": [
                {
                    "type": "CALL",
                    "from": address("a"),
                    "to": address("b"),
                    "input": transfer_calldata(recipient, 123456789),
                    "output": "0x" + "0" * 63 + "1",
                    "calls": [],
                }
            ],
        }
        graph = build_transaction_graph(adapt_calltracer_call_tree(raw_frame))
        annotations = build_runtime_annotations(graph)
        ledger = build_observed_asset_ledger(graph, annotations)
        self.assertEqual(len(ledger.transfer_facts), 1)
        fact = ledger.transfer_facts[0]
        self.assertEqual(fact.status, "COMPLETE")
        self.assertEqual(fact.asset_type, "TOKEN")
        self.assertEqual(fact.asset_address["normalized"], address("b"))
        self.assertEqual(fact.from_address["normalized"], address("a"))
        self.assertEqual(fact.to_address["normalized"], recipient)
        self.assertEqual(fact.amount_raw, "123456789")

    def test_malformed_selector_calldata_is_safely_rejected(self) -> None:
        valid = transfer_calldata(address("c"), 9)
        malformed = {
            "truncated": valid[:-2],
            "overlong": valid + "00",
            "odd_length": valid[:-1],
            "non_hex": valid[:-1] + "z",
            "bad_address_padding": "0xa9059cbb" + abi_address_word(address("c"), padding="1" + "0" * 23) + abi_uint256_word(9),
            "selector_mismatch": "0x23b872dd" + valid[10:],
        }
        expected = {
            "truncated": "ERC20_CALLDATA_LENGTH_INVALID",
            "overlong": "ERC20_CALLDATA_LENGTH_INVALID",
            "odd_length": "ERC20_CALLDATA_ODD_HEX_LENGTH",
            "non_hex": "ERC20_CALLDATA_INVALID_HEX",
            "bad_address_padding": "ERC20_CALLDATA_ADDRESS_PADDING_INVALID",
            "selector_mismatch": "ERC20_CALLDATA_SELECTOR_MISMATCH",
        }
        for case, calldata in malformed.items():
            with self.subTest(case=case):
                graph = make_graph(
                    [{"type": "call", "address": address("b"), "function": "0xa9059cbb", "params": calldata, "children": []}]
                )
                annotations = build_runtime_annotations(graph)
                annotation = annotation_for(annotations, "n002")
                self.assertIn(expected[case], annotation["uncertainties"])
                self.assertIsNone(annotation["observed_operands"]["to_address"])
                self.assertIsNone(annotation["observed_operands"]["amount_raw"])
                fact = build_observed_asset_ledger(graph, annotations).transfer_facts[0]
                self.assertEqual(fact.status, "INCOMPLETE")
                rendered = serialize_runtime_annotations(annotations)
                self.assertNotIn(calldata, rendered)

    def test_function_name_and_selector_conflict_is_not_silently_resolved(self) -> None:
        graph = make_graph(
            [{"type": "call", "address": address("b"), "function": "transfer", "params": f"{address('c')}, 1", "children": []}]
        )
        conflicting_node = replace(graph.nodes[1], function_selector="0x23b872dd")
        conflicting_graph = replace(graph, nodes=(graph.nodes[0], conflicting_node))
        annotation = annotation_for(build_runtime_annotations(conflicting_graph), "n002")
        self.assertEqual(annotation["action_types"], [])
        self.assertEqual(annotation["observed_operands"], {})
        self.assertIn("FUNCTION_NAME_SELECTOR_CONFLICT", annotation["uncertainties"])

    def test_transfer_extracts_address_and_large_integer_without_float(self) -> None:
        graph = make_graph(
            [{"type": "call", "address": address("b"), "function": "transfer", "params": f"{address('C')}, 123456789012345678901234567890 [1.234e29]", "children": []}]
        )
        result = build_runtime_annotations(graph)
        annotation = annotation_for(result, "n002")
        self.assertEqual(annotation["action_types"], ["TOKEN_TRANSFER"])
        self.assertEqual(annotation["observed_operands"]["to_address"]["original"], address("C"))
        self.assertEqual(annotation["observed_operands"]["to_address"]["normalized"], address("c"))
        self.assertEqual(annotation["observed_operands"]["amount_raw"], "123456789012345678901234567890")
        self.assertIsInstance(annotation["observed_operands"]["amount_raw"], str)

    def test_transfer_from_has_two_neutral_action_types_and_standard_operands(self) -> None:
        graph = make_graph(
            [{"type": "call", "address": address("a"), "function": "transferFrom", "params": f"{address('b')}, {address('C')}, 200000000 [2e8]", "children": []}]
        )
        annotation = annotation_for(build_runtime_annotations(graph), "n002")
        self.assertEqual(annotation["action_types"], ["TOKEN_TRANSFER", "TOKEN_TRANSFER_FROM"])
        self.assertEqual(annotation["observed_operands"]["from_address"]["normalized"], address("b"))
        self.assertEqual(annotation["observed_operands"]["to_address"]["normalized"], address("c"))
        self.assertEqual(annotation["observed_operands"]["amount_raw"], "200000000")

    def test_safe_transfer_from_supports_erc721_and_erc1155_argument_layouts(self) -> None:
        graph = make_graph(
            [
                {"type": "call", "address": address("a"), "function": "safeTransferFrom", "params": f"{address('b')}, {address('c')}, 5110, 0x", "children": []},
                {"type": "call", "address": address("d"), "function": "safeTransferFrom", "params": f"{address('b')}, {address('c')}, 77, 5, 0x", "children": []},
            ]
        )
        annotations = build_runtime_annotations(graph)
        erc721 = annotation_for(annotations, "n002")
        erc1155 = annotation_for(annotations, "n003")
        self.assertEqual(erc721["action_types"], ["TOKEN_TRANSFER", "TOKEN_TRANSFER_FROM"])
        self.assertEqual(erc721["observed_operands"]["amount_raw"], "5110")
        self.assertEqual(erc1155["observed_operands"]["amount_raw"], "5")
        self.assertEqual(erc1155["observed_operands"]["to_address"]["normalized"], address("c"))

    def test_approve_and_balance_of_extract_standard_parameters(self) -> None:
        graph = make_graph(
            [
                {"type": "call", "address": address("a"), "function": "approve", "params": f"{address('B')}, 999999999999999999999", "children": []},
                {"type": "call", "address": address("a"), "function": "balanceOf", "params": address("C"), "children": []},
            ]
        )
        result = build_runtime_annotations(graph)
        approve = annotation_for(result, "n002")["observed_operands"]
        balance = annotation_for(result, "n003")["observed_operands"]
        self.assertEqual(approve["spender_address"]["normalized"], address("b"))
        self.assertEqual(approve["amount_raw"], "999999999999999999999")
        self.assertEqual(balance["account_address"]["normalized"], address("c"))

    def test_nested_arrays_are_not_split_as_top_level_arguments(self) -> None:
        raw = f"[{address('a')}, {address('b')}], 200000000 [2e8], 0xdeadbeef"
        self.assertEqual(split_top_level_arguments(raw), [f"[{address('a')}, {address('b')}]", "200000000 [2e8]", "0xdeadbeef"])
        graph = make_graph([{"type": "call", "address": address("d"), "function": "swap", "params": raw, "children": []}])
        operands = annotation_for(build_runtime_annotations(graph), "n002")["observed_operands"]
        self.assertEqual(operands["arguments_raw"], split_top_level_arguments(raw))

    def test_empty_parameters_and_unresolved_selector_are_explicit(self) -> None:
        graph = make_graph(
            [
                {"type": "call", "address": address("a"), "function": "swap", "params": "", "children": []},
                {"type": "call", "address": address("b"), "function": "0xdeadbeef", "params": "", "children": []},
            ]
        )
        result = build_runtime_annotations(graph)
        self.assertEqual(annotation_for(result, "n002")["observed_operands"]["arguments_raw"], [])
        unresolved = annotation_for(result, "n003")
        self.assertEqual(unresolved["action_types"], [])
        self.assertIn("function selector unresolved: 0xdeadbeef", unresolved["uncertainties"])

    def test_callbacks_use_only_explicit_registry(self) -> None:
        graph = make_graph(
            [
                {"type": "call", "address": address("a"), "function": "executeOperation", "params": "", "children": []},
                {"type": "call", "address": address("b"), "function": "ordinaryCall", "params": "", "children": []},
            ]
        )
        result = build_runtime_annotations(graph)
        self.assertEqual(annotation_for(result, "n002")["action_types"], ["CALLBACK"])
        self.assertEqual(annotation_for(result, "n003")["action_types"], [])

    def test_exact_price_read_rules_remain_available_without_real_trace_hits(self) -> None:
        graph = make_graph(
            [
                {"type": "call", "address": address("a"), "function": "queryRate", "params": "", "children": []},
                {"type": "call", "address": address("b"), "function": "latestAnswer", "params": "", "children": []},
                {"type": "call", "address": address("c"), "function": "getAmountsOut", "params": "1, []", "children": []},
            ]
        )
        result = build_runtime_annotations(graph)
        for node_id in ("n002", "n003", "n004"):
            self.assertEqual(annotation_for(result, node_id)["action_types"], ["PRICE_READ"])

    def test_address_profiles_normalize_case_and_aggregate_capabilities(self) -> None:
        mixed_case = "0xAbCd" + "0" * 36
        graph = make_graph(
            [
                {"type": "call", "address": mixed_case, "function": "transfer", "params": f"{address('b')}, 1", "children": []},
                {"type": "call", "address": mixed_case.lower(), "function": "balanceOf", "params": address("b"), "children": []},
            ]
        )
        result = build_runtime_annotations(graph)
        profiles = [item.to_dict() for item in result.address_profiles]
        target = next(item for item in profiles if item["normalized_address"] == mixed_case.lower())
        self.assertEqual(target["address"], mixed_case)
        self.assertEqual(target["observed_capabilities"], ["TOKEN_LIKE"])
        self.assertEqual(target["evidence_node_ids"], ["n002", "n003"])

    def test_graph_is_unchanged_and_annotations_are_deterministic(self) -> None:
        graph = make_graph([{"type": "call", "address": address("a"), "function": "transfer", "params": f"{address('b')}, 1", "children": []}])
        before = copy.deepcopy(graph.to_dict())
        first = serialize_runtime_annotations(build_runtime_annotations(graph))
        second = serialize_runtime_annotations(build_runtime_annotations(graph))
        self.assertEqual(graph.to_dict(), before)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))

    def test_annotation_output_contains_no_forbidden_labels_or_causal_edges(self) -> None:
        graph = make_graph([{"type": "call", "address": address("a"), "function": "transferFrom", "params": f"{address('b')}, {address('c')}, 1", "children": []}])
        payload = build_runtime_annotations(graph).to_dict()
        self.assertTrue(set(payload["metadata"]["action_type_counts"]).issubset(ALLOWED_ACTION_TYPES))
        serialized = json.dumps(payload).upper()
        for forbidden in ("ATTACKER", "VULNERABLE", "ROOT_CAUSE", "MALICIOUS", '"ASSET"', '"PARAM"', '"STATE"', '"RESULT"'):
            self.assertNotIn(forbidden, serialized)
        self.assertNotIn("edges", payload)

    def test_load_and_write_round_trip_requires_phase_one_schema(self) -> None:
        graph = make_graph([{"type": "call", "address": address("a"), "function": "borrow", "params": "", "children": []}])
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "transaction_graph_v1.json"
            output_path = Path(temp_dir) / "runtime_annotations_v1.json"
            write_transaction_graph(graph, graph_path)
            loaded = load_transaction_graph(graph_path)
            written = write_runtime_annotations(build_runtime_annotations(loaded), output_path)
            self.assertEqual(written, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), serialize_runtime_annotations(build_runtime_annotations(loaded)))
            invalid_path = Path(temp_dir) / "invalid.json"
            invalid_path.write_text('{"schema_version":"other"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_transaction_graph(invalid_path)

    def test_real_phase_one_graph_is_read_only_compatible_when_available(self) -> None:
        if not REAL_GRAPH_PATH.exists():
            self.skipTest("No phase-1 real transaction graph is available in this checkout")
        before = REAL_GRAPH_PATH.read_bytes()
        graph = load_transaction_graph(REAL_GRAPH_PATH)
        annotations = build_runtime_annotations(graph)
        self.assertEqual(len(annotations.node_annotations), len(graph.nodes))
        self.assertEqual(REAL_GRAPH_PATH.read_bytes(), before)
        self.assertEqual(
            hashlib.sha256(serialize_transaction_graph(graph).encode("utf-8")).hexdigest(),
            annotations.transaction_graph_sha256,
        )


if __name__ == "__main__":
    unittest.main()

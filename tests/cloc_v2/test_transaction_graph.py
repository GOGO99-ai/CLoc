from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.cloc_v2.call_status import resolve_call_status
from src.cloc_v2.transaction_graph import (
    build_transaction_graph,
    load_call_tree,
    serialize_transaction_graph,
    write_transaction_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "call_tree_legacy.json"


def count_nodes(node: dict) -> int:
    return 1 + sum(count_nodes(child) for child in node.get("children", []) or [])


class TransactionGraphTests(unittest.TestCase):
    def load_fixture(self) -> dict:
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_single_node_tree_has_no_edges(self) -> None:
        raw = {"type": "call", "address": "0xroot", "function": "ping", "children": []}
        graph = build_transaction_graph(raw)
        self.assertEqual(graph.root_id, "n001")
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(len(graph.call_edges), 0)
        self.assertEqual(len(graph.order_edges), 0)
        self.assertEqual(graph.nodes[0].trace_path, "0")
        self.assertIsNone(graph.nodes[0].caller_address)
        self.assertEqual(graph.nodes[0].caller_source, "UNKNOWN")

    def test_nested_tree_has_preorder_metadata_and_call_edges(self) -> None:
        graph = build_transaction_graph(self.load_fixture())
        nodes = graph.nodes
        self.assertEqual([node.node_id for node in nodes], ["n001", "n002", "n003", "n004"])
        self.assertEqual([node.trace_path for node in nodes], ["0", "0/0", "0/1", "0/1/0"])
        self.assertEqual([node.depth for node in nodes], [0, 1, 1, 2])
        self.assertEqual([node.parent_id for node in nodes], [None, "n001", "n001", "n003"])
        self.assertEqual([node.sibling_order for node in nodes], [0, 0, 1, 0])
        self.assertEqual([(edge.src, edge.dst) for edge in graph.call_edges], [("n001", "n002"), ("n001", "n003"), ("n003", "n004")])
        self.assertEqual(len(graph.call_edges), len(nodes) - 1)

    def test_order_edges_link_only_adjacent_siblings(self) -> None:
        raw = {
            "type": "call",
            "children": [
                {"type": "call", "function": "a", "children": []},
                {"type": "call", "function": "b", "children": [{"type": "call", "function": "nested", "children": []}]},
                {"type": "call", "function": "c", "children": []},
            ],
        }
        graph = build_transaction_graph(raw)
        self.assertEqual([(edge.src, edge.dst) for edge in graph.order_edges], [("n002", "n003"), ("n003", "n005")])
        self.assertNotIn(("n002", "n005"), [(edge.src, edge.dst) for edge in graph.order_edges])
        self.assertNotIn(("n003", "n004"), [(edge.src, edge.dst) for edge in graph.order_edges])

    def test_repeated_function_calls_remain_distinct_instances(self) -> None:
        graph = build_transaction_graph(self.load_fixture())
        swaps = [node for node in graph.nodes if node.function_name == "swap"]
        self.assertEqual([node.node_id for node in swaps], ["n002", "n003"])
        self.assertNotEqual(swaps[0].trace_path, swaps[1].trace_path)

    def test_function_selector_and_existing_fields_are_preserved(self) -> None:
        graph = build_transaction_graph(self.load_fixture())
        root, child_a, child_b, grandchild = graph.nodes
        self.assertIsNone(root.function_name)
        self.assertEqual(root.function_selector, "0xdeadbeef")
        self.assertEqual(grandchild.function_selector, "0xcafebabe")
        self.assertEqual(child_a.function_name, "swap")
        self.assertEqual(child_a.decoded_params, {"amount": 10})
        self.assertEqual(child_b.value, "7")
        self.assertEqual(grandchild.gas, 700)

    def test_missing_fields_are_null_and_phase_one_fields_are_fixed(self) -> None:
        graph = build_transaction_graph({"children": [{"children": []}]})
        for node in graph.nodes:
            payload = node.to_dict()
            self.assertIsNone(payload["callee_address"])
            self.assertIsNone(payload["function_name"])
            self.assertIsNone(payload["function_selector"])
            self.assertIsNone(payload["call_type"])
            self.assertIsNone(payload["params_raw"])
            self.assertIsNone(payload["decoded_params"])
            self.assertIsNone(payload["value"])
            self.assertIsNone(payload["gas"])
            self.assertIsNone(payload["gas_used"])
            self.assertEqual(payload["execution_status"], "UNKNOWN")
            self.assertEqual(payload["execution_status_source"], "NOT_REPORTED_BY_CALL_TREE")
            self.assertEqual(
                payload["execution_status_uncertainties"],
                ["CALL_SUCCESS_STATUS_NOT_REPORTED"],
            )
            self.assertFalse(payload["execution_success_evidence"])
            self.assertIsNone(payload["source_ref"])
            self.assertEqual(payload["contract_role"], "UNKNOWN")
            self.assertEqual(payload["semantic_flags"], [])

    def test_caller_addresses_are_derived_only_from_parent(self) -> None:
        graph = build_transaction_graph(self.load_fixture())
        self.assertEqual(graph.nodes[1].caller_address, "0xroot")
        self.assertEqual(graph.nodes[1].caller_source, "DERIVED_FROM_PARENT")
        explicit = build_transaction_graph(
            {"address": "0xroot", "children": [{"from": "0xexplicit", "address": "0xchild", "children": []}]}
        )
        self.assertEqual(explicit.nodes[1].caller_address, "0xexplicit")
        self.assertEqual(explicit.nodes[1].caller_source, "EXPLICIT_FROM")

    def test_input_is_not_modified_and_serialization_is_byte_stable(self) -> None:
        raw = self.load_fixture()
        before = copy.deepcopy(raw)
        first = serialize_transaction_graph(build_transaction_graph(raw))
        second = serialize_transaction_graph(build_transaction_graph(raw))
        self.assertEqual(raw, before)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))

    def test_legacy_wrapper_and_direct_trace_node_shape_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapped_path = Path(temp_dir) / "wrapped.json"
            wrapped_path.write_text(json.dumps({"call_tree": self.load_fixture()}), encoding="utf-8")
            graph = build_transaction_graph(load_call_tree(wrapped_path))
        self.assertEqual(len(graph.nodes), 4)
        trace_node_shape = {"type": "call", "gas": 1, "address": "0xtrace", "function": "ping", "params": "", "call_type": None, "value": None, "children": []}
        self.assertEqual(len(build_transaction_graph(trace_node_shape).nodes), 1)

    def test_all_edges_are_observed_and_only_phase_one_types_exist(self) -> None:
        graph = build_transaction_graph(self.load_fixture())
        for edge in (*graph.call_edges, *graph.order_edges):
            self.assertEqual(edge.evidence_level, "OBSERVED")
            self.assertIn(edge.edge_type, {"CALL", "ORDER"})
            self.assertEqual(len(edge.trace_refs), 2)

    def test_existing_cache_is_read_only_compatible_when_available(self) -> None:
        cache_root = PROJECT_ROOT / "data" / "cache" / "all_tree"
        cached_trees = sorted(cache_root.rglob("*.json")) if cache_root.exists() else []
        if not cached_trees:
            self.skipTest("No existing cached call tree is available in this checkout")
        cache_path = cached_trees[0]
        original_bytes = cache_path.read_bytes()
        raw = json.loads(original_bytes.decode("utf-8"))
        graph = build_transaction_graph(raw)
        self.assertEqual(len(graph.nodes), count_nodes(raw))
        self.assertEqual(len(graph.call_edges), len(graph.nodes) - 1)
        self.assertEqual(cache_path.read_bytes(), original_bytes)

    def test_writer_creates_separate_output_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "transaction_graph_v1.json"
            graph = build_transaction_graph(self.load_fixture())
            written_path = write_transaction_graph(graph, output_path)
            self.assertEqual(written_path, output_path)
            self.assertEqual(written_path.read_text(encoding="utf-8"), serialize_transaction_graph(graph))

    def test_structured_call_status_enters_normalized_node(self) -> None:
        status = resolve_call_status({"output": "0x"}).to_dict()
        graph = build_transaction_graph(
            {
                "type": "CALL",
                "address": "0x" + "a" * 40,
                "function": "0x11111111",
                "gas": "0x100",
                "gas_used": "0x80",
                "call_status": status,
                "children": [],
            }
        )
        node = graph.nodes[0]
        self.assertEqual(node.execution_status, "SUCCESS")
        self.assertEqual(node.execution_status_source, "CALLTRACER_OUTPUT")
        self.assertEqual(node.execution_status_uncertainties, ())
        self.assertTrue(node.execution_success_evidence)
        self.assertEqual(node.gas_used, "0x80")

    def test_legacy_fixture_defaults_to_unreported_unknown_status(self) -> None:
        graph = build_transaction_graph(self.load_fixture())
        for node in graph.nodes:
            self.assertEqual(node.execution_status, "UNKNOWN")
            self.assertEqual(node.execution_status_source, "NOT_REPORTED_BY_CALL_TREE")
            self.assertIn("CALL_SUCCESS_STATUS_NOT_REPORTED", node.execution_status_uncertainties)
            self.assertFalse(node.execution_success_evidence)

    def test_forged_success_without_evidence_is_downgraded(self) -> None:
        graph = build_transaction_graph(
            {
                "call_status": {
                    "status": "SUCCESS",
                    "decision_source": "CALLTRACER_OUTPUT",
                    "output_present": False,
                    "output_valid_hex": False,
                    "error_failure_evidence": False,
                    "revert_reason_failure_evidence": False,
                    "usable_as_success_evidence": False,
                    "uncertainties": [],
                },
                "children": [],
            }
        )
        node = graph.nodes[0]
        self.assertEqual(node.execution_status, "UNKNOWN")
        self.assertEqual(node.execution_status_source, "INVALID_CALL_STATUS")
        self.assertIn("CALL_STATUS_SUCCESS_EVIDENCE_INVALID", node.execution_status_uncertainties)
        self.assertFalse(node.execution_success_evidence)

    def test_structured_status_serialization_is_stable(self) -> None:
        status = resolve_call_status({"output": "0xdeadbeef"}).to_dict()
        first = {"call_status": status, "children": []}
        second = {"children": [], "call_status": dict(reversed(list(status.items())))}
        self.assertEqual(
            serialize_transaction_graph(build_transaction_graph(first)),
            serialize_transaction_graph(build_transaction_graph(second)),
        )


if __name__ == "__main__":
    unittest.main()

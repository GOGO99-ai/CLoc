from __future__ import annotations

import unittest

from src.cloc_v2.calltracer_adapter import (
    adapt_calltracer_call_tree,
    serialize_calltracer_call_tree,
)
from src.cloc_v2.transaction_graph import build_transaction_graph


def frame(
    *,
    selector: str = "0x11111111",
    output: object = "0xdeadbeef",
    error: object = None,
    revert_reason: object = None,
    children: object = (),
) -> dict:
    built = {
        "type": "CALL",
        "from": "0x" + "a" * 40,
        "to": "0x" + "b" * 40,
        "input": selector + "00" * 32,
        "gas": "0x100",
        "gasUsed": "0x80",
        "value": "0x0",
        "calls": list(children),
    }
    if output is not ...:
        built["output"] = output
    if error is not None:
        built["error"] = error
    if revert_reason is not None:
        built["revertReason"] = revert_reason
    return built


class CallTracerAdapterTests(unittest.TestCase):
    def test_nested_success_failure_unknown_are_preserved_in_graph(self) -> None:
        raw = frame(
            children=(
                frame(selector="0x22222222", output="0x"),
                frame(selector="0x33333333", output="0xdead", error="reverted"),
                frame(selector="0x44444444", output=...),
            )
        )
        graph = build_transaction_graph(adapt_calltracer_call_tree(raw))
        self.assertEqual(
            [node.execution_status for node in graph.nodes],
            ["SUCCESS", "SUCCESS", "FAILED", "UNKNOWN"],
        )
        self.assertEqual(graph.nodes[1].execution_status_source, "CALLTRACER_OUTPUT")
        self.assertEqual(graph.nodes[2].execution_status_source, "CALLTRACER_ERROR")
        self.assertIn("CALLTRACER_OUTPUT_MISSING", graph.nodes[3].execution_status_uncertainties)

    def test_zero_length_output_is_explicit_success(self) -> None:
        tree = adapt_calltracer_call_tree(frame(output="0x"))
        self.assertEqual(tree["call_status"]["status"], "SUCCESS")
        self.assertTrue(tree["call_status"]["usable_as_success_evidence"])

    def test_error_overrides_output_and_status_is_sanitized(self) -> None:
        tree = adapt_calltracer_call_tree(frame(output="0xdeadbeef", error="secret error detail"))
        self.assertEqual(tree["call_status"]["status"], "FAILED")
        serialized = serialize_calltracer_call_tree(tree)
        self.assertNotIn("secret error detail", serialized)
        self.assertNotIn("0xdeadbeef", serialized)
        self.assertNotIn('"error"', serialized)
        self.assertNotIn('"output"', serialized)
        self.assertNotIn('"revertReason"', serialized)

    def test_selector_is_extracted_from_legal_input(self) -> None:
        tree = adapt_calltracer_call_tree(frame(selector="0xA9059CBB"))
        graph = build_transaction_graph(tree)
        self.assertEqual(tree["function"], "0xa9059cbb")
        self.assertEqual(graph.nodes[0].function_selector, "0xa9059cbb")
        invalid = adapt_calltracer_call_tree(frame(selector="0x123"))
        self.assertIsNone(invalid["function"])

    def test_child_order_is_stable(self) -> None:
        raw = frame(
            children=(
                frame(selector="0x11111112"),
                frame(selector="0x11111113"),
                frame(selector="0x11111114"),
            )
        )
        tree = adapt_calltracer_call_tree(raw)
        self.assertEqual(
            [child["function"] for child in tree["children"]],
            ["0x11111112", "0x11111113", "0x11111114"],
        )

    def test_output_does_not_retain_raw_status_values(self) -> None:
        raw = frame(output="0xfeedface", error="hidden", revert_reason="also-hidden")
        tree = adapt_calltracer_call_tree(raw)
        serialized = serialize_calltracer_call_tree(tree)
        for forbidden in ("0xfeedface", "hidden", "also-hidden", '"error"', '"output"', '"revertReason"'):
            self.assertNotIn(forbidden, serialized)

    def test_input_dictionary_order_does_not_change_serialization(self) -> None:
        first = {
            "type": "CALL",
            "from": "0x" + "a" * 40,
            "to": "0x" + "b" * 40,
            "input": "0x11111111",
            "output": "0x",
            "calls": [],
        }
        second = {
            "calls": [],
            "output": "0x",
            "input": "0x11111111",
            "to": "0x" + "b" * 40,
            "from": "0x" + "a" * 40,
            "type": "CALL",
        }
        self.assertEqual(
            serialize_calltracer_call_tree(adapt_calltracer_call_tree(first)),
            serialize_calltracer_call_tree(adapt_calltracer_call_tree(second)),
        )

    def test_invalid_root_and_children_fail_safely(self) -> None:
        with self.assertRaisesRegex(ValueError, "root frame"):
            adapt_calltracer_call_tree([])
        with self.assertRaisesRegex(ValueError, "calls"):
            adapt_calltracer_call_tree({"calls": "invalid"})
        with self.assertRaisesRegex(ValueError, "child"):
            adapt_calltracer_call_tree(frame(children=("invalid",)))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from src.cloc_baselines.source_mapping import (
    expand_source_map,
    instruction_program_counters,
    reconstruct_trace_addresses,
)
from src.cloc_baselines.solidity_parser import internal_call_names, parse_solidity_source


class SourceMappingTests(unittest.TestCase):
    def test_packaged_solidity_parser_finds_function_ranges(self) -> None:
        text = "contract Demo { function ping(uint value) external returns (uint) { return value; } }"
        parsed = parse_solidity_source(text)
        self.assertEqual(parsed["parse_status"], "PARSED")
        self.assertEqual(len(parsed["functions"]), 1)
        self.assertEqual(parsed["functions"][0]["function_name"], "ping")
        self.assertEqual(parsed["functions"][0]["canonical_signature"], "ping(uint256)")

    def test_internal_calls_exclude_member_calls_and_comments(self) -> None:
        calls = internal_call_names("function a() { b(); token.b(); /* c(); */ if (true) { d(); } }")
        self.assertEqual(calls, ("a", "b", "d"))

    def test_instruction_program_counters_skip_push_data(self) -> None:
        self.assertEqual(instruction_program_counters("0x6001600201"), (0, 2, 4))
        self.assertEqual(instruction_program_counters("0x7f" + "00" * 32 + "00"), (0, 33))

    def test_source_map_inherits_omitted_fields(self) -> None:
        self.assertEqual(
            expand_source_map("1:2:3:-;:4;;5::6"),
            ((1, 2, 3), (1, 4, 3), (1, 4, 3), (5, 4, 6)),
        )

    def test_trace_address_reconstruction_uses_call_target(self) -> None:
        root = "0x" + "1" * 40
        child = "2" * 40
        logs = [
            {"pc": 0, "op": "PUSH1", "depth": 1, "stack": []},
            {"pc": 2, "op": "CALL", "depth": 1, "stack": ["00", child, "00"]},
            {"pc": 0, "op": "STOP", "depth": 2, "stack": []},
            {"pc": 3, "op": "STOP", "depth": 1, "stack": []},
        ]
        rows = reconstruct_trace_addresses(root, logs)
        self.assertEqual([item["address"] for item in rows], [root, root, "0x" + child, root])


if __name__ == "__main__":
    unittest.main()

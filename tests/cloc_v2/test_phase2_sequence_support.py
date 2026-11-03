from __future__ import annotations

import unittest

from src.cloc_v2.phase2_sequence_support import aggregate_sequence_exact_support


def _row(path: str, signature: str, start: int, status: str) -> dict[str, object]:
    return {
        "relative_file": path,
        "function_signature": signature,
        "start_line": start,
        "end_line": start + 4,
        "selector_match_status": status,
    }


class Phase2SequenceSupportTests(unittest.TestCase):
    def test_exact_identity_is_shared_only_with_siblings_that_contain_it(self):
        withdraw = _row("contracts/Vault.sol", "withdraw(uint256)", 10, "SELECTOR_EXACT_MATCH")
        withdraw_coarse = _row("contracts/Vault.sol", "withdraw(uint256)", 10, "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK")
        deposit = _row("contracts/Vault.sol", "deposit(uint256)", 30, "SELECTOR_EXACT_MATCH")
        pools = {
            "S1": (withdraw,),
            "S2": (withdraw_coarse, deposit),
            "S3": (withdraw_coarse,),
        }
        aliases = {"S1": "G1", "S2": "G1", "S3": "G2"}

        support = aggregate_sequence_exact_support(pools, aliases)

        self.assertEqual(support["S1"], ())
        self.assertEqual(support["S2"], (("contracts/Vault.sol", "withdraw(uint256)", 10, 14),))
        self.assertEqual(support["S3"], ())

    def test_output_is_deterministic_and_requires_anonymous_group_aliases(self):
        first = _row("contracts/Z.sol", "z()", 20, "SELECTOR_EXACT_MATCH")
        second = _row("contracts/A.sol", "a()", 10, "SELECTOR_EXACT_MATCH")
        pools = {"S1": (first, second), "S2": (first, second)}
        self.assertEqual(
            aggregate_sequence_exact_support(pools, {"S1": "G1", "S2": "G1"})["S1"],
            (("contracts/A.sol", "a()", 10, 14), ("contracts/Z.sol", "z()", 20, 24)),
        )
        with self.assertRaises(ValueError):
            aggregate_sequence_exact_support(pools, {"S1": "incident-name", "S2": "incident-name"})


if __name__ == "__main__":
    unittest.main()

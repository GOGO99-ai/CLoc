from __future__ import annotations

import unittest

from src.cloc_v2.function_identity import (
    OfflineFunctionCandidate,
    resolve_function_identity,
    serialize_function_identity,
)


def address(character: str) -> str:
    return "0x" + character * 40


def candidate(
    signature: str,
    *,
    chain_id: int = 1,
    effective_address: str | None = None,
    evidence_source: str = "local_abi",
) -> OfflineFunctionCandidate:
    return OfflineFunctionCandidate(
        blockchain_chain_id=chain_id,
        effective_address=effective_address or address("b"),
        function_name=signature.split("(", 1)[0],
        signature=signature,
        evidence_source=evidence_source,
    )


def resolve(**overrides):
    values = {
        "blockchain_chain_id": 1,
        "runtime_address": address("a"),
        "effective_address": address("b"),
        "trace_selector": None,
        "function_name": "foo",
        "offline_candidates": (),
    }
    values.update(overrides)
    return resolve_function_identity(**values)


class FunctionIdentityTests(unittest.TestCase):
    def test_valid_native_selector_always_has_priority(self) -> None:
        built = resolve(
            trace_selector="0xA9059CBB",
            offline_candidates=(candidate("foo(uint256)"), candidate("foo(address)")),
        )
        self.assertEqual(built.selector, "0xa9059cbb")
        self.assertEqual(built.resolution_source, "TRACE_NATIVE")
        self.assertEqual(built.status, "RESOLVED")
        self.assertTrue(built.usable_for_anchor)
        self.assertEqual(built.candidate_selectors, ())

    def test_unique_offline_abi_candidate_recovers_selector(self) -> None:
        built = resolve(offline_candidates=(candidate("foo(uint256)"),))
        self.assertEqual(built.resolution_source, "OFFLINE_ABI_UNIQUE")
        self.assertEqual(built.status, "RESOLVED")
        self.assertTrue(built.usable_for_anchor)
        self.assertEqual(built.candidate_signatures, ("foo(uint256)",))
        self.assertEqual((built.selector,), built.candidate_selectors)
        self.assertEqual(built.candidate_evidence_sources, ("local_abi",))

    def test_overloads_with_different_selectors_are_ambiguous(self) -> None:
        built = resolve(
            offline_candidates=(candidate("foo(uint256)"), candidate("foo(address)")),
        )
        self.assertIsNone(built.selector)
        self.assertEqual(built.status, "UNCERTAIN")
        self.assertFalse(built.usable_for_anchor)
        self.assertIn("SELECTOR_RECOVERY_AMBIGUOUS", built.uncertainties)
        self.assertEqual(len(built.candidate_selectors), 2)

    def test_missing_offline_source_is_explicit(self) -> None:
        built = resolve()
        self.assertEqual(built.resolution_source, "UNRESOLVED")
        self.assertIn("SELECTOR_RECOVERY_SOURCE_MISSING", built.uncertainties)

    def test_invalid_chain_id_and_addresses_are_uncertain(self) -> None:
        for overrides, reason in (
            ({"blockchain_chain_id": 0}, "BLOCKCHAIN_CHAIN_ID_INVALID"),
            ({"blockchain_chain_id": True}, "BLOCKCHAIN_CHAIN_ID_INVALID"),
            ({"runtime_address": "not-an-address"}, "RUNTIME_ADDRESS_INVALID"),
            ({"effective_address": "0x1234"}, "EFFECTIVE_ADDRESS_INVALID"),
        ):
            with self.subTest(reason=reason):
                built = resolve(trace_selector="0x11111111", **overrides)
                self.assertEqual(built.status, "UNCERTAIN")
                self.assertFalse(built.usable_for_anchor)
                self.assertIn(reason, built.uncertainties)

    def test_empty_function_name_cannot_use_offline_index(self) -> None:
        built = resolve(function_name="  ", offline_candidates=(candidate("foo(uint256)"),))
        self.assertIsNone(built.selector)
        self.assertIn("FUNCTION_NAME_MISSING", built.uncertainties)

    def test_invalid_native_selector_is_retained_as_uncertainty(self) -> None:
        built = resolve(
            trace_selector="0x123",
            offline_candidates=(candidate("foo(uint256)"),),
        )
        self.assertIsNotNone(built.selector)
        self.assertEqual(built.resolution_source, "OFFLINE_ABI_UNIQUE")
        self.assertEqual(built.status, "UNCERTAIN")
        self.assertFalse(built.usable_for_anchor)
        self.assertIn("TRACE_SELECTOR_INVALID", built.uncertainties)

    def test_function_name_matching_is_case_sensitive(self) -> None:
        built = resolve(
            function_name="foo",
            offline_candidates=(candidate("Foo(uint256)"),),
        )
        self.assertIsNone(built.selector)
        self.assertIn("SELECTOR_RECOVERY_FUNCTION_NOT_FOUND", built.uncertainties)

    def test_exact_duplicate_candidates_are_stably_deduplicated(self) -> None:
        item = candidate("foo(uint256)")
        built = resolve(offline_candidates=(item, item, item))
        self.assertEqual(built.status, "RESOLVED")
        self.assertEqual(built.candidate_signatures, ("foo(uint256)",))
        self.assertEqual(len(built.candidate_selectors), 1)
        self.assertEqual(built.candidate_evidence_sources, ("local_abi",))

    def test_candidate_order_does_not_change_serialization(self) -> None:
        first_candidate = candidate("foo(uint256)", evidence_source="source_b")
        second_candidate = candidate("foo(address)", evidence_source="source_a")
        first = resolve(offline_candidates=(first_candidate, second_candidate))
        second = resolve(offline_candidates=(second_candidate, first_candidate))
        self.assertEqual(serialize_function_identity(first), serialize_function_identity(second))


if __name__ == "__main__":
    unittest.main()

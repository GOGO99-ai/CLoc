from __future__ import annotations

import unittest
from decimal import Decimal

from src.cloc_v2.normal_baseline import (
    build_normal_baseline,
    query_normal_baseline,
    serialize_normal_baseline,
    serialize_normal_baseline_query,
)
from src.cloc_v2.normal_distance import AssetBehavior, CallStep, ChainFeatureRecord, MainAnchorKey


def addr(character: str) -> str:
    return "0x" + character * 40


ANCHOR = MainAnchorKey(1, addr("a"), "0x11111111")
CHAIN_VARIANT = MainAnchorKey(10, addr("a"), "0x11111111")
ADDRESS_VARIANT = MainAnchorKey(1, addr("b"), "0x11111111")
SELECTOR_VARIANT = MainAnchorKey(1, addr("a"), "0x22222222")
STEP_A = CallStep(addr("a"), "0x11111111")
STEP_B = CallStep(addr("b"), "0x22222222")
STEP_C = CallStep(addr("c"), "0x33333333")
INFLOW = AssetBehavior(addr("d"), "INFLOW")
OUTFLOW = AssetBehavior(addr("d"), "OUTFLOW")


def record(
    record_id: str,
    *,
    anchor: MainAnchorKey = ANCHOR,
    steps: tuple[CallStep, ...] = (),
    assets: frozenset[AssetBehavior] = frozenset(),
    uncertainties: tuple[str, ...] = (),
) -> ChainFeatureRecord:
    return ChainFeatureRecord(record_id, anchor, steps, assets, uncertainties)


class NormalBaselineTests(unittest.TestCase):
    def test_same_anchor_records_form_one_group(self) -> None:
        baseline = build_normal_baseline((record("a"), record("b")), min_group_size=2)
        self.assertEqual((baseline.record_count, baseline.group_count), (2, 1))
        self.assertEqual([item.record_id for item in baseline.groups[0].records], ["a", "b"])

    def test_chain_address_and_selector_variants_form_distinct_groups(self) -> None:
        baseline = build_normal_baseline(
            (
                record("chain", anchor=CHAIN_VARIANT),
                record("address", anchor=ADDRESS_VARIANT),
                record("selector", anchor=SELECTOR_VARIANT),
                record("base", anchor=ANCHOR),
            ),
            min_group_size=1,
        )
        self.assertEqual(baseline.group_count, 4)
        self.assertTrue(all(len(group.records) == 1 for group in baseline.groups))

    def test_each_group_has_independent_threshold(self) -> None:
        second_anchor = ADDRESS_VARIANT
        baseline = build_normal_baseline(
            (
                record("a1", steps=(STEP_A,)),
                record("a2", steps=(STEP_A,)),
                record("b1", anchor=second_anchor, steps=(STEP_A,)),
                record("b2", anchor=second_anchor, steps=(STEP_B,)),
            ),
            min_group_size=2,
        )
        thresholds = {group.main_anchor: group.calibration.threshold for group in baseline.groups}
        self.assertEqual(thresholds[ANCHOR], Decimal(0))
        self.assertEqual(thresholds[second_anchor], Decimal(1))

    def test_duplicate_record_id_is_rejected_across_whole_baseline(self) -> None:
        with self.assertRaises(ValueError):
            build_normal_baseline((record("same"), record("same", anchor=ADDRESS_VARIANT)), min_group_size=1)

    def test_top_three_sorts_by_chain_distance(self) -> None:
        baseline = build_normal_baseline(
            (
                record("a", steps=(STEP_A,)),
                record("b", steps=(STEP_A, STEP_B)),
                record("c", steps=(STEP_B,)),
                record("d", steps=(STEP_A,), assets=frozenset({INFLOW})),
            ),
            min_group_size=2,
        )
        result = query_normal_baseline(baseline, record("query", steps=(STEP_A,)))
        self.assertEqual([match.normal_record_id for match in result.top_matches], ["a", "b", "d"])

    def test_distance_ties_have_stable_record_id_order(self) -> None:
        baseline = build_normal_baseline(
            (record("c", steps=(STEP_B,)), record("a", steps=(STEP_B,)), record("b", steps=(STEP_B,))),
            min_group_size=2,
        )
        result = query_normal_baseline(baseline, record("query", steps=(STEP_A,)))
        self.assertEqual([match.normal_record_id for match in result.top_matches], ["a", "b", "c"])

    def test_group_with_fewer_than_three_records_returns_all_matches(self) -> None:
        baseline = build_normal_baseline((record("a"), record("b")), min_group_size=2)
        result = query_normal_baseline(baseline, record("query"))
        self.assertEqual(len(result.top_matches), 2)

    def test_distance_equal_to_threshold_passes(self) -> None:
        baseline = build_normal_baseline((record("a"), record("b")), min_group_size=2)
        result = query_normal_baseline(baseline, record("query"))
        self.assertEqual(result.group_threshold, result.nearest_distance)
        self.assertEqual(result.status, "PASS")

    def test_distance_above_threshold_fails(self) -> None:
        baseline = build_normal_baseline((record("a", steps=(STEP_A,)), record("b", steps=(STEP_A,))), min_group_size=2)
        result = query_normal_baseline(baseline, record("query", steps=(STEP_B,)))
        self.assertEqual(result.status, "FAIL")

    def test_missing_group_is_uncertain(self) -> None:
        baseline = build_normal_baseline((record("a"), record("b")), min_group_size=2)
        result = query_normal_baseline(baseline, record("query", anchor=ADDRESS_VARIANT))
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertIn("NORMAL_GROUP_NOT_FOUND", result.reasons)
        self.assertEqual(result.top_matches, ())

    def test_insufficient_group_is_uncertain_but_returns_diagnostic_matches(self) -> None:
        baseline = build_normal_baseline((record("a"), record("b")), min_group_size=3)
        result = query_normal_baseline(baseline, record("query"))
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertEqual(len(result.top_matches), 2)
        self.assertIn("NORMAL_GROUP_THRESHOLD_UNAVAILABLE", result.reasons)

    def test_uncertain_query_cannot_pass(self) -> None:
        baseline = build_normal_baseline((record("a"), record("b")), min_group_size=2)
        result = query_normal_baseline(baseline, record("query", uncertainties=("STRICT_CHAIN_UNCERTAIN",)))
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertIn("QUERY_FEATURE_RECORD_UNCERTAIN", result.reasons)

    def test_uncertain_normal_record_prevents_formal_group_threshold(self) -> None:
        baseline = build_normal_baseline(
            (record("a"), record("b", uncertainties=("STRICT_CHAIN_UNCERTAIN",))),
            min_group_size=2,
        )
        result = query_normal_baseline(baseline, record("query"))
        self.assertEqual(baseline.groups[0].calibration.status, "UNCERTAIN_INPUT")
        self.assertIsNone(result.group_threshold)
        self.assertEqual(result.status, "UNCERTAIN")

    def test_input_order_does_not_change_baseline_or_serialization(self) -> None:
        inputs = (record("b", steps=(STEP_A,)), record("a", steps=(STEP_A,)), record("c", steps=(STEP_B,)))
        first = build_normal_baseline(inputs, min_group_size=2)
        second = build_normal_baseline(tuple(reversed(inputs)), min_group_size=2)
        self.assertEqual(first, second)
        self.assertEqual(serialize_normal_baseline(first), serialize_normal_baseline(second))

    def test_serialized_outputs_exclude_labels_amounts_and_pairing_fields(self) -> None:
        baseline = build_normal_baseline((record("a"), record("b")), min_group_size=2)
        result = query_normal_baseline(baseline, record("query"))
        payload = serialize_normal_baseline(baseline) + serialize_normal_baseline_query(result)
        for forbidden in ("label", "amount", "cause", "location", "pair", "fault", "ground_truth"):
            self.assertNotIn(forbidden, payload.lower())


if __name__ == "__main__":
    unittest.main()

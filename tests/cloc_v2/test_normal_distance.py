from __future__ import annotations

import unittest
from decimal import Decimal

from src.cloc_v2.normal_distance import (
    AssetBehavior,
    CallStep,
    ChainFeatureRecord,
    MainAnchorKey,
    asset_distance,
    abnormal_observed_call_identities,
    call_multiplicity_excess,
    call_distance,
    calibrate_group_threshold,
    compute_chain_distance,
    is_call_topology_reliable,
    serialize_group_threshold_calibration,
)


def addr(character: str) -> str:
    return "0x" + character * 40


ANCHOR = MainAnchorKey(1, addr("a"), "0x11111111")
OTHER_ANCHOR = MainAnchorKey(1, addr("b"), "0x11111111")
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


class NormalDistanceTests(unittest.TestCase):
    def test_identical_call_sequences_have_zero_distance(self) -> None:
        self.assertEqual(call_distance(record("q", steps=(STEP_A, STEP_B)), record("n", steps=(STEP_A, STEP_B))), Decimal(0))

    def test_disjoint_call_sequences_have_one_distance(self) -> None:
        self.assertEqual(call_distance(record("q", steps=(STEP_A,)), record("n", steps=(STEP_B,))), Decimal(1))

    def test_call_order_changes_lcs_distance(self) -> None:
        self.assertEqual(
            call_distance(record("q", steps=(STEP_A, STEP_B)), record("n", steps=(STEP_B, STEP_A))),
            Decimal("0.500000000000000000"),
        )

    def test_repeated_call_steps_are_not_set_normalized(self) -> None:
        self.assertEqual(
            call_distance(record("q", steps=(STEP_A, STEP_A)), record("n", steps=(STEP_A,))),
            Decimal("0.500000000000000000"),
        )

    def test_asset_only_uncertainty_keeps_call_topology_reliable(self) -> None:
        value = record(
            "n",
            steps=(STEP_A,),
            uncertainties=("ASSET_DIRECTION_RELATIVE_TO_ANCHOR_UNRESOLVED",),
        )
        self.assertTrue(is_call_topology_reliable(value))
        self.assertFalse(is_call_topology_reliable(record("bad", steps=(STEP_A,), uncertainties=("CALLTRACER_OUTPUT_MISSING",))))

    def test_call_multiplicity_uses_reliable_exact_anchor_upper_bound(self) -> None:
        normals = (
            record("n1", steps=(STEP_A, STEP_A), uncertainties=("ASSET_DIRECTION_RELATIVE_TO_ANCHOR_UNRESOLVED",)),
            record("n2", steps=(STEP_A, STEP_A)),
            record("n3", steps=(STEP_A, STEP_A, STEP_B)),
            record("ignored-anchor", anchor=OTHER_ANCHOR, steps=(STEP_A,) * 20),
            record("ignored-call", steps=(STEP_A,) * 20, uncertainties=("CALLTRACER_OUTPUT_MISSING",)),
        )
        query = record("q", steps=(STEP_A,) * 6)
        self.assertEqual(call_multiplicity_excess(query, normals, STEP_A), 4)

    def test_call_multiplicity_is_unknown_without_three_reliable_normals(self) -> None:
        query = record("q", steps=(STEP_A,) * 4)
        self.assertIsNone(call_multiplicity_excess(query, (record("n1", steps=(STEP_A,)), record("n2", steps=(STEP_A,))), STEP_A))

    def test_observed_abnormal_identity_uses_partial_query_as_lower_bound(self) -> None:
        query = record("q", steps=(STEP_A,) * 3, uncertainties=("CALL_FUNCTION_IDENTITY_UNRESOLVED",))
        normals = tuple(record(f"n{index}", steps=(STEP_A,)) for index in range(3))
        self.assertEqual(abnormal_observed_call_identities((query,), normals), (STEP_A,))

    def test_observed_abnormal_identity_ignores_unreliable_normal_records(self) -> None:
        query = record("q", steps=(STEP_A,) * 3)
        normals = (
            record("n1", steps=(STEP_A,)),
            record("n2", steps=(STEP_A,)),
            record("hidden-upper-bound", steps=(STEP_A,) * 9, uncertainties=("CALLTRACER_OUTPUT_MISSING",)),
        )
        self.assertEqual(abnormal_observed_call_identities((query,), normals), ())

    def test_both_empty_asset_sets_have_zero_distance(self) -> None:
        self.assertEqual(asset_distance(record("q"), record("n")), Decimal(0))

    def test_one_empty_asset_set_has_one_distance(self) -> None:
        self.assertEqual(asset_distance(record("q", assets=frozenset({INFLOW})), record("n")), Decimal(1))

    def test_asset_direction_is_part_of_behavior_identity(self) -> None:
        self.assertEqual(asset_distance(record("q", assets=frozenset({INFLOW})), record("n", assets=frozenset({OUTFLOW}))), Decimal(1))

    def test_chain_distance_is_maximum_of_call_and_asset_components(self) -> None:
        distance = compute_chain_distance(
            record("q", steps=(STEP_A, STEP_B), assets=frozenset({INFLOW})),
            record("n", steps=(STEP_A, STEP_C), assets=frozenset({INFLOW, OUTFLOW})),
        )
        self.assertEqual(distance.call_distance, Decimal("0.500000000000000000"))
        self.assertEqual(distance.asset_distance, Decimal("0.500000000000000000"))
        self.assertEqual(distance.chain_distance, Decimal("0.500000000000000000"))

    def test_each_record_uses_its_actual_nearest_neighbor(self) -> None:
        calibrated = calibrate_group_threshold(
            (
                record("a", steps=(STEP_A,)),
                record("b", steps=(STEP_A,)),
                record("c", steps=(STEP_B,)),
            ),
            min_group_size=3,
        )
        self.assertEqual(
            [(item.record_id, item.nearest_record_id, item.chain_distance) for item in calibrated.nearest_neighbor_distances],
            [("a", "b", Decimal(0)), ("b", "a", Decimal(0)), ("c", "a", Decimal(1))],
        )

    def test_empirical_nearest_rank_95_threshold(self) -> None:
        calibrated = calibrate_group_threshold(
            (
                record("a", steps=(STEP_A,)),
                record("b", steps=(STEP_A,)),
                record("c", steps=(STEP_B,)),
                record("d", steps=(STEP_C,)),
            ),
            min_group_size=4,
        )
        self.assertEqual(calibrated.threshold, Decimal(1))
        self.assertEqual(calibrated.quantile, Decimal("0.95"))
        self.assertEqual(calibrated.quantile_method, "EMPIRICAL_NEAREST_RANK")

    def test_insufficient_group_returns_no_threshold(self) -> None:
        calibrated = calibrate_group_threshold((record("a"),), min_group_size=1)
        self.assertEqual(calibrated.status, "INSUFFICIENT_GROUP")
        self.assertIsNone(calibrated.threshold)

    def test_mixed_main_anchors_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_group_threshold((record("a"), record("b", anchor=OTHER_ANCHOR)), min_group_size=2)

    def test_uncertain_record_cannot_produce_formal_threshold(self) -> None:
        calibrated = calibrate_group_threshold(
            (record("a"), record("b", uncertainties=("STRICT_CHAIN_UNCERTAIN",))),
            min_group_size=2,
        )
        self.assertEqual(calibrated.status, "UNCERTAIN_INPUT")
        self.assertIsNone(calibrated.threshold)
        self.assertIn("UNCERTAIN_FEATURE_RECORDS_PRESENT", calibrated.uncertainties)

    def test_input_order_does_not_change_calibration_or_serialization(self) -> None:
        inputs = (record("b", steps=(STEP_A,)), record("a", steps=(STEP_A,)), record("c", steps=(STEP_B,)))
        first = calibrate_group_threshold(inputs, min_group_size=3)
        second = calibrate_group_threshold(tuple(reversed(inputs)), min_group_size=3)
        self.assertEqual(first, second)
        self.assertEqual(serialize_group_threshold_calibration(first), serialize_group_threshold_calibration(second))

    def test_serialization_contains_no_amount_or_label_fields(self) -> None:
        features = record("a", steps=(STEP_A,), assets=frozenset({AssetBehavior("NATIVE", "INFLOW")}))
        payload = str(features.to_dict()) + serialize_group_threshold_calibration(
            calibrate_group_threshold((features, record("b", steps=(STEP_A,))), min_group_size=2)
        )
        self.assertNotIn("amount", payload.lower())
        self.assertNotIn("label", payload.lower())
        self.assertNotIn("cause", payload.lower())
        self.assertNotIn("location", payload.lower())


if __name__ == "__main__":
    unittest.main()

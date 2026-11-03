from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.dry_run_phase1_normality import (
    build_synthetic_phase1_dry_run,
    build_synthetic_phase1_preparation,
    serialize_phase1_normality_dry_run,
    write_phase1_normality_dry_run,
)


class Phase1NormalityDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = build_synthetic_phase1_dry_run()

    def test_preparation_is_ready_for_llm(self) -> None:
        self.assertEqual(self.report["preparation_status"], "READY_FOR_LLM")

    def test_shared_synthetic_preparation_is_fully_eligible(self) -> None:
        preparation = build_synthetic_phase1_preparation()
        self.assertEqual(preparation.preparation_status, "READY_FOR_LLM")
        self.assertEqual(preparation.eligibility.eligibility, "ELIGIBLE")
        self.assertTrue(preparation.eligibility.llm_allowed)
        self.assertIsNotNone(preparation.prompt)
        self.assertIsNotNone(preparation.gate_evidence)

    def test_eligibility_is_eligible(self) -> None:
        self.assertEqual(self.report["eligibility"], "ELIGIBLE")

    def test_formal_normal_group_size_is_three(self) -> None:
        self.assertEqual(self.report["baseline_summary"]["min_group_size"], 3)
        self.assertEqual(self.report["baseline_summary"]["normal_group_size"], 3)

    def test_threshold_is_calibrated_and_discriminative(self) -> None:
        summary = self.report["baseline_summary"]
        self.assertEqual(summary["calibration_status"], "CALIBRATED")
        self.assertLess(float(summary["dynamic_threshold"]), 1.0)

    def test_synthetic_query_is_non_exact_but_within_real_calibrated_threshold(self) -> None:
        preparation = build_synthetic_phase1_preparation()
        query = preparation.gate_evidence.query_chains[0]
        nearest = query.normal_references[0]
        self.assertGreater(nearest.chain_distance, 0)
        self.assertEqual(nearest.call_distance, nearest.chain_distance)
        self.assertEqual(nearest.asset_distance, 0)
        self.assertLessEqual(query.nearest_distance, query.dynamic_threshold)
        self.assertLess(query.dynamic_threshold, 1)
        self.assertEqual(query.quantitative_status, "PASS")

    def test_top_three_is_complete(self) -> None:
        self.assertEqual(self.report["query_summary"]["top3_count"], 3)
        self.assertEqual(len(self.report["citation_whitelist"]["reference_aliases"]), 3)

    def test_no_response_awaits_llm(self) -> None:
        result = self.report["route_results"]["NO_RESPONSE"]
        self.assertEqual(result["route"], "AWAITING_LLM")
        self.assertFalse(result["llm_response_processed"])

    def test_normal_response_early_stops(self) -> None:
        result = self.report["route_results"]["NORMAL_RESPONSE"]
        self.assertEqual(result["route"], "EARLY_NORMAL")
        self.assertIn("ALL_USABLE_CLEAN_CHAINS_LLM_NORMAL", result["reason_codes"])

    def test_suspicious_response_proceeds_phase2(self) -> None:
        result = self.report["route_results"]["SUSPICIOUS_RESPONSE"]
        self.assertEqual(result["route"], "PROCEED_PHASE2")
        self.assertIn("LLM_SUSPICIOUS", result["reason_codes"])

    def test_uncertain_response_proceeds_phase2(self) -> None:
        result = self.report["route_results"]["UNCERTAIN_RESPONSE"]
        self.assertEqual(result["route"], "PROCEED_PHASE2")
        self.assertIn("LLM_UNCERTAIN", result["reason_codes"])

    def test_invalid_response_proceeds_phase2(self) -> None:
        result = self.report["route_results"]["INVALID_RESPONSE"]
        self.assertEqual(result["route"], "PROCEED_PHASE2")
        self.assertEqual(result["reason_codes"], ["LLM_RESPONSE_INVALID"])

    def test_normal_references_have_no_amount(self) -> None:
        package = json.loads(self.report["user_message"])
        for reference in package["query_chains"][0]["normal_references"]:
            for asset in reference["asset_evidence"]:
                self.assertNotIn("amount_observation", asset)

    def test_query_amount_is_present_as_explanatory_observation(self) -> None:
        package = json.loads(self.report["user_message"])
        amounts = [
            item["amount_observation"]
            for item in package["query_chains"][0]["query_asset_evidence"]
            if item["amount_observation"] is not None
        ]
        self.assertEqual(
            amounts,
            [{"role": "query_only_explanatory_observation", "value": "123"}],
        )

    def test_output_has_no_raw_or_internal_identity(self) -> None:
        text = serialize_phase1_normality_dry_run(self.report)
        self.assertIsNone(re.search(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])", text))
        self.assertIsNone(re.search(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])", text))
        for forbidden in (
            "synthetic-query-record",
            "synthetic-normal-record-1",
            "synthetic_strict_result_chain_001",
        ):
            self.assertNotIn(forbidden, text)

    def test_output_has_no_sensitive_trace_fields(self) -> None:
        text = serialize_phase1_normality_dry_run(self.report).lower()
        for forbidden in ('"calldata"', '"input"', '"output"', '"error"', '"revertreason"'):
            self.assertNotIn(forbidden, text)
        self.assertTrue(self.report["sanitization_check"]["passed"])

    def test_selector_is_retained(self) -> None:
        self.assertIn("0x11111111", serialize_phase1_normality_dry_run(self.report))

    def test_serialization_is_stable(self) -> None:
        first = serialize_phase1_normality_dry_run(self.report)
        reversed_report = dict(reversed(tuple(self.report.items())))
        second = serialize_phase1_normality_dry_run(reversed_report)
        self.assertEqual(first, second)

    def test_write_uses_atomic_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "report.json"
            with patch("scripts.dry_run_phase1_normality.os.replace", wraps=os.replace) as replace:
                written = write_phase1_normality_dry_run(destination)
            self.assertEqual(written, destination)
            self.assertEqual(replace.call_count, 1)
            self.assertEqual(destination.read_text(encoding="utf-8"), serialize_phase1_normality_dry_run(self.report))
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_external_call_counts_are_zero(self) -> None:
        self.assertEqual(
            self.report["external_call_counts"],
            {"llm": 0, "api": 0, "rpc": 0, "network": 0},
        )

    def test_report_is_not_a_scientific_experiment_result(self) -> None:
        self.assertIs(self.report["scientific_experiment_result"], False)
        self.assertEqual(self.report["dry_run_kind"], "SYNTHETIC_OFFLINE_ENGINEERING_CHECK")

    def test_multiple_runs_are_identical(self) -> None:
        first = serialize_phase1_normality_dry_run(build_synthetic_phase1_dry_run())
        second = serialize_phase1_normality_dry_run(build_synthetic_phase1_dry_run())
        self.assertEqual(first, second)

    def test_refactor_preserves_original_dry_run_bytes(self) -> None:
        serialized = serialize_phase1_normality_dry_run(self.report)
        self.assertEqual(
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "dd4de1f210c461f43e96775dac482f5d100807c3863d1a2ff53d56b40e9e5527",
        )


if __name__ == "__main__":
    unittest.main()

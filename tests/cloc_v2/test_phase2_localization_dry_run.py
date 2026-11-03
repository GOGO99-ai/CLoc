from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dry_run_phase2_localization.py"
_SPEC = importlib.util.spec_from_file_location("dry_run_phase2_localization", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
dry = importlib.util.module_from_spec(_SPEC); sys.modules[_SPEC.name] = dry; _SPEC.loader.exec_module(dry)


class Phase2LocalizationDryRunTests(unittest.TestCase):
    def setUp(self):
        self.report = dry.build_synthetic_phase2_localization_dry_run()

    def test_metadata_and_zero_external_calls(self):
        self.assertEqual(self.report["dry_run_kind"], "SYNTHETIC_OFFLINE_PHASE2_LOCALIZATION_CHECK")
        self.assertFalse(self.report["scientific_experiment_result"])
        self.assertEqual(self.report["external_call_counts"], {"llm": 0, "rpc": 0, "network": 0})

    def test_three_valid_statuses_pass(self):
        validations = self.report["response_validations"]
        self.assertTrue(validations["valid_located"]["valid"])
        self.assertTrue(validations["valid_uncertain"]["valid"])
        self.assertTrue(validations["valid_insufficient"]["valid"])

    def test_v2_machine_decisions_and_local_explanations(self):
        decisions = self.report["decision_v2_validations"]
        for name in ("valid_located", "valid_uncertain", "valid_insufficient"):
            self.assertTrue(decisions[name]["valid"])
        materialized = self.report["materialized_v2_result"]
        self.assertFalse(materialized["provider_free_text_used"])
        self.assertEqual(materialized["explanation_source"], "LOCAL_DETERMINISTIC_EVIDENCE_SUMMARY")

    def test_topk_v3_contract_uses_only_machine_ranking_and_local_interpretation(self):
        validations = self.report["topk_v3_validations"]
        self.assertTrue(validations["valid_ranked"]["valid"])
        self.assertTrue(validations["valid_empty"]["valid"])
        self.assertFalse(validations["invalid_unknown_slice"]["valid"])
        self.assertFalse(validations["invalid_cross_slice_evidence"]["valid"])
        materialized = self.report["materialized_topk_v3_result"]
        self.assertEqual(materialized["status"], "RANKED_CANDIDATES")
        self.assertEqual(materialized["findings"][0]["role"], "PRIMARY_CANDIDATE")
        self.assertFalse(materialized["provider_free_text_used"])
        self.assertNotIn("discarded", json.dumps(materialized, ensure_ascii=False))

    def test_invalid_cases_fail_with_fixed_codes(self):
        validations = self.report["response_validations"]
        for name in ("invalid_unknown_slice", "invalid_cross_slice_evidence", "invalid_too_many_findings", "invalid_extra_field"):
            self.assertFalse(validations[name]["valid"])
            self.assertTrue(validations[name]["reason_codes"])
            self.assertIsNone(validations[name]["response"])

    def test_prompt_has_two_chains_five_edge_types_and_three_slices(self):
        package = json.loads(self.report["prompt"]["user_message"].split("EVIDENCE_PACKAGE\n", 1)[1])
        self.assertEqual(len(package["result_chains"]), 2)
        self.assertEqual({edge["edge_type"] for edge in package["evidence_edges"]}, {"CALL", "ASSET", "ORDER", "PARAM", "STATE"})
        self.assertEqual(len(package["source_slices"]), 3)

    def test_injection_is_data_and_rules_remain(self):
        self.assertTrue(self.report["prompt_injection_check"]["source_instruction_preserved_as_untrusted_json_data"])
        self.assertTrue(self.report["prompt_injection_check"]["system_rule_unchanged"])

    def test_stable_serialization_and_repeated_run(self):
        first = dry.serialize_phase2_localization_dry_run(self.report)
        second = dry.serialize_phase2_localization_dry_run(dry.build_synthetic_phase2_localization_dry_run())
        self.assertEqual(first, second)

    def test_atomic_write_uses_only_requested_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            result = dry.write_phase2_localization_dry_run(path, self.report)
            self.assertEqual(result, path); self.assertTrue(path.exists())
            self.assertEqual(len(list(Path(temp).iterdir())), 1)

    def test_sanitization_scan_passes(self):
        text = dry.serialize_phase2_localization_dry_run(self.report)
        self.assertTrue(self.report["sanitization_check"]["passed"])
        self.assertNotRegex(text, r"0x[0-9a-fA-F]{40}")
        self.assertNotIn("record_id", text)
        self.assertNotIn("reasoning_content", text)


if __name__ == "__main__":
    unittest.main()

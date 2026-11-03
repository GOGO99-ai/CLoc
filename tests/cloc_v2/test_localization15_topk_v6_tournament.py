from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from scripts.run_localization15_topk_v6_tournament import main, run_synthetic_offline_readiness


class Localization15TopKV6TournamentRunnerTests(unittest.TestCase):
    def test_synthetic_closed_loop_is_ready_and_c88_reaches_global_top5(self):
        report = run_synthetic_offline_readiness(100)
        self.assertEqual(report["readiness"]["readiness_status"], "READY_FOR_SYNTHETIC_API_SMOKE")
        self.assertEqual(report["readiness"]["input_candidate_count"], 100)
        self.assertEqual(report["readiness"]["group_candidate_counts"], [20, 20, 20, 20, 20])
        self.assertEqual(report["readiness"]["final_candidate_count"], 25)
        self.assertEqual(report["readiness"]["final_topk_count"], 5)
        self.assertEqual(report["c88_path"]["assigned_group_count"], 1)
        self.assertTrue(report["c88_path"]["entered_group_top5"])
        self.assertTrue(report["c88_path"]["entered_finalists"])
        self.assertTrue(report["c88_path"]["entered_global_top5"])

    def test_runner_is_offline_and_never_reads_environment_credentials(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("os.getenv", side_effect=AssertionError("credential read")):
            report = run_synthetic_offline_readiness(19)
        self.assertEqual(report["rpc_request_count"], 0)
        self.assertEqual(report["source_network_request_count"], 0)
        self.assertEqual(report["deepseek_request_count"], 0)
        self.assertEqual(main(["offline-synthetic"]), 0)

    def test_readiness_report_is_deterministic_and_redacted(self):
        first = run_synthetic_offline_readiness(100)
        second = run_synthetic_offline_readiness(100)
        self.assertEqual(first, second)
        encoded = json.dumps(first, ensure_ascii=False, sort_keys=True)
        self.assertNotRegex(encoded, r"0x[0-9a-fA-F]{40}")
        self.assertNotRegex(encoded, r"0x[0-9a-fA-F]{64}")
        self.assertNotIn("source_text", encoded)
        self.assertNotIn("api_key", encoded.lower())
        self.assertNotIn("http://", encoded.lower())
        self.assertNotIn("https://", encoded.lower())

    def test_small_candidate_sets_still_complete(self):
        for count in (1, 19, 20, 21, 50, 79, 97, 100):
            with self.subTest(count=count):
                report = run_synthetic_offline_readiness(count)
                self.assertEqual(report["readiness"]["readiness_status"], "READY_FOR_SYNTHETIC_API_SMOKE")
                self.assertEqual(sum(report["readiness"]["group_candidate_counts"]), count)
                self.assertLessEqual(max(report["readiness"]["group_candidate_counts"]), 20)


if __name__ == "__main__":
    unittest.main()

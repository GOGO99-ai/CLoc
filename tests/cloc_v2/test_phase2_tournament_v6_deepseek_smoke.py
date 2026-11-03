from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.smoke_phase2_tournament_v6_deepseek import (
    _atomic_json,
    run_synthetic_api_smoke,
    validate_public_report,
)
from src.cloc_v2.phase2_source_tournament import make_provider_response


class _AdaptiveFakeTransport:
    calls = 0

    def send(self, request_payload, api_key, timeout_seconds):
        type(self).calls += 1
        user = request_payload["messages"][1]["content"]
        package = json.loads(user.split("EVIDENCE_PACKAGE\n", 1)[1])
        count = package["limits"]["expected_findings"]
        findings = []
        for rank, candidate in enumerate(package["candidates"][:count], start=1):
            findings.append({
                "rank": rank,
                "source_slice_id": candidate["source_slice_id"],
                "evidence_ids": [candidate["evidence_ids"][0]],
            })
        response = make_provider_response(findings)
        response["choices"][0]["message"]["reasoning_content"] = "must never be saved"
        return response


class _GroupFailureTransport:
    calls = 0

    def send(self, request_payload, api_key, timeout_seconds):
        type(self).calls += 1
        return make_provider_response([])


class TournamentV6DeepSeekSmokeTests(unittest.TestCase):
    def setUp(self):
        _AdaptiveFakeTransport.calls = 0
        _GroupFailureTransport.calls = 0

    def test_two_valid_one_shot_requests_complete_gate2(self):
        report = run_synthetic_api_smoke("SYNTHETIC", _AdaptiveFakeTransport)
        self.assertTrue(report["gate2_success"])
        self.assertEqual(report["group_request_status"], "VALIDATED")
        self.assertEqual(report["group_output_count"], 5)
        self.assertEqual(report["final_request_status"], "VALIDATED")
        self.assertEqual(report["final_output_count"], 5)
        self.assertEqual(report["deepseek_request_count"], 2)
        self.assertEqual(report["retry_count"], 0)
        self.assertEqual(_AdaptiveFakeTransport.calls, 2)

    def test_group_failure_stops_before_final_without_retry(self):
        report = run_synthetic_api_smoke("SYNTHETIC", _GroupFailureTransport)
        self.assertFalse(report["gate2_success"])
        self.assertEqual(report["group_request_status"], "INVALID_RESPONSE")
        self.assertEqual(report["final_request_status"], "NOT_ATTEMPTED")
        self.assertEqual(report["deepseek_request_count"], 1)
        self.assertEqual(_GroupFailureTransport.calls, 1)

    def test_provider_free_text_is_not_retained(self):
        report = run_synthetic_api_smoke("SYNTHETIC", _AdaptiveFakeTransport)
        rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)
        self.assertFalse(report["provider_free_text_used"])
        self.assertNotIn("must never be saved", rendered)
        self.assertNotIn("reasoning_content", rendered)

    def test_report_is_redacted_and_atomically_written(self):
        report = run_synthetic_api_smoke("SYNTHETIC", _AdaptiveFakeTransport)
        validate_public_report(report)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            _atomic_json(path, report)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_default_import_and_nonlive_path_do_not_read_key_or_call_transport(self):
        from scripts.smoke_phase2_tournament_v6_deepseek import main

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "os.getenv", side_effect=AssertionError("credential read")
        ):
            self.assertEqual(main(["--output-dir", str(Path(directory) / "new")]), 2)
        self.assertEqual(_AdaptiveFakeTransport.calls, 0)

    def test_existing_output_directory_is_never_overwritten(self):
        from scripts.smoke_phase2_tournament_v6_deepseek import main

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "SYNTHETIC"}, clear=True):
            target = Path(directory)
            self.assertEqual(main(["--live", "--output-dir", str(target)]), 2)


if __name__ == "__main__":
    unittest.main()


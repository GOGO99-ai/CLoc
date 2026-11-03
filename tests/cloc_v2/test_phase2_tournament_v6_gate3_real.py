from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_phase2_tournament_v6_gate3_real import (
    LOCKED_NEUTRAL_ALIAS,
    SAFE_ALIAS,
    LockedGate3Sample,
    _base_prediction,
    _freeze,
    _sha,
    evaluate_frozen_gate3,
    lock_gate3_sample,
    run_gate3,
    validate_frozen,
    validate_public,
)
from scripts.run_localization15_topk_v4 import LockedLocalizationV4Manifest, LocalizationV4Sample


def _lock() -> LockedGate3Sample:
    return LockedGate3Sample(SAFE_ALIAS, 1, "0x" + "a" * 64, "b" * 64)


def _completed_prediction(lock: LockedGate3Sample) -> dict:
    prediction = _base_prediction(lock)
    prediction.update(
        status="COMPLETED",
        candidate_pool_count=2,
        candidate_rows=[
            {"candidate_alias": "C1", "relative_file": "contracts/A.sol", "function_signature": "good(uint256)", "start_line": 10, "end_line": 20, "visibility": "public"},
            {"candidate_alias": "C2", "relative_file": "contracts/B.sol", "function_signature": "other()", "start_line": 1, "end_line": 4, "visibility": "external"},
        ],
        grouping_audit={"status": "VALID"},
        groups=[{"group_alias": "G1", "candidate_count": 2, "status": "VALIDATED", "winner_candidate_ids": ["C1", "C2"]}],
        finalist_count=2,
        final_status="VALIDATED",
        global_top5=[
            {"rank": 1, "candidate_alias": "C1", "relative_file": "contracts/A.sol", "function": "good(uint256)", "predicted_line_start": 10, "predicted_line_end": 20, "line_range_source": "LOCAL_EXACT_FUNCTION_RANGE", "evidence_ids": ["E1"]},
            {"rank": 2, "candidate_alias": "C2", "relative_file": "contracts/B.sol", "function": "other()", "predicted_line_start": 1, "predicted_line_end": 4, "line_range_source": "LOCAL_EXACT_FUNCTION_RANGE", "evidence_ids": ["E1"]},
        ],
        reason_codes=["TOURNAMENT_V6_ENGINEERING_CHAIN_VALIDATED"],
    )
    return prediction


def _truth():
    return {
        "sample_alias": LOCKED_NEUTRAL_ALIAS,
        "truths": [{
            "relative_file": "contracts/A.sol",
            "function_signature": "good(uint256)",
            "function_name": "good",
            "function_name_unique_in_file": True,
            "truth_start_line": 14,
            "truth_end_line": 14,
        }],
    }


class TournamentV6Gate3RealTests(unittest.TestCase):
    def test_lock_uses_fixed_neutral_alias_but_public_manifest_only_uses_safe_alias(self):
        samples = tuple(LocalizationV4Sample(f"localization_v4_eth_{i:03d}", 1, "0x" + f"{i:064x}", i) for i in range(1, 16))
        manifest = LockedLocalizationV4Manifest(samples)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
            lock = lock_gate3_sample(path)
        self.assertEqual(lock.sample_alias, SAFE_ALIAS)
        self.assertEqual(lock.tx_hash, samples[4].tx_hash)
        public = lock.public_manifest()
        rendered = json.dumps(public)
        self.assertNotIn(LOCKED_NEUTRAL_ALIAS, rendered)
        self.assertNotIn(samples[4].tx_hash, rendered)

    def test_evaluation_distinguishes_pool_group_global_and_three_levels(self):
        lock = _lock()
        frozen = _freeze(lock, _completed_prediction(lock))
        result = evaluate_frozen_gate3(frozen, _truth())
        self.assertTrue(result["truth_function_in_candidate_pool"])
        self.assertTrue(result["truth_function_in_group_top5"])
        self.assertTrue(result["truth_function_in_global_top5"])
        self.assertEqual(result["truth_function_final_rank"], 1)
        self.assertTrue(result["file_hit"])
        self.assertTrue(result["function_hit"])
        self.assertTrue(result["line_range_hit"])

    def test_truth_loader_runs_only_after_frozen_file_exists_and_sha_validates(self):
        lock = _lock()
        calls = []

        def blind_runner(lock, output_dir, chain_config, api_key):
            self.assertFalse((output_dir / "frozen_prediction.json").exists())
            calls.append("blind")
            return _completed_prediction(lock)

        def truth_loader(lock):
            path = root / "frozen_prediction.json"
            self.assertTrue(path.exists())
            validate_frozen(path, lock)
            calls.append("truth")
            return _truth(), "c" * 64

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_gate3(lock, root, {"rpc_url": "not-used"}, "SYNTHETIC", blind_runner=blind_runner, truth_loader=truth_loader)
            self.assertEqual(calls, ["blind", "truth"])
            self.assertTrue(report["gate3_success"])
            self.assertEqual(len(list(root.iterdir())), 3)

    def test_failed_blind_inference_never_reads_truth(self):
        lock = _lock()
        prediction = _base_prediction(lock)
        prediction["reason_codes"] = ["TOURNAMENT_GROUP_INCOMPLETE"]

        def fail_truth(lock):
            raise AssertionError("truth must not be read")

        with tempfile.TemporaryDirectory() as directory:
            report = run_gate3(lock, Path(directory), {}, "", blind_runner=lambda *args, **kwargs: prediction, truth_loader=fail_truth)
        self.assertFalse(report["gate3_success"])
        self.assertIsNone(report["evaluation_truth_sha256"])

    def test_evaluate_never_changes_frozen_bytes_or_sha(self):
        lock = _lock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_gate3(lock, root, {}, "", blind_runner=lambda *args, **kwargs: _completed_prediction(lock), truth_loader=lambda lock: (_truth(), "d" * 64))
            before = (root / "frozen_prediction.json").read_bytes()
            frozen = validate_frozen(root / "frozen_prediction.json", lock)
            evaluate_frozen_gate3(frozen, _truth())
            self.assertEqual((root / "frozen_prediction.json").read_bytes(), before)
            self.assertEqual(frozen["payload_sha256"], report["frozen_prediction_sha256"])

    def test_public_artifacts_reject_identity_trace_provider_text_and_absolute_paths(self):
        lock = _lock()
        validate_public(lock.public_manifest())
        for value in (
            {"safe": "0x" + "1" * 40},
            {"provider_response": "forbidden"},
            {"safe": "D:\\secret\\A.sol"},
            {"cause": "forbidden"},
        ):
            with self.assertRaisesRegex(ValueError, "SENSITIVE"):
                validate_public(value)

    def test_nonlive_mode_never_reads_configuration_or_creates_output(self):
        from scripts.run_phase2_tournament_v6_gate3_real import main

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "os.getenv", side_effect=AssertionError("credential read")
        ):
            target = Path(directory) / "out"
            self.assertEqual(main(["--output-dir", str(target)]), 2)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.run_localization_fault10_blind_legacy_v4 as runner


def _tx(index: int) -> str:
    return "0x" + f"{index:064x}"


def _manifest() -> runner.Fault10Manifest:
    samples = tuple(runner.Fault10Sample(f"fault10_eth_{i:03d}", 1, _tx(1000 + i), i) for i in range(1, 11))
    return runner.Fault10Manifest(samples, 0, 79)


def _prediction(sample: runner.Fault10Sample, *, valid: bool) -> dict:
    return {
        "sample_alias": sample.sample_alias,
        "status": "VALID_TOPK" if valid else "INSUFFICIENT_SOURCE",
        "terminal_kind": "VALID_TOPK" if valid else "SAFE_STOP",
        "reason_codes": [],
        "profile_id": runner.PROFILE_ID,
        "profile_sha256": runner.EXPECTED_PROFILE_SHA256,
        "pre_llm_candidate_count": 3 if valid else 0,
        "prompt_candidate_count": 3 if valid else 0,
        "pre_llm_candidates": [],
        "source_resolution_mode": "EXACT_FUNCTION_SLICE" if valid else None,
        "source_selection_mode": "TRANSACTION_CALL_GRAPH_FALLBACK" if valid else None,
        "selection_mode": "LEGACY_PREFIX_SELECTION",
        "prompt_sha256": "a" * 64 if valid else None,
        "prompt_bytes": 100 if valid else 0,
        "deepseek_called": valid,
        "deepseek_status": "VALIDATED" if valid else "NOT_CALLED",
        "topk_valid": valid,
        "topk": ([{"rank": 1, "relative_file": "contracts/A.sol", "function": "f()", "slice_start_line": 10, "slice_end_line": 12, "predicted_line_start": 10, "predicted_line_end": 12, "line_range_source": "LOCAL_EXACT_FUNCTION_RANGE", "evidence_ids": ["E1"]}] if valid else []),
        "source_summary": {},
        "provider_free_text_used": False,
    }


class FakeRPC:
    def __init__(self):
        self.calls = []

    def call(self, method, params):
        self.calls.append((method, params))
        return "0x1" if method == "eth_chainId" else {"type": "CALL"}


class Fault10BlindLegacyV4Tests(unittest.TestCase):
    def test_profile_gate_is_exact(self):
        value = runner.validate_locked_profile()
        self.assertEqual(value["configuration_sha256"], runner.EXPECTED_PROFILE_SHA256)
        self.assertEqual(value["status"], "RECONSTRUCTED_AND_LOCKED")
        self.assertEqual(value["limits"], {"candidate_function_pool": 20, "final_topk": 5, "internal_function_expansion_depth": 2, "prompt_candidates": 12})
        self.assertFalse(value["deepseek_request_policy"]["retry_enabled"])

    def test_rank_namespace_matches_frozen_formula(self):
        tx = _tx(9)
        expected = hashlib.sha256(f"CLOC_FAULT10_BLIND_V1|20260824|1|{tx}".encode()).hexdigest()
        self.assertEqual(runner._selection_rank(tx), expected)

    def test_selection_is_deterministic_and_excludes_history_conflicts_and_fault007(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); dataset = root / "dataset"; prototypes = root / "prototypes"; normal = root / "normal"
            dataset.mkdir(); prototypes.mkdir(); normal.mkdir()
            for i in range(1, 80):
                (dataset / f"d{i:03d}.json").write_text(json.dumps({"platform": "Ethereum", "fault": {"transaction_hash": _tx(i), "location": "MUST_NOT_BE_READ", "cause": "MUST_NOT_BE_READ"}}), encoding="utf-8")
            historical = _tx(1)
            (prototypes / "locked_manifest.json").write_text(json.dumps({"samples": [{"tx_hash": historical}]}), encoding="utf-8")
            normal_tx = _tx(2)
            normal_manifest = {"samples": [{"chain_key": "eth", "blockchain_chain_id": 1, "tx_hash": normal_tx}]}
            checkpoint = {"samples": {f"eth:{normal_tx}": {"status": "SUCCEEDED"}}}
            (normal / "normal_manifest.json").write_text(json.dumps(normal_manifest), encoding="utf-8")
            (normal / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            first = runner.build_locked_fault10_manifest(dataset, prototypes, normal)
            second = runner.build_locked_fault10_manifest(dataset, prototypes, normal)
            hashes = {sample.tx_hash for sample in first.samples}
            self.assertEqual(first.to_dict(), second.to_dict())
            self.assertNotIn(historical, hashes)
            self.assertNotIn(normal_tx, hashes)
            self.assertNotIn(runner.FAULT007, hashes)
            self.assertEqual([sample.sample_alias for sample in first.samples], [f"fault10_eth_{i:03d}" for i in range(1, 11)])

    def test_selection_function_does_not_reference_location_or_cause(self):
        import inspect
        source = inspect.getsource(runner.build_locked_fault10_manifest)
        self.assertNotIn("location", source.lower())
        self.assertNotIn("cause", source.lower())

    def test_manifest_sha_and_profile_binding_are_stable(self):
        manifest = _manifest(); loaded = manifest.to_dict()
        self.assertEqual(loaded["profile_sha256"], runner.EXPECTED_PROFILE_SHA256)
        self.assertEqual(manifest.manifest_sha256, _manifest().manifest_sha256)
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "manifest.json"; runner._atomic(path, loaded)
            self.assertEqual(runner.load_manifest(path), manifest)

    def test_request_ledger_counts_before_calls_and_enforces_no_retry(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "checkpoint.json"; checkpoint = runner._initial_checkpoint(manifest); ledger = runner.RequestLedger(checkpoint, path)
            ledger.before("deepseek", manifest.samples[0].sample_alias)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["physical_request_counts"]["deepseek"], 1)
            with self.assertRaisesRegex(RuntimeError, "PER_SAMPLE"):
                ledger.before("deepseek", manifest.samples[0].sample_alias)
            self.assertEqual(checkpoint["physical_request_counts"]["deepseek"], 1)

    def test_fake_end_to_end_freezes_before_truth_and_counts_once(self):
        manifest = _manifest(); rpc = FakeRPC(); truth_called = []
        with tempfile.TemporaryDirectory() as value:
            output = Path(value) / "experiment"
            def factory(ledger):
                def process(sample, raw):
                    valid = sample.order <= 5
                    if valid:
                        ledger.before("deepseek", sample.sample_alias)
                    return _prediction(sample, valid=valid)
                return process
            def truth_preparer(directory, locked):
                self.assertTrue((directory / "frozen_predictions.json").exists())
                runner.validate_frozen(directory, locked)
                truth_called.append(True)
                labels = [{"sample_alias": sample.sample_alias, "evaluation_status": "AVAILABLE", "truths": [{"relative_file": "contracts/A.sol", "function_signature": "f()", "function_name": "f", "function_name_unique_in_file": True, "truth_start_line": 11, "truth_end_line": 11}], "reason_codes": []} for sample in locked.samples]
                core = {"schema_version": "localization_fault10_evaluation_truth_v1", "locked_manifest_sha256": locked.manifest_sha256, "labels": labels}
                result = {**core, "ground_truth_sha256": runner._sha(core)}; runner._atomic(directory / "evaluation_ground_truth.json", result); return result
            report = runner.run_isolated(manifest, output, rpc_client=rpc, sample_processor_factory=factory, truth_preparer=truth_preparer)
            self.assertEqual(len(rpc.calls), 11)
            self.assertEqual(sum(method == "debug_traceTransaction" for method, _ in rpc.calls), 10)
            self.assertEqual(truth_called, [True])
            self.assertEqual(report["physical_request_counts"]["rpc"], 11)
            self.assertEqual(report["physical_request_counts"]["deepseek"], 5)
            self.assertTrue(report["request_count_consistent_with_sample_sum"])
            self.assertEqual(report["strict_end_to_end_metrics"]["function_recall_at_5"], 0.5)
            self.assertEqual(report["engineering_metrics"]["terminal_coverage_rate"], 1.0)
            self.assertEqual(report["engineering_metrics"]["valid_topk_coverage_rate"], 0.5)
            self.assertEqual(len(list((output / "predictions").glob("*.json"))), 10)

    def test_unavailable_truth_stays_in_fixed_denominator(self):
        manifest = _manifest(); rpc = FakeRPC()
        with tempfile.TemporaryDirectory() as value:
            output = Path(value) / "experiment"
            def factory(ledger): return lambda sample, raw: _prediction(sample, valid=True)
            def truth_preparer(directory, locked):
                labels = [{"sample_alias": sample.sample_alias, "evaluation_status": "EVALUATION_UNAVAILABLE", "truths": [], "reason_codes": ["GROUND_TRUTH_UNAVAILABLE"]} for sample in locked.samples]
                core = {"schema_version": "localization_fault10_evaluation_truth_v1", "locked_manifest_sha256": locked.manifest_sha256, "labels": labels}
                result = {**core, "ground_truth_sha256": runner._sha(core)}; runner._atomic(directory / "evaluation_ground_truth.json", result); return result
            report = runner.run_isolated(manifest, output, rpc_client=rpc, sample_processor_factory=factory, truth_preparer=truth_preparer)
            self.assertEqual(report["strict_end_to_end_metrics"]["file_recall_at_5"], 0.0)
            self.assertIsNone(report["conditional_metrics"]["conditional_file_recall_at_5"])
            self.assertEqual(report["conditional_metrics"]["denominators"]["file"], 0)
            self.assertEqual(report["engineering_metrics"]["evaluation_unavailable_count"], 10)

    def test_non_live_mode_never_reads_config_or_creates_output(self):
        with tempfile.TemporaryDirectory() as value, patch.object(runner, "load_chain_config") as config:
            output = Path(value) / "out"
            self.assertEqual(runner.main(["--output-dir", str(output)]), 2)
            config.assert_not_called()
            self.assertFalse(output.exists())

    def test_existing_output_stops_before_profile_or_config(self):
        with tempfile.TemporaryDirectory() as value, patch.object(runner, "validate_locked_profile") as profile, patch.object(runner, "load_chain_config") as config:
            output = Path(value); self.assertEqual(runner.main(["--live", "--output-dir", str(output)]), 1)
            profile.assert_not_called(); config.assert_not_called()

    def test_public_artifacts_reject_identity_and_sensitive_fields(self):
        with self.assertRaises(ValueError): runner._validate_public({"address": "0x" + "1" * 40})
        with self.assertRaises(ValueError): runner._validate_public({"cause": "x"})
        with self.assertRaises(ValueError): runner._validate_public({"source": "contract X {}"})
        runner._validate_public({"physical_request_counts": {"source": 3}})
        with self.assertRaises(ValueError): runner._validate_public({"provider_response": {}})


if __name__ == "__main__":
    unittest.main()

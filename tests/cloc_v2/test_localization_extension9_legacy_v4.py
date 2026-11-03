from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.run_localization_extension9_legacy_v4 as runner


def _tx(index: int) -> str:
    return "0x" + f"{index:064x}"


def _manifest() -> runner.ExtensionManifest:
    return runner.ExtensionManifest(tuple(
        runner.ExtensionSample(f"extension_eth_{index:03d}", 1, _tx(1000 + index), index)
        for index in range(1, 10)
    ))


def _prediction(sample: runner.ExtensionSample, valid: bool = True) -> dict:
    topk = [{
        "rank": 1,
        "relative_file": "contracts/A.sol",
        "function": "f()",
        "slice_start_line": 10,
        "slice_end_line": 12,
        "predicted_line_start": 10,
        "predicted_line_end": 12,
        "line_range_source": "LOCAL_EXACT_FUNCTION_RANGE",
        "evidence_ids": ["E1"],
    }] if valid else []
    return {
        "sample_alias": sample.sample_alias,
        "status": "VALID_TOPK" if valid else "INSUFFICIENT_SOURCE",
        "terminal_kind": "VALID_TOPK" if valid else "SAFE_STOP",
        "reason_codes": [],
        "profile_id": runner.PROFILE_ID,
        "profile_sha256": runner.EXPECTED_PROFILE_SHA256,
        "pre_llm_candidate_count": 1 if valid else 0,
        "prompt_candidate_count": 1 if valid else 0,
        "pre_llm_candidates": [],
        "source_resolution_mode": "EXACT_FUNCTION_SLICE" if valid else None,
        "source_selection_mode": "DIRECT_EVIDENCE_SOURCE" if valid else None,
        "selection_mode": "LEGACY_PREFIX_SELECTION",
        "prompt_sha256": "a" * 64 if valid else None,
        "prompt_bytes": 100 if valid else 0,
        "deepseek_called": valid,
        "deepseek_status": "VALIDATED" if valid else "NOT_CALLED",
        "topk_valid": valid,
        "topk": topk,
        "source_summary": {},
        "provider_free_text_used": False,
    }


class FakeRPC:
    def __init__(self, chain_ok: bool = True):
        self.calls = []
        self.chain_ok = chain_ok

    def call(self, method, params):
        self.calls.append((method, params))
        if method == "eth_chainId":
            return "0x1" if self.chain_ok else "0x2"
        return {"type": "CALL"}


class Extension9LegacyV4Tests(unittest.TestCase):
    def test_profile_gate_and_fixed_limits(self):
        profile = runner.validate_locked_profile()
        self.assertEqual(profile["status"], "RECONSTRUCTED_AND_LOCKED")
        self.assertEqual(profile["configuration_sha256"], runner.EXPECTED_PROFILE_SHA256)
        self.assertEqual(profile["limits"]["candidate_function_pool"], 20)
        self.assertEqual(profile["limits"]["prompt_candidates"], 12)
        self.assertEqual(profile["limits"]["final_topk"], 5)
        self.assertFalse(profile["deepseek_request_policy"]["retry_enabled"])

    def test_fixed_aliases_map_without_location_and_are_deterministic(self):
        fault_pool = {_tx(index) for index in range(1, 80)}
        with tempfile.TemporaryDirectory() as value, patch.object(runner, "_fault_pool", return_value=fault_pool):
            prototypes = Path(value)
            first = runner.build_extension_manifest(Path("unused"), prototypes)
            second = runner.build_extension_manifest(Path("unused"), prototypes)
        expected_indexes = (2, 9, 39, 41, 47, 54, 58, 59, 77)
        self.assertEqual([sample.tx_hash for sample in first.samples], [_tx(index) for index in expected_indexes])
        self.assertEqual(first.to_dict(), second.to_dict())
        text = runner._canonical(first.to_dict()).lower()
        self.assertNotIn("location", text)
        self.assertNotIn("cause", text)
        self.assertNotIn("function", text)

    def test_historical_overlap_is_rejected(self):
        fault_pool = {_tx(index) for index in range(1, 80)}
        with tempfile.TemporaryDirectory() as value, patch.object(runner, "_fault_pool", return_value=fault_pool):
            prototypes = Path(value)
            phase2 = prototypes / "localization_old"
            phase2.mkdir()
            (phase2 / "old_manifest.json").write_text(
                json.dumps({"samples": [{"tx_hash": _tx(2)}]}), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "EXTENSION9_HISTORICAL_OVERLAP"):
                runner.build_extension_manifest(Path("unused"), prototypes)

    def test_phase1_manifest_is_ignored_and_final_phase2_queue_is_exact(self):
        with tempfile.TemporaryDirectory() as value:
            prototypes = Path(value)
            phase1 = prototypes / "phase1_only"
            phase1.mkdir()
            (phase1 / "locked_manifest.json").write_text(
                json.dumps({"samples": [{"sample_alias": "p1", "tx_hash": _tx(2)}]}),
                encoding="utf-8",
            )
            final = prototypes / "final_small"
            final.mkdir()
            (final / "locked_inference_manifest.json").write_text(
                json.dumps({"samples": [
                    {"sample_alias": "not_phase2", "tx_hash": _tx(2)},
                    {"sample_alias": "in_phase2", "tx_hash": _tx(3)},
                ]}),
                encoding="utf-8",
            )
            (final / "frozen_phase2_predictions.json").write_text(
                json.dumps({"phase2_queue": ["in_phase2"], "predictions": [
                    {"sample_alias": "in_phase2"},
                ]}),
                encoding="utf-8",
            )
            self.assertEqual(runner._phase2_history_transactions(prototypes), {_tx(3)})

    def test_prepare_writes_only_neutral_manifest(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as value, patch.object(runner, "validate_locked_profile"), patch.object(runner, "build_extension_manifest", return_value=manifest):
            output = Path(value) / "experiment"
            runner.prepare_extension9(output)
            self.assertEqual([path.name for path in output.iterdir()], ["locked_inference_manifest.json"])
            self.assertFalse((output / "evaluation_ground_truth.json").exists())
            self.assertEqual(runner.load_manifest(output / "locked_inference_manifest.json"), manifest)

    def test_request_ledger_enforces_global_and_per_sample_budgets(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as value:
            checkpoint = runner._initial_checkpoint(manifest)
            ledger = runner.ExtensionRequestLedger(checkpoint, Path(value) / "checkpoint.json")
            alias = manifest.samples[0].sample_alias
            ledger.before("rpc")
            ledger.before("rpc", alias)
            with self.assertRaisesRegex(RuntimeError, "PER_SAMPLE"):
                ledger.before("rpc", alias)
            for _ in range(3):
                ledger.before("source", alias)
            with self.assertRaisesRegex(RuntimeError, "PER_SAMPLE"):
                ledger.before("source", alias)
            ledger.before("deepseek", alias)
            with self.assertRaisesRegex(RuntimeError, "PER_SAMPLE"):
                ledger.before("deepseek", alias)

    def test_fake_infer_freezes_nine_without_truth_and_counts_once(self):
        manifest = _manifest()
        rpc = FakeRPC()
        with tempfile.TemporaryDirectory() as value:
            output = Path(value)
            runner._atomic(output / "locked_inference_manifest.json", manifest.to_dict())

            def factory(ledger):
                def process(sample, raw):
                    ledger.before("source", sample.sample_alias)
                    ledger.before("deepseek", sample.sample_alias)
                    return _prediction(sample)
                return process

            frozen = runner.run_blind_inference(
                manifest, output, rpc_client=rpc, sample_processor_factory=factory,
            )
            self.assertEqual(len(rpc.calls), 10)
            self.assertEqual(sum(method == "debug_traceTransaction" for method, _ in rpc.calls), 9)
            self.assertEqual(frozen["physical_request_counts"], {
                "rpc": 10, "source": 9, "deepseek": 9,
                "stage_a": 0, "tournament": 0, "retry": 0,
            })
            self.assertFalse((output / "evaluation_ground_truth.json").exists())
            self.assertEqual(len(list((output / "predictions").glob("*.json"))), 9)
            runner.validate_frozen(output, manifest)

    def test_chain_failure_stops_before_trace_and_freeze(self):
        manifest = _manifest()
        rpc = FakeRPC(chain_ok=False)
        with tempfile.TemporaryDirectory() as value:
            output = Path(value)
            runner._atomic(output / "locked_inference_manifest.json", manifest.to_dict())
            result = runner.run_blind_inference(
                manifest, output, rpc_client=rpc,
                sample_processor=lambda sample, raw: _prediction(sample),
            )
            self.assertEqual(result["status"], "STOPPED_CHAIN_ID")
            self.assertEqual(len(rpc.calls), 1)
            self.assertFalse((output / "frozen_predictions.json").exists())
            self.assertFalse((output / "evaluation_ground_truth.json").exists())

    def test_truth_present_blocks_infer_before_rpc(self):
        manifest = _manifest()
        rpc = FakeRPC()
        with tempfile.TemporaryDirectory() as value:
            output = Path(value)
            runner._atomic(output / "locked_inference_manifest.json", manifest.to_dict())
            runner._atomic(output / "evaluation_ground_truth.json", {"forbidden": True})
            with self.assertRaisesRegex(ValueError, "TRUTH_PRESENT"):
                runner.run_blind_inference(
                    manifest, output, rpc_client=rpc,
                    sample_processor=lambda sample, raw: _prediction(sample),
                )
            self.assertEqual(rpc.calls, [])

    def test_evaluate_reads_truth_only_after_freeze_and_combines_fault006(self):
        manifest = _manifest()
        rpc = FakeRPC()
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            output = root / "extension"
            old_root = root / "fault10"
            old_root.mkdir()
            runner._atomic(output / "locked_inference_manifest.json", manifest.to_dict())
            runner.run_blind_inference(
                manifest, output, rpc_client=rpc,
                sample_processor=lambda sample, raw: _prediction(sample),
            )
            frozen_before = (output / "frozen_predictions.json").read_bytes()
            for name in ("locked_inference_manifest.json", "frozen_predictions.json", "checkpoint.json"):
                (old_root / name).write_text("{}\n", encoding="utf-8")
            digests = {name: runner._file_sha(old_root / name) for name in (
                "locked_inference_manifest.json", "frozen_predictions.json", "checkpoint.json",
            )}
            old_manifest = {
                "profile_sha256": runner.EXPECTED_PROFILE_SHA256,
                "samples": [{"sample_alias": "fault10_eth_006", "tx_hash": _tx(5000)}],
            }
            old_prediction = _prediction(
                runner.ExtensionSample("extension_eth_006", 1, _tx(5000), 6),
            )
            old_prediction["sample_alias"] = "fault10_eth_006"
            old_frozen = {
                "profile_sha256": runner.EXPECTED_PROFILE_SHA256,
                "predictions": [old_prediction],
            }

            truth_calls = []
            def truth_provider(transactions, dataset_root, cache_root):
                self.assertTrue((output / "frozen_predictions.json").exists())
                truth_calls.append(tuple(sorted(transactions)))
                return {
                    alias: [{
                        "relative_file": "contracts/A.sol",
                        "function_signature": "f()",
                        "function_name": "f",
                        "function_name_unique_in_file": True,
                        "truth_start_line": 11,
                        "truth_end_line": 11,
                        "function_available": True,
                        "reason_codes": [],
                        "binding_role": "GROUND_TRUTH_LOCATION_ADDRESS",
                        "path_recovery_applied": False,
                    }]
                    for alias in transactions
                }

            with patch.object(runner, "_truths_for_transactions", side_effect=truth_provider), patch.object(
                runner, "validate_frozen_group",
                return_value=(old_manifest, old_frozen, {"fault10_eth_006": _tx(5000)}, digests),
            ):
                result = runner.evaluate_extension9(
                    output, dataset_root=root / "dataset", cache_root=root / "cache", fault10_root=old_root,
                )
            self.assertEqual(len(truth_calls), 2)
            self.assertEqual(result["extension9"]["fully_evaluable_count"], 9)
            self.assertEqual(result["combined10"]["evaluation_sample_count"], 10)
            self.assertEqual(result["combined10"]["strict_end_to_end_metrics"]["function_recall_at_5"], 1.0)
            self.assertEqual((output / "frozen_predictions.json").read_bytes(), frozen_before)
            self.assertTrue((output / "evaluation_ground_truth.json").exists())

    def test_evaluate_rejects_unfrozen_predictions(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as value:
            output = Path(value)
            runner._atomic(output / "locked_inference_manifest.json", manifest.to_dict())
            with self.assertRaises((OSError, ValueError)):
                runner.evaluate_extension9(output)
            self.assertFalse((output / "evaluation_ground_truth.json").exists())

    def test_non_live_infer_never_reads_external_config(self):
        with tempfile.TemporaryDirectory() as value, patch.object(runner, "load_chain_config") as config:
            output = Path(value)
            self.assertEqual(runner.main(["infer", "--output-dir", str(output)]), 2)
            config.assert_not_called()


if __name__ == "__main__":
    unittest.main()

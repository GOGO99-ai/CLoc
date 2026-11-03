from __future__ import annotations

import contextlib
import inspect
import io
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.run_phase1_balanced20_pilot as pilot
from src.cloc_v2.formal_normal_baseline_loader import load_formal_normal_baseline


def tx(value: int) -> str:
    return "0x" + f"{value:064x}"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def selection_fixture(root: Path, *, seed: int = pilot.SELECTION_SEED):
    formal = root / "formal"
    dataset = root / "dataset"
    normal_rows = [
        {"chain_key": "eth", "blockchain_chain_id": 1, "tx_hash": tx(index), "sample_id": f"n{index}"}
        for index in range(1, 41)
    ]
    states = {f"eth:{row['tx_hash']}": {"status": "SUCCEEDED"} for row in normal_rows}
    write_json(formal / "normal_manifest.json", {"manifest_sha256": "same", "samples": normal_rows})
    write_json(formal / "checkpoint.json", {"manifest_sha256": "same", "samples": states})
    for index in range(101, 131):
        write_json(
            dataset / f"case_{index}.json",
            {
                "platform": "Ethereum",
                "fault": {"transaction_hash": tx(index), "location": "must-not-be-read"},
                "cause": "must-not-be-read",
            },
        )
    normal_exclusions = (root / "old_normal_3.json", root / "old_normal_10.json")
    write_json(normal_exclusions[0], {"samples": [{"tx_hash": tx(index)} for index in range(1, 4)]})
    write_json(normal_exclusions[1], {"samples": [{"tx_hash": tx(index)} for index in range(4, 14)]})
    fault_exclusions = (root / "old_fault.json",)
    write_json(fault_exclusions[0], {"samples": [{"tx_hash": tx(index)} for index in range(101, 104)]})
    return pilot.build_balanced20_selection(
        formal,
        dataset,
        selection_seed=seed,
        excluded_normal_manifests=normal_exclusions,
        excluded_fault_manifests=fault_exclusions,
        expected_succeeded_normal_count=40,
    )


@dataclass
class FakePrepared:
    sample_alias: str


class FakeRpc:
    def __init__(self, fail_at: int | None = None, chain_id: str = "0x1"):
        self.fail_at = fail_at
        self.chain_id = chain_id
        self.request_count = 0
        self.methods = []

    def call(self, method, params):
        self.request_count += 1
        self.methods.append(method)
        if self.fail_at == self.request_count:
            raise RuntimeError("RPC_NETWORK_ERROR")
        return self.chain_id if method == "eth_chainId" else {"frame": self.request_count}


def row(alias: str, *, direct=False, eligible=False, llm=False, suspicious=False):
    reasons = ["KNOWN_NORMAL_TRANSACTION_SELF_MATCH"] if direct else []
    if suspicious:
        reasons.append("LLM_SUSPICIOUS")
    return {
        "sample_alias": alias,
        "stage_counts": {"strict_chain_count": 1},
        "coverage_rate": "1.0000",
        "critical_missing_count": 0,
        "clean_ready_chain_count": 1 if direct or eligible else 0,
        "exact_matched_clean_chain_count": 1 if direct else 0,
        "clean_unmatched_chain_count": 0,
        "uncertain_chain_count": 0 if direct else 1,
        "direct_normal": direct,
        "query_normal_identity_status": "KNOWN_NORMAL_ORIGIN" if direct else "UNRESOLVED",
        "known_transaction_direct_normal": direct,
        "structural_exact_collision": False,
        "decision_kind": "KNOWN_TRANSACTION_DIRECT_NORMAL" if direct else "CONTINUE_STANDARD_PHASE1",
        "preparation_status": "KNOWN_TRANSACTION_DIRECT_NORMAL" if direct else ("READY_FOR_LLM" if eligible else "INELIGIBLE"),
        "eligibility": "DIRECT_NORMAL" if direct else ("ELIGIBLE" if eligible else "INELIGIBLE"),
        "llm_called": llm,
        "deepseek_response_status": "VALIDATED" if llm else "NOT_REQUESTED",
        "deepseek_usage": {"total_tokens": 3} if llm else None,
        "final_route": "PROCEED_PHASE2" if suspicious or (not direct and not llm) else "EARLY_NORMAL",
        "reason_codes": reasons,
    }


def router_for(mode: str):
    def route(prepared, deepseek_runner):
        if mode == "direct":
            return row(prepared.sample_alias, direct=True), 0, False
        if mode == "llm_normal":
            deepseek_runner(prepared)
            return row(prepared.sample_alias, eligible=True, llm=True), 1, False
        if mode == "llm_suspicious":
            deepseek_runner(prepared)
            return row(prepared.sample_alias, eligible=True, llm=True, suspicious=True), 1, False
        return row(prepared.sample_alias), 0, False
    return route


class Phase1Balanced20PilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.selection = selection_fixture(self.root)
        self.loaded = load_formal_normal_baseline()

    def tearDown(self):
        self.temporary.cleanup()

    def write_locked(self, output: Path | None = None):
        output = output or self.root / "output"
        pilot.write_balanced20_selection(output, self.selection)
        return output

    def run_fake(self, *, mode="direct", rpc=None, output=None, retry_failed=False):
        output = self.write_locked(output)
        runner = MagicMock(return_value=object())
        result = pilot.run_balanced20_inference(
            self.selection.inference_manifest,
            live=True,
            output_dir=output,
            rpc_client=rpc or FakeRpc(),
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: FakePrepared(sample.sample_alias),
            sample_router=router_for(mode),
            retry_failed=retry_failed,
        )
        return result, runner, output

    def test_fixed_seed_selects_stable_balanced_unique_samples(self):
        again = selection_fixture(self.root)
        self.assertEqual(
            [item.tx_hash for item in self.selection.inference_manifest.samples],
            [item.tx_hash for item in again.inference_manifest.samples],
        )
        self.assertEqual(len(set(self.selection.normal_hashes)), 10)
        self.assertEqual(len(set(self.selection.fault_hashes)), 10)
        self.assertTrue(set(self.selection.normal_hashes).isdisjoint(self.selection.fault_hashes))
        self.assertEqual(list(self.selection.evaluation_truth.labels.values()).count("NORMAL"), 10)
        self.assertEqual(list(self.selection.evaluation_truth.labels.values()).count("FAULT"), 10)

    def test_old_samples_are_all_excluded(self):
        self.assertTrue(set(self.selection.normal_hashes).isdisjoint({tx(index) for index in range(1, 14)}))
        self.assertTrue(set(self.selection.fault_hashes).isdisjoint({tx(index) for index in range(101, 104)}))

    def test_seed_change_changes_selection(self):
        changed = selection_fixture(self.root, seed=20260815)
        self.assertNotEqual(
            [item.tx_hash for item in self.selection.inference_manifest.samples],
            [item.tx_hash for item in changed.inference_manifest.samples],
        )

    def test_selection_does_not_consult_evidence_or_history(self):
        source = inspect.getsource(pilot.build_balanced20_selection)
        for forbidden in ("feature_records", "stage_counts", "main_anchor", "READY", "threshold"):
            self.assertNotIn(forbidden, source)

    def test_neutral_manifest_aliases_do_not_reveal_truth(self):
        payload = self.selection.inference_manifest.to_dict()
        for item in payload["samples"]:
            self.assertRegex(item["sample_alias"], r"^eval_eth_\d{3}$")
            self.assertNotIn("label", item)
            self.assertNotIn("ground_truth", item)

    def test_inference_api_has_no_truth_parameter(self):
        parameters = inspect.signature(pilot.run_balanced20_inference).parameters
        self.assertNotIn("truth", parameters)
        self.assertNotIn("evaluation_truth", parameters)

    def test_reused_preparer_resolves_identity_from_formal_loader_not_neutral_alias(self):
        source = inspect.getsource(pilot.prepare_normal_pilot_sample)
        self.assertIn("resolve_formal_normal_transaction_identity", source)
        self.assertIn("sample.tx_hash", source)
        self.assertNotIn("sample.source_sample_id", source)

    def test_direct_report_uses_known_transaction_identity_kind(self):
        result = pilot._inference_result(row("eval_eth_001", direct=True), False)
        self.assertEqual(result["query_normal_identity_status"], "KNOWN_NORMAL_ORIGIN")
        self.assertTrue(result["known_transaction_direct_normal"])
        self.assertFalse(result["structural_exact_collision"])
        self.assertEqual(result["decision_kind"], "KNOWN_TRANSACTION_DIRECT_NORMAL")

    def test_default_mode_makes_no_external_request(self):
        rpc = FakeRpc()
        result = pilot.run_balanced20_inference(self.selection.inference_manifest, live=False, rpc_client=rpc)
        self.assertEqual(result.rpc_request_count, 0)
        self.assertEqual(rpc.request_count, 0)

    def test_cli_without_live_does_not_load_config(self):
        with patch.object(pilot, "_load_live_config") as loader, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main([]), 2)
        loader.assert_not_called()

    def test_direct_match_never_calls_deepseek(self):
        result, runner, _ = self.run_fake(mode="direct")
        runner.assert_not_called()
        self.assertEqual(result.rpc_request_count, 21)
        self.assertEqual(result.deepseek_request_count, 0)
        self.assertTrue(all(item["phase1_conclusion"] == "NORMAL_DIRECT" for item in result.sample_results))

    def test_eligible_gray_sample_calls_deepseek_once_each(self):
        result, runner, _ = self.run_fake(mode="llm_normal")
        self.assertEqual(runner.call_count, 20)
        self.assertEqual(result.deepseek_request_count, 20)
        self.assertTrue(all(item["phase1_conclusion"] == "NORMAL_LLM" for item in result.sample_results))

    def test_quantitative_or_evidence_ineligible_does_not_call_deepseek(self):
        result, runner, _ = self.run_fake(mode="ineligible")
        runner.assert_not_called()
        self.assertTrue(all(item["phase1_conclusion"] == "UNCERTAIN_PHASE2" for item in result.sample_results))

    def test_rpc_and_deepseek_hard_limits(self):
        result, runner, _ = self.run_fake(mode="llm_normal")
        self.assertEqual((result.rpc_request_count, result.deepseek_request_count), (21, 20))
        self.assertEqual(runner.call_count, 20)

    def test_network_error_stops_and_retains_checkpoint(self):
        rpc = FakeRpc(fail_at=4)
        result, _, output = self.run_fake(mode="direct", rpc=rpc)
        self.assertEqual(result.run_status, "STOPPED")
        self.assertEqual(result.stop_code, "RPC_NETWORK_ERROR")
        checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(item["status"] == "SUCCEEDED" for item in checkpoint["samples"].values()), 2)
        self.assertEqual(sum(item["status"] == "FAILED" for item in checkpoint["samples"].values()), 1)
        self.assertEqual(len(list((output / "samples").glob("*.json"))), 2)

    def test_restart_skips_success_and_failed_without_explicit_retry(self):
        output = self.root / "resume"
        first, _, _ = self.run_fake(mode="direct", rpc=FakeRpc(fail_at=4), output=output)
        self.assertEqual(first.run_status, "STOPPED")
        second_rpc = FakeRpc()
        second, _, _ = self.run_fake(mode="direct", rpc=second_rpc, output=output)
        self.assertEqual(second_rpc.request_count, 18)  # chain-id plus 17 still-pending samples
        checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(item["status"] == "SUCCEEDED" for item in checkpoint["samples"].values()), 19)
        self.assertEqual(sum(item["status"] == "FAILED" for item in checkpoint["samples"].values()), 1)

    def test_manifest_drift_refuses_resume(self):
        output = self.write_locked(self.root / "drift")
        changed = selection_fixture(self.root, seed=20260815)
        with self.assertRaisesRegex(ValueError, "BALANCED20_MANIFEST_DRIFT"):
            pilot.run_balanced20_inference(
                changed.inference_manifest,
                live=True,
                output_dir=output,
                rpc_client=FakeRpc(),
                deepseek_runner=MagicMock(),
                loaded_baseline=self.loaded,
            )

    def test_deepseek_runner_receives_no_hash_or_truth(self):
        seen = []

        def runner(prepared):
            seen.append(prepared)
            return object()

        output = self.write_locked(self.root / "no_leak")
        result = pilot.run_balanced20_inference(
            self.selection.inference_manifest,
            live=True,
            output_dir=output,
            rpc_client=FakeRpc(),
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: FakePrepared(sample.sample_alias),
            sample_router=router_for("llm_normal"),
        )
        self.assertEqual(len(seen), 20)
        joined = repr(seen)
        for sample in self.selection.inference_manifest.samples:
            self.assertNotIn(sample.tx_hash, joined)
        self.assertNotIn("NORMAL", joined)
        self.assertNotIn("FAULT", joined)
        self.assertEqual(result.run_status, "COMPLETED")

    def test_metrics_treat_fault_phase2_as_abstention_not_detection(self):
        results = []
        for sample in self.selection.inference_manifest.samples:
            truth = self.selection.evaluation_truth.labels[sample.sample_alias]
            if truth == "NORMAL":
                results.append(pilot._inference_result(row(sample.sample_alias, direct=True), False))
            else:
                results.append(pilot._inference_result(row(sample.sample_alias), False))
        run = pilot.Balanced20RunResult("COMPLETED", self.selection.inference_manifest, tuple(results), 21, 0, None)
        report = pilot.build_balanced20_evaluation_report(run, self.selection.evaluation_truth)
        metrics = report["metrics"]
        self.assertEqual(metrics["normal_early_count"], 10)
        self.assertEqual(metrics["fault_suspicious_count"], 0)
        self.assertEqual(metrics["fault_safe_phase2_count"], 10)
        self.assertEqual(metrics["determinate_accuracy"], "1.0000")
        self.assertEqual(metrics["determinate_coverage"], "0.5000")

    def test_metrics_count_fault_normal_as_severe_error(self):
        results = [pilot._inference_result(row(sample.sample_alias, direct=True), False) for sample in self.selection.inference_manifest.samples]
        run = pilot.Balanced20RunResult("COMPLETED", self.selection.inference_manifest, tuple(results), 21, 0, None)
        report = pilot.build_balanced20_evaluation_report(run, self.selection.evaluation_truth)
        self.assertEqual(report["metrics"]["fault_false_normal_count"], 10)
        self.assertEqual(report["metrics"]["fault_false_release_rate"], "1.0000")
        self.assertEqual(report["metrics"]["determinate_accuracy"], "0.5000")

    def test_suspicious_fault_is_explicit_detection(self):
        results = []
        for sample in self.selection.inference_manifest.samples:
            truth = self.selection.evaluation_truth.labels[sample.sample_alias]
            source = row(sample.sample_alias, direct=True) if truth == "NORMAL" else row(sample.sample_alias, eligible=True, llm=True, suspicious=True)
            results.append(pilot._inference_result(source, False))
        run = pilot.Balanced20RunResult("COMPLETED", self.selection.inference_manifest, tuple(results), 21, 10, None)
        report = pilot.build_balanced20_evaluation_report(run, self.selection.evaluation_truth)
        self.assertEqual(report["metrics"]["fault_suspicious_count"], 10)
        self.assertEqual(report["metrics"]["determinate_accuracy"], "1.0000")
        self.assertEqual(report["metrics"]["determinate_coverage"], "1.0000")

    def test_report_and_sample_artifacts_are_sanitized(self):
        result, _, output = self.run_fake(mode="direct")
        report = pilot.build_balanced20_evaluation_report(result, self.selection.evaluation_truth)
        text = pilot.serialize_balanced20_report(report)
        for sample in self.selection.inference_manifest.samples:
            self.assertNotIn(sample.tx_hash, text)
        for forbidden in ("reasoning_content", "calldata", "rpc_url", "api_key"):
            self.assertNotIn(forbidden, text.lower())
        self.assertEqual(len(list((output / "samples").glob("*.json"))), 20)

    def test_selection_and_checkpoint_writes_are_atomic_and_separate(self):
        output = self.write_locked(self.root / "atomic")
        self.assertTrue((output / "locked_inference_manifest.json").is_file())
        truth = json.loads((output / "evaluation_truth.json").read_text(encoding="utf-8"))
        inference = json.loads((output / "locked_inference_manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("ground_truth", json.dumps(inference))
        self.assertNotIn("tx_hash", json.dumps(truth))
        self.assertEqual(list(output.glob("*.tmp")), [])

    def test_actual_workspace_selection_is_balanced_and_reproducible(self):
        actual = pilot.build_balanced20_selection()
        again = pilot.build_balanced20_selection()
        self.assertEqual(actual.inference_manifest.manifest_sha256, again.inference_manifest.manifest_sha256)
        self.assertEqual(len(actual.normal_hashes), 10)
        self.assertEqual(len(actual.fault_hashes), 10)


if __name__ == "__main__":
    unittest.main()

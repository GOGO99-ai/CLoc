from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.run_phase1_normal_pilot as pilot
from scripts.dry_run_phase1_normality import build_synthetic_phase1_preparation
from src.cloc_v2.formal_normal_baseline_loader import (
    FormalNormalTransactionIdentity,
    load_formal_normal_baseline,
)
from src.cloc_v2.normal_features import ChainFeatureExtractionStatus, NormalFeatureExtractionResult
from src.cloc_v2.phase1_direct_normal import evaluate_phase1_direct_normal_match
from src.cloc_v2.phase1_normality_bridge import Phase1NormalityPreparation
from src.cloc_v2.phase1_normality_deepseek import Phase1DeepSeekConfig, execute_phase1_deepseek_review
from src.cloc_v2.phase1_normality_orchestration import evaluate_phase1_eligibility
from src.cloc_v2.phase1_normality_prompt import RESPONSE_SCHEMA_VERSION
from src.cloc_v2.strict_result_chains import ResultChainStatus, StrictResultChain, StrictResultChainsResult


SECRET_KEY = "fake-key-never-persist"
SECRET_REASONING = "fake-reasoning-never-persist"


def strict_and_features(record):
    chain = StrictResultChain(
        strict_chain_id="strict_result_chain_001",
        result_id="result-1",
        result_node_id="node-1",
        result_fact_ids=("fact-1",),
        result_type="R_OUTFLOW",
        asset_type="TOKEN",
        asset_address=next(iter(record.asset_behaviors)).asset_identifier,
        from_address=record.main_anchor.effective_address,
        to_address="0x" + "f" * 40,
        amount_raw="1",
        result_uncertainties=(),
        node_ids=("node-1",),
        edge_ids=(),
        anchor=None,
        status="READY",
        uncertainties=(),
    )
    strict = StrictResultChainsResult(
        blockchain_chain_id=1,
        edges=(),
        chains=(chain,),
        result_statuses=(ResultChainStatus("result-1", "node-1", (chain.strict_chain_id,), "READY", ()),),
        critical_uncertainties=(),
        coverage={"result_candidate_count": 1, "coverage_rate": "1.0000", "critical_missing_count": 0},
    )
    status = ChainFeatureExtractionStatus(
        strict_chain_id=chain.strict_chain_id,
        record_id=record.record_id,
        status="EXTRACTED_CLEAN",
        record_created=True,
        reasons=(),
        issues=(),
    )
    extracted = NormalFeatureExtractionResult(
        transaction_record_id="query",
        records=(record,),
        chain_statuses=(status,),
        input_chain_count=1,
        record_count=1,
        clean_record_count=1,
        uncertain_record_count=0,
        rejected_chain_count=0,
    )
    return strict, extracted


def prepared(sample_alias, loaded, *, direct=True, eligible=True, self_reference=False, collision=False):
    if direct:
        normal = loaded.baseline.groups[0].records[0]
        query = replace(normal, record_id=f"{sample_alias}:query")
        strict, extracted = strict_and_features(query)
        identity = (
            FormalNormalTransactionIdentity(
                "NOT_IN_NORMAL_LIBRARY", ("QUERY_TRANSACTION_NOT_IN_NORMAL_LIBRARY",), False
            )
            if collision
            else FormalNormalTransactionIdentity(
                "KNOWN_NORMAL_ORIGIN", ("QUERY_NORMAL_ORIGIN_RESOLVED",), True,
                _normal_origin_id=loaded.normal_record_transaction_ids[normal.record_id],
                _normalized_transaction_hash="0x" + "1" * 64,
            )
        )
        direct_result = evaluate_phase1_direct_normal_match(
            strict, extracted, loaded.baseline, identity, loaded.normal_record_transaction_ids
        )
        preparation = build_synthetic_phase1_preparation() if collision else None
    else:
        empty_strict = StrictResultChainsResult(1, (), (), (), (), {"result_candidate_count": 0, "coverage_rate": "0.0000", "critical_missing_count": 0})
        empty_features = NormalFeatureExtractionResult("query", (), (), 0, 0, 0, 0, 0)
        identity = FormalNormalTransactionIdentity(
            "NOT_IN_NORMAL_LIBRARY", ("QUERY_TRANSACTION_NOT_IN_NORMAL_LIBRARY",), False
        )
        direct_result = evaluate_phase1_direct_normal_match(
            empty_strict, empty_features, loaded.baseline, identity, loaded.normal_record_transaction_ids
        )
        preparation = build_synthetic_phase1_preparation()
        if not eligible:
            gate = replace(preparation.gate_evidence, result_candidate_coverage_rate="0")
            eligibility = evaluate_phase1_eligibility(gate, preparation.prompt)
            preparation = Phase1NormalityPreparation("INELIGIBLE", gate, eligibility, None, ())
    return pilot.PreparedNormalPilotEvidence(
        sample_alias=sample_alias,
        stage_counts={
            "result_candidate_count": 1,
            "strict_chain_count": 1,
            "strict_ready_count": 1,
            "strict_uncertain_count": 0,
        },
        coverage_rate="1.0000",
        critical_missing_count=0,
        direct_match=direct_result,
        self_reference_match=self_reference,
        preparation=preparation,
    )


def decision_response(preparation):
    scope = preparation.prompt.citation_whitelist.chain_scopes[0]
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "chain_decisions": [{
            "chain_alias": scope.chain_alias,
            "decision": "NORMAL",
            "reference_assessment": "CONSISTENT",
            "supporting_reference_aliases": [scope.reference_aliases[0]],
            "evidence_ids": [scope.query_call_evidence_ids[0], scope.reference_call_evidence_ids[0]],
            "reason_codes": ["CONSISTENT_CALL_SEQUENCE", "WITHIN_CALIBRATED_THRESHOLD"],
            "brief_rationale": "The supplied CALL evidence is consistent.",
        }],
    }


class FakeTransport:
    def __init__(self):
        self.calls = 0
        self.preparation = None

    def send(self, payload, api_key, timeout):
        self.calls += 1
        return {
            "model": "deepseek-v4-pro-test",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(decision_response(self.preparation)),
                    "reasoning_content": SECRET_REASONING,
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


def deepseek_runner():
    transport = FakeTransport()

    def run(preparation):
        transport.preparation = preparation
        return execute_phase1_deepseek_review(preparation, Phase1DeepSeekConfig(), transport, SECRET_KEY)

    run.transport = transport
    return run


class FakeRpc:
    def __init__(self, *, chain_id="0x1", fail_at=None):
        self.chain_id = chain_id
        self.fail_at = fail_at
        self.calls = []

    def call(self, method, params):
        self.calls.append((method, params))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("RPC_NETWORK_ERROR")
        return self.chain_id if method == "eth_chainId" else {"fake_frame": True}


class Phase1NormalPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = pilot.build_locked_normal_manifest()
        cls.loaded = load_formal_normal_baseline()

    def test_selection_is_exact_stable_and_fair(self):
        again = pilot.build_locked_normal_manifest()
        self.assertEqual(
            [item.tx_hash for item in self.manifest.samples],
            [
                "0x01e6371b60c2ca36b3673bab9edf4caeba03e6dc756c317a90f856c5735cd2c4",
                "0x01f5ebdf91aff85812f5f9305944fd59b586c527c2ff35068394caeecbc8274a",
                "0x01f8b613d9f87f0f7fcde83c35a550b4fc758e8d218086de8ab39f31db9e72b0",
            ],
        )
        self.assertEqual(self.manifest.manifest_sha256, again.manifest_sha256)
        self.assertEqual([item.sample_alias for item in self.manifest.samples], ["normal_pilot_eth_001", "normal_pilot_eth_002", "normal_pilot_eth_003"])

    def test_selection_reads_only_manifest_and_checkpoint(self):
        source = Path(pilot.__file__).read_text(encoding="utf-8")
        selection = source[source.index("def build_locked_normal_manifest"):source.index("def prepare_normal_pilot_sample")]
        self.assertIn("normal_manifest.json", selection)
        self.assertIn("checkpoint.json", selection)
        self.assertNotIn("feature_records", selection)
        self.assertNotIn("sample_artifacts", selection)

    def test_random10_selection_is_stable_unique_and_excludes_prior_three(self):
        chosen = pilot.build_locked_normal_manifest(
            selection_mode="random10", selection_seed=20260814
        )
        again = pilot.build_locked_normal_manifest(
            selection_mode="random10", selection_seed=20260814
        )
        hashes = [item.tx_hash for item in chosen.samples]
        self.assertEqual(hashes, [item.tx_hash for item in again.samples])
        self.assertEqual(len(hashes), 10)
        self.assertEqual(len(set(hashes)), 10)
        self.assertTrue(set(hashes).isdisjoint(pilot.PRIOR_LEXICOGRAPHIC3_HASHES))
        self.assertEqual(
            [item.sample_alias for item in chosen.samples],
            [f"normal_random10_eth_{index:03d}" for index in range(1, 11)],
        )
        self.assertEqual(chosen.selection_seed, 20260814)
        self.assertEqual(chosen.selection_algorithm, pilot.RANDOM10_ALGORITHM)

    def test_random10_selection_changes_with_seed_and_default_is_unchanged(self):
        first = pilot.build_locked_normal_manifest(selection_mode="random10", selection_seed=20260814)
        changed = pilot.build_locked_normal_manifest(selection_mode="random10", selection_seed=20260815)
        self.assertNotEqual([item.tx_hash for item in first.samples], [item.tx_hash for item in changed.samples])
        default = pilot.build_locked_normal_manifest()
        self.assertEqual([item.tx_hash for item in default.samples], [item.tx_hash for item in self.manifest.samples])

    def test_random_selection_only_reads_manifest_and_checkpoint(self):
        source = Path(pilot.__file__).read_text(encoding="utf-8")
        selection = source[source.index("def build_locked_normal_manifest"):source.index("def prepare_normal_pilot_sample")]
        self.assertNotIn("feature_records", selection)
        self.assertNotIn("stage_counts", selection)
        self.assertNotIn("main_anchor", selection)

    def test_default_mode_is_fully_offline(self):
        rpc = FakeRpc()
        runner = MagicMock()
        result = pilot.run_phase1_normal_pilot(self.manifest, live=False, rpc_client=rpc, deepseek_runner=runner)
        self.assertEqual(rpc.calls, [])
        runner.assert_not_called()
        self.assertEqual(result.report["external_call_counts"], {"rpc": 0, "deepseek": 0, "network": 0})

    def test_cli_without_live_does_not_load_config(self):
        with patch.object(pilot, "_load_live_chain_config") as loader, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main([]), 2)
        loader.assert_not_called()

    def test_chain_mismatch_stops_after_one_rpc(self):
        rpc = FakeRpc(chain_id="0x38")
        runner = MagicMock()
        result = pilot.run_phase1_normal_pilot(self.manifest, live=True, rpc_client=rpc, deepseek_runner=runner, loaded_baseline=self.loaded)
        self.assertEqual(len(rpc.calls), 1)
        runner.assert_not_called()
        self.assertEqual(result.report["run_status"], "STOPPED")

    def test_exact_match_routes_early_normal_without_deepseek(self):
        runner = MagicMock()
        row, used, error = pilot.route_normal_pilot_sample(prepared("normal_pilot_eth_001", self.loaded, direct=True, self_reference=True), runner)
        self.assertEqual((used, error), (0, False))
        runner.assert_not_called()
        self.assertEqual(row["final_route"], "EARLY_NORMAL")
        self.assertTrue(row["self_reference_match"])
        self.assertEqual(row["clean_ready_chain_count"], 1)
        self.assertEqual(row["exact_matched_clean_chain_count"], 1)
        self.assertEqual(row["uncertain_chain_count"], 0)

    def test_direct_miss_continues_existing_llm_flow(self):
        runner = deepseek_runner()
        row, used, error = pilot.route_normal_pilot_sample(prepared("normal_pilot_eth_001", self.loaded, direct=False), runner)
        self.assertEqual((used, error), (1, False))
        self.assertEqual(row["final_route"], "EARLY_NORMAL")
        self.assertEqual(runner.transport.calls, 1)

    def test_cross_transaction_exact_collision_goes_phase2_without_deepseek(self):
        runner = MagicMock()
        row, used, error = pilot.route_normal_pilot_sample(
            prepared("normal_pilot_eth_001", self.loaded, direct=True, collision=True),
            runner,
        )
        self.assertEqual((used, error), (0, False))
        self.assertEqual(row["decision_kind"], "STRUCTURAL_EXACT_COLLISION_PHASE2")
        self.assertEqual(row["final_route"], "PROCEED_PHASE2")
        runner.assert_not_called()

    def test_direct_miss_and_ineligible_goes_phase2_without_llm(self):
        runner = MagicMock()
        row, used, error = pilot.route_normal_pilot_sample(prepared("normal_pilot_eth_001", self.loaded, direct=False, eligible=False), runner)
        self.assertEqual((used, error), (0, False))
        runner.assert_not_called()
        self.assertEqual(row["final_route"], "PROCEED_PHASE2")

    def test_rpc_cap_four_and_deepseek_cap_three(self):
        runner = deepseek_runner()
        rpc = FakeRpc()
        result = pilot.run_phase1_normal_pilot(
            self.manifest,
            live=True,
            rpc_client=rpc,
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: prepared(sample.sample_alias, loaded, direct=False),
        )
        self.assertEqual(len(rpc.calls), 4)
        self.assertEqual(runner.transport.calls, 3)
        self.assertEqual(result.report["request_limits"], {"rpc": 4, "deepseek": 3})

    def test_random10_rpc_and_deepseek_caps_are_eleven_and_ten(self):
        manifest = pilot.build_locked_normal_manifest(selection_mode="random10", selection_seed=20260814)
        runner = deepseek_runner()
        rpc = FakeRpc()
        result = pilot.run_phase1_normal_pilot(
            manifest,
            live=True,
            rpc_client=rpc,
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: prepared(sample.sample_alias, loaded, direct=False),
        )
        self.assertEqual(len(rpc.calls), 11)
        self.assertEqual(runner.transport.calls, 10)
        self.assertEqual(result.report["request_limits"], {"rpc": 11, "deepseek": 10})

    def test_random10_direct_normal_never_calls_deepseek(self):
        manifest = pilot.build_locked_normal_manifest(selection_mode="random10", selection_seed=20260814)
        runner = MagicMock()
        result = pilot.run_phase1_normal_pilot(
            manifest,
            live=True,
            rpc_client=FakeRpc(),
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: prepared(sample.sample_alias, loaded, direct=True),
        )
        runner.assert_not_called()
        self.assertEqual(result.report["summary"]["early_normal_transaction_count"], 10)

    def test_random10_no_clean_evidence_falls_back_without_llm(self):
        manifest = pilot.build_locked_normal_manifest(selection_mode="random10", selection_seed=20260814)
        runner = MagicMock()
        result = pilot.run_phase1_normal_pilot(
            manifest,
            live=True,
            rpc_client=FakeRpc(),
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: prepared(sample.sample_alias, loaded, direct=False, eligible=False),
        )
        runner.assert_not_called()
        self.assertEqual(result.report["summary"]["phase2_transaction_count"], 10)

    def test_external_error_stops_without_retry(self):
        rpc = FakeRpc(fail_at=2)
        result = pilot.run_phase1_normal_pilot(
            self.manifest,
            live=True,
            rpc_client=rpc,
            deepseek_runner=deepseek_runner(),
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: prepared(sample.sample_alias, loaded),
        )
        self.assertEqual(len(rpc.calls), 2)
        self.assertEqual(result.report["run_status"], "STOPPED")
        self.assertEqual(result.report["sample_results"][1]["preparation_status"], "NOT_RUN")

    def test_fake_rpc_and_direct_match_end_to_end(self):
        runner = MagicMock()
        result = pilot.run_phase1_normal_pilot(
            self.manifest,
            live=True,
            rpc_client=FakeRpc(),
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: prepared(sample.sample_alias, loaded, direct=True, self_reference=True),
        )
        runner.assert_not_called()
        self.assertEqual(result.report["summary"]["early_normal_transaction_count"], 3)
        self.assertEqual(result.report["external_call_counts"], {"rpc": 4, "deepseek": 0, "network": 4})

    def test_report_is_sanitized_and_declares_non_generalization(self):
        result = pilot.run_phase1_normal_pilot(self.manifest, live=False)
        text = pilot.serialize_phase1_normal_pilot_report(result.report)
        self.assertIn("CLOSED_SET_NORMAL_REIDENTIFICATION_ENGINEERING_CHECK", text)
        self.assertFalse(result.report["scientific_generalization_result"])
        for sample in self.manifest.samples:
            self.assertNotIn(sample.tx_hash, text)
            self.assertNotIn(sample.source_sample_id, text)
        self.assertNotIn(SECRET_KEY, text)
        self.assertNotIn(SECRET_REASONING, text)

    def test_atomic_write_uses_only_two_files(self):
        result = pilot.run_phase1_normal_pilot(self.manifest, live=False)
        with tempfile.TemporaryDirectory() as directory:
            paths = pilot.write_phase1_normal_pilot_artifacts(directory, result)
            self.assertEqual({item.name for item in paths}, {"locked_normal_manifest.json", "phase1_normal_pilot_report.json"})
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_serialization_is_stable(self):
        result = pilot.run_phase1_normal_pilot(self.manifest, live=False)
        self.assertEqual(pilot.serialize_locked_normal_manifest(self.manifest), pilot.serialize_locked_normal_manifest(self.manifest))
        self.assertEqual(pilot.serialize_phase1_normal_pilot_report(result.report), pilot.serialize_phase1_normal_pilot_report(result.report))


if __name__ == "__main__":
    unittest.main()

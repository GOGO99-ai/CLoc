from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import scripts.screen_phase1_fault_llm_eligibility as screen
from scripts.dry_run_phase1_normality import build_synthetic_phase1_preparation
from src.cloc_v2.formal_normal_baseline_loader import (
    FormalNormalTransactionIdentity,
    load_formal_normal_baseline,
)
from src.cloc_v2.normal_features import ChainFeatureExtractionStatus, NormalFeatureExtractionResult
from src.cloc_v2.phase1_direct_normal import evaluate_phase1_direct_normal_match
from src.cloc_v2.strict_result_chains import ResultChainStatus, StrictResultChain, StrictResultChainsResult


def _empty_direct(loaded):
    strict = StrictResultChainsResult(1, (), (), (), (), {"coverage_rate": "0.0000", "critical_missing_count": 0})
    features = NormalFeatureExtractionResult("query", (), (), 0, 0, 0, 0, 0)
    identity = FormalNormalTransactionIdentity(
        "NOT_IN_NORMAL_LIBRARY", ("QUERY_TRANSACTION_NOT_IN_NORMAL_LIBRARY",), False
    )
    return evaluate_phase1_direct_normal_match(
        strict, features, loaded.baseline, identity, loaded.normal_record_transaction_ids
    )


def _exact_direct(loaded):
    record = loaded.baseline.groups[0].records[0]
    query = replace(record, record_id="query:strict_result_chain_001")
    chain = StrictResultChain(
        strict_chain_id="strict_result_chain_001",
        result_id="result-1",
        result_node_id="node-1",
        result_fact_ids=("fact-1",),
        result_type="R_OUTFLOW",
        asset_type="TOKEN",
        asset_address=next(iter(query.asset_behaviors)).asset_identifier,
        from_address=query.main_anchor.effective_address,
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
        1, (), (chain,),
        (ResultChainStatus("result-1", "node-1", (chain.strict_chain_id,), "READY", ()),),
        (), {"coverage_rate": "1.0000", "critical_missing_count": 0},
    )
    status = ChainFeatureExtractionStatus(
        chain.strict_chain_id, query.record_id, "EXTRACTED_CLEAN", True, (), ()
    )
    features = NormalFeatureExtractionResult("query", (query,), (status,), 1, 1, 1, 0, 0)
    identity = FormalNormalTransactionIdentity(
        "NOT_IN_NORMAL_LIBRARY", ("QUERY_TRANSACTION_NOT_IN_NORMAL_LIBRARY",), False
    )
    return evaluate_phase1_direct_normal_match(
        strict, features, loaded.baseline, identity, loaded.normal_record_transaction_ids
    )


def _prepared(alias, loaded, mode="ready"):
    preparation = build_synthetic_phase1_preparation()
    direct = _empty_direct(loaded)
    if mode == "direct":
        direct = _exact_direct(loaded)
    elif mode == "ineligible":
        eligibility = replace(
            preparation.eligibility,
            eligibility="INELIGIBLE",
            llm_allowed=False,
            reason_codes=("QUANTITATIVE_NOT_PASS",),
        )
        preparation = replace(preparation, preparation_status="INELIGIBLE", eligibility=eligibility, prompt=None)
    elif mode == "gap":
        gate = replace(preparation.gate_evidence, query_chains=())
        eligibility = replace(
            preparation.eligibility,
            eligibility="INELIGIBLE",
            llm_allowed=False,
            reason_codes=("CHAIN_SET_MISMATCH",),
            query_chain_aliases=(),
        )
        preparation = replace(
            preparation,
            preparation_status="INELIGIBLE",
            gate_evidence=gate,
            eligibility=eligibility,
            prompt=None,
        )
    return screen.PreparedFaultScreenEvidence(
        sample_alias=alias,
        stage_counts={
            "result_candidate_count": 1,
            "strict_chain_count": 2,
            "strict_ready_count": 1,
            "strict_uncertain_count": 1,
            "clean_record_count": 1,
        },
        coverage_rate="1.0000",
        critical_missing_count=0,
        direct_match=direct,
        preparation=preparation,
    )


class FakeRpc:
    def __init__(self, chain_id="0x1", fail_at=None):
        self.chain_id = chain_id
        self.fail_at = fail_at
        self.calls = []

    def call(self, method, params):
        self.calls.append((method, params))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("RPC_NETWORK_ERROR")
        return self.chain_id if method == "eth_chainId" else {"fake_frame": True}


class Phase1FaultLlmEligibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = screen.build_fault_screen_manifest()
        cls.loaded = load_formal_normal_baseline()

    def test_manifest_selection_and_digest_are_stable(self):
        again = screen.build_fault_screen_manifest()
        self.assertEqual(self.manifest, again)
        self.assertEqual(len(self.manifest.samples), 10)
        self.assertEqual(len({item.tx_hash for item in self.manifest.samples}), 10)
        self.assertEqual(len(self.manifest.manifest_sha256), 64)

    def test_first_group_is_exact_historical_evidence_gap_set(self):
        first = self.manifest.samples[:5]
        self.assertEqual(tuple(item.tx_hash for item in first), screen.HISTORICAL_RECHECK_HASHES)
        self.assertTrue(all(item.source_group == "HISTORICAL_RECHECK" for item in first))

    def test_new_selection_uses_hash_only_and_excludes_audited_and_normal(self):
        audited = screen._audited_fault_hashes(screen.BALANCED_ROOT, screen.PRIOR_FAULT_ROOT)
        normal = screen._normal_pool(screen.DEFAULT_FORMAL_NORMAL_BASELINE_DIR, 470)
        new_hashes = {item.tx_hash for item in self.manifest.samples[5:]}
        self.assertFalse(new_hashes & audited)
        self.assertFalse(new_hashes & normal)
        serialized = screen.serialize_manifest(self.manifest).lower()
        self.assertNotIn("cause", serialized)
        self.assertNotIn("location", serialized)

    def test_rank_rule_matches_exact_seed_formula(self):
        for item in self.manifest.samples[5:]:
            import hashlib
            expected = hashlib.sha256(f"20260822|{item.tx_hash}".encode()).hexdigest()
            self.assertEqual(screen._rank_digest(item.tx_hash), expected)

    def test_default_mode_has_zero_external_calls(self):
        rpc = FakeRpc()
        result = screen.run_fault_screen(self.manifest, live=False, rpc_client=rpc)
        self.assertEqual(rpc.calls, [])
        self.assertEqual(result.rpc_request_count, 0)
        self.assertEqual(result.deepseek_request_count, 0)

    def test_cross_transaction_exact_fault_is_collision_phase2_without_llm(self):
        row = screen.classify_fault_screen_sample(_prepared(self.manifest.samples[0].sample_alias, self.loaded, "direct"))
        self.assertEqual(row["engineering_outcome"], "STRUCTURAL_EXACT_COLLISION_PHASE2")
        self.assertEqual(row["final_status"], "PROCEED_PHASE2")
        self.assertTrue(row["structural_exact_collision"])
        self.assertEqual(row["query_normal_identity_status"], "NOT_IN_NORMAL_LIBRARY")
        self.assertFalse(row["deepseek_called"])

    def test_ready_nonzero_pass_is_candidate_without_deepseek(self):
        row = screen.classify_fault_screen_sample(_prepared(self.manifest.samples[0].sample_alias, self.loaded))
        self.assertEqual(row["engineering_outcome"], "READY_FOR_LLM")
        self.assertEqual(row["final_status"], "AWAITING_LLM_NOT_CALLED")
        self.assertGreater(Decimal(row["chain_evidence"][0]["chain_distance"]), 0)
        self.assertFalse(row["deepseek_called"])

    def test_ineligible_or_missing_threshold_goes_phase2(self):
        row = screen.classify_fault_screen_sample(_prepared(self.manifest.samples[0].sample_alias, self.loaded, "ineligible"))
        self.assertEqual(row["engineering_outcome"], "PROCEED_PHASE2")
        self.assertIn("QUANTITATIVE_NOT_PASS", row["reason_codes"])

    def test_chain_evidence_contract_is_complete(self):
        row = screen.classify_fault_screen_sample(_prepared(self.manifest.samples[0].sample_alias, self.loaded))
        expected = {
            "chain_alias", "strict_chain_status", "feature_extraction_status", "call_distance",
            "asset_distance", "chain_distance", "nearest_distance", "dynamic_threshold",
            "quantitative_status", "normal_group_size", "top3_count",
            "top3_distinct_normal_transaction_count", "exact_match", "strictly_below_threshold",
            "equals_threshold",
        }
        self.assertEqual(set(row["chain_evidence"][0]), expected)
        self.assertEqual(row["chain_evidence"][0]["top3_count"], 3)
        self.assertIsNotNone(row["prompt_sha256"])
        self.assertGreater(row["prompt_byte_count"], 0)

    def test_partial_query_evidence_is_not_counted_as_all_clean_pass(self):
        row = screen.classify_fault_screen_sample(
            _prepared(self.manifest.samples[0].sample_alias, self.loaded, "gap")
        )
        self.assertFalse(row["clean_chain_evidence_complete"])
        self.assertEqual(row["missing_clean_chain_evidence_count"], 1)

    def test_three_candidates_stop_after_three_traces(self):
        rpc = FakeRpc()
        result = screen.run_fault_screen(
            self.manifest,
            live=True,
            rpc_client=rpc,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: _prepared(sample.sample_alias, loaded),
        )
        self.assertEqual(len(rpc.calls), 4)
        self.assertEqual(result.rpc_request_count, 4)
        self.assertEqual(result.deepseek_request_count, 0)
        self.assertEqual(result.run_status, "EARLY_STOP_READY_CANDIDATES_FOUND")
        self.assertEqual(sum(row["engineering_outcome"] == "READY_FOR_LLM" for row in result.rows), 3)
        self.assertTrue(all(
            row["engineering_outcome"] == "NOT_PROCESSED_EARLY_STOP" for row in result.rows[3:]
        ))

    def test_structural_collision_does_not_count_toward_candidate_stop(self):
        rpc = FakeRpc()
        def prepare(sample, raw, loaded):
            mode = "direct" if sample.sample_alias.endswith("001") else "ready"
            return _prepared(sample.sample_alias, loaded, mode)
        result = screen.run_fault_screen(
            self.manifest, live=True, rpc_client=rpc, loaded_baseline=self.loaded, sample_preparer=prepare
        )
        self.assertEqual(
            sum(row["engineering_outcome"] == "STRUCTURAL_EXACT_COLLISION_PHASE2" for row in result.rows),
            1,
        )
        self.assertEqual(sum(row["engineering_outcome"] == "READY_FOR_LLM" for row in result.rows), 3)
        self.assertEqual(len(rpc.calls), 5)

    def test_no_clean_chain_goes_phase2_and_does_not_stop(self):
        rpc = FakeRpc()
        result = screen.run_fault_screen(
            self.manifest,
            live=True,
            rpc_client=rpc,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: _prepared(sample.sample_alias, loaded, "ineligible"),
        )
        self.assertEqual(len(rpc.calls), 11)
        self.assertTrue(all(row["engineering_outcome"] == "PROCEED_PHASE2" for row in result.rows))
        self.assertEqual(result.deepseek_request_count, 0)

    def test_network_error_stops_without_retry_and_keeps_checkpoint(self):
        rpc = FakeRpc(fail_at=3)
        checkpoints = []
        result = screen.run_fault_screen(
            self.manifest,
            live=True,
            rpc_client=rpc,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: _prepared(sample.sample_alias, loaded, "ineligible"),
            progress_callback=lambda value: checkpoints.append(value),
        )
        self.assertEqual(len(rpc.calls), 3)
        self.assertEqual(result.run_status, "STOPPED")
        self.assertGreaterEqual(len(checkpoints), 2)
        self.assertEqual(result.deepseek_request_count, 0)

    def test_chain_id_mismatch_uses_one_rpc_only(self):
        rpc = FakeRpc(chain_id="0x38")
        result = screen.run_fault_screen(
            self.manifest, live=True, rpc_client=rpc, loaded_baseline=self.loaded
        )
        self.assertEqual(len(rpc.calls), 1)
        self.assertEqual(result.run_status, "STOPPED")
        self.assertEqual(result.deepseek_request_count, 0)

    def test_report_counts_and_no_truth_in_method_output(self):
        rpc = FakeRpc()
        result = screen.run_fault_screen(
            self.manifest,
            live=True,
            rpc_client=rpc,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: _prepared(sample.sample_alias, loaded),
        )
        report = screen.build_report(result, self.loaded.summary.to_dict())
        self.assertEqual(report["summary"]["ready_for_llm_count"], 3)
        self.assertEqual(report["external_call_counts"]["deepseek"], 0)
        text = screen.serialize_report(report).lower()
        self.assertNotIn("ground_truth", text)
        self.assertNotIn('"label"', text)

    def test_report_sanitization_rejects_sensitive_fields(self):
        for payload in (
            {"input": "0x1234"},
            {"safe": "0x" + "a" * 40},
            {"safe": "reasoning_content"},
        ):
            with self.assertRaises(ValueError):
                screen.validate_sanitized_document(payload)

    def test_atomic_artifact_writes(self):
        result = screen.run_fault_screen(self.manifest, live=False)
        report = screen.build_report(result, self.loaded.summary.to_dict())
        checkpoint = screen.build_checkpoint(self.manifest, result.rows, 0, "LIVE_MODE_REQUIRED")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                screen.write_manifest(root, self.manifest),
                screen.write_checkpoint(root, checkpoint),
                screen.write_report(root, report),
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_formal_baseline_remains_unchanged(self):
        before = self.loaded.baseline.to_dict()
        screen.classify_fault_screen_sample(_prepared(self.manifest.samples[0].sample_alias, self.loaded))
        self.assertEqual(self.loaded.baseline.to_dict(), before)

    def test_main_without_live_never_loads_rpc_config(self):
        with patch.object(screen, "_load_live_config") as loader:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = screen.main([])
        self.assertEqual(code, 2)
        loader.assert_not_called()
        self.assertIn("deepseek_request_count=0", output.getvalue())


if __name__ == "__main__":
    unittest.main()

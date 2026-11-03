from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import scripts.run_phase1_real_llm3_loo_pilot as pilot
from scripts.dry_run_phase1_normality import build_synthetic_phase1_preparation
from src.cloc_v2.formal_normal_baseline_loader import (
    FormalNormalTransactionIdentity,
    load_formal_normal_baseline,
)
from src.cloc_v2.normal_features import ChainFeatureExtractionStatus, NormalFeatureExtractionResult
from src.cloc_v2.phase1_direct_normal import evaluate_phase1_direct_normal_match
from src.cloc_v2.phase1_normality_deepseek import (
    Phase1DeepSeekConfig,
    execute_phase1_deepseek_review,
)
from src.cloc_v2.phase1_normality_prompt import RESPONSE_SCHEMA_VERSION
from src.cloc_v2.strict_result_chains import ResultChainStatus, StrictResultChain, StrictResultChainsResult


SECRET_KEY = "fake-key-never-persist"
SECRET_REASONING = "fake-reasoning-never-persist"


def _strict_and_features(record):
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
        coverage={"coverage_rate": "1.0000", "critical_missing_count": 0},
    )
    status = ChainFeatureExtractionStatus(
        strict_chain_id=chain.strict_chain_id,
        record_id=record.record_id,
        status="EXTRACTED_CLEAN",
        record_created=True,
        reasons=(),
        issues=(),
    )
    features = NormalFeatureExtractionResult(
        transaction_record_id="query",
        records=(record,),
        chain_statuses=(status,),
        input_chain_count=1,
        record_count=1,
        clean_record_count=1,
        uncertain_record_count=0,
        rejected_chain_count=0,
    )
    return strict, features


def _direct_match(loaded):
    normal = loaded.baseline.groups[0].records[0]
    query = replace(normal, record_id="query:strict_result_chain_001")
    strict, features = _strict_and_features(query)
    identity = FormalNormalTransactionIdentity(
        "KNOWN_NORMAL_ORIGIN", ("QUERY_NORMAL_ORIGIN_RESOLVED",), True,
        _normal_origin_id=loaded.normal_record_transaction_ids[normal.record_id],
        _normalized_transaction_hash="0x" + "1" * 64,
    )
    return evaluate_phase1_direct_normal_match(
        strict, features, loaded.baseline, identity, loaded.normal_record_transaction_ids
    )


def _prepared(alias, loaded, *, eligible=True):
    preparation = build_synthetic_phase1_preparation()
    if not eligible:
        preparation = replace(
            preparation,
            preparation_status="INELIGIBLE",
            prompt=None,
            eligibility=replace(
                preparation.eligibility,
                eligibility="INELIGIBLE",
                llm_allowed=False,
                reason_codes=("QUANTITATIVE_NOT_PASS",),
            ),
        )
    return pilot.PreparedLooEvidence(
        sample_alias=alias,
        stage_counts={
            "result_candidate_count": 1,
            "strict_chain_count": 2,
            "strict_ready_count": 1,
            "strict_uncertain_count": 1,
            "clean_record_count": 1,
        },
        full_direct_match=_direct_match(loaded),
        full_self_exact_match=True,
        excluded_normal_record_count=1,
        loo_baseline_record_count=loaded.baseline.record_count - 1,
        loo_preparation=preparation,
    )


def _response(preparation, decision):
    scope = preparation.prompt.citation_whitelist.chain_scopes[0]
    assessment = {
        "NORMAL": "CONSISTENT",
        "SUSPICIOUS": "MATERIAL_DEVIATION",
        "UNCERTAIN": "INSUFFICIENT_EVIDENCE",
    }[decision]
    reason = {
        "NORMAL": "CONSISTENT_CALL_SEQUENCE",
        "SUSPICIOUS": "MATERIAL_CALL_DEVIATION",
        "UNCERTAIN": "REFERENCE_EVIDENCE_INSUFFICIENT",
    }[decision]
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "chain_decisions": [{
            "chain_alias": scope.chain_alias,
            "decision": decision,
            "reference_assessment": assessment,
            "supporting_reference_aliases": [scope.reference_aliases[0]],
            "evidence_ids": [scope.query_call_evidence_ids[0], scope.reference_call_evidence_ids[0]],
            "reason_codes": [reason],
            "brief_rationale": "Evidence-only synthetic test decision.",
        }],
    }


class FakeTransport:
    def __init__(self, decision="NORMAL", mode="valid"):
        self.decision = decision
        self.mode = mode
        self.preparation = None
        self.calls = 0

    def send(self, payload, api_key, timeout):
        self.calls += 1
        if self.mode == "transport":
            raise RuntimeError("hidden-provider-error")
        content = json.dumps(_response(self.preparation, self.decision))
        if self.mode == "invalid":
            content = "not-json"
        if self.mode == "provider":
            return {"error": {"message": "hidden"}, "model": "deepseek-v4-pro-test"}
        return {
            "model": "deepseek-v4-pro-test",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": SECRET_REASONING,
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


def _runner(decision="NORMAL", mode="valid"):
    transport = FakeTransport(decision, mode)

    def run(preparation):
        transport.preparation = preparation
        return execute_phase1_deepseek_review(preparation, Phase1DeepSeekConfig(), transport, SECRET_KEY)

    run.transport = transport
    return run


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


class Phase1RealLlm3LooPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = pilot.build_locked_loo_manifest()
        cls.loaded = load_formal_normal_baseline()

    def test_locked_manifest_exact_order_and_digest_stable(self):
        again = pilot.build_locked_loo_manifest()
        self.assertEqual(self.manifest, again)
        self.assertEqual(
            [(item.source_sample_id, item.tx_hash) for item in self.manifest.samples],
            list(pilot.LOCKED_SAMPLE_ROWS),
        )
        self.assertEqual(len(self.manifest.manifest_sha256), 64)

    def test_default_mode_is_fully_offline(self):
        rpc = FakeRpc()
        runner = _runner()
        artifacts = pilot.run_phase1_real_llm3_loo_pilot(
            self.manifest, live=False, rpc_client=rpc, deepseek_runner=runner
        )
        self.assertEqual(rpc.calls, [])
        self.assertEqual(runner.transport.calls, 0)
        self.assertEqual(artifacts.report["external_call_counts"], {"rpc": 0, "deepseek": 0, "network": 0})

    def test_loo_excludes_all_transaction_records_and_preserves_formal_baseline(self):
        before = self.loaded.baseline.to_dict()
        source = "eth_normal_000106"
        expected = sum(owner == source for owner in self.loaded.normal_record_transaction_ids.values())
        loo, mapping, excluded = pilot.build_leave_one_transaction_out_baseline(self.loaded, source)
        self.assertEqual(excluded, expected)
        self.assertGreater(excluded, 1)
        self.assertNotIn(source, mapping.values())
        self.assertEqual(loo.record_count, self.loaded.baseline.record_count - excluded)
        self.assertEqual(loo.min_group_size, 3)
        self.assertEqual(self.loaded.baseline.to_dict(), before)

    def test_loo_recalibrates_through_formal_builder(self):
        with patch.object(pilot, "build_normal_baseline", wraps=pilot.build_normal_baseline) as builder:
            pilot.build_leave_one_transaction_out_baseline(self.loaded, "eth_normal_000384")
        builder.assert_called_once()
        self.assertEqual(builder.call_args.kwargs, {"min_group_size": 3})

    def test_filtered_top3_excludes_current_transaction(self):
        source = "eth_normal_000384"
        own = [
            record
            for group in self.loaded.baseline.groups
            for record in group.records
            if self.loaded.normal_record_transaction_ids[record.record_id] == source
        ]
        self.assertEqual(len(own), 1)
        strict, features = _strict_and_features(own[0])
        loo, mapping, _ = pilot.build_leave_one_transaction_out_baseline(self.loaded, source)
        from src.cloc_v2.phase1_normality_bridge import prepare_phase1_normality_review
        preparation = prepare_phase1_normality_review(strict, features, loo, mapping)
        self.assertEqual(preparation.preparation_status, "READY_FOR_LLM")
        self.assertTrue(all(mapping[match_id] != source for match_id in mapping))
        self.assertTrue(all(query.nearest_distance > 0 for query in preparation.gate_evidence.query_chains))
        self.assertTrue(all(len(query.normal_references) == 3 for query in preparation.gate_evidence.query_chains))

    def test_loo_view_explicitly_disables_identity_direct_normal(self):
        row, _, _ = pilot.route_loo_sample(
            _prepared(self.manifest.samples[0].sample_alias, self.loaded),
            _runner("NORMAL"),
        )
        self.assertTrue(row["full_baseline_view"]["identity_direct_normal_enabled"])
        self.assertFalse(row["loo_view"]["identity_direct_normal_enabled"])

    def test_ineligible_sample_never_calls_deepseek(self):
        prepared = _prepared(self.manifest.samples[0].sample_alias, self.loaded, eligible=False)
        runner = _runner()
        row, used, stop = pilot.route_loo_sample(prepared, runner)
        self.assertEqual(used, 0)
        self.assertFalse(stop)
        self.assertFalse(row["llm_called"])
        self.assertEqual(runner.transport.calls, 0)

    def test_eligible_sample_calls_deepseek_once_and_normal_routes_early(self):
        prepared = _prepared(self.manifest.samples[0].sample_alias, self.loaded)
        runner = _runner("NORMAL")
        row, used, stop = pilot.route_loo_sample(prepared, runner)
        self.assertEqual((used, runner.transport.calls), (1, 1))
        self.assertFalse(stop)
        self.assertEqual(row["final_route"], "EARLY_NORMAL")
        self.assertEqual(row["llm_chain_decisions"][0]["decision"], "NORMAL")

    def test_suspicious_and_uncertain_route_phase2(self):
        prepared = _prepared(self.manifest.samples[0].sample_alias, self.loaded)
        for decision in ("SUSPICIOUS", "UNCERTAIN"):
            row, used, stop = pilot.route_loo_sample(prepared, _runner(decision))
            self.assertEqual(used, 1)
            self.assertFalse(stop)
            self.assertEqual(row["final_route"], "PROCEED_PHASE2")
            self.assertEqual(row["llm_chain_decisions"][0]["decision"], decision)

    def test_invalid_response_is_safe_but_does_not_stop_following_sample(self):
        row, used, stop = pilot.route_loo_sample(
            _prepared(self.manifest.samples[0].sample_alias, self.loaded), _runner(mode="invalid")
        )
        self.assertEqual(used, 1)
        self.assertFalse(stop)
        self.assertEqual(row["final_route"], "PROCEED_PHASE2")
        self.assertEqual(row["deepseek_response_status"], "INVALID_RESPONSE")

    def test_transport_and_provider_failures_stop_external_sequence(self):
        prepared = _prepared(self.manifest.samples[0].sample_alias, self.loaded)
        for mode in ("transport", "provider"):
            row, used, stop = pilot.route_loo_sample(prepared, _runner(mode=mode))
            self.assertEqual(used, 1)
            self.assertTrue(stop)
            self.assertEqual(row["final_route"], "PROCEED_PHASE2")

    def test_chain_id_mismatch_uses_one_rpc_and_zero_deepseek(self):
        rpc = FakeRpc(chain_id="0x38")
        runner = _runner()
        result = pilot.run_phase1_real_llm3_loo_pilot(
            self.manifest,
            live=True,
            rpc_client=rpc,
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
        )
        self.assertEqual(len(rpc.calls), 1)
        self.assertEqual(runner.transport.calls, 0)
        self.assertEqual(result.report["run_status"], "STOPPED")

    def test_rpc_network_error_stops_and_keeps_progress(self):
        rpc = FakeRpc(fail_at=3)
        runner = _runner()
        progress = []
        result = pilot.run_phase1_real_llm3_loo_pilot(
            self.manifest,
            live=True,
            rpc_client=rpc,
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: _prepared(sample.sample_alias, loaded),
            progress_callback=lambda value: progress.append(value),
        )
        self.assertEqual(len(rpc.calls), 3)
        self.assertEqual(result.report["run_status"], "STOPPED")
        self.assertGreaterEqual(len(progress), 2)
        self.assertEqual(result.checkpoint["sample_results"][0]["final_route"], "EARLY_NORMAL")

    def test_three_eligible_samples_enforce_four_rpc_and_three_deepseek_caps(self):
        rpc = FakeRpc()
        runner = _runner()
        result = pilot.run_phase1_real_llm3_loo_pilot(
            self.manifest,
            live=True,
            rpc_client=rpc,
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: _prepared(sample.sample_alias, loaded),
        )
        self.assertEqual(len(rpc.calls), 4)
        self.assertEqual(runner.transport.calls, 3)
        self.assertEqual(result.report["external_call_counts"]["rpc"], 4)
        self.assertEqual(result.report["external_call_counts"]["deepseek"], 3)

    def test_report_is_sanitized_and_labels_do_not_enter_inference(self):
        rpc = FakeRpc()
        runner = _runner()
        result = pilot.run_phase1_real_llm3_loo_pilot(
            self.manifest,
            live=True,
            rpc_client=rpc,
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, raw, loaded: _prepared(sample.sample_alias, loaded),
        )
        text = pilot.serialize_report(result.report)
        for sample in self.manifest.samples:
            self.assertNotIn(sample.tx_hash, text)
        self.assertNotIn(SECRET_KEY, text)
        self.assertNotIn(SECRET_REASONING, text)
        self.assertNotIn("reasoning_content", text.lower())
        self.assertNotIn('"label"', text.lower())
        self.assertFalse(result.report["safety"]["label_used_during_inference"])

    def test_atomic_manifest_checkpoint_and_report_writes(self):
        artifacts = pilot.run_phase1_real_llm3_loo_pilot(self.manifest, live=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = pilot.write_locked_manifest(root, artifacts.manifest)
            checkpoint_path = pilot.write_checkpoint(root, artifacts.checkpoint)
            report_path = pilot.write_report(root, artifacts.report)
            self.assertEqual(json.loads(manifest_path.read_text("utf-8"))["manifest_sha256"], self.manifest.manifest_sha256)
            self.assertEqual(json.loads(checkpoint_path.read_text("utf-8"))["schema_version"], pilot.CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(json.loads(report_path.read_text("utf-8"))["experiment_kind"], pilot.EXPERIMENT_KIND)
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_main_without_live_does_not_load_config_or_network(self):
        with patch.object(pilot, "_load_live_chain_config") as loader:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = pilot.main([])
        self.assertEqual(code, 2)
        loader.assert_not_called()
        self.assertIn("rpc_request_count=0", output.getvalue())


if __name__ == "__main__":
    unittest.main()

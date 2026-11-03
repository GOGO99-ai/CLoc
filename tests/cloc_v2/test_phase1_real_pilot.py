from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.run_phase1_real_pilot as pilot
from scripts.dry_run_phase1_normality import build_synthetic_phase1_preparation
from src.cloc_v2.formal_normal_baseline_loader import (
    FormalNormalBaselineSummary,
    LoadedFormalNormalBaseline,
    NormalRecordTransactionOrigin,
)
from src.cloc_v2.normal_baseline import build_normal_baseline
from src.cloc_v2.normal_distance import (
    AssetBehavior,
    CallStep,
    ChainFeatureRecord,
    MainAnchorKey,
)
from src.cloc_v2.phase1_normality_bridge import Phase1NormalityPreparation
from src.cloc_v2.phase1_normality_deepseek import (
    Phase1DeepSeekConfig,
    execute_phase1_deepseek_review,
)
from src.cloc_v2.phase1_normality_orchestration import evaluate_phase1_eligibility
from src.cloc_v2.phase1_normality_prompt import RESPONSE_SCHEMA_VERSION


SECRET_KEY = "fake-key-must-never-persist"
SECRET_REASONING = "private-reasoning-must-never-persist"


def address(character: str) -> str:
    return "0x" + character * 40


def fake_loaded_baseline() -> LoadedFormalNormalBaseline:
    anchor = MainAnchorKey(1, address("a"), "0x11111111")
    records = tuple(
        ChainFeatureRecord(
            f"normal-record-{index}",
            anchor,
            (CallStep(anchor.effective_address, anchor.selector),),
            frozenset({AssetBehavior(address("b"), "OUTFLOW")}),
            (),
        )
        for index in range(1, 4)
    )
    baseline = build_normal_baseline(records, min_group_size=3)
    mapping = {record.record_id: f"normal-transaction-{index}" for index, record in enumerate(records, 1)}
    origins = tuple(NormalRecordTransactionOrigin(record.record_id, mapping[record.record_id]) for record in records)
    summary = FormalNormalBaselineSummary(
        schema_version="formal_normal_baseline_summary_v1",
        formal_min_group_size=3,
        successful_ethereum_transaction_count=470,
        clean_record_count=3,
        group_count=1,
        calibrated_group_count=1,
        uncalibrated_group_count=0,
        record_origin_mapping_count=3,
        manifest_consistency_passed=True,
        reconstructed_baseline_match=True,
        ethereum_only=True,
        source_files_read_count=6,
    )
    return LoadedFormalNormalBaseline(baseline, mapping, origins, summary)


def prepared(sample_alias: str, *, eligible: bool = True) -> pilot.PreparedPilotEvidence:
    preparation = build_synthetic_phase1_preparation()
    if not eligible:
        gate = replace(preparation.gate_evidence, result_candidate_coverage_rate=Decimal("0"))
        eligibility = evaluate_phase1_eligibility(gate, preparation.prompt)
        preparation = Phase1NormalityPreparation("INELIGIBLE", gate, eligibility, None, ())
    return pilot.PreparedPilotEvidence(
        sample_alias,
        {
            "call_node_count": 2,
            "result_candidate_count": 1,
            "strict_chain_count": 1,
            "strict_ready_count": 1,
            "strict_uncertain_count": 0,
            "clean_record_count": 1,
            "uncertain_record_count": 0,
            "rejected_record_count": 0,
        },
        (),
        preparation,
    )


def decision_response(preparation: Phase1NormalityPreparation) -> dict:
    scope = preparation.prompt.citation_whitelist.chain_scopes[0]
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "chain_decisions": [
            {
                "chain_alias": scope.chain_alias,
                "decision": "NORMAL",
                "reference_assessment": "CONSISTENT",
                "supporting_reference_aliases": [scope.reference_aliases[0]],
                "evidence_ids": [scope.query_call_evidence_ids[0], scope.reference_call_evidence_ids[0]],
                "reason_codes": ["CONSISTENT_CALL_SEQUENCE", "WITHIN_CALIBRATED_THRESHOLD"],
                "brief_rationale": "Q1 matches R1 using E1 and E3.",
            }
        ],
    }


class FakeTransport:
    def __init__(self, *, invalid: bool = False) -> None:
        self.invalid = invalid
        self.calls = 0
        self.preparation = None

    def send(self, request_payload, api_key, timeout_seconds):
        self.calls += 1
        content = {"schema_version": RESPONSE_SCHEMA_VERSION, "unexpected": True}
        if not self.invalid:
            content = decision_response(self.preparation)
        return {
            "model": "deepseek-v4-pro-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content),
                        "reasoning_content": SECRET_REASONING,
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


class FakeRpc:
    def __init__(self, *, chain_id: object = "0x1", fail_trace: bool = False) -> None:
        self.chain_id = chain_id
        self.fail_trace = fail_trace
        self.calls: list[tuple[str, list]] = []

    def call(self, method: str, params: list):
        self.calls.append((method, params))
        if method == "eth_chainId":
            return self.chain_id
        if self.fail_trace:
            raise RuntimeError("RPC_NETWORK_ERROR")
        return {"synthetic_frame": True}


def deepseek_runner(*, invalid: bool = False):
    transport = FakeTransport(invalid=invalid)

    def run(preparation):
        transport.preparation = preparation
        return execute_phase1_deepseek_review(
            preparation,
            Phase1DeepSeekConfig(),
            transport,
            SECRET_KEY,
        )

    run.transport = transport
    return run


class Phase1RealPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loaded = fake_loaded_baseline()

    def test_locked_manifest_is_exact_and_stable(self) -> None:
        first = pilot.build_locked_manifest()
        second = pilot.build_locked_manifest()
        self.assertEqual([item.sample_alias for item in first.samples], ["pilot_eth_001", "pilot_eth_002", "pilot_eth_003"])
        self.assertEqual(
            [item.tx_hash for item in first.samples],
            [
                "0x0163856aaa678676b22e072186c1bdc55ed44a5a0e2cf44500362800663ddce9",
                "0x047d2f5ca117de92a65de25d04e00db9bc0f1a899ba61140550afc6618d282ac",
                "0x07e5568b71f1f4d9234d5925473c598865c5ee1800348c1c4b044c6f0223f308",
            ],
        )
        self.assertTrue(all(item.blockchain_chain_id == 1 for item in first.samples))
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(first.to_dict()["selection_rule"], pilot.SELECTION_RULE)

    def test_source_contains_no_dataset_read_path(self) -> None:
        source = Path(pilot.__file__).read_text(encoding="utf-8")
        self.assertNotIn("/dataset", source.lower())
        self.assertNotIn("\\dataset", source.lower())
        self.assertNotIn("rglob(", source)

    def test_default_mode_makes_zero_external_requests(self) -> None:
        rpc = FakeRpc()
        runner = MagicMock()
        result = pilot.run_phase1_real_pilot(live=False, rpc_client=rpc, deepseek_runner=runner)
        self.assertEqual(rpc.calls, [])
        runner.assert_not_called()
        self.assertEqual(result.report["external_call_counts"], {"rpc": 0, "deepseek": 0, "network": 0})

    def test_cli_without_live_does_not_load_config_or_network(self) -> None:
        with patch.object(pilot, "_load_live_chain_config") as load_config, contextlib.redirect_stdout(io.StringIO()):
            code = pilot.main([])
        self.assertEqual(code, 2)
        load_config.assert_not_called()

    def test_chain_id_mismatch_stops_before_trace_or_deepseek(self) -> None:
        rpc = FakeRpc(chain_id="0x38")
        runner = MagicMock()
        result = pilot.run_phase1_real_pilot(
            live=True, rpc_client=rpc, deepseek_runner=runner, loaded_baseline=self.loaded
        )
        self.assertEqual(len(rpc.calls), 1)
        runner.assert_not_called()
        self.assertTrue(all(item["reason_codes"] == ["CHAIN_ID_MISMATCH"] for item in result.report["sample_results"]))

    def test_rpc_request_hard_limit_is_four(self) -> None:
        rpc = FakeRpc()
        result = pilot.run_phase1_real_pilot(
            live=True,
            rpc_client=rpc,
            deepseek_runner=deepseek_runner(),
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, frame, loaded: prepared(sample.sample_alias, eligible=False),
        )
        self.assertEqual(len(rpc.calls), 4)
        self.assertEqual(result.report["request_limits"]["rpc"], 4)

    def test_ineligible_samples_never_call_deepseek(self) -> None:
        runner = MagicMock()
        result = pilot.run_phase1_real_pilot(
            live=True,
            rpc_client=FakeRpc(),
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, frame, loaded: prepared(sample.sample_alias, eligible=False),
        )
        runner.assert_not_called()
        self.assertTrue(all(not item["llm_called"] for item in result.report["sample_results"]))

    def test_one_eligible_sample_calls_deepseek_once(self) -> None:
        runner = deepseek_runner()
        item, used = pilot.route_prepared_pilot_sample(prepared("pilot_eth_001"), runner)
        self.assertEqual(used, 1)
        self.assertEqual(runner.transport.calls, 1)
        self.assertEqual(item["final_route"], "EARLY_NORMAL")

    def test_three_eligible_samples_call_deepseek_at_most_three_times(self) -> None:
        runner = deepseek_runner()
        result = pilot.run_phase1_real_pilot(
            live=True,
            rpc_client=FakeRpc(),
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, frame, loaded: prepared(sample.sample_alias),
        )
        self.assertEqual(runner.transport.calls, 3)
        self.assertEqual(result.report["external_call_counts"]["deepseek"], 3)
        self.assertTrue(all(item["final_route"] == "EARLY_NORMAL" for item in result.report["sample_results"]))

    def test_rpc_error_is_not_retried_and_safely_routes_phase2(self) -> None:
        rpc = FakeRpc(fail_trace=True)
        result = pilot.run_phase1_real_pilot(
            live=True,
            rpc_client=rpc,
            deepseek_runner=deepseek_runner(),
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, frame, loaded: prepared(sample.sample_alias),
        )
        self.assertEqual(len(rpc.calls), 4)
        self.assertTrue(all(item["final_route"] == "PROCEED_PHASE2" for item in result.report["sample_results"]))
        self.assertTrue(all(item["reason_codes"] == ["RPC_NETWORK_ERROR"] for item in result.report["sample_results"]))

    def test_deepseek_exception_is_not_retried_and_safely_routes_phase2(self) -> None:
        runner = MagicMock(side_effect=RuntimeError("private-provider-text"))
        item, used = pilot.route_prepared_pilot_sample(prepared("pilot_eth_001"), runner)
        self.assertEqual(used, 1)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(item["final_route"], "PROCEED_PHASE2")
        self.assertEqual(item["reason_codes"], ["DEEPSEEK_CLIENT_ERROR"])

    def test_invalid_llm_response_safely_routes_phase2(self) -> None:
        runner = deepseek_runner(invalid=True)
        item, used = pilot.route_prepared_pilot_sample(prepared("pilot_eth_001"), runner)
        self.assertEqual(used, 1)
        self.assertEqual(item["deepseek_response_status"], "INVALID_RESPONSE")
        self.assertEqual(item["final_route"], "PROCEED_PHASE2")

    def test_report_excludes_hashes_addresses_trace_key_and_reasoning(self) -> None:
        result = pilot.run_phase1_real_pilot(
            live=True,
            rpc_client=FakeRpc(),
            deepseek_runner=deepseek_runner(),
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, frame, loaded: prepared(sample.sample_alias),
        )
        text = pilot.serialize_phase1_real_pilot_report(result.report)
        for _, tx_hash in pilot._LOCKED_ROWS:
            self.assertNotIn(tx_hash, text)
        self.assertNotIn(address("a"), text)
        self.assertNotIn(SECRET_KEY, text)
        self.assertNotIn(SECRET_REASONING, text)
        for token in ("calldata", "revertReason", "raw_trace", "system_message", "user_message"):
            self.assertNotIn(token, text)

    def test_manifest_and_report_are_atomically_written(self) -> None:
        artifacts = pilot.run_phase1_real_pilot(live=False)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, report_path = pilot.write_phase1_real_pilot_artifacts(directory, artifacts)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), artifacts.manifest.to_dict())
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), dict(artifacts.report))
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_formal_baseline_safe_summary_is_checked(self) -> None:
        summary = pilot.validate_formal_baseline(self.loaded)
        self.assertEqual(summary["formal_min_group_size"], 3)
        self.assertTrue(summary["manifest_consistency_passed"])
        self.assertTrue(summary["reconstructed_baseline_match"])
        self.assertTrue(summary["ethereum_only"])

    def test_fake_rpc_and_fake_deepseek_end_to_end(self) -> None:
        rpc = FakeRpc()
        runner = deepseek_runner()
        result = pilot.run_phase1_real_pilot(
            live=True,
            rpc_client=rpc,
            deepseek_runner=runner,
            loaded_baseline=self.loaded,
            sample_preparer=lambda sample, frame, loaded: prepared(sample.sample_alias),
        )
        self.assertEqual(result.report["run_status"], "COMPLETED")
        self.assertEqual(result.report["external_call_counts"], {"rpc": 4, "deepseek": 3, "network": 7})
        self.assertEqual([item["sample_alias"] for item in result.report["sample_results"]], ["pilot_eth_001", "pilot_eth_002", "pilot_eth_003"])

    def test_stable_serialization(self) -> None:
        first = pilot.run_phase1_real_pilot(live=False)
        second = pilot.run_phase1_real_pilot(live=False)
        self.assertEqual(pilot.serialize_locked_manifest(first.manifest), pilot.serialize_locked_manifest(second.manifest))
        self.assertEqual(
            pilot.serialize_phase1_real_pilot_report(first.report),
            pilot.serialize_phase1_real_pilot_report(second.report),
        )

    def test_actual_pipeline_uses_all_formal_stages_in_order(self) -> None:
        sample = pilot.build_locked_manifest().samples[0]
        calls: list[str] = []
        fake_graph = MagicMock(nodes=())
        fake_annotations = MagicMock(node_annotations=())
        fake_ledger = MagicMock(transfer_facts=())
        fake_continuity = MagicMock()
        fake_continuity.to_dict.return_value = {}
        fake_detected = MagicMock(result_candidates=())
        fake_detected.to_dict.return_value = {}
        fake_strict = MagicMock(chains=())
        fake_extracted = MagicMock(
            clean_record_count=0,
            uncertain_record_count=0,
            rejected_chain_count=0,
            chain_statuses=(),
        )
        fake_preparation = prepared(sample.sample_alias, eligible=False).preparation

        def step(name, value):
            def invoke(*args, **kwargs):
                calls.append(name)
                return value
            return invoke

        patches = (
            patch.object(pilot, "adapt_calltracer_call_tree", step("adapter", {})),
            patch.object(pilot, "build_transaction_graph", step("graph", fake_graph)),
            patch.object(pilot, "build_runtime_annotations", step("runtime", fake_annotations)),
            patch.object(pilot, "build_observed_asset_ledger", step("ledger", fake_ledger)),
            patch.object(pilot, "build_asset_continuity", step("continuity", fake_continuity)),
            patch.object(pilot, "detect_result_candidates", step("results", fake_detected)),
            patch.object(pilot, "build_strict_result_chains", step("strict", fake_strict)),
            patch.object(pilot, "extract_normal_features", step("features", fake_extracted)),
            patch.object(pilot, "prepare_phase1_normality_review", step("bridge", fake_preparation)),
        )
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            pilot.prepare_real_pilot_sample(sample, {"frame": True}, self.loaded)
        self.assertEqual(calls, ["adapter", "graph", "runtime", "ledger", "continuity", "results", "strict", "features", "bridge"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekResult

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_phase2_topk_v3_single_pilot.py"
_SPEC = importlib.util.spec_from_file_location("run_phase2_topk_v3_single_pilot", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC); sys.modules[_SPEC.name] = pilot; _SPEC.loader.exec_module(pilot)


class Rpc:
    def __init__(self, values): self.values = list(values); self.calls = []
    def call(self, method, params):
        self.calls.append(method); value = self.values.pop(0)
        if isinstance(value, Exception): raise value
        return value


def _bundle():
    graph = {"schema_version": "transaction_graph_v1", "nodes": [{"node_id": "n1", "callee_address": "0x" + "1" * 40}, {"node_id": "n2", "callee_address": "0x" + "2" * 40}]}
    strict = {"schema_version": "strict_result_chains_v1", "edges": [{"edge_id": "e1", "edge_type": "CALL", "src_node_id": "n1", "dst_node_id": "n2", "uncertainties": []}], "chains": [{"strict_chain_id": "c1", "result_type": "R_OUTFLOW", "node_ids": ["n1", "n2"], "edge_ids": ["e1"], "anchor": {"node_id": "n1"}, "uncertainties": []}], "result_statuses": [], "critical_uncertainties": [], "coverage": {}}
    return pilot.Phase2EvidenceBundle(graph, strict, {"schema_version": "candidate_causal_graph_v1", "edges": []}, {"result_candidate_count": 1, "strict_chain_count": 1}, ("n1",), ("0x" + "1" * 40,))


def _source(bundle, mode="EXACT_FUNCTION"):
    slices = {"schema_version": "source_request_slices_v1", "metadata": {"resolution_mode": mode}, "requests": [{"node_id": "n1", "function_match_status": "RESOLVED", "file_path": "C.sol", "function_name": "f", "signature": "f()", "start_line": 1, "end_line": 1, "source": "1: function f(){}", "uncertainties": []}]}
    return slices, {"source_request_count": 0, "attempted_address_count": 1, "verified_source_count": 1, "cache_hit_count": 1, "resolved_slice_count": 1, "window_count": 0, "source_resolution_mode": mode, "stop_code": None}


def _v3_result(preparation, findings=None):
    findings = findings if findings is not None else [{"rank": 1, "source_slice_id": "S1", "evidence_ids": ["E1"], "role": "PRIMARY_CANDIDATE", "confidence": None, "explanation": "S1由E1(CALL)支持", "explanation_source": "LOCAL_DETERMINISTIC_EVIDENCE_SUMMARY"}]
    response = {"schema_version": "phase2_localization_materialized_topk_v3", "status": "RANKED_CANDIDATES" if findings else "INSUFFICIENT_EVIDENCE", "findings": findings, "summary": "local", "provider_free_text_used": False, "explanation_source": "LOCAL_DETERMINISTIC_EVIDENCE_SUMMARY"}
    return Phase2DeepSeekResult("VALIDATED", preparation.prompt_sha256, "b" * 64, "deepseek-v4-pro", None, ("RESPONSE_VALIDATED_TOPK_V3",), json.dumps(response))


class Phase2TopKV3SinglePilotTests(unittest.TestCase):
    def test_locked_manifest_is_fixed_stable_and_not_historical(self):
        first, second = pilot.build_locked_manifest(), pilot.build_locked_manifest()
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(first.tx_hash, pilot.TX_HASH)
        self.assertNotIn(first.tx_hash, {"0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9", "0x38ecc8363836e43aca99d994bb9325a8f3770d7a1d1fffd62c82cf5230525836"})

    def test_default_mode_is_offline(self):
        report = pilot.run_topk_v3_single_pilot(pilot.build_locked_manifest(), live=False)
        self.assertEqual(report["external_call_counts"], {"rpc": 0, "source": 0, "deepseek": 0})

    def test_chain_mismatch_stops_before_trace_source_and_llm(self):
        rpc = Rpc(["0x38"]); report = pilot.run_topk_v3_single_pilot(pilot.build_locked_manifest(), live=True, rpc_client=rpc, source_builder=lambda bundle: _source(bundle), deepseek_runner=lambda prep: _v3_result(prep))
        self.assertEqual(rpc.calls, ["eth_chainId"]); self.assertEqual(report["external_call_counts"], {"rpc": 1, "source": 0, "deepseek": 0})

    def test_ready_topk_v3_calls_once_and_discards_provider_free_text(self):
        calls = []
        def deep(preparation): calls.append(preparation); return _v3_result(preparation)
        report = pilot.run_topk_v3_single_pilot(pilot.build_locked_manifest(), live=True, rpc_client=Rpc(["0x1", {}]), evidence_builder=lambda raw: _bundle(), source_builder=lambda bundle: _source(bundle), deepseek_runner=deep, location_loader=lambda tx: {"function": "f"})
        self.assertEqual(len(calls), 1); self.assertEqual(report["external_call_counts"], {"rpc": 2, "source": 0, "deepseek": 1})
        self.assertEqual(report["top3"][0]["role"], "PRIMARY_CANDIDATE")
        self.assertFalse(report["llm"]["provider_free_text_used"])

    def test_window_granularity_is_reported_without_changing_request_limits(self):
        report = pilot.run_topk_v3_single_pilot(pilot.build_locked_manifest(), live=True, rpc_client=Rpc(["0x1", {}]), evidence_builder=lambda raw: _bundle(), source_builder=lambda bundle: _source(bundle, "VERIFIED_SOURCE_WINDOW_FALLBACK"), deepseek_runner=_v3_result, location_loader=lambda tx: {"function": "f"})
        self.assertEqual(report["evaluation_granularity"], "SOURCE_WINDOW")
        self.assertEqual(report["evaluation_result"], "TOP1_WINDOW_HIT")

    def test_invalid_or_missing_source_does_not_retry_llm(self):
        report = pilot.run_topk_v3_single_pilot(pilot.build_locked_manifest(), live=True, rpc_client=Rpc(["0x1", {}]), evidence_builder=lambda raw: _bundle(), source_builder=lambda bundle: ({"schema_version": "source_request_slices_v1", "requests": []}, {"source_request_count": 0, "stop_code": None}), deepseek_runner=lambda prep: self.fail("must not call"))
        self.assertEqual(report["external_call_counts"]["deepseek"], 0)
        self.assertEqual(report["run_status"], "COMPLETED_SAFE_STOP")

    def test_atomic_output_and_report_redaction(self):
        report = pilot.run_topk_v3_single_pilot(pilot.build_locked_manifest(), live=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"; pilot._atomic(path, report)
            self.assertTrue(path.exists()); self.assertEqual(len(list(Path(directory).iterdir())), 1)
        rendered = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(pilot.TX_HASH, rendered); self.assertNotRegex(rendered, r"0x[0-9a-fA-F]{40}")


if __name__ == "__main__":
    unittest.main()

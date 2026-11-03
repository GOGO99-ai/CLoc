from __future__ import annotations

import importlib.util
import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekResult

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_phase2_fault3_v2_pilot.py"
_SPEC = importlib.util.spec_from_file_location("run_phase2_fault3_v2_pilot", _PATH)
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


def _source(_bundle):
    return ({"schema_version": "source_request_slices_v1", "requests": [{"node_id": "n1", "function_match_status": "RESOLVED", "file_path": "C.sol", "function_name": "f", "signature": "f()", "start_line": 1, "end_line": 1, "source": "1: function f(){}", "uncertainties": []}]}, {"source_request_count": 0, "verified_source_count": 1, "cache_hit_count": 1, "resolved_slice_count": 1, "stop_code": None})


def _v2_result(prep, status="LOCATED"):
    findings = [] if status == "INSUFFICIENT_EVIDENCE" else [{"rank": 1, "source_slice_id": "S1", "role": "ROOT_CAUSE" if status == "LOCATED" else "UNCERTAIN", "confidence": "HIGH" if status == "LOCATED" else "LOW", "evidence_ids": ["E1"], "explanation": "S1由E1(CALL)支持", "explanation_source": "LOCAL_DETERMINISTIC_EVIDENCE_SUMMARY"}]
    response = {"schema_version": "phase2_localization_materialized_v2", "status": status, "findings": findings, "summary": "状态", "provider_free_text_used": False, "explanation_source": "LOCAL_DETERMINISTIC_EVIDENCE_SUMMARY"}
    return Phase2DeepSeekResult("VALIDATED", prep.prompt_sha256, "b" * 64, "deepseek-v4-pro", None, ("RESPONSE_VALIDATED_V2",), json.dumps(response))


class Phase2Fault3V2PilotTests(unittest.TestCase):
    def test_single_window_manifest_is_only_fault003_with_hard_limits(self):
        manifest = pilot.build_window_single_manifest()
        self.assertEqual(manifest.samples, (pilot.SAMPLES[2],)); self.assertEqual(manifest.request_limits, (2, 0, 1))
        self.assertEqual(manifest.experiment_kind, "PHASE2_VERIFIED_SOURCE_WINDOW_SINGLE_PILOT")
    def test_manifest_fixed_stable_and_excludes_fault007(self):
        one, two = pilot.build_locked_manifest(), pilot.build_locked_manifest()
        self.assertEqual(one.manifest_sha256, two.manifest_sha256); self.assertEqual(len(one.samples), 3)
        self.assertNotIn("0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9", [item[1] for item in one.samples])

    def test_offline_mode_makes_zero_requests(self):
        report = pilot.run_fault3_v2_pilot(pilot.build_locked_manifest(), live=False)
        self.assertEqual(report["external_call_counts"], {"rpc": 0, "source": 0, "deepseek": 0})

    def test_chain_mismatch_stops_before_trace(self):
        rpc = Rpc(["0x38"])
        report = pilot.run_fault3_v2_pilot(pilot.build_locked_manifest(), live=True, rpc_client=rpc, source_builder=_source, deepseek_runner=lambda prep: None)
        self.assertEqual(rpc.calls, ["eth_chainId"]); self.assertEqual(report["run_status"], "STOPPED")

    def test_three_samples_each_use_one_trace_and_one_v2_llm(self):
        rpc = Rpc(["0x1", {}, {}, {}]); calls = []
        def deep(prep): calls.append(prep); return _v2_result(prep)
        report = pilot.run_fault3_v2_pilot(pilot.build_locked_manifest(), live=True, rpc_client=rpc, evidence_builder=lambda raw: _bundle(), source_builder=_source, deepseek_runner=deep, location_loader=lambda tx: None)
        self.assertEqual(report["run_status"], "COMPLETED"); self.assertEqual(len(calls), 3); self.assertEqual(report["external_call_counts"], {"rpc": 4, "source": 0, "deepseek": 3})
        self.assertTrue(all(row["llm"]["provider_free_text_used"] is False for row in report["samples"]))
        self.assertTrue(all(row["top3"][0]["explanation_source"] == "LOCAL_DETERMINISTIC_EVIDENCE_SUMMARY" for row in report["samples"]))

    def test_invalid_core_response_is_not_retried_or_replaced(self):
        rpc = Rpc(["0x1", {}, {}, {}]); calls = []
        def deep(prep):
            calls.append(1)
            return Phase2DeepSeekResult("INVALID_RESPONSE", prep.prompt_sha256, "b" * 64, "deepseek-v4-pro", None, ("FINDING_EVIDENCE_SCOPE_VIOLATION",))
        report = pilot.run_fault3_v2_pilot(pilot.build_locked_manifest(), live=True, rpc_client=rpc, evidence_builder=lambda raw: _bundle(), source_builder=_source, deepseek_runner=deep)
        self.assertEqual(len(calls), 3); self.assertEqual(len(report["samples"]), 3); self.assertTrue(all(row["llm"]["response_status"] == "INVALID_RESPONSE" for row in report["samples"]))

    def test_network_error_stops_and_checkpoint_is_atomic(self):
        snapshots = []; rpc = Rpc(["0x1", RuntimeError("RPC_NETWORK_ERROR")])
        report = pilot.run_fault3_v2_pilot(pilot.build_locked_manifest(), live=True, rpc_client=rpc, evidence_builder=lambda raw: _bundle(), source_builder=_source, deepseek_runner=lambda prep: _v2_result(prep), checkpoint_callback=snapshots.append)
        self.assertEqual(report["run_status"], "STOPPED"); self.assertEqual(report["external_call_counts"]["rpc"], 2); self.assertTrue(snapshots)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"; pilot._atomic(path, {"ok": True}); self.assertEqual(json.loads(path.read_text()), {"ok": True})

    def test_location_does_not_change_v2_prompt_or_result(self):
        rpc = Rpc(["0x1", {}, {}, {}]); prompts = []
        def deep(prep): prompts.append(prep.prompt_sha256); return _v2_result(prep)
        first = pilot.run_fault3_v2_pilot(pilot.build_locked_manifest(), live=True, rpc_client=rpc, evidence_builder=lambda raw: _bundle(), source_builder=_source, deepseek_runner=deep, location_loader=lambda tx: {"function": "f"})
        self.assertEqual(len(set(prompts)), 1); self.assertTrue(all(row["evaluation_result"] in {"TOP1_HIT", "TOP3_HIT"} for row in first["samples"]))

    def test_report_is_redacted_and_has_hard_limits(self):
        report = pilot.build_report(pilot.build_locked_manifest(), [], rpc=0, source=0, deepseek=0, run_status="X")
        self.assertEqual(report["request_limits"], {"rpc": 4, "source": 9, "deepseek": 3})
        report["reason_codes"] = ["0x" + "1" * 40]
        with self.assertRaisesRegex(ValueError, "SENSITIVE"): pilot.validate_report(report)

    def test_cache_only_fallback_never_uses_network_and_window_granularity_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); address = "0x" + "1" * 40; text = "contract C {\nfunction f() external {\nuint x = 1;\n}\n}"
            file = root / "eth" / address / "C.sol"; file.parent.mkdir(parents=True); file.write_bytes(text.encode("utf-8"))
            (file.parent / "manifest.json").write_text(json.dumps({"status": "verified", "verified": True, "requested_address": address, "resolved_address": address, "files": [{"path": "C.sol", "sha256": hashlib.sha256(text.encode()).hexdigest()}], "proxy": {}}), encoding="utf-8")
            graph = {"schema_version": "transaction_graph_v1", "nodes": [{"node_id": "n1", "callee_address": address}]}
            bundle = SimpleNamespace(candidate_addresses=(address,), candidate_nodes=("n1",), graph=SimpleNamespace(to_dict=lambda: graph))
            slices, summary = pilot.acquire_cached_source_slices(bundle, cache_root=root)
            self.assertEqual(summary["source_request_count"], 0); self.assertEqual(summary["source_resolution_mode"], "VERIFIED_SOURCE_WINDOW_FALLBACK"); self.assertEqual(summary["window_count"], 1)

    def test_cache_graph_fallback_can_use_a_connected_non_candidate_node(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source_address = "0x" + "1" * 40; other = "0x" + "2" * 40; result = "0x" + "3" * 40
            text = "contract C { function run() external {} }"; file = root / "eth" / source_address / "C.sol"; file.parent.mkdir(parents=True); file.write_text(text, encoding="utf-8")
            (file.parent / "manifest.json").write_text(json.dumps({"status": "verified", "verified": True, "requested_address": source_address, "resolved_address": source_address, "files": [{"path": "C.sol", "sha256": hashlib.sha256(text.encode()).hexdigest()}], "proxy": {}}), encoding="utf-8")
            graph = {"schema_version": "transaction_graph_v1", "nodes": [{"node_id": "n1", "callee_address": source_address, "function_name": "run", "function_selector": "0x00000000"}, {"node_id": "n2", "callee_address": other}, {"node_id": "n3", "callee_address": result}], "call_edges": [{"edge_id": "call_e1", "edge_type": "CALL", "src": "n1", "dst": "n2"}, {"edge_id": "call_e2", "edge_type": "CALL", "src": "n2", "dst": "n3"}]}
            strict = {"schema_version": "strict_result_chains_v1", "edges": [], "chains": [{"strict_chain_id": "c1", "node_ids": ["n2", "n3"]}]}
            bundle = SimpleNamespace(candidate_addresses=(other,), candidate_nodes=("n2",), graph=SimpleNamespace(to_dict=lambda: graph), strict=SimpleNamespace(to_dict=lambda: strict))
            slices, summary = pilot.acquire_cached_source_slices(bundle, cache_root=root)
            self.assertEqual(summary["source_request_count"], 0)
            self.assertEqual(summary["source_selection_mode"], "TRANSACTION_CALL_GRAPH_FALLBACK")
            self.assertEqual(slices["metadata"]["graph_fallback_associations"][0]["call_edge_ids"], ["call_e1", "call_e2"])


if __name__ == "__main__":
    unittest.main()

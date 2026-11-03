from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "revalidate_phase1_direct_identity.py"
_SPEC = importlib.util.spec_from_file_location("direct_identity_revalidation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC)
import sys
sys.modules[_SPEC.name] = pilot
_SPEC.loader.exec_module(pilot)


class _FakeRpc:
    def __init__(self, values): self.values = list(values); self.calls = []
    def call(self, method, params):
        self.calls.append((method, params))
        value = self.values.pop(0)
        if isinstance(value, Exception): raise value
        return value


def _result(sample, *, direct=False, collision=False):
    return {"sample_alias": sample.sample_alias, "stage_counts": {"call_node_count": 1}, "query_identity": {"status": "KNOWN_NORMAL_ORIGIN" if direct else "NOT_IN_NORMAL_LIBRARY", "reason_codes": [], "usable_for_direct_normal_identity": direct}, "usable_clean_chain_count": 1, "uncertain_chain_count": 0, "known_transaction_direct_normal": direct, "structural_exact_collision": collision, "decision_kind": "KNOWN_TRANSACTION_DIRECT_NORMAL" if direct else "STRUCTURAL_EXACT_COLLISION_PHASE2" if collision else "CONTINUE_STANDARD_PHASE1", "final_route": "EARLY_NORMAL" if direct else "PROCEED_PHASE2", "reason_codes": ["KNOWN_NORMAL_TRANSACTION_SELF_MATCH"] if direct else ["STRUCTURAL_EXACT_MATCH_WITHOUT_TRANSACTION_IDENTITY"], "chain_matches": [{"chain_alias": "Q1", "structural_exact_match": True, "self_origin_exact_match": direct, "reference_origin_status": "SELF_ORIGIN_EXACT_MATCH" if direct else "QUERY_NOT_IN_NORMAL_LIBRARY", "call_distance": "0", "asset_distance": "0", "chain_distance": "0", "normal_group_size": 3, "top3_count": 3, "reason_codes": []}]}


class DirectIdentityRevalidationTests(unittest.TestCase):
    def _history(self, root):
        hashes = ["0xe182381051bd888098e536eb4f7135706f97035ad29e010e87baa90fbb7af0b1", "0xc9e5bf9698852125dba209bfca289a7f712026499d9a2c5994ee6e2eb005470c", "0x509b134247f308ef9e190cf2a9f9b902955c4cfb120c107933162e4dfa561c9e"]
        manifest = root / "manifest.json"; report = root / "report.json"
        manifest.write_text(json.dumps({"samples": [{"sample_alias": f"normal_random10_eth_{i:03d}", "tx_hash": h} for i, h in enumerate(hashes, 2)]}), encoding="utf-8")
        report.write_text(json.dumps({"sample_results": [{"sample_alias": f"normal_random10_eth_{i:03d}", "direct_normal": True, "self_reference_match": False} for i in range(2, 5)]}), encoding="utf-8")
        return manifest, report

    def test_manifest_is_stable_and_locks_fault_and_fourth(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path, report_path = self._history(Path(temp))
            first = pilot.build_locked_identity_revalidation_manifest(random_manifest=manifest_path, random_report=report_path)
            second = pilot.build_locked_identity_revalidation_manifest(random_manifest=manifest_path, random_report=report_path)
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(first.samples[0].sample_alias, "identity_recheck_fault_001")
        self.assertEqual(len(first.samples), 4)
        self.assertEqual(first.samples[-1].tx_hash, "0x509b134247f308ef9e190cf2a9f9b902955c4cfb120c107933162e4dfa561c9e")

    def test_history_ambiguity_locks_only_three(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, report = self._history(root)
            report.write_text(json.dumps({"sample_results": []}), encoding="utf-8")
            value = pilot.build_locked_identity_revalidation_manifest(random_manifest=manifest, random_report=report)
        self.assertEqual(len(value.samples), 3)

    def test_offline_mode_uses_zero_requests(self):
        manifest = pilot.LockedIdentityManifest((pilot.LockedIdentitySample("identity_recheck_fault_001", 1, "0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9"), pilot.LockedIdentitySample("identity_recheck_normal_001", 1, "0x01e6371b60c2ca36b3673bab9edf4caeba03e6dc756c317a90f856c5735cd2c4"), pilot.LockedIdentitySample("identity_recheck_normal_002", 1, "0x01f8b613d9f87f0f7fcde83c35a550b4fc758e8d218086de8ab39f31db9e72b0")), "0" * 64, "HISTORICAL_EVIDENCE_UNAVAILABLE")
        report = pilot.run_direct_identity_revalidation(manifest)
        self.assertEqual(report["external_call_counts"], {"rpc": 0, "deepseek": 0, "network": 0})

    def test_chain_mismatch_does_not_request_trace(self):
        manifest = self._three_manifest()
        rpc = _FakeRpc(["0x38"])
        loaded = type("Loaded", (), {"summary": type("Summary", (), {"formal_min_group_size": 3, "manifest_consistency_passed": True, "reconstructed_baseline_match": True, "ethereum_only": True, "to_dict": lambda self: {}})()})()
        report = pilot.run_direct_identity_revalidation(manifest, live=True, rpc_client=rpc, loaded_baseline=loaded)
        self.assertEqual(len(rpc.calls), 1); self.assertEqual(report["run_status"], "STOPPED")

    def test_network_error_stops_without_retry_and_saves_checkpoint(self):
        manifest = self._three_manifest(); rpc = _FakeRpc(["0x1", RuntimeError("RPC_NETWORK_ERROR")]); checkpoints = []
        report = pilot.run_direct_identity_revalidation(manifest, live=True, rpc_client=rpc, loaded_baseline=self._loaded(), sample_processor=lambda *args: _result(args[0]), checkpoint_callback=checkpoints.append)
        self.assertEqual(len(rpc.calls), 2); self.assertEqual(report["run_status"], "STOPPED"); self.assertTrue(checkpoints)

    def test_direct_and_collision_rows_are_sanitized(self):
        manifest = self._three_manifest(); rpc = _FakeRpc(["0x1", {}, {}, {}])
        values = iter([False, True, False])
        report = pilot.run_direct_identity_revalidation(manifest, live=True, rpc_client=rpc, loaded_baseline=self._loaded(), sample_processor=lambda sample, *_: _result(sample, direct=next(values)) )
        self.assertEqual(report["sample_results"][1]["final_route"], "EARLY_NORMAL")
        text = pilot.serialize_direct_identity_revalidation_report(report)
        self.assertNotIn(manifest.samples[0].tx_hash, text); self.assertEqual(report["external_call_counts"]["deepseek"], 0)

    def test_atomic_artifacts_keep_manifest_hashes_out_of_report(self):
        manifest = self._three_manifest(); report = pilot.build_direct_identity_revalidation_report(manifest, {}, {}, rpc_request_count=0, run_status="LIVE_MODE_REQUIRED")
        with tempfile.TemporaryDirectory() as temp:
            paths = pilot.write_direct_identity_revalidation_artifacts(Path(temp), manifest, report)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertNotIn(manifest.samples[0].tx_hash, paths[2].read_text(encoding="utf-8"))

    def _three_manifest(self):
        return pilot.LockedIdentityManifest((pilot.LockedIdentitySample("identity_recheck_fault_001", 1, "0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9"), pilot.LockedIdentitySample("identity_recheck_normal_001", 1, "0x01e6371b60c2ca36b3673bab9edf4caeba03e6dc756c317a90f856c5735cd2c4"), pilot.LockedIdentitySample("identity_recheck_normal_002", 1, "0x01f8b613d9f87f0f7fcde83c35a550b4fc758e8d218086de8ab39f31db9e72b0")), "1" * 64, "HISTORICAL_EVIDENCE_UNAVAILABLE")

    def _loaded(self):
        return type("Loaded", (), {"summary": type("Summary", (), {"formal_min_group_size": 3, "manifest_consistency_passed": True, "reconstructed_baseline_match": True, "ethereum_only": True, "to_dict": lambda self: {}})()})()


if __name__ == "__main__":
    unittest.main()

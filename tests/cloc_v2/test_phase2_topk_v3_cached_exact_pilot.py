from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_phase2_topk_v3_cached_exact_pilot.py"
_SPEC = importlib.util.spec_from_file_location("run_phase2_topk_v3_cached_exact_pilot", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC); sys.modules[_SPEC.name] = pilot; _SPEC.loader.exec_module(pilot)


class CachedExactPilotTests(unittest.TestCase):
    def test_locked_transaction_manifest_and_limits_are_stable(self):
        first, second = pilot.build_locked_manifest(), pilot.build_locked_manifest()
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(first.samples, ((pilot.SAMPLE_ALIAS, pilot.TX_HASH),))
        self.assertEqual(first.request_limits, (2, 0, 1))
        self.assertEqual(first.experiment_kind, "PHASE2_TOPK_V3_CACHED_EXACT_SINGLE_PILOT")

    def test_offline_preflight_confirms_exact_cached_slice_contract(self):
        address = "0x" + "1" * 40
        graph = SimpleNamespace(nodes=tuple(SimpleNamespace(callee_address=address, node_id=f"n{i}") for i in range(880)))
        binding = SimpleNamespace(artifact_address=address, solidity_file_count=1)
        lookup = SimpleNamespace(status="VERIFIED_SOURCE_CACHE_HIT", binding=binding)
        slices = {"requests": [{"function_match_status": "RESOLVED"}, {"function_match_status": "RESOLVED"}]}
        summary = {"source_request_count": 0, "cache_hit_count": 1, "window_count": 0, "resolution_mode": "EXACT_FUNCTION_SLICE"}
        source_index = {"addresses": [{"status": "verified", "functions": [{} for _ in range(49)]}]}
        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "tree.json"; tree.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
            with patch.object(pilot, "_cached_tree_path", return_value=tree), patch.object(pilot, "build_transaction_graph", return_value=graph), patch.object(pilot, "build_verified_source_cache_index", return_value=object()), patch.object(pilot, "lookup_verified_source_cache", return_value=lookup), patch.object(pilot, "acquire_cached_source_slices", return_value=(slices, summary)), patch.object(pilot, "build_source_index", return_value=source_index):
                result = pilot.preflight_cached_exact()
        self.assertEqual(result["call_node_count"], 880)
        self.assertEqual(result["cache_hit_address_count"], 1)
        self.assertEqual(result["solidity_file_count"], 1)
        self.assertEqual(result["indexed_function_count"], 49)
        self.assertEqual(result["exact_slice_count"], 2)
        self.assertEqual(result["window_count"], 0)
        self.assertEqual(result["source_network_request_count"], 0)

    def test_conflict_or_cache_preflight_failure_stops_before_live_setup(self):
        with patch.object(pilot, "preflight_cached_exact", side_effect=ValueError("VERIFIED_SOURCE_CACHE_CONFLICT")):
            self.assertEqual(pilot.main([]), 1)

    def test_no_live_mode_never_constructs_rpc_or_deepseek(self):
        fixture = {"call_node_count": 880, "cache_hit_address_count": 1, "solidity_file_count": 1, "indexed_function_count": 49, "exact_slice_count": 2, "window_count": 0, "source_resolution_mode": "EXACT_FUNCTION_SLICE", "source_network_request_count": 0}
        with patch.object(pilot, "preflight_cached_exact", return_value=fixture), patch.object(pilot, "load_chain_config", side_effect=self.fail):
            self.assertEqual(pilot.main([]), 2)


if __name__ == "__main__":
    unittest.main()

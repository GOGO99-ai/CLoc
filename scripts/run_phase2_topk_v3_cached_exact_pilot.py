"""One-shot Top-K v3 pilot using only a pre-verified local exact-source cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.run_phase1_deepseek_smoke import DeepSeekHTTPSOneShotTransport
from scripts.run_phase2_fault3_v2_pilot import _atomic, _code, _location_after_inference, acquire_cached_source_slices, run_fault3_v2_pilot
from scripts.run_phase2_real_pilot import Phase2EvidenceBundle
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekConfig, execute_phase2_localization
from src.cloc_v2.source_index import build_source_index
from src.cloc_v2.transaction_graph import build_transaction_graph
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index, lookup_verified_source_cache

KIND = "PHASE2_TOPK_V3_CACHED_EXACT_SINGLE_PILOT"
SAMPLE_ALIAS = "phase2_topk_v3_exact_001"
TX_HASH = "0x44aad3b853866468161735496a5d9cc961ce5aa872924c5d78673076b1cd95aa"
OUTPUT = ROOT / "data" / "prototypes" / "phase2_topk_v3_cached_exact_001_20260822"
CACHE_ROOT = ROOT / "data" / "source_cache"
TREE_ROOT = ROOT / "data" / "cache" / "all_tree"


def _canonical(value: Any) -> str: return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class LockedCachedExactManifest:
    samples: tuple[tuple[str, str], ...] = ((SAMPLE_ALIAS, TX_HASH),)
    experiment_kind: str = KIND
    request_limits: tuple[int, int, int] = (2, 0, 1)
    @property
    def manifest_sha256(self) -> str:
        value = {"schema_version": "phase2_topk_v3_cached_exact_manifest_v1", "experiment_kind": self.experiment_kind, "samples": [{"sample_alias": SAMPLE_ALIAS, "blockchain_chain_id": 1, "tx_hash": TX_HASH}]}
        return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "phase2_topk_v3_cached_exact_manifest_v1", "experiment_kind": self.experiment_kind, "scientific_experiment_result": False, "excluded_from_accuracy_metrics": True, "samples": [{"sample_alias": SAMPLE_ALIAS, "blockchain_chain_id": 1, "tx_hash": TX_HASH}], "manifest_sha256": self.manifest_sha256}


def build_locked_manifest() -> LockedCachedExactManifest: return LockedCachedExactManifest()


def _cached_tree_path() -> Path:
    matches = sorted(TREE_ROOT.rglob(f"{TX_HASH}.json"))
    if len(matches) != 1: raise ValueError("CACHED_CALL_TREE_NOT_UNIQUE")
    return matches[0]


def preflight_cached_exact() -> dict[str, int | str]:
    """Offline preflight: validate cache integrity and exact slices from the saved call tree."""
    tree = json.loads(_cached_tree_path().read_text(encoding="utf-8"))
    graph = build_transaction_graph(tree)
    addresses = tuple(dict.fromkeys(node.callee_address.lower() for node in graph.nodes if isinstance(node.callee_address, str)))
    index = build_verified_source_cache_index(CACHE_ROOT, blockchain_chain_id=1, chain_key="eth")
    hits = [lookup_verified_source_cache(index, runtime_address=address, requested_address=address) for address in addresses]
    if any(item.status == "VERIFIED_SOURCE_CACHE_CONFLICT" for item in hits): raise ValueError("VERIFIED_SOURCE_CACHE_CONFLICT")
    hit_bindings = [item.binding for item in hits if item.binding is not None]
    if len({item.artifact_address for item in hit_bindings}) != 1: raise ValueError("VERIFIED_SOURCE_CACHE_HIT_COUNT_INVALID")
    bundle = Phase2EvidenceBundle(graph, None, {}, {}, tuple(node.node_id for node in graph.nodes), addresses)
    slices, summary = acquire_cached_source_slices(bundle, cache_root=CACHE_ROOT)
    exact = sum(item.get("function_match_status") == "RESOLVED" for item in slices.get("requests", ()))
    source_index = build_source_index({"schema_version": "source_acquisition_v1", "chain": "eth", "addresses": [{"requested_address": hit_bindings[0].artifact_address}]}, CACHE_ROOT)
    function_count = sum(len(item.get("functions", ())) for item in source_index.get("addresses", ()) if item.get("status") == "verified")
    if summary.get("source_request_count") != 0 or summary.get("cache_hit_count") != 1 or exact != 2 or function_count != 49 or summary.get("resolution_mode", summary.get("source_resolution_mode")) not in {"EXACT_FUNCTION_SLICE", None}:
        raise ValueError("CACHED_EXACT_PREFLIGHT_FAILED")
    return {"call_node_count": len(graph.nodes), "cache_hit_address_count": 1, "solidity_file_count": int(hit_bindings[0].solidity_file_count), "indexed_function_count": function_count, "exact_slice_count": exact, "window_count": int(summary.get("window_count", 0)), "source_resolution_mode": "EXACT_FUNCTION_SLICE", "source_network_request_count": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--live", action="store_true"); parser.add_argument("--output-dir", type=Path, default=OUTPUT); args = parser.parse_args(argv)
    try: preflight = preflight_cached_exact()
    except Exception as exc:
        print("pilot_status=STOPPED"); print("reason_code=" + _code(exc, "CACHED_EXACT_PREFLIGHT_FAILED")); return 1
    if not args.live:
        print("live_required=true"); print("rpc_request_count=0"); print("source_request_count=0"); print("deepseek_request_count=0"); return 2
    manifest = build_locked_manifest(); checkpoint = args.output_dir / "checkpoint.json"; report_path = args.output_dir / "phase2_topk_v3_cached_exact_report.json"
    _atomic(args.output_dir / "locked_manifest.json", manifest.to_dict())
    _atomic(checkpoint, {"schema_version": "phase2_topk_v3_cached_exact_checkpoint_v1", "manifest_sha256": manifest.manifest_sha256, "status": "LOCKED", "rpc_request_count": 0, "source_request_count": 0, "deepseek_request_count": 0, "samples": []})
    try:
        config = load_chain_config("eth"); rpc_url = config.get("rpc_url")
        if not isinstance(rpc_url, str) or not rpc_url: raise ValueError("ETH_RPC_CONFIG_NOT_LOADED")
        key = os.getenv("DEEPSEEK_API_KEY")
        if not isinstance(key, str) or not key: raise ValueError("DEEPSEEK_KEY_NOT_LOADED")
        rpc = JsonRpcClient(rpc_url, max_requests=2)
        def deepseek_runner(preparation: Any): return execute_phase2_localization(preparation, Phase2DeepSeekConfig(), DeepSeekHTTPSOneShotTransport(), key)
        report = run_fault3_v2_pilot(manifest, live=True, rpc_client=rpc, source_builder=lambda bundle: acquire_cached_source_slices(bundle, cache_root=CACHE_ROOT), deepseek_runner=deepseek_runner, location_loader=_location_after_inference, checkpoint_callback=lambda value: _atomic(checkpoint, value))
        report["schema_version"] = "phase2_topk_v3_cached_exact_report_v1"; report["preflight"] = preflight; report["excluded_from_accuracy_metrics"] = True
        _atomic(report_path, report)
    except Exception as exc:
        print("pilot_status=STOPPED"); print("reason_code=" + _code(exc, "CACHED_EXACT_SETUP_ERROR")); return 1
    print("pilot_status=" + report["run_status"]); print("report_path=" + str(report_path)); print("rpc_request_count=" + str(report["external_call_counts"]["rpc"])); print("source_request_count=" + str(report["external_call_counts"]["source"])); print("deepseek_request_count=" + str(report["external_call_counts"]["deepseek"])); return 0 if report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__": raise SystemExit(main())

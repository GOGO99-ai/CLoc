"""Three-sample, one-shot phase-2 v2 localization engineering pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase1_deepseek_smoke import DeepSeekHTTPSOneShotTransport
from scripts.run_phase2_real_pilot import (
    MAX_SOURCE_REQUESTS,
    Phase2EvidenceBundle,
    acquire_real_source_slices,
    build_real_phase2_evidence,
    evaluate_top3_locations,
    map_validated_top3,
    prepare_phase2_topk_v3,
)
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.phase2_localization_bridge import prepare_phase2_localization
from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekConfig, Phase2DeepSeekResult, execute_phase2_localization
from src.cloc_v2.source_index import build_source_index
from src.cloc_v2.source_slices import build_source_request_slices
from src.cloc_v2.source_window_fallback import build_verified_source_window_fallback
from src.cloc_v2.phase2_source_graph_fallback import build_phase2_source_graph_fallback
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index, lookup_verified_source_cache

KIND = "PHASE2_FAULT3_V2_ENGINEERING_PILOT"
DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "phase2_fault3_v2_pilot_20260822"
DEFAULT_WINDOW_OUTPUT = ROOT / "data" / "prototypes" / "phase2_window_single_fault003_20260822"
MAX_RPC = 4
MAX_SOURCE = 9
MAX_DEEPSEEK = 3
SAMPLES = (
    ("phase2_fault3_001", "0x0163856aaa678676b22e072186c1bdc55ed44a5a0e2cf44500362800663ddce9"),
    ("phase2_fault3_002", "0xa9a1b8ea288eb9ad315088f17f7c7386b9989c95b4d13c81b69d5ddad7ffe61e"),
    ("phase2_fault3_003", "0x38ecc8363836e43aca99d994bb9325a8f3770d7a1d1fffd62c82cf5230525836"),
)
_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _code(error: Exception, fallback: str) -> str:
    for candidate in (str(error), error.__class__.__name__):
        match = _CODE.search(candidate.upper())
        if match:
            return match.group()
    return fallback


@dataclass(frozen=True)
class LockedFault3Manifest:
    samples: tuple[tuple[str, str], ...] = SAMPLES
    experiment_kind: str = KIND
    request_limits: tuple[int, int, int] = (MAX_RPC, MAX_SOURCE, MAX_DEEPSEEK)

    def __post_init__(self) -> None:
        if len({alias for alias, _ in self.samples}) != len(self.samples):
            raise ValueError("PHASE2_FAULT3_MANIFEST_INVALID")
        if self.experiment_kind == KIND and (len(self.samples) != 3 or tuple(alias for alias, _ in self.samples) != tuple(f"phase2_fault3_{i:03d}" for i in range(1, 4))):
            raise ValueError("PHASE2_FAULT3_MANIFEST_ORDER_INVALID")
        if self.experiment_kind == "PHASE2_VERIFIED_SOURCE_WINDOW_SINGLE_PILOT" and self.samples != (SAMPLES[2],):
            raise ValueError("PHASE2_WINDOW_SINGLE_MANIFEST_INVALID")
        if len(self.request_limits) != 3 or any(not isinstance(value, int) or value < 0 for value in self.request_limits):
            raise ValueError("PHASE2_FAULT3_LIMITS_INVALID")
        if any(not isinstance(tx_hash, str) or _HASH.fullmatch(tx_hash.lower()) is None for _, tx_hash in self.samples):
            raise ValueError("PHASE2_FAULT3_HASH_INVALID")
        if any(tx_hash.lower() == "0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9" for _, tx_hash in self.samples):
            raise ValueError("FAULT007_FORBIDDEN")

    @property
    def manifest_sha256(self) -> str:
        return _sha({"schema_version": "phase2_fault3_v2_manifest_v1", "experiment_kind": self.experiment_kind, "samples": [{"sample_alias": alias, "blockchain_chain_id": 1, "tx_hash": tx_hash.lower()} for alias, tx_hash in self.samples]})

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "phase2_fault3_v2_manifest_v1", "experiment_kind": self.experiment_kind, "scientific_experiment_result": False, "samples": [{"sample_alias": alias, "blockchain_chain_id": 1, "tx_hash": tx_hash.lower()} for alias, tx_hash in self.samples], "manifest_sha256": self.manifest_sha256}


def build_locked_manifest() -> LockedFault3Manifest:
    return LockedFault3Manifest()


def build_window_single_manifest() -> LockedFault3Manifest:
    return LockedFault3Manifest((SAMPLES[2],), "PHASE2_VERIFIED_SOURCE_WINDOW_SINGLE_PILOT", (2, 0, 1))


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno()); temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _location_after_inference(tx_hash: str) -> Any:
    """Read only the independently stored location after a frozen result."""
    for path in sorted((ROOT / "dataset").rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, Mapping):
                fault = item.get("fault")
                if isinstance(fault, Mapping) and str(fault.get("transaction_hash", "")).lower() == tx_hash.lower():
                    return item.get("location")
                stack.extend(value for key, value in item.items() if key in {"data", "items", "records"})
    return None


def acquire_cached_source_slices(bundle: Phase2EvidenceBundle, *, cache_root: Path = ROOT / "data" / "source_cache") -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Offline-only source stage: absent cache is evidence absence, never a fetch."""
    cache_index = build_verified_source_cache_index(cache_root, blockchain_chain_id=1, chain_key="eth")
    addresses = []
    statuses: dict[str, int] = {}
    for address in bundle.candidate_addresses:
        lookup = lookup_verified_source_cache(cache_index, runtime_address=address, requested_address=address)
        statuses[lookup.status] = statuses.get(lookup.status, 0) + 1
        if lookup.status == "VERIFIED_SOURCE_CACHE_HIT" and lookup.binding is not None:
            addresses.append({"requested_address": lookup.binding.artifact_address})
    acquisition = {"schema_version": "source_acquisition_v1", "chain": "eth", "addresses": addresses}
    index = build_source_index(acquisition, cache_root)
    requests = [{"chain_id": "Q", "node_id": node_id, "focus": "phase2_localization", "priority": 1 if position == 0 else 2} for position, node_id in enumerate(bundle.candidate_nodes)]
    exact = build_source_request_slices(index, bundle.graph.to_dict(), cache_root, requests)
    slices = build_verified_source_window_fallback(index, bundle.graph.to_dict(), exact, cache_root)
    strict = getattr(bundle, "strict", None)
    if strict is not None:
        slices = build_phase2_source_graph_fallback(bundle.graph, strict, slices, cache_index, cache_root)
    metadata = slices.get("metadata", {})
    verified = sum(item.get("status") == "verified" and item.get("verified") for item in index.get("addresses", ()) if isinstance(item, Mapping))
    windows = [{"relative_file": item["file_path"], "start_line": item["start_line"], "end_line": item["end_line"]} for item in slices.get("requests", ()) if item.get("function_match_status") == "WINDOW_FALLBACK"]
    return slices, {"source_request_count": 0, "attempted_address_count": len(bundle.candidate_addresses), "verified_source_count": verified, "cache_hit_count": len(addresses), "cache_index_status_counts": dict(sorted(statuses.items())), "resolved_slice_count": metadata.get("resolved_count", 0), "window_count": metadata.get("window_count", 0), "window_file_count": metadata.get("window_file_count", 0), "source_resolution_mode": metadata.get("resolution_mode", "NONE"), "source_selection_mode": metadata.get("source_selection_mode", "DIRECT_EVIDENCE_SOURCE"), "windows": windows, "stop_code": None}


def _sanitized_sample(alias: str, *, bridge_status: str, stage_counts: Mapping[str, int] | None = None, source_summary: Mapping[str, Any] | None = None, result: Phase2DeepSeekResult | None = None, top3: list[Mapping[str, Any]] | None = None, evaluation: str = "EVALUATION_UNAVAILABLE", evaluation_granularity: str | None = None, bridge_reason_codes: tuple[str, ...] = (), reason_codes: tuple[str, ...] = ()) -> dict[str, Any]:
    response = result.validated_response if result and result.validated_response else {}
    return {"sample_alias": alias, "bridge_status": bridge_status, "source_resolution_mode": source_summary.get("source_resolution_mode") if source_summary else None, "bridge_reason_codes": sorted(set(bridge_reason_codes)), "stage_counts": dict(stage_counts or {}), "source_summary": dict(source_summary or {"source_request_count": 0, "verified_source_count": 0, "cache_hit_count": 0, "resolved_slice_count": 0, "window_count": 0, "window_file_count": 0}), "llm": {"called": result is not None, "response_status": result.response_status if result else "NOT_CALLED", "valid": result is not None and result.response_status in {"VALIDATED", "VALIDATED_WITH_NORMALIZATION"}, "status": response.get("status"), "provider_free_text_used": response.get("provider_free_text_used", False), "explanation_source": response.get("explanation_source", None), "reason_codes": list(result.reason_codes) if result else []}, "top3": [dict(row) for row in (top3 or [])], "evaluation_result": evaluation, "evaluation_granularity": evaluation_granularity, "reason_codes": sorted(set(reason_codes))}


def validate_report(report: Mapping[str, Any]) -> None:
    text = _canonical(report)
    blocked = ("reasoning_content", "rpc_url", "api_key", "provider_response", "system_message", "user_message", '"cause"')
    if _RAW_ADDRESS.search(text) or _RAW_HASH.search(text) or any(item in text.lower() for item in blocked):
        raise ValueError("PHASE2_FAULT3_REPORT_SENSITIVE_CONTENT")
    calls, limits = report["external_call_counts"], report["request_limits"]
    if any(calls[key] > limits[key] for key in calls):
        raise ValueError("PHASE2_FAULT3_REQUEST_LIMIT_EXCEEDED")


def build_report(manifest: LockedFault3Manifest, samples: list[Mapping[str, Any]], *, rpc: int, source: int, deepseek: int, run_status: str, reason_codes: tuple[str, ...] = ()) -> dict[str, Any]:
    report = {"schema_version": "phase2_fault3_v2_report_v1", "experiment_kind": manifest.experiment_kind, "scientific_experiment_result": False, "locked_manifest_sha256": manifest.manifest_sha256, "run_status": run_status, "samples": [dict(item) for item in samples], "external_call_counts": {"rpc": rpc, "source": source, "deepseek": deepseek}, "request_limits": {"rpc": manifest.request_limits[0], "source": manifest.request_limits[1], "deepseek": manifest.request_limits[2]}, "retry_count": 0, "endpoint_switch_count": 0, "sample_switch_count": 0, "reason_codes": sorted(set(reason_codes))}
    validate_report(report)
    return report


def run_fault3_v2_pilot(manifest: LockedFault3Manifest, *, live: bool, rpc_client: Any | None = None, evidence_builder: Callable[[Any], Phase2EvidenceBundle] = build_real_phase2_evidence, source_builder: Callable[[Phase2EvidenceBundle], tuple[Mapping[str, Any], Mapping[str, Any]]] | None = None, deepseek_runner: Callable[[Any], Phase2DeepSeekResult] | None = None, location_loader: Callable[[str], Any] | None = None, checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    if not live:
        return build_report(manifest, [], rpc=0, source=0, deepseek=0, run_status="LIVE_MODE_REQUIRED")
    if rpc_client is None or source_builder is None or deepseek_runner is None:
        raise ValueError("PHASE2_FAULT3_LIVE_DEPENDENCY_MISSING")
    rpc_count = source_count = deepseek_count = 0; rows: list[dict[str, Any]] = []
    checkpoint = {"schema_version": "phase2_fault3_v2_checkpoint_v1", "manifest_sha256": manifest.manifest_sha256, "status": "RUNNING", "rpc_request_count": 0, "source_request_count": 0, "deepseek_request_count": 0, "samples": []}
    def save() -> None:
        checkpoint.update(rpc_request_count=rpc_count, source_request_count=source_count, deepseek_request_count=deepseek_count, samples=[{"sample_alias": row["sample_alias"], "status": row["bridge_status"], "reason_codes": row["reason_codes"]} for row in rows])
        if checkpoint_callback: checkpoint_callback(json.loads(_canonical(checkpoint)))
    try:
        rpc_count += 1; chain = rpc_client.call("eth_chainId", [])
    except Exception as exc:
        checkpoint["status"] = "STOPPED"; save(); return build_report(manifest, rows, rpc=rpc_count, source=source_count, deepseek=deepseek_count, run_status="STOPPED", reason_codes=(_code(exc, "RPC_CHAIN_ID_ERROR"),))
    if str(chain).lower() not in {"0x1", "1"}:
        checkpoint["status"] = "STOPPED"; save(); return build_report(manifest, rows, rpc=rpc_count, source=source_count, deepseek=deepseek_count, run_status="STOPPED", reason_codes=("CHAIN_ID_MISMATCH",))
    for alias, tx_hash in manifest.samples:
        try:
            rpc_count += 1; raw = rpc_client.call("debug_traceTransaction", [tx_hash, {"tracer": "callTracer"}])
        except Exception as exc:
            checkpoint["status"] = "STOPPED"; save(); return build_report(manifest, rows, rpc=rpc_count, source=source_count, deepseek=deepseek_count, run_status="STOPPED", reason_codes=(_code(exc, "RPC_TRACE_ERROR"),))
        try:
            bundle = evidence_builder(raw)
        except Exception:
            rows.append(_sanitized_sample(alias, bridge_status="EVIDENCE_PROCESSING_ERROR", reason_codes=("EVIDENCE_PROCESSING_ERROR",))); save(); continue
        try:
            slices, source_summary = source_builder(bundle)
        except Exception:
            checkpoint["status"] = "STOPPED"; save(); return build_report(manifest, rows, rpc=rpc_count, source=source_count, deepseek=deepseek_count, run_status="STOPPED", reason_codes=("SOURCE_PROCESSING_ERROR",))
        source_count += int(source_summary.get("source_request_count", 0))
        if source_count > manifest.request_limits[1]:
            raise ValueError("SOURCE_REQUEST_LIMIT_EXCEEDED")
        if source_summary.get("stop_code"):
            checkpoint["status"] = "STOPPED"; save(); return build_report(manifest, rows, rpc=rpc_count, source=source_count, deepseek=deepseek_count, run_status="STOPPED", reason_codes=(str(source_summary["stop_code"]),))
        preparation = prepare_phase2_topk_v3(prepare_phase2_localization(bundle.strict, bundle.causal, bundle.graph, slices))
        bridge_codes = tuple(item.code for item in preparation.issues)
        if preparation.preparation_status != "READY_FOR_LLM":
            rows.append(_sanitized_sample(alias, bridge_status=preparation.preparation_status, stage_counts=bundle.stage_counts, source_summary=source_summary, bridge_reason_codes=bridge_codes, reason_codes=(preparation.preparation_status,))); save(); continue
        try:
            result = deepseek_runner(preparation); deepseek_count += 1
        except Exception:
            rows.append(_sanitized_sample(alias, bridge_status="READY_FOR_LLM", stage_counts=bundle.stage_counts, source_summary=source_summary, reason_codes=("DEEPSEEK_TRANSPORT_ERROR",))); checkpoint["status"] = "STOPPED"; save(); return build_report(manifest, rows, rpc=rpc_count, source=source_count, deepseek=deepseek_count, run_status="STOPPED", reason_codes=("DEEPSEEK_TRANSPORT_ERROR",))
        if not isinstance(result, Phase2DeepSeekResult):
            rows.append(_sanitized_sample(alias, bridge_status="READY_FOR_LLM", stage_counts=bundle.stage_counts, source_summary=source_summary, reason_codes=("DEEPSEEK_RESULT_INVALID",))); save(); continue
        top3 = map_validated_top3(preparation, result)
        evaluation = evaluate_top3_locations(top3, location_loader(tx_hash) if result.response_status in {"VALIDATED", "VALIDATED_WITH_NORMALIZATION"} and location_loader else None)
        modes = {item.get("resolution_mode") for item in preparation.source_locations.values() if isinstance(item, Mapping)}
        granularity = "SOURCE_WINDOW" if "VERIFIED_SOURCE_WINDOW_FALLBACK" in modes or preparation.source_resolution_mode == "VERIFIED_SOURCE_WINDOW_FALLBACK" else "EXACT_FUNCTION"
        if granularity == "SOURCE_WINDOW":
            evaluation = {"TOP1_HIT": "TOP1_WINDOW_HIT", "TOP3_HIT": "TOP3_WINDOW_HIT"}.get(evaluation, evaluation)
        rows.append(_sanitized_sample(alias, bridge_status="READY_FOR_LLM", stage_counts=bundle.stage_counts, source_summary=source_summary, result=result, top3=top3, evaluation=evaluation, evaluation_granularity=granularity, bridge_reason_codes=bridge_codes, reason_codes=tuple(result.reason_codes))); save()
    checkpoint["status"] = "COMPLETED"; save()
    return build_report(manifest, rows, rpc=rpc_count, source=source_count, deepseek=deepseek_count, run_status="COMPLETED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--live", action="store_true"); parser.add_argument("--only-sample", choices=("phase2_fault3_003",)); parser.add_argument("--output-dir", type=Path); args = parser.parse_args(argv)
    manifest = build_window_single_manifest() if args.only_sample else build_locked_manifest()
    output_dir = args.output_dir or (DEFAULT_WINDOW_OUTPUT if args.only_sample else DEFAULT_OUTPUT)
    if not args.live:
        print("live_required=true"); print("rpc_request_count=0"); print("source_request_count=0"); print("deepseek_request_count=0"); return 2
    _atomic(output_dir / "locked_manifest.json", manifest.to_dict())
    checkpoint_path, report_path = output_dir / "checkpoint.json", output_dir / ("phase2_window_single_report.json" if args.only_sample else "phase2_fault3_v2_report.json")
    _atomic(checkpoint_path, {"schema_version": "phase2_fault3_v2_checkpoint_v1", "manifest_sha256": manifest.manifest_sha256, "status": "LOCKED", "rpc_request_count": 0, "source_request_count": 0, "deepseek_request_count": 0, "samples": []})
    try:
        config = load_chain_config("eth"); rpc_url = config.get("rpc_url")
        if not isinstance(rpc_url, str) or not rpc_url: raise ValueError("ETH_RPC_CONFIG_NOT_LOADED")
        rpc = JsonRpcClient(rpc_url, max_requests=MAX_RPC)
        def source_builder(bundle: Phase2EvidenceBundle): return acquire_cached_source_slices(bundle) if args.only_sample else acquire_real_source_slices(bundle, config)
        def deepseek_runner(preparation: Any):
            key = os.getenv("DEEPSEEK_API_KEY")
            if not key: raise ValueError("DEEPSEEK_KEY_NOT_LOADED")
            return execute_phase2_localization(preparation, Phase2DeepSeekConfig(), DeepSeekHTTPSOneShotTransport(), key)
        report = run_fault3_v2_pilot(manifest, live=True, rpc_client=rpc, source_builder=source_builder, deepseek_runner=deepseek_runner, location_loader=_location_after_inference, checkpoint_callback=lambda value: _atomic(checkpoint_path, value))
        _atomic(report_path, report)
    except Exception as exc:
        print("pilot_status=STOPPED"); print("reason_code=" + _code(exc, "PHASE2_FAULT3_SETUP_ERROR")); return 1
    print("pilot_status=" + report["run_status"]); print("report_path=" + str(report_path)); print("rpc_request_count=" + str(report["external_call_counts"]["rpc"])); print("source_request_count=" + str(report["external_call_counts"]["source"])); print("deepseek_request_count=" + str(report["external_call_counts"]["deepseek"])); return 0 if report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

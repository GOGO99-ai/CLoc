"""One-shot, auditable real pilot for the frozen phase-2 Top-K v3 contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase1_deepseek_smoke import DeepSeekHTTPSOneShotTransport
from scripts.run_phase2_fault3_v2_pilot import _atomic, _code, _location_after_inference
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

KIND = "PHASE2_TOPK_V3_SINGLE_REAL_PILOT"
SAMPLE_ALIAS = "phase2_topk_v3_fault_001"
TX_HASH = "0xd721b0ef2886f14b75548b70d2d1fd82bea085ca24f5de29b833a64cfd8f7a50"
DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "phase2_topk_v3_fault001_20260822"
MAX_RPC = 2
MAX_SOURCE = 3
MAX_DEEPSEEK = 1
_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class LockedTopKV3Manifest:
    sample_alias: str = SAMPLE_ALIAS
    tx_hash: str = TX_HASH
    blockchain_chain_id: int = 1

    def __post_init__(self) -> None:
        if self.sample_alias != SAMPLE_ALIAS or self.blockchain_chain_id != 1 or _HASH.fullmatch(self.tx_hash.lower()) is None:
            raise ValueError("TOPK_V3_MANIFEST_INVALID")
        if self.tx_hash.lower() in {
            "0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9",
            "0x0163856aaa678676b22e072186c1bdc55ed44a5a0e2cf44500362800663ddce9",
            "0xa9a1b8ea288eb9ad315088f17f7c7386b9989c95b4d13c81b69d5ddad7ffe61e",
            "0x38ecc8363836e43aca99d994bb9325a8f3770d7a1d1fffd62c82cf5230525836",
        }:
            raise ValueError("TOPK_V3_HISTORICAL_SAMPLE_FORBIDDEN")

    @property
    def manifest_sha256(self) -> str:
        value = {"schema_version": "phase2_topk_v3_single_manifest_v1", "experiment_kind": KIND, "sample_alias": self.sample_alias, "blockchain_chain_id": self.blockchain_chain_id, "tx_hash": self.tx_hash.lower()}
        return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "phase2_topk_v3_single_manifest_v1", "experiment_kind": KIND, "scientific_experiment_result": False, "sample_alias": self.sample_alias, "blockchain_chain_id": self.blockchain_chain_id, "tx_hash": self.tx_hash.lower(), "manifest_sha256": self.manifest_sha256}


def build_locked_manifest() -> LockedTopKV3Manifest:
    return LockedTopKV3Manifest()


def _summary(source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = {"source_request_count": 0, "attempted_address_count": 0, "verified_source_count": 0, "cache_hit_count": 0, "resolved_slice_count": 0, "window_count": 0, "window_file_count": 0, "source_resolution_mode": "NONE", "stop_code": None}
    base.update(dict(source or {})); return base


def _granularity(preparation: Any) -> str:
    locations = getattr(preparation, "source_locations", {})
    modes = {item.get("resolution_mode") for item in locations.values() if isinstance(item, Mapping)}
    return "SOURCE_WINDOW" if "VERIFIED_SOURCE_WINDOW_FALLBACK" in modes or getattr(preparation, "source_resolution_mode", None) == "VERIFIED_SOURCE_WINDOW_FALLBACK" else "EXACT_FUNCTION"


def _sanitized_report(manifest: LockedTopKV3Manifest, *, run_status: str, stage_counts: Mapping[str, Any] | None = None, source_summary: Mapping[str, Any] | None = None, preparation: Any = None, result: Phase2DeepSeekResult | None = None, top3: list[Mapping[str, Any]] | None = None, evaluation_result: str = "EVALUATION_UNAVAILABLE", evaluation_granularity: str | None = None, reason_codes: tuple[str, ...] = (), rpc: int = 0, source: int = 0, deepseek: int = 0) -> dict[str, Any]:
    response = result.validated_response if result and result.validated_response else {}
    report = {"schema_version": "phase2_topk_v3_single_report_v1", "experiment_kind": KIND, "scientific_experiment_result": False, "sample_alias": manifest.sample_alias, "locked_manifest_sha256": manifest.manifest_sha256, "run_status": run_status, "stage_counts": dict(stage_counts or {}), "bridge_status": preparation.preparation_status if preparation else "NOT_PREPARED", "bridge_reason_codes": sorted(item.code for item in preparation.issues) if preparation else [], "source_summary": _summary(source_summary), "prompt_summary": {"prompt_sha256": preparation.prompt_sha256 if preparation else None, "prompt_bytes": preparation.prompt_bytes if preparation else 0, "q_count": len(preparation.prompt.citation_whitelist.chain_aliases) if preparation and preparation.prompt else 0, "e_count": len(preparation.prompt.citation_whitelist.evidence_ids) if preparation and preparation.prompt else 0, "s_count": len(preparation.prompt.citation_whitelist.source_slice_ids) if preparation and preparation.prompt else 0}, "llm": {"called": result is not None, "response_status": result.response_status if result else "NOT_CALLED", "valid": result is not None and result.response_status in {"VALIDATED", "VALIDATED_WITH_NORMALIZATION"}, "status": response.get("status"), "provider_free_text_used": response.get("provider_free_text_used", False), "explanation_source": response.get("explanation_source"), "reason_codes": list(result.reason_codes) if result else []}, "top3": [dict(item) for item in (top3 or [])], "evaluation_result": evaluation_result, "evaluation_granularity": evaluation_granularity, "external_call_counts": {"rpc": rpc, "source": source, "deepseek": deepseek}, "request_limits": {"rpc": MAX_RPC, "source": MAX_SOURCE, "deepseek": MAX_DEEPSEEK}, "retry_count": 0, "endpoint_switch_count": 0, "sample_switch_count": 0, "reason_codes": sorted(set(reason_codes))}
    validate_report(report); return report


def validate_report(report: Mapping[str, Any]) -> None:
    text = _canonical(report)
    if _RAW_ADDRESS.search(text) or _RAW_HASH.search(text) or any(value in text.lower() for value in ("reasoning_content", "rpc_url", "api_key", "provider_response", "system_message", "user_message", '"cause"')):
        raise ValueError("TOPK_V3_REPORT_SENSITIVE_CONTENT")
    if any(report["external_call_counts"][key] > report["request_limits"][key] for key in ("rpc", "source", "deepseek")):
        raise ValueError("TOPK_V3_REQUEST_LIMIT_EXCEEDED")


def run_topk_v3_single_pilot(manifest: LockedTopKV3Manifest, *, live: bool, rpc_client: Any | None = None, evidence_builder: Callable[[Any], Phase2EvidenceBundle] = build_real_phase2_evidence, source_builder: Callable[[Phase2EvidenceBundle], tuple[Mapping[str, Any], Mapping[str, Any]]] | None = None, deepseek_runner: Callable[[Any], Phase2DeepSeekResult] | None = None, location_loader: Callable[[str], Any] | None = None, checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    if not live: return _sanitized_report(manifest, run_status="LIVE_MODE_REQUIRED")
    if rpc_client is None or source_builder is None or deepseek_runner is None: raise ValueError("TOPK_V3_LIVE_DEPENDENCY_MISSING")
    rpc = source = deepseek = 0
    checkpoint = {"schema_version": "phase2_topk_v3_single_checkpoint_v1", "manifest_sha256": manifest.manifest_sha256, "status": "RUNNING", "rpc_request_count": 0, "source_request_count": 0, "deepseek_request_count": 0, "reason_codes": []}
    def save() -> None:
        checkpoint.update(rpc_request_count=rpc, source_request_count=source, deepseek_request_count=deepseek)
        if checkpoint_callback: checkpoint_callback(json.loads(_canonical(checkpoint)))
    try:
        rpc += 1; chain = rpc_client.call("eth_chainId", [])
    except Exception as exc:
        checkpoint.update(status="STOPPED", reason_codes=[_code(exc, "RPC_CHAIN_ID_ERROR")]); save(); return _sanitized_report(manifest, run_status="STOPPED", reason_codes=tuple(checkpoint["reason_codes"]), rpc=rpc)
    if str(chain).lower() not in {"0x1", "1"}:
        checkpoint.update(status="STOPPED", reason_codes=["CHAIN_ID_MISMATCH"]); save(); return _sanitized_report(manifest, run_status="STOPPED", reason_codes=("CHAIN_ID_MISMATCH",), rpc=rpc)
    try:
        rpc += 1; raw = rpc_client.call("debug_traceTransaction", [manifest.tx_hash, {"tracer": "callTracer"}])
    except Exception as exc:
        checkpoint.update(status="STOPPED", reason_codes=[_code(exc, "RPC_TRACE_ERROR")]); save(); return _sanitized_report(manifest, run_status="STOPPED", reason_codes=tuple(checkpoint["reason_codes"]), rpc=rpc)
    try:
        bundle = evidence_builder(raw)
    except Exception:
        checkpoint.update(status="STOPPED", reason_codes=["EVIDENCE_PROCESSING_ERROR"]); save(); return _sanitized_report(manifest, run_status="STOPPED", reason_codes=("EVIDENCE_PROCESSING_ERROR",), rpc=rpc)
    try:
        slices, source_summary = source_builder(bundle); source_summary = _summary(source_summary); source = int(source_summary["source_request_count"])
    except Exception:
        checkpoint.update(status="STOPPED", reason_codes=["SOURCE_PROCESSING_ERROR"]); save(); return _sanitized_report(manifest, run_status="STOPPED", stage_counts=bundle.stage_counts, reason_codes=("SOURCE_PROCESSING_ERROR",), rpc=rpc)
    if source > MAX_SOURCE or source_summary.get("stop_code"):
        code = str(source_summary.get("stop_code") or "SOURCE_REQUEST_LIMIT_EXCEEDED"); checkpoint.update(status="STOPPED", reason_codes=[code]); save(); return _sanitized_report(manifest, run_status="STOPPED", stage_counts=bundle.stage_counts, source_summary=source_summary, reason_codes=(code,), rpc=rpc, source=source)
    preparation = prepare_phase2_topk_v3(prepare_phase2_localization(bundle.strict, bundle.causal, bundle.graph, slices))
    if preparation.preparation_status != "READY_FOR_LLM":
        checkpoint.update(status="COMPLETED_SAFE_STOP", reason_codes=[preparation.preparation_status]); save(); return _sanitized_report(manifest, run_status="COMPLETED_SAFE_STOP", stage_counts=bundle.stage_counts, source_summary=source_summary, preparation=preparation, reason_codes=(preparation.preparation_status,), rpc=rpc, source=source)
    try:
        result = deepseek_runner(preparation); deepseek = 1
    except Exception:
        checkpoint.update(status="STOPPED", reason_codes=["DEEPSEEK_TRANSPORT_ERROR"]); save(); return _sanitized_report(manifest, run_status="STOPPED", stage_counts=bundle.stage_counts, source_summary=source_summary, preparation=preparation, reason_codes=("DEEPSEEK_TRANSPORT_ERROR",), rpc=rpc, source=source, deepseek=deepseek)
    if not isinstance(result, Phase2DeepSeekResult):
        checkpoint.update(status="COMPLETED_SAFE_STOP", reason_codes=["DEEPSEEK_RESULT_INVALID"]); save(); return _sanitized_report(manifest, run_status="COMPLETED_SAFE_STOP", stage_counts=bundle.stage_counts, source_summary=source_summary, preparation=preparation, reason_codes=("DEEPSEEK_RESULT_INVALID",), rpc=rpc, source=source, deepseek=deepseek)
    top3 = map_validated_top3(preparation, result)
    granularity = _granularity(preparation)
    evaluation = evaluate_top3_locations(top3, location_loader(manifest.tx_hash) if result.response_status in {"VALIDATED", "VALIDATED_WITH_NORMALIZATION"} and location_loader else None)
    if granularity == "SOURCE_WINDOW": evaluation = {"TOP1_HIT": "TOP1_WINDOW_HIT", "TOP3_HIT": "TOP3_WINDOW_HIT"}.get(evaluation, evaluation)
    checkpoint.update(status="COMPLETED", reason_codes=list(result.reason_codes)); save()
    return _sanitized_report(manifest, run_status="COMPLETED", stage_counts=bundle.stage_counts, source_summary=source_summary, preparation=preparation, result=result, top3=top3, evaluation_result=evaluation, evaluation_granularity=granularity, reason_codes=tuple(result.reason_codes), rpc=rpc, source=source, deepseek=deepseek)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--live", action="store_true"); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(argv)
    manifest = build_locked_manifest()
    if not args.live:
        print("live_required=true"); print("rpc_request_count=0"); print("source_request_count=0"); print("deepseek_request_count=0"); return 2
    _atomic(args.output_dir / "locked_manifest.json", manifest.to_dict())
    checkpoint_path = args.output_dir / "checkpoint.json"; report_path = args.output_dir / "phase2_topk_v3_report.json"
    _atomic(checkpoint_path, {"schema_version": "phase2_topk_v3_single_checkpoint_v1", "manifest_sha256": manifest.manifest_sha256, "status": "LOCKED", "rpc_request_count": 0, "source_request_count": 0, "deepseek_request_count": 0, "reason_codes": []})
    try:
        config = load_chain_config("eth"); rpc_url = config.get("rpc_url")
        if not isinstance(rpc_url, str) or not rpc_url: raise ValueError("ETH_RPC_CONFIG_NOT_LOADED")
        rpc_client = JsonRpcClient(rpc_url, max_requests=MAX_RPC)
        def source_builder(bundle: Phase2EvidenceBundle): return acquire_real_source_slices(bundle, config)
        def deepseek_runner(preparation: Any):
            key = os.getenv("DEEPSEEK_API_KEY")
            if not key: raise ValueError("DEEPSEEK_KEY_NOT_LOADED")
            return execute_phase2_localization(preparation, Phase2DeepSeekConfig(), DeepSeekHTTPSOneShotTransport(), key)
        report = run_topk_v3_single_pilot(manifest, live=True, rpc_client=rpc_client, source_builder=source_builder, deepseek_runner=deepseek_runner, location_loader=_location_after_inference, checkpoint_callback=lambda value: _atomic(checkpoint_path, value))
        _atomic(report_path, report)
    except Exception as exc:
        print("pilot_status=STOPPED"); print("reason_code=" + _code(exc, "TOPK_V3_SETUP_ERROR")); return 1
    print("pilot_status=" + report["run_status"]); print("report_path=" + str(report_path)); print("rpc_request_count=" + str(report["external_call_counts"]["rpc"])); print("source_request_count=" + str(report["external_call_counts"]["source"])); print("deepseek_request_count=" + str(report["external_call_counts"]["deepseek"])); return 0 if report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

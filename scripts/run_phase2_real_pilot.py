"""One-sample phase-2 source-localization pilot for locked Ethereum fault 007."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cloc.runtime.execution_policy import ExecutionPolicy
from scripts.run_phase1_deepseek_smoke import DeepSeekHTTPSOneShotTransport
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.asset_continuity import build_asset_continuity
from src.cloc_v2.calltracer_adapter import adapt_calltracer_call_tree
from src.cloc_v2.candidate_causal_graph import build_candidate_causal_graph
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.phase2_localization_bridge import Phase2LocalizationPreparation, prepare_phase2_localization
from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekConfig, Phase2DeepSeekResult, execute_phase2_localization
from src.cloc_v2.phase2_localization_prompt import upgrade_phase2_localization_prompt_to_decision_v2, upgrade_phase2_localization_prompt_to_topk_v3
from src.cloc_v2.result_detection import detect_result_candidates
from src.cloc_v2.result_models import ResultDetectionConfig
from src.cloc_v2.runtime_semantics import build_runtime_annotations
from src.cloc_v2.source_fetcher import SourceFetcher
from src.cloc_v2.source_index import build_source_index
from src.cloc_v2.source_slices import build_source_request_slices
from src.cloc_v2.source_window_fallback import build_verified_source_window_fallback
from src.cloc_v2.phase2_source_graph_fallback import build_phase2_source_graph_fallback
from src.cloc_v2.strict_result_chains import build_strict_result_chains
from src.cloc_v2.transaction_graph import build_transaction_graph
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index, lookup_verified_source_cache

KIND = "ONE_SAMPLE_PHASE2_LOCALIZATION_ENGINEERING_PILOT"
TX_HASH = "0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9"
SAMPLE_ALIAS = "phase2_fault_007"
DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "phase2_real_pilot_fault007_retry2_20260822"
VALID_DEEPSEEK_RESPONSE_STATUSES = frozenset({"VALIDATED", "VALIDATED_WITH_NORMALIZATION"})
SOURCE_CACHE = ROOT / "data" / "source_cache"
MAX_RPC = 2
MAX_SOURCE_REQUESTS = 3
MAX_DEEPSEEK = 1
_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")


def _canonical(value: Any) -> str: return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def _sha(value: Any) -> str: return hashlib.sha256(_canonical(value).encode()).hexdigest()
def _code(exc: Exception, fallback: str) -> str:
    return exc.args[0] if len(exc.args) == 1 and isinstance(exc.args[0], str) and _CODE.fullmatch(exc.args[0]) else fallback
def _chain_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool): return value
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]+", value): return int(value, 16)
    return None


@dataclass(frozen=True)
class LockedPhase2Manifest:
    sample_alias: str = SAMPLE_ALIAS
    blockchain_chain_id: int = 1
    tx_hash: str = field(default=TX_HASH, repr=False)
    manifest_sha256: str = ""
    def __post_init__(self) -> None:
        payload = {"schema_version": "phase2_real_pilot_manifest_v1", "experiment_kind": KIND, "sample_alias": self.sample_alias, "blockchain_chain_id": self.blockchain_chain_id, "tx_hash": self.tx_hash}
        expected = _sha(payload)
        if self.sample_alias != SAMPLE_ALIAS or self.blockchain_chain_id != 1 or self.tx_hash != TX_HASH: raise ValueError("PHASE2_LOCKED_SAMPLE_INVALID")
        if self.manifest_sha256 and self.manifest_sha256 != expected: raise ValueError("PHASE2_MANIFEST_SHA_INVALID")
        object.__setattr__(self, "manifest_sha256", expected)
    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "phase2_real_pilot_manifest_v1", "experiment_kind": KIND, "scientific_experiment_result": False, "sample_alias": self.sample_alias, "blockchain_chain_id": self.blockchain_chain_id, "tx_hash": self.tx_hash, "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class Phase2EvidenceBundle:
    graph: Any = field(repr=False)
    strict: Any = field(repr=False)
    causal: Mapping[str, Any] = field(repr=False)
    stage_counts: Mapping[str, int]
    candidate_nodes: tuple[str, ...] = field(repr=False)
    candidate_addresses: tuple[str, ...] = field(repr=False)


def build_locked_manifest() -> LockedPhase2Manifest: return LockedPhase2Manifest()


def prepare_phase2_decision_v2(preparation: Phase2LocalizationPreparation) -> Phase2LocalizationPreparation:
    """Keep bridge evidence unchanged while requesting only v2 machine decisions."""
    if not isinstance(preparation, Phase2LocalizationPreparation) or preparation.preparation_status != "READY_FOR_LLM" or preparation.prompt is None:
        return preparation
    prompt = upgrade_phase2_localization_prompt_to_decision_v2(preparation.prompt)
    encoded = _canonical(prompt.to_dict()).encode("utf-8")
    return replace(preparation, prompt=prompt, prompt_sha256=hashlib.sha256(encoded).hexdigest(), prompt_bytes=len(encoded))


def prepare_phase2_topk_v3(preparation: Phase2LocalizationPreparation) -> Phase2LocalizationPreparation:
    """Keep bridge evidence unchanged while requesting only v3 rank/slice/evidence fields."""
    if not isinstance(preparation, Phase2LocalizationPreparation) or preparation.preparation_status != "READY_FOR_LLM" or preparation.prompt is None:
        return preparation
    prompt = upgrade_phase2_localization_prompt_to_topk_v3(preparation.prompt)
    encoded = _canonical(prompt.to_dict()).encode("utf-8")
    return replace(preparation, prompt=prompt, prompt_sha256=hashlib.sha256(encoded).hexdigest(), prompt_bytes=len(encoded))


def build_real_phase2_evidence(raw_frame: Any, *, result_config: ResultDetectionConfig | None = None) -> Phase2EvidenceBundle:
    call_tree = adapt_calltracer_call_tree(raw_frame); graph = build_transaction_graph(call_tree); annotations = build_runtime_annotations(graph); ledger = build_observed_asset_ledger(graph, annotations); continuity = build_asset_continuity(graph, annotations, ledger); detected = detect_result_candidates(graph, annotations, ledger, continuity.to_dict(), result_config)
    strict = build_strict_result_chains(graph, continuity.to_dict(), detected.to_dict(), blockchain_chain_id=1, offline_function_candidates=(), offline_effective_address_map={})
    causal = build_candidate_causal_graph(graph, annotations, ledger, continuity.to_dict(), detected.to_dict()).to_dict()
    node_by_id = {node.node_id: node for node in graph.nodes}; anchor_nodes = [chain.anchor.node_id for chain in strict.chains if chain.anchor is not None]; chain_nodes = [node for chain in strict.chains for node in chain.node_ids]
    ordered_nodes = tuple(dict.fromkeys(anchor_nodes + chain_nodes))
    addresses = []
    for node_id in ordered_nodes:
        node = node_by_id.get(node_id); address = node.callee_address.lower() if node and isinstance(node.callee_address, str) else None
        if address and address not in addresses: addresses.append(address)
        if len(addresses) == MAX_SOURCE_REQUESTS: break
    counts = {"call_node_count": len(graph.nodes), "result_candidate_count": len(detected.result_candidates), "strict_chain_count": len(strict.chains), "strict_ready_count": sum(item.status == "READY" for item in strict.chains), "strict_uncertain_count": sum(item.status == "UNCERTAIN" for item in strict.chains), "causal_edge_count": len(causal.get("edges", ())) }
    return Phase2EvidenceBundle(graph, strict, causal, counts, ordered_nodes[:12], tuple(addresses))


class CountingSourceSession:
    def __init__(self, delegate: Any, limit: int = MAX_SOURCE_REQUESTS): self.delegate=delegate; self.limit=limit; self.request_count=0
    def get(self, url, params, timeout):
        if self.request_count >= self.limit: raise OSError("SOURCE_REQUEST_LIMIT_REACHED")
        self.request_count += 1
        return self.delegate.get(url, params, timeout)


class Phase2SourceStageError(Exception):
    def __init__(self, request_count: int):
        super().__init__("SOURCE_PROCESSING_ERROR")
        self.request_count = request_count


def acquire_real_source_slices(bundle: Phase2EvidenceBundle, chain_config: Mapping[str, Any], *, cache_root: Path = SOURCE_CACHE) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Resolve verified local cache bindings first; fetch only explicit cache misses."""
    policy = ExecutionPolicy(allow_explorer=True, allow_artifact_cache_write=True)
    fetcher = SourceFetcher("eth", dict(chain_config), cache_root=cache_root, policy=policy, max_retries=1, max_proxy_depth=3)
    counter = CountingSourceSession(fetcher.session); fetcher.session = counter
    cache_index = build_verified_source_cache_index(cache_root, blockchain_chain_id=1, chain_key="eth")
    acquisition_addresses = []
    cache_statuses: dict[str, int] = {}
    cache_hits = 0
    stop_code = None
    for address in bundle.candidate_addresses:
        lookup = lookup_verified_source_cache(cache_index, runtime_address=address, requested_address=address)
        cache_statuses[lookup.status] = cache_statuses.get(lookup.status, 0) + 1
        if lookup.status == "VERIFIED_SOURCE_CACHE_HIT" and lookup.binding is not None:
            acquisition_addresses.append({"requested_address": lookup.binding.artifact_address})
            cache_hits += 1
            continue
        if lookup.status in {"VERIFIED_SOURCE_CACHE_CONFLICT", "VERIFIED_SOURCE_CACHE_INVALID"}:
            stop_code = lookup.status
            break
        result = fetcher.acquire(address, refresh=False, origins=())
        acquisition_addresses.append(result.to_dict())
        if result.status in {"network_error", "timeout", "rate_limited", "api_error", "parse_error"}:
            stop_code = "SOURCE_FETCH_EXTERNAL_ERROR"; break
    acquisition = {"schema_version": "source_acquisition_v1", "chain": "eth", "addresses": acquisition_addresses}
    try:
        index = build_source_index(acquisition, cache_root)
        requests = [{"chain_id": "Q", "node_id": node_id, "focus": "phase2_localization", "priority": 1 if position == 0 else 2} for position, node_id in enumerate(bundle.candidate_nodes)]
        slices = build_source_request_slices(index, bundle.graph.to_dict(), cache_root, requests)
        slices = build_verified_source_window_fallback(index, bundle.graph.to_dict(), slices, cache_root)
        strict = getattr(bundle, "strict", None)
        if strict is not None:
            slices = build_phase2_source_graph_fallback(bundle.graph, strict, slices, cache_index, cache_root)
    except Exception as exc:
        raise Phase2SourceStageError(counter.request_count) from exc
    metadata = slices.get("metadata", {})
    summary = {"source_request_count": counter.request_count, "attempted_address_count": len(acquisition_addresses), "verified_source_count": sum(item.get("status") == "verified" for item in index.get("addresses", ())), "cache_hit_count": cache_hits, "cache_index_status_counts": dict(sorted(cache_statuses.items())), "resolved_slice_count": metadata.get("resolved_count", 0), "window_count": metadata.get("window_count", 0), "window_file_count": metadata.get("window_file_count", 0), "source_resolution_mode": metadata.get("resolution_mode", "EXACT_FUNCTION_SLICE"), "source_selection_mode": metadata.get("source_selection_mode", "DIRECT_EVIDENCE_SOURCE"), "stop_code": stop_code}
    return slices, summary


def map_validated_top3(preparation: Phase2LocalizationPreparation, result: Phase2DeepSeekResult) -> list[dict[str, Any]]:
    if result.response_status not in VALID_DEEPSEEK_RESPONSE_STATUSES or result.validated_response is None: return []
    locations = preparation.source_locations; rows = []
    for finding in result.validated_response.get("findings", ()):
        location = locations.get(finding["source_slice_id"])
        if location is None: raise ValueError("PHASE2_LOCAL_MAPPING_MISSING")
        rows.append({"rank": finding["rank"], "relative_file": location["relative_file"], "function": location["function"], "start_line": location["start_line"], "end_line": location["end_line"], "role": finding.get("role"), "confidence": finding.get("confidence"), "evidence_ids": list(finding["evidence_ids"]), "explanation": finding.get("explanation"), "explanation_source": finding.get("explanation_source", "MODEL_GENERATED_V1")})
    return rows


def evaluate_top3_locations(findings: list[Mapping[str, Any]], location: Any) -> str:
    """Fixed post-inference evaluator; ambiguous labels are unavailable, never MISS."""
    file_name = function = None; line = None
    if isinstance(location, Mapping):
        file_name = location.get("file") or location.get("file_path")
        function = location.get("function") or location.get("function_name")
        line = location.get("line") or location.get("line_number")
    elif isinstance(location, str):
        file_match = re.search(r"([A-Za-z0-9_.\-/]+\.sol)", location); line_match = re.search(r"(?:line|:|#)\s*(\d+)", location, re.I); function_match = re.search(r"(?:function|func)\s*[:=]?\s*([A-Za-z_$][A-Za-z0-9_$]*)", location, re.I)
        file_name = file_match.group(1) if file_match else None; line = int(line_match.group(1)) if line_match else None; function = function_match.group(1) if function_match else None
    if not ((isinstance(file_name, str) and isinstance(line, int)) or isinstance(function, str)):
        return "EVALUATION_UNAVAILABLE"
    hits = []
    for row in findings:
        file_hit = isinstance(file_name, str) and isinstance(line, int) and str(row.get("relative_file", "")).replace("\\", "/").endswith(file_name.replace("\\", "/")) and row.get("start_line") <= line <= row.get("end_line")
        row_function = str(row.get("function") or "").split("(", 1)[0].strip().lower(); function_hit = isinstance(function, str) and row_function == function.split("(", 1)[0].strip().lower()
        hits.append(file_hit or function_hit)
    if hits and hits[0]: return "TOP1_HIT"
    if any(hits[:3]): return "TOP3_HIT"
    return "MISS"


def build_phase2_report(manifest: LockedPhase2Manifest, *, run_status: str, stage_counts: Mapping[str, Any] | None = None, preparation: Phase2LocalizationPreparation | None = None, source_summary: Mapping[str, Any] | None = None, deepseek_result: Phase2DeepSeekResult | None = None, top3: list[Mapping[str, Any]] | None = None, evaluation_result: str = "EVALUATION_UNAVAILABLE", reason_codes: tuple[str, ...] = (), rpc_count: int = 0) -> dict[str, Any]:
    source_summary = dict(source_summary or {"source_request_count": 0, "attempted_address_count": 0, "verified_source_count": 0, "cache_hit_count": 0, "resolved_slice_count": 0, "stop_code": None})
    counts = dict(preparation.evidence_type_counts) if preparation else {kind: 0 for kind in ("ASSET", "CALL", "ORDER", "PARAM", "STATE")}
    prompt = preparation.prompt if preparation else None
    materialized = deepseek_result.validated_response if deepseek_result and deepseek_result.validated_response else {}
    report = {"schema_version": "phase2_real_pilot_report_v1", "experiment_kind": KIND, "scientific_experiment_result": False, "sample_alias": manifest.sample_alias, "locked_manifest_sha256": manifest.manifest_sha256, "run_status": run_status, "stage_counts": dict(stage_counts or {}), "bridge_status": preparation.preparation_status if preparation else "NOT_PREPARED", "bridge_issues": [item.to_dict() for item in preparation.issues] if preparation else [], "evidence_type_counts": counts, "source_summary": source_summary, "prompt_summary": {"prompt_sha256": preparation.prompt_sha256 if preparation else None, "prompt_bytes": preparation.prompt_bytes if preparation else 0, "q_count": len(prompt.citation_whitelist.chain_aliases) if prompt else 0, "e_count": len(prompt.citation_whitelist.evidence_ids) if prompt else 0, "s_count": len(prompt.citation_whitelist.source_slice_ids) if prompt else 0}, "deepseek": {"called": deepseek_result is not None, "response_status": deepseek_result.response_status if deepseek_result else "NOT_CALLED", "valid": deepseek_result.response_status in VALID_DEEPSEEK_RESPONSE_STATUSES if deepseek_result else False, "provider_free_text_used": materialized.get("provider_free_text_used", True), "explanation_source": materialized.get("explanation_source", "MODEL_GENERATED_V1"), "normalization_applied": deepseek_result.normalization_applied if deepseek_result else False, "normalization_actions": list(deepseek_result.normalization_actions) if deepseek_result else [], "usage": deepseek_result.usage.to_dict() if deepseek_result and deepseek_result.usage else None, "reason_codes": list(deepseek_result.reason_codes) if deepseek_result else []}, "top3": [dict(item) for item in (top3 or [])], "evaluation_result": evaluation_result, "reason_codes": sorted(set(reason_codes)), "external_call_counts": {"rpc": rpc_count, "source": source_summary.get("source_request_count", 0), "deepseek": 1 if deepseek_result is not None else 0}, "request_limits": {"rpc": MAX_RPC, "source": MAX_SOURCE_REQUESTS, "deepseek": MAX_DEEPSEEK}, "retry_count": 0, "endpoint_switch_count": 0, "sample_switch_count": 0}
    validate_phase2_report(report); return report


def validate_phase2_report(report: Mapping[str, Any]) -> None:
    text = _canonical(report)
    if _RAW_HASH.search(text) or _RAW_ADDRESS.search(text) or any(word in text.lower() for word in ("reasoning_content", "rpc_url", "api_key", "provider_response", "system_message", "user_message")): raise ValueError("PHASE2_REPORT_SENSITIVE_CONTENT")
    calls = report["external_call_counts"]; limits = report["request_limits"]
    if any(calls[key] > limits[key] for key in calls): raise ValueError("PHASE2_REPORT_REQUEST_LIMIT_EXCEEDED")


def run_phase2_pilot(manifest: LockedPhase2Manifest, *, live: bool, rpc_client: Any | None = None, evidence_builder: Callable[[Any], Phase2EvidenceBundle] = build_real_phase2_evidence, source_builder: Callable[[Phase2EvidenceBundle], tuple[Mapping[str, Any], Mapping[str, Any]]] | None = None, deepseek_runner: Callable[[Phase2LocalizationPreparation], Phase2DeepSeekResult] | None = None, location_loader: Callable[[], Any] | None = None, checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    if not live: return build_phase2_report(manifest, run_status="LIVE_MODE_REQUIRED")
    if rpc_client is None or source_builder is None or deepseek_runner is None: raise ValueError("PHASE2_LIVE_DEPENDENCY_MISSING")
    rpc_count = 0; checkpoint = {"schema_version": "phase2_real_pilot_checkpoint_v1", "manifest_sha256": manifest.manifest_sha256, "status": "RUNNING", "rpc_request_count": 0, "source_request_count": 0, "deepseek_request_count": 0, "reason_codes": []}
    def save():
        if checkpoint_callback: checkpoint_callback(json.loads(_canonical(checkpoint)))
    try: rpc_count += 1; chain = rpc_client.call("eth_chainId", [])
    except Exception as exc:
        checkpoint.update(status="STOPPED", rpc_request_count=rpc_count, reason_codes=[_code(exc, "RPC_CHAIN_ID_ERROR")]); save(); return build_phase2_report(manifest, run_status="STOPPED", reason_codes=tuple(checkpoint["reason_codes"]), rpc_count=rpc_count)
    if _chain_id(chain) != 1:
        checkpoint.update(status="STOPPED", rpc_request_count=rpc_count, reason_codes=["CHAIN_ID_MISMATCH"]); save(); return build_phase2_report(manifest, run_status="STOPPED", reason_codes=("CHAIN_ID_MISMATCH",), rpc_count=rpc_count)
    try: rpc_count += 1; raw = rpc_client.call("debug_traceTransaction", [manifest.tx_hash, {"tracer": "callTracer"}])
    except Exception as exc:
        checkpoint.update(status="STOPPED", rpc_request_count=rpc_count, reason_codes=[_code(exc, "RPC_TRACE_ERROR")]); save(); return build_phase2_report(manifest, run_status="STOPPED", reason_codes=tuple(checkpoint["reason_codes"]), rpc_count=rpc_count)
    try: bundle = evidence_builder(raw)
    except Exception:
        checkpoint.update(status="STOPPED", rpc_request_count=rpc_count, reason_codes=["EVIDENCE_PROCESSING_ERROR"]); save(); return build_phase2_report(manifest, run_status="STOPPED", reason_codes=("EVIDENCE_PROCESSING_ERROR",), rpc_count=rpc_count)
    try: slices, source_summary = source_builder(bundle)
    except Exception as exc:
        source_count = exc.request_count if isinstance(exc, Phase2SourceStageError) else 0
        source_summary = {"source_request_count": source_count, "attempted_address_count": 0, "verified_source_count": 0, "cache_hit_count": 0, "resolved_slice_count": 0, "stop_code": "SOURCE_PROCESSING_ERROR"}
        checkpoint.update(status="STOPPED", rpc_request_count=rpc_count, source_request_count=source_count, reason_codes=["SOURCE_PROCESSING_ERROR"]); save(); return build_phase2_report(manifest, run_status="STOPPED", stage_counts=bundle.stage_counts, source_summary=source_summary, reason_codes=("SOURCE_PROCESSING_ERROR",), rpc_count=rpc_count)
    checkpoint.update(rpc_request_count=rpc_count, source_request_count=source_summary.get("source_request_count", 0)); save()
    if source_summary.get("stop_code"):
        code = source_summary["stop_code"]; checkpoint.update(status="STOPPED", reason_codes=[code]); save(); return build_phase2_report(manifest, run_status="STOPPED", stage_counts=bundle.stage_counts, source_summary=source_summary, reason_codes=(code,), rpc_count=rpc_count)
    preparation = prepare_phase2_topk_v3(prepare_phase2_localization(bundle.strict, bundle.causal, bundle.graph, slices))
    if preparation.preparation_status != "READY_FOR_LLM":
        checkpoint.update(status="COMPLETED_SAFE_STOP", reason_codes=[preparation.preparation_status]); save(); return build_phase2_report(manifest, run_status="COMPLETED_SAFE_STOP", stage_counts=bundle.stage_counts, preparation=preparation, source_summary=source_summary, reason_codes=(preparation.preparation_status,), rpc_count=rpc_count)
    try:
        result = deepseek_runner(preparation)
    except Exception:
        checkpoint.update(status="COMPLETED_SAFE_STOP", deepseek_request_count=0, reason_codes=["DEEPSEEK_LOCAL_SETUP_ERROR"]); save()
        return build_phase2_report(manifest, run_status="COMPLETED_SAFE_STOP", stage_counts=bundle.stage_counts, preparation=preparation, source_summary=source_summary, reason_codes=("DEEPSEEK_LOCAL_SETUP_ERROR",), rpc_count=rpc_count)
    if not isinstance(result, Phase2DeepSeekResult):
        checkpoint.update(status="COMPLETED_SAFE_STOP", deepseek_request_count=0, reason_codes=["DEEPSEEK_RESULT_INVALID"]); save()
        return build_phase2_report(manifest, run_status="COMPLETED_SAFE_STOP", stage_counts=bundle.stage_counts, preparation=preparation, source_summary=source_summary, reason_codes=("DEEPSEEK_RESULT_INVALID",), rpc_count=rpc_count)
    checkpoint["deepseek_request_count"] = 1; checkpoint["status"] = "COMPLETED" if result.response_status in VALID_DEEPSEEK_RESPONSE_STATUSES else "COMPLETED_SAFE_STOP"; checkpoint["reason_codes"] = list(result.reason_codes); save()
    top3 = map_validated_top3(preparation, result)
    evaluation = evaluate_top3_locations(top3, location_loader()) if result.response_status in VALID_DEEPSEEK_RESPONSE_STATUSES and location_loader is not None else "EVALUATION_UNAVAILABLE"
    return build_phase2_report(manifest, run_status=checkpoint["status"], stage_counts=bundle.stage_counts, preparation=preparation, source_summary=source_summary, deepseek_result=result, top3=top3, evaluation_result=evaluation, reason_codes=tuple(result.reason_codes), rpc_count=rpc_count)


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle: json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        if Path(handle.name).exists(): Path(handle.name).unlink()


def _load_location_after_inference() -> Any:
    for path in sorted((ROOT / "dataset").rglob("*.json")):
        try: value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError): continue
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, list): stack.extend(item)
            elif isinstance(item, Mapping):
                fault = item.get("fault")
                if isinstance(fault, Mapping) and str(fault.get("transaction_hash", "")).lower() == TX_HASH:
                    return item.get("location")
                stack.extend(v for k, v in item.items() if k in {"data", "items", "records"})
    return None


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--live", action="store_true"); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--outflow-lambda", default="0.65"); args=parser.parse_args(argv)
    from src.cloc_v2.outflow_scoring import unit_decimal
    unit_decimal(args.outflow_lambda, "outflow_lambda")
    result_config = ResultDetectionConfig(outflow_lambda=args.outflow_lambda)
    manifest=build_locked_manifest()
    if not args.live: print("live_required=true"); print("rpc_request_count=0"); print("source_request_count=0"); print("deepseek_request_count=0"); return 2
    _atomic(args.output_dir / "locked_manifest.json", manifest.to_dict())
    _atomic(args.output_dir / "result_detection_config.json", result_config.to_dict())
    checkpoint_path=args.output_dir / "checkpoint.json"; report_path=args.output_dir / "phase2_real_pilot_report.json"
    _atomic(checkpoint_path, {"schema_version":"phase2_real_pilot_checkpoint_v1","manifest_sha256":manifest.manifest_sha256,"status":"LOCKED","rpc_request_count":0,"source_request_count":0,"deepseek_request_count":0,"reason_codes":[]})
    try:
        from src.cloc_v2.config_loader import load_chain_config
        config=load_chain_config("eth"); rpc_url=config.get("rpc_url")
        if not isinstance(rpc_url, str) or not rpc_url: raise ValueError("ETH_RPC_CONFIG_NOT_LOADED")
        rpc=JsonRpcClient(rpc_url, max_requests=MAX_RPC)
        def source_builder(bundle): return acquire_real_source_slices(bundle, config)
        def deepseek_runner(preparation):
            api_key=os.getenv("DEEPSEEK_API_KEY")
            if not api_key: raise ValueError("DEEPSEEK_KEY_NOT_LOADED")
            return execute_phase2_localization(preparation, Phase2DeepSeekConfig(), DeepSeekHTTPSOneShotTransport(), api_key)
        report=run_phase2_pilot(manifest, live=True, rpc_client=rpc, evidence_builder=lambda raw: build_real_phase2_evidence(raw, result_config=result_config), source_builder=source_builder, deepseek_runner=deepseek_runner, location_loader=_load_location_after_inference, checkpoint_callback=lambda value:_atomic(checkpoint_path, value))
        _atomic(report_path, report)
    except Exception as exc:
        print("pilot_status=STOPPED"); print("reason_code="+_code(exc,"PHASE2_PILOT_SETUP_ERROR")); return 1
    print("pilot_status="+report["run_status"]); print("report_path="+str(report_path)); print("rpc_request_count="+str(report["external_call_counts"]["rpc"])); print("source_request_count="+str(report["external_call_counts"]["source"])); print("deepseek_request_count="+str(report["external_call_counts"]["deepseek"])); return 0 if report["run_status"] in {"COMPLETED","COMPLETED_SAFE_STOP"} else 1


if __name__ == "__main__": raise SystemExit(main())

"""Blind 10-fault Ethereum localization run for the locked legacy-style v4 profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloc.runtime.execution_policy import ExecutionPolicy
from scripts.run_final_small_experiment import _fault_pool, _normal_pool
from scripts.run_localization15_topk_v4 import Phase2ClassifiedHTTPSOneShotTransport
from scripts.run_phase2_real_pilot import build_real_phase2_evidence
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.formal_normal_baseline_loader import DEFAULT_FORMAL_NORMAL_BASELINE_DIR
from src.cloc_v2.localization_ground_truth import parse_localization_ground_truth, resolve_localization_ground_truth
from src.cloc_v2.phase2_legacy_v4_profile import (
    LOCKED_LEGACY_V4_PROFILE,
    PROFILE_ID,
    PROFILE_STATUS,
    build_legacy_v4_candidates,
    legacy_v4_profile_sha256,
    select_legacy_v4_prompt_candidates,
    serialize_legacy_v4_profile,
)
from src.cloc_v2.phase2_localization_bridge import Phase2LocalizationPreparation, prepare_phase2_localization
from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekConfig, execute_phase2_localization
from src.cloc_v2.phase2_localization_prompt import upgrade_phase2_localization_prompt_to_topk_v4
from src.cloc_v2.source_fetcher import SourceFetcher
from src.cloc_v2.source_index import build_source_index
from src.cloc_v2.source_slices import build_source_request_slices
from src.cloc_v2.source_window_fallback import build_verified_source_window_fallback
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index, lookup_verified_source_cache


EXPERIMENT_TYPE = "CLOC_FAULT10_BLIND_LEGACY_V4_ENGINEERING_EXPERIMENT"
SELECTION_NAMESPACE = "CLOC_FAULT10_BLIND_V1"
SELECTION_SEED = "20260824"
CHAIN_ID = 1
SAMPLE_COUNT = 10
EXPECTED_PROFILE_SHA256 = "617bee01e72f6c71a5e3e42b5f8912b53c325516e494f3ff5e8e66f56b3353c9"
PROFILE_ARTIFACT = ROOT / "data" / "prototypes" / "phase2_v4_legacy_style_profile_lock_20260824" / "locked_profile.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "prototypes" / "localization_fault10_blind_legacy_v4_seed_20260824"
SOURCE_CACHE = ROOT / "data" / "source_cache"
MAX_RPC_REQUESTS = 11
MAX_SOURCE_REQUESTS = 30
MAX_SOURCE_REQUESTS_PER_SAMPLE = 3
MAX_DEEPSEEK_REQUESTS = 10
FAULT007 = "0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9"
_TX = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_FORBIDDEN_PUBLIC_KEYS = frozenset({
    "tx_hash", "transaction_hash", "address", "rpc_url", "api_key", "authorization",
    "raw_trace", "calldata", "input", "output", "source", "source_text", "prompt",
    "system_message", "user_message", "provider_response", "reasoning_content", "cause", "location",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        if Path(handle.name).exists():
            Path(handle.name).unlink()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_load(path: Path) -> Any | None:
    try:
        return _load(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _valid_tx(value: Any) -> str | None:
    value = str(value).lower()
    return value if _TX.fullmatch(value) else None


def _selection_rank(tx_hash: str) -> str:
    return hashlib.sha256(f"{SELECTION_NAMESPACE}|{SELECTION_SEED}|1|{tx_hash.lower()}".encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Fault10Sample:
    sample_alias: str
    blockchain_chain_id: int
    tx_hash: str = field(repr=False)
    order: int

    def __post_init__(self) -> None:
        if (
            self.sample_alias != f"fault10_eth_{self.order:03d}"
            or not 1 <= self.order <= SAMPLE_COUNT
            or self.blockchain_chain_id != 1
            or _valid_tx(self.tx_hash) is None
        ):
            raise ValueError("FAULT10_SAMPLE_INVALID")
        object.__setattr__(self, "tx_hash", self.tx_hash.lower())

    def to_dict(self) -> dict[str, Any]:
        return {"sample_alias": self.sample_alias, "blockchain_chain_id": 1, "tx_hash": self.tx_hash, "order": self.order}


@dataclass(frozen=True, slots=True)
class Fault10Manifest:
    samples: tuple[Fault10Sample, ...]
    excluded_history_count: int
    fault_pool_count: int
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            len(self.samples) != SAMPLE_COUNT
            or len({item.tx_hash for item in self.samples}) != SAMPLE_COUNT
            or tuple(item.order for item in self.samples) != tuple(range(1, SAMPLE_COUNT + 1))
            or self.excluded_history_count < 0
            or self.fault_pool_count < SAMPLE_COUNT
        ):
            raise ValueError("FAULT10_MANIFEST_INVALID")
        expected = _sha(self._core())
        if self.manifest_sha256 and self.manifest_sha256 != expected:
            raise ValueError("FAULT10_MANIFEST_SHA_INVALID")
        object.__setattr__(self, "manifest_sha256", expected)

    def _core(self) -> dict[str, Any]:
        return {
            "schema_version": "localization_fault10_blind_manifest_v1",
            "experiment_type": EXPERIMENT_TYPE,
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_seed": SELECTION_SEED,
            "blockchain_chain_id": 1,
            "profile_id": PROFILE_ID,
            "profile_sha256": EXPECTED_PROFILE_SHA256,
            "samples": [item.to_dict() for item in self.samples],
            "selection_audit": {
                "fault_pool_count": self.fault_pool_count,
                "excluded_history_count": self.excluded_history_count,
                "history_overlap_count": 0,
                "normal_fault_conflict_count": 0,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._core(), "scientific_generalization_result": False, "manifest_sha256": self.manifest_sha256}


def load_manifest(path: Path) -> Fault10Manifest:
    value = _load(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != "localization_fault10_blind_manifest_v1":
        raise ValueError("FAULT10_MANIFEST_SCHEMA_INVALID")
    samples = tuple(Fault10Sample(str(row["sample_alias"]), int(row["blockchain_chain_id"]), str(row["tx_hash"]), int(row["order"])) for row in value.get("samples", ()) if isinstance(row, Mapping))
    audit = value.get("selection_audit") if isinstance(value.get("selection_audit"), Mapping) else {}
    manifest = Fault10Manifest(samples, int(audit.get("excluded_history_count", -1)), int(audit.get("fault_pool_count", -1)), str(value.get("manifest_sha256", "")))
    if value.get("profile_sha256") != EXPECTED_PROFILE_SHA256 or value.get("profile_id") != PROFILE_ID:
        raise ValueError("FAULT10_PROFILE_MISMATCH")
    return manifest


def _history_transactions(prototypes_root: Path) -> set[str]:
    """Read only transaction identity fields from historical manifest artifacts."""
    found: set[str] = set()
    for path in sorted(prototypes_root.rglob("*manifest*.json")) if prototypes_root.exists() else ():
        value = _safe_load(path)
        if value is None:
            continue
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, Mapping):
                for key, child in item.items():
                    normalized = str(key).lower()
                    if normalized in {"tx_hash", "transaction_hash"}:
                        values = child if isinstance(child, (list, tuple)) else (child,)
                        for candidate in values:
                            tx = _valid_tx(candidate)
                            if tx:
                                found.add(tx)
                    elif isinstance(child, (Mapping, list, tuple)):
                        stack.append(child)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
    return found


def build_locked_fault10_manifest(dataset_root: Path, prototypes_root: Path, normal_baseline_root: Path) -> Fault10Manifest:
    fault_pool = _fault_pool(dataset_root)
    if len(fault_pool) != 79:
        raise ValueError("FAULT10_ETHEREUM_FAULT_POOL_COUNT_INVALID")
    normal_manifest = _safe_load(normal_baseline_root / "normal_manifest.json")
    checkpoint = _safe_load(normal_baseline_root / "checkpoint.json")
    if not isinstance(normal_manifest, Mapping) or not isinstance(checkpoint, Mapping):
        raise ValueError("FAULT10_NORMAL_BASELINE_UNAVAILABLE")
    normal_pool = _normal_pool(normal_manifest, checkpoint)
    conflicts = fault_pool & normal_pool
    history = _history_transactions(prototypes_root) | {FAULT007}
    eligible = fault_pool - history - conflicts
    ordered = sorted(eligible, key=lambda tx: (_selection_rank(tx), tx))
    if len(ordered) < SAMPLE_COUNT:
        raise ValueError("FAULT10_SELECTION_POOL_INSUFFICIENT")
    chosen = ordered[:SAMPLE_COUNT]
    if set(chosen) & history or set(chosen) & conflicts:
        raise ValueError("FAULT10_SELECTION_OVERLAP")
    samples = tuple(Fault10Sample(f"fault10_eth_{index:03d}", 1, tx, index) for index, tx in enumerate(chosen, start=1))
    return Fault10Manifest(samples, len(fault_pool & history), len(fault_pool))


def validate_locked_profile() -> Mapping[str, Any]:
    artifact = _load(PROFILE_ARTIFACT)
    serialized = json.loads(serialize_legacy_v4_profile())
    profile = LOCKED_LEGACY_V4_PROFILE
    disabled = set(profile.disabled_capabilities)
    required_disabled = {
        "MULTI_VIEW_CANDIDATE_RANK_FUSION", "PHASE2_COARSE_TO_FINE_TOPK_V5",
        "PHASE2_SOURCE_TOURNAMENT_TOPK_V6", "PHASE2_V4_100_CANDIDATE_POOL",
        "GRAPH_LINKED_SOURCE_FALLBACK_COEXISTS_WITH_RESOLVED_IDENTITY",
    }
    if (
        artifact != serialized
        or artifact.get("status") != PROFILE_STATUS
        or artifact.get("configuration_sha256") != EXPECTED_PROFILE_SHA256
        or legacy_v4_profile_sha256() != EXPECTED_PROFILE_SHA256
        or (profile.candidate_pool_limit, profile.prompt_candidate_limit, profile.topk_limit) != (20, 12, 5)
        or profile.max_deepseek_requests_per_sample != 1
        or profile.retry_enabled
        or not required_disabled.issubset(disabled)
    ):
        raise ValueError("FAULT10_PROFILE_PREFLIGHT_FAILED")
    return artifact


def _validate_public(value: Any) -> None:
    text = _canonical(value)
    if _RAW_ADDRESS.search(text) or _RAW_HASH.search(text):
        raise ValueError("FAULT10_PUBLIC_ARTIFACT_SENSITIVE")
    stack: list[tuple[Any, tuple[str, ...]]] = [(value, ())]
    while stack:
        current, path = stack.pop()
        if isinstance(current, Mapping):
            for key, child in current.items():
                normalized_key = str(key).strip().lower()
                safe_source_counter = (
                    normalized_key == "source"
                    and path[-1:] in {("physical_request_counts",), ("request_counts",)}
                    and isinstance(child, int)
                    and not isinstance(child, bool)
                    and child >= 0
                )
                if normalized_key in _FORBIDDEN_PUBLIC_KEYS and not safe_source_counter:
                    raise ValueError("FAULT10_PUBLIC_ARTIFACT_SENSITIVE")
                stack.append((child, (*path, normalized_key)))
        elif isinstance(current, (list, tuple)):
            stack.extend((child, path) for child in current)


def _prediction_envelope(manifest: Fault10Manifest, row: Mapping[str, Any]) -> dict[str, Any]:
    core = {"schema_version": "localization_fault10_prediction_v1", "locked_manifest_sha256": manifest.manifest_sha256, "prediction": dict(row)}
    return {**core, "payload_sha256": _sha(core)}


class RequestLedger:
    """Persist physical-attempt counters before every external call."""

    def __init__(self, checkpoint: dict[str, Any], checkpoint_path: Path):
        self.checkpoint = checkpoint
        self.path = checkpoint_path

    def save(self) -> None:
        _atomic(self.path, self.checkpoint)

    def before(self, kind: str, sample_alias: str | None = None) -> None:
        limits = {"rpc": MAX_RPC_REQUESTS, "source": MAX_SOURCE_REQUESTS, "deepseek": MAX_DEEPSEEK_REQUESTS}
        counts = self.checkpoint["physical_request_counts"]
        if kind not in limits or counts[kind] >= limits[kind]:
            raise RuntimeError("FAULT10_REQUEST_BUDGET_EXCEEDED")
        if sample_alias is not None:
            state = self.checkpoint["samples"][sample_alias]
            per_limit = {"rpc": 1, "source": MAX_SOURCE_REQUESTS_PER_SAMPLE, "deepseek": 1}[kind]
            if state["request_counts"][kind] >= per_limit:
                raise RuntimeError("FAULT10_PER_SAMPLE_REQUEST_BUDGET_EXCEEDED")
            state["request_counts"][kind] += 1
        counts[kind] += 1
        self.save()


class AtomicSourceSession:
    def __init__(self, delegate: Any, ledger: RequestLedger, sample_alias: str):
        self.delegate = delegate
        self.ledger = ledger
        self.sample_alias = sample_alias

    def get(self, url: str, params: Mapping[str, Any], timeout: int) -> Any:
        self.ledger.before("source", self.sample_alias)
        return self.delegate.get(url, params, timeout)


def acquire_legacy_source_slices(bundle: Any, chain_config: Mapping[str, Any], ledger: RequestLedger, sample_alias: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    fetcher = SourceFetcher("eth", dict(chain_config), cache_root=SOURCE_CACHE, policy=ExecutionPolicy(allow_explorer=True, allow_artifact_cache_write=True), max_retries=1, max_proxy_depth=3)
    fetcher.session = AtomicSourceSession(fetcher.session, ledger, sample_alias)
    cache_index = build_verified_source_cache_index(SOURCE_CACHE, blockchain_chain_id=1, chain_key="eth")
    acquisition_addresses: list[dict[str, Any]] = []
    cache_hits = 0
    stop_code = None
    for address in bundle.candidate_addresses:
        lookup = lookup_verified_source_cache(cache_index, runtime_address=address, requested_address=address)
        if lookup.status == "VERIFIED_SOURCE_CACHE_HIT" and lookup.binding is not None:
            acquisition_addresses.append({"requested_address": lookup.binding.artifact_address}); cache_hits += 1; continue
        if lookup.status in {"VERIFIED_SOURCE_CACHE_CONFLICT", "VERIFIED_SOURCE_CACHE_INVALID"}:
            stop_code = lookup.status; break
        result = fetcher.acquire(address, refresh=False, origins=())
        acquisition_addresses.append(result.to_dict())
        if result.status in {"network_error", "timeout", "rate_limited", "api_error", "parse_error"}:
            stop_code = "SOURCE_FETCH_EXTERNAL_ERROR"; break
    acquisition = {"schema_version": "source_acquisition_v1", "chain": "eth", "addresses": acquisition_addresses}
    index_document = build_source_index(acquisition, SOURCE_CACHE)
    requests = [{"chain_id": "Q", "node_id": node_id, "focus": "phase2_localization", "priority": 1 if position == 0 else 2} for position, node_id in enumerate(bundle.candidate_nodes)]
    slices = build_source_request_slices(index_document, bundle.graph.to_dict(), SOURCE_CACHE, requests)
    slices = build_verified_source_window_fallback(index_document, bundle.graph.to_dict(), slices, SOURCE_CACHE)
    refreshed_index = build_verified_source_cache_index(SOURCE_CACHE, blockchain_chain_id=1, chain_key="eth")
    slices = build_legacy_v4_candidates(bundle.graph, bundle.strict, slices, refreshed_index, SOURCE_CACHE)
    metadata = slices.get("metadata", {})
    summary = {
        "source_request_count": ledger.checkpoint["samples"][sample_alias]["request_counts"]["source"],
        "attempted_address_count": len(acquisition_addresses),
        "cache_hit_count": cache_hits,
        "resolved_slice_count": int(metadata.get("resolved_count", 0)),
        "source_resolution_mode": metadata.get("resolution_mode", "EXACT_FUNCTION_SLICE"),
        "source_selection_mode": metadata.get("source_selection_mode", "DIRECT_EVIDENCE_SOURCE"),
        "stop_code": stop_code,
    }
    return slices, summary


def _upgrade_v4(preparation: Phase2LocalizationPreparation) -> Phase2LocalizationPreparation:
    if preparation.preparation_status != "READY_FOR_LLM" or preparation.prompt is None:
        return preparation
    prompt = upgrade_phase2_localization_prompt_to_topk_v4(preparation.prompt)
    encoded = _canonical(prompt.to_dict()).encode("utf-8")
    return replace(preparation, prompt=prompt, prompt_sha256=hashlib.sha256(encoded).hexdigest(), prompt_bytes=len(encoded))


def build_live_processor(chain_config: Mapping[str, Any], api_key: str, ledger: RequestLedger) -> Callable[[Fault10Sample, Any], Mapping[str, Any]]:
    def process(sample: Fault10Sample, raw: Any) -> Mapping[str, Any]:
        try:
            bundle = build_real_phase2_evidence(raw)
        except Exception:
            return _safe_row(sample, "STRUCTURAL_ERROR", "EVIDENCE_PROCESSING_ERROR")
        try:
            slices, source_summary = acquire_legacy_source_slices(bundle, chain_config, ledger, sample.sample_alias)
        except RuntimeError as exc:
            return _safe_row(sample, "SOURCE_FAILURE", str(exc))
        except Exception:
            return _safe_row(sample, "SOURCE_FAILURE", "SOURCE_PROCESSING_ERROR")
        if source_summary.get("stop_code"):
            return _safe_row(sample, "SOURCE_FAILURE", str(source_summary["stop_code"]), source_summary=source_summary)
        selected = select_legacy_v4_prompt_candidates(slices)
        preparation = prepare_phase2_localization(
            bundle.strict, bundle.causal, bundle.graph, slices,
            max_pre_llm_candidates=LOCKED_LEGACY_V4_PROFILE.candidate_pool_limit,
            max_prompt_candidates=LOCKED_LEGACY_V4_PROFILE.prompt_candidate_limit,
            prompt_contract="v1",
        )
        base = {
            "sample_alias": sample.sample_alias,
            "status": preparation.preparation_status,
            "terminal_kind": "SAFE_STOP",
            "reason_codes": [item.code for item in preparation.issues],
            "profile_id": PROFILE_ID,
            "profile_sha256": EXPECTED_PROFILE_SHA256,
            "pre_llm_candidate_count": preparation.pre_llm_candidate_count,
            "prompt_candidate_count": preparation.prompt_candidate_count,
            "pre_llm_candidates": preparation.pre_llm_candidates,
            "source_resolution_mode": preparation.source_resolution_mode,
            "source_selection_mode": preparation.source_selection_mode,
            "selection_mode": preparation.candidate_selection_audit.get("selection_mode"),
            "prompt_sha256": preparation.prompt_sha256,
            "prompt_bytes": preparation.prompt_bytes,
            "deepseek_called": False,
            "deepseek_status": "NOT_CALLED",
            "topk_valid": False,
            "topk": [],
            "source_summary": source_summary,
            "provider_free_text_used": False,
        }
        expected_ids = [row["candidate_alias"] for row in selected]
        actual_ids = [row["candidate_alias"] for row in preparation.pre_llm_candidates if row.get("included_in_prompt")]
        if preparation.preparation_status == "READY_FOR_LLM" and (actual_ids != expected_ids or preparation.candidate_selection_audit.get("selection_mode") != "LEGACY_PREFIX_SELECTION"):
            return {**base, "status": "STRUCTURAL_ERROR", "terminal_kind": "SAFE_STOP", "reason_codes": ["LEGACY_V4_PROMPT_SELECTION_DRIFT"]}
        if preparation.preparation_status != "READY_FOR_LLM":
            terminal = "INSUFFICIENT_SOURCE" if preparation.preparation_status == "INSUFFICIENT_SOURCE" else "STRUCTURAL_ERROR" if preparation.preparation_status == "STRUCTURAL_ERROR" else "INSUFFICIENT_EVIDENCE"
            return {**base, "status": terminal}
        preparation = _upgrade_v4(preparation)
        ledger.before("deepseek", sample.sample_alias)
        result = execute_phase2_localization(preparation, Phase2DeepSeekConfig(), Phase2ClassifiedHTTPSOneShotTransport(), api_key)
        base.update(deepseek_called=True, deepseek_status=result.response_status, reason_codes=list(result.reason_codes), prompt_sha256=preparation.prompt_sha256, prompt_bytes=preparation.prompt_bytes)
        if result.response_status not in {"VALIDATED", "VALIDATED_WITH_NORMALIZATION"} or result.validated_response is None:
            terminal = "TRANSPORT_FAILURE" if result.response_status == "TRANSPORT_ERROR" else "INVALID_RESPONSE"
            return {**base, "status": terminal}
        rows = []
        for finding in result.validated_response.get("findings", ()):
            local = preparation.source_locations[finding["source_slice_id"]]
            row = {
                "rank": finding["rank"],
                "relative_file": local["relative_file"],
                "function": local["function"],
                "slice_start_line": local["start_line"],
                "slice_end_line": local["end_line"],
                "evidence_ids": list(finding["evidence_ids"]),
                "line_range_source": finding["line_range_source"],
            }
            if "predicted_line_start" in finding:
                row.update(predicted_line_start=finding["predicted_line_start"], predicted_line_end=finding["predicted_line_end"])
            rows.append(row)
        return {**base, "status": "VALID_TOPK", "terminal_kind": "VALID_TOPK", "topk_valid": True, "topk": rows}
    return process


def _safe_row(sample: Fault10Sample, status: str, reason: str, *, source_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "sample_alias": sample.sample_alias, "status": status, "terminal_kind": "SAFE_STOP",
        "reason_codes": [reason], "profile_id": PROFILE_ID, "profile_sha256": EXPECTED_PROFILE_SHA256,
        "pre_llm_candidate_count": 0, "prompt_candidate_count": 0, "pre_llm_candidates": [],
        "source_resolution_mode": None, "source_selection_mode": None, "selection_mode": "LEGACY_PREFIX_SELECTION",
        "prompt_sha256": None, "prompt_bytes": 0, "deepseek_called": False, "deepseek_status": "NOT_CALLED",
        "topk_valid": False, "topk": [], "source_summary": dict(source_summary or {}), "provider_free_text_used": False,
    }


def _initial_checkpoint(manifest: Fault10Manifest) -> dict[str, Any]:
    return {
        "schema_version": "localization_fault10_checkpoint_v1", "status": "LOCKED",
        "manifest_sha256": manifest.manifest_sha256, "profile_sha256": EXPECTED_PROFILE_SHA256,
        "physical_request_counts": {"rpc": 0, "source": 0, "deepseek": 0, "stage_a": 0, "tournament": 0, "retry": 0},
        "samples": {item.sample_alias: {"status": "PENDING", "prediction_artifact": None, "prediction_sha256": None, "request_counts": {"rpc": 0, "source": 0, "deepseek": 0}} for item in manifest.samples},
    }


def run_blind_inference(
    manifest: Fault10Manifest,
    output_dir: Path,
    *,
    rpc_client: Any,
    sample_processor: Callable[[Fault10Sample, Any], Mapping[str, Any]],
    checkpoint: dict[str, Any] | None = None,
) -> Mapping[str, Any]:
    checkpoint_path = output_dir / "checkpoint.json"
    state = checkpoint or _initial_checkpoint(manifest)
    ledger = RequestLedger(state, checkpoint_path)
    state["status"] = "RUNNING"; ledger.save()
    predictions: list[dict[str, Any]] = []
    try:
        ledger.before("rpc")
        chain_id = rpc_client.call("eth_chainId", [])
        if str(chain_id).lower() not in {"1", "0x1"}:
            raise RuntimeError("CHAIN_ID_MISMATCH")
    except Exception:
        state["status"] = "STOPPED_CHAIN_ID"; state["reason_codes"] = ["RPC_CHAIN_ID_ERROR"]; ledger.save(); return state
    for sample in manifest.samples:
        sample_state = state["samples"][sample.sample_alias]
        try:
            ledger.before("rpc", sample.sample_alias)
            raw = rpc_client.call("debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}])
        except Exception:
            row = _safe_row(sample, "TRACE_FAILURE", "RPC_TRACE_ERROR")
        else:
            try:
                row = dict(sample_processor(sample, raw))
            except Exception:
                row = _safe_row(sample, "STRUCTURAL_ERROR", "LOCAL_PROCESSING_ERROR")
        row["sample_alias"] = sample.sample_alias
        _validate_public(row)
        envelope = _prediction_envelope(manifest, row)
        artifact = Path("predictions") / f"{sample.sample_alias}.json"
        _atomic(output_dir / artifact, envelope)
        sample_state.update(status="TERMINAL", prediction_artifact=artifact.as_posix(), prediction_sha256=envelope["payload_sha256"])
        predictions.append(row); ledger.save()
    predictions.sort(key=lambda row: row["sample_alias"])
    core = {
        "schema_version": "localization_fault10_frozen_predictions_v1", "kind": PROFILE_ID,
        "locked_manifest_sha256": manifest.manifest_sha256, "profile_sha256": EXPECTED_PROFILE_SHA256,
        "predictions": predictions, "physical_request_counts": dict(state["physical_request_counts"]),
    }
    frozen = {**core, "payload_sha256": _sha(core)}
    _validate_public(frozen)
    _atomic(output_dir / "frozen_predictions.json", frozen)
    state.update(status="COMPLETED", frozen_predictions_sha256=frozen["payload_sha256"]); ledger.save()
    return frozen


def validate_frozen(output_dir: Path, manifest: Fault10Manifest) -> Mapping[str, Any]:
    frozen = _load(output_dir / "frozen_predictions.json")
    core = {key: frozen[key] for key in ("schema_version", "kind", "locked_manifest_sha256", "profile_sha256", "predictions", "physical_request_counts")}
    checkpoint = _load(output_dir / "checkpoint.json")
    if (
        frozen.get("payload_sha256") != _sha(core)
        or frozen.get("locked_manifest_sha256") != manifest.manifest_sha256
        or frozen.get("profile_sha256") != EXPECTED_PROFILE_SHA256
        or checkpoint.get("status") != "COMPLETED"
        or checkpoint.get("frozen_predictions_sha256") != frozen.get("payload_sha256")
        or len(frozen.get("predictions", ())) != SAMPLE_COUNT
    ):
        raise ValueError("FAULT10_FROZEN_PREDICTIONS_INVALID")
    for row in frozen["predictions"]:
        alias = row["sample_alias"]; artifact = output_dir / checkpoint["samples"][alias]["prediction_artifact"]
        envelope = _load(artifact); envelope_core = {key: envelope[key] for key in ("schema_version", "locked_manifest_sha256", "prediction")}
        if envelope.get("payload_sha256") != _sha(envelope_core) or envelope.get("payload_sha256") != checkpoint["samples"][alias]["prediction_sha256"]:
            raise ValueError("FAULT10_SAMPLE_PREDICTION_SHA_INVALID")
    _validate_public(frozen)
    return frozen


def _flatten_strings(value: Any) -> tuple[str, ...]:
    result: list[str] = []; stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str): result.append(item)
        elif isinstance(item, (list, tuple)): stack.extend(reversed(item))
    return tuple(result)


def prepare_evaluation_truth_after_freeze(output_dir: Path, manifest: Fault10Manifest, dataset_root: Path) -> Mapping[str, Any]:
    validate_frozen(output_dir, manifest)
    locations_by_tx: dict[str, list[str]] = {item.tx_hash: [] for item in manifest.samples}
    for path in sorted(dataset_root.rglob("*.json")):
        value = _safe_load(path)
        if not isinstance(value, Mapping) or str(value.get("platform", "")).lower() not in {"ethereum", "eth"}:
            continue
        fault = value.get("fault")
        if not isinstance(fault, Mapping):
            continue
        txs = {_valid_tx(item) for item in _flatten_strings(fault.get("transaction_hash"))}
        selected = {tx for tx in txs if tx in locations_by_tx}
        if selected:
            locations = list(_flatten_strings(fault.get("location")))
            for tx in selected:
                locations_by_tx[tx].extend(locations)
    cache_index = build_verified_source_cache_index(SOURCE_CACHE, blockchain_chain_id=1, chain_key="eth")
    labels = []
    for sample in manifest.samples:
        truths: set[tuple[Any, ...]] = set(); reasons: set[str] = set()
        for raw in locations_by_tx[sample.tx_hash]:
            try:
                parsed = parse_localization_ground_truth(raw, blockchain_chain_id=1)
                resolved = resolve_localization_ground_truth(parsed, cache_index)
            except ValueError as exc:
                reasons.add(str(exc)); continue
            reasons.update(resolved.reason_codes)
            if resolved.source_file_available and resolved.normalized_source_file:
                truths.add((resolved.normalized_source_file, resolved.function_signature, resolved.function_name, resolved.function_name_unique_in_file, resolved.truth_start_line, resolved.truth_end_line))
        safe_truths = [{"relative_file": item[0], "function_signature": item[1], "function_name": item[2], "function_name_unique_in_file": bool(item[3]), "truth_start_line": item[4], "truth_end_line": item[5]} for item in sorted(truths, key=lambda item: tuple("" if value is None else str(value) for value in item))]
        labels.append({"sample_alias": sample.sample_alias, "evaluation_status": "AVAILABLE" if safe_truths else "EVALUATION_UNAVAILABLE", "truths": safe_truths, "reason_codes": sorted(reasons) if reasons else ([] if safe_truths else ["GROUND_TRUTH_UNAVAILABLE"])})
    core = {"schema_version": "localization_fault10_evaluation_truth_v1", "locked_manifest_sha256": manifest.manifest_sha256, "labels": labels}
    truth = {**core, "ground_truth_sha256": _sha(core)}
    _validate_public(truth)
    _atomic(output_dir / "evaluation_ground_truth.json", truth)
    return truth


def _file_match(predicted: Any, truth: Any) -> bool:
    if not isinstance(predicted, str) or not isinstance(truth, str):
        return False
    left = PurePosixPath(predicted.replace("\\", "/")).as_posix().lstrip("./")
    right = PurePosixPath(truth.replace("\\", "/")).as_posix().lstrip("./")
    return left == right or left.endswith("/" + right) or right.endswith("/" + left)


def _truth_hit(truth: Mapping[str, Any], finding: Mapping[str, Any], level: str) -> bool:
    if not _file_match(finding.get("relative_file"), truth.get("relative_file")):
        return False
    if level == "file":
        return True
    predicted = finding.get("function") or finding.get("function_signature")
    function_hit = isinstance(predicted, str) and (predicted == truth.get("function_signature") or ("(" not in predicted and truth.get("function_name_unique_in_file") and predicted == truth.get("function_name")))
    if level == "function":
        return bool(function_hit)
    start, end = finding.get("predicted_line_start"), finding.get("predicted_line_end")
    return isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool) and isinstance(truth.get("truth_start_line"), int) and isinstance(truth.get("truth_end_line"), int) and start <= truth["truth_end_line"] and end >= truth["truth_start_line"]


def evaluate_after_freeze(output_dir: Path, manifest: Fault10Manifest) -> Mapping[str, Any]:
    frozen = validate_frozen(output_dir, manifest)
    frozen_bytes = (output_dir / "frozen_predictions.json").read_bytes()
    truth = _load(output_dir / "evaluation_ground_truth.json")
    truth_core = {key: truth[key] for key in ("schema_version", "locked_manifest_sha256", "labels")}
    if truth.get("ground_truth_sha256") != _sha(truth_core) or truth.get("locked_manifest_sha256") != manifest.manifest_sha256:
        raise ValueError("FAULT10_EVALUATION_TRUTH_INVALID")
    labels = {row["sample_alias"]: row for row in truth["labels"]}; predictions = {row["sample_alias"]: row for row in frozen["predictions"]}
    hit_counts = {f"{level}_recall_at_{k}": 0 for level in ("file", "function", "line_range") for k in (1, 3, 5)}
    denominators = {level: 0 for level in ("file", "function", "line_range")}; rows = []
    for sample in manifest.samples:
        prediction = predictions[sample.sample_alias]; label = labels[sample.sample_alias]; truths = label.get("truths", ())
        availability = {
            "file": any(isinstance(item.get("relative_file"), str) for item in truths),
            "function": any(isinstance(item.get("function_signature"), str) for item in truths),
            "line_range": any(isinstance(item.get("truth_start_line"), int) and isinstance(item.get("truth_end_line"), int) for item in truths),
        }
        topk = sorted((item for item in prediction.get("topk", ()) if isinstance(item, Mapping)), key=lambda item: int(item.get("rank", 999))) if prediction.get("topk_valid") else []
        sample_hits: dict[str, bool | None] = {}
        for level in ("file", "function", "line_range"):
            denominators[level] += int(availability[level])
            for k in (1, 3, 5):
                hit = availability[level] and any(_truth_hit(truth_row, finding, level) for finding in topk[:k] for truth_row in truths)
                hit_counts[f"{level}_recall_at_{k}"] += int(hit)
                if k == 5:
                    sample_hits[level] = bool(hit) if availability[level] else None
        rows.append({"sample_alias": sample.sample_alias, "candidate_pool_count": prediction.get("pre_llm_candidate_count", 0), "prompt_candidate_count": prediction.get("prompt_candidate_count", 0), "deepseek_called": bool(prediction.get("deepseek_called")), "topk_count": len(topk), "terminal_status": prediction.get("status"), "file_hit": sample_hits["file"], "function_hit": sample_hits["function"], "line_range_hit": sample_hits["line_range"], "evaluation_status": label.get("evaluation_status")})
    total = SAMPLE_COUNT
    metrics = {key: value / total for key, value in hit_counts.items()}
    conditional = {f"conditional_{key}": (value / denominators[key.split("_recall", 1)[0]] if denominators[key.split("_recall", 1)[0]] else None) for key, value in hit_counts.items()}
    status_counts: dict[str, int] = {}
    for prediction in frozen["predictions"]:
        status_counts[str(prediction.get("status"))] = status_counts.get(str(prediction.get("status")), 0) + 1
    checkpoint = _load(output_dir / "checkpoint.json")
    sample_sums = {kind: sum(state["request_counts"][kind] for state in checkpoint["samples"].values()) for kind in ("rpc", "source", "deepseek")}
    counter_consistent = (
        checkpoint["physical_request_counts"]["rpc"] == 1 + sample_sums["rpc"]
        and checkpoint["physical_request_counts"]["source"] == sample_sums["source"]
        and checkpoint["physical_request_counts"]["deepseek"] == sample_sums["deepseek"]
    )
    valid = status_counts.get("VALID_TOPK", 0); safe = total - valid
    report = {
        "schema_version": "localization_fault10_report_v1", "experiment_type": EXPERIMENT_TYPE,
        "scientific_generalization_result": False, "locked_manifest_sha256": manifest.manifest_sha256,
        "profile_id": PROFILE_ID, "profile_sha256": EXPECTED_PROFILE_SHA256, "profile_status": PROFILE_STATUS,
        "evaluation_ground_truth_sha256": truth["ground_truth_sha256"], "evaluation_sample_count": total,
        "per_sample": rows, "strict_end_to_end_metrics": metrics,
        "conditional_metrics": {**conditional, "denominators": denominators},
        "engineering_metrics": {"terminal_count": total, "terminal_coverage_rate": 1.0, "valid_topk_count": valid, "valid_topk_coverage_rate": valid / total, "safe_stop_count": safe, "safe_stop_rate": safe / total, "evaluation_unavailable_count": sum(row["evaluation_status"] == "EVALUATION_UNAVAILABLE" for row in rows), "status_counts": dict(sorted(status_counts.items()))},
        "physical_request_counts": dict(checkpoint["physical_request_counts"]),
        "request_count_consistent_with_sample_sum": counter_consistent,
        "history_overlap_count": 0, "location_read_phase": "AFTER_FROZEN_PREDICTIONS_SHA_VALIDATED",
        "cause_read": False, "production_modules_modified": False, "provider_free_text_used": False,
    }
    _validate_public(report)
    if not counter_consistent:
        raise ValueError("FAULT10_REQUEST_COUNT_INCONSISTENT")
    _atomic(output_dir / "localization_fault10_report.json", report)
    if (output_dir / "frozen_predictions.json").read_bytes() != frozen_bytes:
        raise ValueError("FAULT10_FROZEN_PREDICTIONS_MODIFIED_DURING_EVALUATION")
    return report


def run_isolated(
    manifest: Fault10Manifest,
    output_dir: Path,
    *,
    rpc_client: Any,
    sample_processor_factory: Callable[[RequestLedger], Callable[[Fault10Sample, Any], Mapping[str, Any]]],
    truth_preparer: Callable[[Path, Fault10Manifest], Mapping[str, Any]],
) -> Mapping[str, Any]:
    _atomic(output_dir / "locked_inference_manifest.json", manifest.to_dict())
    checkpoint = _initial_checkpoint(manifest); _atomic(output_dir / "checkpoint.json", checkpoint)
    ledger = RequestLedger(checkpoint, output_dir / "checkpoint.json")
    processor = sample_processor_factory(ledger)
    frozen = run_blind_inference(manifest, output_dir, rpc_client=rpc_client, sample_processor=processor, checkpoint=checkpoint)
    if frozen.get("schema_version") != "localization_fault10_frozen_predictions_v1":
        return frozen
    validate_frozen(output_dir, manifest)
    if (output_dir / "evaluation_ground_truth.json").exists():
        raise ValueError("FAULT10_TRUTH_CREATED_BEFORE_FREEZE")
    truth_preparer(output_dir, manifest)
    return evaluate_after_freeze(output_dir, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if not args.live:
        print("live_required=true"); print("rpc_request_count=0"); print("source_request_count=0"); print("deepseek_request_count=0"); return 2
    if args.output_dir.exists():
        print("experiment_status=STOPPED"); print("reason_code=OUTPUT_DIR_ALREADY_EXISTS"); return 1
    validate_locked_profile()
    chain_config = load_chain_config("eth")
    rpc_url = chain_config.get("rpc_url"); api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not isinstance(rpc_url, str) or not rpc_url or not isinstance(api_key, str) or not api_key:
        print("experiment_status=STOPPED"); print("reason_code=EXTERNAL_CONFIGURATION_MISSING"); return 1
    manifest = build_locked_fault10_manifest(ROOT / "dataset", ROOT / "data" / "prototypes", DEFAULT_FORMAL_NORMAL_BASELINE_DIR)
    rpc = JsonRpcClient(rpc_url, max_requests=MAX_RPC_REQUESTS)
    report = run_isolated(
        manifest, args.output_dir, rpc_client=rpc,
        sample_processor_factory=lambda ledger: build_live_processor(chain_config, api_key, ledger),
        truth_preparer=lambda output, locked: prepare_evaluation_truth_after_freeze(output, locked, ROOT / "dataset"),
    )
    if report.get("schema_version") != "localization_fault10_report_v1":
        print("experiment_status=STOPPED"); return 1
    print("evaluation_status=COMPLETED")
    print("evaluation_sample_count=10")
    print("rpc_request_count=" + str(report["physical_request_counts"]["rpc"]))
    print("source_request_count=" + str(report["physical_request_counts"]["source"]))
    print("deepseek_request_count=" + str(report["physical_request_counts"]["deepseek"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

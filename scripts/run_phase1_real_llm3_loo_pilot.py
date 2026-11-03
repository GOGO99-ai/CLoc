"""Three-sample real phase-1 LLM pilot with leave-one-transaction-out references.

Importing this module is fully offline.  External access is enabled only by
``--live`` and is bounded to one chain-id request, one callTracer request per
locked sample, and at most one DeepSeek request per eligible sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase1_deepseek_smoke import DeepSeekHTTPSOneShotTransport
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.asset_continuity import build_asset_continuity
from src.cloc_v2.calltracer_adapter import adapt_calltracer_call_tree
from src.cloc_v2.formal_normal_baseline_loader import (
    DEFAULT_FORMAL_NORMAL_BASELINE_DIR,
    LoadedFormalNormalBaseline,
    load_formal_normal_baseline,
    resolve_formal_normal_transaction_identity,
)
from src.cloc_v2.normal_baseline import NormalBaseline, build_normal_baseline
from src.cloc_v2.normal_features import extract_normal_features
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.phase1_direct_normal import Phase1DirectNormalMatch, evaluate_phase1_direct_normal_match
from src.cloc_v2.phase1_normality_bridge import Phase1NormalityPreparation, prepare_phase1_normality_review
from src.cloc_v2.phase1_normality_deepseek import (
    Phase1DeepSeekConfig,
    Phase1DeepSeekResult,
    execute_phase1_deepseek_review,
)
from src.cloc_v2.result_detection import detect_result_candidates
from src.cloc_v2.runtime_semantics import build_runtime_annotations
from src.cloc_v2.strict_result_chains import build_strict_result_chains
from src.cloc_v2.transaction_graph import build_transaction_graph


SCHEMA_VERSION = "phase1_real_llm3_loo_report_v2"
MANIFEST_SCHEMA_VERSION = "phase1_real_llm3_loo_locked_manifest_v1"
CHECKPOINT_SCHEMA_VERSION = "phase1_real_llm3_loo_checkpoint_v1"
EXPERIMENT_KIND = "LEAVE_ONE_TRANSACTION_OUT_REAL_LLM_ENGINEERING_CHECK"
BLOCKCHAIN_CHAIN_ID = 1
FORMAL_MIN_GROUP_SIZE = 3
MAX_RPC_REQUESTS = 4
MAX_DEEPSEEK_REQUESTS = 3
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "phase1_real_llm3_loo_20260814"

LOCKED_SAMPLE_ROWS = (
    (
        "eth_normal_000384",
        "0xd17f7b4d5438f921f631f5aa725b7997b5d927b49ba3dfe47581ff98b7ef9640",
    ),
    (
        "eth_normal_000106",
        "0x3a7380dc5d17aca389496788521fe9ead1f6651a7ca6e642da781b9c047700ab",
    ),
    (
        "eth_normal_000151",
        "0x54c1933f660d4e36260a3accb0a4818e72061275c2f1c68303f2c0a1dc40ef08",
    ),
)

_TX_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_TX_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_MACHINE_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FORBIDDEN_KEYS = frozenset(
    {
        "tx_hash", "transaction_hash", "address", "effective_address", "runtime_address",
        "input", "calldata", "output", "error", "revertreason", "reasoning_content",
        "api_key", "rpc_url", "system_message", "user_message", "provider_response",
        "prompt", "record_id", "source_sample_id", "raw_trace", "response",
    }
)
_EXTERNAL_FAILURE_STATUSES = frozenset({"TRANSPORT_ERROR", "PROVIDER_FAILED", "INCOMPLETE", "REFUSED"})


@dataclass(frozen=True)
class LockedLooSample:
    sample_alias: str
    blockchain_chain_id: int
    tx_hash: str = field(repr=False)
    source_sample_id: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.sample_alias not in {row[0] for row in LOCKED_SAMPLE_ROWS}:
            raise ValueError("LOO_SAMPLE_ALIAS_INVALID")
        if self.blockchain_chain_id != BLOCKCHAIN_CHAIN_ID:
            raise ValueError("LOO_SAMPLE_CHAIN_ID_INVALID")
        if not isinstance(self.tx_hash, str) or _TX_HASH.fullmatch(self.tx_hash) is None:
            raise ValueError("LOO_SAMPLE_HASH_INVALID")
        if self.source_sample_id != self.sample_alias:
            raise ValueError("LOO_SAMPLE_ORIGIN_INVALID")

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "sample_alias": self.sample_alias,
            "blockchain_chain_id": self.blockchain_chain_id,
            "tx_hash": self.tx_hash,
        }


@dataclass(frozen=True)
class LockedLooManifest:
    samples: tuple[LockedLooSample, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        expected = tuple(row[0] for row in LOCKED_SAMPLE_ROWS)
        if tuple(item.sample_alias for item in self.samples) != expected:
            raise ValueError("LOO_SAMPLE_ORDER_INVALID")
        if len({item.tx_hash for item in self.samples}) != len(self.samples):
            raise ValueError("LOO_SAMPLE_DUPLICATE")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("LOO_MANIFEST_SHA256_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_kind": EXPERIMENT_KIND,
            "scientific_generalization_result": False,
            "sample_count": len(self.samples),
            "samples": [item.to_manifest_dict() for item in self.samples],
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class PreparedLooEvidence:
    sample_alias: str
    stage_counts: Mapping[str, int]
    full_direct_match: Phase1DirectNormalMatch
    full_self_exact_match: bool
    excluded_normal_record_count: int
    loo_baseline_record_count: int
    loo_preparation: Phase1NormalityPreparation = field(repr=False)

    def __post_init__(self) -> None:
        counts = dict(self.stage_counts)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
            raise ValueError("LOO_STAGE_COUNTS_INVALID")
        if not isinstance(self.full_direct_match, Phase1DirectNormalMatch):
            raise ValueError("LOO_FULL_DIRECT_MATCH_INVALID")
        if not isinstance(self.loo_preparation, Phase1NormalityPreparation):
            raise ValueError("LOO_PREPARATION_INVALID")
        if self.excluded_normal_record_count <= 0 or self.loo_baseline_record_count < 0:
            raise ValueError("LOO_BASELINE_COUNTS_INVALID")
        object.__setattr__(self, "stage_counts", MappingProxyType(dict(sorted(counts.items()))))


@dataclass(frozen=True)
class LooPilotArtifacts:
    manifest: LockedLooManifest = field(repr=False)
    checkpoint: Mapping[str, Any]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        checkpoint = json.loads(_canonical_json(self.checkpoint))
        report = json.loads(_canonical_json(self.report))
        validate_sanitized_document(checkpoint)
        validate_sanitized_document(report)
        object.__setattr__(self, "checkpoint", MappingProxyType(checkpoint))
        object.__setattr__(self, "report", MappingProxyType(report))


def build_locked_loo_manifest() -> LockedLooManifest:
    """Return the immutable user-locked list without scanning any dataset."""
    samples = tuple(LockedLooSample(alias, 1, tx_hash, alias) for alias, tx_hash in LOCKED_SAMPLE_ROWS)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_kind": EXPERIMENT_KIND,
        "scientific_generalization_result": False,
        "samples": [item.to_manifest_dict() for item in samples],
    }
    return LockedLooManifest(samples, _sha256(_canonical_json(payload)))


def build_leave_one_transaction_out_baseline(
    loaded: LoadedFormalNormalBaseline,
    source_sample_id: str,
) -> tuple[NormalBaseline, Mapping[str, str], int]:
    """Exclude every record owned by one transaction and recalibrate in memory."""
    _validate_formal_baseline(loaded)
    if not isinstance(source_sample_id, str) or not source_sample_id:
        raise ValueError("LOO_SOURCE_SAMPLE_INVALID")
    records = tuple(record for group in loaded.baseline.groups for record in group.records)
    excluded_ids = {
        record_id
        for record_id, owner in loaded.normal_record_transaction_ids.items()
        if owner == source_sample_id
    }
    if not excluded_ids:
        raise ValueError("LOO_SOURCE_RECORDS_NOT_FOUND")
    retained = tuple(record for record in records if record.record_id not in excluded_ids)
    mapping = {
        record.record_id: loaded.normal_record_transaction_ids[record.record_id]
        for record in retained
    }
    rebuilt = build_normal_baseline(retained, min_group_size=FORMAL_MIN_GROUP_SIZE)
    return rebuilt, MappingProxyType(dict(sorted(mapping.items()))), len(excluded_ids)


def prepare_loo_sample(
    sample: LockedLooSample,
    raw_calltracer_frame: Any,
    loaded: LoadedFormalNormalBaseline,
) -> PreparedLooEvidence:
    """Build evidence once, then evaluate independent full and LOO baseline views."""
    _validate_formal_baseline(loaded)
    call_tree = adapt_calltracer_call_tree(raw_calltracer_frame)
    graph = build_transaction_graph(call_tree)
    annotations = build_runtime_annotations(graph)
    ledger = build_observed_asset_ledger(graph, annotations)
    continuity = build_asset_continuity(graph, annotations, ledger)
    continuity_document = continuity.to_dict()
    detected = detect_result_candidates(graph, annotations, ledger, continuity_document)
    strict = build_strict_result_chains(
        graph,
        continuity_document,
        detected.to_dict(),
        blockchain_chain_id=BLOCKCHAIN_CHAIN_ID,
        offline_function_candidates=(),
        offline_effective_address_map={},
    )
    extracted = extract_normal_features(
        graph,
        strict,
        ledger,
        transaction_record_id=sample.sample_alias,
        offline_function_candidates=(),
        offline_effective_address_map={},
    )

    query_identity = resolve_formal_normal_transaction_identity(
        loaded,
        sample.blockchain_chain_id,
        sample.tx_hash,
    )
    if (
        query_identity.status != "KNOWN_NORMAL_ORIGIN"
        or query_identity.normal_origin_id != sample.source_sample_id
    ):
        raise ValueError("LOO_QUERY_NORMAL_IDENTITY_MISMATCH")
    full_direct = evaluate_phase1_direct_normal_match(
        strict,
        extracted,
        loaded.baseline,
        query_identity,
        loaded.normal_record_transaction_ids,
    )
    self_exact = full_direct.known_transaction_direct_normal
    loo_baseline, loo_mapping, excluded_count = build_leave_one_transaction_out_baseline(
        loaded, sample.source_sample_id
    )
    preparation = prepare_phase1_normality_review(strict, extracted, loo_baseline, loo_mapping)
    complete_facts = sum(fact.status == "COMPLETE" for fact in ledger.transfer_facts)
    stage_counts = {
        "call_node_count": len(graph.nodes),
        "recognized_transfer_node_count": sum(
            "TOKEN_TRANSFER" in item.action_types for item in annotations.node_annotations
        ),
        "asset_fact_count": len(ledger.transfer_facts),
        "complete_asset_fact_count": complete_facts,
        "incomplete_asset_fact_count": len(ledger.transfer_facts) - complete_facts,
        "result_candidate_count": len(detected.result_candidates),
        "strict_chain_count": len(strict.chains),
        "strict_ready_count": sum(item.status == "READY" for item in strict.chains),
        "strict_uncertain_count": sum(item.status == "UNCERTAIN" for item in strict.chains),
        "clean_record_count": extracted.clean_record_count,
        "uncertain_record_count": extracted.uncertain_record_count,
        "rejected_record_count": extracted.rejected_chain_count,
    }
    return PreparedLooEvidence(
        sample.sample_alias,
        stage_counts,
        full_direct,
        self_exact,
        excluded_count,
        loo_baseline.record_count,
        preparation,
    )


def route_loo_sample(
    prepared: PreparedLooEvidence,
    deepseek_runner: Callable[[Phase1NormalityPreparation], Phase1DeepSeekResult],
) -> tuple[dict[str, Any], int, bool]:
    """Route one prepared sample; return row, LLM calls, and stop-external flag."""
    preparation = prepared.loo_preparation
    chain_reviews = _chain_reviews(preparation)
    base = {
        "sample_alias": prepared.sample_alias,
        "stage_counts": dict(prepared.stage_counts),
        "full_baseline_view": {
            "identity_direct_normal_enabled": True,
            "direct_normal": prepared.full_direct_match.direct_normal,
            "self_exact_match": prepared.full_self_exact_match,
            "query_normal_identity_status": prepared.full_direct_match.query_normal_identity_status,
            "decision_kind": prepared.full_direct_match.decision_kind,
            "clean_ready_chain_count": prepared.full_direct_match.clean_ready_chain_count,
            "exact_matched_clean_chain_count": prepared.full_direct_match.exact_matched_clean_chain_count,
            "self_origin_exact_matched_clean_chain_count": (
                prepared.full_direct_match.self_origin_exact_matched_clean_chain_count
            ),
            "uncertain_chain_count": prepared.full_direct_match.uncertain_chain_count,
        },
        "loo_view": {
            "identity_direct_normal_enabled": False,
            "excluded_normal_record_count": prepared.excluded_normal_record_count,
            "remaining_baseline_record_count": prepared.loo_baseline_record_count,
            "preparation_status": preparation.preparation_status,
            "eligibility": preparation.eligibility.eligibility,
            "total_strict_chain_count": preparation.eligibility.total_strict_chain_count,
            "usable_clean_chain_count": preparation.eligibility.usable_clean_chain_count,
            "excluded_uncertain_chain_count": preparation.eligibility.excluded_uncertain_chain_count,
            "chain_reviews": chain_reviews,
        },
    }
    if preparation.preparation_status != "READY_FOR_LLM":
        reasons = (*preparation.eligibility.reason_codes, *(issue.code for issue in preparation.issues))
        return ({
            **base,
            "llm_called": False,
            "deepseek_response_status": "NOT_REQUESTED",
            "deepseek_usage": None,
            "llm_chain_decisions": [],
            "final_route": "PROCEED_PHASE2",
            "reason_codes": list(_reason_codes(reasons)),
        }, 0, False)

    try:
        result = deepseek_runner(preparation)
    except Exception:
        return (_llm_failure_row(base, "DEEPSEEK_CLIENT_ERROR", "CLIENT_ERROR"), 1, True)
    if not isinstance(result, Phase1DeepSeekResult):
        return (_llm_failure_row(base, "DEEPSEEK_CLIENT_RESULT_INVALID", "CLIENT_ERROR"), 1, True)
    if result.response_status != "VALIDATED" or result.routing_decision is None:
        reason = result.reason_codes[0] if result.reason_codes else "DEEPSEEK_RESPONSE_INVALID"
        row = _llm_failure_row(base, reason, result.response_status)
        row["deepseek_usage"] = result.usage.to_dict() if result.usage is not None else None
        return row, 1, result.response_status in _EXTERNAL_FAILURE_STATUSES

    decisions = [item.to_dict() for item in result.routing_decision.chain_decisions]
    by_alias = {item["chain_alias"]: item["decision"] for item in decisions}
    for chain in base["loo_view"]["chain_reviews"]:
        chain["llm_decision"] = by_alias.get(chain["chain_alias"])
        chain["response_valid"] = True
    return ({
        **base,
        "llm_called": True,
        "deepseek_response_status": result.response_status,
        "deepseek_usage": result.usage.to_dict() if result.usage is not None else None,
        "llm_chain_decisions": decisions,
        "final_route": result.routing_decision.route,
        "reason_codes": list(_reason_codes(result.routing_decision.reason_codes)),
    }, 1, False)


def run_phase1_real_llm3_loo_pilot(
    manifest: LockedLooManifest,
    *,
    live: bool = False,
    rpc_client: Any | None = None,
    deepseek_runner: Callable[[Phase1NormalityPreparation], Phase1DeepSeekResult] | None = None,
    loaded_baseline: LoadedFormalNormalBaseline | None = None,
    sample_preparer: Callable[[LockedLooSample, Any, LoadedFormalNormalBaseline], PreparedLooEvidence] = prepare_loo_sample,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> LooPilotArtifacts:
    """Run the locked pilot once with explicit clients and exact hard caps."""
    if not isinstance(manifest, LockedLooManifest):
        raise ValueError("LOO_MANIFEST_REQUIRED")
    if not live:
        rows = tuple(_not_run_row(item.sample_alias) for item in manifest.samples)
        checkpoint = build_checkpoint(manifest, rows, 0, 0, "LIVE_MODE_REQUIRED")
        report = build_report(manifest, None, rows, 0, 0, "LIVE_MODE_REQUIRED")
        return LooPilotArtifacts(manifest, checkpoint, report)
    if rpc_client is None or not callable(getattr(rpc_client, "call", None)):
        raise ValueError("LOO_RPC_CLIENT_REQUIRED")
    if deepseek_runner is None or not callable(deepseek_runner):
        raise ValueError("LOO_DEEPSEEK_RUNNER_REQUIRED")
    if loaded_baseline is None:
        raise ValueError("LOO_BASELINE_REQUIRED")
    summary = _validate_formal_baseline(loaded_baseline)
    rows: list[dict[str, Any]] = []
    rpc_count = 1
    deepseek_count = 0
    stopped = False
    try:
        chain_id = rpc_client.call("eth_chainId", [])
    except Exception as exc:
        stopped = True
        rows.append(_failure_row(manifest.samples[0].sample_alias, _exception_code(exc, "RPC_CHAIN_ID_ERROR")))
    else:
        if _parse_chain_id(chain_id) != BLOCKCHAIN_CHAIN_ID:
            stopped = True
            rows.append(_failure_row(manifest.samples[0].sample_alias, "CHAIN_ID_MISMATCH"))

    for sample in manifest.samples:
        if rows and rows[-1]["sample_alias"] == sample.sample_alias:
            if progress_callback is not None:
                progress_callback(build_checkpoint(manifest, tuple(rows), rpc_count, deepseek_count, "STOPPED"))
            continue
        if stopped:
            rows.append(_not_run_row(sample.sample_alias, "EARLY_STOP_AFTER_EXTERNAL_ERROR"))
            continue
        try:
            if rpc_count >= MAX_RPC_REQUESTS:
                raise RuntimeError("RPC_REQUEST_LIMIT_REACHED")
            rpc_count += 1
            raw = rpc_client.call("debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}])
        except Exception as exc:
            rows.append(_failure_row(sample.sample_alias, _exception_code(exc, "RPC_TRACE_ERROR")))
            stopped = True
        else:
            try:
                prepared = sample_preparer(sample, raw, loaded_baseline)
            except Exception:
                rows.append(_failure_row(sample.sample_alias, "SAMPLE_PROCESSING_ERROR"))
            else:
                if prepared.sample_alias != sample.sample_alias:
                    rows.append(_failure_row(sample.sample_alias, "SAMPLE_ALIAS_MISMATCH"))
                elif prepared.loo_preparation.preparation_status == "READY_FOR_LLM" and deepseek_count >= MAX_DEEPSEEK_REQUESTS:
                    rows.append(_failure_row(sample.sample_alias, "DEEPSEEK_REQUEST_LIMIT_REACHED"))
                    stopped = True
                else:
                    row, used, stop_external = route_loo_sample(prepared, deepseek_runner)
                    deepseek_count += used
                    rows.append(row)
                    stopped = stop_external
        if progress_callback is not None:
            progress_callback(build_checkpoint(
                manifest, tuple(rows), rpc_count, deepseek_count, "STOPPED" if stopped else "RUNNING"
            ))

    checkpoint = build_checkpoint(
        manifest, tuple(rows), rpc_count, deepseek_count, "STOPPED" if stopped else "COMPLETED"
    )
    report = build_report(
        manifest, summary, tuple(rows), rpc_count, deepseek_count, "STOPPED" if stopped else "COMPLETED"
    )
    return LooPilotArtifacts(manifest, checkpoint, report)


def build_checkpoint(
    manifest: LockedLooManifest,
    rows: tuple[Mapping[str, Any], ...],
    rpc_request_count: int,
    deepseek_request_count: int,
    run_status: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "locked_manifest_sha256": manifest.manifest_sha256,
        "run_status": run_status,
        "completed_sample_aliases": [item["sample_alias"] for item in rows if item.get("preparation_status") != "NOT_RUN"],
        "sample_results": [dict(item) for item in rows],
        "external_call_counts": {
            "rpc": rpc_request_count,
            "deepseek": deepseek_request_count,
            "network": rpc_request_count + deepseek_request_count,
        },
    }
    validate_sanitized_document(payload)
    return payload


def build_report(
    manifest: LockedLooManifest,
    baseline_summary: Mapping[str, Any] | None,
    rows: tuple[Mapping[str, Any], ...],
    rpc_request_count: int,
    deepseek_request_count: int,
    run_status: str,
) -> dict[str, Any]:
    ordered = sorted((dict(item) for item in rows), key=lambda item: _sample_order(item["sample_alias"]))
    aliases = [item.sample_alias for item in manifest.samples]
    if [item["sample_alias"] for item in ordered] != aliases:
        raise ValueError("LOO_REPORT_SAMPLE_SET_INVALID")
    if not 0 <= rpc_request_count <= MAX_RPC_REQUESTS or not 0 <= deepseek_request_count <= MAX_DEEPSEEK_REQUESTS:
        raise ValueError("LOO_REQUEST_COUNT_INVALID")
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": EXPERIMENT_KIND,
        "scientific_generalization_result": False,
        "locked_manifest_sha256": manifest.manifest_sha256,
        "run_status": run_status,
        "formal_normal_baseline_summary": dict(baseline_summary) if baseline_summary is not None else None,
        "sample_results": ordered,
        "summary": {
            "full_baseline_direct_normal_count": sum(bool(item.get("full_baseline_view", {}).get("direct_normal")) for item in ordered),
            "loo_llm_eligible_count": sum(item.get("loo_view", {}).get("preparation_status") == "READY_FOR_LLM" for item in ordered),
            "deepseek_called_count": sum(bool(item.get("llm_called")) for item in ordered),
            "deepseek_valid_response_count": sum(item.get("deepseek_response_status") == "VALIDATED" for item in ordered),
            "llm_normal_count": sum(any(decision.get("decision") == "NORMAL" for decision in item.get("llm_chain_decisions", [])) for item in ordered),
            "llm_suspicious_count": sum(any(decision.get("decision") == "SUSPICIOUS" for decision in item.get("llm_chain_decisions", [])) for item in ordered),
            "llm_uncertain_count": sum(any(decision.get("decision") == "UNCERTAIN" for decision in item.get("llm_chain_decisions", [])) for item in ordered),
            "early_normal_count": sum(item.get("final_route") == "EARLY_NORMAL" for item in ordered),
            "phase2_count": sum(item.get("final_route") == "PROCEED_PHASE2" for item in ordered),
        },
        "request_limits": {"rpc": MAX_RPC_REQUESTS, "deepseek": MAX_DEEPSEEK_REQUESTS},
        "external_call_counts": {
            "rpc": rpc_request_count,
            "deepseek": deepseek_request_count,
            "network": rpc_request_count + deepseek_request_count,
        },
        "safety": {
            "retry_count": 0,
            "sample_replacement_count": 0,
            "formal_baseline_write_count": 0,
            "label_used_during_inference": False,
        },
    }
    validate_sanitized_document(report)
    return report


def serialize_locked_manifest(manifest: LockedLooManifest) -> str:
    return json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_checkpoint(checkpoint: Mapping[str, Any]) -> str:
    validate_sanitized_document(checkpoint)
    return json.dumps(dict(checkpoint), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_report(report: Mapping[str, Any]) -> str:
    validate_sanitized_document(report)
    return json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_locked_manifest(output_dir: str | Path, manifest: LockedLooManifest) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "locked_manifest.json"
    _atomic_write(path, serialize_locked_manifest(manifest))
    return path


def write_checkpoint(output_dir: str | Path, checkpoint: Mapping[str, Any]) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "checkpoint.json"
    _atomic_write(path, serialize_checkpoint(checkpoint))
    return path


def write_report(output_dir: str | Path, report: Mapping[str, Any]) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase1_real_llm3_loo_report.json"
    _atomic_write(path, serialize_report(report))
    return path


def validate_sanitized_document(value: Mapping[str, Any]) -> None:
    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).lower() in _FORBIDDEN_KEYS:
                    raise ValueError("LOO_DOCUMENT_FORBIDDEN_FIELD")
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            if _RAW_TX_HASH.search(item) or _RAW_ADDRESS.search(item):
                raise ValueError("LOO_DOCUMENT_RAW_IDENTIFIER_FORBIDDEN")
            lowered = item.lower()
            if any(token in lowered for token in ("reasoning_content", "calldata", "revertreason", "api key", "rpc_url", "bearer ")):
                raise ValueError("LOO_DOCUMENT_SENSITIVE_CONTENT_FORBIDDEN")
    visit(value)


def _chain_reviews(preparation: Phase1NormalityPreparation) -> list[dict[str, Any]]:
    if preparation.gate_evidence is None:
        return []
    gates = {item.query_chain_alias: item for item in preparation.gate_evidence.strict_chains}
    reviews: list[dict[str, Any]] = []
    for query in preparation.gate_evidence.query_chains:
        nearest = query.normal_references[0] if query.normal_references else None
        gate = gates.get(query.chain_alias)
        reviews.append({
            "chain_alias": query.chain_alias,
            "call_distance": _distance_text(nearest.call_distance if nearest is not None else None),
            "asset_distance": _distance_text(nearest.asset_distance if nearest is not None else None),
            "chain_distance": _distance_text(nearest.chain_distance if nearest is not None else query.nearest_distance),
            "dynamic_threshold": _distance_text(query.dynamic_threshold),
            "top3_count": len(query.normal_references),
            "normal_group_size": gate.normal_group_size if gate is not None else 0,
            "quantitative_status": query.quantitative_status,
            "llm_decision": None,
            "response_valid": False,
        })
    return reviews


def _llm_failure_row(base: Mapping[str, Any], reason: str, status: str) -> dict[str, Any]:
    return {
        **dict(base),
        "llm_called": True,
        "deepseek_response_status": status,
        "deepseek_usage": None,
        "llm_chain_decisions": [],
        "final_route": "PROCEED_PHASE2",
        "reason_codes": [_fixed_reason(reason)],
    }


def _failure_row(sample_alias: str, reason: str) -> dict[str, Any]:
    return {
        "sample_alias": sample_alias,
        "stage_counts": {},
        "full_baseline_view": {
            "identity_direct_normal_enabled": True,
            "direct_normal": False,
            "self_exact_match": False,
            "query_normal_identity_status": "UNRESOLVED",
            "decision_kind": "CONTINUE_STANDARD_PHASE1",
            "clean_ready_chain_count": 0,
            "exact_matched_clean_chain_count": 0,
            "self_origin_exact_matched_clean_chain_count": 0,
            "uncertain_chain_count": 0,
        },
        "loo_view": {
            "identity_direct_normal_enabled": False,
            "excluded_normal_record_count": 0, "remaining_baseline_record_count": 0,
            "preparation_status": "PROCESSING_FAILED", "eligibility": "INELIGIBLE",
            "total_strict_chain_count": 0, "usable_clean_chain_count": 0,
            "excluded_uncertain_chain_count": 0, "chain_reviews": [],
        },
        "llm_called": False,
        "deepseek_response_status": "NOT_REQUESTED",
        "deepseek_usage": None,
        "llm_chain_decisions": [],
        "final_route": "PROCEED_PHASE2",
        "reason_codes": [_fixed_reason(reason)],
    }


def _not_run_row(sample_alias: str, reason: str = "LIVE_MODE_REQUIRED") -> dict[str, Any]:
    row = _failure_row(sample_alias, reason)
    row["loo_view"]["preparation_status"] = "NOT_RUN"
    return row


def _validate_formal_baseline(loaded: LoadedFormalNormalBaseline) -> dict[str, Any]:
    if not isinstance(loaded, LoadedFormalNormalBaseline):
        raise ValueError("FORMAL_BASELINE_INVALID")
    summary = loaded.summary
    if (
        summary.formal_min_group_size != FORMAL_MIN_GROUP_SIZE
        or not summary.manifest_consistency_passed
        or not summary.reconstructed_baseline_match
        or not summary.ethereum_only
        or loaded.baseline.min_group_size != FORMAL_MIN_GROUP_SIZE
        or any(group.main_anchor.blockchain_chain_id != 1 for group in loaded.baseline.groups)
    ):
        raise ValueError("FORMAL_BASELINE_SAFETY_CHECK_FAILED")
    return summary.to_dict()


def _atomic_write(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _reason_codes(values: Any) -> tuple[str, ...]:
    return tuple(sorted({_fixed_reason(value) for value in values}))


def _fixed_reason(value: Any) -> str:
    return value if isinstance(value, str) and _MACHINE_CODE.fullmatch(value) else "NON_MACHINE_READABLE_REASON"


def _exception_code(exc: Exception, fallback: str) -> str:
    return _fixed_reason(exc.args[0]) if len(exc.args) == 1 else fallback


def _parse_chain_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]+", value):
        return int(value, 16)
    return None


def _distance_text(value: Any) -> str | None:
    return format(value, "f") if value is not None else None


def _sample_order(alias: str) -> int:
    return next(index for index, row in enumerate(LOCKED_SAMPLE_ROWS) if row[0] == alias)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_live_chain_config() -> Mapping[str, Any]:
    from src.cloc_v2.config_loader import load_chain_config
    return load_chain_config("eth")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.live:
        print("live_required=true")
        print("rpc_request_count=0")
        print("deepseek_request_count=0")
        return 2
    try:
        if (args.output_dir / "phase1_real_llm3_loo_report.json").exists():
            raise ValueError("LOO_OUTPUT_ALREADY_COMPLETED")
        manifest = build_locked_loo_manifest()
        manifest_path = write_locked_manifest(args.output_dir, manifest)
        initial = build_checkpoint(manifest, (), 0, 0, "LOCKED")
        checkpoint_path = write_checkpoint(args.output_dir, initial)
        loaded = load_formal_normal_baseline(DEFAULT_FORMAL_NORMAL_BASELINE_DIR)
        config = _load_live_chain_config()
        rpc_url = config.get("rpc_url")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not isinstance(rpc_url, str) or not rpc_url or not isinstance(api_key, str) or not api_key:
            raise ValueError("LIVE_CONFIGURATION_MISSING")
        rpc_client = JsonRpcClient(rpc_url, max_requests=MAX_RPC_REQUESTS)
        deepseek_config = Phase1DeepSeekConfig()

        def deepseek_runner(preparation: Phase1NormalityPreparation) -> Phase1DeepSeekResult:
            return execute_phase1_deepseek_review(
                preparation, deepseek_config, DeepSeekHTTPSOneShotTransport(), api_key
            )

        artifacts = run_phase1_real_llm3_loo_pilot(
            manifest,
            live=True,
            rpc_client=rpc_client,
            deepseek_runner=deepseek_runner,
            loaded_baseline=loaded,
            progress_callback=lambda checkpoint: write_checkpoint(args.output_dir, checkpoint),
        )
        checkpoint_path = write_checkpoint(args.output_dir, artifacts.checkpoint)
        report_path = write_report(args.output_dir, artifacts.report)
    except Exception as exc:
        print(f"pilot_status={_exception_code(exc, 'LOO_PILOT_EXECUTION_ERROR')}")
        return 1
    print(f"pilot_status={artifacts.report['run_status']}")
    print(f"manifest_path={manifest_path}")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"report_path={report_path}")
    print(f"rpc_request_count={artifacts.report['external_call_counts']['rpc']}")
    print(f"deepseek_request_count={artifacts.report['external_call_counts']['deepseek']}")
    return 0 if artifacts.report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR", "EXPERIMENT_KIND", "LOCKED_SAMPLE_ROWS", "LooPilotArtifacts",
    "LockedLooManifest", "LockedLooSample", "PreparedLooEvidence", "build_checkpoint",
    "build_leave_one_transaction_out_baseline", "build_locked_loo_manifest", "build_report",
    "main", "prepare_loo_sample", "route_loo_sample", "run_phase1_real_llm3_loo_pilot",
    "serialize_checkpoint", "serialize_locked_manifest", "serialize_report",
    "validate_sanitized_document", "write_checkpoint", "write_locked_manifest", "write_report",
]

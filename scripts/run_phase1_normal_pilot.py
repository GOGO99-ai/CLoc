"""Closed-set normal re-identification pilot for locked ETH samples.

The command is offline by default.  Only ``--live`` enables one chain-id call,
one callTracer call per selected sample, and at most one gated DeepSeek call per
selected sample.  Selection reads
only the formal manifest/checkpoint before any feature or diagnostic artifact.
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
from src.cloc_v2.normal_features import extract_normal_features
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.phase1_direct_normal import (
    DIRECT_NORMAL_REASON,
    Phase1DirectNormalMatch,
    evaluate_phase1_direct_normal_match,
)
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


SCHEMA_VERSION = "phase1_normal_pilot_v2"
MANIFEST_SCHEMA_VERSION = "phase1_normal_pilot_locked_manifest_v1"
PILOT_KIND = "CLOSED_SET_NORMAL_REIDENTIFICATION_ENGINEERING_CHECK"
DEFAULT_SELECTION_MODE = "lexicographic3"
RANDOM10_SELECTION_MODE = "random10"
DEFAULT_SELECTION_RULE = "LEXICOGRAPHIC_FIRST_3_OF_470_SUCCEEDED_ETHEREUM_NORMAL_TRANSACTIONS"
RANDOM10_SELECTION_RULE = "SHA256_RANKED_RANDOM_10_OF_467_EXCLUDING_PRIOR_LEXICOGRAPHIC3_ETHEREUM_NORMAL_TRANSACTIONS"
RANDOM10_ALGORITHM = "SHA256(CLOC_NORMAL_RANDOM10_V1|seed|lowercase_tx_hash), THEN_TX_HASH"
DEFAULT_SELECTION_SEED: int | None = None
RANDOM10_SAMPLE_COUNT = 10
PRIOR_LEXICOGRAPHIC3_HASHES = frozenset({
    "0x01e6371b60c2ca36b3673bab9edf4caeba03e6dc756c317a90f856c5735cd2c4",
    "0x01f5ebdf91aff85812f5f9305944fd59b586c527c2ff35068394caeecbc8274a",
    "0x01f8b613d9f87f0f7fcde83c35a550b4fc758e8d218086de8ab39f31db9e72b0",
})
BLOCKCHAIN_CHAIN_ID = 1
MAX_RPC_REQUESTS = 4
MAX_DEEPSEEK_REQUESTS = 3
EXPECTED_SUCCEEDED_ETHEREUM = 470
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "phase1_normal_pilot"

_TX_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_TX_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_MACHINE_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "tx_hash", "transaction_hash", "address", "effective_address", "runtime_address",
        "input", "calldata", "output", "error", "revertreason", "reasoning_content",
        "api_key", "rpc_url", "system_message", "user_message", "response", "raw_trace",
        "provider_response", "prompt", "record_id", "source_sample_id",
    }
)


@dataclass(frozen=True)
class LockedNormalSample:
    sample_alias: str
    blockchain_chain_id: int
    tx_hash: str = field(repr=False)
    source_sample_id: str = field(repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"normal_(?:pilot|random10)_eth_0(?:0[1-9]|10)", self.sample_alias):
            raise ValueError("NORMAL_PILOT_ALIAS_INVALID")
        if self.blockchain_chain_id != 1:
            raise ValueError("NORMAL_PILOT_CHAIN_ID_INVALID")
        if not isinstance(self.tx_hash, str) or _TX_HASH.fullmatch(self.tx_hash) is None:
            raise ValueError("NORMAL_PILOT_HASH_INVALID")
        if not isinstance(self.source_sample_id, str) or not self.source_sample_id:
            raise ValueError("NORMAL_PILOT_SOURCE_SAMPLE_INVALID")

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "sample_alias": self.sample_alias,
            "blockchain_chain_id": self.blockchain_chain_id,
            "tx_hash": self.tx_hash,
        }


@dataclass(frozen=True)
class LockedNormalManifest:
    samples: tuple[LockedNormalSample, ...]
    manifest_sha256: str
    selection_rule: str = DEFAULT_SELECTION_RULE
    selection_mode: str = DEFAULT_SELECTION_MODE
    selection_seed: int | None = DEFAULT_SELECTION_SEED
    selection_algorithm: str | None = None

    def __post_init__(self) -> None:
        if self.selection_mode not in {DEFAULT_SELECTION_MODE, RANDOM10_SELECTION_MODE}:
            raise ValueError("NORMAL_PILOT_SELECTION_MODE_INVALID")
        expected_count = 3 if self.selection_mode == DEFAULT_SELECTION_MODE else RANDOM10_SAMPLE_COUNT
        if len(self.samples) != expected_count:
            raise ValueError("NORMAL_PILOT_SAMPLE_COUNT_INVALID")
        alias_prefix = "normal_pilot_eth" if self.selection_mode == DEFAULT_SELECTION_MODE else "normal_random10_eth"
        expected_aliases = tuple(f"{alias_prefix}_{index:03d}" for index in range(1, expected_count + 1))
        if tuple(item.sample_alias for item in self.samples) != expected_aliases:
            raise ValueError("NORMAL_PILOT_SAMPLE_ORDER_INVALID")
        if self.selection_mode == DEFAULT_SELECTION_MODE and (self.selection_rule != DEFAULT_SELECTION_RULE or self.selection_seed is not None):
            raise ValueError("NORMAL_PILOT_DEFAULT_SELECTION_METADATA_INVALID")
        if self.selection_mode == RANDOM10_SELECTION_MODE:
            if not isinstance(self.selection_seed, int) or isinstance(self.selection_seed, bool) or self.selection_algorithm != RANDOM10_ALGORITHM:
                raise ValueError("NORMAL_PILOT_RANDOM_SELECTION_METADATA_INVALID")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("NORMAL_PILOT_MANIFEST_SHA256_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "selection_rule": self.selection_rule,
            "selection_mode": self.selection_mode,
            "selection_seed": self.selection_seed,
            "selection_algorithm": self.selection_algorithm,
            "sample_count": len(self.samples),
            "samples": [item.to_manifest_dict() for item in self.samples],
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class PreparedNormalPilotEvidence:
    sample_alias: str
    stage_counts: Mapping[str, int]
    coverage_rate: str | None
    critical_missing_count: int | None
    direct_match: Phase1DirectNormalMatch
    self_reference_match: bool
    preparation: Phase1NormalityPreparation | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        counts = dict(self.stage_counts)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
            raise ValueError("NORMAL_PILOT_STAGE_COUNTS_INVALID")
        object.__setattr__(self, "stage_counts", MappingProxyType(dict(sorted(counts.items()))))
        if self.direct_match.direct_normal and self.preparation is not None:
            raise ValueError("DIRECT_NORMAL_MUST_PRECEDE_STANDARD_PREPARATION")
        if not self.direct_match.direct_normal and not isinstance(self.preparation, Phase1NormalityPreparation):
            raise ValueError("STANDARD_PREPARATION_REQUIRED_AFTER_DIRECT_MISS")


@dataclass(frozen=True)
class NormalPilotArtifacts:
    manifest: LockedNormalManifest = field(repr=False)
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        report = json.loads(_canonical_json(self.report))
        validate_normal_pilot_report(report)
        object.__setattr__(self, "report", MappingProxyType(report))


def build_locked_normal_manifest(
    formal_directory: str | Path = DEFAULT_FORMAL_NORMAL_BASELINE_DIR,
    *,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    selection_seed: int | None = DEFAULT_SELECTION_SEED,
) -> LockedNormalManifest:
    """Select locked ETH hashes before reading any feature artifact."""
    root = Path(formal_directory)
    manifest = _read_json(root / "normal_manifest.json", "NORMAL_MANIFEST_INVALID")
    checkpoint = _read_json(root / "checkpoint.json", "NORMAL_CHECKPOINT_INVALID")
    if not isinstance(manifest, Mapping) or not isinstance(checkpoint, Mapping):
        raise ValueError("NORMAL_SELECTION_DOCUMENT_INVALID")
    if manifest.get("manifest_sha256") != checkpoint.get("manifest_sha256"):
        raise ValueError("NORMAL_SELECTION_MANIFEST_MISMATCH")
    states = checkpoint.get("samples")
    rows = manifest.get("samples")
    if not isinstance(states, Mapping) or not isinstance(rows, list):
        raise ValueError("NORMAL_SELECTION_STRUCTURE_INVALID")
    succeeded: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("chain_key") != "eth" or row.get("blockchain_chain_id") != 1:
            continue
        tx_hash = row.get("tx_hash")
        source_id = row.get("sample_id")
        if not isinstance(tx_hash, str) or _TX_HASH.fullmatch(tx_hash) is None:
            raise ValueError("NORMAL_SELECTION_HASH_INVALID")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("NORMAL_SELECTION_SOURCE_ID_INVALID")
        state = states.get(f"eth:{tx_hash}")
        if isinstance(state, Mapping) and state.get("status") == "SUCCEEDED":
            succeeded.append((tx_hash, source_id))
    succeeded.sort(key=lambda item: item[0])
    if len(succeeded) != EXPECTED_SUCCEEDED_ETHEREUM:
        raise ValueError("NORMAL_SELECTION_SUCCEEDED_COUNT_DRIFT")
    if selection_mode == DEFAULT_SELECTION_MODE:
        if selection_seed is not None:
            raise ValueError("NORMAL_PILOT_DEFAULT_SELECTION_SEED_FORBIDDEN")
        chosen = succeeded[:3]
        alias_prefix = "normal_pilot_eth"
        selection_rule = DEFAULT_SELECTION_RULE
        selection_algorithm = None
    elif selection_mode == RANDOM10_SELECTION_MODE:
        if not isinstance(selection_seed, int) or isinstance(selection_seed, bool):
            raise ValueError("NORMAL_PILOT_RANDOM_SELECTION_SEED_INVALID")
        candidate_rows = [item for item in succeeded if item[0] not in PRIOR_LEXICOGRAPHIC3_HASHES]
        if len(candidate_rows) != EXPECTED_SUCCEEDED_ETHEREUM - len(PRIOR_LEXICOGRAPHIC3_HASHES):
            raise ValueError("NORMAL_PILOT_RANDOM_SELECTION_POOL_DRIFT")
        seed_text = str(selection_seed)
        chosen = sorted(
            candidate_rows,
            key=lambda item: (_sha256(f"CLOC_NORMAL_RANDOM10_V1|{seed_text}|{item[0]}"), item[0]),
        )[:RANDOM10_SAMPLE_COUNT]
        alias_prefix = "normal_random10_eth"
        selection_rule = RANDOM10_SELECTION_RULE
        selection_algorithm = RANDOM10_ALGORITHM
    else:
        raise ValueError("NORMAL_PILOT_SELECTION_MODE_INVALID")
    samples = tuple(
        LockedNormalSample(f"{alias_prefix}_{index:03d}", 1, tx_hash, source_id)
        for index, (tx_hash, source_id) in enumerate(chosen, start=1)
    )
    digest_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selection_rule": selection_rule,
        "selection_mode": selection_mode,
        "selection_seed": selection_seed,
        "selection_algorithm": selection_algorithm,
        "samples": [item.to_manifest_dict() for item in samples],
    }
    return LockedNormalManifest(
        samples, _sha256(_canonical_json(digest_payload)), selection_rule,
        selection_mode, selection_seed, selection_algorithm,
    )


def prepare_normal_pilot_sample(
    sample: LockedNormalSample,
    raw_calltracer_frame: Any,
    loaded_baseline: LoadedFormalNormalBaseline,
) -> PreparedNormalPilotEvidence:
    """Run the formal evidence pipeline, exact branch, then standard bridge if needed."""
    _validate_formal_baseline(loaded_baseline)
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
        blockchain_chain_id=1,
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
        loaded_baseline,
        sample.blockchain_chain_id,
        sample.tx_hash,
    )
    direct = evaluate_phase1_direct_normal_match(
        strict,
        extracted,
        loaded_baseline.baseline,
        query_identity,
        loaded_baseline.normal_record_transaction_ids,
    )
    self_reference = direct.known_transaction_direct_normal
    preparation = None
    if not direct.direct_normal:
        preparation = prepare_phase1_normality_review(
            strict,
            extracted,
            loaded_baseline.baseline,
            loaded_baseline.normal_record_transaction_ids,
        )
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
    return PreparedNormalPilotEvidence(
        sample.sample_alias,
        stage_counts,
        _distance_text(direct.coverage_rate),
        direct.critical_missing_count,
        direct,
        self_reference,
        preparation,
    )


def route_normal_pilot_sample(
    prepared: PreparedNormalPilotEvidence,
    deepseek_runner: Callable[[Phase1NormalityPreparation], Phase1DeepSeekResult],
) -> tuple[dict[str, Any], int, bool]:
    """Return report row, request count, and whether an execution error occurred."""
    base = {
        "sample_alias": prepared.sample_alias,
        "stage_counts": dict(prepared.stage_counts),
        "coverage_rate": prepared.coverage_rate,
        "critical_missing_count": prepared.critical_missing_count,
        "result_candidate_count": prepared.direct_match.result_candidate_count,
        "clean_ready_chain_count": prepared.direct_match.clean_ready_chain_count,
        "exact_matched_clean_chain_count": prepared.direct_match.exact_matched_clean_chain_count,
        "self_origin_exact_matched_clean_chain_count": (
            prepared.direct_match.self_origin_exact_matched_clean_chain_count
        ),
        "clean_unmatched_chain_count": prepared.direct_match.clean_unmatched_chain_count,
        "uncertain_chain_count": prepared.direct_match.uncertain_chain_count,
        "unusable_ready_chain_count": prepared.direct_match.unusable_ready_chain_count,
        "chain_matches": [item.to_dict() for item in prepared.direct_match.chain_matches],
        "direct_normal": prepared.direct_match.direct_normal,
        "direct_normal_reason": prepared.direct_match.direct_normal_reason,
        "self_reference_match": prepared.self_reference_match,
        "query_normal_identity_status": prepared.direct_match.query_normal_identity_status,
        "known_transaction_direct_normal": prepared.direct_match.known_transaction_direct_normal,
        "structural_exact_collision": prepared.direct_match.structural_exact_collision,
        "decision_kind": prepared.direct_match.decision_kind,
    }
    if prepared.direct_match.direct_normal:
        return ({
            **base,
            "preparation_status": "KNOWN_TRANSACTION_DIRECT_NORMAL",
            "eligibility": "DIRECT_NORMAL",
            "llm_called": False,
            "deepseek_response_status": "NOT_REQUESTED",
            "deepseek_usage": None,
            "final_route": "EARLY_NORMAL",
            "correctly_identified_normal": True,
            "reason_codes": [DIRECT_NORMAL_REASON],
        }, 0, False)

    if prepared.direct_match.structural_exact_collision:
        return ({
            **base,
            "preparation_status": "STRUCTURAL_EXACT_COLLISION_PHASE2",
            "eligibility": "INELIGIBLE",
            "llm_called": False,
            "deepseek_response_status": "NOT_REQUESTED",
            "deepseek_usage": None,
            "final_route": "PROCEED_PHASE2",
            "correctly_identified_normal": False,
            "reason_codes": list(prepared.direct_match.reason_codes),
        }, 0, False)

    preparation = prepared.preparation
    if preparation.preparation_status != "READY_FOR_LLM":
        reasons = (*prepared.direct_match.reason_codes, *preparation.eligibility.reason_codes, *(i.code for i in preparation.issues))
        return ({
            **base,
            "preparation_status": preparation.preparation_status,
            "eligibility": preparation.eligibility.eligibility,
            "llm_called": False,
            "deepseek_response_status": "NOT_REQUESTED",
            "deepseek_usage": None,
            "final_route": "PROCEED_PHASE2",
            "correctly_identified_normal": False,
            "reason_codes": list(_reason_codes(reasons)),
        }, 0, False)
    try:
        result = deepseek_runner(preparation)
    except Exception:
        return (_failure_row(prepared.sample_alias, "DEEPSEEK_CLIENT_ERROR", base=base, llm_called=True), 1, True)
    if not isinstance(result, Phase1DeepSeekResult):
        return (_failure_row(prepared.sample_alias, "DEEPSEEK_CLIENT_RESULT_INVALID", base=base, llm_called=True), 1, True)
    if result.response_status != "VALIDATED" or result.routing_decision is None:
        code = result.reason_codes[0] if result.reason_codes else "DEEPSEEK_RESPONSE_INVALID"
        row = _failure_row(prepared.sample_alias, code, base=base, llm_called=True)
        row["deepseek_response_status"] = result.response_status
        row["deepseek_usage"] = result.usage.to_dict() if result.usage is not None else None
        return row, 1, True
    return ({
        **base,
        "preparation_status": preparation.preparation_status,
        "eligibility": preparation.eligibility.eligibility,
        "llm_called": True,
        "deepseek_response_status": result.response_status,
        "deepseek_usage": result.usage.to_dict() if result.usage is not None else None,
        "final_route": result.routing_decision.route,
        "correctly_identified_normal": result.routing_decision.route == "EARLY_NORMAL",
        "reason_codes": list(_reason_codes(result.routing_decision.reason_codes)),
    }, 1, False)


def run_phase1_normal_pilot(
    manifest: LockedNormalManifest,
    *,
    live: bool = False,
    rpc_client: Any | None = None,
    deepseek_runner: Callable[[Phase1NormalityPreparation], Phase1DeepSeekResult] | None = None,
    loaded_baseline: LoadedFormalNormalBaseline | None = None,
    sample_preparer: Callable[[LockedNormalSample, Any, LoadedFormalNormalBaseline], PreparedNormalPilotEvidence] = prepare_normal_pilot_sample,
) -> NormalPilotArtifacts:
    if not isinstance(manifest, LockedNormalManifest):
        raise ValueError("NORMAL_PILOT_MANIFEST_REQUIRED")
    if not live:
        rows = tuple(_not_run_row(item.sample_alias) for item in manifest.samples)
        return NormalPilotArtifacts(manifest, build_normal_pilot_report(manifest, None, rows, 0, 0, "LIVE_MODE_REQUIRED"))
    if rpc_client is None or not callable(getattr(rpc_client, "call", None)):
        raise ValueError("NORMAL_PILOT_RPC_CLIENT_REQUIRED")
    if deepseek_runner is None or not callable(deepseek_runner):
        raise ValueError("NORMAL_PILOT_DEEPSEEK_RUNNER_REQUIRED")
    if loaded_baseline is None:
        raise ValueError("NORMAL_PILOT_BASELINE_REQUIRED")
    baseline_summary = _validate_formal_baseline(loaded_baseline)
    limits = _request_limits(manifest)
    rpc_count = 1
    deepseek_count = 0
    rows: list[dict[str, Any]] = []
    try:
        chain_id = rpc_client.call("eth_chainId", [])
    except Exception as exc:
        code = _exception_code(exc, "RPC_CHAIN_ID_ERROR")
        rows = [_failure_row(item.sample_alias, code) for item in manifest.samples]
        return NormalPilotArtifacts(manifest, build_normal_pilot_report(manifest, baseline_summary, tuple(rows), 1, 0, "STOPPED"))
    if _parse_chain_id(chain_id) != 1:
        rows = [_failure_row(item.sample_alias, "CHAIN_ID_MISMATCH") for item in manifest.samples]
        return NormalPilotArtifacts(manifest, build_normal_pilot_report(manifest, baseline_summary, tuple(rows), 1, 0, "STOPPED"))

    stopped = False
    for index, sample in enumerate(manifest.samples):
        if stopped:
            rows.append(_not_run_row(sample.sample_alias, "EARLY_STOP_AFTER_ERROR"))
            continue
        try:
            if rpc_count >= limits["rpc"]:
                raise RuntimeError("RPC_REQUEST_LIMIT_REACHED")
            rpc_count += 1
            raw = rpc_client.call("debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}])
        except Exception as exc:
            rows.append(_failure_row(sample.sample_alias, _exception_code(exc, "RPC_TRACE_ERROR")))
            stopped = True
            continue
        try:
            prepared = sample_preparer(sample, raw, loaded_baseline)
        except Exception:
            rows.append(_failure_row(sample.sample_alias, "SAMPLE_PROCESSING_ERROR"))
            stopped = True
            continue
        if prepared.sample_alias != sample.sample_alias:
            rows.append(_failure_row(sample.sample_alias, "SAMPLE_ALIAS_MISMATCH"))
            stopped = True
            continue
        if not prepared.direct_match.direct_normal and prepared.preparation.preparation_status == "READY_FOR_LLM" and deepseek_count >= limits["deepseek"]:
            rows.append(_failure_row(sample.sample_alias, "DEEPSEEK_REQUEST_LIMIT_REACHED"))
            stopped = True
            continue
        row, used, error = route_normal_pilot_sample(prepared, deepseek_runner)
        deepseek_count += used
        rows.append(row)
        if error:
            stopped = True
    return NormalPilotArtifacts(
        manifest,
        build_normal_pilot_report(
            manifest, baseline_summary, tuple(rows), rpc_count, deepseek_count,
            "STOPPED" if stopped else "COMPLETED",
        ),
    )


def build_normal_pilot_report(
    manifest: LockedNormalManifest,
    baseline_summary: Mapping[str, Any] | None,
    sample_results: tuple[Mapping[str, Any], ...],
    rpc_request_count: int,
    deepseek_request_count: int,
    run_status: str,
) -> dict[str, Any]:
    limits = _request_limits(manifest)
    if not 0 <= rpc_request_count <= limits["rpc"] or not 0 <= deepseek_request_count <= limits["deepseek"]:
        raise ValueError("NORMAL_PILOT_REQUEST_COUNT_INVALID")
    rows = sorted((dict(item) for item in sample_results), key=lambda item: item["sample_alias"])
    if [item["sample_alias"] for item in rows] != [item.sample_alias for item in manifest.samples]:
        raise ValueError("NORMAL_PILOT_REPORT_SAMPLE_SET_INVALID")
    report = {
        "schema_version": SCHEMA_VERSION,
        "pilot_kind": PILOT_KIND,
        "scientific_generalization_result": False,
        "selection_rule": manifest.selection_rule,
        "selection_mode": manifest.selection_mode,
        "selection_seed": manifest.selection_seed,
        "selection_algorithm": manifest.selection_algorithm,
        "locked_manifest_sha256": manifest.manifest_sha256,
        "run_status": run_status,
        "formal_normal_baseline_summary": dict(baseline_summary) if baseline_summary is not None else None,
        "sample_results": rows,
        "summary": {
            "usable_clean_exact_match_transaction_count": sum(bool(item.get("direct_normal")) for item in rows),
            "early_normal_transaction_count": sum(item.get("final_route") == "EARLY_NORMAL" for item in rows),
            "deepseek_called_transaction_count": sum(bool(item.get("llm_called")) for item in rows),
            "phase2_transaction_count": sum(item.get("final_route") == "PROCEED_PHASE2" for item in rows),
            "self_reference_match_transaction_count": sum(bool(item.get("self_reference_match")) for item in rows),
        },
        "external_call_counts": {"rpc": rpc_request_count, "deepseek": deepseek_request_count, "network": rpc_request_count + deepseek_request_count},
        "request_limits": limits,
    }
    validate_normal_pilot_report(report)
    return report


def serialize_locked_normal_manifest(manifest: LockedNormalManifest) -> str:
    return json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_phase1_normal_pilot_report(report: Mapping[str, Any]) -> str:
    validate_normal_pilot_report(report)
    return json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def validate_normal_pilot_report(report: Mapping[str, Any]) -> None:
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in _FORBIDDEN_REPORT_KEYS:
                    raise ValueError("NORMAL_PILOT_REPORT_FORBIDDEN_FIELD")
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            if _RAW_TX_HASH.search(value) or _RAW_ADDRESS.search(value):
                raise ValueError("NORMAL_PILOT_REPORT_RAW_IDENTIFIER_FORBIDDEN")
            if any(token in value.lower() for token in ("reasoning_content", "calldata", "revertreason", "api key", "rpc_url")):
                raise ValueError("NORMAL_PILOT_REPORT_SENSITIVE_CONTENT_FORBIDDEN")
    visit(report)


def write_phase1_normal_pilot_artifacts(output_dir: str | Path, artifacts: NormalPilotArtifacts) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "locked_normal_manifest.json"
    report_path = root / "phase1_normal_pilot_report.json"
    _atomic_write(manifest_path, serialize_locked_normal_manifest(artifacts.manifest))
    _atomic_write(report_path, serialize_phase1_normal_pilot_report(artifacts.report))
    return manifest_path, report_path


def _atomic_write(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _failure_row(sample_alias: str, code: str, *, base: Mapping[str, Any] | None = None, llm_called: bool = False) -> dict[str, Any]:
    return {
        **(dict(base) if base is not None else {
            "sample_alias": sample_alias, "stage_counts": {}, "coverage_rate": None,
            "critical_missing_count": None, "result_candidate_count": 0,
            "clean_ready_chain_count": 0, "exact_matched_clean_chain_count": 0,
            "self_origin_exact_matched_clean_chain_count": 0,
            "clean_unmatched_chain_count": 0, "uncertain_chain_count": 0,
            "unusable_ready_chain_count": 0, "chain_matches": [],
            "direct_normal": False, "direct_normal_reason": None, "self_reference_match": False,
            "query_normal_identity_status": "UNRESOLVED",
            "known_transaction_direct_normal": False,
            "structural_exact_collision": False,
            "decision_kind": "CONTINUE_STANDARD_PHASE1",
        }),
        "preparation_status": "PROCESSING_FAILED",
        "eligibility": "INELIGIBLE",
        "llm_called": llm_called,
        "deepseek_response_status": "CLIENT_ERROR" if llm_called else "NOT_REQUESTED",
        "deepseek_usage": None,
        "final_route": "PROCEED_PHASE2",
        "correctly_identified_normal": False,
        "reason_codes": [_fixed_reason(code)],
    }


def _not_run_row(sample_alias: str, code: str = "LIVE_MODE_REQUIRED") -> dict[str, Any]:
    row = _failure_row(sample_alias, code)
    row["preparation_status"] = "NOT_RUN"
    return row


def _validate_formal_baseline(loaded: LoadedFormalNormalBaseline) -> dict[str, Any]:
    if not isinstance(loaded, LoadedFormalNormalBaseline):
        raise ValueError("FORMAL_BASELINE_INVALID")
    summary = loaded.summary
    if summary.formal_min_group_size != 3 or not summary.manifest_consistency_passed or not summary.reconstructed_baseline_match or not summary.ethereum_only:
        raise ValueError("FORMAL_BASELINE_SAFETY_CHECK_FAILED")
    if loaded.baseline.min_group_size != 3 or any(group.main_anchor.blockchain_chain_id != 1 for group in loaded.baseline.groups):
        raise ValueError("FORMAL_BASELINE_CHAIN_SAFETY_CHECK_FAILED")
    return summary.to_dict()


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(code) from None


def _load_live_chain_config() -> Mapping[str, Any]:
    from src.cloc_v2.config_loader import load_chain_config
    return load_chain_config("eth")


def _reason_codes(values: Any) -> tuple[str, ...]:
    return tuple(sorted({_fixed_reason(value) for value in values}))


def _fixed_reason(value: Any) -> str:
    return value if isinstance(value, str) and _MACHINE_CODE.fullmatch(value) else "NON_MACHINE_READABLE_REASON"


def _exception_code(exc: Exception, fallback: str) -> str:
    if len(exc.args) == 1 and isinstance(exc.args[0], str) and _MACHINE_CODE.fullmatch(exc.args[0]):
        return exc.args[0]
    return fallback


def _parse_chain_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]+", value):
        return int(value, 16)
    return None


def _distance_text(value: Any) -> str | None:
    return format(value, "f") if value is not None else None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request_limits(manifest: LockedNormalManifest) -> dict[str, int]:
    """One chain-id request and at most one Trace/LLM request per locked sample."""
    return {"rpc": 1 + len(manifest.samples), "deepseek": len(manifest.samples)}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection-mode", choices=(DEFAULT_SELECTION_MODE, RANDOM10_SELECTION_MODE), default=DEFAULT_SELECTION_MODE)
    parser.add_argument("--selection-seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.live:
        print("live_required=true")
        print("rpc_request_count=0")
        print("deepseek_request_count=0")
        return 2
    try:
        if args.selection_mode == RANDOM10_SELECTION_MODE and args.selection_seed is None:
            raise ValueError("NORMAL_PILOT_RANDOM_SELECTION_SEED_REQUIRED")
        manifest = build_locked_normal_manifest(
            DEFAULT_FORMAL_NORMAL_BASELINE_DIR,
            selection_mode=args.selection_mode,
            selection_seed=args.selection_seed,
        )
        loaded = load_formal_normal_baseline(DEFAULT_FORMAL_NORMAL_BASELINE_DIR)
        config = _load_live_chain_config()
        rpc_url = config.get("rpc_url")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not isinstance(rpc_url, str) or not rpc_url or not isinstance(api_key, str) or not api_key:
            raise ValueError("LIVE_CONFIGURATION_MISSING")
        rpc_client = JsonRpcClient(rpc_url, max_requests=_request_limits(manifest)["rpc"])
        deepseek_config = Phase1DeepSeekConfig()

        def deepseek_runner(preparation: Phase1NormalityPreparation) -> Phase1DeepSeekResult:
            return execute_phase1_deepseek_review(
                preparation, deepseek_config, DeepSeekHTTPSOneShotTransport(), api_key
            )

        artifacts = run_phase1_normal_pilot(
            manifest,
            live=True,
            rpc_client=rpc_client,
            deepseek_runner=deepseek_runner,
            loaded_baseline=loaded,
        )
        manifest_path, report_path = write_phase1_normal_pilot_artifacts(args.output_dir, artifacts)
    except Exception as exc:
        print(f"pilot_status={_exception_code(exc, 'NORMAL_PILOT_EXECUTION_ERROR')}")
        return 1
    print(f"pilot_status={artifacts.report['run_status']}")
    print(f"manifest_path={manifest_path}")
    print(f"report_path={report_path}")
    print(f"rpc_request_count={artifacts.report['external_call_counts']['rpc']}")
    print(f"deepseek_request_count={artifacts.report['external_call_counts']['deepseek']}")
    return 0 if artifacts.report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR", "LockedNormalManifest", "LockedNormalSample", "NormalPilotArtifacts",
    "PreparedNormalPilotEvidence", "build_locked_normal_manifest", "build_normal_pilot_report",
    "main", "prepare_normal_pilot_sample", "route_normal_pilot_sample", "run_phase1_normal_pilot",
    "serialize_locked_normal_manifest", "serialize_phase1_normal_pilot_report",
    "validate_normal_pilot_report", "write_phase1_normal_pilot_artifacts",
]

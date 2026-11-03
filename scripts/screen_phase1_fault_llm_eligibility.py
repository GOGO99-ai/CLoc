"""Bounded Ethereum fault screening for formal phase-1 LLM eligibility.

The module is offline by default.  ``--live`` enables at most one chain-id and
ten callTracer requests.  It never reads a DeepSeek key and never calls an LLM.
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
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase1_balanced20_pilot import _fault_pool, _normal_pool
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
from src.cloc_v2.phase1_direct_normal import Phase1DirectNormalMatch, evaluate_phase1_direct_normal_match
from src.cloc_v2.phase1_normality_bridge import Phase1NormalityPreparation, prepare_phase1_normality_review
from src.cloc_v2.phase1_normality_prompt import serialize_phase1_normality_prompt
from src.cloc_v2.result_detection import detect_result_candidates
from src.cloc_v2.runtime_semantics import build_runtime_annotations
from src.cloc_v2.strict_result_chains import build_strict_result_chains
from src.cloc_v2.transaction_graph import build_transaction_graph


SCHEMA_VERSION = "phase1_fault_llm_eligibility_screen_v2"
MANIFEST_SCHEMA_VERSION = "phase1_fault_llm_eligibility_manifest_v1"
CHECKPOINT_SCHEMA_VERSION = "phase1_fault_llm_eligibility_checkpoint_v1"
SCREEN_KIND = "TARGETED_FAULT_LLM_ELIGIBILITY_ENGINEERING_CHECK"
SELECTION_SEED = 20260822
SAMPLE_COUNT = 10
HISTORICAL_COUNT = 5
NEW_COUNT = 5
MAX_RPC_REQUESTS = 11
MAX_DEEPSEEK_REQUESTS = 0
MAX_READY_CANDIDATES = 3
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "phase1_fault_llm_eligibility_screen_seed_20260822"
BALANCED_ROOT = PROJECT_ROOT / "data" / "prototypes" / "phase1_balanced20_seed_20260814"
PRIOR_FAULT_ROOT = PROJECT_ROOT / "data" / "prototypes" / "phase1_real_pilot"

HISTORICAL_RECHECK_HASHES = tuple(sorted((
    "0xa9a1b8ea288eb9ad315088f17f7c7386b9989c95b4d13c81b69d5ddad7ffe61e",
    "0xe7e0474793aad11875c131ebd7582c8b73499dd3c5a473b59e6762d4e373d7b8",
    "0x1149b8285bedd6e858c225d0cd187818e66ae12a0dd4de8de98ea132e3d1470c",
    "0x605488dd5595e9e29a085ca26d7c377dded5fa96099cc33b6193c1b00effed23",
    "0x0163856aaa678676b22e072186c1bdc55ed44a5a0e2cf44500362800663ddce9",
)))

_TX_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_TX_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_MACHINE_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_NETWORK_CODES = frozenset({
    "RPC_NETWORK_ERROR", "RPC_TIMEOUT", "RPC_UNAUTHORIZED", "RPC_QUOTA_EXCEEDED",
    "RPC_HTTP_ERROR", "RPC_INVALID_RESPONSE", "RPC_TRACE_ERROR", "CHAIN_ID_MISMATCH",
    "CALLTRACER_INPUT_INVALID", "SAMPLE_PROCESSING_ERROR",
})
_FORBIDDEN_KEYS = frozenset({
    "tx_hash", "transaction_hash", "address", "runtime_address", "effective_address",
    "input", "calldata", "output", "error", "revertreason", "api_key", "rpc_url",
    "system_message", "user_message", "prompt", "provider_response", "reasoning_content",
    "record_id", "normal_transaction_hash", "cause", "location", "ground_truth", "label",
})


@dataclass(frozen=True)
class FaultScreenSample:
    sample_alias: str
    source_group: str
    blockchain_chain_id: int
    tx_hash: str = field(repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"fault_llm_screen_eth_0(?:0[1-9]|10)", self.sample_alias):
            raise ValueError("FAULT_SCREEN_ALIAS_INVALID")
        if self.source_group not in {"HISTORICAL_RECHECK", "NEW_DETERMINISTIC"}:
            raise ValueError("FAULT_SCREEN_SOURCE_GROUP_INVALID")
        if self.blockchain_chain_id != 1:
            raise ValueError("FAULT_SCREEN_CHAIN_ID_INVALID")
        normalized = self.tx_hash.lower() if isinstance(self.tx_hash, str) else self.tx_hash
        if not isinstance(normalized, str) or _TX_HASH.fullmatch(normalized) is None:
            raise ValueError("FAULT_SCREEN_HASH_INVALID")
        object.__setattr__(self, "tx_hash", normalized)

    @property
    def source_sample_id(self) -> str:
        return self.sample_alias

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "sample_alias": self.sample_alias,
            "source_group": self.source_group,
            "blockchain_chain_id": self.blockchain_chain_id,
            "tx_hash": self.tx_hash,
        }


@dataclass(frozen=True)
class FaultScreenManifest:
    samples: tuple[FaultScreenSample, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if len(samples) != SAMPLE_COUNT or len({item.tx_hash for item in samples}) != SAMPLE_COUNT:
            raise ValueError("FAULT_SCREEN_SAMPLE_SET_INVALID")
        aliases = tuple(f"fault_llm_screen_eth_{index:03d}" for index in range(1, SAMPLE_COUNT + 1))
        if tuple(item.sample_alias for item in samples) != aliases:
            raise ValueError("FAULT_SCREEN_SAMPLE_ORDER_INVALID")
        if sum(item.source_group == "HISTORICAL_RECHECK" for item in samples) != HISTORICAL_COUNT:
            raise ValueError("FAULT_SCREEN_HISTORICAL_COUNT_INVALID")
        if tuple(item.tx_hash for item in samples[:HISTORICAL_COUNT]) != HISTORICAL_RECHECK_HASHES:
            raise ValueError("FAULT_SCREEN_HISTORICAL_SET_INVALID")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("FAULT_SCREEN_MANIFEST_DIGEST_INVALID")
        object.__setattr__(self, "samples", samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "screen_kind": SCREEN_KIND,
            "scientific_generalization_result": False,
            "selection_seed": SELECTION_SEED,
            "new_selection_algorithm": 'SHA256("20260822|" + lowercase_transaction_hash)',
            "sample_count": len(self.samples),
            "samples": [item.to_manifest_dict() for item in self.samples],
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class PreparedFaultScreenEvidence:
    sample_alias: str
    stage_counts: Mapping[str, int]
    coverage_rate: str | None
    critical_missing_count: int | None
    direct_match: Phase1DirectNormalMatch
    preparation: Phase1NormalityPreparation = field(repr=False)

    def __post_init__(self) -> None:
        counts = dict(self.stage_counts)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
            raise ValueError("FAULT_SCREEN_STAGE_COUNTS_INVALID")
        if not isinstance(self.direct_match, Phase1DirectNormalMatch):
            raise ValueError("FAULT_SCREEN_DIRECT_MATCH_INVALID")
        if not isinstance(self.preparation, Phase1NormalityPreparation):
            raise ValueError("FAULT_SCREEN_PREPARATION_INVALID")
        object.__setattr__(self, "stage_counts", MappingProxyType(dict(sorted(counts.items()))))


@dataclass(frozen=True)
class FaultScreenRunResult:
    manifest: FaultScreenManifest = field(repr=False)
    run_status: str
    rows: tuple[Mapping[str, Any], ...]
    rpc_request_count: int
    deepseek_request_count: int
    stop_code: str | None


def build_fault_screen_manifest(
    *,
    dataset_directory: str | Path = DEFAULT_DATASET_DIR,
    formal_directory: str | Path = DEFAULT_FORMAL_NORMAL_BASELINE_DIR,
    balanced_root: str | Path = BALANCED_ROOT,
    prior_fault_root: str | Path = PRIOR_FAULT_ROOT,
) -> FaultScreenManifest:
    """Lock five audited and five deterministic new faults before any RPC."""
    audited = _audited_fault_hashes(Path(balanced_root), Path(prior_fault_root))
    if len(audited) != 13 or not set(HISTORICAL_RECHECK_HASHES) <= audited:
        raise ValueError("FAULT_SCREEN_AUDITED_SET_DRIFT")
    normal_hashes = _normal_pool(Path(formal_directory), 470)
    fault_hashes = _fault_pool(Path(dataset_directory))
    candidates = fault_hashes - audited - normal_hashes
    if len(candidates) < NEW_COUNT:
        raise ValueError("FAULT_SCREEN_NEW_POOL_INSUFFICIENT")
    ranked = sorted(candidates, key=lambda tx_hash: (_rank_digest(tx_hash), tx_hash))[:NEW_COUNT]
    rows = [(tx_hash, "HISTORICAL_RECHECK") for tx_hash in HISTORICAL_RECHECK_HASHES]
    rows.extend((tx_hash, "NEW_DETERMINISTIC") for tx_hash in ranked)
    samples = tuple(
        FaultScreenSample(f"fault_llm_screen_eth_{index:03d}", source, 1, tx_hash)
        for index, (tx_hash, source) in enumerate(rows, start=1)
    )
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "screen_kind": SCREEN_KIND,
        "selection_seed": SELECTION_SEED,
        "samples": [item.to_manifest_dict() for item in samples],
    }
    return FaultScreenManifest(samples, _sha256(_canonical_json(payload)))


def prepare_fault_screen_sample(
    sample: FaultScreenSample,
    raw_calltracer_frame: Any,
    loaded: LoadedFormalNormalBaseline,
) -> PreparedFaultScreenEvidence:
    """Run the formal evidence pipeline and compute eligibility without an LLM."""
    _validate_formal_baseline(loaded)
    tree = adapt_calltracer_call_tree(raw_calltracer_frame)
    graph = build_transaction_graph(tree)
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
    features = extract_normal_features(
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
    direct = evaluate_phase1_direct_normal_match(
        strict,
        features,
        loaded.baseline,
        query_identity,
        loaded.normal_record_transaction_ids,
    )
    preparation = prepare_phase1_normality_review(
        strict,
        features,
        loaded.baseline,
        loaded.normal_record_transaction_ids,
    )
    coverage = strict.coverage
    complete_facts = sum(item.status == "COMPLETE" for item in ledger.transfer_facts)
    counts = {
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
        "clean_record_count": features.clean_record_count,
        "uncertain_record_count": features.uncertain_record_count,
        "rejected_record_count": features.rejected_chain_count,
    }
    return PreparedFaultScreenEvidence(
        sample.sample_alias,
        counts,
        str(coverage.get("coverage_rate")) if coverage.get("coverage_rate") is not None else None,
        coverage.get("critical_missing_count") if isinstance(coverage.get("critical_missing_count"), int) else None,
        direct,
        preparation,
    )


def classify_fault_screen_sample(prepared: PreparedFaultScreenEvidence) -> dict[str, Any]:
    """Create one sanitized result row; this function has no truth or LLM input."""
    preparation = prepared.preparation
    chains = _chain_evidence(preparation)
    usable_clean_count = preparation.eligibility.usable_clean_chain_count
    missing_chain_evidence_count = max(0, usable_clean_count - len(chains))
    if prepared.direct_match.direct_normal:
        outcome = "FAULT_DIRECT_NORMAL_FALSE_RELEASE"
        final_status = "EARLY_NORMAL"
        reasons = prepared.direct_match.reason_codes
    elif prepared.direct_match.structural_exact_collision:
        outcome = "STRUCTURAL_EXACT_COLLISION_PHASE2"
        final_status = "PROCEED_PHASE2"
        reasons = prepared.direct_match.reason_codes
    elif preparation.preparation_status == "READY_FOR_LLM":
        outcome = "READY_FOR_LLM"
        final_status = "AWAITING_LLM_NOT_CALLED"
        reasons = preparation.eligibility.reason_codes
    else:
        outcome = "PROCEED_PHASE2"
        final_status = "PROCEED_PHASE2"
        reasons = (
            *prepared.direct_match.reason_codes,
            *preparation.eligibility.reason_codes,
            *(issue.code for issue in preparation.issues),
        )
    prompt_sha256 = None
    prompt_bytes = 0
    if preparation.prompt is not None:
        prompt_text = serialize_phase1_normality_prompt(preparation.prompt)
        prompt_sha256 = _sha256(prompt_text)
        prompt_bytes = len(prompt_text.encode("utf-8"))
    return {
        "sample_alias": prepared.sample_alias,
        "stage_counts": dict(prepared.stage_counts),
        "coverage_rate": prepared.coverage_rate,
        "critical_missing_count": prepared.critical_missing_count,
        "direct_normal": prepared.direct_match.direct_normal,
        "query_normal_identity_status": prepared.direct_match.query_normal_identity_status,
        "known_transaction_direct_normal": prepared.direct_match.known_transaction_direct_normal,
        "structural_exact_collision": prepared.direct_match.structural_exact_collision,
        "decision_kind": prepared.direct_match.decision_kind,
        "preparation_status": preparation.preparation_status,
        "eligibility": preparation.eligibility.eligibility,
        "total_strict_chain_count": preparation.eligibility.total_strict_chain_count,
        "usable_clean_chain_count": preparation.eligibility.usable_clean_chain_count,
        "excluded_uncertain_chain_count": preparation.eligibility.excluded_uncertain_chain_count,
        "chain_evidence": chains,
        "clean_chain_evidence_complete": missing_chain_evidence_count == 0,
        "missing_clean_chain_evidence_count": missing_chain_evidence_count,
        "prompt_sha256": prompt_sha256,
        "prompt_byte_count": prompt_bytes,
        "engineering_outcome": outcome,
        "final_status": final_status,
        "reason_codes": list(_reason_codes(reasons)),
        "deepseek_called": False,
    }


def run_fault_screen(
    manifest: FaultScreenManifest,
    *,
    live: bool = False,
    rpc_client: Any | None = None,
    loaded_baseline: LoadedFormalNormalBaseline | None = None,
    sample_preparer: Callable[[FaultScreenSample, Any, LoadedFormalNormalBaseline], PreparedFaultScreenEvidence] = prepare_fault_screen_sample,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> FaultScreenRunResult:
    """Screen until three eligible candidates, an external error, or ten traces."""
    if not isinstance(manifest, FaultScreenManifest):
        raise ValueError("FAULT_SCREEN_MANIFEST_REQUIRED")
    if not live:
        rows = tuple(_not_processed(item, "LIVE_MODE_REQUIRED") for item in manifest.samples)
        return FaultScreenRunResult(manifest, "LIVE_MODE_REQUIRED", rows, 0, 0, "LIVE_MODE_REQUIRED")
    if rpc_client is None or not callable(getattr(rpc_client, "call", None)):
        raise ValueError("FAULT_SCREEN_RPC_CLIENT_REQUIRED")
    if loaded_baseline is None:
        raise ValueError("FAULT_SCREEN_BASELINE_REQUIRED")
    _validate_formal_baseline(loaded_baseline)
    rows: list[dict[str, Any]] = []
    rpc_count = 1
    ready_count = 0
    stop_code: str | None = None
    try:
        chain_id = rpc_client.call("eth_chainId", [])
    except Exception as exc:
        stop_code = _exception_code(exc, "RPC_CHAIN_ID_ERROR")
    else:
        if _parse_chain_id(chain_id) != 1:
            stop_code = "CHAIN_ID_MISMATCH"

    for sample in manifest.samples:
        if stop_code is not None:
            rows.append(_not_processed(sample, "NOT_PROCESSED_EXTERNAL_STOP"))
            continue
        if ready_count >= MAX_READY_CANDIDATES:
            rows.append(_not_processed(sample, "NOT_PROCESSED_EARLY_STOP"))
            continue
        try:
            if rpc_count >= MAX_RPC_REQUESTS:
                raise RuntimeError("RPC_REQUEST_LIMIT_REACHED")
            rpc_count += 1
            raw = rpc_client.call("debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}])
        except Exception as exc:
            stop_code = _exception_code(exc, "RPC_TRACE_ERROR")
            rows.append(_processing_failure(sample, stop_code))
        else:
            try:
                prepared = sample_preparer(sample, raw, loaded_baseline)
                if prepared.sample_alias != sample.sample_alias:
                    raise ValueError("SAMPLE_ALIAS_MISMATCH")
                row = classify_fault_screen_sample(prepared)
            except Exception as exc:
                stop_code = _exception_code(exc, "SAMPLE_PROCESSING_ERROR")
                row = _processing_failure(sample, stop_code)
            rows.append(row)
            if row["engineering_outcome"] == "READY_FOR_LLM":
                ready_count += 1
        if progress_callback is not None:
            progress_callback(build_checkpoint(manifest, tuple(rows), rpc_count, stop_code))

    status = "STOPPED" if stop_code is not None else (
        "EARLY_STOP_READY_CANDIDATES_FOUND" if ready_count >= MAX_READY_CANDIDATES else "COMPLETED"
    )
    return FaultScreenRunResult(manifest, status, tuple(rows), rpc_count, 0, stop_code)


def build_checkpoint(
    manifest: FaultScreenManifest,
    rows: tuple[Mapping[str, Any], ...],
    rpc_request_count: int,
    stop_code: str | None,
) -> dict[str, Any]:
    document = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "locked_manifest_sha256": manifest.manifest_sha256,
        "processed_sample_aliases": [
            row["sample_alias"] for row in rows if not row["engineering_outcome"].startswith("NOT_PROCESSED")
        ],
        "sample_results": [dict(row) for row in rows],
        "rpc_request_count": rpc_request_count,
        "deepseek_request_count": 0,
        "stop_code": stop_code,
    }
    validate_sanitized_document(document)
    return document


def build_report(result: FaultScreenRunResult, baseline_summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = sorted((dict(row) for row in result.rows), key=lambda row: _alias_order(row["sample_alias"]))
    if [row["sample_alias"] for row in rows] != [item.sample_alias for item in result.manifest.samples]:
        raise ValueError("FAULT_SCREEN_REPORT_SAMPLE_SET_INVALID")
    processed = [row for row in rows if not row["engineering_outcome"].startswith("NOT_PROCESSED")]
    report = {
        "schema_version": SCHEMA_VERSION,
        "screen_kind": SCREEN_KIND,
        "scientific_generalization_result": False,
        "selection_seed": SELECTION_SEED,
        "locked_manifest_sha256": result.manifest.manifest_sha256,
        "run_status": result.run_status,
        "stop_code": result.stop_code,
        "formal_normal_baseline_summary": dict(baseline_summary),
        "sample_results": rows,
        "summary": {
            "processed_count": len(processed),
            "usable_clean_chain_transaction_count": sum(row.get("usable_clean_chain_count", 0) > 0 for row in processed),
            "all_clean_chains_pass_transaction_count": sum(
                row.get("usable_clean_chain_count", 0) > 0
                and row.get("clean_chain_evidence_complete") is True
                and all(chain["quantitative_status"] == "PASS" for chain in row["chain_evidence"])
                for row in processed
            ),
            "ready_for_llm_count": sum(row["engineering_outcome"] == "READY_FOR_LLM" for row in processed),
            "fault_direct_normal_false_release_count": sum(
                row["engineering_outcome"] == "FAULT_DIRECT_NORMAL_FALSE_RELEASE" for row in processed
            ),
            "phase2_count": sum(
                row["engineering_outcome"] in {
                    "PROCEED_PHASE2", "STRUCTURAL_EXACT_COLLISION_PHASE2"
                }
                for row in processed
            ),
            "not_processed_count": len(rows) - len(processed),
            "ready_candidate_aliases": [
                row["sample_alias"] for row in processed if row["engineering_outcome"] == "READY_FOR_LLM"
            ],
        },
        "request_limits": {"rpc": MAX_RPC_REQUESTS, "deepseek": 0},
        "external_call_counts": {
            "rpc": result.rpc_request_count,
            "deepseek": result.deepseek_request_count,
            "network": result.rpc_request_count,
        },
        "safety": {
            "retry_count": 0,
            "endpoint_switch_count": 0,
            "sample_replacement_count": 0,
            "formal_baseline_write_count": 0,
            "label_used_during_method": False,
        },
    }
    validate_sanitized_document(report)
    return report


def serialize_manifest(manifest: FaultScreenManifest) -> str:
    return json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_checkpoint(checkpoint: Mapping[str, Any]) -> str:
    validate_sanitized_document(checkpoint)
    return json.dumps(dict(checkpoint), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_report(report: Mapping[str, Any]) -> str:
    validate_sanitized_document(report)
    return json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_manifest(output_dir: str | Path, manifest: FaultScreenManifest) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "locked_manifest.json"
    _atomic_write(path, serialize_manifest(manifest))
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
    path = root / "eligibility_screen_report.json"
    _atomic_write(path, serialize_report(report))
    return path


def validate_sanitized_document(value: Mapping[str, Any]) -> None:
    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).lower() in _FORBIDDEN_KEYS:
                    raise ValueError("FAULT_SCREEN_FORBIDDEN_FIELD")
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            if _RAW_TX_HASH.search(item) or _RAW_ADDRESS.search(item):
                raise ValueError("FAULT_SCREEN_RAW_IDENTIFIER_FORBIDDEN")
            lowered = item.lower()
            if any(token in lowered for token in (
                "reasoning_content", "calldata", "revertreason", "rpc_url", "api key", "bearer "
            )):
                raise ValueError("FAULT_SCREEN_SENSITIVE_CONTENT_FORBIDDEN")
    visit(value)


def _chain_evidence(preparation: Phase1NormalityPreparation) -> list[dict[str, Any]]:
    gate_evidence = preparation.gate_evidence
    if gate_evidence is None:
        return []
    gates = {item.query_chain_alias: item for item in gate_evidence.strict_chains}
    rows: list[dict[str, Any]] = []
    for query in gate_evidence.query_chains:
        gate = gates.get(query.chain_alias)
        nearest = query.normal_references[0] if query.normal_references else None
        chain_distance = nearest.chain_distance if nearest is not None else query.nearest_distance
        threshold = query.dynamic_threshold
        rows.append({
            "chain_alias": query.chain_alias,
            "strict_chain_status": gate.strict_chain_status if gate is not None else "READY",
            "feature_extraction_status": "EXTRACTED_CLEAN",
            "call_distance": _distance_text(nearest.call_distance if nearest is not None else None),
            "asset_distance": _distance_text(nearest.asset_distance if nearest is not None else None),
            "chain_distance": _distance_text(chain_distance),
            "nearest_distance": _distance_text(query.nearest_distance),
            "dynamic_threshold": _distance_text(threshold),
            "quantitative_status": query.quantitative_status,
            "normal_group_size": gate.normal_group_size if gate is not None else 0,
            "top3_count": len(query.normal_references),
            "top3_distinct_normal_transaction_count": len({
                reference.normal_transaction_alias for reference in query.normal_references
            }),
            "exact_match": chain_distance == Decimal(0),
            "strictly_below_threshold": chain_distance < threshold,
            "equals_threshold": chain_distance == threshold,
        })
    return rows


def _audited_fault_hashes(balanced_root: Path, prior_root: Path) -> set[str]:
    balanced_manifest = _read_json(balanced_root / "locked_inference_manifest.json", "BALANCED_MANIFEST_INVALID")
    truth = _read_json(balanced_root / "evaluation_truth.json", "BALANCED_TRUTH_INVALID")
    prior_manifest = _read_json(prior_root / "locked_manifest.json", "PRIOR_FAULT_MANIFEST_INVALID")
    if not isinstance(balanced_manifest, Mapping) or not isinstance(truth, Mapping) or not isinstance(prior_manifest, Mapping):
        raise ValueError("FAULT_SCREEN_AUDIT_DOCUMENT_INVALID")
    hash_by_alias = {
        row["sample_alias"]: row["tx_hash"].lower()
        for row in balanced_manifest.get("samples", [])
        if isinstance(row, Mapping) and isinstance(row.get("sample_alias"), str) and isinstance(row.get("tx_hash"), str)
    }
    fault_aliases = {
        row["sample_alias"]
        for row in truth.get("samples", [])
        if isinstance(row, Mapping) and row.get("ground_truth") == "FAULT"
    }
    if not fault_aliases <= set(hash_by_alias):
        raise ValueError("FAULT_SCREEN_BALANCED_TRUTH_MISMATCH")
    hashes = {hash_by_alias[alias] for alias in fault_aliases}
    for row in prior_manifest.get("samples", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("tx_hash"), str):
            raise ValueError("FAULT_SCREEN_PRIOR_SAMPLE_INVALID")
        hashes.add(row["tx_hash"].lower())
    if any(_TX_HASH.fullmatch(value) is None for value in hashes):
        raise ValueError("FAULT_SCREEN_AUDITED_HASH_INVALID")
    return hashes


def _rank_digest(tx_hash: str) -> str:
    return _sha256(f"{SELECTION_SEED}|{tx_hash.lower()}")


def _processing_failure(sample: FaultScreenSample, code: str) -> dict[str, Any]:
    return {
        "sample_alias": sample.sample_alias,
        "stage_counts": {},
        "coverage_rate": None,
        "critical_missing_count": None,
        "direct_normal": False,
        "query_normal_identity_status": "UNRESOLVED",
        "known_transaction_direct_normal": False,
        "structural_exact_collision": False,
        "decision_kind": "CONTINUE_STANDARD_PHASE1",
        "preparation_status": "PROCESSING_FAILED",
        "eligibility": "INELIGIBLE",
        "total_strict_chain_count": 0,
        "usable_clean_chain_count": 0,
        "excluded_uncertain_chain_count": 0,
        "chain_evidence": [],
        "clean_chain_evidence_complete": True,
        "missing_clean_chain_evidence_count": 0,
        "prompt_sha256": None,
        "prompt_byte_count": 0,
        "engineering_outcome": "PROCESSING_FAILED",
        "final_status": "STOPPED",
        "reason_codes": [_fixed_code(code)],
        "deepseek_called": False,
    }


def _not_processed(sample: FaultScreenSample, code: str) -> dict[str, Any]:
    row = _processing_failure(sample, code)
    row["preparation_status"] = "NOT_PROCESSED"
    row["engineering_outcome"] = code
    row["final_status"] = "NOT_PROCESSED"
    return row


def _validate_formal_baseline(loaded: LoadedFormalNormalBaseline) -> dict[str, Any]:
    if not isinstance(loaded, LoadedFormalNormalBaseline):
        raise ValueError("FORMAL_BASELINE_INVALID")
    summary = loaded.summary
    if (
        summary.formal_min_group_size != 3
        or not summary.manifest_consistency_passed
        or not summary.reconstructed_baseline_match
        or not summary.ethereum_only
        or loaded.baseline.min_group_size != 3
        or any(group.main_anchor.blockchain_chain_id != 1 for group in loaded.baseline.groups)
    ):
        raise ValueError("FORMAL_BASELINE_SAFETY_CHECK_FAILED")
    return summary.to_dict()


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(code) from None


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
    return tuple(sorted({_fixed_code(value) for value in values}))


def _fixed_code(value: Any) -> str:
    return value if isinstance(value, str) and _MACHINE_CODE.fullmatch(value) else "NON_MACHINE_READABLE_REASON"


def _exception_code(exc: Exception, fallback: str) -> str:
    return _fixed_code(exc.args[0]) if len(exc.args) == 1 else fallback


def _parse_chain_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]+", value):
        return int(value, 16)
    return None


def _distance_text(value: Any) -> str | None:
    return format(value, "f") if value is not None else None


def _alias_order(alias: str) -> int:
    return int(alias.rsplit("_", 1)[1])


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_live_config() -> Mapping[str, Any]:
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
        if (args.output_dir / "eligibility_screen_report.json").exists():
            raise ValueError("FAULT_SCREEN_OUTPUT_ALREADY_COMPLETED")
        manifest = build_fault_screen_manifest()
        manifest_path = write_manifest(args.output_dir, manifest)
        checkpoint_path = write_checkpoint(args.output_dir, build_checkpoint(manifest, (), 0, None))
        loaded = load_formal_normal_baseline(DEFAULT_FORMAL_NORMAL_BASELINE_DIR)
        config = _load_live_config()
        rpc_url = config.get("rpc_url")
        if not isinstance(rpc_url, str) or not rpc_url:
            raise ValueError("RPC_NOT_CONFIGURED")
        client = JsonRpcClient(rpc_url, max_requests=MAX_RPC_REQUESTS)
        result = run_fault_screen(
            manifest,
            live=True,
            rpc_client=client,
            loaded_baseline=loaded,
            progress_callback=lambda checkpoint: write_checkpoint(args.output_dir, checkpoint),
        )
        checkpoint_path = write_checkpoint(
            args.output_dir,
            build_checkpoint(manifest, result.rows, result.rpc_request_count, result.stop_code),
        )
        report = build_report(result, _validate_formal_baseline(loaded))
        report_path = write_report(args.output_dir, report)
    except Exception as exc:
        print(f"screen_status={_exception_code(exc, 'FAULT_SCREEN_EXECUTION_ERROR')}")
        return 1
    print(f"screen_status={result.run_status}")
    print(f"manifest_path={manifest_path}")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"report_path={report_path}")
    print(f"rpc_request_count={result.rpc_request_count}")
    print("deepseek_request_count=0")
    return 0 if result.run_status != "STOPPED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR", "FaultScreenManifest", "FaultScreenRunResult", "FaultScreenSample",
    "HISTORICAL_RECHECK_HASHES", "PreparedFaultScreenEvidence", "build_checkpoint",
    "build_fault_screen_manifest", "build_report", "classify_fault_screen_sample", "main",
    "prepare_fault_screen_sample", "run_fault_screen", "serialize_checkpoint", "serialize_manifest",
    "serialize_report", "validate_sanitized_document", "write_checkpoint", "write_manifest", "write_report",
]

"""Balanced closed-set phase-1 engineering pilot with safe resumability.

The module is offline by default. Selection locks neutral inference samples and
separate evaluation truth before any RPC call. Formal evidence preparation,
direct matching, prompting, DeepSeek validation, and routing are delegated to
the already-tested production modules.
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
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase1_deepseek_smoke import DeepSeekHTTPSOneShotTransport
from scripts.run_phase1_normal_pilot import (
    PreparedNormalPilotEvidence,
    prepare_normal_pilot_sample,
    route_normal_pilot_sample,
)
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.formal_normal_baseline_loader import (
    DEFAULT_FORMAL_NORMAL_BASELINE_DIR,
    LoadedFormalNormalBaseline,
    load_formal_normal_baseline,
)
from src.cloc_v2.phase1_normality_bridge import Phase1NormalityPreparation
from src.cloc_v2.phase1_normality_deepseek import (
    Phase1DeepSeekConfig,
    Phase1DeepSeekResult,
    execute_phase1_deepseek_review,
)


SCHEMA_VERSION = "phase1_balanced20_report_v2"
MANIFEST_SCHEMA_VERSION = "phase1_balanced20_inference_manifest_v1"
TRUTH_SCHEMA_VERSION = "phase1_balanced20_truth_v1"
CHECKPOINT_SCHEMA_VERSION = "phase1_balanced20_checkpoint_v1"
SAMPLE_SCHEMA_VERSION = "phase1_balanced20_sample_v2"
EXPERIMENT_KIND = "BALANCED20_PHASE1_ENGINEERING_EXPERIMENT"
SCIENTIFIC_GENERALIZATION_RESULT = False
SELECTION_SEED = 20260814
SAMPLE_COUNT_PER_CLASS = 10
TOTAL_SAMPLE_COUNT = 20
MAX_RPC_REQUESTS = 21
MAX_DEEPSEEK_REQUESTS = 20
NORMAL_RANK_PREFIX = "CLOC_BALANCED20_NORMAL_V1"
FAULT_RANK_PREFIX = "CLOC_BALANCED20_FAULT_V1"
ORDER_RANK_PREFIX = "CLOC_BALANCED20_ORDER_V1"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "phase1_balanced20_seed_20260814"
DEFAULT_EXCLUDED_NORMAL_MANIFESTS = (
    PROJECT_ROOT / "data" / "prototypes" / "phase1_normal_pilot_usable_evidence" / "locked_normal_manifest.json",
    PROJECT_ROOT / "data" / "prototypes" / "phase1_normal_random10_seed_20260814" / "locked_normal_manifest.json",
)
DEFAULT_EXCLUDED_FAULT_MANIFESTS = (
    PROJECT_ROOT / "data" / "prototypes" / "phase1_real_pilot" / "locked_manifest.json",
)

_TX_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_TX_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_ALIAS = re.compile(r"^eval_eth_(?:0[0-1][0-9]|020)$")
_MACHINE_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_NETWORK_CODES = frozenset({
    "RPC_NETWORK_ERROR", "RPC_TIMEOUT", "RPC_UNAUTHORIZED", "RPC_QUOTA_EXCEEDED",
    "RPC_HTTP_ERROR", "RPC_INVALID_RESPONSE",
})
_FORBIDDEN_RESULT_KEYS = frozenset({
    "tx_hash", "transaction_hash", "label", "ground_truth", "address",
    "runtime_address", "effective_address", "input", "calldata", "output",
    "error", "revertreason", "api_key", "rpc_url", "system_message",
    "user_message", "prompt", "provider_response", "reasoning_content",
    "record_id", "source_sample_id",
})


@dataclass(frozen=True)
class BalancedInferenceSample:
    sample_alias: str
    blockchain_chain_id: int
    tx_hash: str = field(repr=False)

    def __post_init__(self) -> None:
        normalized = self.tx_hash.lower() if isinstance(self.tx_hash, str) else self.tx_hash
        if _ALIAS.fullmatch(self.sample_alias) is None:
            raise ValueError("BALANCED20_ALIAS_INVALID")
        if self.blockchain_chain_id != 1:
            raise ValueError("BALANCED20_CHAIN_ID_INVALID")
        if not isinstance(normalized, str) or _TX_HASH.fullmatch(normalized) is None:
            raise ValueError("BALANCED20_HASH_INVALID")
        object.__setattr__(self, "tx_hash", normalized)

    @property
    def source_sample_id(self) -> str:
        """Neutral internal ID required by the reused pipeline; never serialized."""
        return self.sample_alias

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_alias": self.sample_alias,
            "blockchain_chain_id": self.blockchain_chain_id,
            "tx_hash": self.tx_hash,
        }


@dataclass(frozen=True)
class BalancedInferenceManifest:
    samples: tuple[BalancedInferenceSample, ...]
    manifest_sha256: str
    selection_seed: int = SELECTION_SEED

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if len(samples) != TOTAL_SAMPLE_COUNT or len({item.tx_hash for item in samples}) != TOTAL_SAMPLE_COUNT:
            raise ValueError("BALANCED20_SAMPLE_SET_INVALID")
        expected_aliases = tuple(f"eval_eth_{index:03d}" for index in range(1, TOTAL_SAMPLE_COUNT + 1))
        if tuple(item.sample_alias for item in samples) != expected_aliases:
            raise ValueError("BALANCED20_SAMPLE_ORDER_INVALID")
        if not isinstance(self.selection_seed, int) or isinstance(self.selection_seed, bool):
            raise ValueError("BALANCED20_SEED_INVALID")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("BALANCED20_MANIFEST_DIGEST_INVALID")
        object.__setattr__(self, "samples", samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "selection_seed": self.selection_seed,
            "selection_algorithms": {
                "normal": f'SHA256("{NORMAL_RANK_PREFIX}|seed|lowercase_tx_hash")',
                "fault": f'SHA256("{FAULT_RANK_PREFIX}|seed|lowercase_tx_hash")',
                "order": f'SHA256("{ORDER_RANK_PREFIX}|seed|lowercase_tx_hash")',
            },
            "sample_count": len(self.samples),
            "samples": [item.to_dict() for item in self.samples],
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class BalancedEvaluationTruth:
    labels: Mapping[str, str]
    manifest_sha256: str

    def __post_init__(self) -> None:
        labels = dict(self.labels)
        expected = {f"eval_eth_{index:03d}" for index in range(1, TOTAL_SAMPLE_COUNT + 1)}
        if set(labels) != expected or set(labels.values()) != {"NORMAL", "FAULT"}:
            raise ValueError("BALANCED20_TRUTH_INVALID")
        if sum(value == "NORMAL" for value in labels.values()) != SAMPLE_COUNT_PER_CLASS:
            raise ValueError("BALANCED20_TRUTH_BALANCE_INVALID")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("BALANCED20_TRUTH_MANIFEST_DIGEST_INVALID")
        object.__setattr__(self, "labels", MappingProxyType(dict(sorted(labels.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRUTH_SCHEMA_VERSION,
            "locked_inference_manifest_sha256": self.manifest_sha256,
            "samples": [
                {"sample_alias": alias, "ground_truth": label}
                for alias, label in self.labels.items()
            ],
        }


@dataclass(frozen=True)
class Balanced20Selection:
    inference_manifest: BalancedInferenceManifest
    evaluation_truth: BalancedEvaluationTruth = field(repr=False)
    normal_hashes: tuple[str, ...] = field(repr=False)
    fault_hashes: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class Balanced20RunResult:
    run_status: str
    inference_manifest: BalancedInferenceManifest = field(repr=False)
    sample_results: tuple[Mapping[str, Any], ...]
    rpc_request_count: int
    deepseek_request_count: int
    stop_code: str | None


def build_balanced20_selection(
    formal_directory: str | Path = DEFAULT_FORMAL_NORMAL_BASELINE_DIR,
    dataset_directory: str | Path = DEFAULT_DATASET_DIR,
    *,
    selection_seed: int = SELECTION_SEED,
    excluded_normal_manifests: Sequence[str | Path] = DEFAULT_EXCLUDED_NORMAL_MANIFESTS,
    excluded_fault_manifests: Sequence[str | Path] = DEFAULT_EXCLUDED_FAULT_MANIFESTS,
    expected_succeeded_normal_count: int | None = 470,
) -> Balanced20Selection:
    """Lock 10+10 samples without consulting any evidence or prior result data."""
    if not isinstance(selection_seed, int) or isinstance(selection_seed, bool):
        raise ValueError("BALANCED20_SEED_INVALID")
    normal_pool = _normal_pool(Path(formal_directory), expected_succeeded_normal_count)
    fault_pool = _fault_pool(Path(dataset_directory))
    excluded_normal = _excluded_hashes(excluded_normal_manifests)
    excluded_fault = _excluded_hashes(excluded_fault_manifests)
    normal_candidates = sorted(normal_pool - excluded_normal)
    fault_candidates = sorted(fault_pool - excluded_fault)
    if len(normal_candidates) < SAMPLE_COUNT_PER_CLASS or len(fault_candidates) < SAMPLE_COUNT_PER_CLASS:
        raise ValueError("BALANCED20_CANDIDATE_POOL_INSUFFICIENT")
    normal_hashes = tuple(_ranked(normal_candidates, NORMAL_RANK_PREFIX, selection_seed)[:SAMPLE_COUNT_PER_CLASS])
    fault_hashes = tuple(_ranked(fault_candidates, FAULT_RANK_PREFIX, selection_seed)[:SAMPLE_COUNT_PER_CLASS])
    tagged = [(tx_hash, "NORMAL") for tx_hash in normal_hashes] + [(tx_hash, "FAULT") for tx_hash in fault_hashes]
    ordered = sorted(tagged, key=lambda item: (_rank_digest(ORDER_RANK_PREFIX, selection_seed, item[0]), item[0]))
    samples = tuple(
        BalancedInferenceSample(f"eval_eth_{index:03d}", 1, tx_hash)
        for index, (tx_hash, _) in enumerate(ordered, start=1)
    )
    digest_payload = {
        "selection_seed": selection_seed,
        "samples": [item.to_dict() for item in samples],
    }
    digest = _sha256(_canonical_json(digest_payload))
    manifest = BalancedInferenceManifest(samples, digest, selection_seed)
    labels = {sample.sample_alias: label for sample, (_, label) in zip(samples, ordered)}
    truth = BalancedEvaluationTruth(labels, digest)
    return Balanced20Selection(manifest, truth, normal_hashes, fault_hashes)


def run_balanced20_inference(
    inference_manifest: BalancedInferenceManifest,
    *,
    live: bool = False,
    output_dir: str | Path | None = None,
    rpc_client: Any | None = None,
    deepseek_runner: Callable[[Phase1NormalityPreparation], Phase1DeepSeekResult] | None = None,
    loaded_baseline: LoadedFormalNormalBaseline | None = None,
    sample_preparer: Callable[[BalancedInferenceSample, Any, LoadedFormalNormalBaseline], PreparedNormalPilotEvidence] = prepare_normal_pilot_sample,
    sample_router: Callable[[PreparedNormalPilotEvidence, Callable[..., Any]], tuple[dict[str, Any], int, bool]] = route_normal_pilot_sample,
    retry_failed: bool = False,
) -> Balanced20RunResult:
    """Run neutral inference only; this function has no truth parameter or label access."""
    if not isinstance(inference_manifest, BalancedInferenceManifest):
        raise ValueError("BALANCED20_INFERENCE_MANIFEST_REQUIRED")
    if not live:
        return Balanced20RunResult("LIVE_MODE_REQUIRED", inference_manifest, (), 0, 0, "LIVE_MODE_REQUIRED")
    if output_dir is None:
        raise ValueError("BALANCED20_OUTPUT_DIR_REQUIRED")
    if rpc_client is None or not callable(getattr(rpc_client, "call", None)):
        raise ValueError("BALANCED20_RPC_CLIENT_REQUIRED")
    if deepseek_runner is None or not callable(deepseek_runner):
        raise ValueError("BALANCED20_DEEPSEEK_RUNNER_REQUIRED")
    if not isinstance(loaded_baseline, LoadedFormalNormalBaseline):
        raise ValueError("BALANCED20_BASELINE_REQUIRED")

    root = Path(output_dir)
    checkpoint = _prepare_inference_workspace(root, inference_manifest)
    starting_rpc = _client_request_count(rpc_client)
    local_rpc = 0
    local_deepseek = 0
    stop_code: str | None = None

    def save_checkpoint() -> None:
        checkpoint["rpc_request_count"] += _client_request_count(rpc_client) - starting_rpc - checkpoint.pop("_synced_rpc", 0)
        checkpoint["_synced_rpc"] = _client_request_count(rpc_client) - starting_rpc
        public = {key: value for key, value in checkpoint.items() if not key.startswith("_")}
        _atomic_json(root / "checkpoint.json", public)

    try:
        local_rpc += 1
        chain_id = rpc_client.call("eth_chainId", [])
    except Exception as exc:
        stop_code = _exception_code(exc, "RPC_CHAIN_ID_ERROR")
        checkpoint["last_stop_code"] = stop_code
        save_checkpoint()
        return _run_result(inference_manifest, root, local_rpc, local_deepseek, "STOPPED", stop_code)
    save_checkpoint()
    if _parse_chain_id(chain_id) != 1:
        stop_code = "CHAIN_ID_MISMATCH"
        checkpoint["last_stop_code"] = stop_code
        save_checkpoint()
        return _run_result(inference_manifest, root, local_rpc, local_deepseek, "STOPPED", stop_code)

    guarded_calls = 0

    def guarded_deepseek(preparation: Phase1NormalityPreparation) -> Phase1DeepSeekResult:
        nonlocal guarded_calls
        if guarded_calls >= MAX_DEEPSEEK_REQUESTS:
            raise RuntimeError("DEEPSEEK_REQUEST_LIMIT_REACHED")
        guarded_calls += 1
        return deepseek_runner(preparation)

    for sample in inference_manifest.samples:
        state = checkpoint["samples"][sample.sample_alias]
        if state["status"] == "SUCCEEDED" or (state["status"] == "FAILED" and not retry_failed):
            continue
        if local_rpc >= MAX_RPC_REQUESTS:
            stop_code = "RPC_REQUEST_LIMIT_REACHED"
            break
        state["attempt_count"] += 1
        try:
            local_rpc += 1
            raw_frame = rpc_client.call("debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}])
        except Exception as exc:
            code = _exception_code(exc, "RPC_TRACE_ERROR")
            state.update({"status": "FAILED", "failure_code": code, "artifact": None})
            checkpoint["last_stop_code"] = code if code in _NETWORK_CODES else None
            save_checkpoint()
            if code in _NETWORK_CODES:
                stop_code = code
                break
            continue
        save_checkpoint()
        try:
            prepared = sample_preparer(sample, raw_frame, loaded_baseline)
            if prepared.sample_alias != sample.sample_alias:
                raise ValueError("SAMPLE_ALIAS_MISMATCH")
        except Exception as exc:
            code = _exception_code(exc, "SAMPLE_PROCESSING_ERROR")
            row = _error_result(sample.sample_alias, code)
            _write_sample_result(root, row)
            state.update({"status": "FAILED", "failure_code": code, "artifact": _sample_relative_path(sample.sample_alias)})
            save_checkpoint()
            continue
        before = guarded_calls
        row, _, route_error = sample_router(prepared, guarded_deepseek)
        used = guarded_calls - before
        if used not in {0, 1}:
            raise ValueError("BALANCED20_DEEPSEEK_PER_SAMPLE_LIMIT_INVALID")
        local_deepseek += used
        result = _inference_result(row, route_error)
        _write_sample_result(root, result)
        checkpoint["deepseek_request_count"] += used
        if route_error:
            code = result["reason_codes"][0] if result["reason_codes"] else "DEEPSEEK_PROVIDER_ERROR"
            state.update({"status": "FAILED", "failure_code": code, "artifact": _sample_relative_path(sample.sample_alias)})
            checkpoint["last_stop_code"] = code
            save_checkpoint()
            stop_code = code
            break
        state.update({"status": "SUCCEEDED", "failure_code": None, "artifact": _sample_relative_path(sample.sample_alias)})
        checkpoint["last_stop_code"] = None
        save_checkpoint()

    checkpoint["last_stop_code"] = stop_code
    save_checkpoint()
    complete = all(state["status"] == "SUCCEEDED" for state in checkpoint["samples"].values())
    status = "COMPLETED" if complete else ("STOPPED" if stop_code else "PARTIAL")
    return _run_result(inference_manifest, root, local_rpc, local_deepseek, status, stop_code)


def build_balanced20_evaluation_report(
    run_result: Balanced20RunResult,
    evaluation_truth: BalancedEvaluationTruth,
) -> dict[str, Any]:
    """Join frozen predictions with separate truth only after inference finishes."""
    if run_result.inference_manifest.manifest_sha256 != evaluation_truth.manifest_sha256:
        raise ValueError("BALANCED20_TRUTH_MANIFEST_MISMATCH")
    rows = []
    for result in sorted((dict(item) for item in run_result.sample_results), key=lambda item: item["sample_alias"]):
        alias = result["sample_alias"]
        rows.append({**result, "ground_truth": evaluation_truth.labels[alias]})
    metrics = _metrics(rows, TOTAL_SAMPLE_COUNT)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": EXPERIMENT_KIND,
        "scientific_generalization_result": SCIENTIFIC_GENERALIZATION_RESULT,
        "selection_seed": run_result.inference_manifest.selection_seed,
        "locked_inference_manifest_sha256": run_result.inference_manifest.manifest_sha256,
        "run_status": run_result.run_status,
        "stop_code": run_result.stop_code,
        "sample_results": rows,
        "metrics": metrics,
        "external_call_counts": {
            "rpc": run_result.rpc_request_count,
            "deepseek": run_result.deepseek_request_count,
            "network": run_result.rpc_request_count + run_result.deepseek_request_count,
        },
        "request_limits": {"rpc": MAX_RPC_REQUESTS, "deepseek": MAX_DEEPSEEK_REQUESTS},
    }
    validate_balanced20_report(report)
    return report


def write_balanced20_selection(
    output_dir: str | Path,
    selection: Balanced20Selection,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    inference_path = root / "locked_inference_manifest.json"
    truth_path = root / "evaluation_truth.json"
    _write_or_verify(inference_path, selection.inference_manifest.to_dict(), "BALANCED20_MANIFEST_DRIFT")
    _write_or_verify(truth_path, selection.evaluation_truth.to_dict(), "BALANCED20_TRUTH_DRIFT")
    return inference_path, truth_path


def write_balanced20_report(output_dir: str | Path, report: Mapping[str, Any]) -> Path:
    validate_balanced20_report(report)
    path = Path(output_dir) / "phase1_balanced20_report.json"
    _atomic_json(path, report)
    return path


def serialize_balanced20_report(report: Mapping[str, Any]) -> str:
    validate_balanced20_report(report)
    return json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def validate_balanced20_report(report: Mapping[str, Any]) -> None:
    def visit(value: Any, *, allow_truth: bool = True) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                lowered = str(key).lower()
                if lowered in _FORBIDDEN_RESULT_KEYS and not (allow_truth and lowered == "ground_truth"):
                    raise ValueError("BALANCED20_REPORT_FORBIDDEN_FIELD")
                visit(nested, allow_truth=allow_truth)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested, allow_truth=allow_truth)
        elif isinstance(value, str):
            if _RAW_TX_HASH.search(value) or _RAW_ADDRESS.search(value):
                raise ValueError("BALANCED20_REPORT_RAW_IDENTIFIER_FORBIDDEN")
            lowered = value.lower()
            if any(token in lowered for token in ("reasoning_content", "calldata", "revertreason", "rpc_url", "api key")):
                raise ValueError("BALANCED20_REPORT_SENSITIVE_CONTENT_FORBIDDEN")
    visit(report)


def _normal_pool(root: Path, expected_count: int | None) -> set[str]:
    manifest = _read_json(root / "normal_manifest.json", "NORMAL_MANIFEST_INVALID")
    checkpoint = _read_json(root / "checkpoint.json", "NORMAL_CHECKPOINT_INVALID")
    if not isinstance(manifest, Mapping) or not isinstance(checkpoint, Mapping):
        raise ValueError("NORMAL_SELECTION_DOCUMENT_INVALID")
    if manifest.get("manifest_sha256") != checkpoint.get("manifest_sha256"):
        raise ValueError("NORMAL_SELECTION_MANIFEST_MISMATCH")
    rows = manifest.get("samples")
    states = checkpoint.get("samples")
    if not isinstance(rows, list) or not isinstance(states, Mapping):
        raise ValueError("NORMAL_SELECTION_STRUCTURE_INVALID")
    hashes: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("chain_key") != "eth" or row.get("blockchain_chain_id") != 1:
            continue
        tx_hash = row.get("tx_hash")
        if not isinstance(tx_hash, str) or _TX_HASH.fullmatch(tx_hash.lower()) is None:
            raise ValueError("NORMAL_SELECTION_HASH_INVALID")
        state = states.get(f"eth:{tx_hash.lower()}")
        if isinstance(state, Mapping) and state.get("status") == "SUCCEEDED":
            hashes.add(tx_hash.lower())
    if expected_count is not None and len(hashes) != expected_count:
        raise ValueError("NORMAL_SELECTION_SUCCEEDED_COUNT_DRIFT")
    return hashes


def _fault_pool(root: Path) -> set[str]:
    hashes: set[str] = set()
    for path in sorted(root.rglob("*.json")):
        platform, raw_hashes = _read_fault_fields(path)
        if not _is_ethereum_platform(platform):
            continue
        for tx_hash in _flatten_strings(raw_hashes):
            normalized = tx_hash.lower()
            if _TX_HASH.fullmatch(normalized) is None:
                raise ValueError("FAULT_SELECTION_HASH_INVALID")
            hashes.add(normalized)
    return hashes


def _read_fault_fields(path: Path) -> tuple[Any, Any]:
    """Extract only platform and fault.transaction_hash fragments."""
    text = path.read_text(encoding="utf-8")
    platform_match = re.search(r'"platform"\s*:\s*("(?:\\.|[^"\\])*")', text, re.IGNORECASE)
    platform = json.loads(platform_match.group(1)) if platform_match is not None else None
    fault_match = re.search(r'"fault"\s*:\s*\{', text, re.IGNORECASE)
    if fault_match is None:
        return platform, None
    start = text.find("{", fault_match.start())
    end = _json_container_end(text, start)
    if end is None:
        raise ValueError("FAULT_OBJECT_INVALID")
    fragment = text[start:end]
    hash_match = re.search(r'"transaction_hash"\s*:', fragment, re.IGNORECASE)
    if hash_match is None:
        return platform, None
    value_start = hash_match.end()
    while value_start < len(fragment) and fragment[value_start].isspace():
        value_start += 1
    value_end = _json_value_end(fragment, value_start)
    if value_end is None:
        raise ValueError("FAULT_TRANSACTION_HASH_STRUCTURE_INVALID")
    try:
        return platform, json.loads(fragment[value_start:value_end])
    except json.JSONDecodeError as exc:
        raise ValueError("FAULT_TRANSACTION_HASH_STRUCTURE_INVALID") from exc


def _json_container_end(text: str, start: int) -> int | None:
    if start < 0 or start >= len(text) or text[start] not in "[{":
        return None
    opening, closing = text[start], "}" if text[start] == "{" else "]"
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _json_value_end(text: str, start: int) -> int | None:
    if start >= len(text):
        return None
    if text[start] in "[{":
        return _json_container_end(text, start)
    if text[start] == '"':
        escaped = False
        for index in range(start + 1, len(text)):
            if escaped:
                escaped = False
            elif text[index] == "\\":
                escaped = True
            elif text[index] == '"':
                return index + 1
        return None
    end = start
    while end < len(text) and text[end] not in ",}":
        end += 1
    return end


def _flatten_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for nested in value:
            yield from _flatten_strings(nested)
    elif value is not None:
        raise ValueError("FAULT_TRANSACTION_HASH_STRUCTURE_INVALID")


def _is_ethereum_platform(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"eth", "ethereum"}


def _excluded_hashes(paths: Sequence[str | Path]) -> set[str]:
    hashes: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError("EXCLUSION_MANIFEST_MISSING")
        payload = _read_json(path, "EXCLUSION_MANIFEST_INVALID")
        rows = payload.get("samples") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError("EXCLUSION_MANIFEST_INVALID")
        for row in rows:
            tx_hash = row.get("tx_hash") if isinstance(row, Mapping) else None
            if not isinstance(tx_hash, str) or _TX_HASH.fullmatch(tx_hash.lower()) is None:
                raise ValueError("EXCLUSION_HASH_INVALID")
            hashes.add(tx_hash.lower())
    return hashes


def _ranked(hashes: Sequence[str], prefix: str, seed: int) -> list[str]:
    return sorted(hashes, key=lambda tx_hash: (_rank_digest(prefix, seed, tx_hash), tx_hash))


def _rank_digest(prefix: str, seed: int, tx_hash: str) -> str:
    return _sha256(f"{prefix}|{seed}|{tx_hash.lower()}")


def _prepare_inference_workspace(root: Path, manifest: BalancedInferenceManifest) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "locked_inference_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("BALANCED20_LOCKED_MANIFEST_REQUIRED")
    existing = _read_json(manifest_path, "BALANCED20_LOCKED_MANIFEST_INVALID")
    if existing != manifest.to_dict():
        raise ValueError("BALANCED20_MANIFEST_DRIFT")
    checkpoint_path = root / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = _read_json(checkpoint_path, "BALANCED20_CHECKPOINT_INVALID")
        if checkpoint.get("manifest_sha256") != manifest.manifest_sha256:
            raise ValueError("BALANCED20_MANIFEST_DRIFT")
        if set(checkpoint.get("samples", {})) != {item.sample_alias for item in manifest.samples}:
            raise ValueError("BALANCED20_CHECKPOINT_SAMPLE_SET_INVALID")
        checkpoint["_synced_rpc"] = 0
        return checkpoint
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "manifest_sha256": manifest.manifest_sha256,
        "rpc_request_count": 0,
        "deepseek_request_count": 0,
        "last_stop_code": None,
        "samples": {
            sample.sample_alias: {"status": "PENDING", "attempt_count": 0, "failure_code": None, "artifact": None}
            for sample in manifest.samples
        },
        "_synced_rpc": 0,
    }
    public = {key: value for key, value in checkpoint.items() if not key.startswith("_")}
    _atomic_json(checkpoint_path, public)
    return checkpoint


def _run_result(
    manifest: BalancedInferenceManifest,
    root: Path,
    rpc_count: int,
    deepseek_count: int,
    status: str,
    stop_code: str | None,
) -> Balanced20RunResult:
    rows = []
    for sample in manifest.samples:
        path = root / _sample_relative_path(sample.sample_alias)
        if path.is_file():
            row = _read_json(path, "BALANCED20_SAMPLE_ARTIFACT_INVALID")
            validate_balanced20_report(row)
            rows.append(row)
    return Balanced20RunResult(status, manifest, tuple(rows), rpc_count, deepseek_count, stop_code)


def _inference_result(row: Mapping[str, Any], route_error: bool) -> dict[str, Any]:
    direct = bool(row.get("direct_normal"))
    llm_called = bool(row.get("llm_called"))
    final_route = row.get("final_route")
    reasons = _reason_codes(row.get("reason_codes", ()))
    if route_error:
        conclusion = "ERROR"
    elif direct and final_route == "EARLY_NORMAL":
        conclusion = "NORMAL_DIRECT"
    elif llm_called and final_route == "EARLY_NORMAL":
        conclusion = "NORMAL_LLM"
    elif llm_called and "LLM_SUSPICIOUS" in reasons:
        conclusion = "SUSPICIOUS_LLM"
    else:
        conclusion = "UNCERTAIN_PHASE2"
    sanitized = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_alias": row["sample_alias"],
        "stage_counts": dict(row.get("stage_counts", {})),
        "coverage_rate": row.get("coverage_rate"),
        "critical_missing_count": row.get("critical_missing_count"),
        "clean_ready_chain_count": int(row.get("clean_ready_chain_count", 0)),
        "exact_matched_clean_chain_count": int(row.get("exact_matched_clean_chain_count", 0)),
        "clean_unmatched_chain_count": int(row.get("clean_unmatched_chain_count", 0)),
        "uncertain_chain_count": int(row.get("uncertain_chain_count", 0)),
        "quantitative_status": _quantitative_status(row),
        "direct_normal": direct,
        "query_normal_identity_status": row.get(
            "query_normal_identity_status",
            "KNOWN_NORMAL_ORIGIN" if direct else "UNRESOLVED",
        ),
        "known_transaction_direct_normal": bool(
            row.get("known_transaction_direct_normal", direct)
        ),
        "structural_exact_collision": bool(row.get("structural_exact_collision", False)),
        "decision_kind": row.get(
            "decision_kind",
            "KNOWN_TRANSACTION_DIRECT_NORMAL" if direct else "CONTINUE_STANDARD_PHASE1",
        ),
        "llm_eligible": row.get("preparation_status") == "READY_FOR_LLM",
        "llm_called": llm_called,
        "llm_response_status": row.get("deepseek_response_status", "NOT_REQUESTED"),
        "llm_usage": row.get("deepseek_usage"),
        "phase1_conclusion": conclusion,
        "final_route": final_route if final_route in {"EARLY_NORMAL", "PROCEED_PHASE2"} else "PROCEED_PHASE2",
        "reason_codes": list(reasons),
    }
    validate_balanced20_report(sanitized)
    return sanitized


def _error_result(alias: str, code: str) -> dict[str, Any]:
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_alias": alias,
        "stage_counts": {},
        "coverage_rate": None,
        "critical_missing_count": None,
        "clean_ready_chain_count": 0,
        "exact_matched_clean_chain_count": 0,
        "clean_unmatched_chain_count": 0,
        "uncertain_chain_count": 0,
        "quantitative_status": "UNAVAILABLE",
        "direct_normal": False,
        "query_normal_identity_status": "UNRESOLVED",
        "known_transaction_direct_normal": False,
        "structural_exact_collision": False,
        "decision_kind": "CONTINUE_STANDARD_PHASE1",
        "llm_eligible": False,
        "llm_called": False,
        "llm_response_status": "NOT_REQUESTED",
        "llm_usage": None,
        "phase1_conclusion": "ERROR",
        "final_route": "PROCEED_PHASE2",
        "reason_codes": [_fixed_code(code)],
    }


def _quantitative_status(row: Mapping[str, Any]) -> str:
    if row.get("direct_normal"):
        return "EXACT_MATCH"
    if row.get("preparation_status") == "READY_FOR_LLM":
        return "PASS"
    reasons = set(row.get("reason_codes", ()))
    if "QUANTITATIVE_NOT_PASS" in reasons:
        return "FAIL_OR_UNCERTAIN"
    return "UNAVAILABLE"


def _metrics(rows: Sequence[Mapping[str, Any]], total_selected: int) -> dict[str, Any]:
    normal = [item for item in rows if item["ground_truth"] == "NORMAL"]
    fault = [item for item in rows if item["ground_truth"] == "FAULT"]
    normal_early = sum(item["phase1_conclusion"] in {"NORMAL_DIRECT", "NORMAL_LLM"} for item in normal)
    normal_abstain = sum(item["phase1_conclusion"] == "UNCERTAIN_PHASE2" for item in normal)
    normal_suspicious = sum(item["phase1_conclusion"] == "SUSPICIOUS_LLM" for item in normal)
    fault_normal = sum(item["phase1_conclusion"] in {"NORMAL_DIRECT", "NORMAL_LLM"} for item in fault)
    fault_suspicious = sum(item["phase1_conclusion"] == "SUSPICIOUS_LLM" for item in fault)
    fault_abstain = sum(item["phase1_conclusion"] == "UNCERTAIN_PHASE2" for item in fault)
    determinate = [item for item in rows if item["phase1_conclusion"] in {"NORMAL_DIRECT", "NORMAL_LLM", "SUSPICIOUS_LLM"}]
    correct_determinate = sum(
        (item["ground_truth"] == "NORMAL" and item["phase1_conclusion"] in {"NORMAL_DIRECT", "NORMAL_LLM"})
        or (item["ground_truth"] == "FAULT" and item["phase1_conclusion"] == "SUSPICIOUS_LLM")
        for item in determinate
    )
    return {
        "processed_sample_count": len(rows),
        "normal_early_count": normal_early,
        "normal_early_rate": _ratio(normal_early, len(normal)),
        "normal_safe_abstain_count": normal_abstain,
        "normal_false_suspicious_count": normal_suspicious,
        "fault_false_normal_count": fault_normal,
        "fault_false_release_rate": _ratio(fault_normal, len(fault)),
        "fault_suspicious_count": fault_suspicious,
        "fault_safe_phase2_count": fault_abstain,
        "llm_eligible_count": sum(bool(item["llm_eligible"]) for item in rows),
        "llm_called_count": sum(bool(item["llm_called"]) for item in rows),
        "llm_valid_response_count": sum(item["llm_response_status"] == "VALIDATED" for item in rows),
        "direct_match_count": sum(item["phase1_conclusion"] == "NORMAL_DIRECT" for item in rows),
        "phase2_escalation_rate": _ratio(sum(item["final_route"] == "PROCEED_PHASE2" for item in rows), len(rows)),
        "determinate_accuracy": _ratio(correct_determinate, len(determinate)),
        "determinate_coverage": _ratio(len(determinate), total_selected),
    }


def _ratio(numerator: int, denominator: int) -> str | None:
    return None if denominator == 0 else f"{numerator / denominator:.4f}"


def _write_sample_result(root: Path, result: Mapping[str, Any]) -> None:
    validate_balanced20_report(result)
    _atomic_json(root / _sample_relative_path(result["sample_alias"]), result)


def _sample_relative_path(alias: str) -> str:
    return f"samples/{alias}.json"


def _write_or_verify(path: Path, payload: Mapping[str, Any], code: str) -> None:
    if path.exists():
        if _read_json(path, code) != dict(payload):
            raise ValueError(code)
    else:
        _atomic_json(path, payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(code) from None


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


def _client_request_count(client: Any) -> int:
    value = getattr(client, "request_count", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_live_config() -> Mapping[str, Any]:
    from src.cloc_v2.config_loader import load_chain_config
    return load_chain_config("eth")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
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
        selection = build_balanced20_selection()
        write_balanced20_selection(args.output_dir, selection)
        loaded = load_formal_normal_baseline(DEFAULT_FORMAL_NORMAL_BASELINE_DIR)
        config = _load_live_config()
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

        run_result = run_balanced20_inference(
            selection.inference_manifest,
            live=True,
            output_dir=args.output_dir,
            rpc_client=rpc_client,
            deepseek_runner=deepseek_runner,
            loaded_baseline=loaded,
            retry_failed=args.retry_failed,
        )
        report = build_balanced20_evaluation_report(run_result, selection.evaluation_truth)
        report_path = write_balanced20_report(args.output_dir, report)
    except Exception as exc:
        print(f"balanced20_status={_exception_code(exc, 'BALANCED20_EXECUTION_ERROR')}")
        return 1
    print(f"balanced20_status={run_result.run_status}")
    print(f"report_path={report_path}")
    print(f"rpc_request_count={run_result.rpc_request_count}")
    print(f"deepseek_request_count={run_result.deepseek_request_count}")
    return 0 if run_result.run_status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Balanced20RunResult", "Balanced20Selection", "BalancedEvaluationTruth",
    "BalancedInferenceManifest", "BalancedInferenceSample", "build_balanced20_evaluation_report",
    "build_balanced20_selection", "main", "run_balanced20_inference",
    "serialize_balanced20_report", "validate_balanced20_report",
    "write_balanced20_report", "write_balanced20_selection",
]

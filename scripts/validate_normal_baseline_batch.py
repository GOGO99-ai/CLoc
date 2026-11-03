"""Resumable, multi-chain validation for the formal normal baseline.

The command is offline by default. Dataset ingestion uses only ``platform``
and ``faultless.transaction_hash``. Live RPC access requires an explicit
``batch --live`` invocation; finalization is always offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_real_normal_baseline import JsonRpcClient, RpcValidationError  # noqa: E402
from src.cloc_v2.asset_continuity import build_asset_continuity  # noqa: E402
from src.cloc_v2.calltracer_adapter import adapt_calltracer_call_tree  # noqa: E402
from src.cloc_v2.normal_baseline import build_normal_baseline, query_normal_baseline  # noqa: E402
from src.cloc_v2.normal_distance import (  # noqa: E402
    AssetBehavior,
    CallStep,
    ChainFeatureRecord,
    MainAnchorKey,
)
from src.cloc_v2.normal_features import extract_normal_features  # noqa: E402
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger  # noqa: E402
from src.cloc_v2.result_detection import detect_result_candidates  # noqa: E402
from src.cloc_v2.runtime_semantics import build_runtime_annotations  # noqa: E402
from src.cloc_v2.strict_result_chains import build_strict_result_chains  # noqa: E402
from src.cloc_v2.transaction_graph import build_transaction_graph  # noqa: E402


DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "formal_normal_baseline_full"
MANIFEST_FILENAME = "normal_manifest.json"
CHECKPOINT_FILENAME = "checkpoint.json"
FINAL_FILENAME = "formal_normal_baseline.json"
DEFAULT_BATCH_SIZE = 8
MAX_BATCH_SIZE = 8
MIN_GROUP_SIZES = (2, 3, 5)
CHAIN_IDS = {"eth": 1, "bsc": 56}
EXPECTED_COUNTS = {"eth": 470, "bsc": 151}
_HASH = re.compile(r"^0x[0-9a-fA-F]{64}$")
_MACHINE_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_NETWORK_STOP_CODES = {
    "RPC_NETWORK_ERROR",
    "RPC_TIMEOUT",
    "RPC_UNAUTHORIZED",
    "RPC_QUOTA_EXCEEDED",
    "RPC_HTTP_ERROR",
    "RPC_INVALID_RESPONSE",
}
_FORBIDDEN_KEYS = {
    "rpc_url",
    "api_key",
    "key",
    "input",
    "output",
    "error",
    "revertreason",
    "calldata",
    "raw_trace",
    "trace_response",
    "fault",
    "cause",
    "location",
    "ground_truth",
    "paired_transaction",
    "amount_raw",
}


@dataclass(frozen=True)
class BatchNormalSample:
    chain_key: str
    blockchain_chain_id: int
    tx_hash: str
    sample_id: str

    @property
    def sample_key(self) -> str:
        return f"{self.chain_key}:{self.tx_hash}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_key": self.chain_key,
            "blockchain_chain_id": self.blockchain_chain_id,
            "tx_hash": self.tx_hash,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True)
class NormalBatchManifest:
    samples: tuple[BatchNormalSample, ...]
    chain_counts: Mapping[str, int]
    manifest_sha256: str

    SCHEMA_VERSION = "formal_normal_batch_manifest_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "chain_counts": {key: self.chain_counts[key] for key in sorted(self.chain_counts)},
            "samples": [sample.to_dict() for sample in self.samples],
        }


def build_normal_manifest(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    *,
    expected_counts: Mapping[str, int] | None = EXPECTED_COUNTS,
) -> NormalBatchManifest:
    """Read only platform and explicitly faultless transaction hashes."""
    by_chain: dict[str, set[str]] = {"eth": set(), "bsc": set()}
    for path in sorted(Path(dataset_dir).rglob("*.json")):
        try:
            platform, raw_hashes = _read_allowed_dataset_fields(path)
        except OSError as exc:
            raise ValueError("DATASET_JSON_INVALID") from exc
        if raw_hashes is None:
            continue
        hashes = tuple(_flatten_hashes(raw_hashes))
        if not hashes:
            continue
        chain_key = _chain_key(platform)
        if chain_key is None:
            raise ValueError("NORMAL_SAMPLE_CHAIN_UNSUPPORTED")
        for tx_hash in hashes:
            if _HASH.fullmatch(tx_hash) is None:
                raise ValueError("NORMAL_SAMPLE_HASH_INVALID")
            by_chain[chain_key].add(tx_hash.lower())

    counts = {chain: len(values) for chain, values in by_chain.items()}
    if expected_counts is not None:
        expected = {chain: int(expected_counts.get(chain, -1)) for chain in CHAIN_IDS}
        if counts != expected:
            raise ValueError("NORMAL_MANIFEST_COUNT_DRIFT")

    samples: list[BatchNormalSample] = []
    for chain_key in sorted(CHAIN_IDS, key=lambda item: CHAIN_IDS[item]):
        for index, tx_hash in enumerate(sorted(by_chain[chain_key]), start=1):
            samples.append(
                BatchNormalSample(
                    chain_key=chain_key,
                    blockchain_chain_id=CHAIN_IDS[chain_key],
                    tx_hash=tx_hash,
                    sample_id=f"{chain_key}_normal_{index:06d}",
                )
            )
    frozen = tuple(samples)
    digest = _manifest_digest(frozen, counts)
    return NormalBatchManifest(frozen, counts, digest)


def _read_allowed_dataset_fields(path: Path) -> tuple[Any, Any | None]:
    """Extract only platform and faultless.transaction_hash JSON fragments.

    Some legacy dataset documents contain malformed content in unrelated
    fields. Those fields are neither parsed nor used by the normal manifest.
    """
    text = path.read_text(encoding="utf-8")
    platform_match = re.search(r'"platform"\s*:\s*("(?:\\.|[^"\\])*")', text, re.IGNORECASE)
    platform = json.loads(platform_match.group(1)) if platform_match is not None else None

    faultless_match = re.search(r'"faultless"\s*:\s*\{', text, re.IGNORECASE)
    if faultless_match is None:
        return platform, None
    object_start = text.find("{", faultless_match.start())
    object_end = _json_container_end(text, object_start)
    if object_end is None:
        raise ValueError("FAULTLESS_OBJECT_INVALID")
    faultless_text = text[object_start:object_end]
    hash_match = re.search(r'"transaction_hash"\s*:', faultless_text, re.IGNORECASE)
    if hash_match is None:
        return platform, None
    value_start = hash_match.end()
    while value_start < len(faultless_text) and faultless_text[value_start].isspace():
        value_start += 1
    value_end = _json_value_end(faultless_text, value_start)
    if value_end is None:
        raise ValueError("FAULTLESS_TRANSACTION_HASH_STRUCTURE_INVALID")
    raw_value = faultless_text[value_start:value_end]
    if raw_value.startswith("["):
        # Legacy files may contain invalid separators or comments inside this
        # one array. Consume only explicit JSON string tokens; every extracted
        # value is still subjected to the exact transaction-hash validator.
        strings = []
        for match in re.finditer(r'"(?:\\.|[^"\\])*"', raw_value):
            try:
                strings.append(json.loads(match.group(0)))
            except json.JSONDecodeError as exc:
                raise ValueError("FAULTLESS_TRANSACTION_HASH_STRUCTURE_INVALID") from exc
        return platform, strings
    try:
        return platform, json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("FAULTLESS_TRANSACTION_HASH_STRUCTURE_INVALID") from exc


def _json_container_end(text: str, start: int) -> int | None:
    if start < 0 or start >= len(text) or text[start] not in "[{":
        return None
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
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
            character = text[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                return index + 1
        return None
    match = re.match(r"(?:null|true|false|-?[0-9]+(?:\.[0-9]+)?)", text[start:])
    return start + len(match.group(0)) if match is not None else None


def run_chain_batch(
    manifest: NormalBatchManifest,
    *,
    chain_key: str,
    output_dir: Path,
    rpc_client: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retry_failed: bool = False,
    processor: Callable[[BatchNormalSample, Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one bounded chain-specific batch and atomically retain progress."""
    if chain_key not in CHAIN_IDS:
        raise ValueError("CHAIN_KEY_UNSUPPORTED")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError("BATCH_SIZE_MUST_BE_BETWEEN_1_AND_8")
    output_path = Path(output_dir)
    checkpoint = _prepare_workspace(output_path, manifest)
    _reconcile_success_artifacts(output_path, manifest, checkpoint)

    starting_requests = int(getattr(rpc_client, "request_count", 0))
    last_synced_requests = starting_requests

    def save_checkpoint() -> None:
        nonlocal last_synced_requests
        current = int(getattr(rpc_client, "request_count", last_synced_requests))
        checkpoint["rpc_request_count"] += current - last_synced_requests
        last_synced_requests = current
        _atomic_json(output_path / CHECKPOINT_FILENAME, checkpoint)

    try:
        reported_chain_id = _parse_chain_id(rpc_client.call("eth_chainId", []))
    except RpcValidationError as exc:
        checkpoint["last_stop_code"] = _safe_code(exc.category)
        save_checkpoint()
        return _batch_result("STOPPED", chain_key, rpc_client, starting_requests, checkpoint["last_stop_code"])
    save_checkpoint()
    if reported_chain_id != CHAIN_IDS[chain_key]:
        checkpoint["last_stop_code"] = "CHAIN_ID_MISMATCH"
        save_checkpoint()
        return _batch_result("STOPPED", chain_key, rpc_client, starting_requests, "CHAIN_ID_MISMATCH")

    selected = []
    for sample in manifest.samples:
        if sample.chain_key != chain_key:
            continue
        state = checkpoint["samples"][sample.sample_key]
        if state["status"] == "SUCCEEDED":
            continue
        if state["status"] == "FAILED" and not retry_failed:
            continue
        if state["status"] == "IN_PROGRESS":
            state["status"] = "FAILED"
            state["failure_code"] = "PREVIOUS_ATTEMPT_INTERRUPTED"
            save_checkpoint()
            if not retry_failed:
                continue
        selected.append(sample)
        if len(selected) == batch_size:
            break

    process = processor or process_calltracer_sample
    succeeded = 0
    failed = 0
    for sample in selected:
        state = checkpoint["samples"][sample.sample_key]
        state["status"] = "IN_PROGRESS"
        state["attempt_count"] += 1
        state["failure_code"] = None
        save_checkpoint()
        try:
            raw_frame = rpc_client.call(
                "debug_traceTransaction",
                [sample.tx_hash, {"tracer": "callTracer"}],
            )
        except RpcValidationError as exc:
            code = _safe_code(exc.category)
            state["status"] = "FAILED"
            state["failure_code"] = code
            failed += 1
            checkpoint["last_stop_code"] = code if code in _NETWORK_STOP_CODES else None
            save_checkpoint()
            if code in _NETWORK_STOP_CODES:
                return _batch_result("STOPPED", chain_key, rpc_client, starting_requests, code, succeeded, failed)
            continue

        try:
            processed = dict(process(sample, raw_frame))
            artifact = _sample_artifact(sample, processed)
            _validate_sanitized(artifact)
            _atomic_json(_sample_path(output_path, sample), artifact)
        except Exception:
            state["status"] = "FAILED"
            state["failure_code"] = "SAMPLE_PROCESSING_ERROR"
            failed += 1
            save_checkpoint()
            continue
        state["status"] = "SUCCEEDED"
        state["failure_code"] = None
        state["artifact"] = str(_sample_relative_path(sample)).replace("\\", "/")
        succeeded += 1
        save_checkpoint()

    checkpoint["last_stop_code"] = None
    save_checkpoint()
    return _batch_result("COMPLETED", chain_key, rpc_client, starting_requests, None, succeeded, failed)


def process_calltracer_sample(sample: BatchNormalSample, raw_frame: Any) -> dict[str, Any]:
    """Run the formal in-memory pipeline and return amount-free features only."""
    call_tree = adapt_calltracer_call_tree(raw_frame)
    graph = build_transaction_graph(call_tree)
    annotations = build_runtime_annotations(graph)
    ledger = build_observed_asset_ledger(graph, annotations)
    continuity = build_asset_continuity(graph, annotations, ledger).to_dict()
    detected = detect_result_candidates(graph, annotations, ledger, continuity)
    strict = build_strict_result_chains(
        graph,
        continuity,
        detected.to_dict(),
        blockchain_chain_id=sample.blockchain_chain_id,
        offline_function_candidates=(),
        offline_effective_address_map={},
    )
    extracted = extract_normal_features(
        graph,
        strict,
        ledger,
        transaction_record_id=sample.sample_id,
        offline_function_candidates=(),
        offline_effective_address_map={},
    )
    reasons = sorted(
        {
            _safe_code(reason)
            for status in extracted.chain_statuses
            for reason in status.reasons
        }
    )
    stage_counts = {
        "call_node_count": len(graph.nodes),
        "recognized_transfer_node_count": sum(
            "TOKEN_TRANSFER" in annotation.action_types
            for annotation in annotations.node_annotations
        ),
        "asset_fact_count": len(ledger.transfer_facts),
        "complete_asset_fact_count": sum(fact.status == "COMPLETE" for fact in ledger.transfer_facts),
        "incomplete_asset_fact_count": sum(fact.status != "COMPLETE" for fact in ledger.transfer_facts),
        "result_candidate_count": len(detected.result_candidates),
        "strict_chain_count": len(strict.chains),
        "strict_ready_count": sum(chain.status == "READY" for chain in strict.chains),
        "strict_uncertain_count": sum(chain.status == "UNCERTAIN" for chain in strict.chains),
        "clean_record_count": extracted.clean_record_count,
        "uncertain_record_count": extracted.uncertain_record_count,
        "rejected_record_count": extracted.rejected_chain_count,
    }
    return {
        "stage_counts": stage_counts,
        "reason_codes": reasons,
        "feature_records": [record.to_dict() for record in extracted.records],
    }


def finalize_normal_baseline(output_dir: Path, *, write: bool = True) -> dict[str, Any]:
    """Offline-only merge, calibration, and cross-transaction LOO diagnostics."""
    output_path = Path(output_dir)
    manifest = _load_manifest(output_path / MANIFEST_FILENAME)
    checkpoint = _load_checkpoint(output_path / CHECKPOINT_FILENAME, manifest)
    records: list[ChainFeatureRecord] = []
    transaction_for_record: dict[str, str] = {}
    successful_transactions: set[str] = set()
    for sample in manifest.samples:
        if checkpoint["samples"][sample.sample_key]["status"] != "SUCCEEDED":
            continue
        artifact = _load_sample_artifact(output_path, sample)
        successful_transactions.add(sample.sample_key)
        for raw_record in artifact["feature_records"]:
            record = _record_from_dict(raw_record)
            if record.record_id in transaction_for_record:
                raise ValueError("FINALIZE_RECORD_ID_DUPLICATE")
            transaction_for_record[record.record_id] = sample.sample_key
            if not record.uncertainties:
                records.append(record)
    clean_records = tuple(sorted(records, key=lambda item: item.record_id))

    settings = []
    leave_one_out = []
    for minimum in MIN_GROUP_SIZES:
        baseline = build_normal_baseline(clean_records, min_group_size=minimum)
        groups = []
        for group in baseline.groups:
            tx_count = len({transaction_for_record[record.record_id] for record in group.records})
            groups.append(
                {
                    "main_anchor": group.main_anchor.to_dict(),
                    "clean_chain_count": len(group.records),
                    "distinct_normal_transaction_count": tx_count,
                    "calibration_status": group.calibration.status,
                    "threshold": group.calibration.to_dict()["threshold"],
                }
            )
        settings.append(
            {
                "min_group_size": minimum,
                "calibrated_group_count": baseline.calibrated_group_count,
                "uncalibrated_group_count": baseline.uncalibrated_group_count,
                "groups": groups,
                "baseline": baseline.to_dict(),
            }
        )

        queries = []
        for query in clean_records:
            query_tx = transaction_for_record[query.record_id]
            remaining = tuple(
                record
                for record in clean_records
                if transaction_for_record[record.record_id] != query_tx
            )
            diagnostic_baseline = build_normal_baseline(remaining, min_group_size=minimum)
            result = query_normal_baseline(diagnostic_baseline, query).to_dict()
            if any(transaction_for_record[match["normal_record_id"]] == query_tx for match in result["top_matches"]):
                raise ValueError("LEAVE_ONE_OUT_SAME_TRANSACTION_MATCH")
            result["query_transaction_key"] = query_tx
            queries.append(result)
        leave_one_out.append({"min_group_size": minimum, "queries": queries})

    result = {
        "schema_version": "formal_normal_batch_finalize_v1",
        "manifest_sha256": manifest.manifest_sha256,
        "successful_transaction_count": len(successful_transactions),
        "clean_record_count": len(clean_records),
        "settings": settings,
        "leave_one_out": leave_one_out,
    }
    _validate_sanitized(result)
    if write:
        _atomic_json(output_path / FINAL_FILENAME, result)
    return result


def _flatten_hashes(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_hashes(item)
    elif value is not None:
        raise ValueError("FAULTLESS_TRANSACTION_HASH_STRUCTURE_INVALID")


def _chain_key(value: Any) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    if normalized in {"ethereum", "eth"}:
        return "eth"
    if normalized in {"bnbchain", "bsc", "binancesmartchain"}:
        return "bsc"
    return None


def _manifest_digest(samples: tuple[BatchNormalSample, ...], counts: Mapping[str, int]) -> str:
    payload = {
        "chain_counts": {key: counts[key] for key in sorted(counts)},
        "samples": [sample.to_dict() for sample in samples],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prepare_workspace(output_dir: Path, manifest: NormalBatchManifest) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        existing = _load_manifest(manifest_path)
        if existing.manifest_sha256 != manifest.manifest_sha256 or existing.to_dict() != manifest.to_dict():
            raise ValueError("NORMAL_MANIFEST_DRIFT")
    else:
        _atomic_json(manifest_path, manifest.to_dict())
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    if checkpoint_path.exists():
        return _load_checkpoint(checkpoint_path, manifest)
    checkpoint = {
        "schema_version": "formal_normal_batch_checkpoint_v1",
        "manifest_sha256": manifest.manifest_sha256,
        "rpc_request_count": 0,
        "last_stop_code": None,
        "samples": {
            sample.sample_key: {
                "status": "PENDING",
                "attempt_count": 0,
                "failure_code": None,
                "artifact": None,
            }
            for sample in manifest.samples
        },
    }
    _atomic_json(checkpoint_path, checkpoint)
    return checkpoint


def _load_manifest(path: Path) -> NormalBatchManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = tuple(
            BatchNormalSample(
                chain_key=item["chain_key"],
                blockchain_chain_id=item["blockchain_chain_id"],
                tx_hash=item["tx_hash"],
                sample_id=item["sample_id"],
            )
            for item in payload["samples"]
        )
        counts = {str(key): int(value) for key, value in payload["chain_counts"].items()}
        manifest = NormalBatchManifest(samples, counts, payload["manifest_sha256"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("NORMAL_MANIFEST_INVALID") from exc
    if manifest.manifest_sha256 != _manifest_digest(samples, counts):
        raise ValueError("NORMAL_MANIFEST_DIGEST_MISMATCH")
    return manifest


def _load_checkpoint(path: Path, manifest: NormalBatchManifest) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("CHECKPOINT_INVALID") from exc
    if checkpoint.get("manifest_sha256") != manifest.manifest_sha256:
        raise ValueError("NORMAL_MANIFEST_DRIFT")
    expected_keys = {sample.sample_key for sample in manifest.samples}
    if set(checkpoint.get("samples", {})) != expected_keys:
        raise ValueError("CHECKPOINT_SAMPLE_SET_MISMATCH")
    return checkpoint


def _reconcile_success_artifacts(output_dir: Path, manifest: NormalBatchManifest, checkpoint: dict[str, Any]) -> None:
    changed = False
    for sample in manifest.samples:
        state = checkpoint["samples"][sample.sample_key]
        if state["status"] == "SUCCEEDED":
            _load_sample_artifact(output_dir, sample)
            continue
        path = _sample_path(output_dir, sample)
        if path.exists():
            _load_sample_artifact(output_dir, sample)
            state["status"] = "SUCCEEDED"
            state["failure_code"] = None
            state["artifact"] = str(_sample_relative_path(sample)).replace("\\", "/")
            changed = True
    if changed:
        _atomic_json(output_dir / CHECKPOINT_FILENAME, checkpoint)


def _sample_artifact(sample: BatchNormalSample, processed: Mapping[str, Any]) -> dict[str, Any]:
    stage_counts = processed.get("stage_counts")
    reason_codes = processed.get("reason_codes")
    records = processed.get("feature_records")
    if not isinstance(stage_counts, Mapping) or not isinstance(reason_codes, (list, tuple)) or not isinstance(records, (list, tuple)):
        raise ValueError("SAMPLE_PROCESSOR_OUTPUT_INVALID")
    return {
        "schema_version": "formal_normal_batch_sample_v1",
        "chain_key": sample.chain_key,
        "blockchain_chain_id": sample.blockchain_chain_id,
        "tx_hash": sample.tx_hash,
        "sample_id": sample.sample_id,
        "stage_counts": {str(key): int(value) for key, value in sorted(stage_counts.items())},
        "reason_codes": sorted({_safe_code(value) for value in reason_codes}),
        "feature_records": [_sanitized_feature_record(record) for record in records],
    }


def _sanitized_feature_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("FEATURE_RECORD_INVALID")
    _validate_sanitized(value)
    record = _record_from_dict(value)
    sanitized = ChainFeatureRecord(
        record_id=record.record_id,
        main_anchor=record.main_anchor,
        call_steps=record.call_steps,
        asset_behaviors=record.asset_behaviors,
        uncertainties=tuple(_safe_code(reason) for reason in record.uncertainties),
        result_scoring_context=record.result_scoring_context,
    )
    return sanitized.to_dict()


def _sample_relative_path(sample: BatchNormalSample) -> Path:
    return Path("samples") / sample.chain_key / f"{sample.tx_hash}.json"


def _sample_path(output_dir: Path, sample: BatchNormalSample) -> Path:
    return output_dir / _sample_relative_path(sample)


def _load_sample_artifact(output_dir: Path, sample: BatchNormalSample) -> dict[str, Any]:
    try:
        payload = json.loads(_sample_path(output_dir, sample).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("SAMPLE_ARTIFACT_INVALID") from exc
    if (
        payload.get("schema_version") != "formal_normal_batch_sample_v1"
        or payload.get("chain_key") != sample.chain_key
        or payload.get("blockchain_chain_id") != sample.blockchain_chain_id
        or payload.get("tx_hash") != sample.tx_hash
        or payload.get("sample_id") != sample.sample_id
    ):
        raise ValueError("SAMPLE_ARTIFACT_IDENTITY_MISMATCH")
    _validate_sanitized(payload)
    return payload


def _record_from_dict(value: Mapping[str, Any]) -> ChainFeatureRecord:
    anchor = value["main_anchor"]
    return ChainFeatureRecord(
        record_id=value["record_id"],
        main_anchor=MainAnchorKey(
            anchor["blockchain_chain_id"],
            anchor["effective_address"],
            anchor["selector"],
        ),
        call_steps=tuple(
            CallStep(item["effective_address"], item["selector"])
            for item in value.get("call_steps", ())
        ),
        asset_behaviors=frozenset(
            AssetBehavior(item["asset_identifier"], item["direction"])
            for item in value.get("asset_behaviors", ())
        ),
        uncertainties=tuple(value.get("uncertainties", ())),
        result_scoring_context=value.get("result_scoring_context", "legacy_independent_weights"),
    )


def _safe_code(value: Any) -> str:
    return value if isinstance(value, str) and _MACHINE_CODE.fullmatch(value) else "NON_MACHINE_READABLE_UNCERTAINTY"


def _parse_chain_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]+", value):
        return int(value, 16)
    return None


def _batch_result(
    status: str,
    chain_key: str,
    client: Any,
    starting_requests: int,
    stop_code: str | None,
    succeeded: int = 0,
    failed: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "chain_key": chain_key,
        "rpc_request_count": int(getattr(client, "request_count", starting_requests)) - starting_requests,
        "succeeded_count": succeeded,
        "failed_count": failed,
        "stop_code": stop_code,
    }


def _validate_sanitized(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError("FORBIDDEN_ARTIFACT_FIELD")
            _validate_sanitized(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_sanitized(nested)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _validate_sanitized(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(handle.name, path)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _configured_client(chain_key: str, max_requests: int) -> JsonRpcClient:
    from src.cloc_v2.config_loader import load_chain_config

    config = load_chain_config(chain_key)
    rpc_url = config.get("rpc_url")
    if not isinstance(rpc_url, str) or not rpc_url:
        raise RpcValidationError("RPC_NOT_CONFIGURED")
    return JsonRpcClient(rpc_url, max_requests=max_requests)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    batch = subparsers.add_parser("batch")
    batch.add_argument("--live", action="store_true")
    batch.add_argument("--chain", choices=tuple(CHAIN_IDS), required=True)
    batch.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    batch.add_argument("--retry-failed", action="store_true")
    batch.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    batch.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "finalize":
        result = finalize_normal_baseline(args.output_dir)
        print(f"FINALIZE_COMPLETE clean_records={result['clean_record_count']}")
        return 0
    if not args.live:
        print("LIVE_MODE_REQUIRED")
        return 0
    manifest = build_normal_manifest(args.dataset_dir)
    client = _configured_client(args.chain, args.batch_size + 1)
    result = run_chain_batch(
        manifest,
        chain_key=args.chain,
        output_dir=args.output_dir,
        rpc_client=client,
        batch_size=args.batch_size,
        retry_failed=args.retry_failed,
    )
    print(
        f"BATCH_{result['status']} chain={result['chain_key']} "
        f"rpc_requests={result['rpc_request_count']} "
        f"succeeded={result['succeeded_count']} failed={result['failed_count']}"
    )
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BatchNormalSample",
    "NormalBatchManifest",
    "build_normal_manifest",
    "finalize_normal_baseline",
    "main",
    "process_calltracer_sample",
    "run_chain_batch",
]

"""Load and validate persisted normal-reference artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cloc.reference.library import NormalBaseline, build_normal_baseline
from cloc.reference.distance import AssetBehavior, CallStep, ChainFeatureRecord, MainAnchorKey
from cloc.review.routing import FORMAL_MIN_NORMAL_GROUP_SIZE


DEFAULT_FORMAL_NORMAL_BASELINE_DIR = Path("data/normal_reference")

_FINAL_SCHEMA = "formal_normal_batch_finalize_v1"
_MANIFEST_SCHEMA = "formal_normal_batch_manifest_v1"
_CHECKPOINT_SCHEMA = "formal_normal_batch_checkpoint_v1"
_SAMPLE_SCHEMA = "formal_normal_batch_sample_v1"
_BASELINE_SCHEMA = "normal_baseline_v1"
_SUMMARY_SCHEMA = "formal_normal_baseline_summary_v1"
_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_IDENTITY_STATUSES = {
    "KNOWN_NORMAL_ORIGIN",
    "NOT_IN_NORMAL_LIBRARY",
    "UNRESOLVED",
}

_FINAL_FIELDS = {
    "schema_version", "manifest_sha256", "successful_transaction_count",
    "clean_record_count", "settings", "leave_one_out",
}
_MANIFEST_FIELDS = {"schema_version", "manifest_sha256", "chain_counts", "samples"}
_MANIFEST_SAMPLE_FIELDS = {"chain_key", "blockchain_chain_id", "tx_hash", "sample_id"}
_CHECKPOINT_FIELDS = {"schema_version", "manifest_sha256", "rpc_request_count", "last_stop_code", "samples"}
_CHECKPOINT_STATE_FIELDS = {"status", "attempt_count", "failure_code", "artifact"}
_SAMPLE_FIELDS = {
    "schema_version", "chain_key", "blockchain_chain_id", "tx_hash", "sample_id",
    "stage_counts", "reason_codes", "feature_records",
}
_SETTING_FIELDS = {
    "min_group_size", "calibrated_group_count", "uncalibrated_group_count", "groups", "baseline",
}
_RECORD_FIELDS = {"record_id", "main_anchor", "call_steps", "asset_behaviors", "uncertainties"}
_ANCHOR_FIELDS = {"blockchain_chain_id", "effective_address", "selector"}
_CALL_FIELDS = {"effective_address", "selector"}
_ASSET_FIELDS = {"asset_identifier", "direction"}


@dataclass(frozen=True)
class NormalRecordTransactionOrigin:
    """Internal explicit ownership evidence; identities stay out of repr."""

    record_id: str = field(repr=False)
    sample_id: str = field(repr=False)


@dataclass(frozen=True)
class FormalNormalTransactionIdentity:
    """Sanitized proof of whether one query transaction is a known normal origin."""

    status: str
    reason_codes: tuple[str, ...]
    usable_for_direct_normal_identity: bool
    _normal_origin_id: str | None = field(default=None, repr=False)
    _normalized_transaction_hash: str | None = field(default=None, repr=False)

    SCHEMA_VERSION = "formal_normal_transaction_identity_v1"

    def __post_init__(self) -> None:
        if self.status not in _IDENTITY_STATUSES:
            raise ValueError("FORMAL_NORMAL_IDENTITY_STATUS_INVALID")
        reasons = _stable_reason_codes(self.reason_codes)
        object.__setattr__(self, "reason_codes", reasons)
        known = self.status == "KNOWN_NORMAL_ORIGIN"
        if self.usable_for_direct_normal_identity != known:
            raise ValueError("FORMAL_NORMAL_IDENTITY_USABILITY_INCONSISTENT")
        if known != (isinstance(self._normal_origin_id, str) and bool(self._normal_origin_id)):
            raise ValueError("FORMAL_NORMAL_IDENTITY_ORIGIN_INCONSISTENT")
        if self._normalized_transaction_hash is not None and (
            not isinstance(self._normalized_transaction_hash, str)
            or _HASH.fullmatch(self._normalized_transaction_hash) is None
        ):
            raise ValueError("FORMAL_NORMAL_IDENTITY_HASH_INVALID")

    @property
    def normal_origin_id(self) -> str | None:
        """Return internal identity evidence without exposing it in serialization."""
        return self._normal_origin_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "usable_for_direct_normal_identity": self.usable_for_direct_normal_identity,
        }


@dataclass(frozen=True)
class FormalNormalBaselineSummary:
    schema_version: str
    formal_min_group_size: int
    successful_ethereum_transaction_count: int
    clean_record_count: int
    group_count: int
    calibrated_group_count: int
    uncalibrated_group_count: int
    record_origin_mapping_count: int
    manifest_consistency_passed: bool
    reconstructed_baseline_match: bool
    ethereum_only: bool
    source_files_read_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "formal_min_group_size": self.formal_min_group_size,
            "successful_ethereum_transaction_count": self.successful_ethereum_transaction_count,
            "clean_record_count": self.clean_record_count,
            "group_count": self.group_count,
            "calibrated_group_count": self.calibrated_group_count,
            "uncalibrated_group_count": self.uncalibrated_group_count,
            "record_origin_mapping_count": self.record_origin_mapping_count,
            "manifest_consistency_passed": self.manifest_consistency_passed,
            "reconstructed_baseline_match": self.reconstructed_baseline_match,
            "ethereum_only": self.ethereum_only,
            "source_files_read_count": self.source_files_read_count,
        }


@dataclass(frozen=True)
class LoadedFormalNormalBaseline:
    baseline: NormalBaseline = field(repr=False)
    normal_record_transaction_ids: Mapping[str, str] = field(repr=False)
    origins: tuple[NormalRecordTransactionOrigin, ...] = field(repr=False)
    summary: FormalNormalBaselineSummary
    normal_transaction_hash_origins: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, NormalBaseline):
            raise ValueError("FORMAL_BASELINE_OBJECT_INVALID")
        if not isinstance(self.summary, FormalNormalBaselineSummary):
            raise ValueError("FORMAL_BASELINE_SUMMARY_INVALID")
        mapping = dict(self.normal_record_transaction_ids)
        transaction_mapping = dict(self.normal_transaction_hash_origins)
        object.__setattr__(self, "normal_record_transaction_ids", MappingProxyType(mapping))
        object.__setattr__(self, "normal_transaction_hash_origins", MappingProxyType(transaction_mapping))
        object.__setattr__(self, "origins", tuple(self.origins))


def load_formal_normal_baseline(
    directory: str | Path = DEFAULT_FORMAL_NORMAL_BASELINE_DIR,
) -> LoadedFormalNormalBaseline:
    """Load, reconstruct, and cross-check the formal Ethereum baseline."""
    root = Path(directory).resolve()
    read_count = 0

    def read_json(path: Path, code: str) -> Any:
        nonlocal read_count
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError(code) from None
        read_count += 1
        return value

    final = read_json(root / "formal_normal_baseline.json", "FORMAL_FINAL_INVALID")
    manifest = read_json(root / "normal_manifest.json", "FORMAL_MANIFEST_INVALID")
    checkpoint = read_json(root / "checkpoint.json", "FORMAL_CHECKPOINT_INVALID")
    _require_mapping_fields(final, _FINAL_FIELDS, "FORMAL_FINAL_FIELDS_INVALID")
    _require_mapping_fields(manifest, _MANIFEST_FIELDS, "FORMAL_MANIFEST_FIELDS_INVALID")
    _require_mapping_fields(checkpoint, _CHECKPOINT_FIELDS, "FORMAL_CHECKPOINT_FIELDS_INVALID")
    if final["schema_version"] != _FINAL_SCHEMA:
        raise ValueError("FORMAL_FINAL_SCHEMA_INVALID")
    if manifest["schema_version"] != _MANIFEST_SCHEMA:
        raise ValueError("FORMAL_MANIFEST_SCHEMA_INVALID")
    if checkpoint["schema_version"] != _CHECKPOINT_SCHEMA:
        raise ValueError("FORMAL_CHECKPOINT_SCHEMA_INVALID")
    hashes = (final["manifest_sha256"], manifest["manifest_sha256"], checkpoint["manifest_sha256"])
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in hashes) or len(set(hashes)) != 1:
        raise ValueError("FORMAL_MANIFEST_SHA256_MISMATCH")

    manifest_samples, manifest_by_key = _validate_manifest(manifest)
    if _manifest_digest(manifest["samples"], manifest["chain_counts"]) != manifest["manifest_sha256"]:
        raise ValueError("FORMAL_MANIFEST_DIGEST_MISMATCH")
    states = _validate_checkpoint(checkpoint, set(manifest_by_key))
    formal_setting = _formal_setting(final)

    all_origins: dict[str, tuple[str, str]] = {}
    clean_records: list[ChainFeatureRecord] = []
    clean_origin: dict[str, str] = {}
    transaction_origins: dict[str, str] = {}
    successful_eth = 0
    for sample in manifest_samples:
        sample_key = _sample_key(sample)
        state = states[sample_key]
        if state["status"] != "SUCCEEDED":
            continue
        if sample["chain_key"] != "eth" or sample["blockchain_chain_id"] != 1:
            raise ValueError("FORMAL_SUCCESSFUL_SAMPLE_NOT_ETHEREUM")
        tx_hash = sample["tx_hash"]
        origin_id = sample["sample_id"]
        existing_origin = transaction_origins.get(tx_hash)
        if existing_origin is not None and existing_origin != origin_id:
            raise ValueError("FORMAL_TRANSACTION_ORIGIN_CONFLICT")
        transaction_origins[tx_hash] = origin_id
        successful_eth += 1
        artifact_path = _safe_artifact_path(root, state["artifact"])
        artifact = read_json(artifact_path, "FORMAL_SAMPLE_ARTIFACT_INVALID")
        _validate_sample_identity(artifact, sample)
        records = artifact["feature_records"]
        for raw_record in records:
            record = _parse_record(raw_record)
            if record.record_id in all_origins:
                raise ValueError("FORMAL_RECORD_ID_DUPLICATE_ACROSS_SAMPLES")
            all_origins[record.record_id] = (sample["sample_id"], sample["chain_key"])
            if record.uncertainties:
                continue
            if record.main_anchor.blockchain_chain_id != 1:
                raise ValueError("FORMAL_BASELINE_NON_ETHEREUM_ANCHOR")
            clean_records.append(record)
            clean_origin[record.record_id] = sample["sample_id"]

    clean = tuple(sorted(clean_records, key=lambda item: item.record_id))
    if len({record.record_id for record in clean}) != len(clean):
        raise ValueError("FORMAL_CLEAN_RECORD_ID_DUPLICATE")
    embedded = formal_setting["baseline"]
    if not isinstance(embedded, Mapping) or embedded.get("schema_version") != _BASELINE_SCHEMA:
        raise ValueError("FORMAL_EMBEDDED_BASELINE_SCHEMA_INVALID")
    if embedded.get("min_group_size") != FORMAL_MIN_NORMAL_GROUP_SIZE:
        raise ValueError("FORMAL_EMBEDDED_MIN_GROUP_SIZE_INVALID")
    embedded_ids = _embedded_record_ids(embedded)
    clean_ids = {record.record_id for record in clean}
    if embedded_ids != clean_ids or set(clean_origin) != clean_ids:
        raise ValueError("FORMAL_BASELINE_RECORD_SOURCE_MISMATCH")
    if any(all_origins[record_id][1] != "eth" for record_id in embedded_ids):
        raise ValueError("FORMAL_BASELINE_SOURCE_NOT_ETHEREUM")

    rebuilt = build_normal_baseline(clean, min_group_size=FORMAL_MIN_NORMAL_GROUP_SIZE)
    if rebuilt.to_dict() != embedded:
        raise ValueError("FORMAL_BASELINE_RECONSTRUCTION_MISMATCH")
    expected_groups = _setting_groups(rebuilt, clean_origin)
    if formal_setting["groups"] != expected_groups:
        raise ValueError("FORMAL_GROUP_SUMMARY_MISMATCH")
    if (
        formal_setting["calibrated_group_count"] != rebuilt.calibrated_group_count
        or formal_setting["uncalibrated_group_count"] != rebuilt.uncalibrated_group_count
    ):
        raise ValueError("FORMAL_SETTING_METADATA_MISMATCH")
    if final["successful_transaction_count"] != successful_eth or final["clean_record_count"] != len(clean):
        raise ValueError("FORMAL_FINAL_COUNT_MISMATCH")

    mapping = {record_id: clean_origin[record_id] for record_id in sorted(clean_ids)}
    origins = tuple(
        NormalRecordTransactionOrigin(record_id, mapping[record_id]) for record_id in sorted(mapping)
    )
    summary = FormalNormalBaselineSummary(
        schema_version=_SUMMARY_SCHEMA,
        formal_min_group_size=FORMAL_MIN_NORMAL_GROUP_SIZE,
        successful_ethereum_transaction_count=successful_eth,
        clean_record_count=len(clean),
        group_count=rebuilt.group_count,
        calibrated_group_count=rebuilt.calibrated_group_count,
        uncalibrated_group_count=rebuilt.uncalibrated_group_count,
        record_origin_mapping_count=len(mapping),
        manifest_consistency_passed=True,
        reconstructed_baseline_match=True,
        ethereum_only=True,
        source_files_read_count=read_count,
    )
    return LoadedFormalNormalBaseline(
        rebuilt,
        mapping,
        origins,
        summary,
        {tx_hash: transaction_origins[tx_hash] for tx_hash in sorted(transaction_origins)},
    )


def resolve_formal_normal_transaction_identity(
    loaded: LoadedFormalNormalBaseline,
    blockchain_chain_id: int,
    transaction_hash: str,
) -> FormalNormalTransactionIdentity:
    """Resolve a query identity only from the already validated formal artifacts."""
    if not isinstance(loaded, LoadedFormalNormalBaseline):
        raise ValueError("FORMAL_BASELINE_OBJECT_INVALID")
    if (
        not isinstance(blockchain_chain_id, int)
        or isinstance(blockchain_chain_id, bool)
        or blockchain_chain_id != 1
    ):
        return FormalNormalTransactionIdentity(
            "UNRESOLVED",
            ("QUERY_NORMAL_CHAIN_ID_MISMATCH",),
            False,
        )
    normalized = transaction_hash.lower() if isinstance(transaction_hash, str) else None
    if normalized is None or _HASH.fullmatch(normalized) is None:
        return FormalNormalTransactionIdentity(
            "UNRESOLVED",
            ("QUERY_TRANSACTION_HASH_INVALID",),
            False,
        )
    mapping = loaded.normal_transaction_hash_origins
    if (
        len(mapping) != loaded.summary.successful_ethereum_transaction_count
        or any(_HASH.fullmatch(key) is None or not isinstance(value, str) or not value for key, value in mapping.items())
    ):
        return FormalNormalTransactionIdentity(
            "UNRESOLVED",
            ("NORMAL_TRANSACTION_ORIGIN_EVIDENCE_INCOMPLETE",),
            False,
            _normalized_transaction_hash=normalized,
        )
    origin_id = mapping.get(normalized)
    if origin_id is None:
        return FormalNormalTransactionIdentity(
            "NOT_IN_NORMAL_LIBRARY",
            ("QUERY_TRANSACTION_NOT_IN_NORMAL_LIBRARY",),
            False,
            _normalized_transaction_hash=normalized,
        )
    return FormalNormalTransactionIdentity(
        "KNOWN_NORMAL_ORIGIN",
        ("QUERY_NORMAL_ORIGIN_RESOLVED",),
        True,
        _normal_origin_id=origin_id,
        _normalized_transaction_hash=normalized,
    )


def serialize_formal_normal_baseline_summary(summary: FormalNormalBaselineSummary) -> str:
    """Serialize only safe counts and booleans; no evidence identity is exposed."""
    if not isinstance(summary, FormalNormalBaselineSummary):
        raise ValueError("FORMAL_BASELINE_SUMMARY_INVALID")
    return json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_formal_normal_transaction_identity(identity: FormalNormalTransactionIdentity) -> str:
    """Serialize only status and reason codes, never query or origin identity."""
    if not isinstance(identity, FormalNormalTransactionIdentity):
        raise ValueError("FORMAL_NORMAL_IDENTITY_INVALID")
    return json.dumps(identity.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _validate_manifest(value: Mapping[str, Any]) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Mapping[str, Any]]]:
    raw_samples = value.get("samples")
    chain_counts = value.get("chain_counts")
    if not isinstance(raw_samples, list) or not isinstance(chain_counts, Mapping):
        raise ValueError("FORMAL_MANIFEST_STRUCTURE_INVALID")
    samples: list[Mapping[str, Any]] = []
    by_key: dict[str, Mapping[str, Any]] = {}
    sample_ids: set[str] = set()
    actual_counts: dict[str, int] = {"eth": 0, "bsc": 0}
    for item in raw_samples:
        _require_mapping_fields(item, _MANIFEST_SAMPLE_FIELDS, "FORMAL_MANIFEST_SAMPLE_INVALID")
        chain = item["chain_key"]
        chain_id = item["blockchain_chain_id"]
        tx_hash = item["tx_hash"]
        sample_id = item["sample_id"]
        if chain not in {"eth", "bsc"} or chain_id != {"eth": 1, "bsc": 56}[chain]:
            raise ValueError("FORMAL_MANIFEST_CHAIN_INVALID")
        if not isinstance(tx_hash, str) or _HASH.fullmatch(tx_hash) is None:
            raise ValueError("FORMAL_MANIFEST_TRANSACTION_INVALID")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("FORMAL_MANIFEST_SAMPLE_ID_INVALID")
        key = _sample_key(item)
        if key in by_key or sample_id in sample_ids:
            raise ValueError("FORMAL_MANIFEST_SAMPLE_DUPLICATE")
        by_key[key] = item
        sample_ids.add(sample_id)
        actual_counts[chain] += 1
        samples.append(item)
    normalized_counts = {str(key): count for key, count in chain_counts.items()}
    if normalized_counts != actual_counts:
        raise ValueError("FORMAL_MANIFEST_CHAIN_COUNT_MISMATCH")
    return tuple(samples), by_key


def _validate_checkpoint(value: Mapping[str, Any], expected_keys: set[str]) -> Mapping[str, Mapping[str, Any]]:
    samples = value.get("samples")
    if not isinstance(samples, Mapping) or set(samples) != expected_keys:
        raise ValueError("FORMAL_CHECKPOINT_SAMPLE_SET_MISMATCH")
    for state in samples.values():
        _require_mapping_fields(state, _CHECKPOINT_STATE_FIELDS, "FORMAL_CHECKPOINT_STATE_INVALID")
        if state["status"] not in {"PENDING", "FAILED", "SUCCEEDED"}:
            raise ValueError("FORMAL_CHECKPOINT_STATUS_INVALID")
        if not isinstance(state["attempt_count"], int) or isinstance(state["attempt_count"], bool) or state["attempt_count"] < 0:
            raise ValueError("FORMAL_CHECKPOINT_ATTEMPT_INVALID")
        if state["status"] == "SUCCEEDED" and (not isinstance(state["artifact"], str) or not state["artifact"]):
            raise ValueError("FORMAL_CHECKPOINT_ARTIFACT_MISSING")
    return samples


def _formal_setting(final: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = final.get("settings")
    if not isinstance(settings, list):
        raise ValueError("FORMAL_SETTINGS_INVALID")
    for item in settings:
        _require_mapping_fields(item, _SETTING_FIELDS, "FORMAL_SETTING_FIELDS_INVALID")
    matches = [item for item in settings if item["min_group_size"] == FORMAL_MIN_NORMAL_GROUP_SIZE]
    if len(matches) != 1:
        raise ValueError("FORMAL_MIN_GROUP_SIZE_SETTING_INVALID")
    return matches[0]


def _safe_artifact_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("FORMAL_CHECKPOINT_ARTIFACT_MISSING")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("FORMAL_ARTIFACT_PATH_ESCAPE") from None
    if not candidate.is_file():
        raise ValueError("FORMAL_SAMPLE_ARTIFACT_MISSING")
    return candidate


def _validate_sample_identity(value: Any, sample: Mapping[str, Any]) -> None:
    _require_mapping_fields(value, _SAMPLE_FIELDS, "FORMAL_SAMPLE_FIELDS_INVALID")
    if value["schema_version"] != _SAMPLE_SCHEMA:
        raise ValueError("FORMAL_SAMPLE_SCHEMA_INVALID")
    for field_name in ("chain_key", "blockchain_chain_id", "tx_hash", "sample_id"):
        if value[field_name] != sample[field_name]:
            raise ValueError("FORMAL_SAMPLE_IDENTITY_MISMATCH")
    if not isinstance(value["feature_records"], list):
        raise ValueError("FORMAL_SAMPLE_FEATURE_RECORDS_INVALID")


def _parse_record(value: Any) -> ChainFeatureRecord:
    _require_mapping_fields(value, _RECORD_FIELDS | ({"result_scoring_context"} if isinstance(value, Mapping) and "result_scoring_context" in value else set()), "FORMAL_FEATURE_RECORD_FIELDS_INVALID")
    anchor = value["main_anchor"]
    _require_mapping_fields(anchor, _ANCHOR_FIELDS, "FORMAL_FEATURE_ANCHOR_FIELDS_INVALID")
    calls = value["call_steps"]
    assets = value["asset_behaviors"]
    uncertainties = value["uncertainties"]
    if not isinstance(calls, list) or not isinstance(assets, list) or not isinstance(uncertainties, list):
        raise ValueError("FORMAL_FEATURE_RECORD_STRUCTURE_INVALID")
    try:
        call_steps = []
        for item in calls:
            _require_mapping_fields(item, _CALL_FIELDS, "FORMAL_FEATURE_CALL_FIELDS_INVALID")
            call_steps.append(CallStep(item["effective_address"], item["selector"]))
        asset_behaviors = []
        for item in assets:
            _require_mapping_fields(item, _ASSET_FIELDS, "FORMAL_FEATURE_ASSET_FIELDS_INVALID")
            asset_behaviors.append(AssetBehavior(item["asset_identifier"], item["direction"]))
        if any(not isinstance(reason, str) or not reason for reason in uncertainties):
            raise ValueError("FORMAL_FEATURE_UNCERTAINTY_INVALID")
        return ChainFeatureRecord(
            record_id=value["record_id"],
            main_anchor=MainAnchorKey(
                anchor["blockchain_chain_id"], anchor["effective_address"], anchor["selector"]
            ),
            call_steps=tuple(call_steps),
            asset_behaviors=frozenset(asset_behaviors),
            uncertainties=tuple(uncertainties),
            result_scoring_context=value.get("result_scoring_context", "legacy_independent_weights"),
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("FORMAL_FEATURE_RECORD_INVALID") from None


def _embedded_record_ids(baseline: Mapping[str, Any]) -> set[str]:
    groups = baseline.get("groups")
    if not isinstance(groups, list):
        raise ValueError("FORMAL_EMBEDDED_BASELINE_INVALID")
    ids: list[str] = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("records"), list):
            raise ValueError("FORMAL_EMBEDDED_BASELINE_INVALID")
        for record in group["records"]:
            if not isinstance(record, Mapping) or not isinstance(record.get("record_id"), str):
                raise ValueError("FORMAL_EMBEDDED_BASELINE_INVALID")
            ids.append(record["record_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("FORMAL_EMBEDDED_RECORD_ID_DUPLICATE")
    return set(ids)


def _setting_groups(baseline: NormalBaseline, origin: Mapping[str, str]) -> list[dict[str, Any]]:
    groups = []
    for group in baseline.groups:
        groups.append(
            {
                "main_anchor": group.main_anchor.to_dict(),
                "clean_chain_count": len(group.records),
                "distinct_normal_transaction_count": len({origin[record.record_id] for record in group.records}),
                "calibration_status": group.calibration.status,
                "threshold": group.calibration.to_dict()["threshold"],
            }
        )
    return groups


def _manifest_digest(samples: Any, counts: Any) -> str:
    payload = {
        "chain_counts": {key: counts[key] for key in sorted(counts)},
        "samples": samples,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sample_key(sample: Mapping[str, Any]) -> str:
    return f"{sample['chain_key']}:{sample['tx_hash']}"


def _require_mapping_fields(value: Any, fields: set[str], code: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(code)


def _stable_reason_codes(values: Any) -> tuple[str, ...]:
    supplied = tuple(values)
    if any(
        not isinstance(item, str)
        or not item
        or not item.replace("_", "").isalnum()
        or item != item.upper()
        for item in supplied
    ):
        raise ValueError("FORMAL_NORMAL_IDENTITY_REASON_INVALID")
    return tuple(sorted(set(supplied)))


__all__ = [
    "DEFAULT_FORMAL_NORMAL_BASELINE_DIR",
    "FormalNormalBaselineSummary",
    "FormalNormalTransactionIdentity",
    "LoadedFormalNormalBaseline",
    "NormalRecordTransactionOrigin",
    "load_formal_normal_baseline",
    "resolve_formal_normal_transaction_identity",
    "serialize_formal_normal_baseline_summary",
    "serialize_formal_normal_transaction_identity",
]

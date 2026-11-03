"""Offline evaluability audit for all Ethereum fault localization labels.

This module is an evaluation-only consumer.  It deliberately does not import
or call any Phase1/Phase2 inference, prompt, ranking, RPC, source-network, or
LLM component.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_v2.localization_ground_truth import (
    LocalizationGroundTruthLabel,
    ResolvedLocalizationGroundTruth,
    parse_localization_ground_truth,
    resolve_localization_ground_truth,
)
from src.cloc_v2.source_index import parse_solidity_source
from src.cloc_v2.verified_source_cache_index import (
    VerifiedSourceCacheBinding,
    VerifiedSourceCacheIndex,
    build_verified_source_cache_index,
    lookup_verified_source_cache,
)


DATASET_ROOT = ROOT / "dataset"
CACHE_ROOT = ROOT / "data" / "source_cache"
DEV15_MANIFEST = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "locked_inference_manifest.json"
BLIND10_ROOT = ROOT / "data" / "prototypes" / "localization_fault10_blind_legacy_v4_seed_20260824"
BLIND10_MANIFEST = BLIND10_ROOT / "locked_inference_manifest.json"
BLIND10_FROZEN = BLIND10_ROOT / "frozen_predictions.json"
DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "full_fault_localization_evaluability_20260824" / "evaluability_audit.json"
EXPECTED_ETHEREUM_FAULT_COUNT = 79

_TX = re.compile(r"^0x[0-9a-fA-F]{64}$")
_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_LOOSE_LOCATION = re.compile(r"^(0x[0-9a-fA-F]{40})#([^#]*)#([0-9]+):([0-9]+)$")
_HEX_IDENTITY = re.compile(r"0x[0-9a-fA-F]{40,64}")
_FULL_SHA = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_SAFE_PATH_MARKERS = frozenset({"contracts", "src", "sources"})

_LEVELS = (
    "FILE_FUNCTION_LINE_EVALUABLE",
    "FILE_ONLY_EVALUABLE",
    "LINE_WITHOUT_FUNCTION_UNSAFE",
    "CACHE_MISSING",
    "CACHE_CONFLICT",
    "LOCATION_INVALID",
    "FUNCTION_AMBIGUOUS",
    "SOURCE_OFFSET_OUT_OF_RANGE",
    "EVALUATION_UNAVAILABLE",
)


def _load(path: Path) -> Any:
    try:
        # The audit never indexes or consumes the dataset's `cause` member.
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _flatten(value: Any) -> list[str]:
    result: list[str] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))
        elif isinstance(current, str):
            result.append(current)
    return result


def _dataset_locations(dataset_root: Path) -> dict[str, tuple[str, ...]]:
    rows: dict[str, set[str]] = {}
    for path in sorted(dataset_root.rglob("*.json")):
        value = _load(path)
        if not isinstance(value, Mapping) or value.get("platform") != "Ethereum":
            continue
        fault = value.get("fault")
        if not isinstance(fault, Mapping):
            continue
        transactions = {
            item.lower()
            for item in _flatten(fault.get("transaction_hash"))
            if _TX.fullmatch(item)
        }
        locations = tuple(_flatten(fault.get("location")))
        for transaction in transactions:
            rows.setdefault(transaction, set()).update(locations)
    return {key: tuple(sorted(value)) for key, value in sorted(rows.items())}


def _manifest_transactions(path: Path) -> dict[str, str]:
    value = _load(path)
    if not isinstance(value, Mapping) or not isinstance(value.get("samples"), list):
        raise ValueError("EVALUABILITY_MANIFEST_INVALID")
    result: dict[str, str] = {}
    for item in value["samples"]:
        if not isinstance(item, Mapping):
            raise ValueError("EVALUABILITY_MANIFEST_INVALID")
        alias, transaction = item.get("sample_alias"), item.get("tx_hash")
        if not isinstance(alias, str) or not isinstance(transaction, str) or not _TX.fullmatch(transaction):
            raise ValueError("EVALUABILITY_MANIFEST_INVALID")
        result[transaction.lower()] = alias
    return result


def _single_binding_index(index: VerifiedSourceCacheIndex, label: LocalizationGroundTruthLabel, binding: VerifiedSourceCacheBinding) -> VerifiedSourceCacheIndex:
    return VerifiedSourceCacheIndex(
        index.blockchain_chain_id,
        index.chain_key,
        index.cache_root,
        (binding,),
        0,
        0,
        {label.contract_address: (binding,)},
    )


def _resolve_conflict_without_guessing(label: LocalizationGroundTruthLabel, index: VerifiedSourceCacheIndex) -> tuple[ResolvedLocalizationGroundTruth | None, dict[str, Any]]:
    bindings = index._by_address.get(label.contract_address, ())
    unique = {(item.source_sha256, item.artifact_relative_path): item for item in bindings}
    resolutions: list[tuple[VerifiedSourceCacheBinding, ResolvedLocalizationGroundTruth]] = []
    for binding in sorted(unique.values(), key=lambda item: (item.source_sha256, item.artifact_relative_path)):
        resolved = resolve_localization_ground_truth(label, _single_binding_index(index, label, binding))
        if resolved.source_file_available and resolved.line_range_available:
            resolutions.append((binding, resolved))
    distinct_valid_sources = {item.source_sha256 for item, _ in resolutions}
    safe = len(distinct_valid_sources) == 1
    selected = resolutions[0][1] if safe else None
    return selected, {
        "binding_count": len(bindings),
        "distinct_source_sha_count": len({item.source_sha256 for item in bindings}),
        "same_sha_duplicate_binding_count": len(bindings) - len({item.source_sha256 for item in bindings}),
        "direct_proxy_mixed": {item.proxy_resolution_source for item in bindings} == {"DIRECT_VERIFIED_CACHE", "VERIFIED_PROXY_IMPLEMENTATION"},
        "safe_truth_only_resolution": safe,
        "valid_source_variant_count": len(distinct_valid_sources),
    }


def _safe_absolute_path_label(raw: str) -> LocalizationGroundTruthLabel | None:
    match = _LOOSE_LOCATION.fullmatch(raw.strip()) if isinstance(raw, str) else None
    if match is None:
        return None
    address, file_value, offset, length = match.groups()
    if not _ADDRESS.fullmatch(address) or int(length) <= 0:
        return None
    normalized = file_value.replace("\\", "/")
    parts = [item for item in normalized.split("/") if item and item != "."]
    marker_positions = [index for index, item in enumerate(parts) if item.lower() in _SAFE_PATH_MARKERS]
    if not marker_positions:
        return None
    suffix = PurePosixPath(*parts[marker_positions[-1]:]).as_posix()
    try:
        return parse_localization_ground_truth(f"{address}#{suffix}#{offset}:{length}")
    except ValueError:
        return None


def _resolution_gap(resolved: ResolvedLocalizationGroundTruth, label: LocalizationGroundTruthLabel, index: VerifiedSourceCacheIndex) -> str | None:
    if resolved.function_available or not resolved.source_file_available:
        return None
    lookup = lookup_verified_source_cache(index, runtime_address=label.contract_address)
    if lookup.status != "VERIFIED_SOURCE_CACHE_HIT" or lookup.binding is None or resolved.normalized_source_file is None:
        return "FUNCTION_SOURCE_UNAVAILABLE"
    artifact = index.cache_root / lookup.binding.artifact_relative_path
    manifest = _load(artifact / "manifest.json")
    if not isinstance(manifest, Mapping):
        return "FUNCTION_SOURCE_UNAVAILABLE"
    matches = []
    for item in manifest.get("files", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            continue
        normalized = item["path"].replace("\\", "/").lstrip("./")
        if normalized != resolved.normalized_source_file:
            continue
        path = (artifact / normalized).resolve()
        try:
            path.relative_to(artifact.resolve())
            text = path.read_text(encoding="utf-8")
        except (ValueError, OSError, UnicodeError):
            return "FUNCTION_SOURCE_UNAVAILABLE"
        matches.append(parse_solidity_source(text))
    if len(matches) != 1:
        return "FUNCTION_SOURCE_UNAVAILABLE"
    parsed = matches[0]
    if parsed.get("parse_status") != "PARSED":
        return "SOURCE_INDEX_PARSE_FAILED"
    if "GROUND_TRUTH_FUNCTION_AMBIGUOUS" in resolved.reason_codes:
        return "FUNCTION_INDEX_AMBIGUOUS"
    return "LOCATION_OUTSIDE_ALL_PARSED_FUNCTIONS"


def _best_level(resolutions: Sequence[ResolvedLocalizationGroundTruth], parsed_count: int, location_count: int) -> str:
    if any(item.source_file_available and item.function_available and item.line_range_available for item in resolutions):
        return "FILE_FUNCTION_LINE_EVALUABLE"
    reasons = {reason for item in resolutions for reason in item.reason_codes}
    if "GROUND_TRUTH_FUNCTION_AMBIGUOUS" in reasons:
        return "FUNCTION_AMBIGUOUS"
    if any(item.source_file_available and item.line_range_available for item in resolutions):
        return "LINE_WITHOUT_FUNCTION_UNSAFE"
    if any(item.source_file_available for item in resolutions):
        return "FILE_ONLY_EVALUABLE"
    if "GROUND_TRUTH_SOURCE_CACHE_CONFLICT" in reasons:
        return "CACHE_CONFLICT"
    if "GROUND_TRUTH_SOURCE_CACHE_MISSING" in reasons:
        return "CACHE_MISSING"
    if "GROUND_TRUTH_OFFSET_OUT_OF_RANGE" in reasons or "GROUND_TRUTH_OFFSET_ENCODING_INVALID" in reasons:
        return "SOURCE_OFFSET_OUT_OF_RANGE"
    if location_count and not parsed_count:
        return "LOCATION_INVALID"
    return "EVALUATION_UNAVAILABLE"


def _audit_transaction(alias: str, locations: Sequence[str], index: VerifiedSourceCacheIndex, group: str) -> dict[str, Any]:
    labels: list[LocalizationGroundTruthLabel] = []
    invalid_count = 0
    address_valid = False
    for raw in locations:
        first = raw.split("#", 1)[0] if isinstance(raw, str) else ""
        address_valid = address_valid or bool(_ADDRESS.fullmatch(first))
        try:
            labels.append(parse_localization_ground_truth(raw))
        except ValueError:
            invalid_count += 1
    resolutions = [resolve_localization_ground_truth(label, index) for label in labels]
    lookups = [lookup_verified_source_cache(index, runtime_address=label.contract_address) for label in labels]
    gaps = sorted({gap for label, resolved in zip(labels, resolutions) if (gap := _resolution_gap(resolved, label, index))})

    repair_resolutions: list[ResolvedLocalizationGroundTruth] = []
    conflict_details: list[dict[str, Any]] = []
    for label, lookup in zip(labels, lookups):
        if lookup.status == "VERIFIED_SOURCE_CACHE_CONFLICT":
            repaired, detail = _resolve_conflict_without_guessing(label, index)
            conflict_details.append(detail)
            if repaired is not None:
                repair_resolutions.append(repaired)
    for raw in locations:
        if any(item.raw_location == raw for item in labels):
            continue
        recovered = _safe_absolute_path_label(raw)
        if recovered is not None:
            repaired = resolve_localization_ground_truth(recovered, index)
            if repaired.source_file_available:
                repair_resolutions.append(repaired)

    current_level = _best_level(resolutions, len(labels), len(locations))
    repaired_all = [*resolutions, *repair_resolutions]
    repair_level = _best_level(repaired_all, len(labels) + len(repair_resolutions), len(locations))
    reason_codes = sorted({reason for item in resolutions for reason in item.reason_codes})
    return {
        "sample_alias": alias,
        "group": group,
        "location_present": bool(locations),
        "strict_location_parseable": bool(labels),
        "invalid_location_label_count": invalid_count,
        "source_file_reported": any(label.source_file is not None for label in labels),
        "contract_address_valid": address_valid,
        "verified_source_cache_present": any(item.status in {"VERIFIED_SOURCE_CACHE_HIT", "VERIFIED_SOURCE_CACHE_CONFLICT"} for item in lookups),
        "verified_source_cache_unique": any(item.status == "VERIFIED_SOURCE_CACHE_HIT" for item in lookups),
        "verified_source_cache_conflict": any(item.status == "VERIFIED_SOURCE_CACHE_CONFLICT" for item in lookups),
        "utf8_offset_in_source": any(item.line_range_available for item in resolutions),
        "source_file_unique": any(item.source_file_available for item in resolutions),
        "source_function_unique": any(item.function_available for item in resolutions),
        "absolute_line_range_available": any(item.line_range_available for item in resolutions),
        "function_resolution_gaps": gaps,
        "reason_codes": reason_codes,
        "evaluability_level": current_level,
        "repair_without_prediction_possible": any(item.function_available for item in repair_resolutions),
        "post_safe_repair_level": repair_level,
        "conflict_resolution_audit": conflict_details,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    levels = Counter(str(item["evaluability_level"]) for item in rows)
    return {
        "transaction_count": len(rows),
        "strict_location_parseable_count": sum(bool(item["strict_location_parseable"]) for item in rows),
        "file_evaluable_count": sum(bool(item["source_file_unique"]) for item in rows),
        "function_evaluable_count": sum(bool(item["source_function_unique"]) for item in rows),
        "line_range_evaluable_count": sum(bool(item["absolute_line_range_available"]) for item in rows),
        "file_without_function_count": sum(bool(item["source_file_unique"]) and not bool(item["source_function_unique"]) for item in rows),
        "cache_missing_count": levels["CACHE_MISSING"],
        "cache_conflict_count": sum(bool(item["verified_source_cache_conflict"]) for item in rows),
        "function_ambiguous_count": levels["FUNCTION_AMBIGUOUS"],
        "location_invalid_count": levels["LOCATION_INVALID"],
        "level_counts": {key: levels[key] for key in _LEVELS},
        "post_safe_repair_file_function_line_count": sum(item["post_safe_repair_level"] == "FILE_FUNCTION_LINE_EVALUABLE" for item in rows),
    }


def _blind_conflicts(frozen_path: Path, index: VerifiedSourceCacheIndex) -> dict[str, Any]:
    frozen = _load(frozen_path)
    rows = frozen.get("predictions", []) if isinstance(frozen, Mapping) else []
    aliases = sorted(
        item.get("sample_alias")
        for item in rows
        if isinstance(item, Mapping)
        and item.get("status") == "SOURCE_FAILURE"
        and "VERIFIED_SOURCE_CACHE_CONFLICT" in item.get("reason_codes", [])
        and isinstance(item.get("sample_alias"), str)
    )
    buckets = [values for values in index._by_address.values() if len({item.source_sha256 for item in values}) > 1]
    uniform_sha_count = sorted({len({item.source_sha256 for item in values}) for values in buckets})
    all_mixed = bool(buckets) and all({item.proxy_resolution_source for item in values} == {"DIRECT_VERIFIED_CACHE", "VERIFIED_PROXY_IMPLEMENTATION"} for values in buckets)
    all_not_duplicate_only = bool(buckets) and all(len({item.source_sha256 for item in values}) > 1 for values in buckets)
    per_sample = [{
        "sample_alias": alias,
        "conflict_address_persisted_in_sanitized_prediction": False,
        "distinct_source_sha_count": uniform_sha_count[0] if len(uniform_sha_count) == 1 else None,
        "same_source_duplicate_only": not all_not_duplicate_only,
        "runtime_proxy_effective_mixed": all_mixed,
        "safe_offline_resolution": False,
        "reason_code": "CONFLICT_QUERY_IDENTITY_NOT_PERSISTED",
    } for alias in aliases]
    return {
        "affected_sample_count": len(aliases),
        "global_conflict_address_count": len(buckets),
        "distinct_source_sha_counts_across_conflict_addresses": uniform_sha_count,
        "all_conflicts_mix_direct_and_verified_proxy_bindings": all_mixed,
        "all_conflicts_have_distinct_source_content": all_not_duplicate_only,
        "samples": per_sample,
        "conclusion": "CANNOT_SAFELY_DISAMBIGUATE_FROZEN_SAMPLE_CONFLICTS",
    }


def build_full_fault_evaluability_audit(
    *,
    dataset_root: Path,
    cache_root: Path,
    dev15_manifest: Path,
    blind10_manifest: Path,
    blind10_frozen: Path,
    expected_transaction_count: int = EXPECTED_ETHEREUM_FAULT_COUNT,
) -> dict[str, Any]:
    locations_by_transaction = _dataset_locations(dataset_root)
    if len(locations_by_transaction) != expected_transaction_count:
        raise ValueError("ETHEREUM_FAULT_TRANSACTION_COUNT_INVALID")
    dev = _manifest_transactions(dev15_manifest)
    blind = _manifest_transactions(blind10_manifest)
    overlap = set(dev) & set(blind)
    if overlap:
        raise ValueError("AUDIT_GROUP_OVERLAP")
    if not set(dev).issubset(locations_by_transaction) or not set(blind).issubset(locations_by_transaction):
        raise ValueError("AUDIT_MANIFEST_TRANSACTION_MISSING")

    index = build_verified_source_cache_index(cache_root, blockchain_chain_id=1, chain_key="eth")
    aliases = {transaction: f"fault_audit_eth_{number:03d}" for number, transaction in enumerate(sorted(locations_by_transaction), 1)}
    rows = []
    for transaction in sorted(locations_by_transaction):
        group = "HISTORICAL_DEV15" if transaction in dev else "LATEST_BLIND10" if transaction in blind else "UNRUN_REMAINDER"
        row = _audit_transaction(aliases[transaction], locations_by_transaction[transaction], index, group)
        if transaction in blind:
            row["blind10_alias"] = blind[transaction]
        rows.append(row)

    groups = {
        name: _summary([item for item in rows if item["group"] == name])
        for name in ("HISTORICAL_DEV15", "LATEST_BLIND10", "UNRUN_REMAINDER")
    }
    blind_details = []
    for item in rows:
        if item["group"] != "LATEST_BLIND10":
            continue
        missing_layer = (
            "LOCATION_FORMAT" if item["evaluability_level"] == "LOCATION_INVALID"
            else "VERIFIED_SOURCE_CACHE" if item["evaluability_level"] == "CACHE_MISSING"
            else "FUNCTION_INDEX" if not item["source_function_unique"] and item["source_file_unique"]
            else "NONE"
        )
        blind_details.append({
            "sample_alias": item["blind10_alias"],
            "file_evaluable": item["source_file_unique"],
            "function_evaluable": item["source_function_unique"],
            "line_range_evaluable": item["absolute_line_range_available"],
            "evaluability_level": item["evaluability_level"],
            "missing_layer": missing_layer,
            "function_resolution_gaps": item["function_resolution_gaps"],
            "reason_codes": item["reason_codes"],
            "recoverable_by_evaluation_only_repair": item["repair_without_prediction_possible"],
        })
    blind_details.sort(key=lambda item: item["sample_alias"])

    overall = _summary(rows)
    structural_upper_bound = overall["strict_location_parseable_count"] + sum(
        item["evaluability_level"] == "LOCATION_INVALID" and item["post_safe_repair_level"] != "LOCATION_INVALID"
        for item in rows
    )
    return {
        "schema_version": "full_fault_localization_evaluability_audit_v1",
        "audit_mode": "POST_FREEZE_EVALUATION_ONLY",
        "scientific_generalization_result": False,
        "cause_read": False,
        "location_used_for_inference": False,
        "rpc_request_count": 0,
        "source_network_request_count": 0,
        "deepseek_request_count": 0,
        "other_network_request_count": 0,
        "cache_modified": False,
        "inference_invoked": False,
        "cache_index_summary": index.summary(),
        "overall": overall,
        "groups": groups,
        "group_overlap_count": 0,
        "samples": rows,
        "latest_blind10": {
            "function_denominator_zero_reason_counts": dict(sorted(Counter(item["missing_layer"] for item in blind_details).items())),
            "recoverable_by_evaluation_only_repair_count": sum(item["recoverable_by_evaluation_only_repair"] for item in blind_details),
            "samples": blind_details,
        },
        "blind10_inference_cache_conflicts": _blind_conflicts(blind10_frozen, index),
        "maximum_fully_evaluable_with_current_verified_cache_and_safe_repairs": overall["post_safe_repair_file_function_line_count"],
        "structural_upper_bound_if_all_missing_verified_sources_are_later_supplied": structural_upper_bound,
        "recommendation": "EVALUATION_TRUTH_REPAIR_REQUIRED",
        "minimum_repair_scope": [
            "ADD_EVALUATION_ONLY_SAFE_ABSOLUTE_PATH_SUFFIX_NORMALIZATION",
            "ADD_TRUTH_ONLY_ROLE_AWARE_CACHE_BINDING_RESOLUTION",
            "ACQUIRE_OR_IMPORT_MISSING_VERIFIED_SOURCES_WITHOUT_USING_PREDICTIONS",
            "KEEP_OUTSIDE_FUNCTION_AND_AMBIGUOUS_LABELS_UNEVALUABLE",
        ],
    }


def _validate_public(value: Mapping[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if _HEX_IDENTITY.search(text) or _FULL_SHA.search(text):
        raise ValueError("EVALUABILITY_AUDIT_SENSITIVE_DATA")
    forbidden = {"transaction_hash", "tx_hash", "address", "location", "cause", "source_text", "cache_path", "artifact_relative_path"}
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if any(str(key).lower() in forbidden for key in current):
                raise ValueError("EVALUABILITY_AUDIT_SENSITIVE_DATA")
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def write_audit(path: Path, audit: Mapping[str, Any]) -> None:
    _validate_public(audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            json.dump(dict(audit), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        Path(handle.name).replace(path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        print("audit_status=STOPPED")
        print("reason_code=OUTPUT_ALREADY_EXISTS")
        return 1
    audit = build_full_fault_evaluability_audit(
        dataset_root=DATASET_ROOT,
        cache_root=CACHE_ROOT,
        dev15_manifest=DEV15_MANIFEST,
        blind10_manifest=BLIND10_MANIFEST,
        blind10_frozen=BLIND10_FROZEN,
    )
    write_audit(args.output, audit)
    print("audit_status=COMPLETED")
    print("ethereum_fault_transaction_count=" + str(audit["overall"]["transaction_count"]))
    print("rpc_request_count=0")
    print("source_network_request_count=0")
    print("deepseek_request_count=0")
    print("other_network_request_count=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

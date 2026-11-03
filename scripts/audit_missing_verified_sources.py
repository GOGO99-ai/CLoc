"""Backfill verified source for the fixed Ethereum cache-miss set."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloc.runtime.execution_policy import ExecutionPolicy
from scripts.audit_full_fault_localization_evaluability import _dataset_locations
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.localization_ground_truth import parse_localization_ground_truth_repair_v1, resolve_localization_ground_truth_repair_v1
from src.cloc_v2.source_cache import atomic_json
from src.cloc_v2.source_fetcher import SourceFetcher, _StdlibSession
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index, lookup_verified_source_cache

DATASET_ROOT = ROOT / "dataset"
CACHE_ROOT = ROOT / "data" / "source_cache"
DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "missing_verified_source_backfill_20260831"
EXPECTED_ETHEREUM_FAULTS = 79
EXPECTED_MISSING_TRANSACTIONS = 45
SOURCE_REQUEST_LIMIT = 90
DIRECT = "DIRECT_SOURCE_AVAILABLE"
PROXY = "PROXY_IMPLEMENTATION_SOURCE_AVAILABLE"
BINDING = "EXISTING_CACHE_BINDING_MISSING"
UNVERIFIED = "UNVERIFIED_SOURCE"
FAILED = "REQUEST_FAILED"
_STOP_NETWORK = frozenset({"network_error", "timeout", "rate_limited"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _content_sha256(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical({key: item for key, item in value.items() if key != field})).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_status(cache_root: Path, address: str) -> str | None:
    path = cache_root / "eth" / address / "manifest.json"
    if not path.is_file():
        return None
    try:
        value = _load(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "invalid"
    return str(value.get("status") or "invalid") if isinstance(value, Mapping) else "invalid"


def build_locked_manifest(dataset_root: Path = DATASET_ROOT, cache_root: Path = CACHE_ROOT, *, expected_faults: int = EXPECTED_ETHEREUM_FAULTS, expected_missing: int = EXPECTED_MISSING_TRANSACTIONS) -> dict[str, Any]:
    """Lock all current Ethereum faults whose truth addresses miss the cache."""
    locations = _dataset_locations(dataset_root)
    if len(locations) != expected_faults:
        raise ValueError("ETHEREUM_FAULT_TRANSACTION_COUNT_INVALID")
    index = build_verified_source_cache_index(cache_root, blockchain_chain_id=1, chain_key="eth")
    selected: list[tuple[str, tuple[str, ...]]] = []
    all_addresses: set[str] = set()
    for transaction, raw_locations in sorted(locations.items()):
        labels = tuple(parse_localization_ground_truth_repair_v1(raw) for raw in raw_locations)
        states = tuple(lookup_verified_source_cache(index, runtime_address=label.contract_address).status for label in labels)
        if states and all(status == "VERIFIED_SOURCE_CACHE_MISS" for status in states):
            addresses = tuple(sorted({label.contract_address for label in labels}))
            selected.append((transaction, addresses))
            all_addresses.update(addresses)
    if len(selected) != expected_missing:
        raise ValueError("VERIFIED_SOURCE_MISSING_TRANSACTION_COUNT_INVALID")
    aliases = {address: f"source_address_{number:03d}" for number, address in enumerate(sorted(all_addresses), 1)}
    manifest: dict[str, Any] = {
        "schema_version": "missing_verified_source_lock_v1",
        "selection_rule": "ALL_ETHEREUM_FAULT_TRANSACTIONS_WITH_TRUTH_ADDRESS_CACHE_MISS",
        "blockchain_chain_id": 1,
        "transaction_count": len(selected),
        "deduplicated_contract_address_count": len(all_addresses),
        "prediction_used_for_selection": False,
        "cause_read": False,
        "transactions": [{"sample_alias": f"missing_source_tx_{number:03d}", "transaction_hash": transaction, "address_aliases": [aliases[address] for address in addresses]} for number, (transaction, addresses) in enumerate(selected, 1)],
        "addresses": [{"address_alias": aliases[address], "contract_address": address} for address in sorted(all_addresses)],
    }
    manifest["manifest_sha256"] = _content_sha256(manifest, "manifest_sha256")
    return manifest


def validate_locked_manifest(value: Mapping[str, Any], *, expected_missing: int = EXPECTED_MISSING_TRANSACTIONS) -> None:
    if value.get("schema_version") != "missing_verified_source_lock_v1":
        raise ValueError("SOURCE_BACKFILL_MANIFEST_SCHEMA_INVALID")
    if value.get("transaction_count") != expected_missing:
        raise ValueError("SOURCE_BACKFILL_MANIFEST_TRANSACTION_COUNT_INVALID")
    addresses = value.get("addresses")
    if not isinstance(addresses, list) or len(addresses) != value.get("deduplicated_contract_address_count"):
        raise ValueError("SOURCE_BACKFILL_MANIFEST_ADDRESS_COUNT_INVALID")
    if value.get("manifest_sha256") != _content_sha256(value, "manifest_sha256"):
        raise ValueError("SOURCE_BACKFILL_MANIFEST_SHA_INVALID")


def _evaluability(dataset_root: Path, cache_root: Path) -> dict[str, Any]:
    locations = _dataset_locations(dataset_root)
    index = build_verified_source_cache_index(cache_root, blockchain_chain_id=1, chain_key="eth")
    file_count = function_count = line_count = full_count = 0
    for raw_locations in locations.values():
        rows = [resolve_localization_ground_truth_repair_v1(parse_localization_ground_truth_repair_v1(raw), index) for raw in raw_locations]
        file_ok = any(row.source_file_available for row in rows)
        function_ok = any(row.function_available for row in rows)
        line_ok = any(row.line_range_available for row in rows)
        file_count += int(file_ok)
        function_count += int(function_ok)
        line_count += int(line_ok)
        full_count += int(file_ok and function_ok and line_ok)
    total = len(locations)
    return {"transaction_count": total, "file_evaluable_count": file_count, "function_evaluable_count": function_count, "line_range_evaluable_count": line_count, "fully_evaluable_count": full_count, "function_evaluation_coverage": function_count / total if total else None}


class SourceRequestBudgetExceeded(RuntimeError):
    pass


class CountingSession:
    """Checkpoint a physical Explorer request before sending it."""
    def __init__(self, delegate: Any, before_request: Any, limit: int = SOURCE_REQUEST_LIMIT):
        self.delegate, self.before_request, self.limit = delegate, before_request, limit

    def get(self, url: str, params: Mapping[str, Any], timeout: int) -> Any:
        if int(self.before_request()) > self.limit:
            raise SourceRequestBudgetExceeded("SOURCE_REQUEST_BUDGET_EXCEEDED")
        return self.delegate.get(url, params=params, timeout=timeout)


def _initial_checkpoint(manifest: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "missing_verified_source_backfill_checkpoint_v1",
        "status": "LOCKED",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_request_limit": SOURCE_REQUEST_LIMIT,
        "physical_source_request_count": 0,
        "attempted_address_aliases": [],
        "completed_address_aliases": [],
        "address_results": [],
        "evaluability_before": dict(before),
        "stop_reason": None,
        "rpc_request_count": 0,
        "deepseek_request_count": 0,
        "trace_request_count": 0,
    }


def _classification(result: Any, preexisting_binding_gap: bool) -> str:
    if preexisting_binding_gap:
        return BINDING
    if bool(getattr(result, "verified", False)):
        proxy = result.proxy if isinstance(getattr(result, "proxy", None), Mapping) else {}
        return PROXY if proxy.get("is_proxy") and proxy.get("resolution") == "verified" else DIRECT
    return UNVERIFIED if getattr(result, "status", None) == "unverified" else FAILED


def _safe_result(alias: str, classification: str, result: Any, *, preexisting_binding_gap: bool) -> dict[str, Any]:
    proxy = result.proxy if isinstance(getattr(result, "proxy", None), Mapping) else {}
    return {
        "address_alias": alias,
        "classification": classification,
        "fetch_status": str(getattr(result, "status", "unknown")),
        "verified": bool(getattr(result, "verified", False)),
        "cache_hit": bool(getattr(result, "cache_hit", False)),
        "solidity_file_count": len(getattr(result, "files", ()) or ()),
        "proxy_reported": bool(proxy.get("is_proxy")),
        "proxy_resolution_status": str(proxy.get("resolution") or "NOT_APPLICABLE"),
        "preexisting_cache_binding_gap": preexisting_binding_gap,
    }


def _write_final_report(checkpoint: dict[str, Any], manifest: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    categories = Counter(row["classification"] for row in checkpoint["address_results"])
    report = {
        "schema_version": "missing_verified_source_backfill_audit_v1",
        "audit_status": checkpoint["status"],
        "locked_transaction_count": manifest["transaction_count"],
        "locked_contract_address_count": manifest["deduplicated_contract_address_count"],
        "processed_contract_address_count": len(checkpoint["completed_address_aliases"]),
        "classification_counts": dict(sorted(categories.items())),
        "successful_source_address_count": categories[DIRECT] + categories[PROXY] + categories[BINDING],
        "request_failure_address_count": categories[FAILED],
        "physical_source_request_count": checkpoint["physical_source_request_count"],
        "source_request_limit": SOURCE_REQUEST_LIMIT,
        "retry_count": 0,
        "evaluability_before": dict(before),
        "evaluability_after": dict(after),
        "new_file_evaluable_count": after["file_evaluable_count"] - before["file_evaluable_count"],
        "new_function_evaluable_count": after["function_evaluable_count"] - before["function_evaluable_count"],
        "new_line_range_evaluable_count": after["line_range_evaluable_count"] - before["line_range_evaluable_count"],
        "prediction_used_for_selection": False,
        "cause_read": False,
        "rpc_request_count": 0,
        "trace_request_count": 0,
        "deepseek_request_count": 0,
        "prompt_or_localization_algorithm_modified": False,
        "address_results": checkpoint["address_results"],
    }
    atomic_json(report_path, report)
    return report


def run_backfill(output_dir: Path = DEFAULT_OUTPUT, *, dataset_root: Path = DATASET_ROOT, cache_root: Path = CACHE_ROOT, config: Mapping[str, Any] | None = None, session: Any | None = None, expected_faults: int = EXPECTED_ETHEREUM_FAULTS, expected_missing: int = EXPECTED_MISSING_TRANSACTIONS) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "locked_manifest.json"
    checkpoint_path = output_dir / "checkpoint.json"
    report_path = output_dir / "verified_source_backfill_audit.json"
    current_manifest = build_locked_manifest(dataset_root, cache_root, expected_faults=expected_faults, expected_missing=expected_missing)
    if manifest_path.exists():
        manifest = _load(manifest_path)
        validate_locked_manifest(manifest, expected_missing=expected_missing)
        if manifest != current_manifest:
            raise ValueError("SOURCE_BACKFILL_LOCK_CHANGED")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest = current_manifest
        atomic_json(manifest_path, manifest)
    if report_path.exists():
        raise FileExistsError("SOURCE_BACKFILL_REPORT_ALREADY_EXISTS")

    before = _evaluability(dataset_root, cache_root)
    checkpoint = _load(checkpoint_path) if checkpoint_path.exists() else _initial_checkpoint(manifest, before)
    if checkpoint.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("SOURCE_BACKFILL_CHECKPOINT_LOCK_MISMATCH")
    atomic_json(checkpoint_path, checkpoint)
    chain_config = dict(config or load_chain_config("eth"))
    if not chain_config.get("key"):
        checkpoint.update(status="STOPPED_CONFIGURATION_MISSING", stop_reason="SOURCE_API_KEY_MISSING")
        atomic_json(checkpoint_path, checkpoint)
        raise RuntimeError("SOURCE_API_KEY_MISSING")

    active = {row["address_alias"]: row["contract_address"] for row in manifest["addresses"]}
    attempted = set(checkpoint.get("attempted_address_aliases", ()))
    completed = set(checkpoint.get("completed_address_aliases", ()))

    def before_request() -> int:
        checkpoint["physical_source_request_count"] = int(checkpoint["physical_source_request_count"]) + 1
        checkpoint["status"] = "RUNNING"
        atomic_json(checkpoint_path, checkpoint)
        return int(checkpoint["physical_source_request_count"])

    fetcher = SourceFetcher(
        "eth",
        chain_config,
        cache_root=cache_root,
        policy=ExecutionPolicy(allow_explorer=True, allow_artifact_cache_write=True),
        session=CountingSession(session or _StdlibSession(), before_request),
        max_proxy_depth=1,
        max_retries=1,
    )
    stopped = False
    for alias in sorted(active):
        if alias in completed or alias in attempted:
            continue
        address = active[alias]
        old_index = build_verified_source_cache_index(cache_root, blockchain_chain_id=1, chain_key="eth")
        old_lookup = lookup_verified_source_cache(old_index, runtime_address=address, requested_address=address)
        binding_gap = _manifest_status(cache_root, address) == "verified" and old_lookup.status == "VERIFIED_SOURCE_CACHE_MISS"
        attempted.add(alias)
        checkpoint.update(status="RUNNING", attempted_address_aliases=sorted(attempted), current_address_alias=alias)
        atomic_json(checkpoint_path, checkpoint)
        try:
            result = fetcher.acquire(address, refresh=True, origins=({"audit_id": "MISSING_VERIFIED_SOURCE_BACKFILL_V1", "address_alias": alias},))
        except SourceRequestBudgetExceeded:
            checkpoint.update(status="STOPPED_BUDGET_EXCEEDED", stop_reason="SOURCE_REQUEST_BUDGET_EXCEEDED")
            atomic_json(checkpoint_path, checkpoint)
            stopped = True
            break
        checkpoint["address_results"].append(_safe_result(alias, _classification(result, binding_gap), result, preexisting_binding_gap=binding_gap))
        completed.add(alias)
        checkpoint["completed_address_aliases"] = sorted(completed)
        checkpoint.pop("current_address_alias", None)
        proxy_resolution = str((result.proxy or {}).get("resolution") or "") if isinstance(result.proxy, Mapping) else ""
        if result.status in _STOP_NETWORK or proxy_resolution in _STOP_NETWORK:
            checkpoint.update(status="STOPPED_NETWORK_ERROR", stop_reason="SOURCE_NETWORK_ERROR")
            atomic_json(checkpoint_path, checkpoint)
            stopped = True
            break
        atomic_json(checkpoint_path, checkpoint)

    after = _evaluability(dataset_root, cache_root)
    categories = Counter(row["classification"] for row in checkpoint["address_results"])
    checkpoint["status"] = checkpoint["status"] if stopped else "COMPLETED"
    checkpoint["evaluability_after"] = after
    checkpoint["classification_counts"] = dict(sorted(categories.items()))
    checkpoint.pop("current_address_alias", None)
    atomic_json(checkpoint_path, checkpoint)
    return _write_final_report(checkpoint, manifest, before, after, report_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_backfill(args.output_dir)
    fields = (
        ("audit_status", result["audit_status"]),
        ("locked_transaction_count", result["locked_transaction_count"]),
        ("locked_contract_address_count", result["locked_contract_address_count"]),
        ("successful_source_address_count", result["successful_source_address_count"]),
        ("physical_source_request_count", result["physical_source_request_count"]),
        ("function_evaluable_count", result["evaluability_after"]["function_evaluable_count"]),
    )
    for key, value in fields:
        print(f"{key}={value}")
    print("rpc_request_count=0")
    print("deepseek_request_count=0")
    return 0 if result["audit_status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

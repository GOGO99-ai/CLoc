"""Build a deterministic acquisition manifest from the native CLoc dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_SCHEMA_VERSION = "cloc_baseline_manifest_v1"
_TX_HASH = re.compile(r"^0x[0-9a-fA-F]{64}$")
_PLATFORMS = {
    "Ethereum": {"chain_id": 1, "chain_key": "eth"},
    "BNBChain": {"chain_id": 56, "chain_key": "bsc"},
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(values: Any) -> list[str]:
    output: list[str] = []
    if isinstance(values, list):
        for value in values:
            output.extend(_flatten(value))
    elif isinstance(values, str):
        output.append(values)
    return output


def _transaction_groups(values: Any) -> list[list[str]]:
    if not isinstance(values, list):
        return []
    if all(isinstance(item, str) for item in values):
        return [[item.lower() for item in values if _TX_HASH.fullmatch(item)]]
    groups: list[list[str]] = []
    for group in values:
        normalized = [item.lower() for item in _flatten(group) if _TX_HASH.fullmatch(item)]
        if normalized:
            groups.append(list(dict.fromkeys(normalized)))
    return groups


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "case"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal_sample_status(samples_root: Path) -> dict[str, bool]:
    status: dict[str, bool] = {}
    if not samples_root.exists():
        return status
    for path in samples_root.rglob("*.json"):
        try:
            raw = _load(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(raw, Mapping):
            continue
        transaction = raw.get("tx_hash") or raw.get("transaction_hash") or path.stem
        if isinstance(transaction, str) and _TX_HASH.fullmatch(transaction):
            status[transaction.lower()] = bool(raw.get("feature_records"))
    return status


def build_cloc_baseline_manifest(
    *,
    dataset_root: str | Path,
    call_tree_root: str | Path | None = None,
    normal_samples_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_root)
    tree_hashes = {
        path.stem.lower()
        for path in Path(call_tree_root).rglob("*.json")
    } if call_tree_root is not None and Path(call_tree_root).exists() else set()
    sample_status = _normal_sample_status(Path(normal_samples_root)) if normal_samples_root is not None else {}
    rows: list[dict[str, Any]] = []
    names: Counter[str] = Counter()
    for index, path in enumerate(sorted(dataset_path.rglob("*.json")), start=1):
        raw = _load(path)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Dataset case must be an object: {path}")
        name = raw.get("name") if isinstance(raw.get("name"), str) else path.stem
        names[name] += 1
        platform = raw.get("platform")
        if platform not in _PLATFORMS:
            raise ValueError(f"Unsupported platform in {path}: {platform!r}")
        fault = raw.get("fault") if isinstance(raw.get("fault"), Mapping) else {}
        faultless = raw.get("faultless") if isinstance(raw.get("faultless"), Mapping) else {}
        groups = _transaction_groups(fault.get("transaction_hash"))
        if not groups:
            raise ValueError(f"No valid fault transactions in {path}")
        all_faults = {transaction for group in groups for transaction in group}
        tree_matches = sorted(all_faults & tree_hashes)
        if len(tree_matches) > 1:
            raise ValueError(f"Multiple locked call trees match one CLoc case: {path}")
        # CLoc evaluates one representative incident transaction per JSON case.
        # The checked-in all_tree cache is the lock that identifies that sample.
        selected_faults = tree_matches or groups[0][:1]
        normal_transactions = [
            item.lower()
            for item in _flatten(faultless.get("transaction_hash"))
            if _TX_HASH.fullmatch(item)
        ]
        normal_transactions = list(dict.fromkeys(normal_transactions))
        locations = [item for item in _flatten(fault.get("location")) if isinstance(item, str)]
        local_fault_trees = [item for item in selected_faults if item in tree_hashes]
        cached_normal = [item for item in normal_transactions if item in sample_status]
        usable_cached_normal = [item for item in normal_transactions if sample_status.get(item)]
        case_id = f"{index:03d}_{_slug(name)}"
        chain = _PLATFORMS[platform]
        rows.append(
            {
                "case_id": case_id,
                "case_name": name,
                "category": path.parent.name,
                "cause": raw.get("cause") if isinstance(raw.get("cause"), str) else "",
                "platform": platform,
                **chain,
                "dataset_file": path.relative_to(dataset_path).as_posix(),
                "dataset_file_sha256": _sha256(path),
                "all_fault_transaction_groups": groups,
                "selected_fault_transactions": selected_faults,
                "normal_transactions": normal_transactions,
                "fault_locations": locations,
                "local_artifacts": {
                    "selected_fault_call_tree_count": len(local_fault_trees),
                    "normal_call_feature_file_count": len(cached_normal),
                    "usable_normal_call_feature_count": len(usable_cached_normal),
                },
                "collection": {
                    "required_fault_trace_count": len(selected_faults),
                    "required_normal_trace_count": len(normal_transactions),
                    "required_total_trace_count": len(selected_faults) + len(normal_transactions),
                    "function_coverage_artifact": f"coverage/{case_id}.json",
                    "grace_graph_artifact": f"grace/{case_id}.json",
                },
                "readiness": {
                    "manifest_ready": True,
                    "function_coverage_ready": False,
                    "grace_graph_ready": False,
                    "reason_codes": [
                        "FUNCTION_LEVEL_OPCODE_SOURCE_MAP_REQUIRED",
                        "CALLTRACER_CACHE_NOT_EQUIVALENT_TO_INTERNAL_FUNCTION_COVERAGE",
                    ],
                },
            }
        )
    if not rows:
        raise ValueError(f"No dataset cases found under {dataset_path}")
    duplicate_names = sorted(name for name, count in names.items() if count > 1)
    all_normal = [tx for row in rows for tx in row["normal_transactions"]]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": {
            "fault_transaction_policy": "ONE_REPRESENTATIVE_TRANSACTION_PER_CLOC_CASE_LOCKED_BY_ALL_TREE_CACHE",
            "normal_transaction_policy": "ALL_LISTED_FAULTLESS_TRANSACTIONS",
            "candidate_policy": "COVERED_BY_AT_LEAST_ONE_FAILING_TRANSACTION",
            "coverage_level": "FUNCTION_VIA_OPCODE_PROGRAM_COUNTER_AND_SOURCE_MAP",
            "ground_truth_policy": "DATASET_LOCATION_CONTAINED_BY_FUNCTION_SOURCE_RANGE",
            "metric_policy": "RECALL_AT_1_3_5_MFR_MAR_WITH_STABLE_ORDINAL_RANKS",
        },
        "summary": {
            "case_count": len(rows),
            "selected_fault_transaction_count": sum(len(row["selected_fault_transactions"]) for row in rows),
            "normal_transaction_reference_count": len(all_normal),
            "unique_normal_transaction_count": len(set(all_normal)),
            "ethereum_case_count": sum(row["chain_key"] == "eth" for row in rows),
            "bsc_case_count": sum(row["chain_key"] == "bsc" for row in rows),
            "case_with_local_fault_call_tree_count": sum(
                row["local_artifacts"]["selected_fault_call_tree_count"] > 0 for row in rows
            ),
            "case_with_usable_normal_call_features_count": sum(
                row["local_artifacts"]["usable_normal_call_feature_count"] > 0 for row in rows
            ),
            "duplicate_case_names": duplicate_names,
        },
        "cases": rows,
    }


def write_manifest(value: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

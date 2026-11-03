from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config_loader import map_platform_to_chain


def scan_dataset_cases(dataset_root: str | Path = "dataset") -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    cases: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        cases.append(_normalize_case(path, raw))
    return cases


def find_case_by_tx(tx_hash: str, dataset_root: str | Path = "dataset") -> Dict[str, Any] | None:
    target = tx_hash.lower()
    for case in scan_dataset_cases(dataset_root):
        if target in case["transaction_hashes"]:
            return case
    return None


def _normalize_case(path: Path, raw: Dict[str, Any]) -> Dict[str, Any]:
    platform = raw.get("platform", "")
    transaction_hashes = [tx.lower() for tx in _flatten_hashes(raw.get("fault", {}).get("transaction_hash", []))]
    locations = raw.get("fault", {}).get("location", [])
    return {
        "path": str(path),
        "category": path.parent.name,
        "case_name": raw.get("name", path.stem),
        "cause": raw.get("cause", ""),
        "platform": platform,
        "mapped_chain": map_platform_to_chain(platform) if platform else "",
        "transaction_hashes": transaction_hashes,
        "location_count": len(locations) if isinstance(locations, list) else 0,
        "has_location": bool(locations),
    }


def _flatten_hashes(values: Iterable[Any]) -> List[str]:
    flattened: List[str] = []
    for value in values:
        if isinstance(value, str):
            flattened.append(value)
        elif isinstance(value, list):
            flattened.extend(_flatten_hashes(value))
    return flattened

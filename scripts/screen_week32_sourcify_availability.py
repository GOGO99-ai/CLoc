"""Check Sourcify source availability for the frozen missing-source addresses."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_v2.source_cache import atomic_json


LOCKED = ROOT / "data" / "prototypes" / "week32_remaining_dataset_screen_20260911" / "source_backfill" / "locked_manifest.json"
OUTPUT = ROOT / "data" / "prototypes" / "week32_remaining_dataset_screen_20260911" / "sourcify_availability.json"
BASE_URL = "https://sourcify.dev/server/v2/contract/1/"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_source_content(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for item in value.values():
        if isinstance(item, Mapping) and isinstance(item.get("content"), str) and item["content"].strip():
            return True
        if isinstance(item, str) and item.strip():
            return True
    return False


def run() -> dict[str, Any]:
    manifest = _load(LOCKED)
    rows = []
    request_count = 0
    for entry in manifest["addresses"]:
        alias = entry["address_alias"]
        address = entry["contract_address"]
        # Verification metadata is returned by default. Only ``sources`` is an
        # optional field; asking for default fields again makes API v2 return 400.
        url = BASE_URL + quote(address, safe="") + "?fields=sources"
        request_count += 1
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                document = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            status = "NOT_AVAILABLE" if error.code == 404 else "HTTP_ERROR"
            rows.append({"address_alias": alias, "status": status, "http_status": error.code})
            continue
        except (OSError, UnicodeError, json.JSONDecodeError):
            rows.append({"address_alias": alias, "status": "NETWORK_OR_PARSE_ERROR", "http_status": None})
            continue
        sources = document.get("sources") if isinstance(document, Mapping) else None
        available = _has_source_content(sources)
        rows.append({
            "address_alias": alias,
            "status": "SOURCE_AVAILABLE" if available else "SOURCE_CONTENT_MISSING",
            "http_status": 200,
            "source_file_count": len(sources) if isinstance(sources, Mapping) else 0,
            "match": document.get("match") if isinstance(document, Mapping) else None,
        })
    counts = Counter(row["status"] for row in rows)
    report = {
        "schema_version": "week32_sourcify_availability_v1",
        "mode": "SOURCE_AVAILABILITY_ONLY",
        "provider": "SOURCIFY_OFFICIAL",
        "checked_address_count": len(rows),
        "request_count": request_count,
        "status_counts": dict(sorted(counts.items())),
        "available_address_aliases": [row["address_alias"] for row in rows if row["status"] == "SOURCE_AVAILABLE"],
        "external_call_counts": {"source": request_count, "rpc": 0, "llm": 0},
        "rows": rows,
    }
    atomic_json(OUTPUT, report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "checked_address_count",
        "status_counts",
        "available_address_aliases",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

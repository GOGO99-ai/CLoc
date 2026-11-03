"""Import the frozen Week 32 Sourcify sources into the verified-source cache."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_v2.source_cache import atomic_json, norm_address, sha
from src.cloc_v2.source_parsing import safe_relative_source_path


LOCKED = ROOT / "data" / "prototypes" / "week32_remaining_dataset_screen_20260911" / "source_backfill" / "locked_manifest.json"
OUTPUT = ROOT / "data" / "prototypes" / "week32_remaining_dataset_screen_20260911" / "sourcify_import.json"
CACHE_ROOT = ROOT / "data" / "source_cache" / "eth"
BASE_URL = "https://sourcify.dev/server/v2/contract/1/"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_sources(cache_dir: Path, sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path, item in sorted(sources.items()):
        relative = safe_relative_source_path(str(raw_path))
        content = item.get("content") if isinstance(item, Mapping) else item
        if relative is None or relative in seen or not isinstance(content, str) or not content.strip():
            continue
        seen.add(relative)
        target = cache_dir / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        files.append({
            "path": (Path("sources") / relative).as_posix(),
            "sha256": sha(target),
            "bytes": target.stat().st_size,
        })
    return files


def run() -> dict[str, Any]:
    manifest = _load(LOCKED)
    rows: list[dict[str, Any]] = []
    for entry in manifest["addresses"]:
        alias = entry["address_alias"]
        address = norm_address(entry["contract_address"])
        url = BASE_URL + quote(address, safe="") + "?fields=sources"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                document = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, OSError, UnicodeError, json.JSONDecodeError) as error:
            rows.append({"address_alias": alias, "status": "DOWNLOAD_FAILED", "detail": type(error).__name__})
            continue
        sources = document.get("sources") if isinstance(document, Mapping) else None
        if not isinstance(sources, Mapping):
            rows.append({"address_alias": alias, "status": "SOURCE_CONTENT_MISSING"})
            continue
        cache_dir = CACHE_ROOT / address
        files = _write_sources(cache_dir, sources)
        solidity_count = sum(Path(item["path"]).suffix.lower() == ".sol" for item in files)
        if solidity_count == 0:
            rows.append({"address_alias": alias, "status": "SOLIDITY_SOURCE_MISSING"})
            continue
        metadata = {
            "provider": "SOURCIFY_OFFICIAL_V2",
            "match": document.get("match"),
            "creation_match": document.get("creationMatch"),
            "runtime_match": document.get("runtimeMatch"),
        }
        atomic_json(cache_dir / "metadata.json", metadata)
        cached_at = datetime.now(timezone.utc).isoformat()
        atomic_json(cache_dir / "manifest.json", {
            "chain": "eth",
            "requested_address": address,
            "resolved_address": address,
            "status": "verified",
            "verified": True,
            "cache_hit": False,
            "source_format": "sourcify_v2",
            "contract_name": "Unknown",
            "files": files,
            "abi_path": None,
            "proxy": {"is_proxy": False, "resolution": "DIRECT_SOURCIFY_MATCH"},
            "failure": None,
            "attempt_count": 1,
            "origins": [{"audit_id": "WEEK32_REMAINING_DATASET_SCREEN", "address_alias": alias}],
            "artifact_hashes": {"metadata.json": sha(cache_dir / "metadata.json")},
            "schema_version": "source_manifest_v1",
            "cache_dir": str(cache_dir),
            "cached_at": cached_at,
        })
        rows.append({
            "address_alias": alias,
            "status": "IMPORTED",
            "source_file_count": len(files),
            "solidity_file_count": solidity_count,
        })
    counts = Counter(row["status"] for row in rows)
    report = {
        "schema_version": "week32_sourcify_import_v1",
        "checked_address_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "external_call_counts": {"source": len(rows), "rpc": 0, "llm": 0},
        "rows": rows,
    }
    atomic_json(OUTPUT, report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "checked_address_count",
        "status_counts",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

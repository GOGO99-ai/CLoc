"""Shared, dependency-free parsing for Etherscan SourceCode payloads."""
from __future__ import annotations

import json
import re
from pathlib import Path


def _safe_contract_filename(contract_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", contract_name or "Contract").strip("._")
    return cleaned or "Contract"


def safe_relative_source_path(raw_path: str) -> str | None:
    normalized = str(raw_path).replace("\\", "/").strip().lstrip("/")
    if not normalized or ".." in normalized.split("/"):
        return None
    return normalized


def write_explorer_source_files(source: str, output_dir: Path, contract_name: str) -> tuple[list[str], str]:
    """Write single-file, standard-json and Etherscan double-brace source safely."""
    src = source.strip()
    if src.startswith("{{") and src.endswith("}}"):
        src = src[1:-1]
    if src.startswith("{"):
        parsed = json.loads(src)
        sources = parsed.get("sources", parsed)
        if not isinstance(sources, dict):
            raise ValueError("standard-json sources is not an object")
        files: list[str] = []
        seen: set[str] = set()
        for raw_path, obj in sources.items():
            safe_path = safe_relative_source_path(str(raw_path))
            content = obj.get("content", obj) if isinstance(obj, dict) else obj
            if safe_path is None or not isinstance(content, str):
                continue
            if safe_path in seen:
                raise ValueError(f"duplicate normalized source path: {safe_path}")
            seen.add(safe_path)
            target = output_dir / safe_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            files.append(safe_path)
        if not files:
            raise ValueError("no safe source files in standard-json")
        return files, "standard_json"
    filename = f"{_safe_contract_filename(contract_name)}.sol"
    target = output_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return [filename], "single_file"

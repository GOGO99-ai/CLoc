from __future__ import annotations

from pathlib import Path
from typing import Dict


def build_run_layout(tx_hash: str, chain: str, root: str | Path = "data/runs") -> Dict[str, str]:
    tx = tx_hash.lower()
    base = Path(root) / chain / tx
    return {
        "run_root": str(base),
        "trace_file": str(base / "trace.txt"),
        "call_tree_file": str(base / "call_tree.json"),
        "address_list_file": str(base / "address_list.json"),
        "source_root": str(base / "source"),
        "manifest_file": str(base / "manifest.json"),
    }

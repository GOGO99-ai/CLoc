from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloc_v2.asset_continuity import build_asset_continuity, load_observed_asset_ledger, write_asset_continuity
from src.cloc_v2.observed_asset_ledger import load_runtime_annotations
from src.cloc_v2.runtime_semantics import load_transaction_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Build candidate asset continuity and explicit temporary-financing episodes.")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(args.graph).resolve(), Path(args.annotations).resolve(), Path(args.ledger).resolve(), Path(args.output).resolve()]
    if len(set(paths)) != 4:
        print("error: graph, annotations, ledger, and output paths must all differ", file=sys.stderr)
        return 2
    try:
        result = build_asset_continuity(load_transaction_graph(paths[0]), load_runtime_annotations(paths[1]), load_observed_asset_ledger(paths[2]))
        write_asset_continuity(result, paths[3])
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "output": str(paths[3]), "schema_version": result.SCHEMA_VERSION, **result.to_dict()["metadata"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloc_v2.observed_asset_ledger import (  # noqa: E402
    build_observed_asset_ledger,
    load_runtime_annotations,
    load_transaction_graph,
    write_observed_asset_ledger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an observed_asset_ledger_v1 from local graph and annotation artifacts.")
    parser.add_argument("--graph", required=True, help="Path to transaction_graph_v1.json")
    parser.add_argument("--annotations", required=True, help="Path to runtime_annotations_v1.json")
    parser.add_argument("--output", required=True, help="Path for the new observed_asset_ledger_v1.json artifact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_path = Path(args.graph).resolve()
    annotations_path = Path(args.annotations).resolve()
    output_path = Path(args.output).resolve()
    if len({graph_path, annotations_path, output_path}) != 3:
        print("error: --graph, --annotations, and --output must refer to different paths", file=sys.stderr)
        return 2
    try:
        transaction_graph = load_transaction_graph(graph_path)
        runtime_annotations = load_runtime_annotations(annotations_path)
        ledger = build_observed_asset_ledger(transaction_graph, runtime_annotations)
        write_observed_asset_ledger(ledger, output_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "output": str(output_path), "schema_version": ledger.SCHEMA_VERSION, **ledger.to_dict()["metadata"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

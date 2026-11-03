from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloc_v2.transaction_graph import (  # noqa: E402
    build_transaction_graph,
    load_call_tree,
    write_transaction_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic transaction_graph_v1 from a local call tree JSON.")
    parser.add_argument("--input", required=True, help="Path to a direct call-tree JSON or {call_tree: ...} wrapper")
    parser.add_argument("--output", required=True, help="Path for the new transaction_graph_v1.json artifact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if input_path == output_path:
        print("error: --input and --output must refer to different paths", file=sys.stderr)
        return 2

    try:
        call_tree = load_call_tree(input_path)
        graph = build_transaction_graph(call_tree)
        write_transaction_graph(graph, output_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "schema_version": graph.SCHEMA_VERSION,
                "node_count": len(graph.nodes),
                "call_edge_count": len(graph.call_edges),
                "order_edge_count": len(graph.order_edges),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

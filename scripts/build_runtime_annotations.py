from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloc_v2.runtime_semantics import (  # noqa: E402
    build_runtime_annotations,
    load_transaction_graph,
    write_runtime_annotations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build neutral runtime_annotations_v1 from a local transaction_graph_v1.")
    parser.add_argument("--graph", required=True, help="Path to transaction_graph_v1.json")
    parser.add_argument("--output", required=True, help="Path for the new runtime_annotations_v1.json artifact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_path = Path(args.graph).resolve()
    output_path = Path(args.output).resolve()
    if graph_path == output_path:
        print("error: --graph and --output must refer to different paths", file=sys.stderr)
        return 2
    try:
        transaction_graph = load_transaction_graph(graph_path)
        annotations = build_runtime_annotations(transaction_graph)
        write_runtime_annotations(annotations, output_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    metadata = annotations.to_dict()["metadata"]
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "schema_version": annotations.SCHEMA_VERSION,
                **metadata,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build an independent phase-2C-2 result-candidate artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloc_v2.result_detection import (
    detect_result_candidates,
    load_asset_continuity,
    load_observed_asset_ledger,
    load_runtime_annotations,
    load_transaction_graph,
    write_result_detection,
)
from src.cloc_v2.result_models import ResultDetectionConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect evidence-bounded result-node candidates from existing phase artifacts.")
    parser.add_argument("--graph", required=True, help="transaction_graph_v1.json")
    parser.add_argument("--annotations", required=True, help="runtime_annotations_v1.json")
    parser.add_argument("--ledger", required=True, help="observed_asset_ledger_v1.json")
    parser.add_argument("--continuity", required=True, help="asset_continuity_v1.json")
    parser.add_argument("--output", required=True, help="new result_candidates_v1.json path")
    parser.add_argument("--outflow-lambda", default="0.65", help="asset/context grouped-mean balance in [0,1]")
    args = parser.parse_args(argv)
    input_paths = [Path(args.graph).resolve(), Path(args.annotations).resolve(), Path(args.ledger).resolve(), Path(args.continuity).resolve()]
    output_path = Path(args.output).resolve()
    if output_path in input_paths or len(set(input_paths)) != len(input_paths):
        parser.error("graph, annotations, ledger, continuity, and output paths must all be distinct")
    try:
        graph = load_transaction_graph(input_paths[0])
        annotations = load_runtime_annotations(input_paths[1])
        ledger = load_observed_asset_ledger(input_paths[2])
        continuity = load_asset_continuity(input_paths[3])
        result = detect_result_candidates(graph, annotations, ledger, continuity, ResultDetectionConfig(outflow_lambda=args.outflow_lambda))
        write_result_detection(result, output_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "output": str(output_path), **result.to_dict()["metadata"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

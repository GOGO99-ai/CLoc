"""Build stage-3B candidate causal chains from graph and backbone artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_v2.causal_chain_assembly import (
    build_candidate_causal_chains,
    load_json_document,
    write_candidate_causal_chains,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble deterministic candidate causal chains with evidence branches."
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--backbones", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    graph_path = Path(args.graph).resolve()
    backbones_path = Path(args.backbones).resolve()
    output_path = Path(args.output).resolve()
    if output_path in {graph_path, backbones_path}:
        parser.error("output path must differ from both input paths")
    try:
        graph = load_json_document(graph_path, "candidate_causal_graph_v2")
        backbones = load_json_document(backbones_path, "causal_backbones_v1")
        result = build_candidate_causal_chains(graph, backbones)
        write_candidate_causal_chains(result, output_path)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "output": str(output_path), **result.assembly_audit}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

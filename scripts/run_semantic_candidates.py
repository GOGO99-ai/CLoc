from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloc_v2.semantic_chains import write_semantic_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate semantic nodes and candidate chains from a call tree JSON.")
    parser.add_argument("--input", required=True, help="Path to call_tree JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to write semantic_nodes.json and candidate_chains.json")
    args = parser.parse_args()

    result = write_semantic_artifacts(args.input, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

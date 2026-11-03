"""Build the locked transaction/source acquisition manifest for CLoc baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_baselines.manifest import build_cloc_baseline_manifest, write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--call-tree-root", type=Path, default=ROOT / "data" / "cache" / "all_tree")
    parser.add_argument(
        "--normal-samples-root",
        type=Path,
        default=ROOT / "data" / "prototypes" / "formal_normal_baseline_full" / "samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparison" / "cloc_baselines" / "manifest.json",
    )
    args = parser.parse_args(argv)
    value = build_cloc_baseline_manifest(
        dataset_root=args.dataset_root,
        call_tree_root=args.call_tree_root,
        normal_samples_root=args.normal_samples_root,
    )
    write_manifest(value, args.output)
    print(json.dumps(value["summary"], ensure_ascii=False, indent=2))
    print(f"manifest={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

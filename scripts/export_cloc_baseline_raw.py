"""Export the locked 40 CLoc JSON cases using stable manifest case IDs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def export_raw(*, manifest_path: Path, dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases") if isinstance(manifest, Mapping) else None
    if not isinstance(cases, list):
        raise ValueError("Invalid CLoc baseline manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for item in cases:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid case row in CLoc baseline manifest")
        source = (dataset_root / item["dataset_file"]).resolve()
        source.relative_to(dataset_root.resolve())
        target = output_dir / f"{item['case_id']}.json"
        shutil.copyfile(source, target)
        exported.append(target.name)
    return {"case_count": len(exported), "files": exported}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "manifest.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export_raw(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

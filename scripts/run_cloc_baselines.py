"""Run function-coverage baselines and common metrics on CLoc artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_baselines.metrics import evaluate_rankings
from src.cloc_baselines.ranking import SUPPORTED_METHODS, rank_case, serialize_rankings
from src.cloc_baselines.schema import CoverageCase
from src.cloc_baselines.cnnfl import rank_cnnfl_case


def _load_expected_case_ids(manifest_path: Path) -> tuple[str, ...]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Invalid baseline manifest")
    return tuple(item["case_id"] for item in rows if isinstance(item, dict) and isinstance(item.get("case_id"), str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "manifest.json")
    parser.add_argument("--coverage-dir", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "coverage")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "results")
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS + ("cnnfl",), default=list(SUPPORTED_METHODS[:3]) + ["cnnfl"])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--cnnfl-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--cnnfl-device")
    parser.add_argument(
        "--cnnfl-malo-label-leakage",
        action="store_true",
        help="Reproduce MALO's ground-truth leakage; never use this for the primary clean result.",
    )
    args = parser.parse_args(argv)
    expected = _load_expected_case_ids(args.manifest)
    available = {path.stem: path for path in args.coverage_dir.glob("*.json")}
    missing = [case_id for case_id in expected if case_id not in available]
    if missing and not args.allow_partial:
        print(f"coverage_status=INCOMPLETE available={len(available)} expected={len(expected)}")
        print("missing_case_ids=" + ",".join(missing))
        return 2
    selected = [case_id for case_id in expected if case_id in available]
    if not selected:
        print("coverage_status=EMPTY")
        return 2
    cases = {case_id: CoverageCase.load(available[case_id]) for case_id in selected}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": "cloc_baseline_results_v1",
        "expected_case_count": len(expected),
        "evaluated_case_count": len(cases),
        "partial": len(cases) != len(expected),
        "methods": {},
    }
    for method in args.methods:
        if method == "cnnfl":
            if len(args.cnnfl_seeds) != 1:
                raise ValueError("The primary result artifact requires exactly one CNNFL seed per run")
            rankings = {
                case_id: rank_cnnfl_case(
                    case,
                    seed=args.cnnfl_seeds[0],
                    device=args.cnnfl_device,
                    malo_label_leakage_compatibility=args.cnnfl_malo_label_leakage,
                )
                for case_id, case in cases.items()
            }
        else:
            rankings = {case_id: rank_case(case, method) for case_id, case in cases.items()}
        metric = evaluate_rankings(rankings)
        payload = {
            "schema_version": "cloc_baseline_method_result_v1",
            "method": method,
            "seed": args.cnnfl_seeds[0] if method == "cnnfl" else None,
            "malo_label_leakage_compatibility": bool(args.cnnfl_malo_label_leakage) if method == "cnnfl" else None,
            "metrics": metric,
            "rankings": {case_id: serialize_rankings(rows) for case_id, rows in rankings.items()},
        }
        target = args.output_dir / f"{method}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["methods"][method] = metric
        print(
            f"{method}: Recall@1={metric['recall']['recall_at_1']:.4f} "
            f"Recall@3={metric['recall']['recall_at_3']:.4f} "
            f"Recall@5={metric['recall']['recall_at_5']:.4f} "
            f"MFR={metric['mfr']:.4f} MAR={metric['mar']:.4f}"
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

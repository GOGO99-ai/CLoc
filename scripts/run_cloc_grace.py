"""Run leakage-free leave-one-case-out Grace on complete CLoc graph artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_baselines.grace import GraceGraphCase, rank_grace_leave_one_out
from src.cloc_baselines.metrics import evaluate_rankings
from src.cloc_baselines.ranking import serialize_rankings
from src.cloc_baselines.schema import CoverageCase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "manifest.json")
    parser.add_argument("--coverage-dir", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "coverage")
    parser.add_argument("--grace-dir", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "grace")
    parser.add_argument("--output", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "results" / "grace.json")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Invalid CLoc baseline manifest")
    expected = tuple(item["case_id"] for item in rows)
    available = tuple(
        case_id
        for case_id in expected
        if (args.coverage_dir / f"{case_id}.json").exists()
        and (args.grace_dir / f"{case_id}.json").exists()
    )
    missing = tuple(case_id for case_id in expected if case_id not in available)
    if missing and not args.allow_partial:
        print(f"grace_status=INCOMPLETE available={len(available)} expected={len(expected)}")
        print("missing_case_ids=" + ",".join(missing))
        return 2
    if len(available) < 2:
        print("grace_status=INSUFFICIENT_CASES")
        return 2
    coverage = {
        case_id: CoverageCase.load(args.coverage_dir / f"{case_id}.json")
        for case_id in available
    }
    graphs = [GraceGraphCase.load(args.grace_dir / f"{case_id}.json") for case_id in available]
    rankings = rank_grace_leave_one_out(
        graphs,
        coverage,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    metrics = evaluate_rankings(rankings)
    payload = {
        "schema_version": "cloc_baseline_method_result_v1",
        "method": "grace_cloc_adaptation",
        "protocol": "LEAVE_ONE_CLOC_CASE_OUT_FIXED_EPOCH",
        "held_out_label_used_for_model_selection": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "expected_case_count": len(expected),
        "evaluated_case_count": len(available),
        "partial": bool(missing),
        "metrics": metrics,
        "rankings": {case_id: serialize_rankings(value) for case_id, value in rankings.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"grace: Recall@1={metrics['recall']['recall_at_1']:.4f} "
        f"Recall@3={metrics['recall']['recall_at_3']:.4f} "
        f"Recall@5={metrics['recall']['recall_at_5']:.4f} "
        f"MFR={metrics['mfr']:.4f} MAR={metrics['mar']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

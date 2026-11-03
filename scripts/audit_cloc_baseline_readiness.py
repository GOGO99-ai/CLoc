"""Audit whether all locked CLoc cases are ready for fair baseline execution."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_baselines.schema import CoverageCase
from src.cloc_baselines.grace import GraceGraphCase


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_readiness(*, manifest_path: Path, coverage_dir: Path, grace_dir: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    cases = manifest.get("cases") if isinstance(manifest, Mapping) else None
    if not isinstance(cases, list):
        raise ValueError("Invalid CLoc baseline manifest")
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for expected in cases:
        if not isinstance(expected, Mapping) or not isinstance(expected.get("case_id"), str):
            raise ValueError("Invalid case row in CLoc baseline manifest")
        case_id = expected["case_id"]
        coverage_path = coverage_dir / f"{case_id}.json"
        grace_path = grace_dir / f"{case_id}.json"
        reasons: list[str] = []
        coverage: CoverageCase | None = None
        if not coverage_path.exists():
            reasons.append("FUNCTION_COVERAGE_MISSING")
        else:
            try:
                coverage = CoverageCase.load(coverage_path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                reasons.append(f"FUNCTION_COVERAGE_INVALID:{type(exc).__name__}")
        if coverage is not None:
            if coverage.case_id != case_id:
                reasons.append("CASE_ID_MISMATCH")
            if coverage.case_name != expected.get("case_name"):
                reasons.append("CASE_NAME_MISMATCH")
            if coverage.chain_id != expected.get("chain_id") or coverage.chain_key != expected.get("chain_key"):
                reasons.append("CHAIN_MISMATCH")
            actual_fail = {item.transaction_hash for item in coverage.tests if item.outcome == "fail"}
            actual_pass = {item.transaction_hash for item in coverage.tests if item.outcome == "pass"}
            expected_fail = set(expected.get("selected_fault_transactions", []))
            expected_pass = set(expected.get("normal_transactions", []))
            if actual_fail != expected_fail:
                reasons.append("FAULT_TRANSACTION_SET_MISMATCH")
            if actual_pass != expected_pass:
                reasons.append("NORMAL_TRANSACTION_SET_MISMATCH")
            if not coverage.call_edges:
                reasons.append("PRFL_CALL_GRAPH_MISSING")
        grace: GraceGraphCase | None = None
        if not grace_path.exists():
            reasons.append("GRACE_GRAPH_MISSING")
        else:
            try:
                grace = GraceGraphCase.load(grace_path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                reasons.append(f"GRACE_GRAPH_INVALID:{type(exc).__name__}")
        if grace is not None and coverage is not None:
            if grace.case_id != case_id or grace.function_ids != coverage.candidate_function_ids:
                reasons.append("GRACE_COVERAGE_MISMATCH")
        reason_counts.update(reasons)
        rows.append(
            {
                "case_id": case_id,
                "case_name": expected.get("case_name"),
                "coverage_ready": coverage is not None and not any(
                    reason.startswith("FUNCTION_COVERAGE")
                    or reason.endswith("MISMATCH")
                    or reason == "PRFL_CALL_GRAPH_MISSING"
                    for reason in reasons
                ),
                "grace_ready": grace is not None and not any(
                    reason.startswith("GRACE_") for reason in reasons
                ),
                "reason_codes": reasons,
            }
        )
    coverage_ready = sum(bool(item["coverage_ready"]) for item in rows)
    grace_ready = sum(bool(item["grace_ready"]) for item in rows)
    return {
        "schema_version": "cloc_baseline_readiness_audit_v1",
        "status": "READY" if coverage_ready == len(rows) and grace_ready == len(rows) else "INCOMPLETE",
        "case_count": len(rows),
        "coverage_ready_case_count": coverage_ready,
        "grace_ready_case_count": grace_ready,
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "manifest.json")
    parser.add_argument("--coverage-dir", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "coverage")
    parser.add_argument("--grace-dir", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "grace")
    parser.add_argument("--output", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "readiness.json")
    args = parser.parse_args(argv)
    result = audit_readiness(
        manifest_path=args.manifest,
        coverage_dir=args.coverage_dir,
        grace_dir=args.grace_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "case_count", "coverage_ready_case_count", "grace_ready_case_count", "reason_code_counts")}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

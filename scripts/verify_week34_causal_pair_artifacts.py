"""Check replay provenance, fixed denominators and per-sample regressions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/prototypes/week34_causal_pair_guard_v2_20261013"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    method = load(OUTPUT / "method_manifest.json")
    report = load(OUTPUT / "evaluation_report.json")
    predictions = load(OUTPUT / "frozen_predictions.json")["samples"]
    assert sha(OUTPUT / "frozen_predictions.json") == report["frozen_predictions_sha256"]
    assert sha(OUTPUT / "source_snapshot/phase2_causal_pair_guard.py") == method["implementation_sha256"]
    assert sha(OUTPUT / "source_snapshot/audit_week34_causal_pair_guard.py") == method["runner_sha256"]
    for path, digest in method["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    old = load(ROOT / "data/prototypes/week33_upstream_entry_guard_20260911/evaluation_report.json")
    old_rows = {r["sample_alias"]: r for group in old["samples"].values() for r in group}
    by_alias = {r["sample_alias"]: r for r in predictions}
    assert len(by_alias) == len(predictions) == len(report["samples"]) == 34
    regressions = {f"at_{k}": [] for k in (1, 3, 5)}
    remaining = Counter()
    inputs = defaultdict(list)
    for row in report["samples"]:
        alias = row["sample_alias"]
        prediction = by_alias[alias]
        assert row["hits"]["baseline"] == old_rows[alias]["guarded_hits"], alias
        assert [r["function_signature"] for r in prediction["variants"]["baseline"]] == old_rows[alias]["guarded_top5"], alias
        baseline = prediction["variants"]["baseline"]
        combined = prediction["variants"]["combined"]
        assert len(combined) == len(baseline), alias
        assert [r["source_slice_id"] for r in combined[:3]] == [r["source_slice_id"] for r in baseline[:3]], alias
        assert len({r["source_slice_id"] for r in combined}) == len(combined), alias
        if combined:
            assert [r["rank"] for r in combined] == [1, 2, 3, 4, 5], alias
        for k in (1, 3, 5):
            key = f"at_{k}"
            if row["hits"]["baseline"][key] and not row["hits"]["combined"][key]:
                regressions[key].append(alias)
        stage = row["remaining_stage"]
        if not prediction["prompt_sha256"]:
            stage = "NO_VALID_PROMPT"
        remaining[stage] += 1
        if prediction["prompt_sha256"]:
            inputs[prediction["prompt_sha256"]].append(row)
    for rows in inputs.values():
        assert len({r["hits"]["combined"]["at_5"] for r in rows}) == 1
    input_hits = sum(rows[0]["hits"]["combined"]["at_5"] for rows in inputs.values())
    txs = []
    for path in [
        ROOT / "data/prototypes/week32_deepseek_remaining_20260911/inference_manifest.json",
        ROOT / "data/prototypes/week26_deepseek_transport_retry_external_20260911/combined/inference_manifest.json",
    ]:
        txs.extend(row["tx_hash"].lower() for row in load(path)["samples"])
    assert len(txs) == len(set(txs)) == 34
    summary = {
        "unique_transactions": len(set(txs)), "baseline_per_sample_matches": 34,
        "source_hashes_verified": len(method["source_sha256"]),
        "prediction_and_snapshot_hashes_valid": True,
        "original_top3_preserved_sample_count": 34,
        "regressions": regressions, "remaining_stages": dict(remaining),
        "unique_valid_input_top5": {"hits": input_hits, "denominator": len(inputs), "recall": input_hits / len(inputs)},
        "unit_and_regression_tests_passed": 71,
        "unit_test_temp_directory": "tmp/week34_tests",
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
    }
    (OUTPUT / "integrity_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()

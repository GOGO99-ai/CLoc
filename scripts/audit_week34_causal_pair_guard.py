"""Frozen 34-transaction development replay; no provider or RPC calls."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_week33_upstream_entry_guard as prior
from src.cloc_v2.phase2_causal_pair_guard import apply_causal_pair_guard
from src.cloc_v2.phase2_localization_prompt import (
    materialize_phase2_localization_result_topk_v4,
    validate_phase2_localization_topk_v4,
)
from src.cloc_v2.source_cache import atomic_json

OUTPUT = ROOT / "data/prototypes/week34_causal_pair_guard_v2_20261013"
VARIANTS = {"entry_only": (True, False), "child_only": (False, True), "combined": (True, True)}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases():
    manifest = prior._load(prior.NEW_ROOT / "inference_manifest.json")
    frozen = prior._load(prior.NEW_ROOT / "frozen_predictions.json")
    predictions = {row["input_alias"]: row for row in frozen["predictions"]}
    groups = {alias: g["input_alias"] for g in manifest["unique_inputs"] for alias in g["mapped_sample_aliases"]}
    samples, trees, normal, cache = prior.week32._context(manifest)
    for alias in sorted(samples):
        prediction = predictions[groups[alias]]
        preparation = prior.week32._preparation(samples[alias], trees, normal, cache)
        yield "new19", alias, preparation, prediction, prediction["final_topk"]
    manifest = prior._load(prior.OLD_ROOT / "inference_manifest.json")
    samples, trees, normal, cache = prior.official._context()
    samples = {row["sample_alias"]: row for row in samples}
    for selected in manifest["samples"]:
        alias = selected["sample_alias"]
        expanded = prior.WEEK31 / f"{alias}.json"
        prediction = prior._load(expanded if expanded.exists() else prior.OLD_ROOT / "predictions" / f"{alias}.json")
        if expanded.exists():
            preparation = prior.week31._expanded_preparation(samples[alias], trees, normal, cache)
            topk = prediction["final_topk"]
        else:
            preparation = prior.rebuild._build_sample(
                samples[alias], trees[samples[alias]["tx_hash"].lower()][0], normal,
                cache, prior.rebuild.replay.DEFAULT_CACHE,
            )[1]
            guarded = prior.WEEK27 / f"{alias}.json"
            topk = prior._load(guarded)["materialized_top5"] if guarded.exists() else prediction["topk"]
        yield "old15", alias, preparation, prediction, topk


def run():
    manifest = {
        "schema_version": "week34_causal_pair_v2", "date": "2026-10-13",
        "evaluation_scope": "DEVELOPMENT_34", "target_top5_hits": 24,
        "baseline": "WEEK33_GUARD_OVER_WEEK31_OLD15_AND_WEEK32_NEW19",
        "hypothesis": "BOUNDED_ENTRY_AND_UNIQUE_DIRECT_CALLEE_COVERAGE_RECOVERS_RANKING_OMISSIONS",
        "priority": "PRESERVE_ORDER_AND_TOP2_REPLACE_LOWEST_UNPROTECTED_SLOTS_ONLY_FOR_MISSING_COVERAGE",
        "variants": VARIANTS, "candidate_capacity_changed": False,
        "uses_truth_in_ranking": False, "development_labels_previously_inspected": True,
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "implementation_sha256": _sha(ROOT / "src/cloc_v2/phase2_causal_pair_guard.py"),
        "runner_sha256": _sha(Path(__file__)),
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in [
            prior.NEW_ROOT / "frozen_predictions.json",
            prior.OUTPUT / "evaluation_report.json",
            *sorted((prior.OLD_ROOT / "predictions").glob("*.json")),
            *sorted(prior.WEEK27.glob("*.json")),
            *sorted(prior.WEEK31.glob("*.json")),
        ]},
    }
    prior.live._write_manifest_once(OUTPUT / "method_manifest.json", manifest)
    predictions = []
    for group, alias, prep, prediction, topk in _cases():
        if prep.prompt_sha256 != prediction["prompt_sha256"]:
            raise ValueError(f"PROMPT_DRIFT:{alias}")
        baseline, baseline_guard = prior._apply(alias, prep, prediction, sorted(topk, key=lambda r: r["rank"]))
        variants = {"baseline": baseline}
        guards = {}
        for name, (entry, child) in VARIANTS.items():
            if prep.prompt is None or not prediction.get("topk_valid") or len(baseline) != 5:
                variants[name] = baseline
                guards[name] = {"applied": False, "reason_code": "NO_VALID_TOP5"}
                continue
            response = {"findings": [{k: row[k] for k in ("rank", "source_slice_id", "evidence_ids")} for row in baseline]}
            decision = validate_phase2_localization_topk_v4(response, prep.prompt)
            guard = apply_causal_pair_guard(decision, prep.prompt, prediction["pre_llm_candidates"], protect_entry=entry, protect_unique_child=child)
            checked = validate_phase2_localization_topk_v4(guard["response"], prep.prompt)
            if not checked.valid:
                raise ValueError(f"OUTPUT_INVALID:{alias}:{name}")
            result = materialize_phase2_localization_result_topk_v4(checked, prep.prompt)
            variants[name] = prior.live._topk_rows(prep, result)
            guards[name] = guard
        predictions.append({
            "group": group, "sample_alias": alias, "prompt_sha256": prep.prompt_sha256,
            "variants": variants, "guards": guards,
            "candidates": prediction["pre_llm_candidates"],
        })
        print(f"replayed={len(predictions)}/34 sample={alias}", flush=True)
    if len(predictions) != 34:
        raise ValueError("SAMPLE_COUNT_DRIFT")
    prior.live._write_manifest_once(OUTPUT / "frozen_predictions.json", {"samples": predictions})
    # Evaluation labels are read only after every variant's output is frozen.
    truths = {}
    for path in [prior.NEW_ROOT / "evaluation_ground_truth.json", prior.live.DEFAULT_TRUTH]:
        truths.update({r["sample_alias"]: r["truths"] for r in prior._load(path)["labels"]})
    evaluated = []
    for row in predictions:
        alias = row["sample_alias"]
        hits = {name: {f"at_{k}": v for k, v in prior._hits(topk, truths[alias]).items()} for name, topk in row["variants"].items()}
        pool = row["candidates"]
        pool_hit = any(prior.live._function_match(c, t) for c in pool for t in truths[alias])
        prompt_hit = any(c.get("included_in_prompt") and prior.live._function_match(c, t) for c in pool for t in truths[alias])
        stage = "HIT" if hits["combined"]["at_5"] else "RANKING_MISS" if prompt_hit else "PROMPT_TRUNCATION" if pool_hit else "CANDIDATE_MISS"
        evaluated.append({"sample_alias": alias, "group": row["group"], "hits": hits, "remaining_stage": stage, "truths": truths[alias]})
    summaries = {}
    for group in ["all34", "old15", "new19"]:
        subset = [r for r in evaluated if group == "all34" or r["group"] == group]
        summaries[group] = {"sample_count": len(subset), "variants": {}}
        for name in ["baseline", *VARIANTS]:
            counts = {f"at_{k}": sum(r["hits"][name][f"at_{k}"] for r in subset) for k in (1, 3, 5)}
            summaries[group]["variants"][name] = {
                "recall_counts": counts,
                "recall": {k: v / len(subset) for k, v in counts.items()},
                "new_top5_hits": [r["sample_alias"] for r in subset if r["hits"][name]["at_5"] and not r["hits"]["baseline"]["at_5"]],
                "top5_regressions": [r["sample_alias"] for r in subset if not r["hits"][name]["at_5"] and r["hits"]["baseline"]["at_5"]],
            }
    previous = prior._load(prior.OUTPUT / "evaluation_report.json")
    for group in ("old15", "new19"):
        expected = previous["old15_regression_check" if group == "old15" else group]["guarded_recall_counts"]
        if summaries[group]["variants"]["baseline"]["recall_counts"] != expected:
            raise ValueError(f"BASELINE_DRIFT:{group}")
    report = {
        "schema_version": "week34_causal_pair_evaluation_v2", "scientific_generalization_result": False,
        "summaries": summaries, "samples": evaluated,
        "frozen_predictions_sha256": _sha(OUTPUT / "frozen_predictions.json"),
        "external_call_counts": manifest["external_call_counts"],
    }
    atomic_json(OUTPUT / "evaluation_report.json", report)
    print(json.dumps(summaries, indent=2))
    return report


if __name__ == "__main__":
    run()

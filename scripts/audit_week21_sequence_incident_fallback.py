"""Offline audit of a small cross-transaction sequence fallback."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_v2.source_cache import atomic_json


MANIFEST = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "locked_inference_manifest.json"
TRUTH = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "evaluation_ground_truth.json"
PREDICTIONS = ROOT / "data" / "prototypes" / "localization15_deepseek_official_20260911" / "predictions"
OUTPUT = ROOT / "data" / "prototypes" / "week21_sequence_incident_fallback_20260911"
SUPPORTED_STATUSES = frozenset({"SELECTOR_EXACT_MATCH", "UNIQUE_NAME_MATCH"})
MAX_SEQUENCE_CANDIDATES = 5
MAX_PROMPT_CANDIDATES = 20

CandidateIdentity = tuple[str, str, int, int]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_hashes(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.lower()]
    if isinstance(value, list):
        return [item for child in value for item in _flatten_hashes(child)]
    return []


def _candidate_identity(row: Mapping[str, Any]) -> CandidateIdentity:
    return (
        str(row["relative_file"]).replace("\\", "/"),
        str(row["function_signature"]),
        int(row["start_line"]),
        int(row["end_line"]),
    )


def _public_identity(identity: CandidateIdentity) -> dict[str, Any]:
    return {
        "relative_file": identity[0],
        "function_signature": identity[1],
        "start_line": identity[2],
        "end_line": identity[3],
    }


def _sequence_aliases(samples: list[dict[str, Any]]) -> dict[str, str]:
    alias_by_tx = {sample["tx_hash"].lower(): sample["sample_alias"] for sample in samples}
    result: dict[str, str] = {}
    next_group = 1
    for path in sorted((ROOT / "dataset").rglob("*.json"), key=lambda item: item.as_posix()):
        document = _load(path)
        raw_groups = document.get("fault", {}).get("transaction_hash", [])
        groups = [raw_groups] if raw_groups and all(isinstance(item, str) for item in raw_groups) else raw_groups
        for raw_group in groups:
            selected = sorted({alias_by_tx[tx] for tx in _flatten_hashes(raw_group) if tx in alias_by_tx})
            if not selected:
                continue
            group_alias = f"G{next_group}"
            next_group += 1
            for sample_alias in selected:
                if sample_alias in result:
                    raise ValueError("SAMPLE_IN_MULTIPLE_SEQUENCES")
                result[sample_alias] = group_alias
    if set(result) != {sample["sample_alias"] for sample in samples}:
        raise ValueError("SEQUENCE_GROUP_MISSING")
    return result


def _freeze_selection() -> dict[str, Any]:
    samples = _load(MANIFEST)["samples"]
    group_by_sample = _sequence_aliases(samples)
    pools: dict[str, list[dict[str, Any]]] = {}
    statuses: dict[str, str] = {}
    for sample in samples:
        alias = sample["sample_alias"]
        prediction = _load(PREDICTIONS / f"{alias}.json")
        pools[alias] = list(prediction.get("pre_llm_candidates", []))
        statuses[alias] = str(prediction.get("preparation_status", "UNKNOWN"))

    members: dict[str, list[str]] = defaultdict(list)
    for alias, group in group_by_sample.items():
        members[group].append(alias)

    rows = []
    for sample in samples:
        alias = sample["sample_alias"]
        base = [_candidate_identity(row) for row in pools[alias] if row.get("included_in_prompt")]
        sibling_support: dict[CandidateIdentity, set[str]] = defaultdict(set)
        sibling_best_rank: dict[CandidateIdentity, int] = {}
        for sibling in members[group_by_sample[alias]]:
            if sibling == alias:
                continue
            for rank, candidate in enumerate(pools[sibling], start=1):
                if candidate.get("selector_match_status") not in SUPPORTED_STATUSES:
                    continue
                if "TRANSACTION_CALL_GRAPH_EXECUTED_FUNCTION" not in candidate.get("candidate_sources", []):
                    continue
                identity = _candidate_identity(candidate)
                sibling_support[identity].add(sibling)
                sibling_best_rank[identity] = min(rank, sibling_best_rank.get(identity, rank))
        supported = sorted(
            sibling_support,
            key=lambda identity: (
                -len(sibling_support[identity]),
                sibling_best_rank[identity],
                identity,
            ),
        )[:MAX_SEQUENCE_CANDIDATES]
        enhanced = list(dict.fromkeys([*supported, *base]))[:MAX_PROMPT_CANDIDATES]
        rows.append({
            "sample_alias": alias,
            "sequence_alias": group_by_sample[alias],
            "base_preparation_status": statuses[alias],
            "base_prompt_candidates": [_public_identity(item) for item in base],
            "sequence_supported_candidates": [
                {
                    **_public_identity(item),
                    "supporting_sibling_count": len(sibling_support[item]),
                    "best_sibling_rank": sibling_best_rank[item],
                }
                for item in supported
            ],
            "counterfactual_prompt_candidates": [_public_identity(item) for item in enhanced],
            "sequence_fallback_status": (
                "SEQUENCE_FALLBACK_CANDIDATES_AVAILABLE"
                if statuses[alias] != "READY_FOR_LLM" and enhanced
                else statuses[alias]
            ),
        })
    return {
        "schema_version": "week21_sequence_fallback_selection_v1",
        "mode": "OFFLINE_LABEL_FREE_COUNTERFACTUAL_SELECTION",
        "sample_count": len(rows),
        "supported_selector_statuses": sorted(SUPPORTED_STATUSES),
        "max_sequence_candidates": MAX_SEQUENCE_CANDIDATES,
        "max_prompt_candidates": MAX_PROMPT_CANDIDATES,
        "samples": rows,
    }


def _truth_hit(candidates: Iterable[Mapping[str, Any]], truths: list[dict[str, Any]]) -> bool:
    return any(
        str(candidate["relative_file"]).replace("\\", "/") == str(truth["relative_file"]).replace("\\", "/")
        and candidate["function_signature"] == truth["function_signature"]
        and int(candidate["start_line"]) <= int(truth["truth_end_line"])
        and int(candidate["end_line"]) >= int(truth["truth_start_line"])
        for candidate in candidates
        for truth in truths
    )


def run() -> dict[str, Any]:
    selection = _freeze_selection()
    atomic_json(OUTPUT / "selection.json", selection)

    truth = _load(TRUTH)
    truth_by_alias = {row["sample_alias"]: row["truths"] for row in truth["labels"]}
    rows = []
    for row in selection["samples"]:
        truths = truth_by_alias[row["sample_alias"]]
        base_hit = _truth_hit(row["base_prompt_candidates"], truths)
        enhanced_hit = _truth_hit(row["counterfactual_prompt_candidates"], truths)
        rows.append({
            "sample_alias": row["sample_alias"],
            "sequence_alias": row["sequence_alias"],
            "base_prompt_truth_hit": base_hit,
            "counterfactual_prompt_truth_hit": enhanced_hit,
            "newly_recovered": enhanced_hit and not base_hit,
            "regressed": base_hit and not enhanced_hit,
            "sequence_supported_candidate_count": len(row["sequence_supported_candidates"]),
            "sequence_fallback_status": row["sequence_fallback_status"],
        })
    report = {
        "schema_version": "week21_sequence_incident_fallback_report_v1",
        "scientific_generalization_result": False,
        "sample_count": len(rows),
        "base_prompt_truth_hit_count": sum(row["base_prompt_truth_hit"] for row in rows),
        "counterfactual_prompt_truth_hit_count": sum(row["counterfactual_prompt_truth_hit"] for row in rows),
        "newly_recovered_aliases": [row["sample_alias"] for row in rows if row["newly_recovered"]],
        "regressed_aliases": [row["sample_alias"] for row in rows if row["regressed"]],
        "sequence_fallback_candidate_available_count": sum(
            row["sequence_fallback_status"] == "SEQUENCE_FALLBACK_CANDIDATES_AVAILABLE" for row in rows
        ),
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": rows,
    }
    atomic_json(OUTPUT / "report.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "sample_count",
        "base_prompt_truth_hit_count",
        "counterfactual_prompt_truth_hit_count",
        "newly_recovered_aliases",
        "regressed_aliases",
        "sequence_fallback_candidate_available_count",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

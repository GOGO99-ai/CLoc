"""Build and validate representative prompts for multi-transaction sequences."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_week21_sequence_incident_fallback as sequence_audit
from scripts import run_localization15_offline_rebuild as offline
from scripts import run_week14_behavior_similarity as rebuild
from src.cloc_v2.phase2_localization_prompt import (
    serialize_phase2_localization_prompt,
    validate_phase2_localization_topk_v4,
)
from src.cloc_v2.source_cache import atomic_json


OUTPUT = ROOT / "data" / "prototypes" / "week22_sequence_prompt_adapter_20260911"
PREDICTIONS = sequence_audit.PREDICTIONS
MANIFEST = sequence_audit.MANIFEST
TRUTH = sequence_audit.TRUTH
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_WINDOWS_PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s\"']+")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["relative_file"]).replace("\\", "/"),
        str(row.get("function_signature", row.get("function"))),
        int(row["start_line"]),
        int(row["end_line"]),
    )


def _truth_hit(candidates: Iterable[tuple[str, str, int, int]], truths: list[dict[str, Any]]) -> bool:
    return any(
        candidate[0] == str(truth["relative_file"]).replace("\\", "/")
        and candidate[1] == truth["function_signature"]
        and candidate[2] <= int(truth["truth_end_line"])
        and candidate[3] >= int(truth["truth_start_line"])
        for candidate in candidates
        for truth in truths
    )


def _synthetic_response(prompt: Any) -> dict[str, Any]:
    package = json.loads(prompt.user_message.split("EVIDENCE_PACKAGE\n", 1)[1])
    count = package["limits"]["expected_findings"]
    return {
        "findings": [
            {
                "rank": rank,
                "source_slice_id": source["source_slice_id"],
                "evidence_ids": [source["allowed_evidence_ids"][0]],
            }
            for rank, source in enumerate(package["source_slices"][:count], start=1)
        ]
    }


def _prompt_checks(prompt: Any) -> dict[str, Any]:
    serialized = serialize_phase2_localization_prompt(prompt)
    llm_visible_text = prompt.system_message + "\n" + prompt.user_message
    package = json.loads(prompt.user_message.split("EVIDENCE_PACKAGE\n", 1)[1])
    validation = validate_phase2_localization_topk_v4(_synthetic_response(prompt), prompt)
    return {
        "prompt_schema_valid": prompt.to_dict().get("schema_version") == "phase2_localization_prompt_v1",
        "evidence_package_valid": package.get("schema_version") == "phase2_localization_evidence_v1",
        "citation_whitelist_consistent": package.get("citation_whitelist") == prompt.citation_whitelist.to_dict(),
        "raw_identifier_scan_clean": not (
            _RAW_ADDRESS.search(llm_visible_text)
            or _RAW_HASH.search(llm_visible_text)
            or _WINDOWS_PATH.search(llm_visible_text)
        ),
        "synthetic_topk_response_valid": validation.valid,
        "synthetic_validation_reason_codes": list(validation.reason_codes),
        "source_slice_count": len(package.get("source_slices", [])),
        "evidence_edge_count": len(package.get("evidence_edges", [])),
        "result_chain_count": len(package.get("result_chains", [])),
        "prompt_bytes": len((prompt.system_message + "\n" + prompt.user_message).encode("utf-8")),
    }


def _freeze_adapter() -> tuple[dict[str, Any], dict[str, tuple[tuple[str, str, int, int], ...]]]:
    samples = _load(MANIFEST)["samples"]
    by_alias = {sample["sample_alias"]: sample for sample in samples}
    group_by_sample = sequence_audit._sequence_aliases(samples)
    members: dict[str, list[str]] = defaultdict(list)
    for alias, group in group_by_sample.items():
        members[group].append(alias)

    predictions = {
        alias: _load(PREDICTIONS / f"{alias}.json")
        for alias in by_alias
    }
    donor_by_group: dict[str, str] = {}
    for group, aliases in members.items():
        if len(aliases) < 2:
            continue
        ready = [alias for alias in aliases if predictions[alias]["preparation_status"] == "READY_FOR_LLM"]
        if not ready:
            raise ValueError(f"SEQUENCE_DONOR_MISSING:{group}")
        donor_by_group[group] = sorted(
            ready,
            key=lambda alias: (-int(predictions[alias]["prompt_bytes"]), int(by_alias[alias]["order"])),
        )[0]

    normal_records = rebuild.replay._load_normal_records(rebuild.replay.DEFAULT_NORMAL)
    cache_index = rebuild.replay.build_verified_source_cache_index(
        rebuild.replay.DEFAULT_CACHE,
        blockchain_chain_id=1,
        chain_key="eth",
    )
    trees = offline._tree_index()
    group_candidates: dict[str, tuple[tuple[str, str, int, int], ...]] = {}
    rows = []
    for group, donor in sorted(donor_by_group.items()):
        sample = by_alias[donor]
        _slices, preparation = rebuild._build_sample(
            sample,
            trees[sample["tx_hash"].lower()][0],
            normal_records,
            cache_index,
            rebuild.replay.DEFAULT_CACHE,
        )
        if preparation.preparation_status != "READY_FOR_LLM" or preparation.prompt is None:
            raise ValueError(f"SEQUENCE_DONOR_NOT_READY:{group}")
        if preparation.prompt_sha256 != predictions[donor]["prompt_sha256"]:
            raise ValueError(f"SEQUENCE_DONOR_PROMPT_DRIFT:{group}")
        checks = _prompt_checks(preparation.prompt)
        if not all(checks[key] for key in (
            "prompt_schema_valid",
            "evidence_package_valid",
            "citation_whitelist_consistent",
            "raw_identifier_scan_clean",
            "synthetic_topk_response_valid",
        )):
            raise ValueError(f"SEQUENCE_PROMPT_VALIDATION_FAILED:{group}")
        atomic_json(OUTPUT / "group_prompts" / f"{group}.json", preparation.prompt.to_dict())
        locations = preparation.source_locations
        identities = tuple(
            _identity(locations[source_slice_id])
            for source_slice_id in preparation.prompt.citation_whitelist.source_slice_ids
        )
        group_candidates[group] = identities
        rows.append({
            "sequence_alias": group,
            "member_aliases": sorted(members[group], key=lambda alias: int(by_alias[alias]["order"])),
            "donor_alias": donor,
            "donor_selection_rule": "MAX_PROMPT_BYTES_THEN_SAMPLE_ORDER",
            "donor_prompt_sha256": preparation.prompt_sha256,
            "candidate_identities": [
                {
                    "relative_file": item[0],
                    "function_signature": item[1],
                    "start_line": item[2],
                    "end_line": item[3],
                }
                for item in identities
            ],
            "checks": checks,
        })
    manifest = {
        "schema_version": "week22_sequence_prompt_adapter_manifest_v1",
        "mode": "OFFLINE_LABEL_FREE_SEQUENCE_PROMPT_ADAPTER",
        "adapter_kind": "REPRESENTATIVE_SIBLING_PROMPT",
        "sequence_group_count": len(rows),
        "samples_covered_by_sequence_prompts": sum(len(row["member_aliases"]) for row in rows),
        "groups": rows,
    }
    return manifest, group_candidates


def run() -> dict[str, Any]:
    adapter, group_candidates = _freeze_adapter()
    atomic_json(OUTPUT / "adapter_manifest.json", adapter)

    samples = _load(MANIFEST)["samples"]
    group_by_sample = sequence_audit._sequence_aliases(samples)
    truths = {
        row["sample_alias"]: row["truths"]
        for row in _load(TRUTH)["labels"]
    }
    rows = []
    for sample in samples:
        alias = sample["sample_alias"]
        prediction = _load(PREDICTIONS / f"{alias}.json")
        base_candidates = tuple(
            _identity(row)
            for row in prediction.get("pre_llm_candidates", [])
            if row.get("included_in_prompt")
        )
        sequence_candidates = group_candidates.get(group_by_sample[alias], ())
        combined = tuple(dict.fromkeys((*base_candidates, *sequence_candidates)))
        base_hit = _truth_hit(base_candidates, truths[alias])
        combined_hit = _truth_hit(combined, truths[alias])
        rows.append({
            "sample_alias": alias,
            "sequence_alias": group_by_sample[alias],
            "base_prompt_truth_hit": base_hit,
            "base_or_sequence_prompt_truth_hit": combined_hit,
            "newly_recovered": combined_hit and not base_hit,
            "sequence_prompt_available": bool(sequence_candidates),
            "sequence_prompt_only": prediction["preparation_status"] != "READY_FOR_LLM" and bool(sequence_candidates),
        })
    report = {
        "schema_version": "week22_sequence_prompt_adapter_report_v1",
        "scientific_generalization_result": False,
        "sample_count": len(rows),
        "sequence_prompt_group_count": adapter["sequence_group_count"],
        "format_valid_group_prompt_count": sum(
            row["checks"]["synthetic_topk_response_valid"] for row in adapter["groups"]
        ),
        "base_prompt_truth_hit_count": sum(row["base_prompt_truth_hit"] for row in rows),
        "base_or_sequence_prompt_truth_hit_count": sum(
            row["base_or_sequence_prompt_truth_hit"] for row in rows
        ),
        "newly_recovered_aliases": [row["sample_alias"] for row in rows if row["newly_recovered"]],
        "sequence_prompt_only_aliases": [row["sample_alias"] for row in rows if row["sequence_prompt_only"]],
        "regressed_aliases": [
            row["sample_alias"]
            for row in rows
            if row["base_prompt_truth_hit"] and not row["base_or_sequence_prompt_truth_hit"]
        ],
        "external_call_counts": {"rpc": 0, "source": 0, "llm": 0},
        "samples": rows,
    }
    atomic_json(OUTPUT / "report.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "sample_count",
        "sequence_prompt_group_count",
        "format_valid_group_prompt_count",
        "base_prompt_truth_hit_count",
        "base_or_sequence_prompt_truth_hit_count",
        "newly_recovered_aliases",
        "sequence_prompt_only_aliases",
        "regressed_aliases",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

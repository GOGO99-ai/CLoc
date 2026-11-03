"""Build the 34-transaction FaultSeeker comparison manifest.

The selected transactions are the union of the CLoc samples marked
``READY_FOR_LLM`` in the frozen 15-case and remaining-case manifests.  Every
selected transaction must already exist in FaultSeeker's dataset-my manifest
and have function-level ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ready_hashes(*manifests: dict) -> list[str]:
    hashes = {
        sample["tx_hash"].lower()
        for manifest in manifests
        for sample in manifest["samples"]
        if sample.get("preparation_status") == "READY_FOR_LLM"
    }
    return sorted(hashes)


def build_manifest(cloc_dev: dict, cloc_remaining: dict, faultseeker: dict) -> dict:
    wanted = _ready_hashes(cloc_dev, cloc_remaining)
    by_hash = {
        case["transaction_hash"].lower(): case for case in faultseeker["cases"]
    }

    missing = [tx_hash for tx_hash in wanted if tx_hash not in by_hash]
    if missing:
        raise ValueError(f"FaultSeeker manifest is missing {len(missing)} transactions: {missing}")

    cases = [by_hash[tx_hash] for tx_hash in wanted]
    ineligible = [
        case["transaction_hash"]
        for case in cases
        if not case.get("evaluation_eligible")
    ]
    if ineligible:
        raise ValueError(f"Selected transactions without function-level ground truth: {ineligible}")
    if len(cases) != 34:
        raise ValueError(f"Expected 34 unique transactions, selected {len(cases)}")
    if {case["chain"] for case in cases} != {"eth"}:
        raise ValueError("The frozen 34-transaction comparison set must contain only Ethereum cases")

    locations = {
        location
        for case in cases
        for location in case.get("ground_truth_locations", [])
    }
    unresolved = {
        location
        for case in cases
        for location in case.get("unresolved_locations", [])
    }
    summary = {
        "incident_files": len({case["source_file"] for case in cases}),
        "transaction_cases": len(cases),
        "unique_transactions": len({case["transaction_hash"].lower() for case in cases}),
        "chains": dict(sorted(Counter(case["chain"] for case in cases).items())),
        "benchmark_transaction_overlaps": sum(
            bool(case.get("benchmark_transaction_overlap")) for case in cases
        ),
        "evaluation_eligible_cases": sum(
            bool(case.get("evaluation_eligible")) for case in cases
        ),
        "unique_ground_truth_locations": len(locations),
        "mapped_unique_locations": len(locations - unresolved),
        "unresolved_unique_locations": len(unresolved),
        "duplicate_transaction_hashes": [],
        "malformed_files": [],
    }
    return {
        "schema_version": faultseeker.get("schema_version", 1),
        "dataset_root": faultseeker.get("dataset_root"),
        "benchmark_root": faultseeker.get("benchmark_root"),
        "selection_provenance": {
            "rule": "CLoc preparation_status == READY_FOR_LLM",
            "cloc_dev_ready_count": sum(
                sample.get("preparation_status") == "READY_FOR_LLM"
                for sample in cloc_dev["samples"]
            ),
            "cloc_remaining_ready_count": sum(
                sample.get("preparation_status") == "READY_FOR_LLM"
                for sample in cloc_remaining["samples"]
            ),
            "selected_unique_transactions": len(cases),
        },
        "summary": summary,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloc-dev", type=Path, required=True)
    parser.add_argument("--cloc-remaining", type=Path, required=True)
    parser.add_argument("--faultseeker-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(
        _load(args.cloc_dev),
        _load(args.cloc_remaining),
        _load(args.faultseeker_manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "incident_name",
                "category",
                "chain",
                "transaction_hash",
                "ground_truth_functions",
            ),
        )
        writer.writeheader()
        for case in manifest["cases"]:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "incident_name": case["incident_name"],
                    "category": case["category"],
                    "chain": case["chain"],
                    "transaction_hash": case["transaction_hash"],
                    "ground_truth_functions": json.dumps(
                        case["ground_truth_functions"], ensure_ascii=False
                    ),
                }
            )

    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(args.output.resolve())
    print(csv_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

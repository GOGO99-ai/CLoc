"""Synthetic, offline engineering dry-run for formal phase-1 normality review.

The script deliberately uses only in-memory synthetic evidence.  It does not
read a real normal baseline, labels, configuration, RPC state, or secrets, and
it performs no LLM, API, RPC, or network call.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloc_v2.normal_baseline import build_normal_baseline
from src.cloc_v2.normal_distance import (
    AssetBehavior,
    CallStep,
    ChainFeatureRecord,
    MainAnchorKey,
)
from src.cloc_v2.normal_features import (
    ChainFeatureExtractionStatus,
    NormalFeatureExtractionResult,
)
from src.cloc_v2.phase1_normality_bridge import prepare_phase1_normality_review
from src.cloc_v2.phase1_normality_orchestration import decide_phase1_route
from src.cloc_v2.phase1_normality_prompt import (
    RESPONSE_SCHEMA_VERSION,
    validate_phase1_normality_response,
)
from src.cloc_v2.strict_result_chains import StrictResultChain, StrictResultChainsResult


SCHEMA_VERSION = "phase1_normality_dry_run_v1"
DRY_RUN_KIND = "SYNTHETIC_OFFLINE_ENGINEERING_CHECK"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "prototypes"
    / "phase1_normality_dry_run"
    / "phase1_normality_dry_run.json"
)

_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_TRANSACTION_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_SENSITIVE_KEYS = frozenset(
    {"calldata", "input", "output", "error", "revertreason", "record_id", "transaction_hash"}
)

_ANCHOR_ADDRESS = "0x" + "a" * 40
_TOKEN_ADDRESS = "0x" + "c" * 40
_RECIPIENT_ADDRESS = "0x" + "f" * 40
_SELECTOR = "0x11111111"
_QUERY_RECORD_ID = "synthetic-query-record"
_STRICT_CHAIN_ID = "synthetic_strict_result_chain_001"
_NORMAL_RECORD_IDS = tuple(f"synthetic-normal-record-{index}" for index in range(1, 4))
_NORMAL_TRANSACTION_IDS = tuple("0x" + str(index) * 64 for index in range(1, 4))
_QUERY_TRANSACTION_ID = "0x" + "9" * 64
_QUERY_AMOUNT = "123"


def _feature_record(record_id: str) -> ChainFeatureRecord:
    anchor = MainAnchorKey(1, _ANCHOR_ADDRESS, _SELECTOR)
    normal_index = (
        _NORMAL_RECORD_IDS.index(record_id) + 1
        if record_id in _NORMAL_RECORD_IDS
        else None
    )
    call_steps = (CallStep(_ANCHOR_ADDRESS, _SELECTOR),)
    if normal_index is not None:
        call_steps += (
            CallStep(_ANCHOR_ADDRESS, f"0x{normal_index + 1:08x}"),
        )
    return ChainFeatureRecord(
        record_id=record_id,
        main_anchor=anchor,
        call_steps=call_steps,
        asset_behaviors=frozenset({AssetBehavior(_TOKEN_ADDRESS, "OUTFLOW")}),
        uncertainties=(),
    )


def _strict_result() -> StrictResultChainsResult:
    chain = StrictResultChain(
        strict_chain_id=_STRICT_CHAIN_ID,
        result_id="synthetic-result-1",
        result_node_id="synthetic-result-node-1",
        result_fact_ids=("synthetic-result-fact-1",),
        result_type="R_OUTFLOW",
        asset_type="TOKEN",
        asset_address=_TOKEN_ADDRESS,
        from_address=_ANCHOR_ADDRESS,
        to_address=_RECIPIENT_ADDRESS,
        amount_raw=_QUERY_AMOUNT,
        result_uncertainties=(),
        node_ids=("synthetic-call-node-1",),
        edge_ids=(),
        anchor=None,
        status="READY",
        uncertainties=(),
    )
    return StrictResultChainsResult(
        blockchain_chain_id=1,
        edges=(),
        chains=(chain,),
        result_statuses=(),
        critical_uncertainties=(),
        coverage={
            "result_candidate_count": 1,
            "chains_formed_result_candidate_count": 1,
            "uncertain_result_candidate_count": 0,
            "critical_missing_count": 0,
            "coverage_rate": "1.0000",
        },
    )


def _query_features(query: ChainFeatureRecord) -> NormalFeatureExtractionResult:
    status = ChainFeatureExtractionStatus(
        strict_chain_id=_STRICT_CHAIN_ID,
        record_id=query.record_id,
        status="EXTRACTED_CLEAN",
        record_created=True,
        reasons=(),
        issues=(),
    )
    return NormalFeatureExtractionResult(
        transaction_record_id=_QUERY_TRANSACTION_ID,
        records=(query,),
        chain_statuses=(status,),
        input_chain_count=1,
        record_count=1,
        clean_record_count=1,
        uncertain_record_count=0,
        rejected_chain_count=0,
    )


def _decision_response(prompt: Any, decision: str) -> dict[str, Any]:
    query = prompt.citation_whitelist.chain_scopes[0]
    query_chain = json.loads(prompt.user_message)["query_chains"][0]
    reference = query_chain["normal_references"][0]
    query_call_id = query.query_call_evidence_ids[0]
    reference_call_id = query.reference_call_evidence_ids[0]
    query_asset_id = query.query_asset_evidence_ids[0]
    reference_asset_id = query.reference_asset_evidence_ids[0]
    reference_alias = reference["reference_alias"]

    if decision == "NORMAL":
        assessment = "CONSISTENT"
        reasons = [
            "CONSISTENT_CALL_SEQUENCE",
            "CONSISTENT_ASSET_BEHAVIOR",
            "WITHIN_CALIBRATED_THRESHOLD",
        ]
        evidence_ids = [query_call_id, query_asset_id, reference_call_id, reference_asset_id]
        rationale = (
            f"{query.chain_alias} matches {reference_alias} in CALL and ASSET evidence "
            f"{query_call_id}, {query_asset_id}, {reference_call_id}, and {reference_asset_id}."
        )
    elif decision == "SUSPICIOUS":
        assessment = "MATERIAL_DEVIATION"
        reasons = ["MATERIAL_CALL_DEVIATION"]
        evidence_ids = [query_call_id, reference_call_id]
        rationale = (
            f"{query.chain_alias} materially differs from {reference_alias} in "
            f"CALL evidence {query_call_id} and {reference_call_id}."
        )
    elif decision == "UNCERTAIN":
        assessment = "INSUFFICIENT_EVIDENCE"
        reasons = ["CRITICAL_EVIDENCE_INCOMPLETE"]
        evidence_ids = [query_call_id]
        rationale = f"{query.chain_alias} cannot be confirmed from evidence {query_call_id}."
    else:
        raise ValueError("SYNTHETIC_DECISION_INVALID")

    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "chain_decisions": [
            {
                "chain_alias": query.chain_alias,
                "decision": decision,
                "reference_assessment": assessment,
                "supporting_reference_aliases": [reference_alias],
                "evidence_ids": evidence_ids,
                "reason_codes": reasons,
                "brief_rationale": rationale,
            }
        ],
    }


def _route_summary(decision: Any) -> dict[str, Any]:
    return {
        "route": decision.route,
        "llm_response_processed": decision.llm_response_processed,
        "reason_codes": list(decision.reason_codes),
        "chain_decisions": [item.to_dict() for item in decision.chain_decisions],
    }


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key).lower() for key in value} | set().union(
            *(_all_keys(item) for item in value.values()), set()
        )
    if isinstance(value, (list, tuple)):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def _sanitization_check(
    report: Mapping[str, Any],
    *,
    forbidden_internal_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    text = json.dumps(report, ensure_ascii=False, sort_keys=True)
    evidence_package = json.loads(str(report["user_message"]))
    query_chains = evidence_package["query_chains"]
    reference_assets = [
        asset
        for chain in query_chains
        for reference in chain["normal_references"]
        for asset in reference["asset_evidence"]
    ]
    query_amounts = [
        asset.get("amount_observation")
        for chain in query_chains
        for asset in chain["query_asset_evidence"]
        if asset.get("amount_observation") is not None
    ]
    keys = _all_keys(report) | _all_keys(evidence_package)
    checks = {
        "no_raw_20_byte_address": _RAW_ADDRESS.search(text) is None,
        "no_raw_32_byte_transaction_hash": _RAW_TRANSACTION_HASH.search(text) is None,
        "no_internal_normal_or_query_identity": all(value not in text for value in forbidden_internal_values),
        "no_sensitive_trace_or_record_fields": not bool(keys & _SENSITIVE_KEYS),
        "normal_references_have_no_amount_observation": all(
            "amount_observation" not in asset for asset in reference_assets
        ),
        "query_amount_is_explanatory_only": bool(query_amounts)
        and all(
            amount == {
                "role": "query_only_explanatory_observation",
                "value": _QUERY_AMOUNT,
            }
            for amount in query_amounts
        ),
        "selector_retained": _SELECTOR in text,
    }
    return {"passed": all(checks.values()), "checks": checks}


def build_synthetic_phase1_preparation():
    """Return the single synthetic preparation shared by offline engineering checks."""
    normals = tuple(_feature_record(record_id) for record_id in _NORMAL_RECORD_IDS)
    baseline = build_normal_baseline(normals, min_group_size=3)
    query_record = _feature_record(_QUERY_RECORD_ID)
    mapping = dict(zip(_NORMAL_RECORD_IDS, _NORMAL_TRANSACTION_IDS, strict=True))
    preparation = prepare_phase1_normality_review(
        _strict_result(),
        _query_features(query_record),
        baseline,
        mapping,
    )
    if (
        preparation.preparation_status != "READY_FOR_LLM"
        or preparation.prompt is None
        or preparation.gate_evidence is None
        or preparation.eligibility.eligibility != "ELIGIBLE"
    ):
        raise ValueError("SYNTHETIC_PREPARATION_NOT_READY")
    return preparation


def build_synthetic_phase1_dry_run() -> dict[str, Any]:
    """Build one deterministic, fully offline phase-1 engineering dry-run."""
    preparation = build_synthetic_phase1_preparation()

    prompt = preparation.prompt
    query_chain = preparation.gate_evidence.query_chains[0]
    gate_chain = preparation.gate_evidence.strict_chains[0]
    normal_response = _decision_response(prompt, "NORMAL")
    suspicious_response = _decision_response(prompt, "SUSPICIOUS")
    uncertain_response = _decision_response(prompt, "UNCERTAIN")
    for response in (normal_response, suspicious_response, uncertain_response):
        validate_phase1_normality_response(response, prompt)
    invalid_response = dict(normal_response)
    invalid_response["transaction_level_decision"] = "NORMAL"

    route_results = {
        "NO_RESPONSE": _route_summary(
            decide_phase1_route(preparation.gate_evidence, prompt, None)
        ),
        "NORMAL_RESPONSE": _route_summary(
            decide_phase1_route(preparation.gate_evidence, prompt, normal_response)
        ),
        "SUSPICIOUS_RESPONSE": _route_summary(
            decide_phase1_route(preparation.gate_evidence, prompt, suspicious_response)
        ),
        "UNCERTAIN_RESPONSE": _route_summary(
            decide_phase1_route(preparation.gate_evidence, prompt, uncertain_response)
        ),
        "INVALID_RESPONSE": _route_summary(
            decide_phase1_route(preparation.gate_evidence, prompt, invalid_response)
        ),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dry_run_kind": DRY_RUN_KIND,
        "scientific_experiment_result": False,
        "preparation_status": preparation.preparation_status,
        "eligibility": preparation.eligibility.eligibility,
        "baseline_summary": {
            "min_group_size": gate_chain.calibration_min_group_size,
            "normal_group_size": gate_chain.normal_group_size,
            "calibration_status": gate_chain.threshold_calibration_status,
            "dynamic_threshold": str(query_chain.dynamic_threshold),
        },
        "query_summary": {
            "quantitative_status": query_chain.quantitative_status,
            "nearest_distance": str(query_chain.nearest_distance),
            "top3_count": len(query_chain.normal_references),
        },
        "system_message": prompt.system_message,
        "user_message": prompt.user_message,
        "response_schema": prompt.response_schema,
        "citation_whitelist": prompt.citation_whitelist.to_dict(),
        "route_results": route_results,
        "external_call_counts": {"llm": 0, "api": 0, "rpc": 0, "network": 0},
    }
    forbidden = (
        _ANCHOR_ADDRESS,
        _TOKEN_ADDRESS,
        _RECIPIENT_ADDRESS,
        _QUERY_RECORD_ID,
        _STRICT_CHAIN_ID,
        *_NORMAL_RECORD_IDS,
        *_NORMAL_TRANSACTION_IDS,
        _QUERY_TRANSACTION_ID,
    )
    report["sanitization_check"] = _sanitization_check(
        report,
        forbidden_internal_values=forbidden,
    )
    if not report["sanitization_check"]["passed"]:
        raise ValueError("PHASE1_DRY_RUN_SANITIZATION_FAILED")
    return report


def serialize_phase1_normality_dry_run(report: Mapping[str, Any]) -> str:
    """Return stable JSON only after rechecking the public report contract."""
    if not isinstance(report, Mapping):
        raise ValueError("PHASE1_DRY_RUN_REPORT_INVALID")
    required = {
        "schema_version",
        "dry_run_kind",
        "scientific_experiment_result",
        "preparation_status",
        "eligibility",
        "baseline_summary",
        "query_summary",
        "system_message",
        "user_message",
        "response_schema",
        "citation_whitelist",
        "route_results",
        "external_call_counts",
        "sanitization_check",
    }
    if set(report) != required:
        raise ValueError("PHASE1_DRY_RUN_REPORT_FIELDS_INVALID")
    if report["schema_version"] != SCHEMA_VERSION or report["dry_run_kind"] != DRY_RUN_KIND:
        raise ValueError("PHASE1_DRY_RUN_REPORT_VERSION_INVALID")
    if report["scientific_experiment_result"] is not False:
        raise ValueError("SCIENTIFIC_EXPERIMENT_RESULT_MUST_BE_FALSE")
    fresh_check = _sanitization_check({key: value for key, value in report.items() if key != "sanitization_check"})
    if not fresh_check["passed"] or report["sanitization_check"] != fresh_check:
        raise ValueError("PHASE1_DRY_RUN_SANITIZATION_FAILED")
    return json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_phase1_normality_dry_run(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Atomically write the single sanitized report after all checks pass."""
    report = build_synthetic_phase1_dry_run()
    serialized = serialize_phase1_normality_dry_run(report)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def main() -> int:
    try:
        destination = write_phase1_normality_dry_run()
    except (OSError, TypeError, ValueError):
        print("PHASE1_NORMALITY_DRY_RUN_FAILED", file=sys.stderr)
        return 1
    print(f"PHASE1_NORMALITY_DRY_RUN_WRITTEN: {destination.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DRY_RUN_KIND",
    "SCHEMA_VERSION",
    "build_synthetic_phase1_dry_run",
    "build_synthetic_phase1_preparation",
    "serialize_phase1_normality_dry_run",
    "write_phase1_normality_dry_run",
]

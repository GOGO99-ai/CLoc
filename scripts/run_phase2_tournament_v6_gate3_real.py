"""One-sample real regression gate for PHASE2_SOURCE_TOURNAMENT_TOPK_V6."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_localization15_topk_v4 import (
    LOCKED_INFERENCE_MANIFEST_SOURCE,
    MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
    Phase2ClassifiedHTTPSOneShotTransport,
    SOURCE_CACHE,
    _attach_formal_candidate_union,
    _single_truth_hit,
    load_run_manifest,
    prepare_localization_v4,
)
from scripts.run_phase2_real_pilot import (
    Phase2SourceStageError,
    acquire_real_source_slices,
    build_real_phase2_evidence,
)
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.phase2_localization_bridge import prepare_phase2_localization
from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekConfig
from src.cloc_v2.phase2_source_tournament import (
    GROUP_INCOMPLETE,
    FINAL_INVALID,
    METHOD_NAME,
    TournamentCandidate,
    TournamentEvidence,
    TournamentEvidenceContext,
    TournamentFinalist,
    TournamentGroup,
    TournamentResultChain,
    build_tournament_candidates_from_union,
    build_tournament_final_prompt,
    build_tournament_group_prompt,
    build_tournament_groups,
    execute_tournament_prompt_with_single_retry,
    merge_tournament_group_winners,
)


EXPERIMENT_TYPE = "PHASE2_SOURCE_TOURNAMENT_TOPK_V6_GATE3_REAL_REGRESSION"
SAFE_ALIAS = "v6_gate3_real_001"
LOCKED_NEUTRAL_ALIAS = "localization_v4_eth_005"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "prototypes" / "phase2_tournament_v6_gate3_real_001_20260901"
GATE2_REPORT = ROOT / "data" / "prototypes" / "phase2_tournament_v6_synthetic_smoke_20260901" / "phase2_tournament_v6_synthetic_smoke_report.json"
MAX_RPC_REQUESTS = 2
MAX_SOURCE_REQUESTS = 3
MAX_GROUP_INITIAL_REQUESTS = 5
MAX_FINAL_INITIAL_REQUESTS = 1
MAX_DEEPSEEK_WITH_RETRY = 12
_TX_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_WINDOWS_PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\r\n]*")
_FORBIDDEN_KEYS = frozenset({
    "tx_hash", "transaction_hash", "address", "rpc_url", "api_key", "authorization",
    "raw_trace", "calldata", "input", "output", "source_text", "system_message",
    "user_message", "provider_response", "reasoning_content", "location", "cause",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        if Path(handle.name).exists():
            Path(handle.name).unlink()


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key).strip().lower().replace("-", "_") for key in value}
        for child in value.values():
            found.update(_all_keys(child))
        return found
    if isinstance(value, (list, tuple)):
        found: set[str] = set()
        for child in value:
            found.update(_all_keys(child))
        return found
    return set()


def validate_public(value: Mapping[str, Any]) -> None:
    rendered = _canonical(value)
    if _RAW_ADDRESS.search(rendered) or _RAW_HASH.search(rendered) or _WINDOWS_PATH.search(rendered):
        raise ValueError("TOURNAMENT_V6_GATE3_SENSITIVE")
    if _FORBIDDEN_KEYS.intersection(_all_keys(value)):
        raise ValueError("TOURNAMENT_V6_GATE3_SENSITIVE")
    lowered = rendered.lower()
    if "http://" in lowered or "https://" in lowered:
        raise ValueError("TOURNAMENT_V6_GATE3_SENSITIVE")


@dataclass(frozen=True)
class LockedGate3Sample:
    sample_alias: str
    blockchain_chain_id: int
    _tx_hash: str = field(repr=False)
    neutral_manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            self.sample_alias != SAFE_ALIAS
            or self.blockchain_chain_id != 1
            or not isinstance(self._tx_hash, str)
            or _TX_HASH.fullmatch(self._tx_hash.lower()) is None
            or not re.fullmatch(r"[0-9a-f]{64}", self.neutral_manifest_sha256)
        ):
            raise ValueError("TOURNAMENT_V6_GATE3_LOCK_INVALID")

    @property
    def tx_hash(self) -> str:
        return self._tx_hash

    def public_manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": "phase2_tournament_v6_gate3_manifest_v1",
            "experiment_type": EXPERIMENT_TYPE,
            "scientific_experiment_result": False,
            "development_regression_sample": True,
            "sample_alias": self.sample_alias,
            "blockchain_chain_id": self.blockchain_chain_id,
            "selection_rule": "FIXED_EXISTING_NEUTRAL_MANIFEST_ALIAS",
            "neutral_manifest_sha256": self.neutral_manifest_sha256,
        }
        return {**core, "manifest_sha256": _sha(core)}


def lock_gate3_sample(manifest_path: Path = LOCKED_INFERENCE_MANIFEST_SOURCE) -> LockedGate3Sample:
    manifest = load_run_manifest(manifest_path, expected_sample_count=15)
    selected = next((item for item in manifest.samples if item.sample_alias == LOCKED_NEUTRAL_ALIAS), None)
    if selected is None or selected.blockchain_chain_id != 1:
        raise ValueError("TOURNAMENT_V6_GATE3_SAMPLE_NOT_FOUND")
    return LockedGate3Sample(SAFE_ALIAS, 1, selected.tx_hash.lower(), manifest.manifest_sha256)


def validate_gate2_report(path: Path = GATE2_REPORT) -> None:
    value = _load(path)
    if (
        not isinstance(value, Mapping)
        or value.get("group_request_status") != "VALIDATED"
        or value.get("final_request_status") != "VALIDATED"
        or value.get("gate2_success") is not True
        or value.get("deepseek_request_count") != 2
        or value.get("retry_count") != 0
    ):
        raise ValueError("TOURNAMENT_V6_GATE2_NOT_VALIDATED")


def _prompt_package(preparation: Any) -> dict[str, Any]:
    if preparation.prompt is None or "EVIDENCE_PACKAGE\n" not in preparation.prompt.user_message:
        raise ValueError("TOURNAMENT_V6_BRIDGE_PROMPT_INVALID")
    return json.loads(preparation.prompt.user_message.split("EVIDENCE_PACKAGE\n", 1)[1])


def _tournament_context(package: Mapping[str, Any]) -> TournamentEvidenceContext:
    chains = tuple(
        TournamentResultChain(
            row["chain_alias"],
            row["result_summary"],
            "READY",
            tuple(row.get("uncertainties", ())),
        )
        for row in package.get("result_chains", ())
    )
    edges = tuple(
        TournamentEvidence(
            row["evidence_id"],
            row["chain_alias"],
            row["edge_type"],
            row["source_alias"],
            row["target_alias"],
            row["fact_description"],
            tuple(row.get("uncertainties", ())),
        )
        for row in package.get("evidence_edges", ())
    )
    return TournamentEvidenceContext("T1", chains, edges)


def _bridge_prepare(bundle: Any, slices: Mapping[str, Any], shortlist: Sequence[str] | None = None) -> Any:
    return prepare_phase2_localization(
        bundle.strict,
        bundle.causal,
        bundle.graph,
        slices,
        max_pre_llm_candidates=MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
        max_prompt_candidates=20,
        max_prompt_bytes=None,
        prompt_contract="topk_v4",
        shortlist_candidate_ids=shortlist,
    )


@dataclass(frozen=True)
class HydratedTournamentInput:
    candidates: tuple[TournamentCandidate, ...]
    context: TournamentEvidenceContext
    public_candidate_rows: tuple[Mapping[str, Any], ...]
    grouping: Any


def hydrate_tournament_input(bundle: Any, slices: Mapping[str, Any]) -> HydratedTournamentInput:
    base = _bridge_prepare(bundle, slices)
    if base.preparation_status != "READY_FOR_LLM" or not base.pre_llm_candidates:
        raise ValueError("TOURNAMENT_V6_CANDIDATE_POOL_UNAVAILABLE")
    rows = [dict(item) for item in base.pre_llm_candidates]
    rows.sort(key=lambda row: int(str(row.get("candidate_alias", "C9999"))[1:]))
    if [row.get("candidate_alias") for row in rows] != [f"C{index}" for index in range(1, len(rows) + 1)]:
        raise ValueError("TOURNAMENT_V6_CANDIDATE_ORDER_INVALID")
    placeholder_rows = []
    for index, row in enumerate(rows, start=1):
        value = dict(row)
        value.update(source_slice_id=f"S{index}", supporting_evidence_ids=["E1"])
        if not isinstance(value.get("fusion_score"), (int, float)):
            value["fusion_score"] = 0.0
        if not isinstance(value.get("fusion_rank"), int):
            value["fusion_rank"] = index
        placeholder_rows.append(value)
    placeholders = build_tournament_candidates_from_union(
        placeholder_rows,
        {f"C{index}": f"function placeholder_{index}() public {{}}" for index in range(1, len(rows) + 1)},
    )
    grouping = build_tournament_groups(placeholders)
    if grouping.audit.status != "VALID":
        raise ValueError("TOURNAMENT_GROUPING_INVALID")
    source_text_by_id: dict[str, str] = {}
    evidence_by_id: dict[str, list[str]] = {}
    context: TournamentEvidenceContext | None = None
    for group in grouping.groups:
        candidate_ids = tuple(item.candidate_id for item in group.candidates)
        preparation = _bridge_prepare(bundle, slices, candidate_ids)
        if preparation.preparation_status != "READY_FOR_LLM" or preparation.prompt_candidate_count != len(candidate_ids):
            raise ValueError("TOURNAMENT_V6_GROUP_SOURCE_HYDRATION_FAILED")
        package = _prompt_package(preparation)
        current_context = _tournament_context(package)
        if context is None:
            context = current_context
        elif context.to_dict() != current_context.to_dict():
            raise ValueError("TOURNAMENT_V6_GROUP_CONTEXT_DRIFT")
        local_by_slice = {item["source_slice_id"]: item for item in package.get("source_slices", ())}
        public_by_slice = {
            item.get("source_slice_id"): item.get("candidate_alias")
            for item in preparation.pre_llm_candidates
            if item.get("included_in_prompt") is True
        }
        if set(public_by_slice) != set(local_by_slice) or set(public_by_slice.values()) != set(candidate_ids):
            raise ValueError("TOURNAMENT_V6_GROUP_SOURCE_ALIGNMENT_INVALID")
        for source_slice_id, candidate_id in public_by_slice.items():
            local = local_by_slice[source_slice_id]
            source_text_by_id[candidate_id] = local["source_text"]
            evidence_by_id[candidate_id] = list(local["allowed_evidence_ids"])
    if context is None or set(source_text_by_id) != {row["candidate_alias"] for row in rows}:
        raise ValueError("TOURNAMENT_V6_CANDIDATE_HYDRATION_INCOMPLETE")
    hydrated_rows = []
    public_rows = []
    for index, row in enumerate(rows, start=1):
        candidate_id = row["candidate_alias"]
        value = dict(row)
        value.update(source_slice_id=f"S{index}", supporting_evidence_ids=evidence_by_id[candidate_id])
        if not isinstance(value.get("fusion_score"), (int, float)):
            value["fusion_score"] = 0.0
        if not isinstance(value.get("fusion_rank"), int):
            value["fusion_rank"] = index
        hydrated_rows.append(value)
        public_rows.append({
            "candidate_alias": candidate_id,
            "relative_file": str(row["relative_file"]).replace("\\", "/"),
            "function_signature": row["function_signature"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "visibility": row.get("visibility", "UNKNOWN"),
        })
    candidates = build_tournament_candidates_from_union(hydrated_rows, source_text_by_id)
    final_grouping = build_tournament_groups(candidates)
    if final_grouping.audit.to_dict() != grouping.audit.to_dict():
        raise ValueError("TOURNAMENT_V6_GROUPING_CHANGED_AFTER_HYDRATION")
    return HydratedTournamentInput(candidates, context, tuple(public_rows), final_grouping)


def _base_prediction(lock: LockedGate3Sample) -> dict[str, Any]:
    return {
        "schema_version": "phase2_tournament_v6_gate3_prediction_v1",
        "method": METHOD_NAME,
        "sample_alias": lock.sample_alias,
        "status": "STOPPED",
        "reason_codes": [],
        "candidate_pool_count": 0,
        "candidate_rows": [],
        "grouping_audit": None,
        "groups": [],
        "finalist_count": 0,
        "final_prompt_bytes": 0,
        "final_status": "NOT_CALLED",
        "global_top5": [],
        "provider_free_text_used": False,
        "deterministic_fallback_used": False,
        "external_call_counts": {"rpc": 0, "source_network": 0, "group_deepseek": 0, "final_deepseek": 0, "deepseek_total": 0},
        "retry_counts": {"group": 0, "final": 0, "total": 0},
        "source_resolution_mode": "UNAVAILABLE",
        "source_selection_mode": "UNAVAILABLE",
    }


def _checkpoint(lock: LockedGate3Sample, status: str, prediction: Mapping[str, Any], frozen_sha: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": "phase2_tournament_v6_gate3_checkpoint_v1",
        "manifest_sha256": lock.public_manifest()["manifest_sha256"],
        "status": status,
        "sample_alias": lock.sample_alias,
        "completed_groups": [row["group_alias"] for row in prediction.get("groups", ()) if row.get("status") == "VALIDATED"],
        "external_call_counts": dict(prediction["external_call_counts"]),
        "retry_counts": dict(prediction["retry_counts"]),
        "frozen_prediction_sha256": frozen_sha,
        "reason_codes": list(prediction.get("reason_codes", ())),
    }


def _freeze(lock: LockedGate3Sample, prediction: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema_version": "phase2_tournament_v6_gate3_frozen_prediction_v1",
        "kind": METHOD_NAME,
        "locked_manifest_sha256": lock.public_manifest()["manifest_sha256"],
        "prediction": dict(prediction),
    }
    return {**core, "payload_sha256": _sha(core)}


def validate_frozen(path: Path, lock: LockedGate3Sample) -> Mapping[str, Any]:
    value = _load(path)
    core = {key: value[key] for key in ("schema_version", "kind", "locked_manifest_sha256", "prediction")}
    if value.get("payload_sha256") != _sha(core) or value.get("locked_manifest_sha256") != lock.public_manifest()["manifest_sha256"]:
        raise ValueError("TOURNAMENT_V6_GATE3_FROZEN_SHA_INVALID")
    validate_public(value)
    return value


def execute_blind_gate3(
    lock: LockedGate3Sample,
    output_dir: Path,
    chain_config: Mapping[str, Any],
    api_key: str,
    *,
    rpc_factory: Callable[..., Any] = JsonRpcClient,
    evidence_builder: Callable[[Any], Any] = build_real_phase2_evidence,
    source_acquirer: Callable[[Any, Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any]]] = acquire_real_source_slices,
    union_builder: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] = _attach_formal_candidate_union,
    tournament_hydrator: Callable[[Any, Mapping[str, Any]], HydratedTournamentInput] = hydrate_tournament_input,
    transport_factory: Callable[[], Any] = Phase2ClassifiedHTTPSOneShotTransport,
) -> Mapping[str, Any]:
    prediction = _base_prediction(lock)
    checkpoint_path = output_dir / "checkpoint.json"
    _atomic(checkpoint_path, _checkpoint(lock, "RUNNING", prediction))
    rpc_url = chain_config.get("rpc_url") if isinstance(chain_config, Mapping) else None
    if not isinstance(rpc_url, str) or not rpc_url or not isinstance(api_key, str) or not api_key:
        prediction["reason_codes"] = ["TOURNAMENT_V6_GATE3_CONFIGURATION_MISSING"]
        return prediction
    rpc = rpc_factory(rpc_url, max_requests=MAX_RPC_REQUESTS)
    try:
        chain_id = rpc.call("eth_chainId", [])
        prediction["external_call_counts"]["rpc"] = int(rpc.request_count)
        if str(chain_id).lower() not in {"1", "0x1"}:
            prediction["reason_codes"] = ["CHAIN_ID_MISMATCH"]
            return prediction
        raw = rpc.call("debug_traceTransaction", [lock.tx_hash, {"tracer": "callTracer"}])
        prediction["external_call_counts"]["rpc"] = int(rpc.request_count)
        bundle = evidence_builder(raw)
        slices, source_summary = source_acquirer(bundle, chain_config)
        source_count = int(source_summary.get("source_request_count", 0))
        prediction["external_call_counts"]["source_network"] = source_count
        prediction["source_resolution_mode"] = source_summary.get("source_resolution_mode", "UNAVAILABLE")
        prediction["source_selection_mode"] = source_summary.get("source_selection_mode", "UNAVAILABLE")
        if source_count > MAX_SOURCE_REQUESTS or source_summary.get("stop_code"):
            prediction["reason_codes"] = [source_summary.get("stop_code") or "SOURCE_REQUEST_LIMIT_EXCEEDED"]
            return prediction
        slices = union_builder(bundle, slices)
        hydrated = tournament_hydrator(bundle, slices)
        grouping = hydrated.grouping
        prediction["candidate_pool_count"] = len(hydrated.candidates)
        prediction["candidate_rows"] = [dict(row) for row in hydrated.public_candidate_rows]
        prediction["grouping_audit"] = grouping.audit.to_dict()
        if grouping.audit.status != "VALID":
            prediction["reason_codes"] = ["TOURNAMENT_GROUPING_INVALID"]
            return prediction
        validations = {}
        group_results = []
        config = Phase2DeepSeekConfig()
        for group in grouping.groups:
            built = build_tournament_group_prompt(group, hydrated.context)
            if built.status != "READY_FOR_LLM" or built.prompt is None:
                prediction["reason_codes"] = list(built.reason_codes) or [GROUP_INCOMPLETE]
                return prediction
            result = execute_tournament_prompt_with_single_retry(built.prompt, config, transport_factory, api_key)
            prediction["external_call_counts"]["group_deepseek"] += result.transport_audit.request_count
            prediction["external_call_counts"]["deepseek_total"] += result.transport_audit.request_count
            prediction["retry_counts"]["group"] += result.transport_audit.retry_count
            prediction["retry_counts"]["total"] += result.transport_audit.retry_count
            winners = []
            if result.validation is not None:
                by_slice = {item.source_slice_id: item for item in group.candidates}
                winners = [by_slice[item.source_slice_id].candidate_id for item in result.validation.findings]
            group_results.append({
                "group_alias": group.group_alias,
                "candidate_count": len(group.candidates),
                "prompt_bytes": built.prompt.prompt_bytes,
                "status": result.status,
                "request_count": result.transport_audit.request_count,
                "retry_count": result.transport_audit.retry_count,
                "winner_candidate_ids": winners,
                "reason_codes": list(result.reason_codes),
            })
            prediction["groups"] = group_results
            _atomic(checkpoint_path, _checkpoint(lock, "RUNNING", prediction))
            if result.status != "VALIDATED" or result.validation is None:
                prediction["reason_codes"] = sorted(set(result.reason_codes) | {GROUP_INCOMPLETE})
                return prediction
            validations[group.group_alias] = result.validation
        finalists = merge_tournament_group_winners(grouping, validations)
        prediction["finalist_count"] = len(finalists)
        final_built = build_tournament_final_prompt(finalists, hydrated.context)
        if final_built.status != "READY_FOR_LLM" or final_built.prompt is None:
            prediction["reason_codes"] = sorted(set(final_built.reason_codes) | {FINAL_INVALID})
            return prediction
        prediction["final_prompt_bytes"] = final_built.prompt.prompt_bytes
        final = execute_tournament_prompt_with_single_retry(final_built.prompt, config, transport_factory, api_key)
        prediction["external_call_counts"]["final_deepseek"] = final.transport_audit.request_count
        prediction["external_call_counts"]["deepseek_total"] += final.transport_audit.request_count
        prediction["retry_counts"]["final"] = final.transport_audit.retry_count
        prediction["retry_counts"]["total"] += final.transport_audit.retry_count
        prediction["final_status"] = final.status
        if final.status != "VALIDATED" or final.validation is None:
            prediction["reason_codes"] = sorted(set(final.reason_codes) | {FINAL_INVALID})
            return prediction
        candidate_by_slice = {item.source_slice_id: item for item in hydrated.candidates}
        public_by_id = {row["candidate_alias"]: row for row in hydrated.public_candidate_rows}
        global_top5 = []
        for finding in final.validation.findings:
            candidate = candidate_by_slice[finding.source_slice_id]
            public = public_by_id[candidate.candidate_id]
            line_start = finding.line_start if finding.line_start is not None else candidate.start_line
            line_end = finding.line_end if finding.line_end is not None else candidate.end_line
            global_top5.append({
                "rank": finding.rank,
                "candidate_alias": candidate.candidate_id,
                "relative_file": public["relative_file"],
                "function": public["function_signature"],
                "predicted_line_start": line_start,
                "predicted_line_end": line_end,
                "line_range_source": "PROVIDER_SELECTED_LINE_RANGE" if finding.line_start is not None else "LOCAL_EXACT_FUNCTION_RANGE",
                "evidence_ids": list(finding.evidence_ids),
            })
        prediction["global_top5"] = global_top5
        prediction["status"] = "COMPLETED"
        prediction["reason_codes"] = ["TOURNAMENT_V6_ENGINEERING_CHAIN_VALIDATED"]
    except Phase2SourceStageError as exc:
        prediction["external_call_counts"]["source_network"] = int(exc.request_count)
        prediction["reason_codes"] = ["SOURCE_PROCESSING_ERROR"]
    except Exception:
        prediction["reason_codes"] = ["TOURNAMENT_V6_GATE3_EXTERNAL_OR_PROCESSING_ERROR"]
    if prediction["external_call_counts"]["rpc"] == 0 and "rpc" in locals() and hasattr(rpc, "request_count"):
        prediction["external_call_counts"]["rpc"] = int(rpc.request_count)
    counts = prediction["external_call_counts"]
    if (
        counts["rpc"] > MAX_RPC_REQUESTS
        or counts["source_network"] > MAX_SOURCE_REQUESTS
        or counts["group_deepseek"] > MAX_GROUP_INITIAL_REQUESTS * 2
        or counts["final_deepseek"] > MAX_FINAL_INITIAL_REQUESTS * 2
        or counts["deepseek_total"] > MAX_DEEPSEEK_WITH_RETRY
    ):
        prediction.update(status="STOPPED", reason_codes=["TOURNAMENT_V6_GATE3_REQUEST_BUDGET_EXCEEDED"])
    validate_public(prediction)
    return prediction


def _truth_loader(lock: LockedGate3Sample) -> tuple[Mapping[str, Any], str]:
    manifest, truth, _ = prepare_localization_v4(ROOT / "dataset", SOURCE_CACHE)
    if manifest.manifest_sha256 != lock.neutral_manifest_sha256:
        raise ValueError("TOURNAMENT_V6_GATE3_TRUTH_MANIFEST_MISMATCH")
    label = next((item for item in truth.labels if item.get("sample_alias") == LOCKED_NEUTRAL_ALIAS), None)
    if not isinstance(label, Mapping):
        raise ValueError("TOURNAMENT_V6_GATE3_TRUTH_UNAVAILABLE")
    return label, truth.ground_truth_sha256


def evaluate_frozen_gate3(
    frozen: Mapping[str, Any],
    label: Mapping[str, Any],
) -> dict[str, Any]:
    prediction = frozen["prediction"]
    candidates = [row for row in prediction.get("candidate_rows", ()) if isinstance(row, Mapping)]
    truths = label.get("truths", ())
    true_candidate_ids = {
        row["candidate_alias"]
        for row in candidates
        if any(_single_truth_hit(truth, row, "function") for truth in truths)
    }
    group_winner_ids = {
        candidate_id
        for group in prediction.get("groups", ())
        for candidate_id in group.get("winner_candidate_ids", ())
    }
    topk = prediction.get("global_top5", ())
    final_truth_rows = [row for row in topk if any(_single_truth_hit(truth, row, "function") for truth in truths)]
    return {
        "truth_function_in_candidate_pool": bool(true_candidate_ids),
        "truth_function_in_group_top5": bool(true_candidate_ids & group_winner_ids),
        "truth_function_in_global_top5": bool(final_truth_rows),
        "truth_function_final_rank": min((row["rank"] for row in final_truth_rows), default=None),
        "file_hit": any(any(_single_truth_hit(truth, row, "file") for truth in truths) for row in topk),
        "function_hit": bool(final_truth_rows),
        "line_range_hit": any(any(_single_truth_hit(truth, row, "line") for truth in truths) for row in topk),
    }


def _report(lock: LockedGate3Sample, frozen: Mapping[str, Any], evaluation: Mapping[str, Any] | None, truth_sha: str | None) -> dict[str, Any]:
    prediction = frozen["prediction"]
    engineering_success = (
        prediction.get("status") == "COMPLETED"
        and prediction.get("final_status") == "VALIDATED"
        and prediction.get("deterministic_fallback_used") is False
        and prediction.get("provider_free_text_used") is False
        and all(group.get("status") == "VALIDATED" for group in prediction.get("groups", ()))
    )
    evaluation = dict(evaluation or {})
    gate3_success = engineering_success and all((
        evaluation.get("truth_function_in_candidate_pool") is True,
        evaluation.get("truth_function_in_group_top5") is True,
        evaluation.get("truth_function_in_global_top5") is True,
    ))
    result = {
        "schema_version": "phase2_tournament_v6_gate3_report_v1",
        "experiment_type": EXPERIMENT_TYPE,
        "scientific_experiment_result": False,
        "development_regression_sample": True,
        "sample_alias": lock.sample_alias,
        "locked_manifest_sha256": lock.public_manifest()["manifest_sha256"],
        "frozen_prediction_sha256": frozen["payload_sha256"],
        "evaluation_truth_sha256": truth_sha,
        "run_status": prediction.get("status"),
        "candidate_pool_count": prediction.get("candidate_pool_count", 0),
        "group_count": len(prediction.get("groups", ())),
        "group_sizes": [row.get("candidate_count", 0) for row in prediction.get("groups", ())],
        "validated_group_count": sum(row.get("status") == "VALIDATED" for row in prediction.get("groups", ())),
        "group_request_count": prediction["external_call_counts"]["group_deepseek"],
        "group_retry_count": prediction["retry_counts"]["group"],
        "finalist_count": prediction.get("finalist_count", 0),
        "final_status": prediction.get("final_status"),
        "global_top5_count": len(prediction.get("global_top5", ())),
        "engineering_chain_success": engineering_success,
        **{
            key: evaluation.get(key)
            for key in (
                "truth_function_in_candidate_pool", "truth_function_in_group_top5",
                "truth_function_in_global_top5", "truth_function_final_rank",
                "file_hit", "function_hit", "line_range_hit",
            )
        },
        "external_call_counts": dict(prediction["external_call_counts"]),
        "retry_counts": dict(prediction["retry_counts"]),
        "provider_free_text_used": False,
        "deterministic_fallback_used": False,
        "reason_codes": list(prediction.get("reason_codes", ())),
        "gate3_success": gate3_success,
        "sanitization_check": "PASSED",
    }
    validate_public(result)
    return result


def run_gate3(
    lock: LockedGate3Sample,
    output_dir: Path,
    chain_config: Mapping[str, Any],
    api_key: str,
    *,
    blind_runner: Callable[..., Mapping[str, Any]] = execute_blind_gate3,
    truth_loader: Callable[[LockedGate3Sample], tuple[Mapping[str, Any], str]] = _truth_loader,
    blind_runner_kwargs: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    prediction = blind_runner(lock, output_dir, chain_config, api_key, **dict(blind_runner_kwargs or {}))
    frozen = _freeze(lock, prediction)
    validate_public(frozen)
    frozen_path = output_dir / "frozen_prediction.json"
    _atomic(frozen_path, frozen)
    verified = validate_frozen(frozen_path, lock)
    frozen_bytes = frozen_path.read_bytes()
    evaluation = None
    truth_sha = None
    if prediction.get("status") == "COMPLETED" and prediction.get("final_status") == "VALIDATED":
        label, truth_sha = truth_loader(lock)
        evaluation = evaluate_frozen_gate3(verified, label)
    if frozen_path.read_bytes() != frozen_bytes:
        raise ValueError("TOURNAMENT_V6_GATE3_FROZEN_MODIFIED_DURING_EVALUATION")
    report = _report(lock, verified, evaluation, truth_sha)
    _atomic(output_dir / "gate3_real_report.json", report)
    _atomic(output_dir / "checkpoint.json", _checkpoint(lock, "COMPLETED" if prediction.get("status") == "COMPLETED" else "STOPPED", prediction, frozen["payload_sha256"]))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if not args.live:
        print("LIVE_MODE_REQUIRED")
        print("rpc_request_count=0")
        print("source_request_count=0")
        print("deepseek_request_count=0")
        return 2
    if args.output_dir.exists():
        print("gate3_status=STOPPED")
        print("reason_code=TOURNAMENT_V6_GATE3_OUTPUT_EXISTS")
        return 2
    validate_gate2_report()
    lock = lock_gate3_sample()
    public_manifest = lock.public_manifest()
    validate_public(public_manifest)
    config = load_chain_config("eth")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not isinstance(config.get("rpc_url"), str) or not config.get("rpc_url") or not isinstance(key, str) or not key:
        print("gate3_status=STOPPED")
        print("reason_code=TOURNAMENT_V6_GATE3_CONFIGURATION_MISSING")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic(args.output_dir / "locked_manifest.json", public_manifest)
    report = run_gate3(lock, args.output_dir, config, key)
    counts = report["external_call_counts"]
    print(f"gate3_status={report['run_status']}")
    print(f"final_status={report['final_status']}")
    print(f"rpc_request_count={counts['rpc']}")
    print(f"source_request_count={counts['source_network']}")
    print(f"deepseek_request_count={counts['deepseek_total']}")
    print(f"gate3_success={str(report['gate3_success']).lower()}")
    return 0 if report["gate3_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


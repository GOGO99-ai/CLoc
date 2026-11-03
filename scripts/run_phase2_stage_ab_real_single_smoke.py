"""Run one blind real Stage-A -> Stage-B engineering smoke without evaluation truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_localization15_topk_v4 import (
    MAX_PROMPT_BYTES_TOPK_V4,
    MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
    Phase2ClassifiedHTTPSOneShotTransport,
    _attach_formal_candidate_union,
)
from scripts.run_phase2_real_pilot import acquire_real_source_slices, build_real_phase2_evidence
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.phase2_candidate_shortlist_prompt import build_phase2_candidate_shortlist_prompt
from src.cloc_v2.phase2_localization_bridge import Phase2LocalizationPreparation, prepare_phase2_localization
from src.cloc_v2.phase2_localization_deepseek import (
    Phase2DeepSeekConfig,
    execute_phase2_candidate_shortlist_with_single_retry,
    execute_phase2_localization_with_single_retry,
)


EXPERIMENT_TYPE = "REAL_STAGE_A_TO_STAGE_B_ENGINEERING_SMOKE"
SAFE_ALIAS = "stage_ab_real_001"
HISTORICAL_DIR = ROOT / "data" / "prototypes" / "localization15_topk_v5_coarse_to_fine_recheck_20260901"
DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "phase2_stage_ab_real_single_smoke_20260901"
MAX_RPC_REQUESTS = 2
MAX_SOURCE_REQUESTS = 3
MAX_STAGE_A_REQUESTS = 2
MAX_STAGE_B_REQUESTS = 2
_TX_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
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
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
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


def _validate_public(value: Mapping[str, Any]) -> None:
    rendered = _canonical(value)
    if _RAW_ADDRESS.search(rendered) or _RAW_HASH.search(rendered):
        raise ValueError("STAGE_AB_REPORT_SENSITIVE")
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                if str(key).strip().lower().replace("-", "_") in _FORBIDDEN_KEYS:
                    raise ValueError("STAGE_AB_REPORT_SENSITIVE")
                stack.append(item)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


@dataclass(frozen=True)
class LockedStageABSample:
    sample_alias: str
    blockchain_chain_id: int
    _tx_hash: str = field(repr=False)
    historical_manifest_sha256: str = ""
    historical_predictions_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            self.sample_alias != SAFE_ALIAS
            or self.blockchain_chain_id != 1
            or not isinstance(self._tx_hash, str)
            or _TX_HASH.fullmatch(self._tx_hash.lower()) is None
            or len(self.historical_manifest_sha256) != 64
            or len(self.historical_predictions_sha256) != 64
        ):
            raise ValueError("STAGE_AB_LOCK_INVALID")

    @property
    def tx_hash(self) -> str:
        return self._tx_hash

    def public_manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": "phase2_stage_ab_real_single_manifest_v1",
            "experiment_type": EXPERIMENT_TYPE,
            "scientific_experiment_result": False,
            "sample_alias": self.sample_alias,
            "blockchain_chain_id": self.blockchain_chain_id,
            "selection_rule": "FIRST_NEUTRAL_ALIAS_WITH_HISTORICAL_STAGE_A_ATTEMPT",
            "historical_manifest_sha256": self.historical_manifest_sha256,
            "historical_predictions_sha256": self.historical_predictions_sha256,
        }
        return {**core, "manifest_sha256": _sha(core)}


def select_locked_sample(
    manifest: Mapping[str, Any],
    frozen: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> LockedStageABSample:
    """Select only from neutral aliases and frozen Stage-A attempt state."""
    if checkpoint.get("status") != "COMPLETED":
        raise ValueError("STAGE_AB_HISTORICAL_CHECKPOINT_INVALID")
    manifest_sha = manifest.get("manifest_sha256")
    frozen_sha = frozen.get("payload_sha256")
    manifest_core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    frozen_core = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    if (
        not isinstance(manifest_sha, str)
        or manifest_sha != _sha(manifest_core)
        or not isinstance(frozen_sha, str)
        or frozen_sha != _sha(frozen_core)
        or frozen.get("locked_manifest_sha256") != manifest_sha
        or checkpoint.get("manifest_sha256") != manifest_sha
        or checkpoint.get("frozen_predictions_sha256") != frozen_sha
    ):
        raise ValueError("STAGE_AB_HISTORICAL_SHA_INVALID")
    attempted = sorted(
        row["sample_alias"]
        for row in frozen.get("predictions", ())
        if isinstance(row, Mapping)
        and isinstance(row.get("sample_alias"), str)
        and isinstance(row.get("stage_a_request_count"), int)
        and not isinstance(row.get("stage_a_request_count"), bool)
        and row["stage_a_request_count"] > 0
    )
    if not attempted:
        raise ValueError("STAGE_AB_NO_HISTORICAL_STAGE_A_SAMPLE")
    by_alias = {
        row.get("sample_alias"): row
        for row in manifest.get("samples", ())
        if isinstance(row, Mapping) and isinstance(row.get("sample_alias"), str)
    }
    selected = by_alias.get(attempted[0])
    if not isinstance(selected, Mapping) or selected.get("blockchain_chain_id") != 1:
        raise ValueError("STAGE_AB_SELECTED_SAMPLE_INVALID")
    return LockedStageABSample(
        SAFE_ALIAS,
        1,
        str(selected.get("tx_hash", "")).lower(),
        manifest_sha,
        frozen_sha,
    )


def _prepare(bundle: Any, slices: Mapping[str, Any], shortlist_ids: Sequence[str] | None = None) -> Phase2LocalizationPreparation:
    return prepare_phase2_localization(
        bundle.strict,
        bundle.causal,
        bundle.graph,
        slices,
        max_pre_llm_candidates=MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
        max_prompt_candidates=20,
        max_prompt_bytes=MAX_PROMPT_BYTES_TOPK_V4,
        prompt_contract="topk_v4",
        shortlist_candidate_ids=shortlist_ids,
    )


def validate_stage_b_alignment(
    selected_ids: Sequence[str], preparation: Phase2LocalizationPreparation
) -> tuple[bool, tuple[str, ...]]:
    prompt_ids = tuple(
        row.get("candidate_alias")
        for row in preparation.pre_llm_candidates
        if isinstance(row, Mapping) and row.get("included_in_prompt") is True
    )
    expected = tuple(selected_ids)
    return prompt_ids == expected and len(set(prompt_ids)) == len(prompt_ids), prompt_ids


def _base_report(lock: LockedStageABSample) -> dict[str, Any]:
    return {
        "schema_version": "phase2_stage_ab_real_single_smoke_report_v1",
        "experiment_type": EXPERIMENT_TYPE,
        "scientific_experiment_result": False,
        "sample_alias": lock.sample_alias,
        "run_status": "STOPPED",
        "bridge_status": "NOT_BUILT",
        "raw_candidate_pool_count": 0,
        "candidate_pool_count": 0,
        "stage_a_input_count": 0,
        "stage_a_output_count": 0,
        "stage_a_status": "NOT_CALLED",
        "stage_a_request_count": 0,
        "stage_a_retry_count": 0,
        "stage_a_fallback_used": False,
        "stage_a_prompt_bytes": 0,
        "stage_a_output_unique": False,
        "stage_a_output_whitelisted": False,
        "stage_b_input_count": 0,
        "stage_b_input_matches_stage_a_output": False,
        "stage_b_topk_count": 0,
        "stage_b_status": "NOT_CALLED",
        "stage_b_request_count": 0,
        "stage_b_retry_count": 0,
        "stage_b_prompt_bytes": 0,
        "source_resolution_mode": "UNAVAILABLE",
        "source_selection_mode": "UNAVAILABLE",
        "exact_function_slice_count": 0,
        "source_window_count": 0,
        "provider_free_text_used": False,
        "two_stage_success": False,
        "reason_codes": [],
        "external_call_counts": {"rpc": 0, "source_network": 0, "stage_a_deepseek": 0, "stage_b_deepseek": 0, "deepseek_total": 0},
        "request_limits": {"rpc": MAX_RPC_REQUESTS, "source_network": MAX_SOURCE_REQUESTS, "stage_a_deepseek": MAX_STAGE_A_REQUESTS, "stage_b_deepseek": MAX_STAGE_B_REQUESTS, "deepseek_total": MAX_STAGE_A_REQUESTS + MAX_STAGE_B_REQUESTS},
        "retry_policy": "NETWORK_ERROR_TIMEOUT_HTTP_429_HTTP_5XX_ONCE_ONLY",
    }


def _checkpoint(lock: LockedStageABSample, status: str, report: Mapping[str, Any]) -> dict[str, Any]:
    counts = report["external_call_counts"]
    return {
        "schema_version": "phase2_stage_ab_real_single_checkpoint_v1",
        "manifest_sha256": lock.public_manifest()["manifest_sha256"],
        "status": status,
        "sample_alias": lock.sample_alias,
        "rpc_request_count": counts["rpc"],
        "source_request_count": counts["source_network"],
        "stage_a_request_count": counts["stage_a_deepseek"],
        "stage_b_request_count": counts["stage_b_deepseek"],
        "reason_codes": list(report["reason_codes"]),
    }


def execute_live_smoke(
    lock: LockedStageABSample,
    output_dir: Path,
    chain_config: Mapping[str, Any],
    api_key: str,
    *,
    rpc_factory: Callable[..., Any] = JsonRpcClient,
    evidence_builder: Callable[[Any], Any] = build_real_phase2_evidence,
    source_acquirer: Callable[[Any, Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any]]] = acquire_real_source_slices,
    union_builder: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] = _attach_formal_candidate_union,
    transport_factory: Callable[[], Any] = Phase2ClassifiedHTTPSOneShotTransport,
) -> Mapping[str, Any]:
    report = _base_report(lock)
    report_path = output_dir / "stage_ab_real_single_smoke_report.json"
    checkpoint_path = output_dir / "checkpoint.json"
    rpc_url = chain_config.get("rpc_url") if isinstance(chain_config, Mapping) else None
    if not isinstance(rpc_url, str) or not rpc_url or not isinstance(api_key, str) or not api_key:
        report["reason_codes"] = ["STAGE_AB_CONFIGURATION_MISSING"]
        _validate_public(report); _atomic(report_path, report); _atomic(checkpoint_path, _checkpoint(lock, "STOPPED", report))
        return report
    rpc = rpc_factory(rpc_url, max_requests=MAX_RPC_REQUESTS)
    _atomic(checkpoint_path, _checkpoint(lock, "RUNNING", report))
    try:
        chain_id = rpc.call("eth_chainId", [])
        report["external_call_counts"]["rpc"] = rpc.request_count
        if str(chain_id).lower() not in {"1", "0x1"}:
            report["reason_codes"] = ["CHAIN_ID_MISMATCH"]
            raise RuntimeError("SAFE_STOP")
        raw = rpc.call("debug_traceTransaction", [lock.tx_hash, {"tracer": "callTracer"}])
        report["external_call_counts"]["rpc"] = rpc.request_count
        bundle = evidence_builder(raw)
        slices, source_summary = source_acquirer(bundle, chain_config)
        source_requests = int(source_summary.get("source_request_count", 0))
        report["external_call_counts"]["source_network"] = source_requests
        if source_requests > MAX_SOURCE_REQUESTS:
            report["reason_codes"] = ["SOURCE_REQUEST_LIMIT_EXCEEDED"]
            raise RuntimeError("SAFE_STOP")
        slices = union_builder(bundle, slices)
        deterministic = _prepare(bundle, slices)
        audit = deterministic.candidate_selection_audit
        report.update(
            bridge_status=deterministic.preparation_status,
            raw_candidate_pool_count=int(audit.get("candidate_count_before_truncation", deterministic.pre_llm_candidate_count)),
            candidate_pool_count=deterministic.pre_llm_candidate_count,
            stage_a_input_count=deterministic.pre_llm_candidate_count,
            source_resolution_mode=deterministic.source_resolution_mode,
            source_selection_mode=deterministic.source_selection_mode,
            exact_function_slice_count=int(source_summary.get("resolved_slice_count", 0)),
            source_window_count=int(source_summary.get("window_count", 0)),
        )
        if deterministic.preparation_status != "READY_FOR_LLM":
            report["reason_codes"] = sorted({item.code for item in deterministic.issues} | {"STAGE_A_NOT_ELIGIBLE"})
            raise RuntimeError("SAFE_STOP")
        built = build_phase2_candidate_shortlist_prompt(deterministic.pre_llm_candidates)
        report["stage_a_prompt_bytes"] = built.prompt.prompt_bytes if built.prompt else 0
        if built.prompt is None:
            report["reason_codes"] = sorted(set(built.reason_codes) | {"STAGE_A_PROMPT_INVALID"})
            raise RuntimeError("SAFE_STOP")
        stage_a = execute_phase2_candidate_shortlist_with_single_retry(
            built.prompt, Phase2DeepSeekConfig(), transport_factory, api_key
        )
        report["stage_a_status"] = stage_a.response_status
        report["stage_a_request_count"] = stage_a.transport_audit.request_count
        report["stage_a_retry_count"] = stage_a.transport_audit.retry_count
        report["external_call_counts"]["stage_a_deepseek"] = stage_a.transport_audit.request_count
        report["external_call_counts"]["deepseek_total"] = stage_a.transport_audit.request_count
        if stage_a.response_status != "VALIDATED":
            report["reason_codes"] = sorted(set(stage_a.reason_codes) | {"STAGE_A_FAILED_NO_FALLBACK"})
            raise RuntimeError("SAFE_STOP")
        selected = tuple(stage_a.selected_candidate_ids)
        whitelist = set(built.prompt.candidate_ids)
        report["stage_a_output_count"] = len(selected)
        report["stage_a_output_unique"] = len(selected) == len(set(selected))
        report["stage_a_output_whitelisted"] = set(selected).issubset(whitelist)
        if (
            len(selected) != min(20, deterministic.pre_llm_candidate_count)
            or not report["stage_a_output_unique"]
            or not report["stage_a_output_whitelisted"]
        ):
            report["reason_codes"] = ["STAGE_A_SELECTION_CONTRACT_INVALID"]
            raise RuntimeError("SAFE_STOP")
        stage_b_preparation = _prepare(bundle, slices, selected)
        aligned, stage_b_ids = validate_stage_b_alignment(selected, stage_b_preparation)
        report["stage_b_input_count"] = len(stage_b_ids)
        report["stage_b_input_matches_stage_a_output"] = aligned
        report["stage_b_prompt_bytes"] = stage_b_preparation.prompt_bytes
        report["bridge_status"] = stage_b_preparation.preparation_status
        if stage_b_preparation.preparation_status != "READY_FOR_LLM" or not aligned:
            report["reason_codes"] = ["STAGE_B_INPUT_ALIGNMENT_INVALID"]
            raise RuntimeError("SAFE_STOP")
        stage_b = execute_phase2_localization_with_single_retry(
            stage_b_preparation, Phase2DeepSeekConfig(), transport_factory, api_key
        )
        result = stage_b.result
        report["stage_b_status"] = result.response_status
        report["stage_b_request_count"] = stage_b.transport_audit.request_count
        report["stage_b_retry_count"] = stage_b.transport_audit.retry_count
        report["external_call_counts"]["stage_b_deepseek"] = stage_b.transport_audit.request_count
        report["external_call_counts"]["deepseek_total"] += stage_b.transport_audit.request_count
        findings = result.validated_response.get("findings", ()) if result.validated_response else ()
        report["stage_b_topk_count"] = len(findings)
        if result.response_status != "VALIDATED" or len(findings) != min(5, len(stage_b_ids)):
            report["reason_codes"] = sorted(set(result.reason_codes) | {"STAGE_B_FAILED"})
            raise RuntimeError("SAFE_STOP")
        report["two_stage_success"] = True
        report["run_status"] = "COMPLETED"
        report["reason_codes"] = ["STAGE_A_TO_STAGE_B_VALIDATED"]
    except RuntimeError as exc:
        if str(exc) != "SAFE_STOP":
            report["reason_codes"] = ["STAGE_AB_INTERNAL_ERROR"]
    except Exception:
        report["reason_codes"] = ["STAGE_AB_EXTERNAL_OR_PROCESSING_ERROR"]
    if report["external_call_counts"]["rpc"] == 0 and hasattr(rpc, "request_count"):
        report["external_call_counts"]["rpc"] = int(rpc.request_count)
    counts = report["external_call_counts"]
    if (
        counts["rpc"] > MAX_RPC_REQUESTS
        or counts["source_network"] > MAX_SOURCE_REQUESTS
        or counts["stage_a_deepseek"] > MAX_STAGE_A_REQUESTS
        or counts["stage_b_deepseek"] > MAX_STAGE_B_REQUESTS
        or counts["deepseek_total"] > MAX_STAGE_A_REQUESTS + MAX_STAGE_B_REQUESTS
    ):
        report.update(run_status="STOPPED", two_stage_success=False, reason_codes=["STAGE_AB_REQUEST_BUDGET_EXCEEDED"])
    _validate_public(report)
    _atomic(report_path, report)
    _atomic(checkpoint_path, _checkpoint(lock, "COMPLETED" if report["two_stage_success"] else "STOPPED", report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.live:
        print("live_required=true")
        print("rpc_request_count=0")
        print("source_request_count=0")
        print("deepseek_request_count=0")
        return 2
    if args.output_dir.exists():
        print("smoke_status=STOPPED")
        print("reason_code=STAGE_AB_OUTPUT_EXISTS")
        return 2
    manifest = _load(HISTORICAL_DIR / "locked_inference_manifest.json")
    frozen = _load(HISTORICAL_DIR / "frozen_predictions.json")
    checkpoint = _load(HISTORICAL_DIR / "checkpoint.json")
    lock = select_locked_sample(manifest, frozen, checkpoint)
    public_manifest = lock.public_manifest()
    _validate_public(public_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic(args.output_dir / "locked_manifest.json", public_manifest)
    config = load_chain_config("eth")
    key = os.getenv("DEEPSEEK_API_KEY")
    report = execute_live_smoke(lock, args.output_dir, config, key or "")
    print(f"smoke_status={report['run_status']}")
    print(f"stage_a_status={report['stage_a_status']}")
    print(f"stage_b_status={report['stage_b_status']}")
    print(f"rpc_request_count={report['external_call_counts']['rpc']}")
    print(f"source_request_count={report['external_call_counts']['source_network']}")
    print(f"deepseek_request_count={report['external_call_counts']['deepseek_total']}")
    return 0 if report["two_stage_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

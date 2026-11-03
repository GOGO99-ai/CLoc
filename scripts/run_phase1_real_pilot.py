"""Controlled phase-1 pilot for three locked Ethereum transactions.

Importing this module performs no file, configuration, key, RPC, LLM, or
network access.  External access is possible only through ``main --live`` or
through caller-injected clients passed to :func:`run_phase1_real_pilot`.
"""

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
from types import MappingProxyType
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase1_deepseek_smoke import DeepSeekHTTPSOneShotTransport
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.asset_continuity import build_asset_continuity
from src.cloc_v2.calltracer_adapter import adapt_calltracer_call_tree
from src.cloc_v2.formal_normal_baseline_loader import (
    DEFAULT_FORMAL_NORMAL_BASELINE_DIR,
    LoadedFormalNormalBaseline,
    load_formal_normal_baseline,
)
from src.cloc_v2.normal_features import extract_normal_features
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.phase1_normality_bridge import (
    Phase1NormalityPreparation,
    prepare_phase1_normality_review,
)
from src.cloc_v2.phase1_normality_deepseek import (
    Phase1DeepSeekConfig,
    Phase1DeepSeekResult,
    execute_phase1_deepseek_review,
)
from src.cloc_v2.result_detection import detect_result_candidates
from src.cloc_v2.runtime_semantics import build_runtime_annotations
from src.cloc_v2.strict_result_chains import build_strict_result_chains
from src.cloc_v2.transaction_graph import build_transaction_graph


SCHEMA_VERSION = "phase1_real_pilot_v1"
MANIFEST_SCHEMA_VERSION = "phase1_real_pilot_locked_manifest_v1"
SELECTION_RULE = "LEXICOGRAPHIC_FIRST_3_OF_79_ETHEREUM_FAULT_TRANSACTIONS"
BLOCKCHAIN_CHAIN_ID = 1
MAX_RPC_REQUESTS = 4
MAX_DEEPSEEK_REQUESTS = 3
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "phase1_real_pilot"

_LOCKED_ROWS = (
    ("pilot_eth_001", "0x0163856aaa678676b22e072186c1bdc55ed44a5a0e2cf44500362800663ddce9"),
    ("pilot_eth_002", "0x047d2f5ca117de92a65de25d04e00db9bc0f1a899ba61140550afc6618d282ac"),
    ("pilot_eth_003", "0x07e5568b71f1f4d9234d5925473c598865c5ee1800348c1c4b044c6f0223f308"),
)
_TX_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_TX_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_MACHINE_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "tx_hash", "transaction_hash", "address", "effective_address",
        "runtime_address", "input", "calldata", "output", "error",
        "revertreason", "reasoning_content", "api_key", "rpc_url",
        "system_message", "user_message", "response", "raw_trace",
        "provider_response", "prompt", "record_id",
    }
)


@dataclass(frozen=True)
class LockedPilotSample:
    sample_alias: str
    blockchain_chain_id: int
    tx_hash: str = field(repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"pilot_eth_00[1-3]", self.sample_alias):
            raise ValueError("PILOT_SAMPLE_ALIAS_INVALID")
        if self.blockchain_chain_id != BLOCKCHAIN_CHAIN_ID:
            raise ValueError("PILOT_SAMPLE_CHAIN_ID_INVALID")
        if not isinstance(self.tx_hash, str) or _TX_HASH.fullmatch(self.tx_hash) is None:
            raise ValueError("PILOT_TRANSACTION_HASH_INVALID")

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "sample_alias": self.sample_alias,
            "blockchain_chain_id": self.blockchain_chain_id,
            "tx_hash": self.tx_hash,
        }


@dataclass(frozen=True)
class LockedPilotManifest:
    samples: tuple[LockedPilotSample, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if tuple(self.samples) != tuple(sorted(self.samples, key=lambda item: item.sample_alias)):
            raise ValueError("PILOT_MANIFEST_ORDER_INVALID")
        if len(self.samples) != 3 or len({item.sample_alias for item in self.samples}) != 3:
            raise ValueError("PILOT_MANIFEST_SAMPLE_SET_INVALID")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("PILOT_MANIFEST_SHA256_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "selection_rule": SELECTION_RULE,
            "sample_count": len(self.samples),
            "samples": [item.to_manifest_dict() for item in self.samples],
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class PreparedPilotEvidence:
    sample_alias: str
    stage_counts: Mapping[str, int]
    reason_codes: tuple[str, ...]
    preparation: Phase1NormalityPreparation = field(repr=False)

    def __post_init__(self) -> None:
        counts = dict(self.stage_counts)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
            raise ValueError("PILOT_STAGE_COUNTS_INVALID")
        object.__setattr__(self, "stage_counts", MappingProxyType(dict(sorted(counts.items()))))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        if not isinstance(self.preparation, Phase1NormalityPreparation):
            raise ValueError("PILOT_PREPARATION_INVALID")


@dataclass(frozen=True)
class PilotRunArtifacts:
    manifest: LockedPilotManifest = field(repr=False)
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        report = json.loads(_canonical_json(self.report))
        validate_pilot_report(report)
        object.__setattr__(self, "report", MappingProxyType(report))


def build_locked_manifest() -> LockedPilotManifest:
    """Return the immutable, code-locked three-sample manifest."""
    samples = tuple(LockedPilotSample(alias, BLOCKCHAIN_CHAIN_ID, tx_hash) for alias, tx_hash in _LOCKED_ROWS)
    digest_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selection_rule": SELECTION_RULE,
        "samples": [item.to_manifest_dict() for item in samples],
    }
    return LockedPilotManifest(samples, _sha256(_canonical_json(digest_payload)))


def validate_formal_baseline(loaded: LoadedFormalNormalBaseline) -> dict[str, Any]:
    """Validate and return only the loader's already-sanitized summary."""
    if not isinstance(loaded, LoadedFormalNormalBaseline):
        raise ValueError("FORMAL_NORMAL_BASELINE_INVALID")
    summary = loaded.summary
    if (
        summary.formal_min_group_size != 3
        or not summary.manifest_consistency_passed
        or not summary.reconstructed_baseline_match
        or not summary.ethereum_only
        or loaded.baseline.min_group_size != 3
        or any(group.main_anchor.blockchain_chain_id != 1 for group in loaded.baseline.groups)
    ):
        raise ValueError("FORMAL_NORMAL_BASELINE_SAFETY_CHECK_FAILED")
    return summary.to_dict()


def prepare_real_pilot_sample(
    sample: LockedPilotSample,
    raw_calltracer_frame: Any,
    loaded_baseline: LoadedFormalNormalBaseline,
) -> PreparedPilotEvidence:
    """Run the existing formal evidence pipeline entirely in memory."""
    validate_formal_baseline(loaded_baseline)
    call_tree = adapt_calltracer_call_tree(raw_calltracer_frame)
    graph = build_transaction_graph(call_tree)
    annotations = build_runtime_annotations(graph)
    ledger = build_observed_asset_ledger(graph, annotations)
    continuity = build_asset_continuity(graph, annotations, ledger)
    detected = detect_result_candidates(graph, annotations, ledger, continuity.to_dict())
    strict = build_strict_result_chains(
        graph,
        continuity.to_dict(),
        detected.to_dict(),
        blockchain_chain_id=BLOCKCHAIN_CHAIN_ID,
        offline_function_candidates=(),
        offline_effective_address_map={},
    )
    extracted = extract_normal_features(
        graph,
        strict,
        ledger,
        transaction_record_id=sample.sample_alias,
        offline_function_candidates=(),
        offline_effective_address_map={},
    )
    preparation = prepare_phase1_normality_review(
        strict,
        extracted,
        loaded_baseline.baseline,
        loaded_baseline.normal_record_transaction_ids,
    )
    complete_facts = sum(fact.status == "COMPLETE" for fact in ledger.transfer_facts)
    stage_counts = {
        "call_node_count": len(graph.nodes),
        "recognized_transfer_node_count": sum(
            "TOKEN_TRANSFER" in annotation.action_types for annotation in annotations.node_annotations
        ),
        "asset_fact_count": len(ledger.transfer_facts),
        "complete_asset_fact_count": complete_facts,
        "incomplete_asset_fact_count": len(ledger.transfer_facts) - complete_facts,
        "result_candidate_count": len(detected.result_candidates),
        "strict_chain_count": len(strict.chains),
        "strict_ready_count": sum(chain.status == "READY" for chain in strict.chains),
        "strict_uncertain_count": sum(chain.status == "UNCERTAIN" for chain in strict.chains),
        "clean_record_count": extracted.clean_record_count,
        "uncertain_record_count": extracted.uncertain_record_count,
        "rejected_record_count": extracted.rejected_chain_count,
    }
    reasons = [item.code for item in preparation.issues]
    reasons.extend(preparation.eligibility.reason_codes)
    reasons.extend(reason for status in extracted.chain_statuses for reason in status.reasons)
    return PreparedPilotEvidence(sample.sample_alias, stage_counts, _reason_codes(reasons), preparation)


def route_prepared_pilot_sample(
    prepared: PreparedPilotEvidence,
    deepseek_runner: Callable[[Phase1NormalityPreparation], Phase1DeepSeekResult],
) -> tuple[dict[str, Any], int]:
    """Apply the hard gate, then make at most one injected DeepSeek call."""
    preparation = prepared.preparation
    base = {
        "sample_alias": prepared.sample_alias,
        "stage_counts": dict(prepared.stage_counts),
        "preparation_status": preparation.preparation_status,
        "eligibility": preparation.eligibility.eligibility,
    }
    if preparation.preparation_status != "READY_FOR_LLM":
        return (
            {
                **base,
                "llm_called": False,
                "deepseek_response_status": "NOT_REQUESTED",
                "deepseek_usage": None,
                "final_route": "PROCEED_PHASE2",
                "reason_codes": list(_reason_codes((*prepared.reason_codes, *preparation.eligibility.reason_codes))),
            },
            0,
        )
    try:
        result = deepseek_runner(preparation)
    except Exception:
        return (
            {
                **base,
                "llm_called": True,
                "deepseek_response_status": "CLIENT_ERROR",
                "deepseek_usage": None,
                "final_route": "PROCEED_PHASE2",
                "reason_codes": ["DEEPSEEK_CLIENT_ERROR"],
            },
            1,
        )
    if not isinstance(result, Phase1DeepSeekResult):
        return (
            {
                **base,
                "llm_called": True,
                "deepseek_response_status": "INVALID_CLIENT_RESULT",
                "deepseek_usage": None,
                "final_route": "PROCEED_PHASE2",
                "reason_codes": ["DEEPSEEK_CLIENT_RESULT_INVALID"],
            },
            1,
        )
    route = result.routing_decision.route if result.routing_decision is not None else "PROCEED_PHASE2"
    route_reasons = result.routing_decision.reason_codes if result.routing_decision is not None else result.reason_codes
    return (
        {
            **base,
            "llm_called": True,
            "deepseek_response_status": result.response_status,
            "deepseek_usage": result.usage.to_dict() if result.usage is not None else None,
            "final_route": route,
            "reason_codes": list(_reason_codes(route_reasons)),
        },
        1,
    )


def run_phase1_real_pilot(
    *,
    live: bool = False,
    rpc_client: Any | None = None,
    deepseek_runner: Callable[[Phase1NormalityPreparation], Phase1DeepSeekResult] | None = None,
    loaded_baseline: LoadedFormalNormalBaseline | None = None,
    sample_preparer: Callable[[LockedPilotSample, Any, LoadedFormalNormalBaseline], PreparedPilotEvidence] = prepare_real_pilot_sample,
) -> PilotRunArtifacts:
    """Orchestrate the locked pilot with explicit clients and hard request caps."""
    manifest = build_locked_manifest()
    if not live:
        report = build_pilot_report(
            manifest,
            baseline_summary=None,
            sample_results=tuple(_not_run_result(item.sample_alias) for item in manifest.samples),
            rpc_request_count=0,
            deepseek_request_count=0,
            run_status="LIVE_MODE_REQUIRED",
        )
        return PilotRunArtifacts(manifest, report)
    if rpc_client is None or not callable(getattr(rpc_client, "call", None)):
        raise ValueError("PILOT_RPC_CLIENT_REQUIRED")
    if deepseek_runner is None or not callable(deepseek_runner):
        raise ValueError("PILOT_DEEPSEEK_RUNNER_REQUIRED")
    if loaded_baseline is None:
        raise ValueError("PILOT_FORMAL_BASELINE_REQUIRED")
    baseline_summary = validate_formal_baseline(loaded_baseline)

    rpc_count = 0
    deepseek_count = 0
    sample_results: list[dict[str, Any]] = []
    try:
        rpc_count += 1
        chain_value = rpc_client.call("eth_chainId", [])
    except Exception as exc:
        code = _exception_code(exc, "RPC_CHAIN_ID_ERROR")
        sample_results.extend(_safe_failure(item.sample_alias, code) for item in manifest.samples)
        return PilotRunArtifacts(
            manifest,
            build_pilot_report(manifest, baseline_summary, tuple(sample_results), rpc_count, 0, "SAFE_FALLBACK"),
        )
    if _parse_chain_id(chain_value) != BLOCKCHAIN_CHAIN_ID:
        sample_results.extend(_safe_failure(item.sample_alias, "CHAIN_ID_MISMATCH") for item in manifest.samples)
        return PilotRunArtifacts(
            manifest,
            build_pilot_report(manifest, baseline_summary, tuple(sample_results), rpc_count, 0, "SAFE_FALLBACK"),
        )

    for sample in manifest.samples:
        if rpc_count >= MAX_RPC_REQUESTS:
            sample_results.append(_safe_failure(sample.sample_alias, "RPC_REQUEST_LIMIT_REACHED"))
            continue
        try:
            rpc_count += 1
            raw_frame = rpc_client.call(
                "debug_traceTransaction",
                [sample.tx_hash, {"tracer": "callTracer"}],
            )
        except Exception as exc:
            sample_results.append(_safe_failure(sample.sample_alias, _exception_code(exc, "RPC_TRACE_ERROR")))
            continue
        try:
            prepared = sample_preparer(sample, raw_frame, loaded_baseline)
        except Exception:
            sample_results.append(_safe_failure(sample.sample_alias, "SAMPLE_PROCESSING_ERROR"))
            continue
        if prepared.sample_alias != sample.sample_alias:
            sample_results.append(_safe_failure(sample.sample_alias, "SAMPLE_ALIAS_MISMATCH"))
            continue
        if prepared.preparation.preparation_status == "READY_FOR_LLM":
            if deepseek_count >= MAX_DEEPSEEK_REQUESTS:
                sample_results.append(_safe_failure(sample.sample_alias, "DEEPSEEK_REQUEST_LIMIT_REACHED"))
                continue
            result, used = route_prepared_pilot_sample(prepared, deepseek_runner)
            deepseek_count += used
            sample_results.append(result)
        else:
            result, used = route_prepared_pilot_sample(prepared, deepseek_runner)
            deepseek_count += used
            sample_results.append(result)

    report = build_pilot_report(
        manifest,
        baseline_summary,
        tuple(sample_results),
        rpc_count,
        deepseek_count,
        "COMPLETED",
    )
    return PilotRunArtifacts(manifest, report)


def build_pilot_report(
    manifest: LockedPilotManifest,
    baseline_summary: Mapping[str, Any] | None,
    sample_results: tuple[Mapping[str, Any], ...],
    rpc_request_count: int,
    deepseek_request_count: int,
    run_status: str,
) -> dict[str, Any]:
    if rpc_request_count < 0 or rpc_request_count > MAX_RPC_REQUESTS:
        raise ValueError("RPC_REQUEST_COUNT_INVALID")
    if deepseek_request_count < 0 or deepseek_request_count > MAX_DEEPSEEK_REQUESTS:
        raise ValueError("DEEPSEEK_REQUEST_COUNT_INVALID")
    ordered = sorted((dict(item) for item in sample_results), key=lambda item: item["sample_alias"])
    if [item["sample_alias"] for item in ordered] != [item.sample_alias for item in manifest.samples]:
        raise ValueError("PILOT_REPORT_SAMPLE_SET_INVALID")
    report = {
        "schema_version": SCHEMA_VERSION,
        "pilot_kind": "CONTROLLED_REAL_ETHEREUM_PHASE1_PILOT",
        "selection_rule": SELECTION_RULE,
        "locked_manifest_sha256": manifest.manifest_sha256,
        "run_status": run_status,
        "formal_normal_baseline_summary": dict(baseline_summary) if baseline_summary is not None else None,
        "sample_results": ordered,
        "external_call_counts": {
            "rpc": rpc_request_count,
            "deepseek": deepseek_request_count,
            "network": rpc_request_count + deepseek_request_count,
        },
        "request_limits": {"rpc": MAX_RPC_REQUESTS, "deepseek": MAX_DEEPSEEK_REQUESTS},
    }
    validate_pilot_report(report)
    return report


def serialize_locked_manifest(manifest: LockedPilotManifest) -> str:
    if not isinstance(manifest, LockedPilotManifest):
        raise ValueError("PILOT_MANIFEST_INVALID")
    return json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_phase1_real_pilot_report(report: Mapping[str, Any]) -> str:
    validate_pilot_report(report)
    return json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def validate_pilot_report(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise ValueError("PILOT_REPORT_INVALID")

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in _FORBIDDEN_REPORT_KEYS:
                    raise ValueError("PILOT_REPORT_FORBIDDEN_FIELD")
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            if _RAW_TX_HASH.search(value) or _RAW_ADDRESS.search(value):
                raise ValueError("PILOT_REPORT_RAW_CHAIN_IDENTIFIER_FORBIDDEN")
            lowered = value.lower()
            if any(token in lowered for token in ("reasoning_content", "calldata", "revertreason", "api key", "rpc_url")):
                raise ValueError("PILOT_REPORT_SENSITIVE_CONTENT_FORBIDDEN")

    visit(report)


def write_phase1_real_pilot_artifacts(
    output_dir: str | Path,
    artifacts: PilotRunArtifacts,
) -> tuple[Path, Path]:
    """Atomically write the only two allowed pilot artifacts."""
    if not isinstance(artifacts, PilotRunArtifacts):
        raise ValueError("PILOT_ARTIFACTS_INVALID")
    root = Path(output_dir)
    manifest_path = root / "locked_manifest.json"
    report_path = root / "phase1_real_pilot_report.json"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write(manifest_path, serialize_locked_manifest(artifacts.manifest))
    _atomic_write(report_path, serialize_phase1_real_pilot_report(artifacts.report))
    return manifest_path, report_path


def _atomic_write(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _not_run_result(sample_alias: str) -> dict[str, Any]:
    return {
        "sample_alias": sample_alias,
        "stage_counts": {},
        "preparation_status": "NOT_RUN",
        "eligibility": "NOT_EVALUATED",
        "llm_called": False,
        "deepseek_response_status": "NOT_REQUESTED",
        "deepseek_usage": None,
        "final_route": "PROCEED_PHASE2",
        "reason_codes": ["LIVE_MODE_REQUIRED"],
    }


def _safe_failure(sample_alias: str, code: str) -> dict[str, Any]:
    return {
        "sample_alias": sample_alias,
        "stage_counts": {},
        "preparation_status": "PROCESSING_FAILED",
        "eligibility": "INELIGIBLE",
        "llm_called": False,
        "deepseek_response_status": "NOT_REQUESTED",
        "deepseek_usage": None,
        "final_route": "PROCEED_PHASE2",
        "reason_codes": [_fixed_reason(code)],
    }


def _reason_codes(values: Any) -> tuple[str, ...]:
    return tuple(sorted({_fixed_reason(value) for value in values}))


def _fixed_reason(value: Any) -> str:
    return value if isinstance(value, str) and _MACHINE_CODE.fullmatch(value) else "NON_MACHINE_READABLE_REASON"


def _exception_code(exc: Exception, fallback: str) -> str:
    return _fixed_reason(exc.args[0]) if len(exc.args) == 1 else fallback


def _parse_chain_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]+", value):
        return int(value, 16)
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="explicitly enable the bounded real pilot")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_live_chain_config() -> Mapping[str, Any]:
    """Import and read chain configuration only after explicit live consent."""
    from src.cloc_v2.config_loader import load_chain_config

    return load_chain_config("eth")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.live:
        print("live_required=true")
        print("rpc_request_count=0")
        print("deepseek_request_count=0")
        return 2

    try:
        config = _load_live_chain_config()
        rpc_url = config.get("rpc_url")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not isinstance(rpc_url, str) or not rpc_url or not isinstance(api_key, str) or not api_key:
            raise ValueError("LIVE_CONFIGURATION_MISSING")
        loaded = load_formal_normal_baseline(DEFAULT_FORMAL_NORMAL_BASELINE_DIR)
        client = JsonRpcClient(rpc_url, max_requests=MAX_RPC_REQUESTS)
        deepseek_config = Phase1DeepSeekConfig()

        def deepseek_runner(preparation: Phase1NormalityPreparation) -> Phase1DeepSeekResult:
            return execute_phase1_deepseek_review(
                preparation,
                deepseek_config,
                DeepSeekHTTPSOneShotTransport(),
                api_key,
            )

        artifacts = run_phase1_real_pilot(
            live=True,
            rpc_client=client,
            deepseek_runner=deepseek_runner,
            loaded_baseline=loaded,
        )
        manifest_path, report_path = write_phase1_real_pilot_artifacts(args.output_dir, artifacts)
    except Exception as exc:
        print(f"pilot_status={_exception_code(exc, 'PILOT_EXECUTION_ERROR')}")
        return 1
    print("pilot_status=COMPLETED")
    print(f"manifest_path={manifest_path}")
    print(f"report_path={report_path}")
    print(f"rpc_request_count={artifacts.report['external_call_counts']['rpc']}")
    print(f"deepseek_request_count={artifacts.report['external_call_counts']['deepseek']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKCHAIN_CHAIN_ID",
    "DEFAULT_OUTPUT_DIR",
    "LockedPilotManifest",
    "LockedPilotSample",
    "MAX_DEEPSEEK_REQUESTS",
    "MAX_RPC_REQUESTS",
    "PilotRunArtifacts",
    "PreparedPilotEvidence",
    "SELECTION_RULE",
    "build_locked_manifest",
    "build_pilot_report",
    "main",
    "prepare_real_pilot_sample",
    "route_prepared_pilot_sample",
    "run_phase1_real_pilot",
    "serialize_locked_manifest",
    "serialize_phase1_real_pilot_report",
    "validate_formal_baseline",
    "validate_pilot_report",
    "write_phase1_real_pilot_artifacts",
]

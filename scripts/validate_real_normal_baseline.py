"""Validate the formal normal baseline on a small live Ethereum sample.

The command is offline by default.  Only an explicit ``--live`` enables one
chain-id request and at most eight ``debug_traceTransaction/callTracer``
requests.  Raw RPC frames remain in memory and are never serialized.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import socket
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloc_v2.asset_continuity import build_asset_continuity  # noqa: E402
from src.cloc_v2.calltracer_adapter import adapt_calltracer_call_tree  # noqa: E402
from src.cloc_v2.normal_baseline import (  # noqa: E402
    build_normal_baseline,
    query_normal_baseline,
)
from src.cloc_v2.normal_distance import ChainFeatureRecord, MainAnchorKey  # noqa: E402
from src.cloc_v2.normal_features import extract_normal_features  # noqa: E402
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger  # noqa: E402
from src.cloc_v2.result_detection import detect_result_candidates  # noqa: E402
from src.cloc_v2.runtime_semantics import build_runtime_annotations  # noqa: E402
from src.cloc_v2.strict_result_chains import build_strict_result_chains  # noqa: E402
from src.cloc_v2.transaction_graph import build_transaction_graph  # noqa: E402


DEFAULT_TRACE_DIR = PROJECT_ROOT / "data" / "cache" / "traces"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "formal_normal_baseline_smoke"
OUTPUT_FILENAMES = (
    "normal_input_manifest.json",
    "normal_feature_audit.json",
    "baseline_sensitivity.json",
    "leave_one_out_queries.json",
    "validation_report.json",
)
MIN_GROUP_SIZES = (2, 3, 5)
MAX_RPC_REQUESTS = 9
SAMPLE_COUNT = 8
_TRACE_NAME = re.compile(r"^eth_(0x[0-9a-fA-F]{64})\.txt$")
_FORBIDDEN_OUTPUT_KEYS = {
    "rpc_url",
    "api_key",
    "key",
    "input",
    "output",
    "error",
    "revertreason",
    "calldata",
    "raw_trace",
    "trace_response",
    "fault",
    "cause",
    "location",
    "ground_truth",
    "paired_transaction",
}


class RpcValidationError(RuntimeError):
    """A sanitized RPC failure category with no provider response content."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class NormalSample:
    sample_id: str
    blockchain_chain_id: int
    tx_hash: str
    is_normal: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockchain_chain_id": self.blockchain_chain_id,
            "tx_hash": self.tx_hash,
            "is_normal": self.is_normal,
            "sample_id": self.sample_id,
        }


class JsonRpcClient:
    """One-endpoint JSON-RPC client with a hard, testable request ceiling."""

    def __init__(
        self,
        rpc_url: str,
        *,
        max_requests: int = MAX_RPC_REQUESTS,
        transport: Callable[[str, Mapping[str, Any], float], Any] | None = None,
    ) -> None:
        if not isinstance(rpc_url, str) or not rpc_url:
            raise RpcValidationError("RPC_NOT_CONFIGURED")
        self._rpc_url = rpc_url
        self._max_requests = max_requests
        self._transport = transport or _urllib_transport
        self.request_count = 0

    def call(self, method: str, params: list[Any]) -> Any:
        if method not in {"eth_chainId", "debug_traceTransaction"}:
            raise RpcValidationError("RPC_METHOD_NOT_ALLOWED")
        if self.request_count >= self._max_requests:
            raise RpcValidationError("RPC_REQUEST_LIMIT_EXCEEDED")
        self.request_count += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_count,
            "method": method,
            "params": params,
        }
        try:
            response = self._transport(self._rpc_url, payload, 55.0)
        except RpcValidationError:
            raise
        except TimeoutError as exc:
            raise RpcValidationError("RPC_TIMEOUT") from exc
        except Exception as exc:
            raise RpcValidationError("RPC_NETWORK_ERROR") from exc
        if not isinstance(response, Mapping):
            raise RpcValidationError("RPC_INVALID_RESPONSE")
        if response.get("error") is not None:
            raise RpcValidationError(_classify_rpc_error(response.get("error")))
        if "result" not in response:
            raise RpcValidationError("RPC_INVALID_RESPONSE")
        return response["result"]


def select_normal_samples(trace_dir: Path, *, limit: int = SAMPLE_COUNT) -> tuple[NormalSample, ...]:
    """Select by sanitized filename only; trace text and dataset paths are unread."""
    hashes: set[str] = set()
    for path in trace_dir.glob("eth_*.txt"):
        match = _TRACE_NAME.fullmatch(path.name)
        if match is not None:
            hashes.add(match.group(1).lower())
    selected = sorted(hashes)[:limit]
    if len(selected) < limit:
        raise ValueError(f"At least {limit} valid eth_ trace filenames are required")
    return tuple(
        NormalSample(
            sample_id=f"normal_sample_{index:03d}",
            blockchain_chain_id=1,
            tx_hash=tx_hash,
        )
        for index, tx_hash in enumerate(selected, start=1)
    )


def run_validation(
    *,
    live: bool = False,
    live_runner: Callable[..., Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Refuse all RPC/config work unless the caller explicitly enables live mode."""
    if not live:
        return {"status": "LIVE_MODE_REQUIRED", "rpc_request_count": 0}
    runner = live_runner or run_live_validation
    return runner(**kwargs)


def run_live_validation(
    *,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    rpc_client: JsonRpcClient | None = None,
) -> Mapping[str, Any]:
    """Execute exactly one bounded live pass and write only sanitized artifacts."""
    samples = select_normal_samples(Path(trace_dir))
    _assert_existing_sample_manifest(Path(output_dir), samples)
    client = rpc_client or _configured_ethereum_client()
    chain_id_result = client.call("eth_chainId", [])
    if _parse_chain_id(chain_id_result) != 1:
        raise RpcValidationError("RPC_CHAIN_ID_MISMATCH")

    audit_samples: list[dict[str, Any]] = []
    all_records: list[ChainFeatureRecord] = []
    successful_samples = 0
    for sample in samples:
        raw_frame = client.call(
            "debug_traceTransaction",
            [sample.tx_hash, {"tracer": "callTracer"}],
        )
        summary, records = process_calltracer_sample(sample, raw_frame)
        audit_samples.append(summary)
        all_records.extend(records)
        successful_samples += 1

    clean_records = tuple(sorted((item for item in all_records if not item.uncertainties), key=lambda item: item.record_id))
    sensitivity = build_baseline_sensitivity(clean_records)
    leave_one_out = build_leave_one_out_queries(clean_records)
    manifest = {
        "schema_version": "formal_normal_input_manifest_v1",
        "samples": [sample.to_dict() for sample in samples],
    }
    reason_counts = Counter(
        reason
        for sample in audit_samples
        for reason in sample["feature_reason_codes"]
    )
    feature_audit = {
        "schema_version": "formal_normal_feature_audit_v1",
        "samples": sorted(audit_samples, key=lambda item: item["sample_id"]),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    validation_report = _validation_report(
        samples=samples,
        audit_samples=audit_samples,
        clean_records=clean_records,
        sensitivity=sensitivity,
        leave_one_out=leave_one_out,
        rpc_request_count=client.request_count,
        successful_samples=successful_samples,
    )
    artifacts = {
        "normal_input_manifest.json": manifest,
        "normal_feature_audit.json": feature_audit,
        "baseline_sensitivity.json": sensitivity,
        "leave_one_out_queries.json": leave_one_out,
        "validation_report.json": validation_report,
    }
    _validate_artifacts(artifacts)
    _write_artifacts(Path(output_dir), artifacts)
    return validation_report


def process_calltracer_sample(
    sample: NormalSample,
    raw_frame: Any,
) -> tuple[dict[str, Any], tuple[ChainFeatureRecord, ...]]:
    """Run the full in-memory evidence pipeline without retaining the raw frame."""
    call_tree = adapt_calltracer_call_tree(raw_frame)
    transaction_graph = build_transaction_graph(call_tree)
    runtime_annotations = build_runtime_annotations(transaction_graph)
    asset_ledger = build_observed_asset_ledger(transaction_graph, runtime_annotations)
    continuity = build_asset_continuity(transaction_graph, runtime_annotations, asset_ledger)
    continuity_document = continuity.to_dict()
    detected = detect_result_candidates(
        transaction_graph,
        runtime_annotations,
        asset_ledger,
        continuity_document,
    )
    strict = build_strict_result_chains(
        transaction_graph,
        continuity_document,
        detected.to_dict(),
        blockchain_chain_id=1,
        offline_function_candidates=(),
        offline_effective_address_map={},
    )
    extracted = extract_normal_features(
        transaction_graph,
        strict,
        asset_ledger,
        transaction_record_id=sample.sample_id,
        offline_function_candidates=(),
        offline_effective_address_map={},
    )
    recognized_transfer_nodes = sum(
        "TOKEN_TRANSFER" in annotation.action_types
        for annotation in runtime_annotations.node_annotations
    )
    complete_asset_facts = sum(fact.status == "COMPLETE" for fact in asset_ledger.transfer_facts)
    incomplete_asset_facts = len(asset_ledger.transfer_facts) - complete_asset_facts
    feature_reason_codes = sorted(
        {
            _fixed_reason_code(reason)
            for status in extracted.chain_statuses
            for reason in status.reasons
        }
    )
    summary = {
        "sample_id": sample.sample_id,
        "tx_hash": sample.tx_hash,
        "status": "PROCESSED",
        "call_node_count": len(transaction_graph.nodes),
        "recognized_transfer_node_count": recognized_transfer_nodes,
        "asset_fact_count": len(asset_ledger.transfer_facts),
        "complete_asset_fact_count": complete_asset_facts,
        "incomplete_asset_fact_count": incomplete_asset_facts,
        "result_candidate_count": len(detected.result_candidates),
        "strict_chain_count": len(strict.chains),
        "strict_ready_count": sum(chain.status == "READY" for chain in strict.chains),
        "strict_uncertain_count": sum(chain.status == "UNCERTAIN" for chain in strict.chains),
        "clean_record_count": extracted.clean_record_count,
        "uncertain_record_count": extracted.uncertain_record_count,
        "rejected_record_count": extracted.rejected_chain_count,
        "feature_reason_codes": feature_reason_codes,
    }
    return summary, extracted.records


def build_baseline_sensitivity(
    clean_records: Iterable[ChainFeatureRecord],
) -> dict[str, Any]:
    """Calibrate each explicit minimum independently, using clean records only."""
    records = tuple(sorted(clean_records, key=lambda item: item.record_id))
    if any(record.uncertainties for record in records):
        raise ValueError("Only clean records may enter formal baseline sensitivity")
    settings: list[dict[str, Any]] = []
    for minimum in MIN_GROUP_SIZES:
        baseline = build_normal_baseline(records, min_group_size=minimum)
        groups = [
            {
                "main_anchor": group.main_anchor.to_dict(),
                "record_count": len(group.records),
                "calibration_status": group.calibration.status,
                "threshold": group.calibration.to_dict()["threshold"],
            }
            for group in baseline.groups
        ]
        settings.append(
            {
                "min_group_size": minimum,
                "total_record_count": baseline.record_count,
                "group_count": baseline.group_count,
                "group_size_distribution": _group_size_distribution(baseline.groups),
                "calibrated_group_count": baseline.calibrated_group_count,
                "uncalibrated_group_count": baseline.uncalibrated_group_count,
                "groups": groups,
            }
        )
    return {
        "schema_version": "formal_normal_baseline_sensitivity_v1",
        "clean_record_count": len(records),
        "settings": settings,
    }


def build_leave_one_out_queries(
    clean_records: Iterable[ChainFeatureRecord],
) -> dict[str, Any]:
    """Remove each query record before Top-3 lookup; never permit self matching."""
    records = tuple(sorted(clean_records, key=lambda item: item.record_id))
    if any(record.uncertainties for record in records):
        raise ValueError("Only clean records may enter leave-one-out validation")
    anchor_counts = Counter(record.main_anchor for record in records)
    settings: list[dict[str, Any]] = []
    for minimum in MIN_GROUP_SIZES:
        queries: list[dict[str, Any]] = []
        for query in records:
            if anchor_counts[query.main_anchor] < 2:
                continue
            remaining = tuple(record for record in records if record.record_id != query.record_id)
            baseline = build_normal_baseline(remaining, min_group_size=minimum)
            result = query_normal_baseline(baseline, query)
            if any(match.normal_record_id == query.record_id for match in result.top_matches):
                raise ValueError("leave-one-out query matched itself")
            queries.append(result.to_dict())
        settings.append({"min_group_size": minimum, "queries": queries})
    return {
        "schema_version": "formal_normal_leave_one_out_v1",
        "settings": settings,
    }


def _validation_report(
    *,
    samples: tuple[NormalSample, ...],
    audit_samples: list[dict[str, Any]],
    clean_records: tuple[ChainFeatureRecord, ...],
    sensitivity: Mapping[str, Any],
    leave_one_out: Mapping[str, Any],
    rpc_request_count: int,
    successful_samples: int,
) -> dict[str, Any]:
    anchor_counts = Counter(record.main_anchor for record in clean_records)
    largest_group = max(anchor_counts.values(), default=0)
    calibrated_by_minimum = {
        str(setting["min_group_size"]): setting["calibrated_group_count"]
        for setting in sensitivity["settings"]
    }
    formal_threshold_available = any(calibrated_by_minimum.values())
    leave_one_out_count = sum(
        len(setting["queries"])
        for setting in leave_one_out["settings"]
    )
    return {
        "schema_version": "formal_normal_baseline_validation_report_v1",
        "selected_sample_count": len(samples),
        "successful_sample_count": successful_samples,
        "rpc_request_count": rpc_request_count,
        "sample_results": [
            {
                key: sample[key]
                for key in (
                    "sample_id",
                    "tx_hash",
                    "call_node_count",
                    "recognized_transfer_node_count",
                    "asset_fact_count",
                    "complete_asset_fact_count",
                    "incomplete_asset_fact_count",
                    "result_candidate_count",
                    "strict_chain_count",
                    "strict_ready_count",
                    "strict_uncertain_count",
                    "clean_record_count",
                    "uncertain_record_count",
                    "rejected_record_count",
                )
            }
            for sample in sorted(audit_samples, key=lambda item: item["sample_id"])
        ],
        "clean_record_count": len(clean_records),
        "clean_anchor_group_count": len(anchor_counts),
        "clean_anchor_group_sizes": sorted(anchor_counts.values()),
        "calibrated_group_count_by_min_group_size": calibrated_by_minimum,
        "formal_threshold_available": formal_threshold_available,
        "leave_one_out_query_count": leave_one_out_count,
        "leave_one_out_top3_available": any(
            query["top_matches"]
            for setting in leave_one_out["settings"]
            for query in setting["queries"]
        ),
        "normal_sample_grouping_status": "CALIBRATED" if formal_threshold_available else "NORMAL_SAMPLE_GROUPING_INSUFFICIENT",
        "minimum_additional_clean_records_by_min_group_size": {
            str(minimum): max(0, minimum - largest_group)
            for minimum in MIN_GROUP_SIZES
        },
        "minimum_additional_clean_records_for_leave_one_out_by_min_group_size": {
            str(minimum): max(0, minimum + 1 - largest_group)
            for minimum in MIN_GROUP_SIZES
        },
    }


def _configured_ethereum_client() -> JsonRpcClient:
    """Load the existing Ethereum endpoint only after explicit live consent."""
    from src.cloc_v2.config_loader import load_chain_config

    config = load_chain_config("eth")
    rpc_url = config.get("rpc_url")
    if not isinstance(rpc_url, str) or not rpc_url:
        raise RpcValidationError("RPC_NOT_CONFIGURED")
    return JsonRpcClient(rpc_url, max_requests=MAX_RPC_REQUESTS)


def _urllib_transport(rpc_url: str, payload: Mapping[str, Any], timeout: float) -> Any:
    """POST JSON-RPC through the standard library without retaining response text."""
    body = json.dumps(dict(payload), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    request = urllib_request.Request(
        rpc_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
    except urllib_error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RpcValidationError("RPC_UNAUTHORIZED") from exc
        if exc.code == 429:
            raise RpcValidationError("RPC_QUOTA_EXCEEDED") from exc
        raise RpcValidationError("RPC_HTTP_ERROR") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RpcValidationError("RPC_TIMEOUT") from exc
    except (urllib_error.URLError, OSError) as exc:
        raise RpcValidationError("RPC_NETWORK_ERROR") from exc
    try:
        return json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RpcValidationError("RPC_INVALID_RESPONSE") from exc


def _classify_rpc_error(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=True).lower() if value is not None else ""
    if any(token in text for token in ("unauthorized", "forbidden", "authentication")):
        return "RPC_UNAUTHORIZED"
    if any(token in text for token in ("quota", "rate limit", "credits", "limit exceeded")):
        return "RPC_QUOTA_EXCEEDED"
    if any(token in text for token in ("method not found", "unsupported", "not supported")):
        return "RPC_CALLTRACER_UNSUPPORTED"
    return "RPC_REMOTE_ERROR"


def _parse_chain_id(value: Any) -> int | None:
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]+", value):
        return int(value, 16)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _group_size_distribution(groups: Iterable[Any]) -> dict[str, int]:
    distribution = Counter(len(group.records) for group in groups)
    return {str(size): count for size, count in sorted(distribution.items())}


def _assert_existing_sample_manifest(output_dir: Path, samples: tuple[NormalSample, ...]) -> None:
    """Prevent an accidental sample change between repeated live validations."""
    path = output_dir / "normal_input_manifest.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing = tuple(
            (item["sample_id"], item["tx_hash"])
            for item in payload["samples"]
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Existing normal input manifest is invalid") from exc
    selected = tuple((sample.sample_id, sample.tx_hash) for sample in samples)
    if existing != selected:
        raise ValueError("Selected samples do not match the existing normal input manifest")


def _fixed_reason_code(value: Any) -> str:
    """Persist only stable machine codes, never free-form evidence text."""
    if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
        return value
    mappings = {
        "observed asset ledger is partial": "RESULT_EVIDENCE_LEDGER_PARTIAL",
        "token decimals and real balance changes are unknown": "RESULT_EVIDENCE_TOKEN_BALANCE_UNKNOWN",
    }
    return mappings.get(value, "NON_MACHINE_READABLE_UNCERTAINTY")


def _anchor_order(anchor: MainAnchorKey) -> tuple[int, str, str]:
    return anchor.blockchain_chain_id, anchor.effective_address, anchor.selector


def _validate_artifacts(artifacts: Mapping[str, Any]) -> None:
    if tuple(sorted(artifacts)) != tuple(sorted(OUTPUT_FILENAMES)):
        raise ValueError("Unexpected validation artifact set")

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in _FORBIDDEN_OUTPUT_KEYS:
                    raise ValueError(f"Forbidden output field: {key}")
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(artifacts)


def _write_artifacts(output_dir: Path, artifacts: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILENAMES:
        payload = artifacts[filename]
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="explicitly enable the bounded live RPC validation")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.live:
        print("LIVE_MODE_REQUIRED: 未执行RPC，也未写入验证产物。")
        return 0
    try:
        report = run_validation(
            live=True,
            trace_dir=args.trace_dir,
            output_dir=args.output_dir,
        )
    except RpcValidationError as exc:
        print(f"LIVE_VALIDATION_STOPPED: {exc.category}")
        return 2
    print(
        "LIVE_VALIDATION_COMPLETE: "
        f"rpc_requests={report['rpc_request_count']} "
        f"successful_samples={report['successful_sample_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

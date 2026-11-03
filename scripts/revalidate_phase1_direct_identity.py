"""One-shot, identity-qualified direct-normal revalidation for four locked ETH txs.

The module is offline by default.  ``--live`` makes at most one chain-id and
one callTracer request for each already locked sample; it never uses DeepSeek.
Only a manifest may contain public transaction hashes.  The checkpoint and
report are deliberately identifier-free.
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
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.asset_continuity import build_asset_continuity
from src.cloc_v2.calltracer_adapter import adapt_calltracer_call_tree
from src.cloc_v2.formal_normal_baseline_loader import (
    DEFAULT_FORMAL_NORMAL_BASELINE_DIR,
    LoadedFormalNormalBaseline,
    load_formal_normal_baseline,
    resolve_formal_normal_transaction_identity,
)
from src.cloc_v2.normal_features import extract_normal_features
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.phase1_direct_normal import Phase1DirectNormalMatch, evaluate_phase1_direct_normal_match
from src.cloc_v2.result_detection import detect_result_candidates
from src.cloc_v2.runtime_semantics import build_runtime_annotations
from src.cloc_v2.strict_result_chains import build_strict_result_chains
from src.cloc_v2.transaction_graph import build_transaction_graph

EXPERIMENT_KIND = "DIRECT_NORMAL_IDENTITY_FIX_REAL_REVALIDATION"
MANIFEST_SCHEMA = "phase1_direct_identity_revalidation_manifest_v1"
CHECKPOINT_SCHEMA = "phase1_direct_identity_revalidation_checkpoint_v1"
REPORT_SCHEMA = "phase1_direct_identity_revalidation_report_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "phase1_direct_identity_revalidation_20260822"
DEFAULT_RANDOM_MANIFEST = PROJECT_ROOT / "data" / "prototypes" / "phase1_normal_random10_seed_20260814" / "locked_normal_manifest.json"
DEFAULT_RANDOM_REPORT = PROJECT_ROOT / "data" / "prototypes" / "phase1_normal_random10_seed_20260814" / "phase1_normal_pilot_report.json"
_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FORBIDDEN_KEYS = frozenset({"tx_hash", "transaction_hash", "record_id", "origin_id", "input", "calldata", "output", "error", "revertreason", "rpc_url", "api_key", "reasoning_content"})


@dataclass(frozen=True)
class LockedIdentitySample:
    sample_alias: str
    blockchain_chain_id: int
    tx_hash: str = field(repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"identity_recheck_(?:fault|normal)_00[1-3]", self.sample_alias):
            raise ValueError("IDENTITY_RECHECK_ALIAS_INVALID")
        if self.blockchain_chain_id != 1 or not isinstance(self.tx_hash, str) or _HASH.fullmatch(self.tx_hash) is None:
            raise ValueError("IDENTITY_RECHECK_SAMPLE_INVALID")

    def to_manifest_dict(self) -> dict[str, Any]:
        return {"sample_alias": self.sample_alias, "blockchain_chain_id": self.blockchain_chain_id, "tx_hash": self.tx_hash}


@dataclass(frozen=True)
class LockedIdentityManifest:
    samples: tuple[LockedIdentitySample, ...]
    manifest_sha256: str
    fourth_sample_status: str

    def __post_init__(self) -> None:
        expected = ("identity_recheck_fault_001", "identity_recheck_normal_001", "identity_recheck_normal_002")
        aliases = tuple(item.sample_alias for item in self.samples)
        if aliases[:3] != expected or len(self.samples) not in {3, 4}:
            raise ValueError("IDENTITY_RECHECK_MANIFEST_ORDER_INVALID")
        if len(self.samples) == 4 and aliases[3] != "identity_recheck_normal_003":
            raise ValueError("IDENTITY_RECHECK_FOURTH_ALIAS_INVALID")
        if self.fourth_sample_status not in {"LOCKED_FROM_HISTORICAL_REPORT", "HISTORICAL_EVIDENCE_UNAVAILABLE"}:
            raise ValueError("IDENTITY_RECHECK_FOURTH_STATUS_INVALID")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("IDENTITY_RECHECK_MANIFEST_HASH_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": MANIFEST_SCHEMA, "experiment_kind": EXPERIMENT_KIND, "scientific_generalization_result": False, "sample_count": len(self.samples), "fourth_sample_status": self.fourth_sample_status, "samples": [item.to_manifest_dict() for item in self.samples], "manifest_sha256": self.manifest_sha256}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(code) from None


def _historical_fourth(random_manifest: Path, random_report: Path) -> str | None:
    manifest = _read_json(random_manifest, "IDENTITY_RECHECK_RANDOM_MANIFEST_INVALID")
    report = _read_json(random_report, "IDENTITY_RECHECK_RANDOM_REPORT_INVALID")
    if not isinstance(manifest, Mapping) or not isinstance(report, Mapping):
        raise ValueError("IDENTITY_RECHECK_RANDOM_HISTORY_INVALID")
    by_alias: dict[str, str] = {}
    for row in manifest.get("samples", []):
        if not isinstance(row, Mapping):
            raise ValueError("IDENTITY_RECHECK_RANDOM_MANIFEST_ROWS_INVALID")
        alias, tx_hash = row.get("sample_alias"), row.get("tx_hash")
        if not isinstance(alias, str) or not isinstance(tx_hash, str) or _HASH.fullmatch(tx_hash) is None:
            raise ValueError("IDENTITY_RECHECK_RANDOM_MANIFEST_ROWS_INVALID")
        by_alias[alias] = tx_hash
    candidates: list[str] = []
    for row in report.get("sample_results", []):
        if not isinstance(row, Mapping):
            raise ValueError("IDENTITY_RECHECK_RANDOM_REPORT_ROWS_INVALID")
        if row.get("direct_normal") is True and row.get("self_reference_match") is False:
            alias = row.get("sample_alias")
            if not isinstance(alias, str) or alias not in by_alias:
                raise ValueError("IDENTITY_RECHECK_RANDOM_HISTORY_MAPPING_INVALID")
            candidates.append(by_alias[alias])
    return sorted(set(candidates))[0] if len(set(candidates)) == 3 else None


def build_locked_identity_revalidation_manifest(
    *,
    random_manifest: str | Path = DEFAULT_RANDOM_MANIFEST,
    random_report: str | Path = DEFAULT_RANDOM_REPORT,
) -> LockedIdentityManifest:
    """Lock the three fixed samples and a fourth only when old evidence proves its rule."""
    samples = [
        LockedIdentitySample("identity_recheck_fault_001", 1, "0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9"),
        LockedIdentitySample("identity_recheck_normal_001", 1, "0x01e6371b60c2ca36b3673bab9edf4caeba03e6dc756c317a90f856c5735cd2c4"),
        LockedIdentitySample("identity_recheck_normal_002", 1, "0x01f8b613d9f87f0f7fcde83c35a550b4fc758e8d218086de8ab39f31db9e72b0"),
    ]
    fourth = _historical_fourth(Path(random_manifest), Path(random_report))
    status = "HISTORICAL_EVIDENCE_UNAVAILABLE"
    if fourth is not None:
        samples.append(LockedIdentitySample("identity_recheck_normal_003", 1, fourth))
        status = "LOCKED_FROM_HISTORICAL_REPORT"
    payload = {"schema_version": MANIFEST_SCHEMA, "experiment_kind": EXPERIMENT_KIND, "fourth_sample_status": status, "samples": [item.to_manifest_dict() for item in samples]}
    return LockedIdentityManifest(tuple(samples), _sha256(_canonical_json(payload)), status)


def _parse_chain_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]+", value):
        return int(value, 16)
    return None


def _exception_code(exc: Exception, fallback: str) -> str:
    if len(exc.args) == 1 and isinstance(exc.args[0], str) and _CODE.fullmatch(exc.args[0]):
        return exc.args[0]
    return fallback


def _validate_baseline(loaded: LoadedFormalNormalBaseline) -> Mapping[str, Any]:
    summary = loaded.summary
    if summary.formal_min_group_size != 3 or not summary.manifest_consistency_passed or not summary.reconstructed_baseline_match or not summary.ethereum_only:
        raise ValueError("FORMAL_BASELINE_SAFETY_CHECK_FAILED")
    return summary.to_dict()


def process_identity_revalidation_sample(sample: LockedIdentitySample, raw_frame: Any, loaded: LoadedFormalNormalBaseline) -> dict[str, Any]:
    """Build formal evidence in memory and expose only the direct-match audit."""
    call_tree = adapt_calltracer_call_tree(raw_frame)
    graph = build_transaction_graph(call_tree)
    annotations = build_runtime_annotations(graph)
    ledger = build_observed_asset_ledger(graph, annotations)
    continuity = build_asset_continuity(graph, annotations, ledger)
    detected = detect_result_candidates(graph, annotations, ledger, continuity.to_dict())
    strict = build_strict_result_chains(graph, continuity.to_dict(), detected.to_dict(), blockchain_chain_id=1, offline_function_candidates=(), offline_effective_address_map={})
    features = extract_normal_features(graph, strict, ledger, transaction_record_id=sample.sample_alias, offline_function_candidates=(), offline_effective_address_map={})
    identity = resolve_formal_normal_transaction_identity(loaded, 1, sample.tx_hash)
    direct = evaluate_phase1_direct_normal_match(strict, features, loaded.baseline, identity, loaded.normal_record_transaction_ids)
    group_sizes: dict[str, int] = {}
    for status in features.chain_statuses:
        if not status.record_created:
            continue
        record = next((item for item in features.records if item.record_id == status.record_id), None)
        if record is not None:
            group_sizes[status.strict_chain_id] = len(next((group.records for group in loaded.baseline.groups if group.main_anchor == record.main_anchor), ()))
    chain_rows = []
    for item in direct.chain_matches:
        chain_rows.append({"chain_alias": item.chain_alias, "structural_exact_match": item.structural_exact_match, "self_origin_exact_match": item.self_origin_exact_match, "reference_origin_status": item.reference_origin_status, "call_distance": _decimal_text(item.call_distance), "asset_distance": _decimal_text(item.asset_distance), "chain_distance": _decimal_text(item.chain_distance), "normal_group_size": group_sizes.get(item._strict_chain_id, 0), "top3_count": min(3, group_sizes.get(item._strict_chain_id, 0)), "reason_codes": list(item.reason_codes)})
    stage_counts = {"call_node_count": len(graph.nodes), "result_candidate_count": len(detected.result_candidates), "strict_chain_count": len(strict.chains), "strict_ready_count": sum(item.status == "READY" for item in strict.chains), "strict_uncertain_count": sum(item.status == "UNCERTAIN" for item in strict.chains), "clean_record_count": features.clean_record_count, "uncertain_record_count": features.uncertain_record_count, "rejected_record_count": features.rejected_chain_count}
    return {"sample_alias": sample.sample_alias, "stage_counts": stage_counts, "query_identity": identity.to_dict(), "usable_clean_chain_count": direct.clean_ready_chain_count, "uncertain_chain_count": direct.uncertain_chain_count, "known_transaction_direct_normal": direct.known_transaction_direct_normal, "structural_exact_collision": direct.structural_exact_collision, "decision_kind": direct.decision_kind, "final_route": direct.final_route, "reason_codes": list(direct.reason_codes), "chain_matches": chain_rows}


def _decimal_text(value: Any) -> str | None:
    return format(value, "f") if value is not None else None


def build_direct_identity_revalidation_report(manifest: LockedIdentityManifest, baseline_summary: Mapping[str, Any] | None, sample_results: Mapping[str, Mapping[str, Any]], *, rpc_request_count: int, run_status: str) -> dict[str, Any]:
    if not 0 <= rpc_request_count <= 1 + len(manifest.samples):
        raise ValueError("IDENTITY_RECHECK_RPC_COUNT_INVALID")
    rows = []
    for sample in manifest.samples:
        row = dict(sample_results.get(sample.sample_alias, {"sample_alias": sample.sample_alias, "status": "NOT_RUN", "reason_codes": ["NOT_RUN"]}))
        if row.get("sample_alias") != sample.sample_alias:
            raise ValueError("IDENTITY_RECHECK_RESULT_ALIAS_INVALID")
        rows.append(row)
    report = {"schema_version": REPORT_SCHEMA, "experiment_kind": EXPERIMENT_KIND, "scientific_generalization_result": False, "locked_manifest_sha256": manifest.manifest_sha256, "run_status": run_status, "formal_normal_baseline_summary": dict(baseline_summary) if baseline_summary is not None else None, "sample_results": rows, "summary": {"known_transaction_direct_normal_count": sum(item.get("known_transaction_direct_normal") is True for item in rows), "structural_exact_collision_count": sum(item.get("structural_exact_collision") is True for item in rows), "phase2_count": sum(item.get("final_route") == "PROCEED_PHASE2" for item in rows)}, "external_call_counts": {"rpc": rpc_request_count, "deepseek": 0, "network": rpc_request_count}, "request_limits": {"rpc": 1 + len(manifest.samples), "deepseek": 0}}
    validate_direct_identity_revalidation_report(report)
    return report


def run_direct_identity_revalidation(manifest: LockedIdentityManifest, *, live: bool = False, rpc_client: Any | None = None, loaded_baseline: LoadedFormalNormalBaseline | None = None, sample_processor: Callable[[LockedIdentitySample, Any, LoadedFormalNormalBaseline], Mapping[str, Any]] = process_identity_revalidation_sample, checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run no network unless live is explicit; stop after the first external/local error."""
    if not live:
        return build_direct_identity_revalidation_report(manifest, None, {}, rpc_request_count=0, run_status="LIVE_MODE_REQUIRED")
    if rpc_client is None or not callable(getattr(rpc_client, "call", None)) or loaded_baseline is None:
        raise ValueError("IDENTITY_RECHECK_LIVE_DEPENDENCY_MISSING")
    baseline_summary = _validate_baseline(loaded_baseline)
    results: dict[str, Mapping[str, Any]] = {}
    state: dict[str, Any] = {"schema_version": CHECKPOINT_SCHEMA, "locked_manifest_sha256": manifest.manifest_sha256, "samples": {}, "rpc_request_count": 0, "last_stop_code": None}
    def checkpoint() -> None:
        if checkpoint_callback is not None:
            checkpoint_callback(json.loads(_canonical_json(state)))
    try:
        chain_value = rpc_client.call("eth_chainId", [])
        state["rpc_request_count"] = 1
    except Exception as exc:
        state["last_stop_code"] = _exception_code(exc, "RPC_CHAIN_ID_ERROR"); checkpoint()
        return build_direct_identity_revalidation_report(manifest, baseline_summary, results, rpc_request_count=1, run_status="STOPPED")
    if _parse_chain_id(chain_value) != 1:
        state["last_stop_code"] = "CHAIN_ID_MISMATCH"; checkpoint()
        return build_direct_identity_revalidation_report(manifest, baseline_summary, results, rpc_request_count=1, run_status="STOPPED")
    for sample in manifest.samples:
        try:
            raw = rpc_client.call("debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}])
            state["rpc_request_count"] += 1
        except Exception as exc:
            # JsonRpcClient increments immediately before its one transport attempt.
            # Count that attempted request even when the endpoint reports an error.
            state["rpc_request_count"] += 1
            code = _exception_code(exc, "RPC_TRACE_ERROR")
            results[sample.sample_alias] = {"sample_alias": sample.sample_alias, "status": "FAILED", "reason_codes": [code]}
            state["samples"][sample.sample_alias] = {"status": "FAILED", "reason_codes": [code]}; state["last_stop_code"] = code; checkpoint()
            return build_direct_identity_revalidation_report(manifest, baseline_summary, results, rpc_request_count=state["rpc_request_count"], run_status="STOPPED")
        try:
            result = dict(sample_processor(sample, raw, loaded_baseline))
            if result.get("sample_alias") != sample.sample_alias:
                raise ValueError("SAMPLE_ALIAS_MISMATCH")
        except Exception as exc:
            code = _exception_code(exc, "SAMPLE_PROCESSING_ERROR")
            results[sample.sample_alias] = {"sample_alias": sample.sample_alias, "status": "FAILED", "reason_codes": [code]}
            state["samples"][sample.sample_alias] = {"status": "FAILED", "reason_codes": [code]}; state["last_stop_code"] = code; checkpoint()
            return build_direct_identity_revalidation_report(manifest, baseline_summary, results, rpc_request_count=state["rpc_request_count"], run_status="STOPPED")
        results[sample.sample_alias] = result
        state["samples"][sample.sample_alias] = {"status": "SUCCEEDED", "reason_codes": []}; checkpoint()
    return build_direct_identity_revalidation_report(manifest, baseline_summary, results, rpc_request_count=state["rpc_request_count"], run_status="COMPLETED")


def validate_direct_identity_revalidation_report(report: Mapping[str, Any]) -> None:
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in _FORBIDDEN_KEYS:
                    raise ValueError("IDENTITY_RECHECK_REPORT_FORBIDDEN_FIELD")
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value: visit(nested)
        elif isinstance(value, str):
            if _RAW_HASH.search(value) or _RAW_ADDRESS.search(value) or any(part in value.lower() for part in ("calldata", "revertreason", "reasoning_content", "rpc_url", "api key")):
                raise ValueError("IDENTITY_RECHECK_REPORT_SENSITIVE_CONTENT")
    visit(report)


def serialize_locked_identity_manifest(manifest: LockedIdentityManifest) -> str:
    return json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_direct_identity_revalidation_report(report: Mapping[str, Any]) -> str:
    validate_direct_identity_revalidation_report(report)
    return json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno()); temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def write_direct_identity_revalidation_artifacts(output_dir: str | Path, manifest: LockedIdentityManifest, report: Mapping[str, Any], checkpoint: Mapping[str, Any] | None = None) -> tuple[Path, Path, Path]:
    root = Path(output_dir)
    manifest_path, checkpoint_path, report_path = root / "locked_manifest.json", root / "checkpoint.json", root / "direct_identity_revalidation_report.json"
    _atomic_write(manifest_path, serialize_locked_identity_manifest(manifest))
    safe_checkpoint = dict(checkpoint or {"schema_version": CHECKPOINT_SCHEMA, "locked_manifest_sha256": manifest.manifest_sha256, "samples": {}, "rpc_request_count": report.get("external_call_counts", {}).get("rpc", 0), "last_stop_code": None})
    _atomic_write(checkpoint_path, json.dumps(safe_checkpoint, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    _atomic_write(report_path, serialize_direct_identity_revalidation_report(report))
    return manifest_path, checkpoint_path, report_path


def _load_live_chain_config() -> Mapping[str, Any]:
    from src.cloc_v2.config_loader import load_chain_config
    return load_chain_config("eth")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--live", action="store_true"); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR); args = parser.parse_args(argv)
    if not args.live:
        print("live_required=true"); print("rpc_request_count=0"); print("deepseek_request_count=0"); return 2
    try:
        manifest = build_locked_identity_revalidation_manifest()
        _atomic_write(args.output_dir / "locked_manifest.json", serialize_locked_identity_manifest(manifest))
        loaded = load_formal_normal_baseline(DEFAULT_FORMAL_NORMAL_BASELINE_DIR)
        config = _load_live_chain_config(); rpc_url = config.get("rpc_url")
        if not isinstance(rpc_url, str) or not rpc_url: raise ValueError("ETH_RPC_CONFIG_NOT_LOADED")
        checkpoints: list[Mapping[str, Any]] = []
        report = run_direct_identity_revalidation(manifest, live=True, rpc_client=JsonRpcClient(rpc_url, max_requests=1 + len(manifest.samples)), loaded_baseline=loaded, checkpoint_callback=lambda value: (checkpoints.append(value), _atomic_write(args.output_dir / "checkpoint.json", json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")))
        paths = write_direct_identity_revalidation_artifacts(args.output_dir, manifest, report, checkpoints[-1] if checkpoints else None)
    except Exception as exc:
        print("revalidation_status=STOPPED"); print("reason_code=" + _exception_code(exc, "REVALIDATION_SETUP_ERROR")); return 1
    print("revalidation_status=" + str(report["run_status"])); print("manifest_path=" + str(paths[0])); print("report_path=" + str(paths[2])); print("rpc_request_count=" + str(report["external_call_counts"]["rpc"])); print("deepseek_request_count=0")
    return 0 if report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

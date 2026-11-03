"""Frozen-method final small Ethereum experiment coordinator.

This module deliberately separates selection, blind inference, and evaluation.
Importing it is offline.  The command line refuses external access unless the
``run --live`` mode is explicitly selected.
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
from typing import Any, Callable, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEED = "20260901"
CHAIN_ID = 1
NORMAL_COUNT = FAULT_COUNT = 20
PHASE2_QUEUE_LIMIT = 10
MAX_RPC_REQUESTS = 41
MAX_SOURCE_REQUESTS = 30
MAX_PHASE1_DEEPSEEK_REQUESTS = 40
MAX_PHASE2_DEEPSEEK_REQUESTS = 10
MAX_DEEPSEEK_REQUESTS = 50
DEFAULT_OUTPUT_DIR = ROOT / "data" / "prototypes" / "final_small_experiment_eth_seed_20260901"
EXPERIMENT_KIND = "CLOC_FINAL_SMALL_ETHEREUM_ENGINEERING_EXPERIMENT"
_TX = re.compile(r"^0x[0-9a-f]{64}$")
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_FORBIDDEN = frozenset({"cause", "location", "category", "label", "ground_truth", "pair", "rpc_url", "api_key", "input", "calldata", "output", "error", "revertreason", "reasoning_content", "provider_response", "prompt"})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _rank(namespace: str, tx_hash: str) -> str:
    return hashlib.sha256(f"{namespace}|{SEED}|{tx_hash.lower()}".encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(value)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(path)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _valid_tx(value: Any) -> str | None:
    value = str(value).lower()
    return value if _TX.fullmatch(value) else None


@dataclass(frozen=True)
class InferenceSample:
    sample_alias: str
    blockchain_chain_id: int
    tx_hash: str = field(repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"final_eth_\d{3}", self.sample_alias) or self.blockchain_chain_id != CHAIN_ID or not _valid_tx(self.tx_hash):
            raise ValueError("FINAL_INFERENCE_SAMPLE_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {"sample_alias": self.sample_alias, "blockchain_chain_id": self.blockchain_chain_id, "tx_hash": self.tx_hash}


@dataclass(frozen=True)
class LockedInferenceManifest:
    samples: tuple[InferenceSample, ...]
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if len(self.samples) != 40 or len({item.tx_hash for item in self.samples}) != 40 or tuple(item.sample_alias for item in self.samples) != tuple(f"final_eth_{index:03d}" for index in range(1, 41)):
            raise ValueError("FINAL_MANIFEST_SAMPLE_SET_INVALID")
        payload = {"schema_version": "final_small_inference_manifest_v1", "experiment_kind": EXPERIMENT_KIND, "selection_seed": SEED, "samples": [item.to_dict() for item in self.samples]}
        expected = _sha(payload)
        if self.manifest_sha256 and self.manifest_sha256 != expected:
            raise ValueError("FINAL_MANIFEST_SHA_INVALID")
        object.__setattr__(self, "manifest_sha256", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "final_small_inference_manifest_v1", "experiment_kind": EXPERIMENT_KIND, "scientific_generalization_result": False, "selection_seed": SEED, "samples": [item.to_dict() for item in self.samples], "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class EvaluationTruth:
    manifest_sha256: str
    labels: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.labels) != 40 or set(self.labels.values()) != {"NORMAL", "FAULT"} or list(self.labels.values()).count("NORMAL") != 20 or list(self.labels.values()).count("FAULT") != 20:
            raise ValueError("FINAL_TRUTH_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "final_small_evaluation_truth_v1", "locked_manifest_sha256": self.manifest_sha256, "labels": dict(sorted(self.labels.items()))}


@dataclass(frozen=True)
class FinalSelection:
    manifest: LockedInferenceManifest
    truth: EvaluationTruth = field(repr=False)


@dataclass
class ExternalRequestCounters:
    rpc: int = 0
    source: int = 0
    phase1_deepseek: int = 0
    phase2_deepseek: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"rpc": self.rpc, "source": self.source, "phase1_deepseek": self.phase1_deepseek, "phase2_deepseek": self.phase2_deepseek}


@dataclass(frozen=True)
class LiveAdapters:
    rpc_client: Any = field(repr=False)
    loaded_baseline: Any = field(repr=False)
    phase1_processor: Callable[[InferenceSample, Any], Mapping[str, Any]] = field(repr=False)
    phase2_processor: Callable[[InferenceSample, Any], Mapping[str, Any]] = field(repr=False)
    counters: ExternalRequestCounters = field(repr=False)


def build_live_adapters(*, chain_config: Mapping[str, Any], api_key: str, rpc_client: Any | None = None) -> LiveAdapters:
    """Connect the frozen Phase-1/Phase-2 pilots without changing their methods."""
    from scripts.run_phase1_deepseek_smoke import DeepSeekHTTPSOneShotTransport
    from scripts.run_phase1_normal_pilot import prepare_normal_pilot_sample, route_normal_pilot_sample
    from scripts.run_phase2_real_pilot import acquire_real_source_slices, build_real_phase2_evidence, map_validated_top3, prepare_phase2_topk_v3
    from scripts.validate_real_normal_baseline import JsonRpcClient
    from src.cloc_v2.formal_normal_baseline_loader import DEFAULT_FORMAL_NORMAL_BASELINE_DIR, load_formal_normal_baseline
    from src.cloc_v2.phase1_normality_deepseek import Phase1DeepSeekConfig, execute_phase1_deepseek_review
    from src.cloc_v2.phase2_localization_bridge import prepare_phase2_localization
    from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekConfig, execute_phase2_localization
    rpc_url = chain_config.get("rpc_url")
    if not isinstance(rpc_url, str) or not rpc_url: raise ValueError("ETH_RPC_CONFIG_NOT_LOADED")
    if not isinstance(api_key, str) or not api_key: raise ValueError("DEEPSEEK_KEY_NOT_LOADED")
    counters = ExternalRequestCounters()
    client = rpc_client or JsonRpcClient(rpc_url, max_requests=MAX_RPC_REQUESTS)
    loaded = load_formal_normal_baseline(DEFAULT_FORMAL_NORMAL_BASELINE_DIR)

    def phase1_deepseek(preparation):
        if counters.phase1_deepseek >= MAX_PHASE1_DEEPSEEK_REQUESTS or counters.phase1_deepseek + counters.phase2_deepseek >= MAX_DEEPSEEK_REQUESTS: raise RuntimeError("DEEPSEEK_REQUEST_LIMIT_REACHED")
        counters.phase1_deepseek += 1
        return execute_phase1_deepseek_review(preparation, Phase1DeepSeekConfig(), DeepSeekHTTPSOneShotTransport(), api_key)

    def phase1_processor(sample: InferenceSample, raw: Any) -> Mapping[str, Any]:
        prepared = prepare_normal_pilot_sample(sample, raw, loaded)
        row, _, route_error = route_normal_pilot_sample(prepared, phase1_deepseek)
        return {**row, "route_error": bool(route_error)}

    def phase2_processor(sample: InferenceSample, raw: Any) -> Mapping[str, Any]:
        bundle = build_real_phase2_evidence(raw)
        slices, source_summary = acquire_real_source_slices(bundle, chain_config)
        used = int(source_summary.get("source_request_count", 0)); counters.source += used
        if counters.source > MAX_SOURCE_REQUESTS: raise RuntimeError("SOURCE_REQUEST_LIMIT_REACHED")
        preparation = prepare_phase2_topk_v3(prepare_phase2_localization(bundle.strict, bundle.causal, bundle.graph, slices))
        base = {"bridge_status": preparation.preparation_status, "source_summary": dict(source_summary), "reason_codes": [preparation.preparation_status] if preparation.preparation_status != "READY_FOR_LLM" else [], "llm_called": False, "topk_frozen": False, "top3": []}
        if preparation.preparation_status != "READY_FOR_LLM": return base
        if counters.phase2_deepseek >= MAX_PHASE2_DEEPSEEK_REQUESTS or counters.phase1_deepseek + counters.phase2_deepseek >= MAX_DEEPSEEK_REQUESTS: raise RuntimeError("DEEPSEEK_REQUEST_LIMIT_REACHED")
        counters.phase2_deepseek += 1
        result = execute_phase2_localization(preparation, Phase2DeepSeekConfig(), DeepSeekHTTPSOneShotTransport(), api_key)
        top3 = map_validated_top3(preparation, result)
        valid = bool(top3) or (result.response_status == "VALIDATED" and result.validated_response is not None)
        return {**base, "llm_called": True, "llm_status": result.response_status, "topk_frozen": valid, "top3": top3, "reason_codes": list(result.reason_codes)}

    return LiveAdapters(client, loaded, phase1_processor, phase2_processor, counters)


def _history_transactions(prototypes_root: Path) -> set[str]:
    values: set[str] = set()
    if not prototypes_root.exists():
        return values
    for path in sorted(prototypes_root.rglob("*manifest*.json")):
        if "formal_normal_baseline_full" in path.parts:
            continue
        value = _read_json(path)
        if value is None:
            continue
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, list): stack.extend(item)
            elif isinstance(item, Mapping):
                for key, child in item.items():
                    if key in {"tx_hash", "transaction_hash"}:
                        tx = _valid_tx(child)
                        if tx: values.add(tx)
                    elif key in {"samples", "locked_samples", "items", "records", "data"}:
                        stack.append(child)
    return values


def _normal_pool(normal_manifest: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> set[str]:
    states = {str(key): value for key, value in (checkpoint.get("samples") or {}).items() if isinstance(value, Mapping) and value.get("status") == "SUCCEEDED"}
    result: set[str] = set()
    for row in normal_manifest.get("samples", []):
        if not isinstance(row, Mapping) or row.get("chain_key") != "eth" or row.get("blockchain_chain_id") != 1:
            continue
        tx = _valid_tx(row.get("tx_hash"))
        checkpoint_key = f"eth:{tx}" if tx else ""
        if tx and checkpoint_key in states:
            result.add(tx)
    return result


def _fault_pool(dataset_root: Path) -> set[str]:
    """Read only platform and fault.transaction_hash; never inspect labels/cause/location."""
    result: set[str] = set()
    for path in sorted(dataset_root.rglob("*.json")):
        value = _read_json(path)
        if not isinstance(value, Mapping) or str(value.get("platform", "")).lower() not in {"ethereum", "eth"}:
            continue
        fault = value.get("fault")
        if not isinstance(fault, Mapping):
            continue
        stack = [fault.get("transaction_hash")]
        while stack:
            value = stack.pop()
            if isinstance(value, (list, tuple)):
                stack.extend(value)
                continue
            tx = _valid_tx(value)
            if tx:
                result.add(tx)
    return result


def build_final_selection(normal_manifest: Mapping[str, Any], checkpoint: Mapping[str, Any], dataset_root: Path, prototypes_root: Path) -> FinalSelection:
    history = _history_transactions(prototypes_root)
    normal = _normal_pool(normal_manifest, checkpoint) - history
    fault = _fault_pool(dataset_root) - history
    overlap = normal & fault
    normal -= overlap
    fault -= overlap
    chosen_normal = sorted(normal, key=lambda tx: (_rank("CLOC_FINAL_SMALL_NORMAL_V1", tx), tx))[:NORMAL_COUNT]
    chosen_fault = sorted(fault, key=lambda tx: (_rank("CLOC_FINAL_SMALL_FAULT_V1", tx), tx))[:FAULT_COUNT]
    if len(chosen_normal) != NORMAL_COUNT or len(chosen_fault) != FAULT_COUNT:
        raise ValueError("FINAL_SELECTION_POOL_INSUFFICIENT")
    labels = {tx: "NORMAL" for tx in chosen_normal} | {tx: "FAULT" for tx in chosen_fault}
    ordered = sorted(labels, key=lambda tx: (_rank("CLOC_FINAL_SMALL_ORDER_V1", tx), tx))
    samples = tuple(InferenceSample(f"final_eth_{index:03d}", CHAIN_ID, tx) for index, tx in enumerate(ordered, 1))
    manifest = LockedInferenceManifest(samples)
    truth = EvaluationTruth(manifest.manifest_sha256, {sample.sample_alias: labels[sample.tx_hash] for sample in samples})
    return FinalSelection(manifest, truth)


def write_prepared_final_experiment(output_dir: Path, selection: FinalSelection) -> None:
    _atomic_json(output_dir / "locked_inference_manifest.json", selection.manifest.to_dict())
    _atomic_json(output_dir / "evaluation_truth.json", selection.truth.to_dict())


def load_prepared_final_experiment(output_dir: Path) -> FinalSelection:
    raw_manifest=_read_json(output_dir/"locked_inference_manifest.json"); raw_truth=_read_json(output_dir/"evaluation_truth.json")
    if not isinstance(raw_manifest,Mapping) or not isinstance(raw_truth,Mapping): raise ValueError("FINAL_PREPARED_ARTIFACTS_MISSING")
    samples=tuple(InferenceSample(str(x["sample_alias"]),int(x["blockchain_chain_id"]),str(x["tx_hash"])) for x in raw_manifest.get("samples",[]) if isinstance(x,Mapping))
    manifest=LockedInferenceManifest(samples,str(raw_manifest.get("manifest_sha256","")))
    truth=EvaluationTruth(str(raw_truth.get("locked_manifest_sha256","")),dict(raw_truth.get("labels") or {}))
    return FinalSelection(manifest,truth)


def select_phase2_queue(phase1_predictions: Iterable[Mapping[str, Any]], manifest: LockedInferenceManifest) -> tuple[str, ...]:
    tx_by_alias = {item.sample_alias: item.tx_hash for item in manifest.samples}
    aliases = [str(row.get("sample_alias")) for row in phase1_predictions if row.get("final_route") == "PROCEED_PHASE2" and str(row.get("sample_alias")) in tx_by_alias]
    return tuple(sorted(set(aliases), key=lambda alias: (_rank("CLOC_FINAL_SMALL_PHASE2_QUEUE_V1", tx_by_alias[alias]), tx_by_alias[alias]))[:PHASE2_QUEUE_LIMIT])


def _sanitize(value: Any) -> None:
    encoded = _canonical(value)
    if _HASH.search(encoded) or _ADDRESS.search(encoded): raise ValueError("FINAL_REPORT_RAW_IDENTITY_FORBIDDEN")
    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).lower() in _FORBIDDEN: raise ValueError("FINAL_REPORT_FORBIDDEN_FIELD")
                visit(child)
        elif isinstance(item, list):
            for child in item: visit(child)
    visit(value)


def _frozen_blob(kind: str, manifest_sha: str, predictions: list[Mapping[str, Any]], queue: tuple[str, ...] = ()) -> dict[str, Any]:
    payload = {"schema_version": "final_small_frozen_predictions_v1", "kind": kind, "locked_manifest_sha256": manifest_sha, "phase2_queue": list(queue), "predictions": predictions}
    _sanitize(payload)
    return {**payload, "payload_sha256": _sha(payload)}


def freeze_predictions(path: Path, blob: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path)
        if existing != blob: raise ValueError("FINAL_FROZEN_PREDICTIONS_IMMUTABLE")
        return
    _atomic_json(path, blob)


def _load_frozen(path: Path, manifest_sha: str, kind: str) -> Mapping[str, Any]:
    blob = _read_json(path)
    if not isinstance(blob, Mapping) or blob.get("kind") != kind or blob.get("locked_manifest_sha256") != manifest_sha: raise ValueError("FINAL_FROZEN_PREDICTIONS_INVALID")
    payload = {key: blob[key] for key in ("schema_version", "kind", "locked_manifest_sha256", "phase2_queue", "predictions") if key in blob}
    if blob.get("payload_sha256") != _sha(payload): raise ValueError("FINAL_FROZEN_PREDICTIONS_HASH_INVALID")
    _sanitize(payload)
    return blob


def infer_final_experiment(manifest: LockedInferenceManifest, output_dir: Path, *, live: bool, rpc_client: Any | None = None, phase1_processor: Callable[[InferenceSample, Any], Mapping[str, Any]] | None = None, phase2_processor: Callable[[InferenceSample, Any], Mapping[str, Any]] | None = None, counters: ExternalRequestCounters | None = None) -> Mapping[str, Any]:
    """Blind inference boundary: deliberately accepts neither truth nor location loader."""
    if not live:
        return {"run_status": "LIVE_MODE_REQUIRED", "external_call_counts": {"rpc": 0, "source": 0, "phase1_deepseek": 0, "phase2_deepseek": 0}}
    if rpc_client is None or not callable(getattr(rpc_client, "call", None)) or phase1_processor is None or phase2_processor is None: raise ValueError("FINAL_INFERENCE_CLIENTS_REQUIRED")
    counters = counters or ExternalRequestCounters()
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = _read_json(checkpoint_path) if checkpoint_path.exists() else None
    if checkpoint is None:
        checkpoint = {"schema_version":"final_small_checkpoint_v2","locked_manifest_sha256":manifest.manifest_sha256,"status":"RUNNING","samples":{s.sample_alias:{"status":"PENDING","trace_acquired":False,"phase1_artifact":None} for s in manifest.samples},"phase2_queue":[],"phase2":{},"external_call_counts":counters.to_dict()}
    if not isinstance(checkpoint, Mapping) or checkpoint.get("locked_manifest_sha256") != manifest.manifest_sha256: raise ValueError("FINAL_CHECKPOINT_INVALID")
    checkpoint = json.loads(_canonical(checkpoint)); raws: dict[str, Any] = {}
    stored_counts=checkpoint.get("external_call_counts") or {}
    for name in ("rpc","source","phase1_deepseek","phase2_deepseek"):
        setattr(counters,name,max(getattr(counters,name),int(stored_counts.get(name,0))))
    def save(): checkpoint["external_call_counts"] = counters.to_dict(); _atomic_json(checkpoint_path, checkpoint)
    if counters.rpc == 0:
        counters.rpc += 1
        try:
            chain = rpc_client.call("eth_chainId", [])
            parsed = int(chain, 16) if isinstance(chain, str) else chain
            if parsed != CHAIN_ID: raise ValueError("CHAIN_ID_MISMATCH")
        except Exception:
            checkpoint["status"]="STOPPED"; save(); return {"run_status":"STOPPED","reason_code":"RPC_CHAIN_ID_ERROR","external_call_counts":counters.to_dict()}
    phase1_rows: list[Mapping[str, Any]] = []
    for sample in manifest.samples:
        state=checkpoint["samples"][sample.sample_alias]
        if state["status"] in {"SUCCEEDED","FAILED"}:
            artifact=state.get("phase1_artifact"); value=_read_json(output_dir / artifact) if artifact else None
            if isinstance(value,Mapping): phase1_rows.append(value)
            continue
        if counters.rpc >= MAX_RPC_REQUESTS: raise ValueError("RPC_REQUEST_LIMIT_REACHED")
        counters.rpc += 1
        try: raw = rpc_client.call("debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}])
        except Exception:
            state.update(status="FAILED",trace_acquired=False); checkpoint["status"]="STOPPED"; save(); return {"run_status":"STOPPED","reason_code":"RPC_TRACE_ERROR","external_call_counts":counters.to_dict()}
        raws[sample.sample_alias] = raw
        try: row = dict(phase1_processor(sample, raw)); row["sample_alias"] = sample.sample_alias; _sanitize(row)
        except Exception:
            row={"sample_alias":sample.sample_alias,"final_route":"PROCEED_PHASE2","reason_codes":["PHASE1_PROCESSING_ERROR"],"llm_called":False}; state["status"]="FAILED"
        else: state["status"]="SUCCEEDED"
        artifact=f"predictions/{sample.sample_alias}.json"; _atomic_json(output_dir/artifact,row); state.update(trace_acquired=True,phase1_artifact=artifact); phase1_rows.append(row); save()
    if any(checkpoint["samples"][s.sample_alias]["status"] not in {"SUCCEEDED","FAILED"} for s in manifest.samples): checkpoint["status"]="STOPPED"; save(); return {"run_status":"STOPPED","external_call_counts":counters.to_dict()}
    completed = {str(row["sample_alias"]) for row in phase1_rows}
    if len(completed)!=40: raise ValueError("FINAL_PHASE1_ARTIFACT_SET_INCOMPLETE")
    phase1_blob = _frozen_blob("phase1", manifest.manifest_sha256, sorted(phase1_rows, key=lambda row: row["sample_alias"]))
    freeze_predictions(output_dir / "frozen_phase1_predictions.json", phase1_blob)
    queue = tuple(checkpoint.get("phase2_queue") or ())
    if not queue: queue=select_phase2_queue(phase1_rows,manifest); checkpoint["phase2_queue"]=list(queue); checkpoint["phase2"]={a:{"status":"PENDING","artifact":None} for a in queue}; save()
    phase2_rows: list[Mapping[str, Any]] = []
    for alias in queue:
        sample = next(item for item in manifest.samples if item.sample_alias == alias)
        state=checkpoint["phase2"][alias]
        if state["status"] in {"SUCCEEDED","FAILED"}:
            value=_read_json(output_dir/state["artifact"]) if state.get("artifact") else None
            if isinstance(value,Mapping): phase2_rows.append(value)
            continue
        if alias not in raws: row={"sample_alias":alias,"status":"TRACE_UNAVAILABLE_AFTER_RESTART","reason_codes":["TRACE_UNAVAILABLE_AFTER_RESTART"],"llm_called":False,"topk_frozen":False}; state["status"]="FAILED"
        else:
            try: row=dict(phase2_processor(sample,raws[alias])); row["sample_alias"]=alias; _sanitize(row); state["status"]="SUCCEEDED"
            except Exception: row={"sample_alias":alias,"status":"PHASE2_PROCESSING_ERROR","reason_codes":["PHASE2_PROCESSING_ERROR"],"llm_called":False,"topk_frozen":False}; state["status"]="FAILED"
        artifact=f"predictions/{alias}_phase2.json"; _atomic_json(output_dir/artifact,row); state["artifact"]=artifact; phase2_rows.append(row); save()
    phase2_blob = _frozen_blob("phase2", manifest.manifest_sha256, sorted(phase2_rows, key=lambda row: row["sample_alias"]), queue)
    freeze_predictions(output_dir / "frozen_phase2_predictions.json", phase2_blob)
    checkpoint["status"]="COMPLETED"; save()
    return checkpoint


def evaluate_final_experiment(output_dir: Path, manifest: LockedInferenceManifest, truth: EvaluationTruth, *, location_loader: Callable[[str], Any] | None = None) -> Mapping[str, Any]:
    """Evaluation boundary.  Location is read only after both frozen blobs verify."""
    if truth.manifest_sha256 != manifest.manifest_sha256: raise ValueError("FINAL_TRUTH_MANIFEST_MISMATCH")
    p1 = _load_frozen(output_dir / "frozen_phase1_predictions.json", manifest.manifest_sha256, "phase1")
    p2 = _load_frozen(output_dir / "frozen_phase2_predictions.json", manifest.manifest_sha256, "phase2")
    rows = {str(row["sample_alias"]): row for row in p1["predictions"]}
    if set(rows) != {item.sample_alias for item in manifest.samples}: raise ValueError("FINAL_PHASE1_TERMINAL_SET_INVALID")
    phase2 = {str(row["sample_alias"]): row for row in p2["predictions"]}
    normal = [alias for alias, label in truth.labels.items() if label == "NORMAL"]; fault = [alias for alias, label in truth.labels.items() if label == "FAULT"]
    direct_normal = lambda row: row.get("final_route") == "EARLY_NORMAL"
    normal_early = sum(direct_normal(rows[alias]) for alias in normal); fault_early = sum(direct_normal(rows[alias]) for alias in fault)
    phase2_fault = [alias for alias in p2.get("phase2_queue", []) if truth.labels.get(alias) == "FAULT"]
    evaluations: dict[str, str] = {}
    for alias in phase2_fault:
        row = phase2.get(alias, {})
        if row.get("topk_frozen") and location_loader is not None:
            evaluations[alias] = str(location_loader(alias))
        else: evaluations[alias] = "EVALUATION_UNAVAILABLE"
    report = {"schema_version": "final_small_experiment_report_v1", "experiment_kind": EXPERIMENT_KIND, "scientific_generalization_result": False, "locked_manifest_sha256": manifest.manifest_sha256, "phase1": {"normal_early_normal_count": normal_early, "normal_early_normal_rate": normal_early / 20, "fault_error_normal_count": fault_early, "fault_error_normal_rate": fault_early / 20, "normal_count": 20, "fault_count": 20, "phase2_upgrade_count": sum(row.get("final_route") == "PROCEED_PHASE2" for row in rows.values())}, "phase2": {"blind_queue_count": len(p2.get("phase2_queue", [])), "fault_queue_count": len(phase2_fault), "evaluations": evaluations, "top1_hit_count": sum(item in {"TOP1_HIT", "TOP1_WINDOW_HIT"} for item in evaluations.values()), "top3_hit_count": sum(item in {"TOP3_HIT", "TOP3_WINDOW_HIT", "TOP1_HIT", "TOP1_WINDOW_HIT"} for item in evaluations.values()), "evaluation_unavailable_count": sum(item == "EVALUATION_UNAVAILABLE" for item in evaluations.values())}, "frozen_prediction_hashes": {"phase1": p1["payload_sha256"], "phase2": p2["payload_sha256"]}}
    _sanitize(report)
    _atomic_json(output_dir / "final_small_experiment_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("prepare", "run")); parser.add_argument("--live", action="store_true"); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR); args = parser.parse_args(argv)
    # Preparation is intentionally offline, while a complete external run needs
    # the explicit opt-in below.  This keeps accidental imports and test runs
    # from loading any credentials.
    if args.command == "prepare":
        from src.cloc_v2.formal_normal_baseline_loader import DEFAULT_FORMAL_NORMAL_BASELINE_DIR
        baseline_dir = DEFAULT_FORMAL_NORMAL_BASELINE_DIR
        normal_manifest = _read_json(baseline_dir / "normal_manifest.json")
        checkpoint = _read_json(baseline_dir / "checkpoint.json")
        if not isinstance(normal_manifest, Mapping) or not isinstance(checkpoint, Mapping):
            print("prepare_status=FORMAL_NORMAL_LIBRARY_UNAVAILABLE"); return 1
        selection = build_final_selection(normal_manifest, checkpoint, ROOT / "dataset", ROOT / "data" / "prototypes")
        manifest_path = args.output_dir / "locked_inference_manifest.json"; truth_path = args.output_dir / "evaluation_truth.json"
        for path, expected in ((manifest_path, selection.manifest.to_dict()), (truth_path, selection.truth.to_dict())):
            if path.exists() and _read_json(path) != expected:
                print("prepare_status=LOCKED_ARTIFACT_CONFLICT"); return 1
        write_prepared_final_experiment(args.output_dir, selection)
        print("prepare_status=LOCKED"); print("rpc_request_count=0"); print("source_request_count=0"); print("deepseek_request_count=0"); return 0
    if not args.live:
        print("live_required=true"); print("rpc_request_count=0"); print("source_request_count=0"); print("deepseek_request_count=0"); return 2
    try:
        if not (args.output_dir/"locked_inference_manifest.json").exists():
            from src.cloc_v2.formal_normal_baseline_loader import DEFAULT_FORMAL_NORMAL_BASELINE_DIR
            nm=_read_json(DEFAULT_FORMAL_NORMAL_BASELINE_DIR/"normal_manifest.json"); cp=_read_json(DEFAULT_FORMAL_NORMAL_BASELINE_DIR/"checkpoint.json")
            if not isinstance(nm,Mapping) or not isinstance(cp,Mapping): raise ValueError("FORMAL_NORMAL_LIBRARY_UNAVAILABLE")
            write_prepared_final_experiment(args.output_dir,build_final_selection(nm,cp,ROOT/"dataset",ROOT/"data"/"prototypes"))
        selection=load_prepared_final_experiment(args.output_dir)
        from src.cloc_v2.config_loader import load_chain_config
        config=load_chain_config("eth"); key=os.getenv("DEEPSEEK_API_KEY")
        adapters=build_live_adapters(chain_config=config,api_key=key or "")
        checkpoint=infer_final_experiment(selection.manifest,args.output_dir,live=True,rpc_client=adapters.rpc_client,phase1_processor=adapters.phase1_processor,phase2_processor=adapters.phase2_processor,counters=adapters.counters)
        if checkpoint.get("status")!="COMPLETED": print("experiment_status=STOPPED"); return 1
        p2=_load_frozen(args.output_dir/"frozen_phase2_predictions.json",selection.manifest.manifest_sha256,"phase2"); by_alias={r["sample_alias"]:r for r in p2["predictions"]}
        tx_by_alias={s.sample_alias:s.tx_hash for s in selection.manifest.samples}
        def location_eval(alias:str)->str:
            from scripts.run_phase2_real_pilot import evaluate_top3_locations
            wanted=tx_by_alias[alias]
            for path in sorted((ROOT/"dataset").rglob("*.json")):
                value=_read_json(path)
                if isinstance(value,Mapping) and str(value.get("platform","")).lower() in {"ethereum","eth"}:
                    fault=value.get("fault")
                    if isinstance(fault,Mapping):
                        hashes=fault.get("transaction_hash"); hashes=hashes if isinstance(hashes,list) else [hashes]
                        if wanted in [str(x).lower() for x in hashes]: return evaluate_top3_locations(list(by_alias.get(alias,{}).get("top3",[])),fault.get("location"))
            return "EVALUATION_UNAVAILABLE"
        evaluate_final_experiment(args.output_dir,selection.manifest,selection.truth,location_loader=location_eval)
    except Exception as exc:
        print("experiment_status=STOPPED"); print("reason_code="+(exc.args[0] if len(exc.args)==1 and isinstance(exc.args[0],str) and re.fullmatch(r"[A-Z][A-Z0-9_]*",exc.args[0]) else "FINAL_EXPERIMENT_ERROR")); return 1
    print("experiment_status=COMPLETED"); print("rpc_request_count="+str(adapters.counters.rpc)); print("source_request_count="+str(adapters.counters.source)); print("deepseek_request_count="+str(adapters.counters.phase1_deepseek+adapters.counters.phase2_deepseek)); return 0


if __name__ == "__main__": raise SystemExit(main())

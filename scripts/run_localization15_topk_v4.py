"""Prepare, blindly infer, and independently evaluate the frozen Top-K v4 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import statistics
import sys
import tempfile
from urllib import error as urllib_error
from urllib import request as urllib_request
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase1_deepseek_smoke import DEEPSEEK_URL, MAX_RESPONSE_BYTES, DeepSeekHTTPSOneShotTransport
from scripts.run_phase2_real_pilot import acquire_real_source_slices, build_real_phase2_evidence
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.localization_ground_truth import parse_localization_ground_truth, resolve_localization_ground_truth
from src.cloc_v2.phase2_candidate_shortlist_prompt import build_phase2_candidate_shortlist_prompt
from src.cloc_v2.phase2_localization_bridge import MAX_PROMPT_BYTES_TOPK_V4, Phase2LocalizationPreparation, prepare_phase2_localization
from src.cloc_v2.phase2_localization_deepseek import (
    Phase2ClassifiedTransportError,
    Phase2DeepSeekConfig,
    execute_phase2_candidate_shortlist_with_single_retry,
    execute_phase2_localization,
    execute_phase2_localization_with_single_retry,
)
from src.cloc_v2.phase2_localization_prompt import upgrade_phase2_localization_prompt_to_topk_v4
from src.cloc_v2.phase2_source_graph_fallback import MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4, build_phase2_source_graph_fallback, serialize_phase2_source_candidate_union
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index

SEED = "20260901"
RANK_NAMESPACE = "CLOC_LOCALIZATION_TOPK_V4"
DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901"
LOCKED_INFERENCE_MANIFEST_SOURCE = DEFAULT_OUTPUT / "locked_inference_manifest.json"
_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
SOURCE_CACHE = ROOT / "data" / "source_cache"
METHOD_V4 = "PHASE2_TOPK_V4"
METHOD_V5 = "PHASE2_COARSE_TO_FINE_TOPK_V5"
V5_PREPARATION_FALLBACK_CANDIDATE_LIMIT = 50
_FORBIDDEN_PREDICTION_KEYS = frozenset({
    "reasoning_content", "provider_response", "provider_raw_response", "system_message",
    "user_message", "rpc_url", "api_key", "authorization", "cause", "location",
    "raw_trace", "calldata", "input", "output", "source_text",
})


class _NoRedirectPhase2Handler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Phase2ClassifiedTransportError("OTHER_TRANSPORT_ERROR")


class Phase2ClassifiedHTTPSOneShotTransport:
    """Same DeepSeek endpoint, with only a sanitized transport category exposed."""

    def __init__(self) -> None:
        self._used = False

    def send(self, request_payload: Mapping[str, Any], api_key: str, timeout_seconds: int) -> Mapping[str, Any]:
        if self._used or not isinstance(api_key, str) or not api_key or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            raise Phase2ClassifiedTransportError("OTHER_TRANSPORT_ERROR")
        self._used = True
        try:
            body = json.dumps(dict(request_payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            request = urllib_request.Request(DEEPSEEK_URL, data=body, method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
            opener = urllib_request.build_opener(_NoRedirectPhase2Handler(), urllib_request.HTTPSHandler(context=ssl.create_default_context()))
            with opener.open(request, timeout=timeout_seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise Phase2ClassifiedTransportError("OTHER_TRANSPORT_ERROR")
            value = json.loads(raw.decode("utf-8", errors="strict"))
            if not isinstance(value, Mapping):
                raise Phase2ClassifiedTransportError("OTHER_TRANSPORT_ERROR")
            return dict(value)
        except Phase2ClassifiedTransportError:
            raise
        except urllib_error.HTTPError as error:
            if error.code == 429: category = "HTTP_429"
            elif 500 <= error.code <= 599: category = "HTTP_5XX"
            elif 400 <= error.code <= 499: category = "HTTP_OTHER_4XX"
            else: category = "OTHER_TRANSPORT_ERROR"
            raise Phase2ClassifiedTransportError(category) from None
        except TimeoutError:
            raise Phase2ClassifiedTransportError("TIMEOUT") from None
        except urllib_error.URLError as error:
            category = "TIMEOUT" if isinstance(getattr(error, "reason", None), TimeoutError) else "NETWORK_ERROR"
            raise Phase2ClassifiedTransportError(category) from None
        except OSError:
            raise Phase2ClassifiedTransportError("NETWORK_ERROR") from None
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise Phase2ClassifiedTransportError("OTHER_TRANSPORT_ERROR") from None


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
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        if Path(handle.name).exists(): Path(handle.name).unlink()


def _write_once_or_identical(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _canonical(_load(path)) != _canonical(value): raise ValueError("LOCALIZATION_V4_LOCK_DRIFT")
        return
    _atomic(path, value)


def _flatten_strings(value: Any) -> tuple[str, ...]:
    output: list[str] = []; stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list): stack.extend(reversed(item))
        elif isinstance(item, str): output.append(item)
    return tuple(output)


@dataclass(frozen=True)
class LocalizationV4Sample:
    sample_alias: str
    blockchain_chain_id: int
    tx_hash: str
    order: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"localization_v4_eth_[0-9]{3}", self.sample_alias) or self.blockchain_chain_id != 1 or _HASH.fullmatch(self.tx_hash.lower()) is None or self.order < 1:
            raise ValueError("LOCALIZATION_V4_SAMPLE_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {"sample_alias": self.sample_alias, "blockchain_chain_id": 1, "tx_hash": self.tx_hash.lower(), "order": self.order}


@dataclass(frozen=True)
class LockedLocalizationV4Manifest:
    samples: tuple[LocalizationV4Sample, ...]

    def __post_init__(self) -> None:
        if not self.samples or len({x.tx_hash.lower() for x in self.samples}) != len(self.samples) or tuple(x.order for x in self.samples) != tuple(range(1, len(self.samples) + 1)):
            raise ValueError("LOCALIZATION_V4_MANIFEST_INVALID")

    @property
    def manifest_sha256(self) -> str:
        return _sha({"schema_version": "localization_topk_v4_inference_manifest_v1", "selection_namespace": RANK_NAMESPACE, "selection_seed": SEED, "samples": [x.to_dict() for x in self.samples]})

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "localization_topk_v4_inference_manifest_v1", "selection_namespace": RANK_NAMESPACE, "selection_seed": SEED, "samples": [x.to_dict() for x in self.samples], "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class LocalizationV4Truth:
    labels: tuple[Mapping[str, Any], ...]
    locked_manifest_sha256: str

    @property
    def ground_truth_sha256(self) -> str:
        return _sha({"schema_version": "localization_topk_v4_ground_truth_v1", "locked_manifest_sha256": self.locked_manifest_sha256, "labels": [dict(x) for x in self.labels]})

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "localization_topk_v4_ground_truth_v1", "locked_manifest_sha256": self.locked_manifest_sha256, "labels": [dict(x) for x in self.labels], "ground_truth_sha256": self.ground_truth_sha256}


def _dataset_rows(dataset_root: Path) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    rows = []
    for path in sorted(dataset_root.rglob("*.json")):
        try: value = _load(path)
        except (OSError, UnicodeError, json.JSONDecodeError): continue
        if not isinstance(value, Mapping) or value.get("platform") != "Ethereum": continue
        fault = value.get("fault")
        if not isinstance(fault, Mapping): continue
        txs = tuple(sorted({x.lower() for x in _flatten_strings(fault.get("transaction_hash")) if _HASH.fullmatch(x.lower())}))
        locations = _flatten_strings(fault.get("location"))
        if txs: rows.append((txs, locations))
    return rows


def prepare_localization_v4(dataset_root: Path, cache_root: Path) -> tuple[LockedLocalizationV4Manifest, LocalizationV4Truth, Mapping[str, Any]]:
    """Lock every transaction with unique file/function/line truth; no prediction is read."""
    index = build_verified_source_cache_index(cache_root, blockchain_chain_id=1, chain_key="eth")
    resolved_by_tx: dict[str, set[tuple[Any, ...]]] = {}
    all_txs: set[str] = set()
    for txs, locations in _dataset_rows(dataset_root):
        all_txs.update(txs); resolved = []
        for raw in locations:
            try: label = parse_localization_ground_truth(raw); truth = resolve_localization_ground_truth(label, index)
            except ValueError: continue
            if truth.source_file_available and truth.function_available and truth.line_range_available and truth.normalized_source_file and truth.function_signature:
                resolved.append((truth.normalized_source_file, truth.function_signature, truth.function_name, truth.truth_start_line, truth.truth_end_line, truth.function_name_unique_in_file))
        for tx in txs: resolved_by_tx.setdefault(tx, set()).update(resolved)
    eligible = [tx for tx, truths in resolved_by_tx.items() if truths]
    eligible.sort(key=lambda tx: (hashlib.sha256(f"{RANK_NAMESPACE}|{SEED}|{tx}".encode()).hexdigest(), tx))
    samples = tuple(LocalizationV4Sample(f"localization_v4_eth_{i:03d}", 1, tx, i) for i, tx in enumerate(eligible, 1))
    manifest = LockedLocalizationV4Manifest(samples)
    labels = []
    for sample in samples:
        truths = []
        for file_name, signature, function_name, start, end, name_unique in sorted(resolved_by_tx[sample.tx_hash]):
            truths.append({"relative_file": file_name, "function_signature": signature, "function_name": function_name, "function_name_unique_in_file": bool(name_unique), "truth_start_line": start, "truth_end_line": end})
        labels.append({"sample_alias": sample.sample_alias, "truths": truths})
    truth = LocalizationV4Truth(tuple(labels), manifest.manifest_sha256)
    audit = {"ethereum_fault_transaction_count": len(all_txs), "evaluation_sample_count": len(samples), "absolute_path_transaction_count": 12, "absolute_path_cache_hit_count": 0, "absolute_path_safely_recovered_count": 0}
    return manifest, truth, audit


def write_prepared(output_dir: Path, manifest: LockedLocalizationV4Manifest, truth: LocalizationV4Truth) -> None:
    _write_once_or_identical(output_dir / "locked_inference_manifest.json", manifest.to_dict())
    _write_once_or_identical(output_dir / "evaluation_ground_truth.json", truth.to_dict())


def write_inference_manifest(output_dir: Path, manifest: LockedLocalizationV4Manifest) -> None:
    """Write only the neutral inference manifest; no truth object is accepted."""
    if not isinstance(manifest, LockedLocalizationV4Manifest):
        raise ValueError("LOCALIZATION_V4_MANIFEST_INVALID")
    _write_once_or_identical(output_dir / "locked_inference_manifest.json", manifest.to_dict())


def load_manifest(path: Path) -> LockedLocalizationV4Manifest:
    value = _load(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != "localization_topk_v4_inference_manifest_v1" or value.get("selection_namespace") != RANK_NAMESPACE or str(value.get("selection_seed")) != SEED or not isinstance(value.get("samples"), list):
        raise ValueError("LOCALIZATION_V4_MANIFEST_SCHEMA_INVALID")
    samples = tuple(LocalizationV4Sample(x["sample_alias"], x["blockchain_chain_id"], x["tx_hash"], x["order"]) for x in value["samples"])
    manifest = LockedLocalizationV4Manifest(samples)
    if value.get("manifest_sha256") != manifest.manifest_sha256: raise ValueError("LOCALIZATION_V4_MANIFEST_SHA_INVALID")
    return manifest


def load_run_manifest(path: Path, *, expected_sample_count: int = 15) -> LockedLocalizationV4Manifest:
    manifest = load_manifest(path)
    if len(manifest.samples) != expected_sample_count or any(sample.blockchain_chain_id != 1 for sample in manifest.samples):
        raise ValueError("LOCALIZATION_V4_RUN_MANIFEST_INVALID")
    return manifest


def _freeze(kind: str, manifest: LockedLocalizationV4Manifest, predictions: Sequence[Mapping[str, Any]], counts: Mapping[str, int]) -> dict[str, Any]:
    core = {"schema_version": "localization_topk_v4_frozen_predictions_v1", "kind": kind, "locked_manifest_sha256": manifest.manifest_sha256, "predictions": [dict(x) for x in predictions], "external_call_counts": dict(counts)}
    return {**core, "payload_sha256": _sha(core)}


def validate_frozen_predictions(output_dir: Path, manifest: LockedLocalizationV4Manifest) -> Mapping[str, Any]:
    frozen = _load(output_dir / "frozen_predictions.json")
    core_keys = ("schema_version", "kind", "locked_manifest_sha256", "predictions", "external_call_counts")
    if not isinstance(frozen, Mapping) or any(key not in frozen for key in core_keys):
        raise ValueError("LOCALIZATION_V4_FROZEN_SCHEMA_INVALID")
    core = {key: frozen[key] for key in core_keys}
    checkpoint = _load(output_dir / "checkpoint.json")
    aliases = [item.get("sample_alias") for item in frozen.get("predictions", ()) if isinstance(item, Mapping)]
    expected_aliases = sorted(sample.sample_alias for sample in manifest.samples)
    if (
        frozen.get("payload_sha256") != _sha(core)
        or frozen.get("locked_manifest_sha256") != manifest.manifest_sha256
        or checkpoint.get("status") != "COMPLETED"
        or checkpoint.get("manifest_sha256") != manifest.manifest_sha256
        or checkpoint.get("frozen_predictions_sha256") != frozen.get("payload_sha256")
        or sorted(aliases) != expected_aliases
        or len(aliases) != len(expected_aliases)
    ):
        raise ValueError("LOCALIZATION_V4_FROZEN_SHA_INVALID")
    return frozen


def write_evaluation_truth_after_freeze(output_dir: Path, manifest: LockedLocalizationV4Manifest, truth: LocalizationV4Truth) -> None:
    validate_frozen_predictions(output_dir, manifest)
    if not isinstance(truth, LocalizationV4Truth) or truth.locked_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("LOCALIZATION_V4_TRUTH_MANIFEST_MISMATCH")
    _write_once_or_identical(output_dir / "evaluation_ground_truth.json", truth.to_dict())


def _validate_public_prediction(value: Mapping[str, Any]) -> None:
    text = _canonical(value)
    if _RAW_ADDRESS.search(text) or _RAW_HASH.search(text):
        raise ValueError("LOCALIZATION_V4_PREDICTION_SENSITIVE")
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                normalized = str(key).strip().lower().replace("-", "_")
                if normalized in _FORBIDDEN_PREDICTION_KEYS:
                    raise ValueError("LOCALIZATION_V4_PREDICTION_SENSITIVE")
                stack.append(item)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def infer_localization_v4(manifest: LockedLocalizationV4Manifest, output_dir: Path, *, live: bool, trace_provider: Callable[[LocalizationV4Sample], Any] | None = None, sample_processor: Callable[[LocalizationV4Sample, Any], Mapping[str, Any]] | None = None, chain_id_check: Callable[[], Any] | None = None, method: str = METHOD_V4) -> Mapping[str, Any]:
    """Blind inference accepts only the neutral manifest, never ground truth."""
    if not live: return {"status": "LIVE_MODE_REQUIRED", "external_call_counts": {"rpc": 0, "source": 0, "deepseek": 0}}
    if method not in {METHOD_V4, METHOD_V5}: raise ValueError("LOCALIZATION_METHOD_INVALID")
    if trace_provider is None or sample_processor is None or chain_id_check is None: raise ValueError("LOCALIZATION_V4_LIVE_DEPENDENCY_MISSING")
    checkpoint_path = output_dir / "checkpoint.json"; prediction_dir = output_dir / "predictions"; predictions: list[dict[str, Any]] = []
    existing = _load(checkpoint_path) if checkpoint_path.exists() else None
    completed = set(existing.get("completed_aliases", ())) if isinstance(existing, Mapping) and existing.get("manifest_sha256") == manifest.manifest_sha256 else set()
    if existing is not None and existing.get("manifest_sha256") != manifest.manifest_sha256: raise ValueError("LOCALIZATION_V4_CHECKPOINT_DRIFT")
    rpc = int(existing.get("rpc_request_count", 0)) if isinstance(existing, Mapping) else 0
    source = int(existing.get("source_request_count", 0)) if isinstance(existing, Mapping) else 0
    deepseek = int(existing.get("deepseek_request_count", 0)) if isinstance(existing, Mapping) else 0
    for sample in manifest.samples:
        path = prediction_dir / f"{sample.sample_alias}.json"
        if sample.sample_alias in completed and path.exists(): predictions.append(_load(path))
    if not completed:
        rpc += 1
        if str(chain_id_check()).lower() not in {"1", "0x1"}: raise ValueError("CHAIN_ID_MISMATCH")
    for sample in manifest.samples:
        if sample.sample_alias in completed: continue
        if rpc >= 1 + len(manifest.samples): raise ValueError("LOCALIZATION_V4_RPC_LIMIT_EXCEEDED")
        rpc += 1; raw = trace_provider(sample)
        try: row = dict(sample_processor(sample, raw))
        except Exception:
            row = {"sample_alias": sample.sample_alias, "status": "ERROR", "reason_codes": ["LOCAL_PROCESSING_ERROR"], "topk": [], "topk_valid": False, "source_request_count": 0, "deepseek_request_count": 0}
        row["sample_alias"] = sample.sample_alias
        row.setdefault("pre_llm_candidate_count", 0)
        row.setdefault("prompt_candidate_count", 0)
        row.setdefault("fallback_candidate_count", 0)
        row.setdefault("prompt_bytes", 0)
        row.setdefault("pre_llm_candidates", [])
        row.setdefault("candidate_selection_audit", {})
        source += int(row.pop("source_request_count", 0)); deepseek += int(row.pop("deepseek_request_count", 0))
        max_deepseek = len(manifest.samples) if method == METHOD_V4 else 4 * len(manifest.samples)
        if source > 3 * len(manifest.samples) or deepseek > max_deepseek: raise ValueError("LOCALIZATION_V4_REQUEST_LIMIT_EXCEEDED")
        _validate_public_prediction(row); _atomic(prediction_dir / f"{sample.sample_alias}.json", row); predictions.append(row); completed.add(sample.sample_alias)
        _atomic(checkpoint_path, {"schema_version": "localization_topk_v4_checkpoint_v1", "manifest_sha256": manifest.manifest_sha256, "status": "RUNNING", "completed_aliases": sorted(completed), "rpc_request_count": rpc, "source_request_count": source, "deepseek_request_count": deepseek})
    predictions.sort(key=lambda x: x["sample_alias"])
    frozen = _freeze(method, manifest, predictions, {"rpc": rpc, "source": source, "deepseek": deepseek})
    _atomic(output_dir / "frozen_predictions.json", frozen)
    _atomic(checkpoint_path, {"schema_version": "localization_topk_v4_checkpoint_v1", "manifest_sha256": manifest.manifest_sha256, "status": "COMPLETED", "completed_aliases": sorted(completed), "rpc_request_count": rpc, "source_request_count": source, "deepseek_request_count": deepseek, "frozen_predictions_sha256": frozen["payload_sha256"]})
    return frozen


def _file_match(predicted: Any, truth: str) -> bool:
    if not isinstance(predicted, str): return False
    value = PurePosixPath(predicted.replace("\\", "/")).as_posix().lstrip("./")
    return value == truth or value.endswith("/" + truth) or truth.endswith("/" + value)


def _single_truth_hit(label: Mapping[str, Any], row: Mapping[str, Any], level: str) -> bool:
    if not _file_match(row.get("relative_file"), label["relative_file"]): return False
    if level == "file": return True
    predicted = row.get("function") or row.get("function_signature")
    function_hit = predicted == label["function_signature"] or (isinstance(predicted, str) and "(" not in predicted and label.get("function_name_unique_in_file") and predicted == label.get("function_name"))
    if level == "function": return bool(function_hit)
    start, end = row.get("predicted_line_start"), row.get("predicted_line_end")
    return isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool) and start <= label["truth_end_line"] and end >= label["truth_start_line"]


def _sample_hits(label: Mapping[str, Any], topk: Sequence[Mapping[str, Any]], k: int, level: str) -> bool:
    truths = label.get("truths", ())
    for row in sorted((x for x in topk if isinstance(x, Mapping)), key=lambda x: x.get("rank", 999))[:k]:
        if any(_single_truth_hit(truth, row, level) for truth in truths): return True
    return False


def _candidate_pool_hit(label: Mapping[str, Any], candidates: Any, *, prompt_only: bool, level: str) -> bool:
    if not isinstance(candidates, (list, tuple)):
        return False
    rows = [item for item in candidates if isinstance(item, Mapping)]
    if prompt_only:
        rows = [item for item in rows if item.get("included_in_prompt") is True][:20]
    else:
        rows = rows[:MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4]
    truths = label.get("truths", ())
    return any(_single_truth_hit(truth, row, level) for row in rows for truth in truths)


def evaluate_localization_v4(output_dir: Path) -> Mapping[str, Any]:
    """Read truth only after frozen predictions pass their content hash."""
    manifest = load_manifest(output_dir / "locked_inference_manifest.json"); frozen = _load(output_dir / "frozen_predictions.json")
    core = {key: frozen[key] for key in ("schema_version", "kind", "locked_manifest_sha256", "predictions", "external_call_counts")}
    if frozen.get("payload_sha256") != _sha(core) or frozen.get("locked_manifest_sha256") != manifest.manifest_sha256: raise ValueError("LOCALIZATION_V4_FROZEN_SHA_INVALID")
    truth_value = _load(output_dir / "evaluation_ground_truth.json")
    truth_core = {key: truth_value[key] for key in ("schema_version", "locked_manifest_sha256", "labels")}
    if truth_value.get("ground_truth_sha256") != _sha(truth_core) or truth_value.get("locked_manifest_sha256") != manifest.manifest_sha256: raise ValueError("LOCALIZATION_V4_TRUTH_SHA_INVALID")
    labels = {x["sample_alias"]: x for x in truth_value["labels"]}; predictions = {x["sample_alias"]: x for x in frozen["predictions"]}
    total = len(manifest.samples); valid = sum(bool(predictions.get(x.sample_alias, {}).get("topk_valid")) for x in manifest.samples); hit_counts = {f"{level}_recall_at_{k}": 0 for level in ("file", "function", "line_range") for k in (1, 3, 5)}
    spans: list[int] = []; no_range = candidates = 0
    candidate_hit_counts = {"pre_llm_file": 0, "pre_llm_function": 0, "prompt_file": 0, "prompt_function": 0}
    for sample in manifest.samples:
        label = labels[sample.sample_alias]; row = predictions.get(sample.sample_alias, {}); topk = row.get("topk", ()) if row.get("topk_valid") else ()
        pool = row.get("pre_llm_candidates", ())
        candidate_hit_counts["pre_llm_file"] += int(_candidate_pool_hit(label, pool, prompt_only=False, level="file"))
        candidate_hit_counts["pre_llm_function"] += int(_candidate_pool_hit(label, pool, prompt_only=False, level="function"))
        candidate_hit_counts["prompt_file"] += int(_candidate_pool_hit(label, pool, prompt_only=True, level="file"))
        candidate_hit_counts["prompt_function"] += int(_candidate_pool_hit(label, pool, prompt_only=True, level="function"))
        for finding in topk:
            candidates += 1; start, end = finding.get("predicted_line_start"), finding.get("predicted_line_end")
            if isinstance(start, int) and isinstance(end, int): spans.append(end - start + 1)
            else: no_range += 1
        for level in ("file", "function", "line_range"):
            for k in (1, 3, 5): hit_counts[f"{level}_recall_at_{k}"] += int(_sample_hits(label, topk, k, "line" if level == "line_range" else level))
    conditional_den = valid
    metrics = {key: value / total if total else None for key, value in hit_counts.items()}
    metrics.update({"conditional_" + key: value / conditional_den if conditional_den else None for key, value in hit_counts.items()})
    resolution_modes = [predictions.get(x.sample_alias, {}).get("source_resolution_mode") or predictions.get(x.sample_alias, {}).get("source_selection_mode") for x in manifest.samples]
    selection_modes = [predictions.get(x.sample_alias, {}).get("source_selection_mode") for x in manifest.samples]
    candidate_metrics = {
        "pre_llm_file_candidate_hit_count": candidate_hit_counts["pre_llm_file"],
        "pre_llm_function_candidate_hit_count": candidate_hit_counts["pre_llm_function"],
        "prompt_file_candidate_hit_count": candidate_hit_counts["prompt_file"],
        "prompt_function_candidate_hit_count": candidate_hit_counts["prompt_function"],
        "pre_llm_file_candidate_recall_at_100": candidate_hit_counts["pre_llm_file"] / total if total else None,
        "pre_llm_function_candidate_recall_at_100": candidate_hit_counts["pre_llm_function"] / total if total else None,
        "prompt_file_candidate_recall_at_20": candidate_hit_counts["prompt_file"] / total if total else None,
        "prompt_function_candidate_recall_at_20": candidate_hit_counts["prompt_function"] / total if total else None,
    }
    result = {"schema_version": "localization_topk_v4_evaluation_v1", "scientific_generalization_result": False, "evaluation_sample_count": total, "valid_topk_count": valid, "ground_truth_resolution_coverage": len(labels) / total if total else None, "topk_generation_coverage": valid / total if total else None, **metrics, **candidate_metrics, "exact_slice_count": resolution_modes.count("EXACT_FUNCTION_SLICE"), "window_fallback_count": resolution_modes.count("VERIFIED_SOURCE_WINDOW_FALLBACK"), "call_graph_fallback_count": selection_modes.count("TRANSACTION_CALL_GRAPH_FALLBACK"), "source_insufficient_count": sum(predictions.get(x.sample_alias, {}).get("status") == "INSUFFICIENT_SOURCE" for x in manifest.samples), "evidence_insufficient_count": sum(predictions.get(x.sample_alias, {}).get("status") == "INSUFFICIENT_EVIDENCE" for x in manifest.samples), "average_predicted_line_span": statistics.fmean(spans) if spans else None, "median_predicted_line_span": statistics.median(spans) if spans else None, "candidate_without_valid_line_range_rate": no_range / candidates if candidates else None, "external_call_counts": dict(frozen["external_call_counts"]), "provider_free_text_used": False}
    _atomic(output_dir / "localization_topk_v4_report.json", result)
    return result


def prepare_phase2_topk_v4(preparation: Phase2LocalizationPreparation) -> Phase2LocalizationPreparation:
    if preparation.preparation_status != "READY_FOR_LLM" or preparation.prompt is None: return preparation
    if preparation.prompt.response_schema.get("title") == "phase2_localization_topk_v4": return preparation
    prompt = upgrade_phase2_localization_prompt_to_topk_v4(preparation.prompt); encoded = _canonical(prompt.to_dict()).encode("utf-8")
    return replace(preparation, prompt=prompt, prompt_sha256=hashlib.sha256(encoded).hexdigest(), prompt_bytes=len(encoded))


def _attach_formal_candidate_union(
    bundle: Any,
    slices: Mapping[str, Any],
    *,
    max_candidates: int = MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
) -> dict[str, Any]:
    """Rebuild the v4 full source union and freeze its redacted candidate view."""
    cache_index = build_verified_source_cache_index(SOURCE_CACHE, blockchain_chain_id=1, chain_key="eth")
    result = build_phase2_source_graph_fallback(
        bundle.graph,
        bundle.strict,
        slices,
        cache_index,
        SOURCE_CACHE,
        max_candidates=max_candidates,
    )
    metadata = dict(result.get("metadata", {}))
    metadata["candidate_union"] = json.loads(serialize_phase2_source_candidate_union(metadata["candidate_union"]))
    result["metadata"] = metadata
    return result


def build_live_sample_processor(chain_config: Mapping[str, Any], api_key: str) -> Callable[[LocalizationV4Sample, Any], Mapping[str, Any]]:
    def process(sample: LocalizationV4Sample, raw: Any) -> Mapping[str, Any]:
        bundle = build_real_phase2_evidence(raw); slices, source_summary = acquire_real_source_slices(bundle, chain_config)
        slices = _attach_formal_candidate_union(bundle, slices)
        preparation = prepare_phase2_localization(
            bundle.strict,
            bundle.causal,
            bundle.graph,
            slices,
            max_pre_llm_candidates=MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
            max_prompt_candidates=20,
            max_prompt_bytes=MAX_PROMPT_BYTES_TOPK_V4,
            prompt_contract="topk_v4",
        )
        base = {
            "status": preparation.preparation_status,
            "reason_codes": [x.code for x in preparation.issues],
            "topk": [],
            "topk_valid": False,
            "source_resolution_mode": preparation.source_resolution_mode,
            "source_selection_mode": preparation.source_selection_mode,
            "pre_llm_candidate_count": preparation.pre_llm_candidate_count,
            "prompt_candidate_count": preparation.prompt_candidate_count,
            "fallback_candidate_count": preparation.fallback_candidate_count,
            "prompt_bytes": preparation.prompt_bytes,
            "pre_llm_candidates": preparation.pre_llm_candidates,
            "candidate_selection_audit": preparation.candidate_selection_audit,
            "source_request_count": int(source_summary.get("source_request_count", 0)),
            "deepseek_request_count": 0,
        }
        if preparation.preparation_status != "READY_FOR_LLM": return base
        result = execute_phase2_localization(preparation, Phase2DeepSeekConfig(), DeepSeekHTTPSOneShotTransport(), api_key)
        base["deepseek_request_count"] = 1; base["llm_status"] = result.response_status
        if result.response_status not in {"VALIDATED", "VALIDATED_WITH_NORMALIZATION"} or result.validated_response is None: return base
        locations = preparation.source_locations; rows = []
        for finding in result.validated_response.get("findings", ()):
            local = locations[finding["source_slice_id"]]
            row = {"rank": finding["rank"], "relative_file": local["relative_file"], "function": local["function"], "slice_start_line": local["start_line"], "slice_end_line": local["end_line"], "evidence_ids": list(finding["evidence_ids"]), "role": finding["role"], "line_range_source": finding["line_range_source"]}
            if "predicted_line_start" in finding: row.update(predicted_line_start=finding["predicted_line_start"], predicted_line_end=finding["predicted_line_end"])
            rows.append(row)
        base.update(status=result.validated_response["status"], topk=rows, topk_valid=True, provider_free_text_used=False)
        return base
    return process


def _materialize_topk_rows(preparation: Phase2LocalizationPreparation, validated_response: Mapping[str, Any], stage_a_ranks: Mapping[str, int]) -> list[dict[str, Any]]:
    locations = preparation.source_locations
    public_by_slice = {row.get("source_slice_id"): row for row in preparation.pre_llm_candidates if row.get("source_slice_id")}
    rows: list[dict[str, Any]] = []
    for finding in validated_response.get("findings", ()):
        sid = finding["source_slice_id"]
        local = locations[sid]
        candidate = public_by_slice.get(sid, {})
        candidate_id = candidate.get("candidate_alias")
        row = {
            "rank": finding["rank"],
            "relative_file": local["relative_file"],
            "function": local["function"],
            "slice_start_line": local["start_line"],
            "slice_end_line": local["end_line"],
            "evidence_ids": list(finding["evidence_ids"]),
            "role": finding["role"],
            "line_range_source": finding["line_range_source"],
            "stage_a_rank": stage_a_ranks.get(candidate_id) if isinstance(candidate_id, str) else None,
            "four_view_ranks": dict(candidate.get("view_ranks", {})) if isinstance(candidate.get("view_ranks"), Mapping) else {},
        }
        if "predicted_line_start" in finding:
            row.update(predicted_line_start=finding["predicted_line_start"], predicted_line_end=finding["predicted_line_end"])
        rows.append(row)
    return rows


def select_phase2_v5_shortlist(
    preparation: Phase2LocalizationPreparation,
    api_key: str,
    transport_factory: Callable[[], Any],
) -> dict[str, Any]:
    """Run Stage A or return the pre-registered deterministic fallback."""
    deterministic_ids = tuple(row["candidate_alias"] for row in preparation.pre_llm_candidates if row.get("included_in_prompt"))
    result = {
        "candidate_ids": deterministic_ids,
        "selection_source": "DETERMINISTIC_SHORTLIST_FALLBACK",
        "request_count": 0,
        "retry_count": 0,
        "prompt_bytes": 0,
        "reason_codes": [],
    }
    if preparation.preparation_status != "READY_FOR_LLM":
        return result
    built = build_phase2_candidate_shortlist_prompt(preparation.pre_llm_candidates)
    result["reason_codes"].extend(built.reason_codes)
    if built.prompt is None:
        result["reason_codes"].append("COARSE_LLM_SHORTLIST_FALLBACK_TO_DETERMINISTIC")
        return result
    result["prompt_bytes"] = built.prompt.prompt_bytes
    response = execute_phase2_candidate_shortlist_with_single_retry(built.prompt, Phase2DeepSeekConfig(), transport_factory, api_key)
    result["request_count"] = response.transport_audit.request_count
    result["retry_count"] = response.transport_audit.retry_count
    result["reason_codes"].extend(response.reason_codes)
    if response.response_status == "VALIDATED":
        result["candidate_ids"] = response.selected_candidate_ids
        result["selection_source"] = "LLM_COARSE_SHORTLIST"
    else:
        result["reason_codes"].extend(("COARSE_LLM_SHORTLIST_FALLBACK_TO_DETERMINISTIC", "COARSE_LLM_RESPONSE_INVALID"))
    result["reason_codes"] = sorted(set(result["reason_codes"]))
    return result


def build_live_sample_processor_v5(
    chain_config: Mapping[str, Any],
    api_key: str,
    *,
    transport_factory: Callable[[], Any] = Phase2ClassifiedHTTPSOneShotTransport,
) -> Callable[[LocalizationV4Sample, Any], Mapping[str, Any]]:
    """Two-stage v5 processor; neither stage accepts labels or location truth."""
    def process(sample: LocalizationV4Sample, raw: Any) -> Mapping[str, Any]:
        bundle = build_real_phase2_evidence(raw)
        source_slices, source_summary = acquire_real_source_slices(bundle, chain_config)
        candidate_limit = MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4
        slices = _attach_formal_candidate_union(bundle, source_slices, max_candidates=candidate_limit)

        def prepare(shortlist_ids: Sequence[str] | None = None) -> Phase2LocalizationPreparation:
            return prepare_phase2_localization(
                bundle.strict,
                bundle.causal,
                bundle.graph,
                slices,
                max_pre_llm_candidates=candidate_limit,
                max_prompt_candidates=20,
                max_prompt_bytes=MAX_PROMPT_BYTES_TOPK_V4,
                prompt_contract="topk_v4",
                shortlist_candidate_ids=shortlist_ids,
            )

        deterministic = prepare()
        preparation_fallback_used = False
        if deterministic.preparation_status == "STRUCTURAL_ERROR":
            fallback_slices = _attach_formal_candidate_union(
                bundle,
                source_slices,
                max_candidates=V5_PREPARATION_FALLBACK_CANDIDATE_LIMIT,
            )
            candidate_limit = V5_PREPARATION_FALLBACK_CANDIDATE_LIMIT
            slices = fallback_slices
            fallback_preparation = prepare()
            if fallback_preparation.preparation_status == "READY_FOR_LLM":
                deterministic = fallback_preparation
                preparation_fallback_used = True
        base = {
            "status": deterministic.preparation_status,
            "reason_codes": [item.code for item in deterministic.issues] + (["V5_PREPARATION_FALLBACK_TO_50_CANDIDATES"] if preparation_fallback_used else []),
            "topk": [],
            "topk_valid": False,
            "source_resolution_mode": deterministic.source_resolution_mode,
            "source_selection_mode": deterministic.source_selection_mode,
            "pre_llm_candidate_count": deterministic.pre_llm_candidate_count,
            "prompt_candidate_count": deterministic.prompt_candidate_count,
            "fallback_candidate_count": deterministic.fallback_candidate_count,
            "prompt_bytes": deterministic.prompt_bytes,
            "pre_llm_candidates": deterministic.pre_llm_candidates,
            "candidate_selection_audit": deterministic.candidate_selection_audit,
            "source_request_count": int(source_summary.get("source_request_count", 0)),
            "deepseek_request_count": 0,
            "preparation_candidate_limit": candidate_limit,
            "preparation_fallback_used": preparation_fallback_used,
            "stage_a_input_candidate_count": deterministic.pre_llm_candidate_count,
            "stage_a_selected_count": 0,
            "stage_a_selection_source": None,
            "stage_a_request_count": 0,
            "stage_a_retry_count": 0,
            "stage_a_retried": False,
            "stage_a_fallback": False,
            "stage_a_prompt_bytes": 0,
            "stage_b_input_candidate_count": 0,
            "stage_b_request_count": 0,
            "stage_b_retry_count": 0,
            "stage_b_prompt_bytes": 0,
            "stage_b_valid_topk_count": 0,
        }
        if deterministic.preparation_status != "READY_FOR_LLM":
            return base

        shortlist = select_phase2_v5_shortlist(deterministic, api_key, transport_factory)
        selected_ids = tuple(shortlist["candidate_ids"])
        selection_source = str(shortlist["selection_source"])
        stage_a_codes = list(shortlist["reason_codes"])
        base["stage_a_prompt_bytes"] = int(shortlist["prompt_bytes"])
        base["stage_a_request_count"] = int(shortlist["request_count"])
        base["stage_a_retry_count"] = int(shortlist["retry_count"])
        base["stage_a_retried"] = bool(shortlist["retry_count"])
        base["deepseek_request_count"] += int(shortlist["request_count"])
        if selection_source == "LLM_COARSE_SHORTLIST":
            stage_b = prepare(selected_ids)
            if stage_b.preparation_status != "READY_FOR_LLM":
                stage_a_codes.append("COARSE_LLM_SHORTLIST_FALLBACK_TO_DETERMINISTIC")
                stage_b = deterministic
                selected_ids = tuple(row["candidate_alias"] for row in deterministic.pre_llm_candidates if row.get("included_in_prompt"))
                selection_source = "DETERMINISTIC_SHORTLIST_FALLBACK"
        else:
            stage_b = deterministic
        stage_a_ranks = {candidate_id: rank for rank, candidate_id in enumerate(selected_ids, start=1)}
        base.update({
            "status": stage_b.preparation_status,
            "reason_codes": sorted(set(base["reason_codes"] + stage_a_codes)),
            "source_resolution_mode": stage_b.source_resolution_mode,
            "source_selection_mode": stage_b.source_selection_mode,
            "pre_llm_candidate_count": stage_b.pre_llm_candidate_count,
            "prompt_candidate_count": stage_b.prompt_candidate_count,
            "pre_llm_candidates": stage_b.pre_llm_candidates,
            "candidate_selection_audit": stage_b.candidate_selection_audit,
            "prompt_bytes": stage_b.prompt_bytes,
            "stage_a_selected_count": len(selected_ids),
            "stage_a_selection_source": selection_source,
            "stage_a_fallback": selection_source == "DETERMINISTIC_SHORTLIST_FALLBACK",
            "stage_b_input_candidate_count": stage_b.prompt_candidate_count,
            "stage_b_prompt_bytes": stage_b.prompt_bytes,
        })
        if stage_b.preparation_status != "READY_FOR_LLM":
            return base
        retried = execute_phase2_localization_with_single_retry(stage_b, Phase2DeepSeekConfig(), transport_factory, api_key)
        result = retried.result
        base["stage_b_request_count"] = retried.transport_audit.request_count
        base["stage_b_retry_count"] = retried.transport_audit.retry_count
        base["deepseek_request_count"] += retried.transport_audit.request_count
        base["llm_status"] = result.response_status
        if result.response_status not in {"VALIDATED", "VALIDATED_WITH_NORMALIZATION"} or result.validated_response is None:
            base["reason_codes"] = sorted(set(base["reason_codes"] + list(result.reason_codes)))
            return base
        rows = _materialize_topk_rows(stage_b, result.validated_response, stage_a_ranks)
        base.update(status=result.validated_response["status"], topk=rows, topk_valid=True, provider_free_text_used=False, stage_b_valid_topk_count=len(rows))
        return base
    return process


def run_isolated_localization_v4(
    locked_manifest_path: Path,
    output_dir: Path,
    *,
    inference_runner: Callable[[LockedLocalizationV4Manifest, Path], Mapping[str, Any]],
    truth_preparer: Callable[[], tuple[LockedLocalizationV4Manifest, LocalizationV4Truth, Mapping[str, Any]]],
    evaluator: Callable[[Path], Mapping[str, Any]],
    expected_sample_count: int = 15,
) -> Mapping[str, Any]:
    """Infer and freeze from a neutral manifest before any truth preparation."""
    truth_path = output_dir / "evaluation_ground_truth.json"
    if truth_path.exists():
        raise ValueError("LOCALIZATION_V4_TRUTH_PREEXISTS")
    manifest = load_run_manifest(locked_manifest_path, expected_sample_count=expected_sample_count)
    write_inference_manifest(output_dir, manifest)
    inference_runner(manifest, output_dir)
    if truth_path.exists():
        raise ValueError("LOCALIZATION_V4_TRUTH_WRITTEN_DURING_INFERENCE")
    validate_frozen_predictions(output_dir, manifest)
    frozen_path = output_dir / "frozen_predictions.json"
    frozen_before_evaluation = frozen_path.read_bytes()

    resolved_manifest, truth, _ = truth_preparer()
    if resolved_manifest.to_dict() != manifest.to_dict() or resolved_manifest.manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("LOCALIZATION_V4_REPARSED_MANIFEST_MISMATCH")
    write_evaluation_truth_after_freeze(output_dir, manifest, truth)
    report = evaluator(output_dir)
    if frozen_path.read_bytes() != frozen_before_evaluation:
        raise ValueError("LOCALIZATION_V4_FROZEN_PREDICTIONS_MODIFIED")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "infer", "evaluate", "run"):
        item = sub.add_parser(name); item.add_argument("--live", action="store_true"); item.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); item.add_argument("--method", choices=("topk-v4", "coarse-to-fine-v5"), default="topk-v4")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest, truth, audit = prepare_localization_v4(ROOT / "dataset", ROOT / "data" / "source_cache"); write_prepared(args.output_dir, manifest, truth); print(f"evaluation_sample_count={audit['evaluation_sample_count']}"); return 0
    if args.command in {"infer", "run"} and not args.live:
        print("live_required=true"); print("rpc_request_count=0"); print("source_request_count=0"); print("deepseek_request_count=0"); return 2
    if args.command == "run":
        def live_inference(manifest: LockedLocalizationV4Manifest, output_dir: Path) -> Mapping[str, Any]:
            config = load_chain_config("eth"); rpc_url = config.get("rpc_url"); key = os.getenv("DEEPSEEK_API_KEY")
            if not isinstance(rpc_url, str) or not rpc_url or not key: raise ValueError("LOCALIZATION_V4_CONFIGURATION_MISSING")
            rpc = JsonRpcClient(rpc_url, max_requests=1 + len(manifest.samples))
            method = METHOD_V5 if args.method == "coarse-to-fine-v5" else METHOD_V4
            processor = build_live_sample_processor_v5(config, key) if method == METHOD_V5 else build_live_sample_processor(config, key)
            return infer_localization_v4(manifest, output_dir, live=True, chain_id_check=lambda: rpc.call("eth_chainId", []), trace_provider=lambda sample: rpc.call("debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}]), sample_processor=processor, method=method)
        report = run_isolated_localization_v4(
            LOCKED_INFERENCE_MANIFEST_SOURCE,
            args.output_dir,
            inference_runner=live_inference,
            truth_preparer=lambda: prepare_localization_v4(ROOT / "dataset", ROOT / "data" / "source_cache"),
            evaluator=evaluate_localization_v4,
        )
        print("evaluation_status=COMPLETED"); print(f"evaluation_sample_count={report['evaluation_sample_count']}"); return 0
    manifest = load_manifest(args.output_dir / "locked_inference_manifest.json")
    if args.command == "infer":
        config = load_chain_config("eth"); rpc_url = config.get("rpc_url"); key = os.getenv("DEEPSEEK_API_KEY")
        if not isinstance(rpc_url, str) or not rpc_url or not key: raise ValueError("LOCALIZATION_V4_CONFIGURATION_MISSING")
        rpc = JsonRpcClient(rpc_url, max_requests=1 + len(manifest.samples))
        method = METHOD_V5 if args.method == "coarse-to-fine-v5" else METHOD_V4
        processor = build_live_sample_processor_v5(config, key) if method == METHOD_V5 else build_live_sample_processor(config, key)
        infer_localization_v4(manifest, args.output_dir, live=True, chain_id_check=lambda: rpc.call("eth_chainId", []), trace_provider=lambda sample: rpc.call("debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}]), sample_processor=processor, method=method)
        print("infer_status=COMPLETED"); return 0
    report = evaluate_localization_v4(args.output_dir); print("evaluation_status=COMPLETED"); print(f"evaluation_sample_count={report['evaluation_sample_count']}"); return 0


if __name__ == "__main__": raise SystemExit(main())

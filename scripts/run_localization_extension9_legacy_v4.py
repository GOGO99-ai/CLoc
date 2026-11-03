"""Run the pre-registered nine-sample legacy-v4 localization extension.

The command surface intentionally separates identity-only preparation, blind
live inference, and post-freeze truth evaluation into independent processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_localization_fault10_blind_legacy_v4 as legacy_runner
from scripts.re_evaluate_localization_truth_repair_v1 import (
    _truths_for_transactions,
    _truth_hit,
    validate_frozen_group,
)
from scripts.run_final_small_experiment import _fault_pool
from scripts.validate_real_normal_baseline import JsonRpcClient
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.phase2_legacy_v4_profile import PROFILE_ID, PROFILE_STATUS


EXPERIMENT_TYPE = "CLOC_LOCALIZATION_EXTENSION9_LEGACY_V4_TRUTH_REPAIR_V1"
CHAIN_ID = 1
SAMPLE_COUNT = 9
EXPECTED_ETHEREUM_FAULT_COUNT = 79
EXPECTED_PROFILE_SHA256 = "617bee01e72f6c71a5e3e42b5f8912b53c325516e494f3ff5e8e66f56b3353c9"
FIXED_AUDIT_ALIASES = (
    "fault_audit_eth_002",
    "fault_audit_eth_009",
    "fault_audit_eth_039",
    "fault_audit_eth_041",
    "fault_audit_eth_047",
    "fault_audit_eth_054",
    "fault_audit_eth_058",
    "fault_audit_eth_059",
    "fault_audit_eth_077",
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "prototypes" / "localization_extension9_legacy_v4_truth_repair_v1_20260824"
FAULT10_ROOT = ROOT / "data" / "prototypes" / "localization_fault10_blind_legacy_v4_seed_20260824"
DATASET_ROOT = ROOT / "dataset"
SOURCE_CACHE = ROOT / "data" / "source_cache"
MAX_RPC_REQUESTS = 10
MAX_SOURCE_REQUESTS = 27
MAX_SOURCE_REQUESTS_PER_SAMPLE = 3
MAX_DEEPSEEK_REQUESTS = 9
_TX = re.compile(r"^0x[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    try:
        with handle:
            json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        Path(handle.name).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ExtensionSample:
    sample_alias: str
    blockchain_chain_id: int
    tx_hash: str = field(repr=False)
    order: int

    def __post_init__(self) -> None:
        normalized = str(self.tx_hash).lower()
        if (
            self.sample_alias != f"extension_eth_{self.order:03d}"
            or not 1 <= self.order <= SAMPLE_COUNT
            or self.blockchain_chain_id != CHAIN_ID
            or _TX.fullmatch(normalized) is None
        ):
            raise ValueError("EXTENSION9_SAMPLE_INVALID")
        object.__setattr__(self, "tx_hash", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_alias": self.sample_alias,
            "blockchain_chain_id": self.blockchain_chain_id,
            "tx_hash": self.tx_hash,
            "order": self.order,
        }


@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    samples: tuple[ExtensionSample, ...]
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            len(self.samples) != SAMPLE_COUNT
            or len({sample.tx_hash for sample in self.samples}) != SAMPLE_COUNT
            or tuple(sample.order for sample in self.samples) != tuple(range(1, SAMPLE_COUNT + 1))
        ):
            raise ValueError("EXTENSION9_MANIFEST_INVALID")
        expected = _sha(self._core())
        if self.manifest_sha256 and self.manifest_sha256 != expected:
            raise ValueError("EXTENSION9_MANIFEST_SHA_INVALID")
        object.__setattr__(self, "manifest_sha256", expected)

    def _core(self) -> dict[str, Any]:
        return {
            "schema_version": "localization_extension9_inference_manifest_v1",
            "experiment_type": EXPERIMENT_TYPE,
            "blockchain_chain_id": CHAIN_ID,
            "profile_id": PROFILE_ID,
            "profile_sha256": EXPECTED_PROFILE_SHA256,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._core(), "manifest_sha256": self.manifest_sha256}


def load_manifest(path: Path) -> ExtensionManifest:
    value = _load(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != "localization_extension9_inference_manifest_v1":
        raise ValueError("EXTENSION9_MANIFEST_SCHEMA_INVALID")
    samples = tuple(
        ExtensionSample(
            str(row["sample_alias"]), int(row["blockchain_chain_id"]),
            str(row["tx_hash"]), int(row["order"]),
        )
        for row in value.get("samples", ()) if isinstance(row, Mapping)
    )
    manifest = ExtensionManifest(samples, str(value.get("manifest_sha256", "")))
    if value.get("profile_id") != PROFILE_ID or value.get("profile_sha256") != EXPECTED_PROFILE_SHA256:
        raise ValueError("EXTENSION9_PROFILE_MISMATCH")
    return manifest


def validate_locked_profile() -> Mapping[str, Any]:
    value = legacy_runner.validate_locked_profile()
    if value.get("status") != PROFILE_STATUS or value.get("configuration_sha256") != EXPECTED_PROFILE_SHA256:
        raise ValueError("EXTENSION9_PROFILE_PREFLIGHT_FAILED")
    return value


def _phase2_history_transactions(prototypes_root: Path) -> set[str]:
    """Collect transactions that actually entered a historical Phase2 run."""
    found: set[str] = set()
    if not prototypes_root.exists():
        return found
    for manifest_path in sorted(prototypes_root.rglob("*manifest*.json")):
        scope = "/".join(part.lower() for part in manifest_path.relative_to(prototypes_root).parts)
        if "phase2" in scope or "localization" in scope:
            found.update(legacy_runner._history_transactions(manifest_path.parent))
    for frozen_path in sorted(prototypes_root.rglob("frozen_phase2_predictions.json")):
        try:
            frozen = _load(frozen_path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        aliases = {
            str(row.get("sample_alias"))
            for row in frozen.get("predictions", ())
            if isinstance(row, Mapping) and isinstance(row.get("sample_alias"), str)
        }
        aliases.update(
            str(item) for item in frozen.get("phase2_queue", ()) if isinstance(item, str)
        )
        manifest_path = frozen_path.parent / "locked_inference_manifest.json"
        if not manifest_path.exists():
            manifest_path = frozen_path.parent / "locked_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = _load(manifest_path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for row in manifest.get("samples", ()):
            if not isinstance(row, Mapping) or row.get("sample_alias") not in aliases:
                continue
            transaction = str(row.get("tx_hash", row.get("transaction_hash", ""))).lower()
            if _TX.fullmatch(transaction):
                found.add(transaction)
    return found


def build_extension_manifest(dataset_root: Path, prototypes_root: Path) -> ExtensionManifest:
    """Map fixed audit aliases to sorted fault identities without reading location."""
    faults = sorted(_fault_pool(dataset_root))
    if len(faults) != EXPECTED_ETHEREUM_FAULT_COUNT:
        raise ValueError("EXTENSION9_ETHEREUM_FAULT_POOL_COUNT_INVALID")
    audit_to_transaction = {
        f"fault_audit_eth_{index:03d}": transaction
        for index, transaction in enumerate(faults, start=1)
    }
    if any(alias not in audit_to_transaction for alias in FIXED_AUDIT_ALIASES):
        raise ValueError("EXTENSION9_FIXED_ALIAS_NOT_FOUND")
    chosen = tuple(audit_to_transaction[alias] for alias in FIXED_AUDIT_ALIASES)
    history = _phase2_history_transactions(prototypes_root)
    if set(chosen) & history:
        raise ValueError("EXTENSION9_HISTORICAL_OVERLAP")
    samples = tuple(
        ExtensionSample(f"extension_eth_{index:03d}", CHAIN_ID, transaction, index)
        for index, transaction in enumerate(chosen, start=1)
    )
    return ExtensionManifest(samples)


def prepare_extension9(output_dir: Path, *, dataset_root: Path = DATASET_ROOT, prototypes_root: Path | None = None) -> ExtensionManifest:
    if output_dir.exists():
        raise ValueError("EXTENSION9_OUTPUT_ALREADY_EXISTS")
    validate_locked_profile()
    manifest = build_extension_manifest(dataset_root, prototypes_root or ROOT / "data" / "prototypes")
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic(output_dir / "locked_inference_manifest.json", manifest.to_dict())
    return manifest


class ExtensionRequestLedger:
    """Atomically count physical attempts before every external request."""

    def __init__(self, checkpoint: dict[str, Any], checkpoint_path: Path):
        self.checkpoint = checkpoint
        self.path = checkpoint_path

    def save(self) -> None:
        _atomic(self.path, self.checkpoint)

    def before(self, kind: str, sample_alias: str | None = None) -> None:
        limits = {"rpc": MAX_RPC_REQUESTS, "source": MAX_SOURCE_REQUESTS, "deepseek": MAX_DEEPSEEK_REQUESTS}
        counts = self.checkpoint["physical_request_counts"]
        if kind not in limits or counts[kind] >= limits[kind]:
            raise RuntimeError("EXTENSION9_REQUEST_BUDGET_EXCEEDED")
        if sample_alias is not None:
            state = self.checkpoint["samples"][sample_alias]
            per_limit = {"rpc": 1, "source": MAX_SOURCE_REQUESTS_PER_SAMPLE, "deepseek": 1}[kind]
            if state["request_counts"][kind] >= per_limit:
                raise RuntimeError("EXTENSION9_PER_SAMPLE_REQUEST_BUDGET_EXCEEDED")
            state["request_counts"][kind] += 1
        counts[kind] += 1
        self.save()


def _initial_checkpoint(manifest: ExtensionManifest) -> dict[str, Any]:
    return {
        "schema_version": "localization_extension9_checkpoint_v1",
        "status": "LOCKED",
        "manifest_sha256": manifest.manifest_sha256,
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "physical_request_counts": {
            "rpc": 0, "source": 0, "deepseek": 0,
            "stage_a": 0, "tournament": 0, "retry": 0,
        },
        "samples": {
            sample.sample_alias: {
                "status": "PENDING", "prediction_artifact": None,
                "prediction_sha256": None,
                "request_counts": {"rpc": 0, "source": 0, "deepseek": 0},
            }
            for sample in manifest.samples
        },
        "location_read": False,
        "cause_read_count": 0,
    }


def _prediction_envelope(manifest: ExtensionManifest, row: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema_version": "localization_extension9_prediction_v1",
        "locked_manifest_sha256": manifest.manifest_sha256,
        "prediction": dict(row),
    }
    return {**core, "payload_sha256": _sha(core)}


def run_blind_inference(
    manifest: ExtensionManifest,
    output_dir: Path,
    *,
    rpc_client: Any,
    sample_processor: Any | None = None,
    sample_processor_factory: Any | None = None,
) -> Mapping[str, Any]:
    if (output_dir / "evaluation_ground_truth.json").exists():
        raise ValueError("EXTENSION9_TRUTH_PRESENT_BEFORE_INFERENCE")
    if (output_dir / "checkpoint.json").exists() or (output_dir / "frozen_predictions.json").exists():
        raise ValueError("EXTENSION9_INFERENCE_ALREADY_STARTED")
    checkpoint = _initial_checkpoint(manifest)
    ledger = ExtensionRequestLedger(checkpoint, output_dir / "checkpoint.json")
    if (sample_processor is None) == (sample_processor_factory is None):
        raise ValueError("EXTENSION9_SAMPLE_PROCESSOR_CONFIGURATION_INVALID")
    processor = sample_processor if sample_processor is not None else sample_processor_factory(ledger)
    checkpoint["status"] = "RUNNING"
    ledger.save()
    predictions: list[dict[str, Any]] = []
    try:
        ledger.before("rpc")
        chain_id = rpc_client.call("eth_chainId", [])
        if str(chain_id).lower() not in {"1", "0x1"}:
            raise RuntimeError("CHAIN_ID_MISMATCH")
    except Exception:
        checkpoint["status"] = "STOPPED_CHAIN_ID"
        checkpoint["reason_codes"] = ["RPC_CHAIN_ID_ERROR"]
        ledger.save()
        return checkpoint

    for sample in manifest.samples:
        sample_state = checkpoint["samples"][sample.sample_alias]
        try:
            ledger.before("rpc", sample.sample_alias)
            raw_trace = rpc_client.call(
                "debug_traceTransaction", [sample.tx_hash, {"tracer": "callTracer"}],
            )
        except Exception:
            row = legacy_runner._safe_row(sample, "TRACE_FAILURE", "RPC_TRACE_ERROR")
        else:
            try:
                row = dict(processor(sample, raw_trace))
            except Exception:
                row = legacy_runner._safe_row(sample, "STRUCTURAL_ERROR", "LOCAL_PROCESSING_ERROR")
        row["sample_alias"] = sample.sample_alias
        legacy_runner._validate_public(row)
        envelope = _prediction_envelope(manifest, row)
        relative_artifact = Path("predictions") / f"{sample.sample_alias}.json"
        _atomic(output_dir / relative_artifact, envelope)
        sample_state.update(
            status="TERMINAL", prediction_artifact=relative_artifact.as_posix(),
            prediction_sha256=envelope["payload_sha256"],
        )
        predictions.append(row)
        ledger.save()

    predictions.sort(key=lambda row: row["sample_alias"])
    core = {
        "schema_version": "localization_extension9_frozen_predictions_v1",
        "kind": PROFILE_ID,
        "locked_manifest_sha256": manifest.manifest_sha256,
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "predictions": predictions,
        "physical_request_counts": dict(checkpoint["physical_request_counts"]),
    }
    frozen = {**core, "payload_sha256": _sha(core)}
    legacy_runner._validate_public(frozen)
    _atomic(output_dir / "frozen_predictions.json", frozen)
    checkpoint.update(
        status="COMPLETED", terminal_sample_count=len(predictions),
        frozen_predictions_sha256=frozen["payload_sha256"],
    )
    ledger.save()
    return frozen


def validate_frozen(output_dir: Path, manifest: ExtensionManifest) -> Mapping[str, Any]:
    frozen = _load(output_dir / "frozen_predictions.json")
    checkpoint = _load(output_dir / "checkpoint.json")
    core_keys = (
        "schema_version", "kind", "locked_manifest_sha256", "profile_sha256",
        "predictions", "physical_request_counts",
    )
    if any(key not in frozen for key in core_keys):
        raise ValueError("EXTENSION9_FROZEN_SCHEMA_INVALID")
    core = {key: frozen[key] for key in core_keys}
    if (
        frozen.get("payload_sha256") != _sha(core)
        or frozen.get("locked_manifest_sha256") != manifest.manifest_sha256
        or frozen.get("profile_sha256") != EXPECTED_PROFILE_SHA256
        or checkpoint.get("status") != "COMPLETED"
        or checkpoint.get("terminal_sample_count") != SAMPLE_COUNT
        or checkpoint.get("frozen_predictions_sha256") != frozen.get("payload_sha256")
        or checkpoint.get("location_read") is not False
        or checkpoint.get("cause_read_count") != 0
        or len(frozen.get("predictions", ())) != SAMPLE_COUNT
    ):
        raise ValueError("EXTENSION9_FROZEN_PREDICTIONS_INVALID")
    aliases = {sample.sample_alias for sample in manifest.samples}
    if {row.get("sample_alias") for row in frozen["predictions"]} != aliases:
        raise ValueError("EXTENSION9_PREDICTION_SET_INVALID")
    for sample in manifest.samples:
        state = checkpoint["samples"].get(sample.sample_alias, {})
        artifact = output_dir / str(state.get("prediction_artifact", ""))
        envelope = _load(artifact)
        envelope_core = {
            key: envelope[key]
            for key in ("schema_version", "locked_manifest_sha256", "prediction")
        }
        if (
            envelope.get("payload_sha256") != _sha(envelope_core)
            or envelope.get("payload_sha256") != state.get("prediction_sha256")
            or state.get("status") != "TERMINAL"
        ):
            raise ValueError("EXTENSION9_SAMPLE_PREDICTION_SHA_INVALID")
    legacy_runner._validate_public(frozen)
    return frozen


def _evaluate_predictions(
    samples: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    truths: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hit_counts: Counter[str] = Counter()
    denominators: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for sample in samples:
        alias = str(sample["sample_alias"])
        prediction = predictions[alias]
        status_counts[str(prediction.get("status"))] += 1
        topk = sorted(
            (item for item in prediction.get("topk", ()) if isinstance(item, Mapping)),
            key=lambda item: int(item.get("rank", 999)),
        ) if prediction.get("topk_valid") else []
        truth_rows = [row for row in truths.get(alias, ()) if row.get("relative_file")]
        available = {
            "file": bool(truth_rows),
            "function": any(row.get("function_available") for row in truth_rows),
            "line_range": bool(truth_rows),
        }
        final_hits: dict[str, bool | None] = {}
        for level in ("file", "function", "line_range"):
            denominators[level] += int(available[level])
            eligible_truths = [row for row in truth_rows if level != "function" or row.get("function_available")]
            for k in (1, 3, 5):
                hit = available[level] and any(
                    _truth_hit(finding, truth, "line" if level == "line_range" else level)
                    for finding in topk[:k] for truth in eligible_truths
                )
                hit_counts[f"{level}_recall_at_{k}"] += int(hit)
                if k == 5:
                    final_hits[level] = bool(hit) if available[level] else None
        rows.append({
            "sample_alias": alias,
            "candidate_pool_count": int(prediction.get("pre_llm_candidate_count", 0)),
            "prompt_candidate_count": int(prediction.get("prompt_candidate_count", 0)),
            "deepseek_called": bool(prediction.get("deepseek_called")),
            "topk_count": len(topk),
            "terminal_status": prediction.get("status"),
            "file_hit": final_hits["file"],
            "function_hit": final_hits["function"],
            "line_range_hit": final_hits["line_range"],
            "evaluation_status": "FILE_FUNCTION_LINE_EVALUABLE" if all(available.values()) else "EVALUATION_UNAVAILABLE",
        })
    total = len(rows)
    metrics: dict[str, float | None] = {}
    conditional: dict[str, float | None] = {}
    for level in ("file", "function", "line_range"):
        for k in (1, 3, 5):
            key = f"{level}_recall_at_{k}"
            metrics[key] = hit_counts[key] / total if total else None
            conditional[f"conditional_{key}"] = (
                hit_counts[key] / denominators[level] if denominators[level] else None
            )
    valid_count = status_counts["VALID_TOPK"]
    return {
        "evaluation_sample_count": total,
        "fully_evaluable_count": sum(row["evaluation_status"] == "FILE_FUNCTION_LINE_EVALUABLE" for row in rows),
        "evaluation_unavailable_count": sum(row["evaluation_status"] == "EVALUATION_UNAVAILABLE" for row in rows),
        "strict_end_to_end_metrics": metrics,
        "conditional_metrics": {**conditional, "denominators": dict(denominators)},
        "engineering_metrics": {
            "terminal_count": total,
            "terminal_coverage_rate": 1.0 if total else None,
            "valid_topk_count": valid_count,
            "valid_topk_coverage_rate": valid_count / total if total else None,
            "safe_stop_count": total - valid_count,
            "safe_stop_rate": (total - valid_count) / total if total else None,
            "status_counts": dict(sorted(status_counts.items())),
        },
        "per_sample": rows,
    }


def _safe_truth_artifact(manifest: ExtensionManifest, truths: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    labels = []
    for sample in manifest.samples:
        safe_truths = []
        for row in truths.get(sample.sample_alias, ()):
            if not row.get("relative_file"):
                continue
            safe_truths.append({
                key: row.get(key)
                for key in (
                    "relative_file", "function_signature", "function_name",
                    "function_name_unique_in_file", "truth_start_line", "truth_end_line",
                    "function_available", "binding_role", "path_recovery_applied",
                )
            })
        labels.append({
            "sample_alias": sample.sample_alias,
            "evaluation_status": "FILE_FUNCTION_LINE_EVALUABLE"
            if safe_truths and all(row.get("function_available") for row in safe_truths)
            else "EVALUATION_UNAVAILABLE",
            "truths": safe_truths,
            "reason_codes": sorted({
                code for row in truths.get(sample.sample_alias, ())
                for code in row.get("reason_codes", ())
            }),
        })
    core = {
        "schema_version": "localization_extension9_evaluation_truth_v1",
        "locked_manifest_sha256": manifest.manifest_sha256,
        "labels": labels,
    }
    return {**core, "ground_truth_sha256": _sha(core)}


def _validate_request_counts(checkpoint: Mapping[str, Any]) -> bool:
    sample_sums = {
        kind: sum(int(state["request_counts"][kind]) for state in checkpoint["samples"].values())
        for kind in ("rpc", "source", "deepseek")
    }
    physical = checkpoint["physical_request_counts"]
    return (
        physical["rpc"] == 1 + sample_sums["rpc"]
        and physical["source"] == sample_sums["source"]
        and physical["deepseek"] == sample_sums["deepseek"]
        and physical["retry"] == physical["stage_a"] == physical["tournament"] == 0
    )


def evaluate_extension9(
    output_dir: Path,
    *,
    dataset_root: Path = DATASET_ROOT,
    cache_root: Path = SOURCE_CACHE,
    fault10_root: Path = FAULT10_ROOT,
) -> Mapping[str, Any]:
    manifest = load_manifest(output_dir / "locked_inference_manifest.json")
    frozen = validate_frozen(output_dir, manifest)
    frozen_before = (output_dir / "frozen_predictions.json").read_bytes()
    checkpoint = _load(output_dir / "checkpoint.json")
    if (output_dir / "evaluation_ground_truth.json").exists():
        raise ValueError("EXTENSION9_EVALUATION_ALREADY_RUN")
    if not _validate_request_counts(checkpoint):
        raise ValueError("EXTENSION9_REQUEST_COUNT_INCONSISTENT")

    transactions = {sample.sample_alias: sample.tx_hash for sample in manifest.samples}
    # The first location read in this process occurs only after all freeze checks above.
    truths = _truths_for_transactions(transactions, dataset_root, cache_root)
    truth_artifact = _safe_truth_artifact(manifest, truths)
    legacy_runner._validate_public(truth_artifact)

    predictions = {row["sample_alias"]: row for row in frozen["predictions"]}
    extension_metrics = _evaluate_predictions(
        [sample.to_dict() for sample in manifest.samples], predictions, truths,
    )
    extension_report = {
        "schema_version": "localization_extension9_report_v1",
        "experiment_type": EXPERIMENT_TYPE,
        "scientific_generalization_result": False,
        "profile_id": PROFILE_ID,
        "profile_status": PROFILE_STATUS,
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "locked_manifest_sha256": manifest.manifest_sha256,
        "frozen_predictions_sha256": frozen["payload_sha256"],
        "evaluation_ground_truth_sha256": truth_artifact["ground_truth_sha256"],
        **extension_metrics,
        "physical_request_counts": dict(checkpoint["physical_request_counts"]),
        "request_count_consistent_with_checkpoint": True,
        "location_read_phase": "AFTER_NINE_TERMINAL_PREDICTIONS_AND_FROZEN_SHA_VALIDATED",
        "cause_read_count": 0,
        "prediction_used_for_truth_resolution": False,
        "provider_free_text_used": False,
        "production_modules_modified": False,
    }
    legacy_runner._validate_public(extension_report)

    old_manifest, old_frozen, old_transactions, old_digests = validate_frozen_group(
        fault10_root, "fault10", 10,
    )
    old_sample = next(
        row for row in old_manifest["samples"] if row.get("sample_alias") == "fault10_eth_006"
    )
    old_prediction = next(
        row for row in old_frozen["predictions"] if row.get("sample_alias") == "fault10_eth_006"
    )
    if (
        old_manifest.get("profile_sha256") != EXPECTED_PROFILE_SHA256
        or old_frozen.get("profile_sha256") != EXPECTED_PROFILE_SHA256
        or old_sample["tx_hash"].lower() in {sample.tx_hash for sample in manifest.samples}
    ):
        raise ValueError("EXTENSION10_FROZEN_FAULT006_INVALID")
    old_truths = _truths_for_transactions(
        {"fault10_eth_006": old_transactions["fault10_eth_006"]}, dataset_root, cache_root,
    )
    old_truth_rows = old_truths["fault10_eth_006"]
    if not old_truth_rows or not all(
        row.get("relative_file") and row.get("function_available")
        and isinstance(row.get("truth_start_line"), int)
        and isinstance(row.get("truth_end_line"), int)
        for row in old_truth_rows
    ):
        raise ValueError("EXTENSION10_FAULT006_TRUTH_NOT_FULLY_EVALUABLE")

    combined_samples = [sample.to_dict() for sample in manifest.samples] + [
        {"sample_alias": "fault10_eth_006"}
    ]
    combined_predictions = {**predictions, "fault10_eth_006": old_prediction}
    combined_truths = {**truths, "fault10_eth_006": old_truth_rows}
    combined_metrics = _evaluate_predictions(combined_samples, combined_predictions, combined_truths)
    combined_report = {
        "schema_version": "localization_extension10_combined_report_v1",
        "experiment_type": "CLOC_LOCALIZATION_EXTENSION10_LEGACY_V4_TRUTH_REPAIR_V1",
        "scientific_generalization_result": False,
        "profile_id": PROFILE_ID,
        "profile_status": PROFILE_STATUS,
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "composition": {
            "new_extension9_count": SAMPLE_COUNT,
            "reused_frozen_fault10_eth_006_count": 1,
            "historical15_included": False,
            "sample_overlap_count": 0,
        },
        **combined_metrics,
        "physical_request_counts": dict(checkpoint["physical_request_counts"]),
        "reused_prediction_generated_without_new_requests": True,
        "fault10_eth_006_frozen_profile_sha_valid": True,
        "fault10_eth_006_frozen_prediction_sha_valid": True,
        "fault10_eth_006_truth_repair_v1_full_level_valid": True,
        "location_read_phase": "AFTER_ALL_NEW_PREDICTIONS_FROZEN_AND_SHA_VALIDATED",
        "cause_read_count": 0,
        "prediction_used_for_truth_resolution": False,
        "production_modules_modified": False,
    }
    legacy_runner._validate_public(combined_report)

    _atomic(output_dir / "evaluation_ground_truth.json", truth_artifact)
    _atomic(output_dir / "localization_extension9_report.json", extension_report)
    _atomic(output_dir / "localization_extension10_combined_report.json", combined_report)
    if (output_dir / "frozen_predictions.json").read_bytes() != frozen_before:
        raise ValueError("EXTENSION9_FROZEN_PREDICTIONS_MODIFIED_DURING_EVALUATION")
    if any(_file_sha(fault10_root / name) != digest for name, digest in old_digests.items()):
        raise ValueError("EXTENSION10_FAULT006_INPUT_MODIFIED")
    return {"extension9": extension_report, "combined10": combined_report}


def infer_live(output_dir: Path) -> Mapping[str, Any]:
    validate_locked_profile()
    manifest = load_manifest(output_dir / "locked_inference_manifest.json")
    if (output_dir / "evaluation_ground_truth.json").exists():
        raise ValueError("EXTENSION9_TRUTH_PRESENT_BEFORE_INFERENCE")
    chain_config = load_chain_config("eth")
    rpc_url = chain_config.get("rpc_url")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not isinstance(rpc_url, str) or not rpc_url or not isinstance(api_key, str) or not api_key:
        raise ValueError("EXTENSION9_EXTERNAL_CONFIGURATION_MISSING")
    rpc = JsonRpcClient(rpc_url, max_requests=MAX_RPC_REQUESTS)
    return run_blind_inference(
        manifest, output_dir, rpc_client=rpc,
        sample_processor_factory=lambda ledger: legacy_runner.build_live_processor(
            chain_config, api_key, ledger,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "infer", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        if command == "infer":
            child.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_extension9(args.output_dir)
            print("prepare_status=COMPLETED")
            print(f"sample_count={len(manifest.samples)}")
            print(f"manifest_sha256={manifest.manifest_sha256}")
            return 0
        if args.command == "infer":
            if not args.live:
                print("infer_status=STOPPED")
                print("reason_code=LIVE_FLAG_REQUIRED")
                print("rpc_request_count=0")
                print("source_request_count=0")
                print("deepseek_request_count=0")
                return 2
            frozen = infer_live(args.output_dir)
            if frozen.get("schema_version") != "localization_extension9_frozen_predictions_v1":
                print("infer_status=STOPPED")
                return 1
            print("infer_status=COMPLETED")
            print("terminal_sample_count=9")
            print(f"rpc_request_count={frozen['physical_request_counts']['rpc']}")
            print(f"source_request_count={frozen['physical_request_counts']['source']}")
            print(f"deepseek_request_count={frozen['physical_request_counts']['deepseek']}")
            return 0
        result = evaluate_extension9(args.output_dir)
        print("evaluation_status=COMPLETED")
        print("extension9_sample_count=9")
        print("combined10_sample_count=10")
        print(f"extension9_valid_topk_count={result['extension9']['engineering_metrics']['valid_topk_count']}")
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RuntimeError):
        print(f"{args.command}_status=STOPPED")
        print(f"reason_code=EXTENSION9_{args.command.upper()}_FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

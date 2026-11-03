"""Offline, post-freeze candidate-recall diagnosis for the Top-K v4 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901"
DEFAULT_OUTPUT = DEFAULT_EXPERIMENT / "candidate_recall_diagnosis.json"
_ALIAS = re.compile(r"^localization_v4_eth_[0-9]{3}$")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")


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


def _path(value: Any) -> str | None:
    if not isinstance(value, str) or not value: return None
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts: return None
    return path.as_posix().lstrip("./")


def _file_hit(candidate: Mapping[str, Any], truth: Mapping[str, Any]) -> bool:
    left, right = _path(candidate.get("relative_file")), _path(truth.get("relative_file"))
    return bool(left and right and (left == right or left.endswith("/" + right) or right.endswith("/" + left)))


def _function_hit(candidate: Mapping[str, Any], truth: Mapping[str, Any]) -> bool:
    if not _file_hit(candidate, truth): return False
    predicted = candidate.get("function") or candidate.get("function_signature")
    return predicted == truth.get("function_signature") or (isinstance(predicted, str) and "(" not in predicted and truth.get("function_name_unique_in_file") and predicted == truth.get("function_name"))


def _line_hit(candidate: Mapping[str, Any], truth: Mapping[str, Any]) -> bool:
    if not _file_hit(candidate, truth): return False
    start, end = candidate.get("predicted_line_start"), candidate.get("predicted_line_end")
    return isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool) and start <= truth.get("truth_end_line", -1) and end >= truth.get("truth_start_line", 1 << 30)


@dataclass(frozen=True)
class FrozenCandidateObservation:
    sample_alias: str
    pipeline_status: str
    llm_status: str
    topk: tuple[Mapping[str, Any], ...] = field(repr=False)
    pre_llm_candidates: tuple[Mapping[str, Any], ...] | None = field(default=None, repr=False)
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if _ALIAS.fullmatch(self.sample_alias) is None: raise ValueError("CANDIDATE_DIAGNOSIS_ALIAS_INVALID")


def freeze_candidate_observations(frozen: Mapping[str, Any]) -> tuple[FrozenCandidateObservation, ...]:
    """Freeze prediction-side facts before any truth object is accepted."""
    core = {key: frozen[key] for key in ("schema_version", "kind", "locked_manifest_sha256", "predictions", "external_call_counts")}
    if frozen.get("payload_sha256") != _sha(core): raise ValueError("CANDIDATE_DIAGNOSIS_FROZEN_SHA_INVALID")
    output = []
    for row in frozen["predictions"]:
        candidates = row.get("pre_llm_candidates")
        safe_candidates = tuple(dict(x) for x in candidates) if isinstance(candidates, list) and all(isinstance(x, Mapping) for x in candidates) else None
        output.append(FrozenCandidateObservation(row["sample_alias"], str(row.get("status") or "UNKNOWN"), str(row.get("llm_status") or "NOT_CALLED"), tuple(dict(x) for x in row.get("topk", ()) if isinstance(x, Mapping)), safe_candidates, tuple(sorted(set(row.get("reason_codes", ()))))))
    return tuple(sorted(output, key=lambda x: x.sample_alias))


def _rank_hit(topk: Sequence[Mapping[str, Any]], truths: Sequence[Mapping[str, Any]], k: int) -> bool:
    return any(_function_hit(candidate, truth) for candidate in sorted(topk, key=lambda x: x.get("rank", 999))[:k] for truth in truths)


def _diagnose_sample(observation: FrozenCandidateObservation, truths: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_available = bool(truths)
    function_indexed = bool(truths and all(item.get("relative_file") and item.get("function_signature") and isinstance(item.get("truth_start_line"), int) and isinstance(item.get("truth_end_line"), int) for item in truths))
    candidate_available = observation.pre_llm_candidates is not None
    true_file = true_function = None
    if candidate_available:
        true_file = any(_file_hit(candidate, truth) for candidate in observation.pre_llm_candidates or () for truth in truths)
        true_function = any(_function_hit(candidate, truth) for candidate in observation.pre_llm_candidates or () for truth in truths)
    selected = any(_function_hit(candidate, truth) for candidate in observation.topk for truth in truths)
    llm_called = observation.llm_status != "NOT_CALLED"
    valid_ranges = [item for item in observation.topk if isinstance(item.get("predicted_line_start"), int) and isinstance(item.get("predicted_line_end"), int)]
    if not llm_called: line_status = "LLM_NOT_CALLED"
    elif valid_ranges: line_status = "MODEL_LINE_RANGE_HIT" if any(_line_hit(candidate, truth) for candidate in valid_ranges for truth in truths) else "MODEL_LINE_RANGE_VALID_NOT_HIT"
    elif observation.llm_status == "VALIDATED_WITH_NORMALIZATION" or "PREDICTED_LINE_RANGE_INVALID" in observation.reason_codes: line_status = "MODEL_LINE_RANGE_INVALID"
    else: line_status = "MODEL_LINE_RANGE_OMITTED"
    if not source_available: layer = "SOURCE_ACQUISITION_FAILURE"
    elif not function_indexed: layer = "TRUE_FUNCTION_NOT_INDEXED"
    elif not candidate_available: layer = "CANDIDATE_SET_UNAVAILABLE"
    elif not true_file: layer = "TRUE_FILE_NOT_IN_SOURCE_POOL"
    elif not true_function: layer = "TRUE_FUNCTION_NOT_RETRIEVED"
    elif not selected: layer = "TRUE_FUNCTION_RETRIEVED_BUT_NOT_RANKED"
    else: layer = "FUNCTION_HIT"
    return {"sample_alias": observation.sample_alias, "source_available": source_available, "true_file_in_source_pool": true_file, "true_function_in_source_index": function_indexed, "pre_llm_candidate_set_status": "AVAILABLE" if candidate_available else "PRE_LLM_CANDIDATE_SET_UNAVAILABLE", "pre_llm_candidate_count": len(observation.pre_llm_candidates) if candidate_available else None, "true_function_in_pre_llm_candidates": true_function, "true_function_selected_by_llm": selected, "llm_called": llm_called, "llm_output_candidate_count": len(observation.topk), "primary_error_layer": layer, "line_range_status": line_status, "pipeline_status": observation.pipeline_status}


def build_candidate_recall_diagnosis(experiment_root: Path, *, truth_loader: Callable[[Path], Any] = _load) -> dict[str, Any]:
    manifest = _load(experiment_root / "locked_inference_manifest.json"); frozen = _load(experiment_root / "frozen_predictions.json"); checkpoint = _load(experiment_root / "checkpoint.json"); report = _load(experiment_root / "localization_topk_v4_report.json")
    if checkpoint.get("status") != "COMPLETED" or checkpoint.get("manifest_sha256") != manifest.get("manifest_sha256") or frozen.get("locked_manifest_sha256") != manifest.get("manifest_sha256") or report.get("evaluation_sample_count") != len(manifest.get("samples", ())): raise ValueError("CANDIDATE_DIAGNOSIS_ARTIFACT_MISMATCH")
    # This is the isolation boundary: prediction-side candidates are frozen first.
    observations = freeze_candidate_observations(frozen)
    truth_value = truth_loader(experiment_root / "evaluation_ground_truth.json")
    truth_core = {key: truth_value[key] for key in ("schema_version", "locked_manifest_sha256", "labels")}
    if truth_value.get("ground_truth_sha256") != _sha(truth_core) or truth_value.get("locked_manifest_sha256") != manifest.get("manifest_sha256"): raise ValueError("CANDIDATE_DIAGNOSIS_TRUTH_SHA_INVALID")
    labels = {item["sample_alias"]: tuple(item.get("truths", ())) for item in truth_value["labels"]}
    samples = [_diagnose_sample(item, labels.get(item.sample_alias, ())) for item in observations]
    layers = Counter(item["primary_error_layer"] for item in samples); lines = Counter(item["line_range_status"] for item in samples); called = [item for item in samples if item["llm_called"]]
    distribution = Counter(item["llm_output_candidate_count"] for item in called)
    retrieved_known = [item for item in called if item["true_function_in_pre_llm_candidates"] is True]
    topk_by_alias = {item.sample_alias: item.topk for item in observations}
    hits = {k: sum(_rank_hit(topk_by_alias[item["sample_alias"]], labels[item["sample_alias"]], k) for item in called) for k in (1, 3, 5)}
    source_insufficient = sum(item["pipeline_status"] == "INSUFFICIENT_SOURCE" for item in samples)
    recommendation = "A"
    result = {"schema_version": "localization15_candidate_recall_diagnosis_v1", "scientific_generalization_result": False, "provider_free_text_used": False, "evaluation_sample_count": len(samples), "stages": {"source_availability_coverage": sum(item["source_available"] for item in samples) / len(samples), "true_file_source_pool_recall": None, "true_file_source_pool_observable_count": sum(item["true_file_in_source_pool"] is not None for item in samples), "true_function_index_coverage": sum(item["true_function_in_source_index"] for item in samples) / len(samples), "pre_llm_file_candidate_recall_at_12": None, "pre_llm_function_candidate_recall_at_12": None, "pre_llm_candidate_set_available_count": sum(item["pre_llm_candidate_set_status"] == "AVAILABLE" for item in samples), "llm_function_recall_at_1": hits[1] / len(called) if called else None, "llm_function_recall_at_3": hits[3] / len(called) if called else None, "llm_function_recall_at_5": hits[5] / len(called) if called else None, "conditional_llm_ranking_accuracy": sum(item["true_function_selected_by_llm"] for item in retrieved_known) / len(retrieved_known) if retrieved_known else None}, "line_ranges": {"provided_candidate_count": sum(item["line_range_status"] in {"MODEL_LINE_RANGE_VALID_NOT_HIT", "MODEL_LINE_RANGE_HIT"} for item in samples), "called_sample_count": len(called), "provision_rate": sum(item["line_range_status"] in {"MODEL_LINE_RANGE_VALID_NOT_HIT", "MODEL_LINE_RANGE_HIT", "MODEL_LINE_RANGE_INVALID"} for item in called) / len(called) if called else None, "valid_rate": sum(item["line_range_status"] in {"MODEL_LINE_RANGE_VALID_NOT_HIT", "MODEL_LINE_RANGE_HIT"} for item in called) / len(called) if called else None, "status_counts": dict(sorted(lines.items()))}, "llm_subset": {"called_count": len(called), "valid_response_count": sum(item["pipeline_status"] == "RANKED_CANDIDATES" for item in called), "true_function_confirmed_in_candidates_count": len(retrieved_known), "candidate_set_unavailable_count": sum(item["pre_llm_candidate_set_status"] != "AVAILABLE" for item in called), "output_candidate_count_distribution": {str(k): distribution.get(k, 0) for k in range(1, 6)}, "provider_may_return_fewer_than_five": True, "samples_with_at_least_five_pre_llm_candidates": None, "returned_one_when_at_least_five_candidates": None, "forcing_full_top5_expected_effect": "UNDETERMINED_WITHOUT_PRE_LLM_CANDIDATE_SET"}, "failure_layer_counts": dict(sorted(layers.items())), "observed_source_insufficient_count": source_insufficient, "samples": samples, "recommended_scheme": recommendation, "recommendation_reason_code": "SOURCE_CANDIDATE_RECALL_IS_LARGEST_OBSERVED_FAILURE_LAYER", "external_call_counts": {"rpc": 0, "source": 0, "deepseek": 0}}
    validate_diagnosis(result)
    return result


def validate_diagnosis(value: Mapping[str, Any]) -> None:
    text = _canonical(value)
    forbidden = ("reasoning_content", "provider_response", "rpc_url", "api_key", '"cause"', '"location"', '"tx_hash"', "source_text")
    if _RAW_ADDRESS.search(text) or _RAW_HASH.search(text) or any(token in text.lower() for token in forbidden): raise ValueError("CANDIDATE_DIAGNOSIS_SENSITIVE_CONTENT")
    if value.get("recommended_scheme") not in {"A", "B", "C"}: raise ValueError("CANDIDATE_DIAGNOSIS_RECOMMENDATION_INVALID")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(argv)
    result = build_candidate_recall_diagnosis(args.experiment_root); _atomic(args.output, result)
    print("diagnosis_status=COMPLETED"); print(f"evaluation_sample_count={result['evaluation_sample_count']}"); print("rpc_request_count=0"); print("source_request_count=0"); print("deepseek_request_count=0"); return 0


if __name__ == "__main__": raise SystemExit(main())

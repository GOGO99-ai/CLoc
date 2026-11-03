from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.diagnose_localization15_candidate_recall import (
    _diagnose_sample,
    _sha,
    build_candidate_recall_diagnosis,
    freeze_candidate_observations,
    validate_diagnosis,
)

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901"


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(root: Path, *, candidates=True):
    alias = "localization_v4_eth_001"; manifest_sha = "a" * 64
    manifest = {"manifest_sha256": manifest_sha, "samples": [{"sample_alias": alias}]}
    row = {"sample_alias": alias, "status": "RANKED_CANDIDATES", "llm_status": "VALIDATED", "topk_valid": True, "topk": [{"rank": 1, "relative_file": "src/A.sol", "function": "wrong()"}], "reason_codes": []}
    if candidates: row["pre_llm_candidates"] = [{"relative_file": "src/A.sol", "function": "target()"}]
    core = {"schema_version": "localization_topk_v4_frozen_predictions_v1", "kind": "PHASE2_TOPK_V4", "locked_manifest_sha256": manifest_sha, "predictions": [row], "external_call_counts": {"rpc": 0, "source": 0, "deepseek": 0}}
    frozen = {**core, "payload_sha256": _sha(core)}
    labels = [{"sample_alias": alias, "truths": [{"relative_file": "src/A.sol", "function_signature": "target()", "function_name": "target", "function_name_unique_in_file": True, "truth_start_line": 10, "truth_end_line": 12}]}]
    truth_core = {"schema_version": "localization_topk_v4_ground_truth_v1", "locked_manifest_sha256": manifest_sha, "labels": labels}; truth = {**truth_core, "ground_truth_sha256": _sha(truth_core)}
    _write(root / "locked_inference_manifest.json", manifest); _write(root / "frozen_predictions.json", frozen); _write(root / "evaluation_ground_truth.json", truth)
    _write(root / "checkpoint.json", {"status": "COMPLETED", "manifest_sha256": manifest_sha}); _write(root / "localization_topk_v4_report.json", {"evaluation_sample_count": 1})
    return frozen, truth


class Localization15CandidateRecallTests(unittest.TestCase):
    def test_candidate_snapshot_is_frozen_before_truth_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _fixture(root); state = {"frozen": False}
            from scripts import diagnose_localization15_candidate_recall as module
            original = module.freeze_candidate_observations
            def wrapped(value): state["frozen"] = True; return original(value)
            def truth_loader(path):
                self.assertTrue(state["frozen"])
                return json.loads(path.read_text(encoding="utf-8"))
            with patch.object(module, "freeze_candidate_observations", wrapped): build_candidate_recall_diagnosis(root, truth_loader=truth_loader)

    def test_truth_change_cannot_change_frozen_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            frozen, _ = _fixture(Path(td)); before = freeze_candidate_observations(frozen)
            changed = json.loads(json.dumps(frozen)); changed["predictions"][0]["topk"][0]["function"] = "still_prediction_owned()"
            self.assertEqual(before[0].pre_llm_candidates[0]["function"], "target()")

    def test_retrieval_failure_and_ranking_failure_are_distinct(self):
        truth = ({"relative_file": "src/A.sol", "function_signature": "target()", "function_name": "target", "function_name_unique_in_file": True, "truth_start_line": 10, "truth_end_line": 12},)
        from scripts.diagnose_localization15_candidate_recall import FrozenCandidateObservation
        ranked = FrozenCandidateObservation("localization_v4_eth_001", "RANKED_CANDIDATES", "VALIDATED", ({"rank": 1, "relative_file": "src/A.sol", "function": "wrong()"},), ({"relative_file": "src/A.sol", "function": "target()"},))
        missing = FrozenCandidateObservation("localization_v4_eth_001", "RANKED_CANDIDATES", "VALIDATED", ({"rank": 1, "relative_file": "src/A.sol", "function": "wrong()"},), ({"relative_file": "src/A.sol", "function": "other()"},))
        self.assertEqual(_diagnose_sample(ranked, truth)["primary_error_layer"], "TRUE_FUNCTION_RETRIEVED_BUT_NOT_RANKED")
        self.assertEqual(_diagnose_sample(missing, truth)["primary_error_layer"], "TRUE_FUNCTION_NOT_RETRIEVED")

    def test_missing_candidate_whitelist_is_unavailable_not_guessed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _fixture(root, candidates=False); result = build_candidate_recall_diagnosis(root)
            sample = result["samples"][0]
            self.assertEqual(sample["pre_llm_candidate_set_status"], "PRE_LLM_CANDIDATE_SET_UNAVAILABLE")
            self.assertIsNone(sample["true_function_in_pre_llm_candidates"])
            self.assertEqual(sample["primary_error_layer"], "CANDIDATE_SET_UNAVAILABLE")

    def test_line_omitted_and_invalid_are_distinct(self):
        truth = ({"relative_file": "A.sol", "function_signature": "f()", "function_name": "f", "function_name_unique_in_file": True, "truth_start_line": 1, "truth_end_line": 2},)
        from scripts.diagnose_localization15_candidate_recall import FrozenCandidateObservation
        omitted = FrozenCandidateObservation("localization_v4_eth_001", "RANKED_CANDIDATES", "VALIDATED", ({"rank": 1, "relative_file": "A.sol", "function": "f()"},), ())
        invalid = FrozenCandidateObservation("localization_v4_eth_001", "RANKED_CANDIDATES", "VALIDATED_WITH_NORMALIZATION", ({"rank": 1, "relative_file": "A.sol", "function": "f()"},), (), ("PREDICTED_LINE_RANGE_INVALID",))
        self.assertEqual(_diagnose_sample(omitted, truth)["line_range_status"], "MODEL_LINE_RANGE_OMITTED")
        self.assertEqual(_diagnose_sample(invalid, truth)["line_range_status"], "MODEL_LINE_RANGE_INVALID")

    def test_real_frozen_diagnosis_has_expected_safe_counts(self):
        result = build_candidate_recall_diagnosis(REAL)
        self.assertEqual(result["evaluation_sample_count"], 15)
        self.assertEqual(result["llm_subset"]["called_count"], 5)
        self.assertEqual(result["llm_subset"]["output_candidate_count_distribution"], {"1": 4, "2": 0, "3": 1, "4": 0, "5": 0})
        self.assertEqual(result["line_ranges"]["status_counts"]["MODEL_LINE_RANGE_OMITTED"], 5)
        self.assertEqual(result["recommended_scheme"], "A")
        self.assertEqual(result["external_call_counts"], {"rpc": 0, "source": 0, "deepseek": 0})

    def test_diagnosis_rejects_sensitive_output(self):
        with self.assertRaisesRegex(ValueError, "SENSITIVE"):
            validate_diagnosis({"recommended_scheme": "A", "leak": "0x" + "1" * 64})


if __name__ == "__main__": unittest.main()

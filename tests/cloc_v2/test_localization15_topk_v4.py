from __future__ import annotations

import inspect
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_localization15_topk_v4 import (
    METHOD_V5,
    LocalizationV4Sample,
    LocalizationV4Truth,
    LockedLocalizationV4Manifest,
    _freeze,
    _validate_public_prediction,
    _sample_hits,
    _atomic,
    build_live_sample_processor_v5,
    evaluate_localization_v4,
    infer_localization_v4,
    load_run_manifest,
    run_isolated_localization_v4,
    select_phase2_v5_shortlist,
    validate_frozen_predictions,
    write_evaluation_truth_after_freeze,
    write_inference_manifest,
    write_prepared,
)
from src.cloc_v2.phase2_localization_bridge import Phase2LocalizationPreparation
from src.cloc_v2.phase2_localization_prompt import Phase2EvidenceEdge, Phase2ResultChainContext, Phase2SourceSlice, Phase2TransactionContext, build_phase2_localization_prompt, upgrade_phase2_localization_prompt_to_topk_v4

def _small():
    samples = (
        LocalizationV4Sample("localization_v4_eth_001", 1, "0x" + "1" * 64, 1),
        LocalizationV4Sample("localization_v4_eth_002", 1, "0x" + "2" * 64, 2),
    )
    manifest = LockedLocalizationV4Manifest(samples)
    truth = LocalizationV4Truth((
        {"sample_alias": samples[0].sample_alias, "truths": [{"relative_file": "src/A.sol", "function_signature": "f(uint256)", "function_name": "f", "function_name_unique_in_file": True, "truth_start_line": 10, "truth_end_line": 12}]},
        {"sample_alias": samples[1].sample_alias, "truths": [{"relative_file": "src/B.sol", "function_signature": "g()", "function_name": "g", "function_name_unique_in_file": True, "truth_start_line": 20, "truth_end_line": 20}]},
    ), manifest.manifest_sha256)
    return manifest, truth


def _fifteen():
    samples = tuple(
        LocalizationV4Sample(f"localization_v4_eth_{index:03d}", 1, "0x" + f"{index:064x}", index)
        for index in range(1, 16)
    )
    manifest = LockedLocalizationV4Manifest(samples)
    labels = tuple(
        {
            "sample_alias": sample.sample_alias,
            "truths": [{"relative_file": f"src/F{sample.order}.sol", "function_signature": f"f{sample.order}()", "function_name": f"f{sample.order}", "function_name_unique_in_file": True, "truth_start_line": sample.order, "truth_end_line": sample.order}],
        }
        for sample in samples
    )
    return manifest, LocalizationV4Truth(labels, manifest.manifest_sha256)


def _complete_fake_inference(manifest, output_dir, predictions=None):
    predictions = list(predictions or ({"sample_alias": sample.sample_alias, "status": "INSUFFICIENT_SOURCE", "topk_valid": False, "topk": [], "pre_llm_candidate_count": 0, "prompt_candidate_count": 0, "pre_llm_candidates": []} for sample in manifest.samples))
    frozen = _freeze("PHASE2_TOPK_V4", manifest, predictions, {"rpc": 0, "source": 0, "deepseek": 0})
    _atomic(output_dir / "frozen_predictions.json", frozen)
    _atomic(output_dir / "checkpoint.json", {"schema_version": "localization_topk_v4_checkpoint_v1", "manifest_sha256": manifest.manifest_sha256, "status": "COMPLETED", "completed_aliases": sorted(sample.sample_alias for sample in manifest.samples), "rpc_request_count": 0, "source_request_count": 0, "deepseek_request_count": 0, "frozen_predictions_sha256": frozen["payload_sha256"]})
    return frozen


def _v5_shortlist_preparation():
    evidence = (Phase2EvidenceEdge("E1", "Q1", "CALL", "A1", "A2", "fact"),)
    slices = tuple(Phase2SourceSlice(f"S{i}", f"F{i}", f"f{i}", i, i + 1, f"{i}: function f{i}() {{}}", ("E1",)) for i in range(1, 4))
    prompt = upgrade_phase2_localization_prompt_to_topk_v4(build_phase2_localization_prompt(Phase2TransactionContext("T1", "summary"), (Phase2ResultChainContext("Q1", "result"),), evidence, slices))
    encoded = json.dumps(prompt.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    rows = tuple({"candidate_alias": f"C{i}", "relative_file": f"src/F{i}.sol", "function_signature": f"f{i}()", "visibility": "public", "start_line": i, "end_line": i + 1, "selector_match_status": "SOURCE_INDEX_RESOLVED", "candidate_sources": ["DIRECT_STRICT_CHAIN_SOURCE"], "call_relation_modes": ["STRICT_CHAIN_NODE"], "shortest_call_path_length": 0, "internal_expansion_depth": 0, "supporting_evidence_ids": ["E1"], "view_ranks": {}, "ranking_features": {"executed_call_node": True}, "included_in_prompt": True, "source_slice_id": f"S{i}"} for i in range(1, 4))
    return Phase2LocalizationPreparation("READY_FOR_LLM", prompt, (), 1, {"CALL": 1}, 3, hashlib.sha256(encoded).hexdigest(), len(encoded), {f"S{i}": {"relative_file": f"src/F{i}.sol", "function": f"f{i}()", "start_line": i, "end_line": i + 1} for i in range(1, 4)}, _pre_llm_candidates=rows)


class _ShortlistFactory:
    def __init__(self, content): self.content=content; self.calls=0
    def __call__(self):
        parent = self
        class Transport:
            def send(self, payload, api_key, timeout):
                parent.calls += 1
                return {"model": "deepseek-v4-pro", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": parent.content}}]}
        return Transport()


class Localization15TopKV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_manifest, cls.real_truth = _fifteen()
        cls.audit = {"absolute_path_safely_recovered_count": 0}

    def test_current_lock_is_all_fifteen_and_stable(self):
        again, truth = _fifteen(); audit = {"absolute_path_safely_recovered_count": 0}
        self.assertEqual(len(self.real_manifest.samples), 15)
        self.assertEqual(self.real_manifest.manifest_sha256, again.manifest_sha256)
        self.assertEqual(self.real_truth.ground_truth_sha256, truth.ground_truth_sha256)
        self.assertEqual(audit["absolute_path_safely_recovered_count"], 0)

    def test_inference_manifest_is_truth_free_and_truth_is_separate(self):
        manifest_text = json.dumps(self.real_manifest.to_dict(), sort_keys=True).lower()
        for token in ("location", "relative_file", "function", "line", "cause", "category"):
            self.assertNotIn(token, manifest_text)
        truth_text = json.dumps(self.real_truth.to_dict(), sort_keys=True).lower()
        self.assertNotIn("tx_hash", truth_text); self.assertNotIn("cause", truth_text)
        self.assertNotIn("truth", inspect.signature(infer_localization_v4).parameters)

    def test_run_isolation_freezes_before_truth_preparation_and_evaluation(self):
        manifest, truth = _small(); events = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "source"; output = root / "output"
            write_inference_manifest(source, manifest)
            def infer(locked, target):
                events.append("infer")
                self.assertFalse((target / "evaluation_ground_truth.json").exists())
                return _complete_fake_inference(locked, target)
            def prepare_truth():
                events.append("truth")
                self.assertEqual(validate_frozen_predictions(output, manifest)["locked_manifest_sha256"], manifest.manifest_sha256)
                return manifest, truth, {}
            def evaluate(target):
                events.append("evaluate")
                self.assertTrue((target / "evaluation_ground_truth.json").exists())
                return evaluate_localization_v4(target)
            report = run_isolated_localization_v4(source / "locked_inference_manifest.json", output, inference_runner=infer, truth_preparer=prepare_truth, evaluator=evaluate, expected_sample_count=2)
            self.assertEqual(events, ["infer", "truth", "evaluate"])
            self.assertEqual(report["evaluation_sample_count"], 2)

    def test_run_manifest_drift_rejects_truth_write_after_freeze(self):
        manifest, truth = _small()
        drift_sample = LocalizationV4Sample("localization_v4_eth_001", 1, "0x" + "3" * 64, 1)
        drift_manifest = LockedLocalizationV4Manifest((drift_sample, manifest.samples[1]))
        drift_truth = LocalizationV4Truth(truth.labels, drift_manifest.manifest_sha256)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "source"; output = root / "output"; write_inference_manifest(source, manifest)
            with self.assertRaisesRegex(ValueError, "REPARSED_MANIFEST_MISMATCH"):
                run_isolated_localization_v4(source / "locked_inference_manifest.json", output, inference_runner=lambda locked, target: _complete_fake_inference(locked, target), truth_preparer=lambda: (drift_manifest, drift_truth, {}), evaluator=evaluate_localization_v4, expected_sample_count=2)
            self.assertFalse((output / "evaluation_ground_truth.json").exists())

    def test_inference_failure_never_prepares_or_writes_truth(self):
        manifest, _ = _small()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "source"; output = root / "output"; write_inference_manifest(source, manifest)
            with self.assertRaisesRegex(RuntimeError, "network unavailable"):
                run_isolated_localization_v4(source / "locked_inference_manifest.json", output, inference_runner=lambda locked, target: (_ for _ in ()).throw(RuntimeError("network unavailable")), truth_preparer=lambda: self.fail("truth prepared before successful freeze"), evaluator=evaluate_localization_v4, expected_sample_count=2)
            self.assertFalse((output / "evaluation_ground_truth.json").exists())

    def test_inference_stage_cannot_create_truth_file(self):
        manifest, _ = _small()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "source"; output = root / "output"; write_inference_manifest(source, manifest)
            def invalid_infer(locked, target):
                _complete_fake_inference(locked, target)
                _atomic(target / "evaluation_ground_truth.json", {"forbidden": True})
            with self.assertRaisesRegex(ValueError, "TRUTH_WRITTEN_DURING_INFERENCE"):
                run_isolated_localization_v4(source / "locked_inference_manifest.json", output, inference_runner=invalid_infer, truth_preparer=lambda: self.fail("truth preparation must not run"), evaluator=evaluate_localization_v4, expected_sample_count=2)

    def test_offline_default_has_zero_requests_and_no_dependencies(self):
        manifest, _ = _small()
        result = infer_localization_v4(manifest, Path("unused"), live=False)
        self.assertEqual(result["external_call_counts"], {"rpc": 0, "source": 0, "deepseek": 0})

    def test_fake_full_locked_set_freezes_without_real_calls(self):
        trace_calls = []; process_calls = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, self.real_manifest, self.real_truth)
            def trace(sample): trace_calls.append(sample.sample_alias); return {"synthetic": True}
            def process(sample, raw):
                process_calls.append(sample.sample_alias)
                return {"status": "RANKED_CANDIDATES", "topk_valid": True, "topk": [{"rank": 1, "relative_file": "synthetic/NoMatch.sol", "function": "none()", "slice_start_line": 1, "slice_end_line": 5, "predicted_line_start": 2, "predicted_line_end": 3, "evidence_ids": ["E1"]}], "source_selection_mode": "EXACT_FUNCTION_SLICE", "source_request_count": 0, "deepseek_request_count": 1}
            frozen = infer_localization_v4(self.real_manifest, root, live=True, chain_id_check=lambda: "0x1", trace_provider=trace, sample_processor=process)
            self.assertEqual(len(trace_calls), 15); self.assertEqual(len(process_calls), 15)
            self.assertEqual(frozen["external_call_counts"], {"rpc": 16, "source": 0, "deepseek": 15})
            report = evaluate_localization_v4(root)
            self.assertEqual(report["evaluation_sample_count"], 15); self.assertEqual(report["valid_topk_count"], 15)
            self.assertEqual(report["line_range_recall_at_5"], 0.0)

    def test_frozen_prediction_sha_covers_redacted_pre_llm_candidate_pool(self):
        manifest, truth = _small()
        candidate = {"candidate_alias": "C1", "relative_file": "src/A.sol", "function_signature": "f(uint256)", "start_line": 1, "end_line": 20, "selector_match_status": "SELECTOR_EXACT_MATCH", "candidate_sources": ["DIRECT_STRICT_CHAIN_SOURCE"], "call_relation_modes": ["STRICT_CHAIN_NODE"], "shortest_call_path_length": 0, "internal_expansion_depth": 0, "supporting_evidence_ids": ["E1"], "included_in_prompt": True, "source_slice_id": "S1"}
        audit = {"selection_mode": "MULTI_VIEW_CANDIDATE_RANK_FUSION", "quota_selected_count": 1, "fusion_fill_selected_count": 0, "prompt_bytes": 4321}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, manifest, truth)
            def process(sample, raw):
                return {"status": "INSUFFICIENT_EVIDENCE", "topk_valid": False, "topk": [], "source_selection_mode": "EXACT_FUNCTION_SLICE", "pre_llm_candidate_count": 1, "prompt_candidate_count": 1, "fallback_candidate_count": 1, "prompt_bytes": 4321, "pre_llm_candidates": [candidate], "candidate_selection_audit": audit}
            frozen = infer_localization_v4(manifest, root, live=True, chain_id_check=lambda: 1, trace_provider=lambda sample: {}, sample_processor=process)
            self.assertEqual(frozen["predictions"][0]["pre_llm_candidates"], [candidate])
            self.assertEqual(frozen["predictions"][0]["fallback_candidate_count"], 1)
            self.assertEqual(frozen["predictions"][0]["prompt_bytes"], 4321)
            self.assertEqual(frozen["predictions"][0]["candidate_selection_audit"], audit)
            core = {key: frozen[key] for key in ("schema_version", "kind", "locked_manifest_sha256", "predictions", "external_call_counts")}
            expected = hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            self.assertEqual(frozen["payload_sha256"], expected)
            changed = json.loads(json.dumps(core)); changed["predictions"][0]["pre_llm_candidates"][0]["end_line"] = 21
            changed_sha = hashlib.sha256(json.dumps(changed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            self.assertNotEqual(changed_sha, frozen["payload_sha256"])

    def test_file_function_and_line_recall_at_1_3_5(self):
        label = {"truths": [{"relative_file": "src/A.sol", "function_signature": "f(uint256)", "function_name": "f", "function_name_unique_in_file": True, "truth_start_line": 10, "truth_end_line": 12}]}
        rows = [
            {"rank": 1, "relative_file": "X.sol", "function": "x()", "predicted_line_start": 1, "predicted_line_end": 1},
            {"rank": 2, "relative_file": "src/A.sol", "function": "other()", "predicted_line_start": 1, "predicted_line_end": 2},
            {"rank": 3, "relative_file": "src/A.sol", "function": "f(uint256)"},
            {"rank": 4, "relative_file": "Y.sol", "function": "y()"},
            {"rank": 5, "relative_file": "src/A.sol", "function": "f(uint256)", "predicted_line_start": 11, "predicted_line_end": 11},
        ]
        self.assertFalse(_sample_hits(label, rows, 1, "file")); self.assertTrue(_sample_hits(label, rows, 3, "file"))
        self.assertFalse(_sample_hits(label, rows, 1, "function")); self.assertTrue(_sample_hits(label, rows, 3, "function"))
        self.assertFalse(_sample_hits(label, rows, 3, "line")); self.assertTrue(_sample_hits(label, rows, 5, "line"))

    def test_no_predicted_range_never_uses_slice_as_line_prediction(self):
        label = {"truths": [{"relative_file": "A.sol", "function_signature": "f()", "function_name": "f", "function_name_unique_in_file": True, "truth_start_line": 10, "truth_end_line": 10}]}
        row = {"rank": 1, "relative_file": "A.sol", "function": "f()", "slice_start_line": 1, "slice_end_line": 20}
        self.assertTrue(_sample_hits(label, [row], 1, "file")); self.assertTrue(_sample_hits(label, [row], 1, "function")); self.assertFalse(_sample_hits(label, [row], 1, "line"))

    def test_evaluation_uses_fixed_end_to_end_and_conditional_denominators(self):
        manifest, truth = _small()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, manifest, truth)
            predictions = [
                {"sample_alias": manifest.samples[0].sample_alias, "status": "RANKED_CANDIDATES", "topk_valid": True, "source_selection_mode": "EXACT_FUNCTION_SLICE", "topk": [{"rank": 1, "relative_file": "src/A.sol", "function": "f(uint256)", "predicted_line_start": 10, "predicted_line_end": 10}]},
                {"sample_alias": manifest.samples[1].sample_alias, "status": "INSUFFICIENT_SOURCE", "topk_valid": False, "source_selection_mode": "NONE", "topk": []},
            ]
            frozen = _freeze("PHASE2_TOPK_V4", manifest, predictions, {"rpc": 0, "source": 0, "deepseek": 0}); _atomic(root / "frozen_predictions.json", frozen)
            report = evaluate_localization_v4(root)
            self.assertEqual(report["file_recall_at_1"], 0.5); self.assertEqual(report["conditional_file_recall_at_1"], 1.0)
            self.assertEqual(report["line_range_recall_at_5"], 0.5); self.assertEqual(report["source_insufficient_count"], 1)

    def test_pre_llm_hundred_and_prompt_twenty_candidate_recall_are_separate(self):
        manifest, truth = _small()
        def candidate(index, file_name, signature, included):
            return {"candidate_alias": f"C{index}", "relative_file": file_name, "function_signature": signature, "start_line": 1, "end_line": 2, "selector_match_status": "SOURCE_INDEX_RESOLVED", "candidate_sources": ["DIRECT_STRICT_CHAIN_SOURCE"], "call_relation_modes": ["STRICT_CHAIN_NODE"], "shortest_call_path_length": 0, "internal_expansion_depth": 0, "supporting_evidence_ids": ["E1"], "included_in_prompt": included, "source_slice_id": f"S{index}" if included else None}
        first_pool = [candidate(i, f"src/X{i}.sol", f"x{i}()", i <= 20) for i in range(1, 21)] + [candidate(21, "src/A.sol", "f(uint256)", False)]
        second_pool = [candidate(1, "src/B.sol", "g()", True)]
        predictions = [
            {"sample_alias": manifest.samples[0].sample_alias, "status": "RANKED_CANDIDATES", "topk_valid": False, "topk": [], "pre_llm_candidate_count": 21, "prompt_candidate_count": 20, "pre_llm_candidates": first_pool},
            {"sample_alias": manifest.samples[1].sample_alias, "status": "RANKED_CANDIDATES", "topk_valid": False, "topk": [], "pre_llm_candidate_count": 1, "prompt_candidate_count": 1, "pre_llm_candidates": second_pool},
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, manifest, truth); frozen = _freeze("PHASE2_TOPK_V4", manifest, predictions, {"rpc": 0, "source": 0, "deepseek": 0}); _atomic(root / "frozen_predictions.json", frozen)
            before = (root / "frozen_predictions.json").read_bytes(); report = evaluate_localization_v4(root)
            self.assertEqual(report["pre_llm_file_candidate_hit_count"], 2)
            self.assertEqual(report["pre_llm_function_candidate_hit_count"], 2)
            self.assertEqual(report["prompt_file_candidate_hit_count"], 1)
            self.assertEqual(report["prompt_function_candidate_hit_count"], 1)
            self.assertEqual(report["pre_llm_function_candidate_recall_at_100"], 1.0)
            self.assertEqual(report["prompt_function_candidate_recall_at_20"], 0.5)
            self.assertEqual((root / "frozen_predictions.json").read_bytes(), before)

    def test_missing_candidate_pool_is_zero_hit_not_removed_from_denominator(self):
        manifest, truth = _small()
        predictions = [{"sample_alias": sample.sample_alias, "status": "INSUFFICIENT_SOURCE", "topk_valid": False, "topk": []} for sample in manifest.samples]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, manifest, truth); _atomic(root / "frozen_predictions.json", _freeze("PHASE2_TOPK_V4", manifest, predictions, {"rpc": 0, "source": 0, "deepseek": 0}))
            report = evaluate_localization_v4(root)
            self.assertEqual(report["pre_llm_file_candidate_hit_count"], 0)
            self.assertEqual(report["prompt_function_candidate_hit_count"], 0)
            self.assertEqual(report["pre_llm_file_candidate_recall_at_100"], 0.0)
            self.assertEqual(report["prompt_function_candidate_recall_at_20"], 0.0)

    def test_checkpoint_resume_skips_completed_trace(self):
        manifest, truth = _small(); calls = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, manifest, truth)
            first = {"sample_alias": manifest.samples[0].sample_alias, "status": "INSUFFICIENT_EVIDENCE", "topk_valid": False, "topk": [], "source_selection_mode": "NONE"}
            _atomic(root / "predictions" / f"{manifest.samples[0].sample_alias}.json", first)
            _atomic(root / "checkpoint.json", {"manifest_sha256": manifest.manifest_sha256, "completed_aliases": [manifest.samples[0].sample_alias]})
            infer_localization_v4(manifest, root, live=True, chain_id_check=lambda: self.fail("chain check repeated"), trace_provider=lambda sample: calls.append(sample.sample_alias) or {}, sample_processor=lambda sample, raw: {"status": "INSUFFICIENT_EVIDENCE", "topk_valid": False, "topk": [], "source_selection_mode": "NONE"})
            self.assertEqual(calls, [manifest.samples[1].sample_alias])

    def test_request_budget_rejects_overrun(self):
        manifest, truth = _small()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, manifest, truth)
            with self.assertRaisesRegex(ValueError, "REQUEST_LIMIT"):
                infer_localization_v4(manifest, root, live=True, chain_id_check=lambda: 1, trace_provider=lambda sample: {}, sample_processor=lambda sample, raw: {"status": "X", "topk_valid": False, "topk": [], "source_selection_mode": "NONE", "source_request_count": 7})

    def test_v5_allows_two_stages_with_one_retry_each_and_freezes_audit(self):
        manifest, truth = _small()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, manifest, truth)
            def process(sample, raw):
                return {
                    "status": "RANKED_CANDIDATES", "topk_valid": False, "topk": [],
                    "source_request_count": 0, "deepseek_request_count": 4,
                    "stage_a_input_candidate_count": 100, "stage_a_selected_count": 20,
                    "stage_a_selection_source": "LLM_COARSE_SHORTLIST", "stage_a_request_count": 2,
                    "stage_a_retry_count": 1, "stage_a_retried": True, "stage_a_fallback": False,
                    "stage_a_prompt_bytes": 24000, "stage_b_input_candidate_count": 20,
                    "stage_b_request_count": 2, "stage_b_retry_count": 1,
                    "stage_b_prompt_bytes": 58000, "stage_b_valid_topk_count": 0,
                }
            frozen = infer_localization_v4(manifest, root, live=True, chain_id_check=lambda: 1, trace_provider=lambda sample: {}, sample_processor=process, method=METHOD_V5)
            self.assertEqual(frozen["kind"], METHOD_V5)
            self.assertEqual(frozen["external_call_counts"]["deepseek"], 8)
            self.assertEqual(frozen["predictions"][0]["stage_a_selection_source"], "LLM_COARSE_SHORTLIST")

    def test_v5_stage_a_order_and_invalid_response_fallback_are_fixed(self):
        prep = _v5_shortlist_preparation()
        valid_factory = _ShortlistFactory(json.dumps({"candidate_ids": ["C3", "C1", "C2"]}))
        valid = select_phase2_v5_shortlist(prep, "key", valid_factory)
        self.assertEqual(valid["selection_source"], "LLM_COARSE_SHORTLIST")
        self.assertEqual(valid["candidate_ids"], ("C3", "C1", "C2"))
        invalid_factory = _ShortlistFactory(json.dumps({"candidate_ids": ["C1"]}))
        fallback = select_phase2_v5_shortlist(prep, "key", invalid_factory)
        self.assertEqual(fallback["selection_source"], "DETERMINISTIC_SHORTLIST_FALLBACK")
        self.assertEqual(fallback["candidate_ids"], ("C1", "C2", "C3"))
        self.assertIn("COARSE_LLM_RESPONSE_INVALID", fallback["reason_codes"])
        self.assertEqual(invalid_factory.calls, 1)

    def test_v5_retries_structural_preparation_with_fifty_candidates(self):
        structural = Phase2LocalizationPreparation("STRUCTURAL_ERROR", None, (), 0, {}, 0, None, 0, {})
        ready = _v5_shortlist_preparation()
        bundle = SimpleNamespace(strict={}, causal={}, graph={})
        factory = _ShortlistFactory(json.dumps({"candidate_ids": ["C1"]}))
        sample = LocalizationV4Sample("localization_v4_eth_001", 1, "0x" + "1" * 64, 1)

        with (
            patch("scripts.run_localization15_topk_v4.build_real_phase2_evidence", return_value=bundle),
            patch("scripts.run_localization15_topk_v4.acquire_real_source_slices", return_value=({"metadata": {}}, {"source_request_count": 0})),
            patch("scripts.run_localization15_topk_v4._attach_formal_candidate_union", side_effect=lambda _bundle, source, max_candidates: {**source, "candidate_limit": max_candidates}) as attach,
            patch("scripts.run_localization15_topk_v4.prepare_phase2_localization", side_effect=[structural, ready]) as prepare,
        ):
            result = build_live_sample_processor_v5({}, "key", transport_factory=factory)(sample, {})

        self.assertEqual([call.kwargs["max_candidates"] for call in attach.call_args_list], [100, 50])
        self.assertEqual(prepare.call_count, 2)
        self.assertTrue(result["preparation_fallback_used"])
        self.assertEqual(result["preparation_candidate_limit"], 50)
        self.assertIn("V5_PREPARATION_FALLBACK_TO_50_CANDIDATES", result["reason_codes"])

    def test_safe_provider_response_reason_code_is_not_mistaken_for_raw_response(self):
        value = {"status": "ERROR", "reason_codes": ["PROVIDER_RESPONSE_INVALID", "COARSE_LLM_RESPONSE_INVALID"], "topk": []}
        _validate_public_prediction(value)

    def test_sensitive_prediction_keys_and_raw_identities_are_still_rejected(self):
        for value in (
            {"provider_response": {"raw": True}},
            {"nested": {"reasoning_content": "secret"}},
            {"api_key": "secret"},
            {"location": "hidden"},
            {"safe": "0x" + "a" * 40},
            {"safe": "0x" + "b" * 64},
        ):
            with self.assertRaisesRegex(ValueError, "PREDICTION_SENSITIVE"):
                _validate_public_prediction(value)

    def test_v5_rejects_more_than_four_deepseek_attempts_per_sample_budget(self):
        manifest, truth = _small()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, manifest, truth)
            with self.assertRaisesRegex(ValueError, "REQUEST_LIMIT"):
                infer_localization_v4(manifest, root, live=True, chain_id_check=lambda: 1, trace_provider=lambda sample: {}, sample_processor=lambda sample, raw: {"status": "X", "topk_valid": False, "topk": [], "deepseek_request_count": 5}, method=METHOD_V5)

    def test_frozen_sha_and_truth_sha_are_required(self):
        manifest, truth = _small()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); write_prepared(root, manifest, truth)
            frozen = _freeze("PHASE2_TOPK_V4", manifest, [], {"rpc": 0, "source": 0, "deepseek": 0}); frozen["payload_sha256"] = "0" * 64; _atomic(root / "frozen_predictions.json", frozen)
            with self.assertRaisesRegex(ValueError, "FROZEN_SHA"):
                evaluate_localization_v4(root)


if __name__ == "__main__": unittest.main()

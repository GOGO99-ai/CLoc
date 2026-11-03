from __future__ import annotations

import copy
import unittest

from src.cloc_v2.phase2_causal_pair_guard import apply_causal_pair_guard
from src.cloc_v2.phase2_localization_prompt import validate_phase2_localization_topk_v4
from tests.cloc_v2.test_phase2_upstream_entry_guard import _prompt, _decision, _candidate


def _rows():
    rows = [_candidate(f"S{i}", endpoint=i not in (1, 6), upstream=i in (1, 6)) for i in range(1, 7)]
    for i, row in enumerate(rows, 1):
        row.update(contract_alias="K1", relative_file="source.sol", function_signature=f"f{i}()", start_line=i, end_line=i)
        row["ranking_features"]["call_asset_joint_support"] = True
    return rows


def _bind_child(rows, *, parent_index=5, count=1):
    parent = rows[parent_index]
    child = rows[0]
    child["internal_expansion_depth"] = 1
    child["selector_match_status"] = "INTERNAL_UNIQUE_CALL_EXPANSION"
    child["unique_parent_function_identity"] = {
        key: parent[key] for key in ("contract_alias", "relative_file", "function_signature", "start_line", "end_line", "selector_match_status")
    }
    child["unique_parent_function_identity"]["direct_internal_callee_count"] = count


class CausalPairGuardTests(unittest.TestCase):
    def test_reserves_entry_without_endpoint_threshold_and_preserves_top_two(self):
        prompt = _prompt()
        rows = _rows()
        for row in rows:
            row["ranking_features"]["asset_result_endpoint"] = False
        before = copy.deepcopy(rows)
        result = apply_causal_pair_guard(_decision(prompt), prompt, rows, protect_unique_child=False)
        self.assertEqual([r["source_slice_id"] for r in result["response"]["findings"]], ["S2", "S3", "S4", "S5", "S1"])
        self.assertEqual(rows, before)
        self.assertEqual(result["response"]["findings"][4]["evidence_ids"], ["E1"])

    def test_unique_child_keeps_its_rank_five_parent(self):
        prompt = _prompt()
        rows = _rows()
        _bind_child(rows)
        result = apply_causal_pair_guard(_decision(prompt), prompt, rows)
        self.assertEqual([r["source_slice_id"] for r in result["response"]["findings"]], ["S2", "S3", "S4", "S6", "S1"])
        self.assertTrue(validate_phase2_localization_topk_v4(result["response"], prompt).valid)

    def test_child_requires_unique_direct_parent_and_full_identity(self):
        prompt = _prompt()
        for mutation in ("multiple_callees", "wrong_contract", "wrong_file", "missing_parent", "depth_two"):
            with self.subTest(mutation=mutation):
                rows = _rows()
                _bind_child(rows)
                if mutation == "multiple_callees":
                    rows[0]["unique_parent_function_identity"]["direct_internal_callee_count"] = 2
                elif mutation == "wrong_contract":
                    rows[0]["unique_parent_function_identity"]["contract_alias"] = "K2"
                elif mutation == "wrong_file":
                    rows[0]["unique_parent_function_identity"]["relative_file"] = "other.sol"
                elif mutation == "missing_parent":
                    rows[0]["unique_parent_function_identity"] = None
                else:
                    rows[0]["internal_expansion_depth"] = 2
                result = apply_causal_pair_guard(_decision(prompt), prompt, rows, protect_entry=False)
                self.assertFalse(result["applied"])

    def test_entry_requires_identity_joint_evidence_and_prompt_membership(self):
        prompt = _prompt()
        for mutation in ("coarse", "no_joint", "no_execution", "no_strict", "not_in_prompt", "no_whitelist"):
            with self.subTest(mutation=mutation):
                rows = _rows()
                rows[5]["ranking_features"]["asset_result_endpoint"] = True
                if mutation == "coarse":
                    rows[0]["selector_match_status"] = "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK"
                elif mutation == "not_in_prompt":
                    rows[0]["included_in_prompt"] = False
                elif mutation == "no_whitelist":
                    rows[0]["source_slice_id"] = "S999"
                else:
                    feature = {"no_joint": "call_asset_joint_support", "no_execution": "executed_call_node", "no_strict": "direct_strict_evidence"}[mutation]
                    rows[0]["ranking_features"][feature] = False
                self.assertFalse(apply_causal_pair_guard(_decision(prompt), prompt, rows)["applied"])

    def test_disabled_components_and_missing_metadata_leave_response_unchanged(self):
        prompt = _prompt()
        decision = _decision(prompt)
        for rows, options in [(_rows(), {"protect_entry": False, "protect_unique_child": False}), (_rows()[:4], {})]:
            result = apply_causal_pair_guard(decision, prompt, rows, **options)
            self.assertEqual(result["response"], dict(decision.decision))

    def test_names_do_not_affect_ranking_when_parent_identity_is_consistent(self):
        prompt = _prompt()
        rows = _rows()
        _bind_child(rows)
        original = apply_causal_pair_guard(_decision(prompt), prompt, rows)
        for row in rows:
            row["function_signature"] = "renamed_" + row["function_signature"]
        rows[0]["unique_parent_function_identity"]["function_signature"] = rows[5]["function_signature"]
        self.assertEqual(original, apply_causal_pair_guard(_decision(prompt), prompt, rows))

    def test_rejects_invalid_decision_and_duplicate_candidate_ids(self):
        prompt = _prompt()
        with self.assertRaisesRegex(ValueError, "INPUT_INVALID"):
            apply_causal_pair_guard(validate_phase2_localization_topk_v4({"findings": []}, prompt), prompt, [])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_SLICE"):
            apply_causal_pair_guard(_decision(prompt), prompt, _rows() + [_rows()[0]])

    def test_tampered_inserted_evidence_is_rejected(self):
        prompt = _prompt()
        result = apply_causal_pair_guard(_decision(prompt), prompt, _rows())
        result["response"]["findings"][4]["evidence_ids"] = ["E999"]
        self.assertFalse(validate_phase2_localization_topk_v4(result["response"], prompt).valid)

    def test_existing_coverage_never_reorders_findings(self):
        prompt = _prompt()
        rows = _rows()
        rows[0]["ranking_features"]["call_asset_joint_support"] = False
        decision = _decision(prompt)
        result = apply_causal_pair_guard(decision, prompt, rows)
        self.assertFalse(result["applied"])
        self.assertEqual(result["response"], dict(decision.decision))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from src.cloc_v2.phase2_endpoint_saturation_guard import apply_endpoint_saturation_guard
from src.cloc_v2.phase2_localization_prompt import (
    Phase2EvidenceEdge,
    Phase2ResultChainContext,
    Phase2SourceSlice,
    Phase2TransactionContext,
    build_phase2_localization_prompt,
    upgrade_phase2_localization_prompt_to_topk_v4,
    validate_phase2_localization_topk_v4,
)


def _prompt():
    edges = tuple(
        Phase2EvidenceEdge(f"E{i}", "Q1", "CALL", "A1", "A2", f"fact {i}")
        for i in range(1, 9)
    )
    slices = tuple(
        Phase2SourceSlice(f"S{i}", f"F{i}", f"f{i}", i, i, f"function f{i}(){{}}", (f"E{i}",))
        for i in range(1, 9)
    )
    return upgrade_phase2_localization_prompt_to_topk_v4(build_phase2_localization_prompt(
        Phase2TransactionContext("T1", "summary"),
        (Phase2ResultChainContext("Q1", "result"),),
        edges,
        slices,
    ))


def _candidate(source_slice_id: str, *, endpoint: bool, controller: bool = False):
    return {
        "source_slice_id": source_slice_id,
        "included_in_prompt": True,
        "selector_match_status": "SELECTOR_EXACT_MATCH",
        "internal_expansion_depth": 0,
        "ranking_features": {
            "executed_call_node": controller or endpoint,
            "direct_strict_evidence": controller or endpoint,
            "call_asset_joint_support": controller or endpoint,
            "asset_result_endpoint": endpoint,
        },
    }


def _decision(prompt):
    response = {
        "findings": [
            {"rank": rank, "source_slice_id": f"S{rank + 3}", "evidence_ids": [f"E{rank + 3}"]}
            for rank in range(1, 6)
        ]
    }
    return validate_phase2_localization_topk_v4(response, prompt)


class Phase2EndpointSaturationGuardTests(unittest.TestCase):
    def test_prepends_three_evidenced_controllers(self):
        prompt = _prompt()
        candidates = [
            *(_candidate(f"S{i}", endpoint=False, controller=True) for i in range(1, 4)),
            *(_candidate(f"S{i}", endpoint=True) for i in range(4, 9)),
        ]
        result = apply_endpoint_saturation_guard(_decision(prompt), prompt, candidates)
        self.assertTrue(result["applied"])
        self.assertEqual(result["controller_source_slice_ids"], ["S1", "S2", "S3"])
        self.assertEqual(
            [row["source_slice_id"] for row in result["response"]["findings"]],
            ["S1", "S2", "S3", "S4", "S5"],
        )
        self.assertEqual(result["response"]["findings"][0]["evidence_ids"], ["E1"])
        self.assertTrue(validate_phase2_localization_topk_v4(result["response"], prompt).valid)

    def test_does_not_change_a_mixed_top5(self):
        prompt = _prompt()
        candidates = [
            _candidate("S1", endpoint=False, controller=True),
            _candidate("S4", endpoint=False),
            *(_candidate(f"S{i}", endpoint=True) for i in range(5, 9)),
        ]
        decision = _decision(prompt)
        result = apply_endpoint_saturation_guard(decision, prompt, candidates)
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason_code"], "TOP5_NOT_ENDPOINT_SATURATED")
        self.assertEqual(result["response"], dict(decision.decision))

    def test_saturated_top5_without_controller_is_unchanged(self):
        prompt = _prompt()
        candidates = [_candidate(f"S{i}", endpoint=True) for i in range(4, 9)]
        decision = _decision(prompt)
        result = apply_endpoint_saturation_guard(decision, prompt, candidates)
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason_code"], "NO_ELIGIBLE_CONTROLLER")
        self.assertEqual(result["response"], dict(decision.decision))

    def test_rejects_an_invalid_provider_decision(self):
        prompt = _prompt()
        invalid = validate_phase2_localization_topk_v4({"findings": []}, prompt)
        with self.assertRaisesRegex(ValueError, "ENDPOINT_SATURATION_GUARD_INPUT_INVALID"):
            apply_endpoint_saturation_guard(invalid, prompt, [])


if __name__ == "__main__":
    unittest.main()

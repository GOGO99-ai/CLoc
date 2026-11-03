from __future__ import annotations

import unittest

from src.cloc_v2.phase2_localization_prompt import (
    Phase2EvidenceEdge,
    Phase2ResultChainContext,
    Phase2SourceSlice,
    Phase2TransactionContext,
    build_phase2_localization_prompt,
    upgrade_phase2_localization_prompt_to_topk_v4,
    validate_phase2_localization_topk_v4,
)
from src.cloc_v2.phase2_upstream_entry_guard import apply_upstream_entry_guard


def _prompt():
    edges = tuple(
        Phase2EvidenceEdge(f"E{i}", "Q1", "CALL", "A1", "A2", f"fact {i}")
        for i in range(1, 7)
    )
    slices = tuple(
        Phase2SourceSlice(f"S{i}", f"F{i}", f"f{i}", i, i, f"function f{i}(){{}}", (f"E{i}",))
        for i in range(1, 7)
    )
    return upgrade_phase2_localization_prompt_to_topk_v4(build_phase2_localization_prompt(
        Phase2TransactionContext("T1", "summary"),
        (Phase2ResultChainContext("Q1", "result"),),
        edges,
        slices,
    ))


def _candidate(source_slice_id: str, *, endpoint: bool, upstream: bool = False):
    return {
        "source_slice_id": source_slice_id,
        "included_in_prompt": True,
        "selector_match_status": "UNIQUE_NAME_MATCH",
        "internal_expansion_depth": 0,
        "ranking_features": {
            "executed_call_node": endpoint or upstream,
            "direct_strict_evidence": endpoint or upstream,
            "asset_result_endpoint": endpoint,
        },
    }


def _decision(prompt):
    response = {"findings": [
        {"rank": rank, "source_slice_id": f"S{rank + 1}", "evidence_ids": [f"E{rank + 1}"]}
        for rank in range(1, 6)
    ]}
    return validate_phase2_localization_topk_v4(response, prompt)


class Phase2UpstreamEntryGuardTests(unittest.TestCase):
    def test_inserts_one_upstream_entry_at_rank_three(self):
        prompt = _prompt()
        candidates = [
            _candidate("S1", endpoint=False, upstream=True),
            *(_candidate(f"S{i}", endpoint=i != 3) for i in range(2, 7)),
        ]
        result = apply_upstream_entry_guard(_decision(prompt), prompt, candidates)
        self.assertTrue(result["applied"])
        self.assertEqual(result["endpoint_count"], 4)
        self.assertEqual(result["upstream_entry_source_slice_id"], "S1")
        self.assertEqual(
            [row["source_slice_id"] for row in result["response"]["findings"]],
            ["S2", "S3", "S1", "S4", "S5"],
        )
        self.assertTrue(validate_phase2_localization_topk_v4(result["response"], prompt).valid)

    def test_does_not_change_top5_with_only_three_endpoints(self):
        prompt = _prompt()
        candidates = [
            _candidate("S1", endpoint=False, upstream=True),
            *(_candidate(f"S{i}", endpoint=i in {2, 4, 6}) for i in range(2, 7)),
        ]
        decision = _decision(prompt)
        result = apply_upstream_entry_guard(decision, prompt, candidates)
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason_code"], "TOP5_NOT_ENDPOINT_HEAVY")
        self.assertEqual(result["response"], dict(decision.decision))

    def test_internal_candidate_is_not_inserted(self):
        prompt = _prompt()
        candidates = [
            _candidate("S1", endpoint=False, upstream=True),
            *(_candidate(f"S{i}", endpoint=True) for i in range(2, 7)),
        ]
        candidates[0]["internal_expansion_depth"] = 1
        result = apply_upstream_entry_guard(_decision(prompt), prompt, candidates)
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason_code"], "NO_ELIGIBLE_UPSTREAM_ENTRY")

    def test_rejects_invalid_provider_decision(self):
        prompt = _prompt()
        invalid = validate_phase2_localization_topk_v4({"findings": []}, prompt)
        with self.assertRaisesRegex(ValueError, "UPSTREAM_ENTRY_GUARD_INPUT_INVALID"):
            apply_upstream_entry_guard(invalid, prompt, [])


if __name__ == "__main__":
    unittest.main()

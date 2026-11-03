from __future__ import annotations

import json
import unittest

from src.cloc_v2.phase2_candidate_shortlist_prompt import (
    MAX_PROMPT_BYTES,
    build_phase2_candidate_shortlist_prompt,
    serialize_phase2_candidate_shortlist_prompt,
    validate_phase2_candidate_shortlist_response,
)
from src.cloc_v2.phase2_localization_deepseek import (
    Phase2DeepSeekConfig,
    build_phase2_shortlist_deepseek_request,
)


def _rows(count: int):
    return [
        {
            "candidate_alias": f"C{i}",
            "relative_file": f"src/F{i}.sol",
            "function_signature": f"f{i}(uint256)",
            "visibility": "public" if i % 2 else "external",
            "candidate_sources": ["DIRECT_STRICT_CHAIN_SOURCE" if i == 1 else "GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK"],
            "view_ranks": {
                "IDENTITY_EVIDENCE_VIEW": i,
                "CALL_CAUSAL_PROXIMITY_VIEW": i,
                "ASSET_RESULT_RELEVANCE_VIEW": i,
                "EXECUTED_SOURCE_COVERAGE_VIEW": i,
            },
            "fusion_score": 1 / (60 + i),
            "ranking_features": {
                "direct_strict_evidence": i == 1,
                "asset_result_endpoint": i == 2,
                "call_asset_joint_support": i <= 3,
                "address_match_source": "RUNTIME_ADDRESS_MATCH",
                "executed_call_node": True,
            },
            "selector_match_status": "SELECTOR_EXACT_MATCH" if i == 1 else "SOURCE_INDEX_RESOLVED",
            "call_relation_modes": ["STRICT_CHAIN_NODE" if i == 1 else "UNDIRECTED_CALL_CORRIDOR"],
            "shortest_call_path_length": i - 1,
            "internal_expansion_depth": 0,
            "uncertainty_codes": ["FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK"],
        }
        for i in range(1, count + 1)
    ]


class Phase2CandidateShortlistPromptTests(unittest.TestCase):
    def test_json_mode_instruction_is_explicit_without_changing_request_contract(self):
        prompt = build_phase2_candidate_shortlist_prompt(_rows(25)).prompt
        rendered = f"{prompt.system_message}\n{prompt.user_message}"
        self.assertIn("JSON", rendered)
        self.assertIn("Return exactly one JSON object and no other text.", rendered)
        self.assertIn("JSON root object must contain only candidate_ids", rendered)
        self.assertIn("Markdown code fences", rendered)
        self.assertIn("all other free text are forbidden", rendered)

        request = build_phase2_shortlist_deepseek_request(prompt, Phase2DeepSeekConfig())
        self.assertEqual(
            set(request.payload),
            {"model", "messages", "thinking", "reasoning_effort", "response_format", "max_tokens"},
        )
        self.assertEqual(request.payload["model"], "deepseek-v4-pro")
        self.assertEqual(request.payload["thinking"], {"type": "enabled"})
        self.assertEqual(request.payload["reasoning_effort"], "high")
        self.assertEqual(request.payload["response_format"], {"type": "json_object"})
        self.assertEqual(request.payload["max_tokens"], 8192)
        self.assertEqual([row["role"] for row in request.payload["messages"]], ["system", "user"])
        self.assertTrue(all(isinstance(row["content"], str) for row in request.payload["messages"]))

    def test_all_hundred_candidates_enter_metadata_prompt(self):
        result = build_phase2_candidate_shortlist_prompt(_rows(100))
        self.assertEqual(result.status, "READY_FOR_LLM")
        self.assertEqual(len(result.prompt.candidate_ids), 100)
        self.assertEqual(result.prompt.expected_candidate_count, 20)
        self.assertLessEqual(result.prompt.prompt_bytes, MAX_PROMPT_BYTES)
        rendered = serialize_phase2_candidate_shortlist_prompt(result.prompt)
        self.assertNotIn(".sol", rendered)
        self.assertNotIn("source_text", rendered)

    def test_exact_twenty_and_all_when_fewer(self):
        for count, expected in ((100, 20), (7, 7)):
            prompt = build_phase2_candidate_shortlist_prompt(_rows(count)).prompt
            response = {"candidate_ids": list(prompt.candidate_ids[:expected])}
            valid = validate_phase2_candidate_shortlist_response(response, prompt)
            self.assertTrue(valid.valid)
            self.assertEqual(len(valid.candidate_ids), expected)

    def test_unknown_duplicate_and_wrong_count_are_rejected(self):
        prompt = build_phase2_candidate_shortlist_prompt(_rows(25)).prompt
        cases = [
            {"candidate_ids": ["C1"] * 20},
            {"candidate_ids": [f"C{i}" for i in range(1, 20)] + ["C99"]},
            {"candidate_ids": [f"C{i}" for i in range(1, 20)]},
        ]
        for response in cases:
            self.assertFalse(validate_phase2_candidate_shortlist_response(response, prompt).valid)

    def test_extra_root_fields_and_free_text_are_rejected(self):
        prompt = build_phase2_candidate_shortlist_prompt(_rows(3)).prompt
        response = {"candidate_ids": ["C1", "C2", "C3"], "summary": "provider text"}
        validation = validate_phase2_candidate_shortlist_response(response, prompt)
        self.assertFalse(validation.valid)
        self.assertNotIn("provider text", json.dumps(validation.to_dict()))

    def test_serialization_is_deterministic(self):
        first = build_phase2_candidate_shortlist_prompt(_rows(100)).prompt
        second = build_phase2_candidate_shortlist_prompt(_rows(100)).prompt
        self.assertEqual(serialize_phase2_candidate_shortlist_prompt(first), serialize_phase2_candidate_shortlist_prompt(second))

    def test_raw_identity_is_rejected(self):
        rows = _rows(1)
        rows[0]["function_signature"] = "f_0x" + "a" * 40
        self.assertEqual(build_phase2_candidate_shortlist_prompt(rows).status, "INVALID_CANDIDATE_PACKAGE")


if __name__ == "__main__":
    unittest.main()

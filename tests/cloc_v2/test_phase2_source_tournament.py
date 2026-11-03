from __future__ import annotations

import json
import unittest

from src.cloc_v2.phase2_localization_deepseek import Phase2ClassifiedTransportError, Phase2DeepSeekConfig
from src.cloc_v2.phase2_source_tournament import (
    FINAL_INVALID,
    GROUP_INCOMPLETE,
    GROUPING_INVALID,
    MAX_FINALISTS,
    MAX_FINAL_PROMPT_BYTES,
    MAX_GROUP_SIZE,
    NOT_READY,
    READY,
    TournamentCandidate,
    TournamentEvidence,
    TournamentEvidenceContext,
    TournamentFinalist,
    TournamentGroup,
    TournamentPrompt,
    TournamentResultChain,
    build_tournament_final_prompt,
    build_tournament_group_prompt,
    build_tournament_groups,
    build_tournament_candidates_from_union,
    execute_tournament_prompt_with_single_retry,
    make_provider_response,
    merge_tournament_group_winners,
    run_tournament_offline,
    serialize_tournament_grouping,
    serialize_tournament_readiness,
    tournament_future_request_budget,
    validate_tournament_response,
)


def _context() -> TournamentEvidenceContext:
    return TournamentEvidenceContext(
        "T1",
        (TournamentResultChain("Q1", "Synthetic strict result chain."),),
        (
            TournamentEvidence("E1", "Q1", "CALL", "A1", "A2", "Synthetic real CALL edge."),
            TournamentEvidence("E2", "Q1", "ASSET", "A2", "A3", "Synthetic result ASSET edge."),
        ),
    )


def _candidates(count: int, *, source_size: int = 40, one_contract: bool = False) -> tuple[TournamentCandidate, ...]:
    return tuple(
        TournamentCandidate(
            candidate_id=f"C{i}",
            source_slice_id=f"S{i}",
            file_alias=f"F{1 + ((i - 1) % 9)}",
            contract_alias="K1" if one_contract else f"K{1 + ((i - 1) % 7)}",
            function_signature=f"f{i}(uint256)",
            visibility="public",
            start_line=i,
            end_line=i + 1,
            _source_text="function f(uint256 value) public { value; }" + ("x" * source_size),
            evidence_ids=("E1", "E2") if i % 2 else ("E1",),
            candidate_sources=("DIRECT_STRICT_CHAIN_SOURCE",) if i <= 5 else ("GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK",),
            view_ranks={"IDENTITY_EVIDENCE_VIEW": i, "CALL_CAUSAL_PROXIMITY_VIEW": i},
            rrf_rank=i,
            rrf_score=f"{1 / (60 + i):.12f}",
            shortest_call_distance=i % 5,
            call_relation_grade="STRICT_CHAIN_NODE" if i <= 5 else "UNDIRECTED_CALL_CORRIDOR",
            strict_chain_direct_evidence=i <= 5,
            selector_exact_identity=i <= 5,
            coarse_fallback=i > 5,
            address_match_evidence="RUNTIME_ADDRESS_MATCH",
            uncertainty_codes=() if i <= 5 else ("FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK",),
        )
        for i in range(1, count + 1)
    )


def _valid_findings(prompt: TournamentPrompt, *, force_slice: str | None = None):
    selected = list(prompt.source_slice_ids[: prompt.expected_findings])
    if force_slice is not None and force_slice in prompt.source_slice_ids and force_slice not in selected:
        selected[-1] = force_slice
    return [
        {"rank": rank, "source_slice_id": sid, "evidence_ids": [prompt.evidence_by_slice[sid][0]]}
        for rank, sid in enumerate(selected, start=1)
    ]


class _ResponseTransport:
    def __init__(self, response):
        self.response = response

    def send(self, request_payload, api_key, timeout_seconds):
        return self.response


class Phase2SourceTournamentGroupingTests(unittest.TestCase):
    def test_existing_union_adapter_preserves_all_rows_and_hides_paths(self):
        rows = []
        sources = {}
        for index in range(1, 101):
            rows.append({
                "candidate_alias": f"C{index}",
                "contract_alias": f"K{1 + ((index - 1) % 7)}",
                "relative_file": f"contracts/Module{1 + ((index - 1) % 4)}.sol",
                "function_signature": f"f{index}(uint256)",
                "visibility": "public",
                "start_line": index,
                "end_line": index + 1,
                "supporting_evidence_ids": ["E1"],
                "candidate_sources": ["GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK"],
                "view_ranks": {"IDENTITY_EVIDENCE_VIEW": index},
                "fusion_rank": index,
                "fusion_score": 1 / (60 + index),
                "call_relation_modes": ["UNDIRECTED_CALL_CORRIDOR"],
                "shortest_call_path_length": index % 5,
                "ranking_features": {"address_match_source": "RUNTIME_ADDRESS_MATCH", "executed_call_node": True},
            })
            sources[f"C{index}"] = f"function f{index}(uint256 value) public {{ value; }}"
        candidates = build_tournament_candidates_from_union(rows, sources)
        self.assertEqual(len(candidates), 100)
        self.assertEqual([item.candidate_id for item in candidates], [f"C{i}" for i in range(1, 101)])
        public = json.dumps([item.to_public_dict() for item in candidates], ensure_ascii=False)
        self.assertNotIn("contracts/", public)
        self.assertNotIn(".sol", public)
        self.assertNotIn("source_text", public)

    def test_required_candidate_counts_are_exactly_preserved(self):
        for count in (1, 19, 20, 21, 50, 79, 97, 100):
            with self.subTest(count=count):
                candidates = _candidates(count)
                grouping = build_tournament_groups(candidates)
                assigned = [item.candidate_id for group in grouping.groups for item in group.candidates]
                self.assertEqual(grouping.audit.status, "VALID")
                self.assertTrue(grouping.audit.all_candidates_assigned)
                self.assertTrue(grouping.audit.candidate_assignment_unique)
                self.assertTrue(grouping.audit.group_size_limit_satisfied)
                self.assertTrue(grouping.audit.candidate_count_preserved)
                self.assertEqual(set(assigned), {item.candidate_id for item in candidates})
                self.assertEqual(len(assigned), count)
                self.assertLessEqual(max(grouping.audit.group_candidate_counts), MAX_GROUP_SIZE)

    def test_fixed_snake_and_same_contract_dispersion_are_deterministic(self):
        candidates = _candidates(10, one_contract=True)
        first = build_tournament_groups(candidates)
        second = build_tournament_groups(candidates)
        self.assertEqual(serialize_tournament_grouping(first), serialize_tournament_grouping(second))
        self.assertEqual(
            [[item.candidate_id for item in group.candidates] for group in first.groups],
            [["C1", "C10"], ["C2", "C9"], ["C3", "C8"], ["C4", "C7"], ["C5", "C6"]],
        )
        self.assertEqual({group.candidates[0].contract_alias for group in first.groups}, {"K1"})
        self.assertTrue(all(len(group.candidates) == 2 for group in first.groups))

    def test_duplicate_candidate_assignment_is_invalid_and_builds_no_groups(self):
        one = _candidates(1)[0]
        grouping = build_tournament_groups((one, one))
        self.assertEqual(grouping.audit.status, GROUPING_INVALID)
        self.assertFalse(grouping.audit.candidate_assignment_unique)
        self.assertEqual(grouping.groups, ())


class Phase2SourceTournamentPromptTests(unittest.TestCase):
    def setUp(self):
        self.context = _context()
        self.group = build_tournament_groups(_candidates(21)).groups[0]
        self.prompt = build_tournament_group_prompt(self.group, self.context).prompt

    def test_group_prompt_has_strict_exact_topk_and_no_raw_identity(self):
        self.assertEqual(self.prompt.expected_findings, min(5, len(self.group.candidates)))
        text = self.prompt.user_message
        self.assertIn("OUTPUT_CONTRACT", text)
        self.assertIn("EVIDENCE_PACKAGE", text)
        self.assertNotRegex(text, r"0x[0-9a-fA-F]{40}")
        self.assertNotIn("D:\\", text)

    def test_group_response_rejects_unknown_duplicate_cross_group_and_extra_text(self):
        valid = _valid_findings(self.prompt)
        cases = []
        unknown = json.loads(json.dumps(valid)); unknown[0]["source_slice_id"] = "S999"
        cases.append({"findings": unknown})
        duplicate = json.loads(json.dumps(valid)); duplicate[-1]["source_slice_id"] = duplicate[0]["source_slice_id"]
        cases.append({"findings": duplicate})
        cross_group = json.loads(json.dumps(valid))
        other_slice = next(item.source_slice_id for item in _candidates(21) if item.source_slice_id not in self.prompt.source_slice_ids)
        cross_group[0]["source_slice_id"] = other_slice
        cases.append({"findings": cross_group})
        cases.append({"findings": valid, "summary": "free text"})
        for response in cases:
            self.assertFalse(validate_tournament_response(response, self.prompt).valid)

    def test_line_range_and_evidence_scope_are_strict(self):
        valid = _valid_findings(self.prompt)
        valid[0].update(line_start=self.group.candidates[0].start_line, line_end=self.group.candidates[0].end_line)
        self.assertTrue(validate_tournament_response({"findings": valid}, self.prompt).valid)
        invalid_line = json.loads(json.dumps(valid)); invalid_line[0]["line_end"] = 99999
        self.assertFalse(validate_tournament_response({"findings": invalid_line}, self.prompt).valid)
        invalid_evidence = json.loads(json.dumps(valid)); invalid_evidence[0]["evidence_ids"] = ["E999"]
        self.assertFalse(validate_tournament_response({"findings": invalid_evidence}, self.prompt).valid)

    def test_finalists_come_only_from_valid_group_winners_and_are_at_most_25(self):
        grouping = build_tournament_groups(_candidates(100))
        validations = {}
        for group in grouping.groups:
            prompt = build_tournament_group_prompt(group, self.context).prompt
            validations[group.group_alias] = validate_tournament_response({"findings": _valid_findings(prompt)}, prompt)
        finalists = merge_tournament_group_winners(grouping, validations)
        self.assertEqual(len(finalists), MAX_FINALISTS)
        winner_slices = {finding.source_slice_id for value in validations.values() for finding in value.findings}
        self.assertEqual({item.candidate.source_slice_id for item in finalists}, winner_slices)
        final = build_tournament_final_prompt(finalists, self.context)
        self.assertEqual(final.status, "READY_FOR_LLM")
        self.assertEqual(final.prompt.expected_findings, 5)

    def test_invalid_group_blocks_final_merge(self):
        grouping = build_tournament_groups(_candidates(50))
        validations = {}
        for group in grouping.groups:
            prompt = build_tournament_group_prompt(group, self.context).prompt
            validations[group.group_alias] = validate_tournament_response({"findings": _valid_findings(prompt)}, prompt)
        validations["G3"] = validate_tournament_response({"findings": []}, build_tournament_group_prompt(grouping.groups[2], self.context).prompt)
        with self.assertRaisesRegex(ValueError, GROUP_INCOMPLETE):
            merge_tournament_group_winners(grouping, validations)

    def test_final_byte_limit_reduces_whole_candidates_deterministically(self):
        candidates = _candidates(25, source_size=3_400)
        finalists = tuple(TournamentFinalist(f"G{1 + ((i - 1) % 5)}", 1 + ((i - 1) // 5), item) for i, item in enumerate(candidates, start=1))
        first = build_tournament_final_prompt(finalists, self.context)
        second = build_tournament_final_prompt(finalists, self.context)
        self.assertEqual(first.status, "READY_FOR_LLM")
        self.assertLessEqual(first.prompt.prompt_bytes, MAX_FINAL_PROMPT_BYTES)
        self.assertGreater(first.prompt.reduced_candidate_count, 0)
        self.assertIn("TOURNAMENT_FINAL_CANDIDATES_REDUCED_FOR_BYTE_LIMIT", first.prompt.reason_codes)
        self.assertEqual(first.prompt.source_slice_ids, second.prompt.source_slice_ids)

    def test_final_response_rejects_unknown_duplicate_and_noncontinuous_rank(self):
        finalists = tuple(TournamentFinalist("G1", 1 + ((i - 1) % 5), item) for i, item in enumerate(_candidates(10), start=1))
        prompt = build_tournament_final_prompt(finalists, self.context).prompt
        valid = _valid_findings(prompt)
        bad_unknown = json.loads(json.dumps(valid)); bad_unknown[0]["source_slice_id"] = "S999"
        bad_duplicate = json.loads(json.dumps(valid)); bad_duplicate[-1]["source_slice_id"] = bad_duplicate[0]["source_slice_id"]
        bad_rank = json.loads(json.dumps(valid)); bad_rank[1]["rank"] = 3
        for rows in (bad_unknown, bad_duplicate, bad_rank):
            self.assertFalse(validate_tournament_response({"findings": rows}, prompt).valid)


class Phase2SourceTournamentExecutionTests(unittest.TestCase):
    def test_retryable_error_retries_once_and_nonretryable_does_not(self):
        prompt = build_tournament_group_prompt(TournamentGroup("G1", _candidates(5)), _context()).prompt
        attempts = {"count": 0}

        class RetryTransport:
            def send(self, payload, api_key, timeout_seconds):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise Phase2ClassifiedTransportError("TIMEOUT")
                return make_provider_response(_valid_findings(prompt))

        retried = execute_tournament_prompt_with_single_retry(prompt, Phase2DeepSeekConfig(), RetryTransport, "SYNTHETIC")
        self.assertEqual(retried.status, "VALIDATED")
        self.assertEqual((retried.transport_audit.request_count, retried.transport_audit.retry_count), (2, 1))

        calls = {"count": 0}

        class NoRetryTransport:
            def send(self, payload, api_key, timeout_seconds):
                calls["count"] += 1
                raise Phase2ClassifiedTransportError("HTTP_OTHER_4XX")

        stopped = execute_tournament_prompt_with_single_retry(prompt, Phase2DeepSeekConfig(), NoRetryTransport, "SYNTHETIC")
        self.assertEqual(stopped.status, "TRANSPORT_ERROR")
        self.assertEqual(calls["count"], 1)

    def test_schema_invalid_is_not_retried(self):
        prompt = build_tournament_group_prompt(TournamentGroup("G1", _candidates(5)), _context()).prompt
        calls = {"count": 0}

        class InvalidTransport:
            def send(self, payload, api_key, timeout_seconds):
                calls["count"] += 1
                return make_provider_response([])

        result = execute_tournament_prompt_with_single_retry(prompt, Phase2DeepSeekConfig(), InvalidTransport, "SYNTHETIC")
        self.assertEqual(result.status, "INVALID_RESPONSE")
        self.assertEqual(calls["count"], 1)

    def test_c88_survives_group_and_final_fake_topk(self):
        candidates = _candidates(100)

        def router(prompt):
            force = "S88" if "S88" in prompt.source_slice_ids else None
            response = make_provider_response(_valid_findings(prompt, force_slice=force))
            return lambda: _ResponseTransport(response)

        result = run_tournament_offline(candidates, _context(), router)
        assigned = [group.group_alias for group in result.grouping.groups for item in group.candidates if item.candidate_id == "C88"]
        self.assertEqual(len(assigned), 1)
        self.assertIn("C88", {item.candidate.candidate_id for item in result.finalists})
        self.assertIn("S88", {item.source_slice_id for item in result.final_llm_result.validation.findings})
        self.assertEqual(result.readiness.readiness_status, READY)
        self.assertFalse(result.readiness.provider_free_text_used)
        self.assertEqual(json.loads(serialize_tournament_readiness(result.readiness))["final_topk_count"], 5)

    def test_group_failure_yields_not_ready_and_no_final(self):
        candidates = _candidates(21)

        def router(prompt):
            rows = [] if prompt.group_alias == "G2" else _valid_findings(prompt)
            return lambda: _ResponseTransport(make_provider_response(rows))

        result = run_tournament_offline(candidates, _context(), router)
        self.assertEqual(result.readiness.readiness_status, NOT_READY)
        self.assertIn(GROUP_INCOMPLETE, result.readiness.reason_codes)
        self.assertIsNone(result.final_llm_result)

    def test_future_budget_is_frozen(self):
        self.assertEqual(
            tournament_future_request_budget(15),
            {
                "per_sample_group_initial_max": 5,
                "per_sample_final_initial_max": 1,
                "per_sample_deepseek_base_max": 6,
                "per_sample_deepseek_with_retry_max": 12,
                "batch_deepseek_base_max": 90,
                "batch_deepseek_with_retry_max": 180,
                "batch_rpc_max": 16,
                "batch_source_network_max": 45,
            },
        )


if __name__ == "__main__":
    unittest.main()

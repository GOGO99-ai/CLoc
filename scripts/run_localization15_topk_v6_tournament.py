"""Gate-1 offline runner for PHASE2_SOURCE_TOURNAMENT_TOPK_V6.

This entry point deliberately has no live mode.  It exercises the complete
group/final contract with FakeTransport and synthetic aliases only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Mapping

from src.cloc_v2.phase2_source_tournament import (
    READY,
    TournamentCandidate,
    TournamentEvidence,
    TournamentEvidenceContext,
    TournamentPrompt,
    TournamentResultChain,
    make_provider_response,
    run_tournament_offline,
    serialize_tournament_readiness,
    tournament_future_request_budget,
)


EXPERIMENT_TYPE = "SYNTHETIC_OFFLINE_PHASE2_SOURCE_TOURNAMENT_CHECK"


def build_synthetic_candidates(count: int = 100) -> tuple[TournamentCandidate, ...]:
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 100:
        raise ValueError("SYNTHETIC_TOURNAMENT_CANDIDATE_COUNT_INVALID")
    return tuple(
        TournamentCandidate(
            candidate_id=f"C{index}",
            source_slice_id=f"S{index}",
            file_alias=f"F{1 + ((index - 1) % 9)}",
            contract_alias=f"K{1 + ((index - 1) % 7)}",
            function_signature=f"function_{index}(uint256)",
            visibility="public" if index % 2 else "external",
            start_line=index * 3,
            end_line=index * 3 + 2,
            _source_text=f"function function_{index}(uint256 value) public {{ value; }}",
            evidence_ids=("E1", "E2") if index % 3 == 0 else ("E1",),
            candidate_sources=("DIRECT_STRICT_CHAIN_SOURCE",) if index <= 5 else ("GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK",),
            view_ranks={
                "ASSET_RESULT_RELEVANCE_VIEW": index,
                "CALL_CAUSAL_PROXIMITY_VIEW": index,
                "EXECUTED_SOURCE_COVERAGE_VIEW": index,
                "IDENTITY_EVIDENCE_VIEW": index,
            },
            rrf_rank=index,
            rrf_score=f"{1 / (60 + index):.12f}",
            shortest_call_distance=(index - 1) % 6,
            call_relation_grade="STRICT_CHAIN_NODE" if index <= 5 else "UNDIRECTED_CALL_CORRIDOR",
            strict_chain_direct_evidence=index <= 5,
            selector_exact_identity=index <= 5,
            coarse_fallback=index > 5,
            asset_result_endpoint=index % 4 == 0,
            call_asset_joint_support=index % 3 == 0,
            address_match_evidence="RUNTIME_ADDRESS_MATCH",
            executed_call_node=True,
            uncertainty_codes=() if index <= 5 else ("FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK",),
        )
        for index in range(1, count + 1)
    )


def build_synthetic_context() -> TournamentEvidenceContext:
    return TournamentEvidenceContext(
        "T1",
        (TournamentResultChain("Q1", "Synthetic result chain used only for contract validation."),),
        (
            TournamentEvidence("E1", "Q1", "CALL", "A1", "A2", "Synthetic CALL edge reaches the result corridor."),
            TournamentEvidence("E2", "Q1", "ASSET", "A2", "A3", "Synthetic ASSET edge reaches the result endpoint."),
        ),
    )


@dataclass
class _FakeTransport:
    response: Mapping[str, Any]
    calls: list[Mapping[str, Any]]

    def send(self, request_payload: Mapping[str, Any], api_key: str, timeout_seconds: int) -> Mapping[str, Any]:
        self.calls.append(dict(request_payload))
        return self.response


class SyntheticTournamentTransportRouter:
    def __init__(self) -> None:
        self.request_count = 0

    @staticmethod
    def _findings(prompt: TournamentPrompt) -> list[dict[str, Any]]:
        ids = list(prompt.source_slice_ids)
        chosen = ids[: prompt.expected_findings]
        if "S88" in ids and "S88" not in chosen:
            chosen[-1] = "S88"
        return [
            {
                "rank": rank,
                "source_slice_id": source_slice_id,
                "evidence_ids": [prompt.evidence_by_slice[source_slice_id][0]],
            }
            for rank, source_slice_id in enumerate(chosen, start=1)
        ]

    def for_prompt(self, prompt: TournamentPrompt):
        response = make_provider_response(self._findings(prompt))

        def factory() -> _FakeTransport:
            self.request_count += 1
            return _FakeTransport(response, [])

        return factory


def run_synthetic_offline_readiness(candidate_count: int = 100) -> dict[str, Any]:
    candidates = build_synthetic_candidates(candidate_count)
    router = SyntheticTournamentTransportRouter()
    result = run_tournament_offline(candidates, build_synthetic_context(), router.for_prompt)
    assignments = [
        group.group_alias
        for group in result.grouping.groups
        for candidate in group.candidates
        if candidate.candidate_id == "C88"
    ]
    finalist_ids = [item.candidate.candidate_id for item in result.finalists]
    final_slice_ids = (
        [item.source_slice_id for item in result.final_llm_result.validation.findings]
        if result.final_llm_result is not None and result.final_llm_result.validation is not None
        else []
    )
    return {
        "experiment_type": EXPERIMENT_TYPE,
        "scientific_experiment_result": False,
        "rpc_request_count": 0,
        "source_network_request_count": 0,
        "deepseek_request_count": 0,
        "fake_transport_invocation_count": router.request_count,
        "readiness": result.readiness.to_dict(),
        "c88_path": {
            "assigned_group_count": len(assignments),
            "group_alias": assignments[0] if len(assignments) == 1 else None,
            "entered_group_top5": "C88" in finalist_ids,
            "entered_finalists": "C88" in finalist_ids,
            "entered_global_top5": "S88" in final_slice_ids,
        },
        "future_request_budget": tournament_future_request_budget(15),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline-only Top-K v6 tournament readiness check")
    parser.add_argument("command", nargs="?", default="offline-synthetic", choices=("offline-synthetic",))
    args = parser.parse_args(argv)
    if args.command != "offline-synthetic":
        raise ValueError("TOURNAMENT_GATE1_OFFLINE_ONLY")
    report = run_synthetic_offline_readiness()
    print(f"readiness_status={report['readiness']['readiness_status']}")
    print(f"input_candidate_count={report['readiness']['input_candidate_count']}")
    print(f"group_count={report['readiness']['group_count']}")
    print(f"final_topk_count={report['readiness']['final_topk_count']}")
    return 0 if report["readiness"]["readiness_status"] == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())


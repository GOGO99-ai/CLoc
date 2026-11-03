"""Baseline suspiciousness equations over validated function coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .schema import CoverageCase, TestRecord


SUPPORTED_METHODS = ("tarantula", "ochiai", "prfl", "prfl_malo")


@dataclass(frozen=True)
class RankedFunction:
    function_id: str
    score: float
    rank: int
    is_fault: bool
    ef: int
    ep: int
    nf: int
    np: int

    def to_dict(self) -> dict[str, object]:
        return {
            "function_id": self.function_id,
            "score": self.score,
            "rank": self.rank,
            "is_fault": self.is_fault,
            "ef": self.ef,
            "ep": self.ep,
            "nf": self.nf,
            "np": self.np,
        }


def spectrum_counts(case: CoverageCase) -> dict[str, tuple[int, int, int, int]]:
    candidates = case.candidate_function_ids
    failing = tuple(item for item in case.tests if item.outcome == "fail")
    passing = tuple(item for item in case.tests if item.outcome == "pass")
    output: dict[str, tuple[int, int, int, int]] = {}
    for function_id in candidates:
        ef = sum(function_id in item.covered_function_ids for item in failing)
        ep = sum(function_id in item.covered_function_ids for item in passing)
        output[function_id] = (ef, ep, len(failing) - ef, len(passing) - ep)
    return output


def tarantula_score(ef: float, ep: float, nf: float, np: float) -> float:
    fail_rate = ef / (ef + nf) if ef + nf else 0.0
    pass_rate = ep / (ep + np) if ep + np else 0.0
    denominator = fail_rate + pass_rate
    return fail_rate / denominator if denominator else 0.0


def ochiai_score(ef: float, ep: float, nf: float, _np: float) -> float:
    denominator = math.sqrt((ef + nf) * (ef + ep))
    return ef / denominator if denominator else 0.0


def _score_spectrum(case: CoverageCase, scorer: Callable[[float, float, float, float], float]) -> dict[str, float]:
    return {function_id: scorer(*counts) for function_id, counts in spectrum_counts(case).items()}


def _normalize_max(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in values}
    return {key: value / maximum for key, value in values.items()}


def _prfl_stationary_scores(
    *,
    functions: tuple[str, ...],
    tests: tuple[TestRecord, ...],
    call_edges: tuple[tuple[str, str], ...],
    failing: bool,
    damping: float = 0.7,
    alpha: float = 0.001,
    delta: float = 1.0,
    iterations: int = 25,
) -> dict[str, float]:
    covered = {function_id for test in tests for function_id in test.covered_function_ids}
    method_ids = tuple(function_id for function_id in functions if function_id in covered)
    if not method_ids or not tests:
        return {}
    method_set = set(method_ids)
    test_ids = tuple(test.transaction_hash for test in tests)
    test_by_id = {test.transaction_hash: test for test in tests}
    method_to_tests = {
        function_id: tuple(test.transaction_hash for test in tests if function_id in test.covered_function_ids)
        for function_id in method_ids
    }
    incoming: dict[str, list[tuple[str, float]]] = {function_id: [] for function_id in method_ids}
    for source, target in call_edges:
        if source in method_set and target in method_set:
            incoming[target].append((source, 1.0))
            incoming[source].append((target, delta))
    if failing:
        raw_teleport = {
            test_id: 1.0 / len(set(test_by_id[test_id].covered_function_ids) & method_set)
            for test_id in test_ids
        }
        total = sum(raw_teleport.values())
        teleport = {key: value / total for key, value in raw_teleport.items()}
    else:
        teleport = {test_id: 1.0 / len(test_ids) for test_id in test_ids}
    denominator = len(method_ids) + len(test_ids)
    method_rank = {function_id: 1.0 / denominator for function_id in method_ids}
    test_rank = {test_id: 1.0 / denominator for test_id in test_ids}
    for _ in range(iterations):
        updated_methods: dict[str, float] = {}
        for function_id in method_ids:
            from_tests = sum(
                test_rank[test_id]
                / len(set(test_by_id[test_id].covered_function_ids) & method_set)
                for test_id in test_ids
                if function_id in test_by_id[test_id].covered_function_ids
            )
            from_calls = sum(weight * method_rank[source] for source, weight in incoming[function_id])
            updated_methods[function_id] = damping * (from_tests + alpha * from_calls)
        updated_tests: dict[str, float] = {}
        for test_id in test_ids:
            from_methods = sum(
                method_rank[function_id] / len(method_to_tests[function_id])
                for function_id in method_ids
                if test_id in method_to_tests[function_id]
            )
            updated_tests[test_id] = damping * from_methods + (1.0 - damping) * teleport[test_id]
        method_rank = _normalize_max(updated_methods)
        test_rank = _normalize_max(updated_tests)
    return _normalize_max(method_rank)


def prfl_scores(case: CoverageCase) -> dict[str, float]:
    """Reproduce the reference PRFL PageRank + PR_Ochiai calculation."""
    all_functions = tuple(item.function_id for item in case.functions)
    failing = tuple(item for item in case.tests if item.outcome == "fail")
    passing = tuple(item for item in case.tests if item.outcome == "pass")
    fail_scores = _prfl_stationary_scores(
        functions=all_functions,
        tests=failing,
        call_edges=case.call_edges,
        failing=True,
    )
    pass_scores = _prfl_stationary_scores(
        functions=all_functions,
        tests=passing,
        call_edges=case.call_edges,
        failing=False,
    )
    result: dict[str, float] = {}
    for function_id in case.candidate_function_ids:
        pr_ef = fail_scores.get(function_id, 0.0) * len(failing)
        pr_ep = pass_scores.get(function_id, 0.0) * len(passing)
        pr_nf = len(failing) - pr_ef
        pr_np = len(passing) - pr_ep
        result[function_id] = ochiai_score(pr_ef, pr_ep, pr_nf, pr_np)
    return result


def _weighted_personalized_pagerank(
    adjacency: Mapping[str, tuple[tuple[str, float], ...]],
    personalization: Mapping[str, float],
    *,
    alpha: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1.0e-12,
) -> dict[str, float]:
    nodes = tuple(adjacency)
    if not nodes:
        return {}
    rank = {node: 1.0 / len(nodes) for node in nodes}
    for _ in range(iterations):
        dangling = sum(rank[node] for node in nodes if not adjacency[node])
        updated = {
            node: (1.0 - alpha) * personalization.get(node, 0.0)
            + alpha * dangling * personalization.get(node, 0.0)
            for node in nodes
        }
        for source in nodes:
            total_weight = sum(weight for _, weight in adjacency[source])
            if total_weight <= 0:
                continue
            for target, weight in adjacency[source]:
                updated[target] += alpha * rank[source] * weight / total_weight
        error = sum(abs(updated[node] - rank[node]) for node in nodes)
        rank = updated
        if error < tolerance:
            break
    return rank


def prfl_malo_scores(case: CoverageCase) -> dict[str, float]:
    """Compatibility implementation of the public MALO repository's PRFL.py."""
    candidates = case.candidate_function_ids
    all_function_ids = tuple(item.function_id for item in case.functions)
    test_nodes = tuple(f"case:{item.transaction_hash}" for item in case.tests)
    function_nodes = tuple(f"function:{item}" for item in all_function_ids)
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in test_nodes + function_nodes}
    coverage_sizes: dict[str, int] = {}
    for test, test_node in zip(case.tests, test_nodes):
        covered = tuple(item for item in test.covered_function_ids if item in all_function_ids)
        if not covered:
            raise ValueError(f"MALO PRFL requires non-empty candidate coverage for {test.transaction_hash}")
        coverage_sizes[test_node] = len(covered)
        weight = 1.0 / len(covered)
        for function_id in covered:
            function_node = f"function:{function_id}"
            adjacency[test_node].append((function_node, weight))
            adjacency[function_node].append((test_node, weight))
    inverse_total = sum(1.0 / coverage_sizes[node] for node in test_nodes)
    personalization = {
        node: (1.0 / coverage_sizes[node] / inverse_total if node in coverage_sizes else 0.0)
        for node in adjacency
    }
    page_rank = _weighted_personalized_pagerank(
        {key: tuple(value) for key, value in adjacency.items()}, personalization
    )
    function_total = sum(page_rank.get(node, 0.0) for node in function_nodes)
    weights = {
        function_id: page_rank.get(f"function:{function_id}", 0.0) / function_total
        for function_id in all_function_ids
    }
    counts = spectrum_counts(case)
    total_cases = len(case.tests)
    output: dict[str, float] = {}
    for function_id in candidates:
        ef, ep, _nf, _np = counts[function_id]
        weighted_ef = weights[function_id] * ef
        weighted_ep = weights[function_id] * ep
        output[function_id] = ochiai_score(
            weighted_ef,
            weighted_ep,
            total_cases - weighted_ef,
            total_cases - weighted_ep,
        )
    return output


def rank_case(case: CoverageCase, method: str) -> tuple[RankedFunction, ...]:
    method = method.lower()
    if method == "tarantula":
        scores = _score_spectrum(case, tarantula_score)
    elif method == "ochiai":
        scores = _score_spectrum(case, ochiai_score)
    elif method == "prfl":
        scores = prfl_scores(case)
    elif method == "prfl_malo":
        scores = prfl_malo_scores(case)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")
    return rank_scores(case, scores)


def rank_scores(case: CoverageCase, scores: Mapping[str, float]) -> tuple[RankedFunction, ...]:
    expected = set(case.candidate_function_ids)
    if set(scores) != expected:
        missing = sorted(expected - set(scores))
        extra = sorted(set(scores) - expected)
        raise ValueError(f"Score keys do not match candidates; missing={missing}, extra={extra}")
    if not all(math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("Scores must all be finite")
    counts = spectrum_counts(case)
    order = {function_id: index for index, function_id in enumerate(case.candidate_function_ids)}
    sorted_ids = sorted(scores, key=lambda function_id: (-scores[function_id], order[function_id]))
    faults = set(case.fault_function_ids)
    return tuple(
        RankedFunction(
            function_id=function_id,
            score=float(scores[function_id]),
            rank=rank,
            is_fault=function_id in faults,
            ef=counts[function_id][0],
            ep=counts[function_id][1],
            nf=counts[function_id][2],
            np=counts[function_id][3],
        )
        for rank, function_id in enumerate(sorted_ids, start=1)
    )


def serialize_rankings(rows: Iterable[RankedFunction]) -> list[dict[str, object]]:
    return [item.to_dict() for item in rows]

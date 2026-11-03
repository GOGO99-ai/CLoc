"""Common Recall@K, MFR and MAR evaluation for comparison methods."""

from __future__ import annotations

from typing import Mapping, Sequence

from .ranking import RankedFunction


def evaluate_rankings(
    rankings: Mapping[str, Sequence[RankedFunction]],
    *,
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, object]:
    if not rankings:
        raise ValueError("No case rankings supplied")
    hit_counts = {k: 0 for k in top_ks}
    first_ranks: list[float] = []
    average_ranks: list[float] = []
    case_rows: list[dict[str, object]] = []
    for case_id, rows in rankings.items():
        if not rows:
            raise ValueError(f"Empty ranking for {case_id}")
        fault_ranks = [item.rank for item in rows if item.is_fault]
        if not fault_ranks:
            raise ValueError(f"No labeled fault appears in ranking for {case_id}")
        first_rank = float(min(fault_ranks))
        average_rank = float(sum(fault_ranks) / len(fault_ranks))
        first_ranks.append(first_rank)
        average_ranks.append(average_rank)
        for k in top_ks:
            if first_rank <= k:
                hit_counts[k] += 1
        case_rows.append(
            {
                "case_id": case_id,
                "first_fault_rank": first_rank,
                "average_fault_rank": average_rank,
                "fault_count": len(fault_ranks),
                "candidate_count": len(rows),
            }
        )
    denominator = len(rankings)
    return {
        "case_count": denominator,
        "recall": {f"recall_at_{k}": hit_counts[k] / denominator for k in top_ks},
        "hit_counts": {f"top_{k}": hit_counts[k] for k in top_ks},
        "mfr": sum(first_ranks) / denominator,
        "mar": sum(average_ranks) / denominator,
        "rank_policy": "ordinal_after_stable_score_sort",
        "cases": case_rows,
    }

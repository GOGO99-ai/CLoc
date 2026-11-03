"""Small, label-free helper for sharing exact function evidence in one sequence."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping, Sequence, TypeAlias


CandidateIdentity: TypeAlias = tuple[str, str, int, int]
_ANONYMOUS_SEQUENCE_ALIAS = re.compile(r"G[1-9][0-9]*")


def _candidate_identity(row: Mapping[str, Any]) -> CandidateIdentity:
    relative_file = row.get("relative_file")
    signature = row.get("function_signature")
    start_line = row.get("start_line")
    end_line = row.get("end_line")
    if (
        not isinstance(relative_file, str)
        or not relative_file
        or relative_file.startswith(("/", "\\"))
        or ".." in relative_file.replace("\\", "/").split("/")
        or not isinstance(signature, str)
        or not signature
        or not isinstance(start_line, int)
        or isinstance(start_line, bool)
        or not isinstance(end_line, int)
        or isinstance(end_line, bool)
        or start_line < 1
        or end_line < start_line
    ):
        raise ValueError("SEQUENCE_CANDIDATE_IDENTITY_INVALID")
    return (relative_file.replace("\\", "/"), signature, start_line, end_line)


def aggregate_sequence_exact_support(
    candidate_pools_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    sequence_alias_by_sample: Mapping[str, str],
) -> dict[str, tuple[CandidateIdentity, ...]]:
    """Return candidate identities supported by an exact match in a sibling sample.

    The caller supplies anonymous group aliases such as ``G1``.  Incident names,
    fault labels, cause text and ground-truth locations are neither accepted nor
    inspected by this function.
    """
    if set(candidate_pools_by_sample) != set(sequence_alias_by_sample):
        raise ValueError("SEQUENCE_SAMPLE_SET_MISMATCH")

    available: dict[str, set[CandidateIdentity]] = {}
    exact_owners: dict[tuple[str, CandidateIdentity], set[str]] = defaultdict(set)
    for sample_alias, rows in candidate_pools_by_sample.items():
        group_alias = sequence_alias_by_sample.get(sample_alias)
        if (
            not isinstance(sample_alias, str)
            or not sample_alias
            or not isinstance(group_alias, str)
            or _ANONYMOUS_SEQUENCE_ALIAS.fullmatch(group_alias) is None
            or not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes))
        ):
            raise ValueError("SEQUENCE_SUPPORT_INPUT_INVALID")
        sample_candidates: set[CandidateIdentity] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("SEQUENCE_SUPPORT_INPUT_INVALID")
            identity = _candidate_identity(row)
            sample_candidates.add(identity)
            if row.get("selector_match_status") == "SELECTOR_EXACT_MATCH":
                exact_owners[(group_alias, identity)].add(sample_alias)
        available[sample_alias] = sample_candidates

    support: dict[str, tuple[CandidateIdentity, ...]] = {}
    for sample_alias in sorted(candidate_pools_by_sample):
        group_alias = sequence_alias_by_sample[sample_alias]
        support[sample_alias] = tuple(sorted(
            identity
            for identity in available[sample_alias]
            if exact_owners.get((group_alias, identity), set()) - {sample_alias}
        ))
    return support

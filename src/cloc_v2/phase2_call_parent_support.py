"""Map exact abnormal CALL identities to redacted unique-parent keys."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .normal_distance import CallStep
from .source_index import selector_for_signature


def _source_key(value: Mapping[str, Any]) -> tuple[str, str, int, int] | None:
    file_name = value.get("file_path") or value.get("relative_file")
    signature = value.get("signature") or value.get("function_signature")
    start = value.get("start_line")
    end = value.get("end_line")
    if (
        not isinstance(file_name, str)
        or not isinstance(signature, str)
        or not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        return None
    return file_name.replace("\\", "/"), signature, start, end


def map_abnormal_calls_to_unique_parent_keys(
    source_slices: Mapping[str, Any],
    abnormal_calls: Iterable[CallStep],
) -> tuple[tuple[str, str, str, int, int], ...]:
    """Return contract-scoped parent keys backed by one exact abnormal CALL."""
    if not isinstance(source_slices, Mapping) or source_slices.get("schema_version") != "source_request_slices_v1":
        raise ValueError("source_slices must use source_request_slices_v1")
    calls = frozenset(abnormal_calls)
    if any(not isinstance(item, CallStep) for item in calls):
        raise ValueError("abnormal_calls must contain only CallStep values")
    metadata = source_slices.get("metadata")
    union = metadata.get("candidate_union") if isinstance(metadata, Mapping) else None
    rows = union.get("candidates") if isinstance(union, Mapping) else None
    requests = source_slices.get("requests")
    if not isinstance(rows, list) or not isinstance(requests, list):
        return ()

    public_by_child = {
        key: row
        for row in rows
        if isinstance(row, Mapping) and (key := _source_key(row)) is not None
    }
    result: set[tuple[str, str, str, int, int]] = set()
    for request in requests:
        if not isinstance(request, Mapping):
            continue
        child_key = _source_key(request)
        public = public_by_child.get(child_key)
        parent = request.get("unique_parent_function_identity")
        public_parent = public.get("unique_parent_function_identity") if isinstance(public, Mapping) else None
        address = request.get("source_address")
        if not isinstance(parent, Mapping) or not isinstance(public_parent, Mapping) or not isinstance(address, str):
            continue
        parent_key = _source_key(parent)
        public_parent_key = _source_key(public_parent)
        contract_alias = public_parent.get("contract_alias")
        if (
            parent_key is None
            or parent_key != public_parent_key
            or parent.get("direct_internal_callee_count") != 1
            or public_parent.get("direct_internal_callee_count") != 1
            or not isinstance(contract_alias, str)
        ):
            continue
        selector = selector_for_signature(parent_key[1])
        try:
            parent_call = CallStep(address, selector)
        except ValueError:
            continue
        if parent_call in calls:
            result.add((contract_alias, *parent_key))
    return tuple(sorted(result))


__all__ = ["map_abnormal_calls_to_unique_parent_keys"]

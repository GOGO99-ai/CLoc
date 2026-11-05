"""Resolve runtime selectors to available function identities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from cloc.source.index import selector_for_signature


_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_SELECTOR = re.compile(r"^0x[0-9a-fA-F]{8}$")
_SIGNATURE = re.compile(r"^([A-Za-z_$][A-Za-z0-9_$]*)\(.*\)$")


@dataclass(frozen=True)
class OfflineFunctionCandidate:
    """One explicitly supplied offline function-index entry."""

    blockchain_chain_id: int
    effective_address: str
    function_name: str
    signature: str
    selector: str | None = None
    evidence_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockchain_chain_id": self.blockchain_chain_id,
            "effective_address": self.effective_address,
            "function_name": self.function_name,
            "signature": self.signature,
            "selector": self.selector,
            "evidence_source": self.evidence_source,
        }


@dataclass(frozen=True)
class FunctionIdentityResolution:
    """Stable result of resolving one call's function identity."""

    blockchain_chain_id: int | None
    runtime_address: str | None
    effective_address: str | None
    original_function_name: str | None
    trace_selector: str | None
    selector: str | None
    resolution_source: str
    candidate_signatures: tuple[str, ...]
    candidate_selectors: tuple[str, ...]
    candidate_evidence_sources: tuple[str, ...]
    usable_for_anchor: bool
    status: str
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockchain_chain_id": self.blockchain_chain_id,
            "runtime_address": self.runtime_address,
            "effective_address": self.effective_address,
            "original_function_name": self.original_function_name,
            "trace_selector": self.trace_selector,
            "selector": self.selector,
            "resolution_source": self.resolution_source,
            "candidate_signatures": list(self.candidate_signatures),
            "candidate_selectors": list(self.candidate_selectors),
            "candidate_evidence_sources": list(self.candidate_evidence_sources),
            "usable_for_anchor": self.usable_for_anchor,
            "status": self.status,
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class _NormalizedCandidate:
    signature: str
    selector: str
    evidence_source: str | None


def resolve_function_identity(
    *,
    blockchain_chain_id: Any,
    runtime_address: Any,
    effective_address: Any = None,
    trace_selector: Any = None,
    function_name: Any = None,
    offline_candidates: Iterable[OfflineFunctionCandidate] = (),
) -> FunctionIdentityResolution:
    """Resolve a selector using only explicit trace and offline-index input."""
    chain_id = _normalize_chain_id(blockchain_chain_id)
    runtime = _normalize_address(runtime_address)
    effective_was_supplied = effective_address is not None
    effective = _normalize_address(effective_address) if effective_was_supplied else runtime
    original_name = function_name if isinstance(function_name, str) else None
    reasons: list[str] = []

    if chain_id is None:
        reasons.append("BLOCKCHAIN_CHAIN_ID_INVALID")
    if runtime is None:
        reasons.append("RUNTIME_ADDRESS_INVALID")
    if effective_was_supplied and effective is None:
        reasons.append("EFFECTIVE_ADDRESS_INVALID")

    native_claimed = trace_selector is not None
    native = _normalize_selector(trace_selector)
    if native_claimed and native is None:
        reasons.append("TRACE_SELECTOR_INVALID")

    if native is not None:
        return _resolution(
            chain_id=chain_id,
            runtime=runtime,
            effective=effective,
            original_name=original_name,
            trace_selector=native,
            selector=native,
            resolution_source="TRACE_NATIVE",
            candidates=(),
            reasons=reasons,
        )

    if not isinstance(original_name, str) or not original_name.strip():
        reasons.append("FUNCTION_NAME_MISSING")
        return _resolution(
            chain_id=chain_id,
            runtime=runtime,
            effective=effective,
            original_name=original_name,
            trace_selector=None,
            selector=None,
            resolution_source="UNRESOLVED",
            candidates=(),
            reasons=reasons,
        )

    supplied = tuple(offline_candidates)
    address_evidence = False
    matching: list[_NormalizedCandidate] = []
    if chain_id is not None and effective is not None:
        for candidate in supplied:
            candidate_chain = _normalize_chain_id(candidate.blockchain_chain_id)
            candidate_address = _normalize_address(candidate.effective_address)
            if candidate_chain != chain_id or candidate_address != effective:
                continue
            address_evidence = True
            if candidate.function_name != original_name:
                continue
            normalized_candidate = _normalize_candidate(candidate)
            if normalized_candidate is not None:
                matching.append(normalized_candidate)

    candidates = tuple(
        sorted(
            set(matching),
            key=lambda item: (item.signature, item.selector, item.evidence_source or ""),
        )
    )
    selectors = sorted({candidate.selector for candidate in candidates})
    if not supplied or not address_evidence:
        reasons.append("SELECTOR_RECOVERY_SOURCE_MISSING")
    elif not candidates:
        same_name_exists = any(
            _normalize_chain_id(candidate.blockchain_chain_id) == chain_id
            and _normalize_address(candidate.effective_address) == effective
            and candidate.function_name == original_name
            for candidate in supplied
        )
        reasons.append(
            "SELECTOR_RECOVERY_NO_VALID_SELECTOR"
            if same_name_exists
            else "SELECTOR_RECOVERY_FUNCTION_NOT_FOUND"
        )
    elif len(selectors) > 1:
        reasons.append("SELECTOR_RECOVERY_AMBIGUOUS")

    selector = selectors[0] if len(selectors) == 1 else None
    source = "OFFLINE_ABI_UNIQUE" if selector is not None else "UNRESOLVED"
    return _resolution(
        chain_id=chain_id,
        runtime=runtime,
        effective=effective,
        original_name=original_name,
        trace_selector=None,
        selector=selector,
        resolution_source=source,
        candidates=candidates,
        reasons=reasons,
    )


def serialize_function_identity(result: FunctionIdentityResolution) -> str:
    """Return stable JSON without writing an artifact."""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def _normalize_chain_id(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _normalize_address(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and _ADDRESS.fullmatch(value) else None


def _normalize_selector(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and _SELECTOR.fullmatch(value) else None


def _normalize_candidate(candidate: OfflineFunctionCandidate) -> _NormalizedCandidate | None:
    if not isinstance(candidate.signature, str):
        return None
    signature_match = _SIGNATURE.fullmatch(candidate.signature)
    if signature_match is None or signature_match.group(1) != candidate.function_name:
        return None
    derived = _normalize_selector(selector_for_signature(candidate.signature))
    supplied = _normalize_selector(candidate.selector) if candidate.selector is not None else None
    if candidate.selector is not None and supplied is None:
        return None
    if supplied is not None and supplied != derived:
        return None
    selector = supplied or derived
    if selector is None:
        return None
    evidence_source = (
        candidate.evidence_source
        if isinstance(candidate.evidence_source, str) and candidate.evidence_source
        else None
    )
    return _NormalizedCandidate(candidate.signature, selector, evidence_source)


def _resolution(
    *,
    chain_id: int | None,
    runtime: str | None,
    effective: str | None,
    original_name: str | None,
    trace_selector: str | None,
    selector: str | None,
    resolution_source: str,
    candidates: tuple[_NormalizedCandidate, ...],
    reasons: list[str],
) -> FunctionIdentityResolution:
    uncertainties = tuple(sorted(set(reasons)))
    usable = selector is not None and not uncertainties
    return FunctionIdentityResolution(
        blockchain_chain_id=chain_id,
        runtime_address=runtime,
        effective_address=effective,
        original_function_name=original_name,
        trace_selector=trace_selector,
        selector=selector,
        resolution_source=resolution_source,
        candidate_signatures=tuple(sorted({item.signature for item in candidates})),
        candidate_selectors=tuple(sorted({item.selector for item in candidates})),
        candidate_evidence_sources=tuple(
            sorted({item.evidence_source for item in candidates if item.evidence_source is not None})
        ),
        usable_for_anchor=usable,
        status="RESOLVED" if usable else "UNCERTAIN",
        uncertainties=uncertainties,
    )


__all__ = [
    "FunctionIdentityResolution",
    "OfflineFunctionCandidate",
    "resolve_function_identity",
    "serialize_function_identity",
]

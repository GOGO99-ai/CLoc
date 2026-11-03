"""Usable-clean-evidence exact normal recognition before phase-1 LLM use."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .formal_normal_baseline_loader import FormalNormalTransactionIdentity
from .normal_baseline import NormalBaseline
from .normal_distance import compute_chain_distance
from .normal_features import NormalFeatureExtractionResult
from .strict_result_chains import StrictResultChainsResult


DIRECT_NORMAL_REASON = "KNOWN_NORMAL_TRANSACTION_SELF_MATCH"
STRUCTURAL_EXACT_COLLISION_REASON = "STRUCTURAL_EXACT_MATCH_WITHOUT_TRANSACTION_IDENTITY"
_ZERO = Decimal(0)
_REFERENCE_ORIGIN_STATUSES = {
    "NOT_APPLICABLE",
    "NO_STRUCTURAL_EXACT_MATCH",
    "SELF_ORIGIN_EXACT_MATCH",
    "OTHER_ORIGIN_EXACT_MATCH",
    "QUERY_NOT_IN_NORMAL_LIBRARY",
    "QUERY_ORIGIN_UNRESOLVED",
    "MAPPING_INCOMPLETE",
    "MAPPING_CONFLICT",
}


@dataclass(frozen=True)
class Phase1DirectChainMatch:
    """One sanitized strict-chain audit; identities remain internal only."""

    chain_alias: str
    strict_status: str
    clean_feature: bool
    usable_clean: bool
    structural_exact_match: bool
    self_origin_exact_match: bool
    reference_origin_status: str
    call_distance: Decimal | None
    asset_distance: Decimal | None
    chain_distance: Decimal | None
    reason_codes: tuple[str, ...]
    _strict_chain_id: str = field(repr=False)
    _query_record_id: str | None = field(default=None, repr=False)
    _structural_match_record_ids: tuple[str, ...] = field(default=(), repr=False)
    _self_match_record_ids: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.chain_alias, str) or not self.chain_alias.startswith("Q"):
            raise ValueError("DIRECT_CHAIN_ALIAS_INVALID")
        if self.strict_status not in {"READY", "UNCERTAIN"}:
            raise ValueError("DIRECT_CHAIN_STATUS_INVALID")
        if self.reference_origin_status not in _REFERENCE_ORIGIN_STATUSES:
            raise ValueError("DIRECT_REFERENCE_ORIGIN_STATUS_INVALID")
        for name in ("call_distance", "asset_distance", "chain_distance"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _distance(value))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        structural_ids = tuple(sorted(set(self._structural_match_record_ids)))
        self_ids = tuple(sorted(set(self._self_match_record_ids)))
        object.__setattr__(self, "_structural_match_record_ids", structural_ids)
        object.__setattr__(self, "_self_match_record_ids", self_ids)
        if self.usable_clean != (self.strict_status == "READY" and self.clean_feature):
            raise ValueError("DIRECT_USABLE_CLEAN_INCONSISTENT")
        if self.structural_exact_match != (
            self.usable_clean
            and self.call_distance == _ZERO
            and self.asset_distance == _ZERO
            and self.chain_distance == _ZERO
            and bool(structural_ids)
        ):
            raise ValueError("DIRECT_CHAIN_MATCH_INVARIANT_VIOLATED")
        if self.self_origin_exact_match != bool(self_ids):
            raise ValueError("DIRECT_SELF_MATCH_INVARIANT_VIOLATED")
        if self.self_origin_exact_match and (
            not self.structural_exact_match or not set(self_ids).issubset(structural_ids)
        ):
            raise ValueError("DIRECT_SELF_MATCH_NOT_STRUCTURAL")

    @property
    def exact_match(self) -> bool:
        """Compatibility alias for the structural, not identity-qualified, match."""
        return self.structural_exact_match

    @property
    def matched_normal_record_id(self) -> str | None:
        return self._structural_match_record_ids[0] if self._structural_match_record_ids else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_alias": self.chain_alias,
            "strict_status": self.strict_status,
            "clean_feature": self.clean_feature,
            "usable_clean": self.usable_clean,
            "structural_exact_match": self.structural_exact_match,
            "self_origin_exact_match": self.self_origin_exact_match,
            "reference_origin_status": self.reference_origin_status,
            "call_distance": _distance_text(self.call_distance),
            "asset_distance": _distance_text(self.asset_distance),
            "chain_distance": _distance_text(self.chain_distance),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class Phase1DirectNormalMatch:
    """Transaction decision from all usable clean chain evidence only."""

    direct_normal: bool
    final_route: str
    direct_normal_reason: str | None
    query_normal_identity_status: str
    known_transaction_direct_normal: bool
    structural_exact_collision: bool
    decision_kind: str
    coverage_rate: Decimal | None
    critical_missing_count: int | None
    result_candidate_count: int
    strict_chain_count: int
    clean_ready_chain_count: int
    exact_matched_clean_chain_count: int
    self_origin_exact_matched_clean_chain_count: int
    clean_unmatched_chain_count: int
    uncertain_chain_count: int
    unusable_ready_chain_count: int
    chain_matches: tuple[Phase1DirectChainMatch, ...]
    reason_codes: tuple[str, ...]

    SCHEMA_VERSION = "phase1_direct_normal_v4"

    def __post_init__(self) -> None:
        if self.final_route not in {"EARLY_NORMAL", "PROCEED_PHASE2", "CONTINUE_STANDARD_PHASE1"}:
            raise ValueError("DIRECT_NORMAL_ROUTE_INVALID")
        if self.query_normal_identity_status not in {
            "KNOWN_NORMAL_ORIGIN", "NOT_IN_NORMAL_LIBRARY", "UNRESOLVED"
        }:
            raise ValueError("DIRECT_QUERY_IDENTITY_STATUS_INVALID")
        if self.decision_kind not in {
            "KNOWN_TRANSACTION_DIRECT_NORMAL",
            "STRUCTURAL_EXACT_COLLISION_PHASE2",
            "CONTINUE_STANDARD_PHASE1",
        }:
            raise ValueError("DIRECT_DECISION_KIND_INVALID")
        if self.coverage_rate is not None:
            object.__setattr__(self, "coverage_rate", _distance(self.coverage_rate))
        if self.critical_missing_count is not None and (
            not isinstance(self.critical_missing_count, int)
            or isinstance(self.critical_missing_count, bool)
            or self.critical_missing_count < 0
        ):
            raise ValueError("DIRECT_CRITICAL_MISSING_COUNT_INVALID")
        matches = tuple(sorted(self.chain_matches, key=lambda item: _alias_number(item.chain_alias)))
        if len({item.chain_alias for item in matches}) != len(matches):
            raise ValueError("DIRECT_CHAIN_ALIAS_DUPLICATE")
        object.__setattr__(self, "chain_matches", matches)
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        if self.strict_chain_count != len(matches):
            raise ValueError("DIRECT_STRICT_CHAIN_COUNT_MISMATCH")
        if self.clean_ready_chain_count != sum(item.usable_clean for item in matches):
            raise ValueError("DIRECT_CLEAN_READY_COUNT_MISMATCH")
        if self.exact_matched_clean_chain_count != sum(item.structural_exact_match for item in matches):
            raise ValueError("DIRECT_EXACT_CLEAN_COUNT_MISMATCH")
        if self.self_origin_exact_matched_clean_chain_count != sum(
            item.self_origin_exact_match for item in matches
        ):
            raise ValueError("DIRECT_SELF_EXACT_CLEAN_COUNT_MISMATCH")
        if self.clean_unmatched_chain_count != sum(
            item.usable_clean and not item.structural_exact_match for item in matches
        ):
            raise ValueError("DIRECT_CLEAN_UNMATCHED_COUNT_MISMATCH")
        if self.uncertain_chain_count != sum(item.strict_status == "UNCERTAIN" for item in matches):
            raise ValueError("DIRECT_UNCERTAIN_COUNT_MISMATCH")
        if self.unusable_ready_chain_count != sum(item.strict_status == "READY" and not item.clean_feature for item in matches):
            raise ValueError("DIRECT_UNUSABLE_READY_COUNT_MISMATCH")
        all_structural_exact = (
            self.result_candidate_count > 0
            and self.strict_chain_count > 0
            and self.coverage_rate == Decimal(1)
            and self.critical_missing_count == 0
            and self.clean_ready_chain_count > 0
            and self.clean_unmatched_chain_count == 0
            and self.exact_matched_clean_chain_count == self.clean_ready_chain_count
        )
        expected_direct = (
            all_structural_exact
            and self.query_normal_identity_status == "KNOWN_NORMAL_ORIGIN"
            and self.self_origin_exact_matched_clean_chain_count == self.clean_ready_chain_count
            and self.reason_codes == (DIRECT_NORMAL_REASON,)
        )
        expected_collision = all_structural_exact and not expected_direct
        if (
            self.direct_normal != expected_direct
            or self.known_transaction_direct_normal != expected_direct
            or self.structural_exact_collision != expected_collision
            or self.direct_normal != (self.final_route == "EARLY_NORMAL")
        ):
            raise ValueError("DIRECT_NORMAL_INVARIANT_VIOLATED")
        if self.direct_normal_reason != (DIRECT_NORMAL_REASON if self.direct_normal else None):
            raise ValueError("DIRECT_NORMAL_REASON_INCONSISTENT")
        expected_kind = (
            "KNOWN_TRANSACTION_DIRECT_NORMAL" if expected_direct
            else "STRUCTURAL_EXACT_COLLISION_PHASE2" if expected_collision
            else "CONTINUE_STANDARD_PHASE1"
        )
        expected_route = "EARLY_NORMAL" if expected_direct else "PROCEED_PHASE2" if expected_collision else "CONTINUE_STANDARD_PHASE1"
        if self.decision_kind != expected_kind or self.final_route != expected_route:
            raise ValueError("DIRECT_DECISION_KIND_INCONSISTENT")
        if expected_collision and STRUCTURAL_EXACT_COLLISION_REASON not in self.reason_codes:
            raise ValueError("DIRECT_COLLISION_REASON_MISSING")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "direct_normal": self.direct_normal,
            "final_route": self.final_route,
            "direct_normal_reason": self.direct_normal_reason,
            "query_normal_identity_status": self.query_normal_identity_status,
            "known_transaction_direct_normal": self.known_transaction_direct_normal,
            "structural_exact_collision": self.structural_exact_collision,
            "decision_kind": self.decision_kind,
            "coverage_rate": _distance_text(self.coverage_rate),
            "critical_missing_count": self.critical_missing_count,
            "result_candidate_count": self.result_candidate_count,
            "strict_chain_count": self.strict_chain_count,
            "clean_ready_chain_count": self.clean_ready_chain_count,
            "exact_matched_clean_chain_count": self.exact_matched_clean_chain_count,
            "self_origin_exact_matched_clean_chain_count": self.self_origin_exact_matched_clean_chain_count,
            "clean_unmatched_chain_count": self.clean_unmatched_chain_count,
            "uncertain_chain_count": self.uncertain_chain_count,
            "unusable_ready_chain_count": self.unusable_ready_chain_count,
            "chain_matches": [item.to_dict() for item in self.chain_matches],
            "reason_codes": list(self.reason_codes),
        }


def evaluate_phase1_direct_normal_match(
    strict_result_chains: StrictResultChainsResult,
    query_features: NormalFeatureExtractionResult,
    normal_baseline: NormalBaseline,
    query_normal_identity: FormalNormalTransactionIdentity,
    normal_record_transaction_ids: Mapping[str, str],
) -> Phase1DirectNormalMatch:
    """Allow direct normal only for identity-proven, self-origin exact matches."""
    if not isinstance(strict_result_chains, StrictResultChainsResult):
        raise ValueError("DIRECT_STRICT_RESULT_CHAINS_INVALID")
    if not isinstance(query_features, NormalFeatureExtractionResult):
        raise ValueError("DIRECT_QUERY_FEATURES_INVALID")
    if not isinstance(normal_baseline, NormalBaseline):
        raise ValueError("DIRECT_NORMAL_BASELINE_INVALID")
    if not isinstance(query_normal_identity, FormalNormalTransactionIdentity):
        raise ValueError("DIRECT_QUERY_NORMAL_IDENTITY_INVALID")
    if not isinstance(normal_record_transaction_ids, Mapping):
        raise ValueError("DIRECT_NORMAL_RECORD_ORIGIN_MAPPING_INVALID")

    mapping = dict(normal_record_transaction_ids)
    mapping_reasons = _mapping_reasons(normal_baseline, mapping)

    coverage = _coverage(strict_result_chains.coverage.get("coverage_rate"))
    critical = _nonnegative_int(strict_result_chains.coverage.get("critical_missing_count"))
    candidate_count = _nonnegative_int(strict_result_chains.coverage.get("result_candidate_count"))
    if candidate_count is None:
        candidate_count = len(strict_result_chains.result_statuses)
    global_reasons: set[str] = set()
    if candidate_count == 0:
        global_reasons.add("NO_RESULT_CANDIDATE")
    if coverage != Decimal(1):
        global_reasons.add("RESULT_COVERAGE_INCOMPLETE")
    if critical != 0:
        global_reasons.add("CRITICAL_EVIDENCE_MISSING")
    global_reasons.update(mapping_reasons)
    if query_normal_identity.status == "UNRESOLVED":
        global_reasons.add("QUERY_NORMAL_ORIGIN_UNRESOLVED")

    chains = tuple(sorted(strict_result_chains.chains, key=lambda item: item.strict_chain_id))
    if not chains:
        global_reasons.add("NO_STRICT_RESULT_CHAIN")
    statuses: dict[str, list[Any]] = {}
    for status in query_features.chain_statuses:
        statuses.setdefault(status.strict_chain_id, []).append(status)
    records: dict[str, list[Any]] = {}
    for record in query_features.records:
        records.setdefault(record.record_id, []).append(record)

    audits: list[Phase1DirectChainMatch] = []
    used_records: set[str] = set()
    clean_unmatched = 0
    for index, chain in enumerate(chains, start=1):
        reasons: set[str] = set()
        clean = False
        query_record = None
        if chain.status == "READY":
            status_values = statuses.get(chain.strict_chain_id, [])
            if len(status_values) != 1:
                reasons.add("CHAIN_FEATURE_MAPPING_MISSING" if not status_values else "CHAIN_FEATURE_MAPPING_DUPLICATE")
            else:
                status = status_values[0]
                record_values = records.get(status.record_id, [])
                if len(record_values) != 1 or not status.record_created:
                    reasons.add("CHAIN_FEATURE_RECORD_MISSING" if not record_values else "CHAIN_FEATURE_RECORD_DUPLICATE")
                elif status.record_id in used_records:
                    reasons.add("CHAIN_FEATURE_RECORD_REUSED")
                else:
                    used_records.add(status.record_id)
                    query_record = record_values[0]
                    clean = status.status == "EXTRACTED_CLEAN" and not status.reasons and not query_record.uncertainties
                    if not clean:
                        reasons.add("QUERY_FEATURE_NOT_CLEAN")
        else:
            reasons.add("STRICT_CHAIN_NOT_READY")

        call_distance = asset_distance = chain_distance = None
        structural_match_ids: tuple[str, ...] = ()
        self_match_ids: tuple[str, ...] = ()
        reference_origin_status = "NOT_APPLICABLE"
        if chain.status == "READY" and clean and query_record is not None:
            group = next(
                (item for item in normal_baseline.groups if item.main_anchor == query_record.main_anchor),
                None,
            )
            if group is None or not group.records:
                reasons.add("NORMAL_GROUP_NOT_FOUND")
                reasons.add("CLEAN_CHAIN_EXACT_MATCH_NOT_FOUND")
                clean_unmatched += 1
                reference_origin_status = "NO_STRUCTURAL_EXACT_MATCH"
            else:
                distances = tuple(sorted(
                    (compute_chain_distance(query_record, record) for record in group.records),
                    key=lambda item: (
                        item.chain_distance,
                        item.call_distance,
                        item.asset_distance,
                        item.normal_record_id,
                    ),
                ))
                nearest = distances[0]
                call_distance = nearest.call_distance
                asset_distance = nearest.asset_distance
                chain_distance = nearest.chain_distance
                structural_match_ids = tuple(
                    item.normal_record_id for item in distances
                    if item.call_distance == _ZERO
                    and item.asset_distance == _ZERO
                    and item.chain_distance == _ZERO
                )
                if structural_match_ids:
                    if "NORMAL_REFERENCE_ORIGIN_MAPPING_MISSING" in mapping_reasons:
                        reference_origin_status = "MAPPING_INCOMPLETE"
                        reasons.add("NORMAL_REFERENCE_ORIGIN_MAPPING_MISSING")
                    elif "NORMAL_REFERENCE_ORIGIN_MAPPING_CONFLICT" in mapping_reasons:
                        reference_origin_status = "MAPPING_CONFLICT"
                        reasons.add("NORMAL_REFERENCE_ORIGIN_MAPPING_CONFLICT")
                    elif query_normal_identity.status == "KNOWN_NORMAL_ORIGIN":
                        self_match_ids = tuple(
                            record_id for record_id in structural_match_ids
                            if mapping.get(record_id) == query_normal_identity.normal_origin_id
                        )
                        if self_match_ids:
                            reference_origin_status = "SELF_ORIGIN_EXACT_MATCH"
                        else:
                            reference_origin_status = "OTHER_ORIGIN_EXACT_MATCH"
                            reasons.add("SELF_ORIGIN_EXACT_MATCH_NOT_FOUND")
                    elif query_normal_identity.status == "NOT_IN_NORMAL_LIBRARY":
                        reference_origin_status = "QUERY_NOT_IN_NORMAL_LIBRARY"
                    else:
                        reference_origin_status = "QUERY_ORIGIN_UNRESOLVED"
                        reasons.add("QUERY_NORMAL_ORIGIN_UNRESOLVED")
                else:
                    reference_origin_status = "NO_STRUCTURAL_EXACT_MATCH"
                    if call_distance != _ZERO:
                        reasons.add("EXACT_CALL_DISTANCE_NOT_ZERO")
                    if asset_distance != _ZERO:
                        reasons.add("EXACT_ASSET_DISTANCE_NOT_ZERO")
                    if chain_distance != _ZERO:
                        reasons.add("EXACT_CHAIN_DISTANCE_NOT_ZERO")
                    reasons.add("CLEAN_CHAIN_EXACT_MATCH_NOT_FOUND")
                    clean_unmatched += 1
        structural_exact = bool(structural_match_ids)
        self_exact = bool(self_match_ids)
        audits.append(
            Phase1DirectChainMatch(
                f"Q{index}",
                chain.status,
                clean,
                chain.status == "READY" and clean,
                structural_exact,
                self_exact,
                reference_origin_status,
                call_distance,
                asset_distance,
                chain_distance,
                tuple(reasons),
                chain.strict_chain_id,
                query_record.record_id if query_record is not None else None,
                structural_match_ids,
                self_match_ids,
            )
        )

    clean_ready = sum(item.usable_clean for item in audits)
    exact_clean = sum(item.structural_exact_match for item in audits)
    self_exact_clean = sum(item.self_origin_exact_match for item in audits)
    if clean_ready == 0:
        global_reasons.add("NO_USABLE_CLEAN_READY_CHAIN")
    if clean_unmatched:
        global_reasons.add("CLEAN_CHAIN_EXACT_MATCH_NOT_FOUND")
    all_structural_exact = (
        candidate_count > 0
        and bool(chains)
        and coverage == Decimal(1)
        and critical == 0
        and clean_ready > 0
        and clean_unmatched == 0
        and exact_clean == clean_ready
    )
    direct = (
        all_structural_exact
        and query_normal_identity.status == "KNOWN_NORMAL_ORIGIN"
        and query_normal_identity.usable_for_direct_normal_identity
        and self_exact_clean == clean_ready
        and not mapping_reasons
    )
    collision = all_structural_exact and not direct
    if direct:
        reasons = (DIRECT_NORMAL_REASON,)
        final_route = "EARLY_NORMAL"
        decision_kind = "KNOWN_TRANSACTION_DIRECT_NORMAL"
    elif collision:
        reasons = tuple(sorted({*global_reasons, STRUCTURAL_EXACT_COLLISION_REASON}))
        final_route = "PROCEED_PHASE2"
        decision_kind = "STRUCTURAL_EXACT_COLLISION_PHASE2"
    else:
        reasons = tuple(global_reasons)
        final_route = "CONTINUE_STANDARD_PHASE1"
        decision_kind = "CONTINUE_STANDARD_PHASE1"
    return Phase1DirectNormalMatch(
        direct,
        final_route,
        DIRECT_NORMAL_REASON if direct else None,
        query_normal_identity.status,
        direct,
        collision,
        decision_kind,
        coverage,
        critical,
        candidate_count,
        len(audits),
        clean_ready,
        exact_clean,
        self_exact_clean,
        clean_unmatched,
        sum(item.strict_status == "UNCERTAIN" for item in audits),
        sum(item.strict_status == "READY" and not item.clean_feature for item in audits),
        tuple(audits),
        reasons,
    )


def serialize_phase1_direct_normal_match(result: Phase1DirectNormalMatch) -> str:
    if not isinstance(result, Phase1DirectNormalMatch):
        raise ValueError("DIRECT_NORMAL_RESULT_INVALID")
    return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _mapping_reasons(
    baseline: NormalBaseline,
    normal_record_transaction_ids: Mapping[str, str],
) -> tuple[str, ...]:
    baseline_ids = {
        record.record_id
        for group in baseline.groups
        for record in group.records
    }
    supplied_ids = set(normal_record_transaction_ids)
    reasons: set[str] = set()
    if baseline_ids - supplied_ids:
        reasons.add("NORMAL_REFERENCE_ORIGIN_MAPPING_MISSING")
    if supplied_ids - baseline_ids or any(
        not isinstance(origin_id, str) or not origin_id
        for origin_id in normal_record_transaction_ids.values()
    ):
        reasons.add("NORMAL_REFERENCE_ORIGIN_MAPPING_CONFLICT")
    return tuple(sorted(reasons))


def _coverage(value: Any) -> Decimal | None:
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not result.is_finite() or result < 0 or result > 1:
        return None
    return result.quantize(Decimal("0.0001"))


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _distance(value: Any) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("DIRECT_DISTANCE_INVALID")
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("DIRECT_DISTANCE_INVALID") from exc
    if not result.is_finite() or result < 0 or result > 1:
        raise ValueError("DIRECT_DISTANCE_INVALID")
    return result.quantize(Decimal("0.0001"))


def _distance_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _reason_codes(values: Any) -> tuple[str, ...]:
    supplied = tuple(values)
    if any(not isinstance(item, str) or not item or not item.replace("_", "").isalnum() or item != item.upper() for item in supplied):
        raise ValueError("DIRECT_REASON_CODE_INVALID")
    return tuple(sorted(set(supplied)))


def _alias_number(value: str) -> int:
    try:
        return int(value[1:])
    except (TypeError, ValueError):
        return 0


__all__ = [
    "DIRECT_NORMAL_REASON", "STRUCTURAL_EXACT_COLLISION_REASON",
    "Phase1DirectChainMatch", "Phase1DirectNormalMatch",
    "evaluate_phase1_direct_normal_match", "serialize_phase1_direct_normal_match",
]

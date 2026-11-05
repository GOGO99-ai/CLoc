"""Gate review requests and route validated chain decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Any, Iterable, Mapping

from cloc.review.prompt import (
    ReviewPrompt,
    ReviewQueryChain,
    validate_review_response,
)


FORMAL_MIN_NORMAL_GROUP_SIZE = 3
ELIGIBILITY_STATES = frozenset({"ELIGIBLE", "INELIGIBLE"})
ROUTE_STATES = frozenset({"AWAITING_LLM", "EARLY_NORMAL", "PROCEED_RANKING"})
REASON_CODES = frozenset(
    {
        "NO_RESULT_CHAIN",
        "RESULT_COVERAGE_INCOMPLETE",
        "CRITICAL_EVIDENCE_MISSING",
        "STRICT_CHAIN_NOT_READY",
        "NO_USABLE_CLEAN_CHAIN",
        "CHAIN_SET_MISMATCH",
        "QUANTITATIVE_NOT_PASS",
        "THRESHOLD_UNAVAILABLE",
        "THRESHOLD_NON_DISCRIMINATIVE",
        "NORMAL_GROUP_TOO_SMALL",
        "FORMAL_MIN_GROUP_SIZE_MISMATCH",
        "TOP3_REFERENCE_INCOMPLETE",
        "LLM_RESPONSE_INVALID",
        "LLM_SUSPICIOUS",
        "LLM_UNCERTAIN",
        "ALL_CHAINS_NORMAL",
    }
)

_STRICT_CHAIN_ID = re.compile(r"^[a-z][a-z0-9_:-]*$")
_QUERY_ALIAS = re.compile(r"^Q[1-9][0-9]*$")
_DISTANCE_QUANTUM = Decimal("0.000000000000000001")


@dataclass(frozen=True)
class ReviewChainGateEvidence:
    """Program-owned hard-gate evidence for one strict result chain."""

    strict_chain_id: str
    strict_chain_status: str
    query_chain_alias: str
    quantitative_status: str
    dynamic_threshold: Decimal | None
    threshold_calibration_status: str
    calibration_min_group_size: int
    normal_group_size: int
    normal_reference_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.strict_chain_id, str) or _STRICT_CHAIN_ID.fullmatch(self.strict_chain_id) is None:
            raise ValueError("STRICT_CHAIN_ID_INVALID")
        if self.strict_chain_status not in {"READY", "UNCERTAIN"}:
            raise ValueError("STRICT_CHAIN_STATUS_INVALID")
        if not isinstance(self.query_chain_alias, str) or _QUERY_ALIAS.fullmatch(self.query_chain_alias) is None:
            raise ValueError("QUERY_CHAIN_ALIAS_INVALID")
        if self.quantitative_status not in {"PASS", "FAIL", "UNCERTAIN"}:
            raise ValueError("QUANTITATIVE_STATUS_INVALID")
        if self.threshold_calibration_status not in {"CALIBRATED", "UNAVAILABLE"}:
            raise ValueError("THRESHOLD_CALIBRATION_STATUS_INVALID")
        if self.dynamic_threshold is not None:
            object.__setattr__(self, "dynamic_threshold", _distance(self.dynamic_threshold))
        for name in ("calibration_min_group_size", "normal_group_size", "normal_reference_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name.upper()}_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strict_chain_id": self.strict_chain_id,
            "strict_chain_status": self.strict_chain_status,
            "query_chain_alias": self.query_chain_alias,
            "quantitative_status": self.quantitative_status,
            "dynamic_threshold": _distance_text(self.dynamic_threshold),
            "threshold_calibration_status": self.threshold_calibration_status,
            "calibration_min_group_size": self.calibration_min_group_size,
            "normal_group_size": self.normal_group_size,
            "normal_reference_count": self.normal_reference_count,
        }


@dataclass(frozen=True)
class ReviewTransactionGateEvidence:
    """Transaction coverage plus all program-owned per-chain gate evidence."""

    result_candidate_coverage_rate: Decimal
    critical_missing_count: int
    strict_chains: tuple[ReviewChainGateEvidence, ...]
    query_chains: tuple[ReviewQueryChain, ...]
    total_strict_chain_count: int | None = None
    excluded_uncertain_chain_count: int = 0

    @property
    def usable_clean_chain_count(self) -> int:
        return len(self.strict_chains)

    def __post_init__(self) -> None:
        coverage = _distance(self.result_candidate_coverage_rate)
        if not isinstance(self.critical_missing_count, int) or isinstance(self.critical_missing_count, bool) or self.critical_missing_count < 0:
            raise ValueError("CRITICAL_MISSING_COUNT_INVALID")
        strict_chains = tuple(self.strict_chains)
        query_chains = tuple(self.query_chains)
        if any(not isinstance(item, ReviewChainGateEvidence) for item in strict_chains):
            raise ValueError("STRICT_CHAIN_GATE_EVIDENCE_INVALID")
        if any(not isinstance(item, ReviewQueryChain) for item in query_chains):
            raise ValueError("REVIEW_QUERY_CHAIN_INVALID")
        excluded = self.excluded_uncertain_chain_count
        total = self.total_strict_chain_count
        if not isinstance(excluded, int) or isinstance(excluded, bool) or excluded < 0:
            raise ValueError("EXCLUDED_UNCERTAIN_CHAIN_COUNT_INVALID")
        if total is None:
            total = len(strict_chains) + excluded
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ValueError("TOTAL_STRICT_CHAIN_COUNT_INVALID")
        if total != len(strict_chains) + excluded:
            raise ValueError("STRICT_CHAIN_AUDIT_COUNT_INCONSISTENT")
        object.__setattr__(self, "result_candidate_coverage_rate", coverage)
        object.__setattr__(
            self,
            "strict_chains",
            tuple(sorted(strict_chains, key=lambda item: (item.strict_chain_id, item.query_chain_alias))),
        )
        object.__setattr__(
            self,
            "query_chains",
            tuple(sorted(query_chains, key=lambda item: item.chain_alias)),
        )
        object.__setattr__(self, "total_strict_chain_count", total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_candidate_coverage_rate": _distance_text(self.result_candidate_coverage_rate),
            "critical_missing_count": self.critical_missing_count,
            "total_strict_chain_count": self.total_strict_chain_count,
            "usable_clean_chain_count": self.usable_clean_chain_count,
            "excluded_uncertain_chain_count": self.excluded_uncertain_chain_count,
            "strict_chains": [item.to_dict() for item in self.strict_chains],
            "query_chain_aliases": [item.chain_alias for item in self.query_chains],
        }


@dataclass(frozen=True)
class ReviewEligibilityResult:
    eligibility: str
    llm_allowed: bool
    reason_codes: tuple[str, ...]
    strict_chain_ids: tuple[str, ...]
    query_chain_aliases: tuple[str, ...]
    total_strict_chain_count: int = 0
    usable_clean_chain_count: int = 0
    excluded_uncertain_chain_count: int = 0

    def __post_init__(self) -> None:
        if self.eligibility not in ELIGIBILITY_STATES:
            raise ValueError("ELIGIBILITY_STATE_INVALID")
        if self.llm_allowed != (self.eligibility == "ELIGIBLE"):
            raise ValueError("ELIGIBILITY_LLM_PERMISSION_INCONSISTENT")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(self, "strict_chain_ids", tuple(sorted(self.strict_chain_ids)))
        object.__setattr__(self, "query_chain_aliases", tuple(sorted(self.query_chain_aliases)))
        counts = (
            self.total_strict_chain_count,
            self.usable_clean_chain_count,
            self.excluded_uncertain_chain_count,
        )
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in counts):
            raise ValueError("ELIGIBILITY_CHAIN_AUDIT_COUNT_INVALID")
        if self.total_strict_chain_count != self.usable_clean_chain_count + self.excluded_uncertain_chain_count:
            raise ValueError("ELIGIBILITY_CHAIN_AUDIT_COUNT_INCONSISTENT")

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligibility": self.eligibility,
            "llm_allowed": self.llm_allowed,
            "reason_codes": list(self.reason_codes),
            "strict_chain_ids": list(self.strict_chain_ids),
            "query_chain_aliases": list(self.query_chain_aliases),
            "total_strict_chain_count": self.total_strict_chain_count,
            "usable_clean_chain_count": self.usable_clean_chain_count,
            "excluded_uncertain_chain_count": self.excluded_uncertain_chain_count,
        }


@dataclass(frozen=True)
class ReviewChainRouteDecision:
    chain_alias: str
    decision: str

    def __post_init__(self) -> None:
        if not isinstance(self.chain_alias, str) or _QUERY_ALIAS.fullmatch(self.chain_alias) is None:
            raise ValueError("QUERY_CHAIN_ALIAS_INVALID")
        if self.decision not in {"NORMAL", "SUSPICIOUS", "UNCERTAIN"}:
            raise ValueError("CHAIN_DECISION_INVALID")

    def to_dict(self) -> dict[str, str]:
        return {"chain_alias": self.chain_alias, "decision": self.decision}


@dataclass(frozen=True)
class ReviewRoutingDecision:
    eligibility: ReviewEligibilityResult
    route: str
    llm_response_processed: bool
    reason_codes: tuple[str, ...]
    chain_decisions: tuple[ReviewChainRouteDecision, ...] = ()

    SCHEMA_VERSION = "review_normality_routing_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.eligibility, ReviewEligibilityResult):
            raise ValueError("ELIGIBILITY_RESULT_INVALID")
        if self.route not in ROUTE_STATES:
            raise ValueError("ROUTE_STATE_INVALID")
        decisions = tuple(sorted(self.chain_decisions, key=lambda item: item.chain_alias))
        if any(not isinstance(item, ReviewChainRouteDecision) for item in decisions):
            raise ValueError("CHAIN_ROUTE_DECISION_INVALID")
        if len({item.chain_alias for item in decisions}) != len(decisions):
            raise ValueError("CHAIN_ROUTE_DECISION_DUPLICATE")
        if self.route == "EARLY_NORMAL" and (
            self.eligibility.eligibility != "ELIGIBLE"
            or not decisions
            or any(item.decision != "NORMAL" for item in decisions)
        ):
            raise ValueError("EARLY_NORMAL_SAFETY_INVARIANT_VIOLATED")
        if self.route == "AWAITING_LLM" and self.eligibility.eligibility != "ELIGIBLE":
            raise ValueError("AWAITING_LLM_SAFETY_INVARIANT_VIOLATED")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(self, "chain_decisions", decisions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "eligibility": self.eligibility.to_dict(),
            "route": self.route,
            "llm_response_processed": self.llm_response_processed,
            "reason_codes": list(self.reason_codes),
            "chain_decisions": [item.to_dict() for item in self.chain_decisions],
        }


def evaluate_review_eligibility(
    evidence: ReviewTransactionGateEvidence,
    prompt: ReviewPrompt | None = None,
) -> ReviewEligibilityResult:
    """Evaluate every program-owned precondition before any LLM may be used."""
    if not isinstance(evidence, ReviewTransactionGateEvidence):
        raise ValueError("TRANSACTION_GATE_EVIDENCE_INVALID")
    if prompt is not None and not isinstance(prompt, ReviewPrompt):
        raise ValueError("PROMPT_CONTRACT_INVALID")

    reasons: set[str] = set()
    strict = evidence.strict_chains
    queries = evidence.query_chains
    if evidence.total_strict_chain_count == 0:
        reasons.add("NO_RESULT_CHAIN")
    if not strict:
        reasons.add("NO_USABLE_CLEAN_CHAIN")
    if evidence.result_candidate_coverage_rate != Decimal(1):
        reasons.add("RESULT_COVERAGE_INCOMPLETE")
    if evidence.critical_missing_count != 0:
        reasons.add("CRITICAL_EVIDENCE_MISSING")
    if evidence.excluded_uncertain_chain_count or any(item.strict_chain_status != "READY" for item in strict):
        reasons.add("STRICT_CHAIN_NOT_READY")

    strict_ids = tuple(item.strict_chain_id for item in strict)
    mapped_aliases = tuple(item.query_chain_alias for item in strict)
    query_aliases = tuple(item.chain_alias for item in queries)
    if (
        len(set(strict_ids)) != len(strict_ids)
        or len(set(mapped_aliases)) != len(mapped_aliases)
        or len(set(query_aliases)) != len(query_aliases)
        or set(mapped_aliases) != set(query_aliases)
        or len(strict) != len(queries)
    ):
        reasons.add("CHAIN_SET_MISMATCH")

    query_by_alias = {item.chain_alias: item for item in queries}
    prompt_scopes = (
        {item.chain_alias: item for item in prompt.citation_whitelist.chain_scopes}
        if prompt is not None
        else None
    )
    if prompt is not None and set(prompt.allowed_chain_aliases) != set(query_aliases):
        reasons.add("CHAIN_SET_MISMATCH")

    for item in strict:
        query = query_by_alias.get(item.query_chain_alias)
        if item.quantitative_status != "PASS" or (query is not None and query.quantitative_status != "PASS"):
            reasons.add("QUANTITATIVE_NOT_PASS")
        if item.dynamic_threshold is None or item.threshold_calibration_status != "CALIBRATED":
            reasons.add("THRESHOLD_UNAVAILABLE")
        elif item.dynamic_threshold >= Decimal(1):
            reasons.add("THRESHOLD_NON_DISCRIMINATIVE")
        if item.calibration_min_group_size != FORMAL_MIN_NORMAL_GROUP_SIZE:
            reasons.add("FORMAL_MIN_GROUP_SIZE_MISMATCH")
        if item.normal_group_size < FORMAL_MIN_NORMAL_GROUP_SIZE:
            reasons.add("NORMAL_GROUP_TOO_SMALL")

        query_reference_count = len(query.normal_references) if query is not None else -1
        prompt_reference_count = (
            len(prompt_scopes[item.query_chain_alias].reference_aliases)
            if prompt_scopes is not None and item.query_chain_alias in prompt_scopes
            else query_reference_count
        )
        if (
            item.normal_reference_count != 3
            or query_reference_count != 3
            or prompt_reference_count != 3
        ):
            reasons.add("TOP3_REFERENCE_INCOMPLETE")
        if query is not None and item.dynamic_threshold is not None and query.dynamic_threshold != item.dynamic_threshold:
            reasons.add("CHAIN_SET_MISMATCH")

    frozen_reasons = tuple(sorted(reasons))
    eligible = not frozen_reasons
    return ReviewEligibilityResult(
        eligibility="ELIGIBLE" if eligible else "INELIGIBLE",
        llm_allowed=eligible,
        reason_codes=frozen_reasons,
        strict_chain_ids=tuple(sorted(set(strict_ids))),
        query_chain_aliases=tuple(sorted(set(query_aliases))),
        total_strict_chain_count=evidence.total_strict_chain_count,
        usable_clean_chain_count=len(strict),
        excluded_uncertain_chain_count=evidence.excluded_uncertain_chain_count,
    )


def decide_review_route(
    evidence: ReviewTransactionGateEvidence,
    prompt: ReviewPrompt,
    llm_response: Mapping[str, Any] | str | None = None,
) -> ReviewRoutingDecision:
    """Route only after hard gates; an LLM response cannot override them."""
    eligibility = evaluate_review_eligibility(evidence, prompt)
    if eligibility.eligibility != "ELIGIBLE":
        return ReviewRoutingDecision(
            eligibility=eligibility,
            route="PROCEED_RANKING",
            llm_response_processed=False,
            reason_codes=eligibility.reason_codes,
        )
    if llm_response is None:
        return ReviewRoutingDecision(
            eligibility=eligibility,
            route="AWAITING_LLM",
            llm_response_processed=False,
            reason_codes=(),
        )

    try:
        validated = validate_review_response(llm_response, prompt)
    except (TypeError, ValueError):
        return ReviewRoutingDecision(
            eligibility=eligibility,
            route="PROCEED_RANKING",
            llm_response_processed=True,
            reason_codes=("LLM_RESPONSE_INVALID",),
        )

    decisions = tuple(
        ReviewChainRouteDecision(item["unit_id"], item["verdict"])
        for item in validated["reviews"]
    )
    decision_values = {item.decision for item in decisions}
    if "SUSPICIOUS" in decision_values or "UNCERTAIN" in decision_values:
        reasons = []
        if "SUSPICIOUS" in decision_values:
            reasons.append("LLM_SUSPICIOUS")
        if "UNCERTAIN" in decision_values:
            reasons.append("LLM_UNCERTAIN")
        return ReviewRoutingDecision(
            eligibility=eligibility,
            route="PROCEED_RANKING",
            llm_response_processed=True,
            reason_codes=tuple(reasons),
            chain_decisions=decisions,
        )
    return ReviewRoutingDecision(
        eligibility=eligibility,
        route="EARLY_NORMAL",
        llm_response_processed=True,
        reason_codes=("ALL_CHAINS_NORMAL",),
        chain_decisions=decisions,
    )


def serialize_review_routing_decision(decision: ReviewRoutingDecision) -> str:
    """Return stable JSON without writing a runtime artifact."""
    if not isinstance(decision, ReviewRoutingDecision):
        raise ValueError("ROUTING_DECISION_INVALID")
    return json.dumps(decision.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _reason_codes(values: Iterable[str]) -> tuple[str, ...]:
    supplied = tuple(values)
    if any(item not in REASON_CODES for item in supplied):
        raise ValueError("ROUTING_REASON_CODE_INVALID")
    return tuple(sorted(set(supplied)))


def _distance(value: Any) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("DISTANCE_INVALID")
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("DISTANCE_INVALID") from exc
    if not result.is_finite() or result < 0 or result > 1:
        raise ValueError("DISTANCE_INVALID")
    return result.quantize(_DISTANCE_QUANTUM, rounding=ROUND_HALF_EVEN)


def _distance_text(value: Decimal | None) -> str | None:
    return (
        format(value.quantize(_DISTANCE_QUANTUM, rounding=ROUND_HALF_EVEN), "f")
        if value is not None
        else None
    )


__all__ = [
    "ELIGIBILITY_STATES",
    "FORMAL_MIN_NORMAL_GROUP_SIZE",
    "ReviewChainGateEvidence",
    "ReviewChainRouteDecision",
    "ReviewEligibilityResult",
    "ReviewRoutingDecision",
    "ReviewTransactionGateEvidence",
    "REASON_CODES",
    "ROUTE_STATES",
    "decide_review_route",
    "evaluate_review_eligibility",
    "serialize_review_routing_decision",
]

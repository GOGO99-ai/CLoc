"""Pure offline bridge from formal chain evidence to phase-1 review inputs.

Only caller-supplied in-memory objects are consumed.  The bridge reuses the
formal baseline query, prompt builder, and eligibility evaluator; it performs
no file, configuration, label, LLM, API, RPC, or network access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .normal_baseline import NormalBaseline, NormalBaselineGroup, query_normal_baseline
from .normal_distance import AssetBehavior, ChainFeatureRecord
from .normal_features import NormalFeatureExtractionResult
from .phase1_normality_orchestration import (
    Phase1ChainGateEvidence,
    Phase1EligibilityResult,
    Phase1TransactionGateEvidence,
    evaluate_phase1_eligibility,
)
from .phase1_normality_prompt import (
    Phase1AssetEvidence,
    Phase1CallEvidence,
    Phase1NormalReference,
    Phase1NormalityPrompt,
    Phase1QueryChain,
    SanitizedMainAnchor,
    build_phase1_normality_prompt,
)
from .strict_result_chains import StrictResultChain, StrictResultChainsResult


_AMOUNT = re.compile(r"^(?:0|[1-9][0-9]*)$")
_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")
_TRANSACTION_HASH = re.compile(r"^0x[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class Phase1BridgeIssue:
    """One sanitized machine issue; no raw source identifier is retained."""

    code: str
    severity: str
    chain_alias: str | None = None
    reference_alias: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code or not self.code.replace("_", "").isalnum() or self.code != self.code.upper():
            raise ValueError("BRIDGE_ISSUE_CODE_INVALID")
        if self.severity not in {"FATAL", "INFORMATIONAL"}:
            raise ValueError("BRIDGE_ISSUE_SEVERITY_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "chain_alias": self.chain_alias,
            "reference_alias": self.reference_alias,
        }


@dataclass(frozen=True)
class Phase1NormalityPreparation:
    """Sanitized bridge output; only eligible preparations expose a prompt."""

    preparation_status: str
    gate_evidence: Phase1TransactionGateEvidence | None
    eligibility: Phase1EligibilityResult
    prompt: Phase1NormalityPrompt | None
    issues: tuple[Phase1BridgeIssue, ...]

    SCHEMA_VERSION = "phase1_normality_preparation_v2"

    def __post_init__(self) -> None:
        if self.preparation_status not in {"READY_FOR_LLM", "INELIGIBLE"}:
            raise ValueError("PREPARATION_STATUS_INVALID")
        if not isinstance(self.eligibility, Phase1EligibilityResult):
            raise ValueError("ELIGIBILITY_RESULT_INVALID")
        if any(not isinstance(item, Phase1BridgeIssue) for item in self.issues):
            raise ValueError("BRIDGE_ISSUES_INVALID")
        ordered = tuple(sorted(set(self.issues), key=_issue_order))
        object.__setattr__(self, "issues", ordered)
        if self.preparation_status == "READY_FOR_LLM":
            if self.prompt is None or self.gate_evidence is None or self.eligibility.eligibility != "ELIGIBLE":
                raise ValueError("READY_FOR_LLM_INVARIANT_VIOLATED")
            if any(item.severity == "FATAL" for item in ordered):
                raise ValueError("READY_FOR_LLM_HAS_FATAL_ISSUE")
        elif self.prompt is not None:
            raise ValueError("INELIGIBLE_PROMPT_FORBIDDEN")
        if self.prompt is not None and self.eligibility.eligibility != "ELIGIBLE":
            raise ValueError("PROMPT_REQUIRES_ELIGIBLE")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "preparation_status": self.preparation_status,
            "gate_evidence": self.gate_evidence.to_dict() if self.gate_evidence is not None else None,
            "eligibility": self.eligibility.to_dict(),
            "prompt": self.prompt.to_dict() if self.prompt is not None else None,
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class _ChainPlan:
    raw_chain: StrictResultChain
    chain_alias: str
    sanitized_gate_id: str
    query_record: ChainFeatureRecord | None
    baseline_query: Any | None
    group: NormalBaselineGroup | None
    references: tuple[ChainFeatureRecord, ...]


def prepare_phase1_normality_review(
    strict_result_chains: StrictResultChainsResult,
    query_features: NormalFeatureExtractionResult,
    normal_baseline: NormalBaseline,
    normal_record_transaction_ids: Mapping[str, str],
) -> Phase1NormalityPreparation:
    """Translate formal evidence into a safely gated, fully sanitized review."""
    _validate_inputs(strict_result_chains, query_features, normal_baseline, normal_record_transaction_ids)
    issues: list[Phase1BridgeIssue] = []
    coverage = _coverage(strict_result_chains.coverage, issues)
    critical_missing = _critical_missing(strict_result_chains.coverage, issues)

    raw_chains = tuple(sorted(strict_result_chains.chains, key=lambda item: item.strict_chain_id))
    duplicate_chain_ids = _duplicates(item.strict_chain_id for item in raw_chains)
    if duplicate_chain_ids:
        issues.append(Phase1BridgeIssue("STRICT_CHAIN_ID_DUPLICATE", "FATAL"))

    statuses: dict[str, list[Any]] = {}
    for status in query_features.chain_statuses:
        statuses.setdefault(status.strict_chain_id, []).append(status)
    records: dict[str, list[ChainFeatureRecord]] = {}
    for record in query_features.records:
        records.setdefault(record.record_id, []).append(record)
    baseline_records: dict[str, list[ChainFeatureRecord]] = {}
    for group in normal_baseline.groups:
        for record in group.records:
            baseline_records.setdefault(record.record_id, []).append(record)
    if any(len(values) > 1 for values in baseline_records.values()):
        issues.append(Phase1BridgeIssue("NORMAL_RECORD_ID_DUPLICATE", "FATAL"))

    raw_chain_ids = {item.strict_chain_id for item in raw_chains}
    if set(statuses) - raw_chain_ids:
        issues.append(Phase1BridgeIssue("FEATURE_CHAIN_SET_MISMATCH", "FATAL"))
    referenced_record_ids = {
        status.record_id
        for values in statuses.values()
        for status in values
        if getattr(status, "record_created", False)
    }
    if any(not record.uncertainties and record.record_id not in referenced_record_ids for record in query_features.records):
        issues.append(Phase1BridgeIssue("CLEAN_FEATURE_RECORD_ORPHAN", "FATAL"))

    usable_inputs: list[tuple[StrictResultChain, ChainFeatureRecord]] = []
    excluded_uncertain_chain_count = 0
    used_query_record_ids: set[str] = set()
    for chain in raw_chains:
        status_values = statuses.get(chain.strict_chain_id, [])
        if chain.status == "UNCERTAIN":
            excluded_uncertain_chain_count += 1
            issues.append(Phase1BridgeIssue("UNCERTAIN_CHAIN_EXCLUDED", "INFORMATIONAL"))
            continue
        if chain.status != "READY":
            issues.append(Phase1BridgeIssue("STRICT_CHAIN_STATUS_INVALID", "FATAL"))
            continue
        if len(status_values) != 1:
            code = "FEATURE_CHAIN_MAPPING_MISSING" if not status_values else "FEATURE_CHAIN_MAPPING_DUPLICATE"
            issues.append(Phase1BridgeIssue(code, "FATAL"))
            continue
        status = status_values[0]
        if status.status != "EXTRACTED_CLEAN" or not status.record_created:
            excluded_uncertain_chain_count += 1
            issues.append(Phase1BridgeIssue("NON_CLEAN_CHAIN_EXCLUDED", "INFORMATIONAL"))
            continue
        record_values = records.get(status.record_id, [])
        if len(record_values) != 1:
            code = "FEATURE_RECORD_MISSING" if len(record_values) < 1 else "FEATURE_RECORD_DUPLICATE"
            issues.append(Phase1BridgeIssue(code, "FATAL"))
            continue
        query_record = record_values[0]
        if status.reasons or query_record.uncertainties:
            issues.append(Phase1BridgeIssue("CLEAN_FEATURE_STATUS_CONTRADICTORY", "FATAL"))
            continue
        if not query_record.call_steps or not query_record.asset_behaviors:
            issues.append(Phase1BridgeIssue("CLEAN_FEATURE_RECORD_INCOMPLETE", "FATAL"))
            continue
        if status.record_id in used_query_record_ids:
            issues.append(Phase1BridgeIssue("QUERY_RECORD_REUSED", "FATAL"))
            continue
        used_query_record_ids.add(status.record_id)
        usable_inputs.append((chain, query_record))

    if any(item.severity == "FATAL" for item in issues):
        return _ineligible_without_gate(issues)

    chain_aliases = {
        chain.strict_chain_id: f"Q{index}"
        for index, (chain, _) in enumerate(usable_inputs, start=1)
    }

    plans: list[_ChainPlan] = []
    for index, (chain, query_record) in enumerate(usable_inputs, start=1):
        alias = chain_aliases[chain.strict_chain_id]
        baseline_query = query_normal_baseline(normal_baseline, query_record)
        group = None
        references: tuple[ChainFeatureRecord, ...] = ()
        if baseline_query.query_record_id != query_record.record_id or baseline_query.main_anchor != query_record.main_anchor:
            issues.append(Phase1BridgeIssue("NORMAL_QUERY_IDENTITY_CONTRADICTORY", "FATAL", alias))
        else:
            matching_groups = tuple(group for group in normal_baseline.groups if group.main_anchor == query_record.main_anchor)
            if len(matching_groups) != 1:
                code = "NORMAL_GROUP_NOT_FOUND" if not matching_groups else "NORMAL_GROUP_DUPLICATE"
                severity = "INFORMATIONAL" if not matching_groups else "FATAL"
                issues.append(Phase1BridgeIssue(code, severity, alias))
            else:
                group = matching_groups[0]
                found: list[ChainFeatureRecord] = []
                matched_record_ids = tuple(
                    match.normal_record_id for match in baseline_query.top_matches
                )
                if len(set(matched_record_ids)) != len(matched_record_ids):
                    issues.append(
                        Phase1BridgeIssue(
                            "NORMAL_REFERENCE_DUPLICATE_WITHIN_CHAIN",
                            "FATAL",
                            alias,
                        )
                    )
                for match in baseline_query.top_matches:
                    values = baseline_records.get(match.normal_record_id, [])
                    if len(values) != 1:
                        code = "NORMAL_RECORD_NOT_FOUND" if not values else "NORMAL_RECORD_ID_DUPLICATE"
                        issues.append(Phase1BridgeIssue(code, "FATAL", alias))
                        continue
                    reference = values[0]
                    if reference.main_anchor != query_record.main_anchor or reference not in group.records:
                        issues.append(Phase1BridgeIssue("NORMAL_REFERENCE_ANCHOR_MISMATCH", "FATAL", alias))
                        continue
                    if match.normal_record_id not in normal_record_transaction_ids:
                        issues.append(Phase1BridgeIssue("NORMAL_TRANSACTION_MAPPING_MISSING", "FATAL", alias))
                        continue
                    transaction_id = normal_record_transaction_ids[match.normal_record_id]
                    if not isinstance(transaction_id, str) or not transaction_id:
                        issues.append(Phase1BridgeIssue("NORMAL_TRANSACTION_MAPPING_INVALID", "FATAL", alias))
                        continue
                    found.append(reference)
                references = tuple(found)
        plans.append(
            _ChainPlan(
                raw_chain=chain,
                chain_alias=alias,
                sanitized_gate_id=f"chain_q{index:03d}",
                query_record=query_record,
                baseline_query=baseline_query,
                group=group,
                references=references,
            )
        )

    alias_maps = _alias_maps(plans, normal_record_transaction_ids)
    query_chains: list[Phase1QueryChain] = []
    gate_chains: list[Phase1ChainGateEvidence] = []
    for plan in plans:
        query, plan_issues = _build_query_chain(plan, alias_maps, normal_record_transaction_ids)
        issues.extend(plan_issues)
        if query is not None:
            query_chains.append(query)
        gate, gate_issue = _build_gate(plan, query)
        if gate_issue is not None:
            issues.append(gate_issue)
        if gate is not None:
            gate_chains.append(gate)

    fatal = any(item.severity == "FATAL" for item in issues)
    if coverage is None or critical_missing is None:
        return _ineligible_without_gate(issues)
    try:
        gate_evidence = Phase1TransactionGateEvidence(
            result_candidate_coverage_rate=coverage,
            critical_missing_count=critical_missing,
            strict_chains=tuple(gate_chains),
            query_chains=tuple(query_chains),
            total_strict_chain_count=len(raw_chains),
            excluded_uncertain_chain_count=excluded_uncertain_chain_count,
        )
    except ValueError:
        issues.append(Phase1BridgeIssue("TRANSACTION_GATE_CONSTRUCTION_FAILED", "FATAL"))
        return _ineligible_without_gate(issues)

    candidate_prompt: Phase1NormalityPrompt | None = None
    fatal = any(item.severity == "FATAL" for item in issues)
    if not fatal and query_chains and len(query_chains) == len(plans):
        try:
            candidate_prompt = build_phase1_normality_prompt(tuple(query_chains))
        except ValueError:
            issues.append(Phase1BridgeIssue("PROMPT_CONSTRUCTION_FAILED", "FATAL"))
    eligibility = evaluate_phase1_eligibility(gate_evidence, candidate_prompt)
    fatal = any(item.severity == "FATAL" for item in issues)
    if candidate_prompt is not None and eligibility.eligibility == "ELIGIBLE" and not fatal:
        return Phase1NormalityPreparation(
            "READY_FOR_LLM", gate_evidence, eligibility, candidate_prompt, tuple(issues)
        )
    return Phase1NormalityPreparation(
        "INELIGIBLE", gate_evidence, eligibility, None, tuple(issues)
    )


def serialize_phase1_normality_preparation(preparation: Phase1NormalityPreparation) -> str:
    """Return a stable sanitized JSON document without writing an artifact."""
    if not isinstance(preparation, Phase1NormalityPreparation):
        raise ValueError("PREPARATION_INVALID")
    text = json.dumps(preparation.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if _ADDRESS.search(text) or _TRANSACTION_HASH.search(text):
        raise ValueError("SERIALIZED_PREPARATION_CONTAINS_RAW_CHAIN_IDENTIFIER")
    return text


def _build_query_chain(
    plan: _ChainPlan,
    aliases: Mapping[str, Mapping[str, str]],
    transaction_ids: Mapping[str, str],
) -> tuple[Phase1QueryChain | None, tuple[Phase1BridgeIssue, ...]]:
    issues: list[Phase1BridgeIssue] = []
    record = plan.query_record
    result = plan.baseline_query
    if record is None or result is None or plan.group is None:
        return None, ()
    if result.status != "PASS" or result.group_threshold is None or result.nearest_distance is None:
        return None, ()
    query_calls = tuple(
        Phase1CallEvidence(
            aliases["evidence"][f"{plan.chain_alias}:query:call:{index}"],
            aliases["address"][step.effective_address],
            step.selector,
            index,
        )
        for index, step in enumerate(record.call_steps, start=1)
    )
    amount_index, amount_value, amount_reason = _amount_observation(plan.raw_chain, record)
    query_assets = []
    for index, behavior in enumerate(sorted(record.asset_behaviors, key=_asset_order), start=1):
        uncertainty = (amount_reason,) if amount_reason is not None and (amount_index is None or amount_index == index) else ()
        query_assets.append(
            Phase1AssetEvidence(
                aliases["evidence"][f"{plan.chain_alias}:query:asset:{index}"],
                aliases["asset"][behavior.asset_identifier],
                behavior.direction,
                "NATIVE" if behavior.asset_identifier == "NATIVE" else "TOKEN",
                amount_value if amount_index == index else None,
                uncertainty,
            )
        )
    if amount_reason is not None:
        issues.append(Phase1BridgeIssue(amount_reason, "INFORMATIONAL", plan.chain_alias))

    references: list[Phase1NormalReference] = []
    record_by_id = {item.record_id: item for item in plan.references}
    for match in result.top_matches:
        reference_record = record_by_id.get(match.normal_record_id)
        if reference_record is None:
            continue
        occurrence_key = _reference_occurrence_key(
            plan.chain_alias,
            match.rank,
            match.normal_record_id,
        )
        reference_alias = aliases["reference"][occurrence_key]
        reference_calls = tuple(
            Phase1CallEvidence(
                aliases["evidence"][f"{occurrence_key}:call:{index}"],
                aliases["address"][step.effective_address],
                step.selector,
                index,
            )
            for index, step in enumerate(reference_record.call_steps, start=1)
        )
        reference_assets = tuple(
            Phase1AssetEvidence(
                aliases["evidence"][f"{occurrence_key}:asset:{index}"],
                aliases["asset"][behavior.asset_identifier],
                behavior.direction,
                "NATIVE" if behavior.asset_identifier == "NATIVE" else "TOKEN",
            )
            for index, behavior in enumerate(sorted(reference_record.asset_behaviors, key=_asset_order), start=1)
        )
        references.append(
            Phase1NormalReference(
                reference_alias=reference_alias,
                normal_transaction_alias=aliases["transaction"][transaction_ids[match.normal_record_id]],
                call_evidence=reference_calls,
                asset_evidence=reference_assets,
                chain_distance=match.distance.chain_distance,
                call_distance=match.distance.call_distance,
                asset_distance=match.distance.asset_distance,
                uncertainties=reference_record.uncertainties,
            )
        )
    try:
        return (
            Phase1QueryChain(
                chain_alias=plan.chain_alias,
                main_anchor=SanitizedMainAnchor(
                    record.main_anchor.blockchain_chain_id,
                    aliases["address"][record.main_anchor.effective_address],
                    record.main_anchor.selector,
                ),
                dynamic_threshold=result.group_threshold,
                nearest_distance=result.nearest_distance,
                quantitative_status=result.status,
                call_evidence=query_calls,
                asset_evidence=tuple(query_assets),
                normal_references=tuple(references),
                uncertainties=tuple(sorted(set(record.uncertainties))),
            ),
            tuple(issues),
        )
    except ValueError:
        issues.append(Phase1BridgeIssue("QUERY_CHAIN_CONSTRUCTION_FAILED", "FATAL", plan.chain_alias))
        return None, tuple(issues)


def _build_gate(
    plan: _ChainPlan,
    query: Phase1QueryChain | None,
) -> tuple[Phase1ChainGateEvidence | None, Phase1BridgeIssue | None]:
    result = plan.baseline_query
    if result is None:
        return None, Phase1BridgeIssue("CHAIN_GATE_SOURCE_MISSING", "FATAL", plan.chain_alias)
    if plan.raw_chain.status not in {"READY", "UNCERTAIN"}:
        return None, Phase1BridgeIssue("STRICT_CHAIN_STATUS_INVALID", "FATAL", plan.chain_alias)
    if result.status not in {"PASS", "FAIL", "UNCERTAIN"}:
        return None, Phase1BridgeIssue("QUANTITATIVE_STATUS_INVALID", "FATAL", plan.chain_alias)
    calibration_status = (
        "CALIBRATED"
        if plan.group is not None and plan.group.calibration.status == "CALIBRATED"
        else "UNAVAILABLE"
    )
    try:
        return (
            Phase1ChainGateEvidence(
                strict_chain_id=plan.sanitized_gate_id,
                strict_chain_status=plan.raw_chain.status,
                query_chain_alias=plan.chain_alias,
                quantitative_status=result.status,
                dynamic_threshold=result.group_threshold,
                threshold_calibration_status=calibration_status,
                calibration_min_group_size=(plan.group.calibration.min_group_size if plan.group is not None else 0),
                normal_group_size=(len(plan.group.records) if plan.group is not None else 0),
                normal_reference_count=(len(query.normal_references) if query is not None else len(plan.references)),
            ),
            (
                Phase1BridgeIssue("THRESHOLD_CALIBRATION_UNAVAILABLE", "INFORMATIONAL", plan.chain_alias)
                if calibration_status == "UNAVAILABLE"
                else None
            ),
        )
    except ValueError:
        return None, Phase1BridgeIssue("CHAIN_GATE_CONSTRUCTION_FAILED", "FATAL", plan.chain_alias)


def _alias_maps(
    plans: tuple[_ChainPlan, ...] | list[_ChainPlan],
    transaction_ids: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    addresses: set[str] = set()
    assets: set[str] = set()
    transactions: set[str] = set()
    evidence_keys: list[str] = []
    ordered_plans = tuple(sorted(plans, key=lambda item: _query_alias_order(item.chain_alias)))
    reference_occurrences: list[tuple[int, int, str, str]] = []
    records_by_plan = {
        plan.chain_alias: {item.record_id: item for item in plan.references}
        for plan in ordered_plans
    }
    for plan in ordered_plans:
        if plan.query_record is not None:
            addresses.add(plan.query_record.main_anchor.effective_address)
            for index, step in enumerate(plan.query_record.call_steps, start=1):
                addresses.add(step.effective_address)
                evidence_keys.append(f"{plan.chain_alias}:query:call:{index}")
            for index, behavior in enumerate(sorted(plan.query_record.asset_behaviors, key=_asset_order), start=1):
                assets.add(behavior.asset_identifier)
                evidence_keys.append(f"{plan.chain_alias}:query:asset:{index}")
        for reference in plan.references:
            transactions.add(transaction_ids[reference.record_id])
            for step in reference.call_steps:
                addresses.add(step.effective_address)
            for behavior in reference.asset_behaviors:
                assets.add(behavior.asset_identifier)
        if plan.baseline_query is not None:
            available = records_by_plan[plan.chain_alias]
            for match in plan.baseline_query.top_matches:
                if match.normal_record_id in available:
                    reference_occurrences.append(
                        (
                            _query_alias_order(plan.chain_alias),
                            match.rank,
                            match.normal_record_id,
                            _reference_occurrence_key(
                                plan.chain_alias,
                                match.rank,
                                match.normal_record_id,
                            ),
                        )
                    )
    ordered_occurrences = tuple(sorted(reference_occurrences, key=lambda item: item[:3]))
    reference_alias = {
        occurrence_key: f"R{index}"
        for index, (_, _, _, occurrence_key) in enumerate(ordered_occurrences, start=1)
    }
    for _, _, record_id, occurrence_key in ordered_occurrences:
        reference = next(
            item
            for plan in ordered_plans
            for item in plan.references
            if item.record_id == record_id
        )
        for index, _ in enumerate(reference.call_steps, start=1):
            evidence_keys.append(f"{occurrence_key}:call:{index}")
        for index, _ in enumerate(sorted(reference.asset_behaviors, key=_asset_order), start=1):
            evidence_keys.append(f"{occurrence_key}:asset:{index}")
    return {
        "address": {value: f"A{index}" for index, value in enumerate(sorted(addresses), start=1)},
        "asset": {value: f"AS{index}" for index, value in enumerate(sorted(assets), start=1)},
        "reference": reference_alias,
        "transaction": {value: f"H{index}" for index, value in enumerate(sorted(transactions), start=1)},
        "evidence": {value: f"E{index}" for index, value in enumerate(evidence_keys, start=1)},
    }


def _amount_observation(
    chain: StrictResultChain,
    record: ChainFeatureRecord,
) -> tuple[int | None, str | None, str | None]:
    identifier = "NATIVE" if chain.asset_type == "NATIVE" else (
        chain.asset_address.lower() if isinstance(chain.asset_address, str) and _ADDRESS.fullmatch(chain.asset_address.lower()) else None
    )
    ordered = tuple(sorted(record.asset_behaviors, key=_asset_order))
    matches = tuple(index for index, item in enumerate(ordered, start=1) if item.asset_identifier == identifier)
    if len(matches) > 1:
        return None, None, "QUERY_AMOUNT_ASSET_MATCH_AMBIGUOUS"
    if len(matches) != 1 or not isinstance(chain.amount_raw, str) or _AMOUNT.fullmatch(chain.amount_raw) is None:
        return matches[0] if matches else None, None, "QUERY_AMOUNT_OBSERVATION_UNAVAILABLE"
    return matches[0], chain.amount_raw, None


def _coverage(value: Mapping[str, Any], issues: list[Phase1BridgeIssue]) -> Decimal | None:
    raw = value.get("coverage_rate") if isinstance(value, Mapping) else None
    if isinstance(raw, bool) or not isinstance(raw, (str, int, Decimal)):
        issues.append(Phase1BridgeIssue("RESULT_COVERAGE_FIELD_INVALID", "FATAL"))
        return None
    try:
        parsed = Decimal(raw)
    except (InvalidOperation, TypeError, ValueError):
        issues.append(Phase1BridgeIssue("RESULT_COVERAGE_FIELD_INVALID", "FATAL"))
        return None
    if not parsed.is_finite() or parsed < 0 or parsed > 1:
        issues.append(Phase1BridgeIssue("RESULT_COVERAGE_FIELD_INVALID", "FATAL"))
        return None
    count_keys = (
        "result_candidate_count",
        "chains_formed_result_candidate_count",
        "uncertain_result_candidate_count",
    )
    supplied_count_keys = tuple(key for key in count_keys if key in value)
    if supplied_count_keys and len(supplied_count_keys) != len(count_keys):
        issues.append(Phase1BridgeIssue("RESULT_COVERAGE_FIELDS_INCOMPLETE", "FATAL"))
        return None
    if supplied_count_keys:
        counts = tuple(value[key] for key in count_keys)
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in counts):
            issues.append(Phase1BridgeIssue("RESULT_COVERAGE_FIELD_INVALID", "FATAL"))
            return None
        total, formed, uncertain = counts
        expected = Decimal(1) if total == 0 else Decimal(formed) / Decimal(total)
        if formed > total or uncertain > total or parsed.quantize(Decimal("0.0001")) != expected.quantize(Decimal("0.0001")):
            issues.append(Phase1BridgeIssue("RESULT_COVERAGE_FIELDS_CONTRADICTORY", "FATAL"))
            return None
    return parsed


def _critical_missing(value: Mapping[str, Any], issues: list[Phase1BridgeIssue]) -> int | None:
    raw = value.get("critical_missing_count") if isinstance(value, Mapping) else None
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        issues.append(Phase1BridgeIssue("CRITICAL_MISSING_COUNT_FIELD_INVALID", "FATAL"))
        return None
    return raw


def _ineligible_without_gate(issues: list[Phase1BridgeIssue]) -> Phase1NormalityPreparation:
    reasons = []
    if any(item.code == "RESULT_COVERAGE_FIELD_INVALID" for item in issues):
        reasons.append("RESULT_COVERAGE_INCOMPLETE")
    if any(item.code == "CRITICAL_MISSING_COUNT_FIELD_INVALID" for item in issues):
        reasons.append("CRITICAL_EVIDENCE_MISSING")
    if not reasons:
        reasons.append("CHAIN_SET_MISMATCH")
    eligibility = Phase1EligibilityResult("INELIGIBLE", False, tuple(reasons), (), ())
    return Phase1NormalityPreparation("INELIGIBLE", None, eligibility, None, tuple(issues))


def _validate_inputs(strict: Any, features: Any, baseline: Any, mapping: Any) -> None:
    if not isinstance(strict, StrictResultChainsResult):
        raise ValueError("STRICT_RESULT_CHAINS_INPUT_INVALID")
    if not isinstance(features, NormalFeatureExtractionResult):
        raise ValueError("NORMAL_FEATURES_INPUT_INVALID")
    if not isinstance(baseline, NormalBaseline):
        raise ValueError("NORMAL_BASELINE_INPUT_INVALID")
    if not isinstance(mapping, Mapping):
        raise ValueError("NORMAL_TRANSACTION_MAPPING_INVALID")


def _duplicates(values: Any) -> set[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return duplicate


def _asset_order(item: AssetBehavior) -> tuple[str, str]:
    return item.asset_identifier, item.direction


def _query_alias_order(value: str) -> int:
    return int(value[1:])


def _reference_occurrence_key(chain_alias: str, rank: int, record_id: str) -> str:
    """Internal-only identity for one reference occurrence; never serialized."""
    return f"{chain_alias}:reference:{rank}:{record_id}"


def _issue_order(item: Phase1BridgeIssue) -> tuple[str, str, str, str]:
    return item.code, item.severity, item.chain_alias or "", item.reference_alias or ""


__all__ = [
    "Phase1BridgeIssue",
    "Phase1NormalityPreparation",
    "prepare_phase1_normality_review",
    "serialize_phase1_normality_preparation",
]

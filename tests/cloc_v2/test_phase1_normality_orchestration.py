from __future__ import annotations

import json
import unittest
from dataclasses import replace
from decimal import Decimal

from src.cloc_v2.phase1_normality_orchestration import (
    Phase1ChainGateEvidence,
    Phase1TransactionGateEvidence,
    decide_phase1_route,
    evaluate_phase1_eligibility,
    serialize_phase1_routing_decision,
)
from src.cloc_v2.phase1_normality_prompt import (
    RESPONSE_SCHEMA_VERSION,
    Phase1AssetEvidence,
    Phase1CallEvidence,
    Phase1NormalReference,
    Phase1QueryChain,
    SanitizedMainAnchor,
    build_phase1_normality_prompt,
)


def _call(evidence_id: str, address_alias: str, selector: str) -> Phase1CallEvidence:
    return Phase1CallEvidence(evidence_id, address_alias, selector, 1)


def _asset(evidence_id: str, asset_alias: str, amount: str | None = None) -> Phase1AssetEvidence:
    return Phase1AssetEvidence(
        evidence_id=evidence_id,
        asset_alias=asset_alias,
        direction="OUTFLOW",
        asset_type="TOKEN",
        amount_observation=amount,
    )


def _reference(
    number: int,
    *,
    address_alias: str,
    asset_alias: str,
    selector: str,
    call_evidence_id: str,
    asset_evidence_id: str,
) -> Phase1NormalReference:
    return Phase1NormalReference(
        reference_alias=f"R{number}",
        normal_transaction_alias=f"H{number}",
        call_evidence=(_call(call_evidence_id, address_alias, selector),),
        asset_evidence=(_asset(asset_evidence_id, asset_alias),),
        chain_distance=Decimal("0.100000000000000000"),
        call_distance=Decimal("0.100000000000000000"),
        asset_distance=Decimal("0.000000000000000000"),
    )


def _query(
    number: int = 1,
    *,
    reference_count: int = 3,
    amount: str = "100",
    nearest_distance: Decimal = Decimal("0.100000000000000000"),
) -> Phase1QueryChain:
    if number == 1:
        address_alias, asset_alias, selector = "A1", "T1", "0x11111111"
        query_call_id, query_asset_id = "E1", "E2"
        reference_numbers = (1, 2, 3)
        reference_ids = (("E3", "E4"), ("E5", "E6"), ("E7", "E8"))
    else:
        address_alias, asset_alias, selector = "A2", "T2", "0x22222222"
        query_call_id, query_asset_id = "E11", "E12"
        reference_numbers = (4, 5, 6)
        reference_ids = (("E13", "E14"), ("E15", "E16"), ("E17", "E18"))
    references = tuple(
        _reference(
            reference_number,
            address_alias=address_alias,
            asset_alias=asset_alias,
            selector=selector,
            call_evidence_id=ids[0],
            asset_evidence_id=ids[1],
        )
        for reference_number, ids in zip(reference_numbers[:reference_count], reference_ids[:reference_count])
    )
    return Phase1QueryChain(
        chain_alias=f"Q{number}",
        main_anchor=SanitizedMainAnchor(1, address_alias, selector),
        dynamic_threshold=Decimal("0.200000000000000000"),
        nearest_distance=nearest_distance,
        quantitative_status="PASS",
        call_evidence=(_call(query_call_id, address_alias, selector),),
        asset_evidence=(_asset(query_asset_id, asset_alias, amount),),
        normal_references=references,
    )


def _gate(
    number: int = 1,
    *,
    status: str = "READY",
    quantitative: str = "PASS",
    threshold: Decimal | None = Decimal("0.200000000000000000"),
    threshold_status: str = "CALIBRATED",
    calibration_min_group_size: int = 3,
    group_size: int = 3,
    reference_count: int = 3,
    query_alias: str | None = None,
) -> Phase1ChainGateEvidence:
    return Phase1ChainGateEvidence(
        strict_chain_id=f"chain_c{number:03d}",
        strict_chain_status=status,
        query_chain_alias=query_alias or f"Q{number}",
        quantitative_status=quantitative,
        dynamic_threshold=threshold,
        threshold_calibration_status=threshold_status,
        calibration_min_group_size=calibration_min_group_size,
        normal_group_size=group_size,
        normal_reference_count=reference_count,
    )


def _evidence(
    *,
    gates: tuple[Phase1ChainGateEvidence, ...] | None = None,
    queries: tuple[Phase1QueryChain, ...] | None = None,
    coverage: Decimal = Decimal("1"),
    critical_missing: int = 0,
    total_strict_chain_count: int | None = None,
    excluded_uncertain_chain_count: int = 0,
) -> Phase1TransactionGateEvidence:
    return Phase1TransactionGateEvidence(
        result_candidate_coverage_rate=coverage,
        critical_missing_count=critical_missing,
        strict_chains=(_gate(),) if gates is None else gates,
        query_chains=(_query(),) if queries is None else queries,
        total_strict_chain_count=total_strict_chain_count,
        excluded_uncertain_chain_count=excluded_uncertain_chain_count,
    )


def _decision_item(number: int, decision: str) -> dict:
    if number == 1:
        reference_alias, query_call_id, reference_call_id = "R1", "E1", "E3"
    else:
        reference_alias, query_call_id, reference_call_id = "R4", "E11", "E13"
    if decision == "NORMAL":
        assessment = "CONSISTENT"
        evidence_ids = [query_call_id]
        reasons = ["WITHIN_CALIBRATED_THRESHOLD"]
    elif decision == "SUSPICIOUS":
        assessment = "MATERIAL_DEVIATION"
        evidence_ids = [query_call_id, reference_call_id]
        reasons = ["MATERIAL_CALL_DEVIATION"]
    else:
        assessment = "INSUFFICIENT_EVIDENCE"
        evidence_ids = [query_call_id]
        reasons = ["CRITICAL_EVIDENCE_INCOMPLETE"]
    return {
        "chain_alias": f"Q{number}",
        "decision": decision,
        "reference_assessment": assessment,
        "supporting_reference_aliases": [reference_alias],
        "evidence_ids": evidence_ids,
        "reason_codes": reasons,
        "brief_rationale": f"Q{number} is assessed using {reference_alias} and {query_call_id}.",
    }


def _response(*decisions: tuple[int, str]) -> dict:
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "chain_decisions": [_decision_item(number, decision) for number, decision in decisions],
    }


class Phase1NormalityOrchestrationTests(unittest.TestCase):
    def test_all_hard_gates_satisfied_is_eligible(self) -> None:
        evidence = _evidence()
        prompt = build_phase1_normality_prompt(evidence.query_chains)
        result = evaluate_phase1_eligibility(evidence, prompt)
        self.assertEqual(result.eligibility, "ELIGIBLE")
        self.assertTrue(result.llm_allowed)
        self.assertEqual(result.reason_codes, ())

    def test_eligible_without_response_awaits_llm(self) -> None:
        evidence = _evidence()
        result = decide_phase1_route(evidence, build_phase1_normality_prompt(evidence.query_chains))
        self.assertEqual(result.route, "AWAITING_LLM")
        self.assertFalse(result.llm_response_processed)

    def test_all_normal_chains_early_stop_normal(self) -> None:
        evidence = _evidence()
        result = decide_phase1_route(
            evidence,
            build_phase1_normality_prompt(evidence.query_chains),
            _response((1, "NORMAL")),
        )
        self.assertEqual(result.route, "EARLY_NORMAL")
        self.assertEqual(result.reason_codes, ("ALL_USABLE_CLEAN_CHAINS_LLM_NORMAL",))

    def test_any_suspicious_chain_proceeds_to_phase2(self) -> None:
        evidence = _evidence(gates=(_gate(1), _gate(2)), queries=(_query(1), _query(2)))
        result = decide_phase1_route(
            evidence,
            build_phase1_normality_prompt(evidence.query_chains),
            _response((1, "NORMAL"), (2, "SUSPICIOUS")),
        )
        self.assertEqual(result.route, "PROCEED_PHASE2")
        self.assertEqual(result.reason_codes, ("LLM_SUSPICIOUS",))

    def test_any_uncertain_chain_proceeds_to_phase2(self) -> None:
        evidence = _evidence(gates=(_gate(1), _gate(2)), queries=(_query(1), _query(2)))
        result = decide_phase1_route(
            evidence,
            build_phase1_normality_prompt(evidence.query_chains),
            _response((1, "NORMAL"), (2, "UNCERTAIN")),
        )
        self.assertEqual(result.route, "PROCEED_PHASE2")
        self.assertEqual(result.reason_codes, ("LLM_UNCERTAIN",))

    def test_incomplete_coverage_is_ineligible(self) -> None:
        result = evaluate_phase1_eligibility(_evidence(coverage=Decimal("0.9999")))
        self.assertIn("RESULT_COVERAGE_INCOMPLETE", result.reason_codes)

    def test_critical_missing_evidence_is_ineligible(self) -> None:
        result = evaluate_phase1_eligibility(_evidence(critical_missing=1))
        self.assertIn("CRITICAL_EVIDENCE_MISSING", result.reason_codes)

    def test_no_result_chain_is_ineligible_and_routes_phase2(self) -> None:
        evidence = _evidence(gates=(), queries=(_query(),))
        prompt = build_phase1_normality_prompt(evidence.query_chains)
        result = decide_phase1_route(evidence, prompt, _response((1, "NORMAL")))
        self.assertEqual(result.route, "PROCEED_PHASE2")
        self.assertIn("NO_RESULT_CHAIN", result.reason_codes)
        self.assertFalse(result.llm_response_processed)

    def test_non_ready_strict_chain_is_ineligible(self) -> None:
        result = evaluate_phase1_eligibility(_evidence(gates=(_gate(status="UNCERTAIN"),)))
        self.assertIn("STRICT_CHAIN_NOT_READY", result.reason_codes)

    def test_excluded_uncertain_chains_do_not_block_usable_clean_chain(self) -> None:
        evidence = _evidence(total_strict_chain_count=4, excluded_uncertain_chain_count=3)
        prompt = build_phase1_normality_prompt(evidence.query_chains)
        result = evaluate_phase1_eligibility(evidence, prompt)
        self.assertEqual(result.eligibility, "ELIGIBLE")
        self.assertEqual(result.total_strict_chain_count, 4)
        self.assertEqual(result.usable_clean_chain_count, 1)
        self.assertEqual(result.excluded_uncertain_chain_count, 3)

    def test_no_usable_clean_chain_is_ineligible(self) -> None:
        result = evaluate_phase1_eligibility(
            _evidence(
                gates=(),
                queries=(),
                total_strict_chain_count=3,
                excluded_uncertain_chain_count=3,
            )
        )
        self.assertIn("NO_USABLE_CLEAN_CHAIN", result.reason_codes)
        self.assertNotIn("NO_RESULT_CHAIN", result.reason_codes)

    def test_all_exact_clean_queries_require_direct_normal_route(self) -> None:
        query = _query(nearest_distance=Decimal(0))
        evidence = _evidence(queries=(query,))
        prompt = build_phase1_normality_prompt(evidence.query_chains)
        eligibility = evaluate_phase1_eligibility(evidence, prompt)
        self.assertEqual(eligibility.eligibility, "INELIGIBLE")
        self.assertIn("DIRECT_NORMAL_ROUTE_REQUIRED", eligibility.reason_codes)
        route = decide_phase1_route(evidence, prompt, _response((1, "NORMAL")))
        self.assertEqual(route.route, "PROCEED_PHASE2")
        self.assertFalse(route.llm_response_processed)

    def test_quantitative_fail_or_uncertain_is_ineligible(self) -> None:
        for status in ("FAIL", "UNCERTAIN"):
            with self.subTest(status=status):
                result = evaluate_phase1_eligibility(_evidence(gates=(_gate(quantitative=status),)))
                self.assertIn("QUANTITATIVE_NOT_PASS", result.reason_codes)

    def test_missing_threshold_is_ineligible(self) -> None:
        result = evaluate_phase1_eligibility(
            _evidence(gates=(_gate(threshold=None, threshold_status="UNAVAILABLE"),))
        )
        self.assertIn("THRESHOLD_UNAVAILABLE", result.reason_codes)

    def test_diagnostic_calibration_policy_cannot_enter_formal_phase1(self) -> None:
        result = evaluate_phase1_eligibility(
            _evidence(gates=(_gate(calibration_min_group_size=2),))
        )
        self.assertIn("FORMAL_MIN_GROUP_SIZE_MISMATCH", result.reason_codes)

    def test_threshold_equal_to_one_is_non_discriminative(self) -> None:
        query = replace(
            _query(),
            dynamic_threshold=Decimal("1"),
            nearest_distance=Decimal("0.1"),
        )
        result = evaluate_phase1_eligibility(
            _evidence(gates=(_gate(threshold=Decimal("1")),), queries=(query,))
        )
        self.assertIn("THRESHOLD_NON_DISCRIMINATIVE", result.reason_codes)

    def test_normal_group_smaller_than_three_is_ineligible(self) -> None:
        result = evaluate_phase1_eligibility(_evidence(gates=(_gate(group_size=2),)))
        self.assertIn("NORMAL_GROUP_TOO_SMALL", result.reason_codes)

    def test_top_three_reference_must_be_complete(self) -> None:
        evidence = _evidence(
            gates=(_gate(reference_count=2),),
            queries=(_query(reference_count=2),),
        )
        result = evaluate_phase1_eligibility(evidence)
        self.assertIn("TOP3_REFERENCE_INCOMPLETE", result.reason_codes)

    def test_input_chain_set_mismatch_is_ineligible(self) -> None:
        evidence = _evidence(gates=(_gate(query_alias="Q2"),), queries=(_query(1),))
        result = evaluate_phase1_eligibility(evidence)
        self.assertIn("CHAIN_SET_MISMATCH", result.reason_codes)

    def test_llm_response_missing_or_duplicate_chain_is_invalid(self) -> None:
        evidence = _evidence(gates=(_gate(1), _gate(2)), queries=(_query(1), _query(2)))
        prompt = build_phase1_normality_prompt(evidence.query_chains)
        for response in (
            _response((1, "NORMAL")),
            _response((1, "NORMAL"), (1, "NORMAL")),
        ):
            with self.subTest(response=response):
                result = decide_phase1_route(evidence, prompt, response)
                self.assertEqual(result.route, "PROCEED_PHASE2")
                self.assertEqual(result.reason_codes, ("LLM_RESPONSE_INVALID",))

    def test_unknown_evidence_and_extra_fields_are_invalid_responses(self) -> None:
        evidence = _evidence()
        prompt = build_phase1_normality_prompt(evidence.query_chains)
        unknown = _response((1, "NORMAL"))
        unknown["chain_decisions"][0]["evidence_ids"] = ["E999"]
        extra = _response((1, "NORMAL"))
        extra["transaction_decision"] = "NORMAL"
        for response in (unknown, extra):
            with self.subTest(response=response):
                result = decide_phase1_route(evidence, prompt, response)
                self.assertEqual(result.reason_codes, ("LLM_RESPONSE_INVALID",))

    def test_ineligible_cannot_be_overridden_by_all_normal_response(self) -> None:
        evidence = _evidence(coverage=Decimal("0.5"))
        result = decide_phase1_route(
            evidence,
            build_phase1_normality_prompt(evidence.query_chains),
            _response((1, "NORMAL")),
        )
        self.assertEqual(result.route, "PROCEED_PHASE2")
        self.assertFalse(result.llm_response_processed)
        self.assertIn("RESULT_COVERAGE_INCOMPLETE", result.reason_codes)

    def test_query_amount_changes_do_not_affect_eligibility_or_route(self) -> None:
        small = _evidence(queries=(_query(amount="1"),))
        large = _evidence(queries=(_query(amount="999999999999999999999"),))
        small_route = decide_phase1_route(
            small,
            build_phase1_normality_prompt(small.query_chains),
            _response((1, "NORMAL")),
        )
        large_route = decide_phase1_route(
            large,
            build_phase1_normality_prompt(large.query_chains),
            _response((1, "NORMAL")),
        )
        self.assertEqual(small_route.route, large_route.route)
        self.assertEqual(small_route.reason_codes, large_route.reason_codes)

    def test_reason_codes_and_serialization_are_stable(self) -> None:
        left = _evidence(
            gates=(
                _gate(
                    status="UNCERTAIN",
                    quantitative="FAIL",
                    threshold=Decimal("1"),
                    group_size=2,
                    reference_count=2,
                ),
            ),
            queries=(_query(reference_count=2),),
            coverage=Decimal("0.5"),
            critical_missing=2,
        )
        prompt = build_phase1_normality_prompt(left.query_chains)
        first = decide_phase1_route(left, prompt)
        second = decide_phase1_route(left, prompt, _response((1, "NORMAL")))
        self.assertEqual(first.reason_codes, tuple(sorted(set(first.reason_codes))))
        self.assertEqual(first.reason_codes, second.reason_codes)
        self.assertEqual(
            serialize_phase1_routing_decision(first),
            serialize_phase1_routing_decision(decide_phase1_route(left, prompt)),
        )
        self.assertEqual(json.loads(serialize_phase1_routing_decision(first))["route"], "PROCEED_PHASE2")

    def test_routing_serialization_retains_excluded_chain_audit_counts(self) -> None:
        evidence = _evidence(total_strict_chain_count=9, excluded_uncertain_chain_count=8)
        prompt = build_phase1_normality_prompt(evidence.query_chains)
        payload = json.loads(serialize_phase1_routing_decision(decide_phase1_route(evidence, prompt)))
        eligibility = payload["eligibility"]
        self.assertEqual(eligibility["total_strict_chain_count"], 9)
        self.assertEqual(eligibility["usable_clean_chain_count"], 1)
        self.assertEqual(eligibility["excluded_uncertain_chain_count"], 8)


if __name__ == "__main__":
    unittest.main()

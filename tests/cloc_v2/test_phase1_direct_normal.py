from __future__ import annotations

import json
import unittest

from src.cloc_v2.formal_normal_baseline_loader import FormalNormalTransactionIdentity
from src.cloc_v2.normal_baseline import build_normal_baseline
from src.cloc_v2.normal_distance import AssetBehavior, CallStep, ChainFeatureRecord, MainAnchorKey
from src.cloc_v2.normal_features import ChainFeatureExtractionStatus, NormalFeatureExtractionResult
from src.cloc_v2.phase1_direct_normal import (
    DIRECT_NORMAL_REASON,
    STRUCTURAL_EXACT_COLLISION_REASON,
    evaluate_phase1_direct_normal_match,
    serialize_phase1_direct_normal_match,
)
from src.cloc_v2.strict_result_chains import ResultChainStatus, StrictResultChain, StrictResultChainsResult


def address(ch): return "0x" + ch * 40
ANCHOR = MainAnchorKey(1, address("a"), "0x11111111")
ANCHOR_TWO = MainAnchorKey(1, address("d"), "0x22222222")
TOKEN = address("b")
QUERY_HASH = "0x" + "1" * 64
QUERY_ORIGIN = "verified-query-origin"
OTHER_ORIGIN = "verified-other-origin"


def record(record_id, *, selector=None, direction="OUTFLOW", uncertainties=(), anchor=ANCHOR):
    selector_value = selector or anchor.selector
    return ChainFeatureRecord(
        record_id,
        anchor,
        (CallStep(anchor.effective_address, selector_value),),
        frozenset({AssetBehavior(TOKEN, direction)}),
        tuple(uncertainties),
    )


def chain(chain_id, *, status="READY", result_id="result-1"):
    return StrictResultChain(
        chain_id, result_id, "node-1", ("fact-1",), "R_OUTFLOW", "TOKEN", TOKEN,
        ANCHOR.effective_address, address("c"), "1", (), ("node-1",), (), None,
        status, () if status == "READY" else ("STRICT_CHAIN_UNCERTAIN",),
    )


def strict(*chains_, coverage="1.0000", critical=0, candidates=1):
    statuses = (
        ResultChainStatus(
            "result-1", "node-1", tuple(item.strict_chain_id for item in chains_), "READY", ()
        ),
    ) if candidates else ()
    return StrictResultChainsResult(
        1, (), tuple(chains_), statuses, (),
        {"result_candidate_count": candidates, "coverage_rate": coverage, "critical_missing_count": critical},
    )


def features(*pairs, uncertain_ids=()):
    records = tuple(item[1] for item in pairs)
    statuses = tuple(
        ChainFeatureExtractionStatus(
            chain_id,
            item.record_id,
            "EXTRACTED_UNCERTAIN" if chain_id in uncertain_ids else "EXTRACTED_CLEAN",
            True,
            ("FEATURE_UNCERTAIN",) if chain_id in uncertain_ids else (),
            (),
        )
        for chain_id, item in pairs
    )
    return NormalFeatureExtractionResult(
        "query", records, statuses, len(pairs), len(records),
        len(records) - len(uncertain_ids), len(uncertain_ids), 0,
    )


def identity(status="KNOWN_NORMAL_ORIGIN", origin=QUERY_ORIGIN):
    if status == "KNOWN_NORMAL_ORIGIN":
        return FormalNormalTransactionIdentity(
            status, ("QUERY_NORMAL_ORIGIN_RESOLVED",), True,
            _normal_origin_id=origin, _normalized_transaction_hash=QUERY_HASH,
        )
    reason = (
        "QUERY_TRANSACTION_NOT_IN_NORMAL_LIBRARY"
        if status == "NOT_IN_NORMAL_LIBRARY"
        else "QUERY_TRANSACTION_HASH_INVALID"
    )
    return FormalNormalTransactionIdentity(status, (reason,), False)


def mapping_for(baseline, default_origin=QUERY_ORIGIN, overrides=None):
    result = {
        item.record_id: default_origin
        for group in baseline.groups
        for item in group.records
    }
    result.update(overrides or {})
    return result


class Phase1DirectNormalTests(unittest.TestCase):
    def setUp(self):
        self.ready = chain("strict_result_chain_001")
        self.query = record("query")
        self.normal = record("normal")

    def evaluate(
        self,
        strict_value=None,
        features_value=None,
        baseline_value=None,
        identity_value=None,
        mapping_value=None,
    ):
        baseline = baseline_value or build_normal_baseline((self.normal,), min_group_size=3)
        origin_mapping = mapping_value if mapping_value is not None else mapping_for(baseline)
        return evaluate_phase1_direct_normal_match(
            strict_value or strict(self.ready),
            features_value or features((self.ready.strict_chain_id, self.query)),
            baseline,
            identity_value or identity(),
            origin_mapping,
        )

    def test_known_normal_self_origin_exact_allows_direct(self):
        result = self.evaluate()
        self.assertTrue(result.direct_normal)
        self.assertEqual(result.final_route, "EARLY_NORMAL")
        self.assertEqual(result.decision_kind, "KNOWN_TRANSACTION_DIRECT_NORMAL")
        self.assertTrue(result.chain_matches[0].self_origin_exact_match)

    def test_one_clean_ready_exact_plus_uncertain_allows_direct(self):
        uncertain = chain("strict_result_chain_002", status="UNCERTAIN")
        result = self.evaluate(strict(self.ready, uncertain), features((self.ready.strict_chain_id, self.query)))
        self.assertTrue(result.direct_normal)
        self.assertEqual(
            (result.clean_ready_chain_count, result.exact_matched_clean_chain_count, result.uncertain_chain_count),
            (1, 1, 1),
        )

    def test_not_in_library_other_origin_exact_is_collision_phase2(self):
        baseline = build_normal_baseline((self.normal,), min_group_size=3)
        result = self.evaluate(
            baseline_value=baseline,
            identity_value=identity("NOT_IN_NORMAL_LIBRARY", None),
            mapping_value=mapping_for(baseline, OTHER_ORIGIN),
        )
        self.assertFalse(result.direct_normal)
        self.assertTrue(result.structural_exact_collision)
        self.assertEqual(result.final_route, "PROCEED_PHASE2")
        self.assertEqual(result.decision_kind, "STRUCTURAL_EXACT_COLLISION_PHASE2")
        self.assertIn(STRUCTURAL_EXACT_COLLISION_REASON, result.reason_codes)

    def test_known_identity_with_only_other_origin_exact_is_collision(self):
        baseline = build_normal_baseline((self.normal,), min_group_size=3)
        result = self.evaluate(
            baseline_value=baseline,
            mapping_value=mapping_for(baseline, OTHER_ORIGIN),
        )
        self.assertTrue(result.chain_matches[0].structural_exact_match)
        self.assertFalse(result.chain_matches[0].self_origin_exact_match)
        self.assertEqual(result.final_route, "PROCEED_PHASE2")

    def test_one_self_chain_and_one_other_only_chain_is_collision(self):
        second = chain("strict_result_chain_002")
        query_two = record("query-two", anchor=ANCHOR_TWO)
        self_normal = record("self-normal")
        other_normal = record("other-normal", anchor=ANCHOR_TWO)
        baseline = build_normal_baseline((self_normal, other_normal), min_group_size=3)
        result = self.evaluate(
            strict(self.ready, second),
            features((self.ready.strict_chain_id, self.query), (second.strict_chain_id, query_two)),
            baseline,
            mapping_value={self_normal.record_id: QUERY_ORIGIN, other_normal.record_id: OTHER_ORIGIN},
        )
        self.assertEqual(result.self_origin_exact_matched_clean_chain_count, 1)
        self.assertTrue(result.structural_exact_collision)

    def test_self_exact_outside_top3_is_found_in_complete_group(self):
        records = tuple(record(f"normal-{index}") for index in range(1, 5)) + (record("zz-self"),)
        baseline = build_normal_baseline(records, min_group_size=3)
        mapping = mapping_for(baseline, OTHER_ORIGIN, {"zz-self": QUERY_ORIGIN})
        result = self.evaluate(baseline_value=baseline, mapping_value=mapping)
        self.assertTrue(result.direct_normal)
        self.assertTrue(result.chain_matches[0].self_origin_exact_match)
        self.assertEqual(result.chain_matches[0].reference_origin_status, "SELF_ORIGIN_EXACT_MATCH")

    def test_top3_first_other_but_complete_group_has_self(self):
        records = (record("aa-other"), record("bb-other"), record("cc-other"), record("zz-self"))
        baseline = build_normal_baseline(records, min_group_size=3)
        mapping = mapping_for(baseline, OTHER_ORIGIN, {"zz-self": QUERY_ORIGIN})
        result = self.evaluate(baseline_value=baseline, mapping_value=mapping)
        self.assertEqual(result.chain_matches[0].matched_normal_record_id, "aa-other")
        self.assertTrue(result.direct_normal)

    def test_missing_record_origin_mapping_blocks_direct(self):
        result = self.evaluate(mapping_value={})
        self.assertFalse(result.direct_normal)
        self.assertTrue(result.structural_exact_collision)
        self.assertIn("NORMAL_REFERENCE_ORIGIN_MAPPING_MISSING", result.reason_codes)

    def test_conflicting_extra_record_origin_mapping_blocks_direct(self):
        baseline = build_normal_baseline((self.normal,), min_group_size=3)
        mapping = mapping_for(baseline)
        mapping["not-in-baseline"] = OTHER_ORIGIN
        result = self.evaluate(baseline_value=baseline, mapping_value=mapping)
        self.assertFalse(result.direct_normal)
        self.assertIn("NORMAL_REFERENCE_ORIGIN_MAPPING_CONFLICT", result.reason_codes)

    def test_unresolved_identity_does_not_direct(self):
        result = self.evaluate(identity_value=identity("UNRESOLVED", None))
        self.assertFalse(result.direct_normal)
        self.assertTrue(result.structural_exact_collision)
        self.assertIn("QUERY_NORMAL_ORIGIN_UNRESOLVED", result.reason_codes)

    def test_not_in_library_and_unresolved_are_distinct(self):
        not_in = self.evaluate(identity_value=identity("NOT_IN_NORMAL_LIBRARY", None))
        unresolved = self.evaluate(identity_value=identity("UNRESOLVED", None))
        self.assertEqual(not_in.query_normal_identity_status, "NOT_IN_NORMAL_LIBRARY")
        self.assertEqual(unresolved.query_normal_identity_status, "UNRESOLVED")

    def test_multiple_clean_chains_all_self_exact_allow_direct(self):
        second = chain("strict_result_chain_002")
        result = self.evaluate(
            strict(self.ready, second),
            features((self.ready.strict_chain_id, self.query), (second.strict_chain_id, record("query-2"))),
        )
        self.assertTrue(result.direct_normal)
        self.assertEqual(result.self_origin_exact_matched_clean_chain_count, 2)

    def test_any_clean_unmatched_chain_blocks_direct(self):
        second = chain("strict_result_chain_002")
        result = self.evaluate(
            strict(self.ready, second),
            features((self.ready.strict_chain_id, self.query), (second.strict_chain_id, record("query-2", direction="INFLOW"))),
        )
        self.assertFalse(result.direct_normal)
        self.assertFalse(result.structural_exact_collision)
        self.assertEqual(result.final_route, "CONTINUE_STANDARD_PHASE1")

    def test_no_clean_chain_goes_to_standard_phase1(self):
        uncertain = chain("strict_result_chain_001", status="UNCERTAIN")
        result = self.evaluate(strict(uncertain), features())
        self.assertFalse(result.direct_normal)
        self.assertEqual(result.final_route, "CONTINUE_STANDARD_PHASE1")
        self.assertIn("NO_USABLE_CLEAN_READY_CHAIN", result.reason_codes)

    def test_uncertain_chain_never_participates_or_becomes_ready(self):
        uncertain = chain("strict_result_chain_002", status="UNCERTAIN")
        result = self.evaluate(strict(self.ready, uncertain), features((self.ready.strict_chain_id, self.query)))
        audit = next(item for item in result.chain_matches if item.strict_status == "UNCERTAIN")
        self.assertTrue(result.direct_normal)
        self.assertFalse(audit.usable_clean)
        self.assertFalse(audit.structural_exact_match)
        self.assertEqual(audit.strict_status, "UNCERTAIN")

    def test_call_and_total_distance_nonzero_continue_standard_phase1(self):
        baseline = build_normal_baseline((record("normal", selector="0x22222222"),), min_group_size=3)
        result = self.evaluate(baseline_value=baseline)
        self.assertFalse(result.direct_normal)
        self.assertEqual(result.final_route, "CONTINUE_STANDARD_PHASE1")
        self.assertIn("EXACT_CALL_DISTANCE_NOT_ZERO", result.chain_matches[0].reason_codes)
        self.assertNotEqual(result.chain_matches[0].chain_distance, 0)

    def test_asset_distance_nonzero_continues_standard_phase1(self):
        result = self.evaluate(
            baseline_value=build_normal_baseline((record("normal", direction="INFLOW"),), min_group_size=3)
        )
        self.assertFalse(result.direct_normal)
        self.assertIn("EXACT_ASSET_DISTANCE_NOT_ZERO", result.chain_matches[0].reason_codes)

    def test_incomplete_coverage_and_critical_missing_block(self):
        self.assertFalse(self.evaluate(strict(self.ready, coverage="0.5")).direct_normal)
        self.assertFalse(self.evaluate(strict(self.ready, critical=1)).direct_normal)

    def test_uncalibrated_group_self_exact_match_allows_direct(self):
        result = self.evaluate(baseline_value=build_normal_baseline((self.normal,), min_group_size=3))
        self.assertTrue(result.direct_normal)

    def test_required_identity_arguments_have_no_dangerous_defaults(self):
        with self.assertRaises(TypeError):
            evaluate_phase1_direct_normal_match(
                strict(self.ready),
                features((self.ready.strict_chain_id, self.query)),
                build_normal_baseline((self.normal,), min_group_size=3),
            )

    def test_serialization_is_stable_and_hides_all_internal_identities(self):
        result = self.evaluate()
        text = serialize_phase1_direct_normal_match(result)
        self.assertEqual(text, serialize_phase1_direct_normal_match(result))
        self.assertEqual(json.loads(text)["reason_codes"], [DIRECT_NORMAL_REASON])
        for forbidden in ('"query"', '"normal"', QUERY_HASH, QUERY_ORIGIN, OTHER_ORIGIN, ANCHOR.effective_address):
            self.assertNotIn(forbidden, text)
        self.assertNotIn('"record_id"', text)


if __name__ == "__main__": unittest.main()

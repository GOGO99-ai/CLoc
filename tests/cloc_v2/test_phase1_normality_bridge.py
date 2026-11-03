from __future__ import annotations

import json
import unittest
from dataclasses import replace
from decimal import Decimal
from unittest.mock import patch

from src.cloc_v2.normal_baseline import (
    NormalBaseline,
    NormalBaselineQuery,
    TopNormalMatch,
    build_normal_baseline,
)
from src.cloc_v2.normal_distance import (
    AssetBehavior,
    CallStep,
    ChainDistance,
    ChainFeatureRecord,
    MainAnchorKey,
)
from src.cloc_v2.normal_features import (
    ChainFeatureExtractionStatus,
    NormalFeatureExtractionResult,
)
from src.cloc_v2.phase1_normality_bridge import (
    Phase1NormalityPreparation,
    prepare_phase1_normality_review,
    serialize_phase1_normality_preparation,
)
from src.cloc_v2.strict_result_chains import StrictResultChain, StrictResultChainsResult


def addr(character: str) -> str:
    return "0x" + character * 40


ANCHOR_1 = MainAnchorKey(1, addr("a"), "0x11111111")
ANCHOR_2 = MainAnchorKey(1, addr("b"), "0x22222222")
TOKEN_1 = addr("c")
TOKEN_2 = addr("d")
TX_HASHES = {
    "n1": "0x" + "1" * 64,
    "n2": "0x" + "2" * 64,
    "n3": "0x" + "3" * 64,
    "m1": "0x" + "4" * 64,
    "m2": "0x" + "5" * 64,
    "m3": "0x" + "6" * 64,
}


def record(
    record_id: str,
    *,
    anchor: MainAnchorKey = ANCHOR_1,
    steps: tuple[CallStep, ...] | None = None,
    assets: frozenset[AssetBehavior] | None = None,
    uncertainties: tuple[str, ...] = (),
) -> ChainFeatureRecord:
    return ChainFeatureRecord(
        record_id,
        anchor,
        steps if steps is not None else (CallStep(anchor.effective_address, anchor.selector),),
        assets if assets is not None else frozenset({AssetBehavior(TOKEN_1, "OUTFLOW")}),
        uncertainties,
    )


def normal_records(anchor: MainAnchorKey = ANCHOR_1, prefix: str = "n") -> tuple[ChainFeatureRecord, ...]:
    return tuple(
        record(
            f"normal-record-{prefix}{index}",
            anchor=anchor,
            steps=(
                CallStep(anchor.effective_address, anchor.selector),
                CallStep(addr(str(index + 3)), f"0x{index + 1:08x}"),
            ),
        )
        for index in range(1, 4)
    )


def baseline(*records_: ChainFeatureRecord, min_group_size: int = 3) -> NormalBaseline:
    supplied = records_ or normal_records()
    return build_normal_baseline(supplied, min_group_size=min_group_size)


def strict_chain(
    strict_chain_id: str = "strict_result_chain_001",
    *,
    status: str = "READY",
    anchor: MainAnchorKey = ANCHOR_1,
    asset_type: str = "TOKEN",
    asset_address: str | None = TOKEN_1,
    amount_raw: str | None = "123",
) -> StrictResultChain:
    return StrictResultChain(
        strict_chain_id=strict_chain_id,
        result_id="result-" + strict_chain_id,
        result_node_id="result-node",
        result_fact_ids=("fact-1",),
        result_type="R_OUTFLOW",
        asset_type=asset_type,
        asset_address=asset_address,
        from_address=anchor.effective_address,
        to_address=addr("f"),
        amount_raw=amount_raw,
        result_uncertainties=(),
        node_ids=("node-1",),
        edge_ids=(),
        anchor=None,
        status=status,
        uncertainties=() if status == "READY" else ("STRICT_EVIDENCE_UNCERTAIN",),
    )


def strict_result(
    *chains: StrictResultChain,
    coverage: object = "1.0000",
    critical_missing: object = 0,
) -> StrictResultChainsResult:
    return StrictResultChainsResult(
        blockchain_chain_id=1,
        edges=(),
        chains=tuple(chains) if chains else (strict_chain(),),
        result_statuses=(),
        critical_uncertainties=(),
        coverage={"coverage_rate": coverage, "critical_missing_count": critical_missing},
    )


def features(
    *pairs: tuple[str, ChainFeatureRecord],
    duplicate_status: bool = False,
) -> NormalFeatureExtractionResult:
    supplied = pairs or (("strict_result_chain_001", record("query-record-1")),)
    records = tuple(item[1] for item in supplied)
    statuses = [
        ChainFeatureExtractionStatus(
            strict_chain_id=chain_id,
            record_id=item.record_id,
            status="EXTRACTED_CLEAN" if not item.uncertainties else "EXTRACTED_UNCERTAIN",
            record_created=True,
            reasons=item.uncertainties,
            issues=(),
        )
        for chain_id, item in supplied
    ]
    if duplicate_status:
        statuses.append(statuses[0])
    return NormalFeatureExtractionResult(
        transaction_record_id="query-transaction",
        records=records,
        chain_statuses=tuple(statuses),
        input_chain_count=len(supplied),
        record_count=len(records),
        clean_record_count=sum(not item.uncertainties for item in records),
        uncertain_record_count=sum(bool(item.uncertainties) for item in records),
        rejected_chain_count=0,
    )


def mapping(*records_: ChainFeatureRecord) -> dict[str, str]:
    values = records_ or normal_records()
    return {item.record_id: TX_HASHES[item.record_id[-2:]] for item in values}


def prepare(
    strict: StrictResultChainsResult | None = None,
    extracted: NormalFeatureExtractionResult | None = None,
    normal: NormalBaseline | None = None,
    ownership: dict[str, str] | None = None,
) -> Phase1NormalityPreparation:
    normal_value = normal or baseline()
    ownership_value = ownership
    if ownership_value is None:
        ownership_value = {
            item.record_id: TX_HASHES[item.record_id[-2:]]
            for group in normal_value.groups
            for item in group.records
        }
    return prepare_phase1_normality_review(
        strict or strict_result(),
        extracted or features(),
        normal_value,
        ownership_value,
    )


class Phase1NormalityBridgeTests(unittest.TestCase):
    def test_single_clean_pass_is_ready_for_llm(self) -> None:
        result = prepare()
        self.assertEqual(result.preparation_status, "READY_FOR_LLM")
        self.assertEqual(result.eligibility.eligibility, "ELIGIBLE")
        self.assertIsNotNone(result.prompt)

    def test_multiple_clean_chains_get_stable_query_aliases(self) -> None:
        records_1 = normal_records(ANCHOR_1, "n")
        records_2 = tuple(
            record(
                f"normal-record-m{index}",
                anchor=ANCHOR_2,
                assets=frozenset({AssetBehavior(TOKEN_2, "OUTFLOW")}),
            )
            for index in range(1, 4)
        )
        normal = baseline(*(records_1 + records_2))
        query_1 = record("query-record-1", anchor=ANCHOR_1)
        query_2 = record("query-record-2", anchor=ANCHOR_2, assets=frozenset({AssetBehavior(TOKEN_2, "OUTFLOW")}))
        extracted = features(
            ("strict_result_chain_002", query_2),
            ("strict_result_chain_001", query_1),
        )
        strict = strict_result(
            strict_chain("strict_result_chain_002", anchor=ANCHOR_2, asset_address=TOKEN_2),
            strict_chain("strict_result_chain_001", anchor=ANCHOR_1),
        )
        ownership = {item.record_id: TX_HASHES[item.record_id[-2:]] for item in records_1 + records_2}
        result = prepare(strict, extracted, normal, ownership)
        self.assertEqual(result.prompt.allowed_chain_aliases, ("Q1", "Q2"))

    def test_call_order_and_duplicates_are_preserved(self) -> None:
        step_a = CallStep(addr("a"), "0x11111111")
        step_b = CallStep(addr("b"), "0x22222222")
        query_record = record("query-record-1", steps=(step_a, step_b, step_a))
        normals = tuple(record(item.record_id, steps=query_record.call_steps) for item in normal_records())
        result = prepare(
            extracted=features(("strict_result_chain_001", query_record)),
            normal=baseline(*normals),
        )
        calls = result.gate_evidence.query_chains[0].call_evidence
        self.assertEqual([item.order for item in calls], [1, 2, 3])
        self.assertEqual([item.selector for item in calls], ["0x11111111", "0x22222222", "0x11111111"])
        self.assertEqual(calls[0].address_alias, calls[2].address_alias)

    def test_asset_direction_is_preserved(self) -> None:
        query_record = record(
            "query-record-1",
            assets=frozenset({AssetBehavior(TOKEN_1, "INFLOW")}),
        )
        normal = baseline(*tuple(
            record(item.record_id, steps=item.call_steps, assets=query_record.asset_behaviors)
            for item in normal_records()
        ))
        result = prepare(
            strict=strict_result(strict_chain()),
            extracted=features(("strict_result_chain_001", query_record)),
            normal=normal,
        )
        self.assertIsNotNone(result.prompt)
        self.assertEqual(result.gate_evidence.query_chains[0].asset_evidence[0].direction, "INFLOW")

    def test_top_three_distances_and_order_come_from_formal_query(self) -> None:
        query_record = record("query-record-1")
        normals = (
            record("normal-record-n1"),
            record("normal-record-n2", steps=(CallStep(addr("a"), "0x11111111"), CallStep(addr("e"), "0xeeeeeeee"))),
            record("normal-record-n3", steps=(CallStep(addr("e"), "0xeeeeeeee"),)),
        )
        normal = baseline(*normals)
        result = prepare(extracted=features(("strict_result_chain_001", query_record)), normal=normal)
        references = result.gate_evidence.query_chains[0].normal_references
        self.assertEqual([item.reference_alias for item in references], ["R1", "R2", "R3"])
        self.assertEqual([item.chain_distance for item in references], [Decimal("0"), Decimal("0.5"), Decimal("1")])

    def test_baseline_record_is_looked_up_and_aliases_reuse_without_collision(self) -> None:
        result = prepare()
        query = result.gate_evidence.query_chains[0]
        aliases = [item.normal_transaction_alias for item in query.normal_references]
        self.assertEqual(len(set(aliases)), 3)
        self.assertEqual(query.main_anchor.address_alias, query.call_evidence[0].address_alias)
        self.assertEqual(query.asset_evidence[0].asset_alias, query.normal_references[0].asset_evidence[0].asset_alias)

    def test_references_have_no_amount_and_unique_query_amount_is_retained(self) -> None:
        result = prepare()
        query = result.gate_evidence.query_chains[0]
        self.assertEqual(query.asset_evidence[0].amount_observation, "123")
        self.assertTrue(all(asset.amount_observation is None for ref in query.normal_references for asset in ref.asset_evidence))
        package = json.loads(result.prompt.user_message)
        self.assertNotIn("amount_observation", package["query_chains"][0]["normal_references"][0]["asset_evidence"][0])

    def test_missing_or_ambiguous_amount_is_not_inferred(self) -> None:
        missing = prepare(strict=strict_result(strict_chain(amount_raw=None)))
        self.assertIsNone(missing.gate_evidence.query_chains[0].asset_evidence[0].amount_observation)
        self.assertIn("QUERY_AMOUNT_OBSERVATION_UNAVAILABLE", {item.code for item in missing.issues})

        assets = frozenset({AssetBehavior(TOKEN_1, "INFLOW"), AssetBehavior(TOKEN_1, "OUTFLOW")})
        normals = tuple(record(item.record_id, assets=assets) for item in normal_records())
        ambiguous = prepare(
            extracted=features(("strict_result_chain_001", record("query-record-1", assets=assets))),
            normal=baseline(*normals),
        )
        self.assertTrue(all(item.amount_observation is None for item in ambiguous.gate_evidence.query_chains[0].asset_evidence))
        self.assertIn("QUERY_AMOUNT_ASSET_MATCH_AMBIGUOUS", {item.code for item in ambiguous.issues})

    def test_amount_change_does_not_change_quantitative_status_or_eligibility(self) -> None:
        small = prepare(strict=strict_result(strict_chain(amount_raw="1")))
        large = prepare(strict=strict_result(strict_chain(amount_raw="999999999999999999999")))
        self.assertEqual(small.eligibility.to_dict(), large.eligibility.to_dict())
        self.assertEqual(
            small.gate_evidence.strict_chains[0].quantitative_status,
            large.gate_evidence.strict_chains[0].quantitative_status,
        )

    def test_missing_normal_record_or_transaction_mapping_blocks_prompt(self) -> None:
        normal = baseline()
        broken_group = replace(normal.groups[0], records=normal.groups[0].records[1:])
        broken = replace(normal, groups=(broken_group,))
        missing_record = prepare(normal=broken)
        self.assertEqual(missing_record.preparation_status, "INELIGIBLE")
        self.assertIsNone(missing_record.prompt)

        ownership = mapping(*normal.groups[0].records)
        ownership.pop(normal.groups[0].records[0].record_id)
        missing_mapping = prepare(normal=normal, ownership=ownership)
        self.assertIn("NORMAL_TRANSACTION_MAPPING_MISSING", {item.code for item in missing_mapping.issues})
        self.assertIsNone(missing_mapping.prompt)

    def test_anchor_mismatch_blocks_prompt(self) -> None:
        normal = baseline()
        wrong = replace(normal.groups[0].records[0], main_anchor=ANCHOR_2)
        broken_group = replace(normal.groups[0], records=(wrong, *normal.groups[0].records[1:]))
        result = prepare(normal=replace(normal, groups=(broken_group,)))
        self.assertEqual(result.preparation_status, "INELIGIBLE")
        self.assertIsNone(result.prompt)

    def test_missing_or_duplicate_feature_mapping_blocks_prompt(self) -> None:
        missing = prepare(extracted=features(("other_chain", record("query-record-1"))))
        self.assertIn("FEATURE_CHAIN_MAPPING_MISSING", {item.code for item in missing.issues})
        self.assertIsNone(missing.prompt)
        duplicate = prepare(extracted=features(duplicate_status=True))
        self.assertIn("FEATURE_CHAIN_MAPPING_DUPLICATE", {item.code for item in duplicate.issues})
        self.assertIsNone(duplicate.prompt)

    def test_same_query_record_cannot_map_to_multiple_strict_chains(self) -> None:
        shared = record("query-record-1")
        base = features(("strict_result_chain_001", shared))
        second_status = replace(base.chain_statuses[0], strict_chain_id="strict_result_chain_002")
        extracted = replace(
            base,
            chain_statuses=(base.chain_statuses[0], second_status),
            input_chain_count=2,
        )
        result = prepare(
            strict=strict_result(strict_chain("strict_result_chain_001"), strict_chain("strict_result_chain_002")),
            extracted=extracted,
        )
        self.assertIn("QUERY_RECORD_REUSED", {item.code for item in result.issues})
        self.assertIsNone(result.prompt)

    def test_uncertain_chain_is_excluded_without_blocking_clean_ready_chain(self) -> None:
        clean = strict_chain("strict_result_chain_001")
        uncertain_chains = tuple(
            strict_chain(f"strict_result_chain_{index:03d}", status="UNCERTAIN")
            for index in range(2, 5)
        )
        uncertain_records = tuple(
            record(f"query-record-{index}", uncertainties=("TRACE_EVIDENCE_INCOMPLETE",))
            for index in range(2, 5)
        )
        result = prepare(
            strict=strict_result(clean, *uncertain_chains),
            extracted=features(
                (clean.strict_chain_id, record("query-record-1")),
                *((chain.strict_chain_id, item) for chain, item in zip(uncertain_chains, uncertain_records)),
            ),
        )
        self.assertEqual(result.preparation_status, "READY_FOR_LLM")
        self.assertEqual(result.gate_evidence.total_strict_chain_count, 4)
        self.assertEqual(len(result.gate_evidence.strict_chains), 1)
        self.assertEqual(result.gate_evidence.excluded_uncertain_chain_count, 3)
        self.assertEqual(result.prompt.allowed_chain_aliases, ("Q1",))
        self.assertEqual(result.prompt.citation_whitelist.chain_aliases, ("Q1",))
        self.assertIn("UNCERTAIN_CHAIN_EXCLUDED", {item.code for item in result.issues})

    def test_filtered_aliases_are_stable_when_input_order_changes(self) -> None:
        ready_1 = strict_chain("strict_result_chain_001")
        uncertain = strict_chain("strict_result_chain_002", status="UNCERTAIN")
        ready_3 = strict_chain("strict_result_chain_003")
        pairs = (
            (ready_1.strict_chain_id, record("query-record-1")),
            (uncertain.strict_chain_id, record("query-record-2", uncertainties=("TRACE_EVIDENCE_INCOMPLETE",))),
            (ready_3.strict_chain_id, record("query-record-3")),
        )
        left = prepare(strict=strict_result(ready_1, uncertain, ready_3), extracted=features(*pairs))
        right = prepare(
            strict=strict_result(ready_3, uncertain, ready_1),
            extracted=features(*reversed(pairs)),
        )
        self.assertEqual(left.prompt.allowed_chain_aliases, ("Q1", "Q2"))
        self.assertEqual(
            serialize_phase1_normality_preparation(left),
            serialize_phase1_normality_preparation(right),
        )

    def test_quantitative_fail_or_uncertain_clean_chain_is_ineligible(self) -> None:

        fail_query = record("query-record-1", steps=(CallStep(addr("f"), "0xffffffff"),))
        fail = prepare(extracted=features(("strict_result_chain_001", fail_query)))
        self.assertIn("QUANTITATIVE_NOT_PASS", fail.eligibility.reason_codes)
        self.assertIsNone(fail.prompt)

        uncertain_record = record("query-record-1", uncertainties=("QUERY_UNCERTAIN",))
        uncertain = prepare(extracted=features(("strict_result_chain_001", uncertain_record)))
        self.assertIn("NO_USABLE_CLEAN_CHAIN", uncertain.eligibility.reason_codes)
        self.assertIsNone(uncertain.prompt)

    def test_exact_clean_chain_requires_existing_direct_normal_route(self) -> None:
        exact_normals = tuple(record(f"normal-record-n{index}") for index in range(1, 4))
        result = prepare(
            extracted=features(("strict_result_chain_001", record("query-record-1"))),
            normal=baseline(*exact_normals),
            ownership=mapping(*exact_normals),
        )
        self.assertEqual(result.preparation_status, "INELIGIBLE")
        self.assertIsNone(result.prompt)
        self.assertIn("DIRECT_NORMAL_ROUTE_REQUIRED", result.eligibility.reason_codes)

    def test_ready_chain_mapping_errors_remain_fatal(self) -> None:
        missing = prepare(extracted=replace(features(), chain_statuses=()))
        self.assertIn("FEATURE_CHAIN_MAPPING_MISSING", {item.code for item in missing.issues})
        duplicate = prepare(extracted=features(duplicate_status=True))
        self.assertIn("FEATURE_CHAIN_MAPPING_DUPLICATE", {item.code for item in duplicate.issues})
        self.assertEqual(missing.preparation_status, "INELIGIBLE")
        self.assertEqual(duplicate.preparation_status, "INELIGIBLE")

    def test_missing_threshold_small_group_and_threshold_one_are_ineligible(self) -> None:
        small_normal = baseline(*normal_records()[:2], min_group_size=3)
        unavailable = prepare(normal=small_normal)
        self.assertIn("THRESHOLD_UNAVAILABLE", unavailable.eligibility.reason_codes)
        self.assertIn("NORMAL_GROUP_TOO_SMALL", unavailable.eligibility.reason_codes)

        diverse = (
            record("normal-record-n1", steps=(CallStep(addr("a"), "0x11111111"),)),
            record("normal-record-n2", steps=(CallStep(addr("e"), "0xeeeeeeee"),)),
            record("normal-record-n3", steps=(CallStep(addr("f"), "0xffffffff"),)),
        )
        one = prepare(extracted=features(("strict_result_chain_001", diverse[0])), normal=baseline(*diverse))
        self.assertIn("THRESHOLD_NON_DISCRIMINATIVE", one.eligibility.reason_codes)
        self.assertIsNone(one.prompt)

    def test_top_three_incomplete_is_ineligible(self) -> None:
        normal = baseline(*normal_records()[:2], min_group_size=2)
        result = prepare(normal=normal)
        self.assertIn("TOP3_REFERENCE_INCOMPLETE", result.eligibility.reason_codes)
        self.assertIsNone(result.prompt)

    def test_invalid_or_incomplete_coverage_and_critical_missing_are_safe(self) -> None:
        for coverage in (None, "not-a-number", Decimal("1.1")):
            with self.subTest(coverage=coverage):
                result = prepare(strict=strict_result(coverage=coverage))
                self.assertEqual(result.preparation_status, "INELIGIBLE")
                self.assertIsNone(result.prompt)
        incomplete = prepare(strict=strict_result(coverage="0.9999"))
        self.assertIn("RESULT_COVERAGE_INCOMPLETE", incomplete.eligibility.reason_codes)
        critical = prepare(strict=strict_result(critical_missing=1))
        self.assertIn("CRITICAL_EVIDENCE_MISSING", critical.eligibility.reason_codes)
        self.assertIsNone(critical.prompt)

    def test_serialization_is_sanitized(self) -> None:
        result = prepare()
        text = serialize_phase1_normality_preparation(result)
        for forbidden in (
            addr("a"), TOKEN_1, *TX_HASHES.values(), "normal-record-n1", "query-record-1",
            "calldata", "revertReason",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIn("0x11111111", text)

    def test_input_order_does_not_change_serialization(self) -> None:
        records_1 = normal_records(ANCHOR_1, "n")
        records_2 = tuple(
            record(
                f"normal-record-m{index}",
                anchor=ANCHOR_2,
                assets=frozenset({AssetBehavior(TOKEN_2, "OUTFLOW")}),
            )
            for index in range(1, 4)
        )
        normal_left = baseline(*(records_1 + records_2))
        normal_right = baseline(*(reversed(records_1 + records_2)))
        query_1 = record("query-record-1", anchor=ANCHOR_1)
        query_2 = record("query-record-2", anchor=ANCHOR_2, assets=frozenset({AssetBehavior(TOKEN_2, "OUTFLOW")}))
        chains = (
            strict_chain("strict_result_chain_001", anchor=ANCHOR_1),
            strict_chain("strict_result_chain_002", anchor=ANCHOR_2, asset_address=TOKEN_2),
        )
        ownership = {item.record_id: TX_HASHES[item.record_id[-2:]] for item in records_1 + records_2}
        left = prepare(strict_result(*chains), features((chains[0].strict_chain_id, query_1), (chains[1].strict_chain_id, query_2)), normal_left, ownership)
        right = prepare(strict_result(*reversed(chains)), features((chains[1].strict_chain_id, query_2), (chains[0].strict_chain_id, query_1)), normal_right, ownership)
        self.assertEqual(serialize_phase1_normality_preparation(left), serialize_phase1_normality_preparation(right))

    def test_invalid_input_has_no_prompt_and_no_transaction_decision(self) -> None:
        result = prepare(strict=strict_result(coverage="bad"))
        payload = json.loads(serialize_phase1_normality_preparation(result))
        self.assertIsNone(payload["prompt"])
        self.assertNotIn("transaction_decision", payload)
        self.assertNotIn("EARLY_NORMAL", json.dumps(payload))

    def test_formal_query_missing_record_id_blocks_without_fabrication(self) -> None:
        fake = NormalBaselineQuery(
            query_record_id="query-record-1",
            main_anchor=ANCHOR_1,
            group_status="CALIBRATED",
            group_threshold=Decimal("0"),
            top_matches=(
                TopNormalMatch(
                    "record-that-does-not-exist",
                    1,
                    ChainDistance("query-record-1", "record-that-does-not-exist", Decimal(0), Decimal(0), Decimal(0)),
                ),
            ),
            nearest_distance=Decimal(0),
            status="PASS",
            reasons=("NEAREST_DISTANCE_WITHIN_GROUP_THRESHOLD",),
        )
        with patch("src.cloc_v2.phase1_normality_bridge.query_normal_baseline", return_value=fake):
            result = prepare()
        self.assertIn("NORMAL_RECORD_NOT_FOUND", {item.code for item in result.issues})
        self.assertIsNone(result.prompt)

    def test_duplicate_normal_record_id_blocks_prompt(self) -> None:
        normal = baseline()
        records_ = normal.groups[0].records
        duplicated_group = replace(normal.groups[0], records=(records_[0], records_[0], *records_[1:]))
        result = prepare(normal=replace(normal, groups=(duplicated_group,)))
        self.assertIn("NORMAL_RECORD_ID_DUPLICATE", {item.code for item in result.issues})
        self.assertIsNone(result.prompt)

    def test_calibration_minimum_other_than_three_is_ineligible(self) -> None:
        result = prepare(normal=baseline(*normal_records(), min_group_size=2))
        self.assertIn("FORMAL_MIN_GROUP_SIZE_MISMATCH", result.eligibility.reason_codes)
        self.assertIsNone(result.prompt)

    def test_missing_coverage_field_is_rejected(self) -> None:
        malformed = replace(strict_result(), coverage={"critical_missing_count": 0})
        result = prepare(strict=malformed)
        self.assertIn("RESULT_COVERAGE_FIELD_INVALID", {item.code for item in result.issues})
        self.assertIsNone(result.prompt)

    def test_contradictory_coverage_counts_are_rejected(self) -> None:
        malformed = replace(
            strict_result(),
            coverage={
                "result_candidate_count": 2,
                "chains_formed_result_candidate_count": 1,
                "uncertain_result_candidate_count": 0,
                "critical_missing_count": 0,
                "coverage_rate": "1.0000",
            },
        )
        result = prepare(strict=malformed)
        self.assertIn("RESULT_COVERAGE_FIELDS_CONTRADICTORY", {item.code for item in result.issues})
        self.assertIsNone(result.prompt)

    def test_invalid_critical_missing_type_is_rejected(self) -> None:
        malformed = replace(strict_result(), coverage={"coverage_rate": "1.0000", "critical_missing_count": "0"})
        result = prepare(strict=malformed)
        self.assertIn("CRITICAL_MISSING_COUNT_FIELD_INVALID", {item.code for item in result.issues})
        self.assertIsNone(result.prompt)

    def test_same_normal_transaction_reuses_one_h_alias(self) -> None:
        normal = baseline()
        ownership = {item.record_id: TX_HASHES["n1"] for item in normal.groups[0].records}
        result = prepare(normal=normal, ownership=ownership)
        aliases = {item.normal_transaction_alias for item in result.prompt.citation_whitelist.chain_scopes and result.gate_evidence.query_chains[0].normal_references}
        self.assertEqual(aliases, {"H1"})

    def test_missing_mapping_is_not_guessed_from_record_id(self) -> None:
        embedded = record("normal-" + TX_HASHES["n1"])
        other = normal_records()[1:]
        normal = baseline(embedded, *other)
        ownership = {item.record_id: TX_HASHES[item.record_id[-2:]] for item in other}
        result = prepare(normal=normal, ownership=ownership)
        self.assertIn("NORMAL_TRANSACTION_MAPPING_MISSING", {item.code for item in result.issues})
        self.assertIsNone(result.prompt)

    def test_selector_is_retained_as_function_identity(self) -> None:
        result = prepare()
        self.assertEqual(result.gate_evidence.query_chains[0].main_anchor.selector, "0x11111111")
        self.assertIn("0x11111111", result.prompt.user_message)

    def test_serialization_has_no_label_or_fault_fields(self) -> None:
        payload = json.loads(serialize_phase1_normality_preparation(prepare()))
        serialized = json.dumps(payload, sort_keys=True).lower()
        for forbidden in ("cause", "location", "ground_truth", "fault_label", "paired_transaction"):
            self.assertNotIn(forbidden, serialized)

    def test_no_result_chain_is_ineligible_without_prompt(self) -> None:
        empty_strict = replace(strict_result(), chains=())
        empty_features = NormalFeatureExtractionResult(
            transaction_record_id="query-transaction",
            records=(),
            chain_statuses=(),
            input_chain_count=0,
            record_count=0,
            clean_record_count=0,
            uncertain_record_count=0,
            rejected_chain_count=0,
        )
        result = prepare(strict=empty_strict, extracted=empty_features)
        self.assertIn("NO_RESULT_CHAIN", result.eligibility.reason_codes)
        self.assertIsNone(result.prompt)

    def test_shared_normal_record_across_query_chains_has_distinct_occurrence_aliases(self) -> None:
        query_1 = record("query-record-1")
        query_2 = record("query-record-2")
        extracted = features(
            ("strict_result_chain_001", query_1),
            ("strict_result_chain_002", query_2),
        )
        strict = strict_result(
            strict_chain("strict_result_chain_001"),
            strict_chain("strict_result_chain_002"),
        )
        result = prepare(strict=strict, extracted=extracted)
        self.assertEqual(result.preparation_status, "READY_FOR_LLM")
        self.assertIsNotNone(result.prompt)
        first, second = result.gate_evidence.query_chains
        self.assertEqual(
            [item.reference_alias for item in first.normal_references],
            ["R1", "R2", "R3"],
        )
        self.assertEqual(
            [item.reference_alias for item in second.normal_references],
            ["R4", "R5", "R6"],
        )

    def test_shared_record_reuses_h_a_and_as_but_not_evidence_ids(self) -> None:
        query_1 = record("query-record-1")
        query_2 = record("query-record-2")
        result = prepare(
            strict=strict_result(
                strict_chain("strict_result_chain_001"),
                strict_chain("strict_result_chain_002"),
            ),
            extracted=features(
                ("strict_result_chain_001", query_1),
                ("strict_result_chain_002", query_2),
            ),
        )
        first, second = result.gate_evidence.query_chains
        left = first.normal_references[0]
        right = second.normal_references[0]
        self.assertNotEqual(left.reference_alias, right.reference_alias)
        self.assertEqual(left.normal_transaction_alias, right.normal_transaction_alias)
        self.assertEqual(left.call_evidence[0].address_alias, right.call_evidence[0].address_alias)
        self.assertEqual(left.asset_evidence[0].asset_alias, right.asset_evidence[0].asset_alias)
        self.assertNotEqual(left.call_evidence[0].evidence_id, right.call_evidence[0].evidence_id)
        self.assertNotEqual(left.asset_evidence[0].evidence_id, right.asset_evidence[0].evidence_id)

    def test_each_chain_scope_only_contains_own_reference_and_evidence_ids(self) -> None:
        query_1 = record("query-record-1")
        query_2 = record("query-record-2")
        result = prepare(
            strict=strict_result(
                strict_chain("strict_result_chain_001"),
                strict_chain("strict_result_chain_002"),
            ),
            extracted=features(
                ("strict_result_chain_001", query_1),
                ("strict_result_chain_002", query_2),
            ),
        )
        scopes = {item.chain_alias: item for item in result.prompt.citation_whitelist.chain_scopes}
        self.assertEqual(scopes["Q1"].reference_aliases, ("R1", "R2", "R3"))
        self.assertEqual(scopes["Q2"].reference_aliases, ("R4", "R5", "R6"))
        self.assertTrue(set(scopes["Q1"].evidence_ids).isdisjoint(set(scopes["Q2"].evidence_ids)))

    def test_shared_reference_input_order_has_stable_serialization(self) -> None:
        query_1 = record("query-record-1")
        query_2 = record("query-record-2")
        chains = (
            strict_chain("strict_result_chain_001"),
            strict_chain("strict_result_chain_002"),
        )
        left = prepare(
            strict=strict_result(*chains),
            extracted=features(
                ("strict_result_chain_001", query_1),
                ("strict_result_chain_002", query_2),
            ),
        )
        right = prepare(
            strict=strict_result(*reversed(chains)),
            extracted=features(
                ("strict_result_chain_002", query_2),
                ("strict_result_chain_001", query_1),
            ),
        )
        self.assertEqual(
            serialize_phase1_normality_preparation(left),
            serialize_phase1_normality_preparation(right),
        )

    def test_duplicate_normal_record_inside_one_top3_is_rejected(self) -> None:
        repeated = "normal-record-n1"
        duplicate_query = NormalBaselineQuery(
            query_record_id="query-record-1",
            main_anchor=ANCHOR_1,
            group_status="CALIBRATED",
            group_threshold=Decimal("0"),
            top_matches=(
                TopNormalMatch(
                    repeated,
                    1,
                    ChainDistance("query-record-1", repeated, Decimal(0), Decimal(0), Decimal(0)),
                ),
                TopNormalMatch(
                    repeated,
                    2,
                    ChainDistance("query-record-1", repeated, Decimal(0), Decimal(0), Decimal(0)),
                ),
                TopNormalMatch(
                    "normal-record-n2",
                    3,
                    ChainDistance("query-record-1", "normal-record-n2", Decimal(0), Decimal(0), Decimal(0)),
                ),
            ),
            nearest_distance=Decimal(0),
            status="PASS",
            reasons=("NEAREST_DISTANCE_WITHIN_GROUP_THRESHOLD",),
        )
        with patch("src.cloc_v2.phase1_normality_bridge.query_normal_baseline", return_value=duplicate_query):
            result = prepare()
        self.assertIn("NORMAL_REFERENCE_DUPLICATE_WITHIN_CHAIN", {item.code for item in result.issues})
        self.assertEqual(result.preparation_status, "INELIGIBLE")
        self.assertIsNone(result.prompt)


if __name__ == "__main__":
    unittest.main()

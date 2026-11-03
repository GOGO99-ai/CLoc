from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from src.cloc_v2.normal_baseline import build_normal_baseline
from src.cloc_v2.normal_distance import AssetBehavior, CallStep, ChainFeatureRecord, MainAnchorKey
from src.cloc_v2.normal_features import ChainFeatureExtractionStatus, NormalFeatureExtractionResult
from src.cloc_v2.phase1_normality_bridge import (
    Phase1NormalityPreparation,
    prepare_phase1_normality_review,
)
from src.cloc_v2.phase1_normality_llm import (
    Phase1NormalityLLMConfig,
    build_phase1_normality_request,
    execute_phase1_normality_review,
    parse_phase1_normality_provider_response,
    serialize_phase1_normality_request,
    serialize_phase1_normality_result,
)
from src.cloc_v2.phase1_normality_orchestration import Phase1EligibilityResult
from src.cloc_v2.phase1_normality_prompt import RESPONSE_SCHEMA_VERSION
from src.cloc_v2.strict_result_chains import StrictResultChain, StrictResultChainsResult


ADDRESS = "0x" + "a" * 40
TOKEN = "0x" + "c" * 40
RECIPIENT = "0x" + "f" * 40
SELECTOR = "0x11111111"
NORMAL_IDS = tuple(f"hidden-normal-record-{index}" for index in range(1, 4))
NORMAL_TX_IDS = tuple("0x" + str(index) * 64 for index in range(1, 4))
API_KEY = "api-key-sentinel-must-never-be-serialized"
PROVIDER_SECRET_ERROR = "provider-secret-error-body"


def feature(record_id: str) -> ChainFeatureRecord:
    anchor = MainAnchorKey(1, ADDRESS, SELECTOR)
    normal_index = NORMAL_IDS.index(record_id) + 1 if record_id in NORMAL_IDS else None
    call_steps = (CallStep(ADDRESS, SELECTOR),)
    if normal_index is not None:
        call_steps += (CallStep(ADDRESS, f"0x{normal_index + 1:08x}"),)
    return ChainFeatureRecord(
        record_id=record_id,
        main_anchor=anchor,
        call_steps=call_steps,
        asset_behaviors=frozenset({AssetBehavior(TOKEN, "OUTFLOW")}),
        uncertainties=(),
    )


def ready_preparation() -> Phase1NormalityPreparation:
    normal_records = tuple(feature(record_id) for record_id in NORMAL_IDS)
    baseline = build_normal_baseline(normal_records, min_group_size=3)
    query = feature("hidden-query-record")
    strict_id = "strict_result_chain_001"
    strict_chain = StrictResultChain(
        strict_chain_id=strict_id,
        result_id="hidden-result",
        result_node_id="hidden-result-node",
        result_fact_ids=("hidden-fact",),
        result_type="R_OUTFLOW",
        asset_type="TOKEN",
        asset_address=TOKEN,
        from_address=ADDRESS,
        to_address=RECIPIENT,
        amount_raw="123",
        result_uncertainties=(),
        node_ids=("hidden-node",),
        edge_ids=(),
        anchor=None,
        status="READY",
        uncertainties=(),
    )
    strict = StrictResultChainsResult(
        blockchain_chain_id=1,
        edges=(),
        chains=(strict_chain,),
        result_statuses=(),
        critical_uncertainties=(),
        coverage={
            "result_candidate_count": 1,
            "chains_formed_result_candidate_count": 1,
            "uncertain_result_candidate_count": 0,
            "critical_missing_count": 0,
            "coverage_rate": "1.0000",
        },
    )
    extracted = NormalFeatureExtractionResult(
        transaction_record_id="hidden-query-transaction",
        records=(query,),
        chain_statuses=(
            ChainFeatureExtractionStatus(
                strict_chain_id=strict_id,
                record_id=query.record_id,
                status="EXTRACTED_CLEAN",
                record_created=True,
                reasons=(),
                issues=(),
            ),
        ),
        input_chain_count=1,
        record_count=1,
        clean_record_count=1,
        uncertain_record_count=0,
        rejected_chain_count=0,
    )
    return prepare_phase1_normality_review(
        strict,
        extracted,
        baseline,
        dict(zip(NORMAL_IDS, NORMAL_TX_IDS, strict=True)),
    )


def chain_response(preparation: Phase1NormalityPreparation, decision: str = "NORMAL") -> dict:
    prompt = preparation.prompt
    assert prompt is not None
    scope = prompt.citation_whitelist.chain_scopes[0]
    reference = scope.reference_aliases[0]
    if decision == "NORMAL":
        assessment = "CONSISTENT"
        evidence = [scope.query_call_evidence_ids[0], scope.reference_call_evidence_ids[0]]
        reasons = ["CONSISTENT_CALL_SEQUENCE", "WITHIN_CALIBRATED_THRESHOLD"]
    elif decision == "SUSPICIOUS":
        assessment = "MATERIAL_DEVIATION"
        evidence = [scope.query_call_evidence_ids[0], scope.reference_call_evidence_ids[0]]
        reasons = ["MATERIAL_CALL_DEVIATION"]
    elif decision == "UNCERTAIN":
        assessment = "INSUFFICIENT_EVIDENCE"
        evidence = [scope.query_call_evidence_ids[0]]
        reasons = ["CRITICAL_EVIDENCE_INCOMPLETE"]
    else:
        raise AssertionError(decision)
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "chain_decisions": [
            {
                "chain_alias": scope.chain_alias,
                "decision": decision,
                "reference_assessment": assessment,
                "supporting_reference_aliases": [reference],
                "evidence_ids": evidence,
                "reason_codes": reasons,
                "brief_rationale": f"{scope.chain_alias} compared with {reference} using {' and '.join(evidence)}.",
            }
        ],
    }


def provider_response(
    preparation: Phase1NormalityPreparation,
    decision: str = "NORMAL",
    **overrides,
) -> dict:
    value = {
        "status": "completed",
        "model": "gpt-5.6-terra-2026-08-01",
        "usage": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": json.dumps(chain_response(preparation, decision))}
                ],
            }
        ],
    }
    value.update(overrides)
    return value


class FakeTransport:
    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[dict, str, int]] = []

    def send(self, request_payload, api_key, timeout_seconds):
        self.calls.append((request_payload, api_key, timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.response


class Phase1NormalityLLMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Phase1NormalityLLMConfig()
        self.preparation = ready_preparation()
        self.request = build_phase1_normality_request(self.preparation, self.config)

    def execute(self, response: dict):
        transport = FakeTransport(response)
        result = execute_phase1_normality_review(
            self.preparation, self.config, transport, API_KEY
        )
        return result, transport

    def test_formal_default_config(self) -> None:
        self.assertEqual(self.config.provider, "OPENAI")
        self.assertEqual(self.config.endpoint_path, "/v1/responses")
        self.assertEqual(self.config.max_output_tokens, 8192)
        self.assertEqual(self.config.max_request_attempts, 1)

    def test_request_uses_terra_model(self) -> None:
        self.assertEqual(self.request.payload["model"], "gpt-5.6-terra")

    def test_reasoning_effort_is_medium(self) -> None:
        self.assertEqual(self.request.payload["reasoning"], {"effort": "medium"})

    def test_text_verbosity_is_low(self) -> None:
        self.assertEqual(self.request.payload["text"]["verbosity"], "low")

    def test_store_is_false(self) -> None:
        self.assertIs(self.request.payload["store"], False)

    def test_request_omits_temperature(self) -> None:
        self.assertNotIn("temperature", self.request.payload)

    def test_request_omits_tools_and_conversation_state(self) -> None:
        for field in ("tools", "previous_response_id", "conversation"):
            self.assertNotIn(field, self.request.payload)

    def test_structured_output_uses_strict_schema(self) -> None:
        structured = self.request.payload["text"]["format"]
        self.assertEqual(structured["type"], "json_schema")
        self.assertIs(structured["strict"], True)
        self.assertNotIn("$schema", structured["schema"])
        self.assertIs(structured["schema"]["additionalProperties"], False)
        self.assertIn("required", structured["schema"])

    def test_request_hash_and_serialization_are_stable(self) -> None:
        another = build_phase1_normality_request(ready_preparation(), Phase1NormalityLLMConfig())
        self.assertEqual(self.request.prompt_sha256, another.prompt_sha256)
        self.assertEqual(self.request.request_sha256, another.request_sha256)
        self.assertEqual(
            serialize_phase1_normality_request(self.request),
            serialize_phase1_normality_request(another),
        )

    def test_ineligible_cannot_construct_request(self) -> None:
        ineligible = Phase1NormalityPreparation(
            "INELIGIBLE",
            self.preparation.gate_evidence,
            Phase1EligibilityResult("INELIGIBLE", False, ("CHAIN_SET_MISMATCH",), (), ()),
            None,
            (),
        )
        with self.assertRaisesRegex(ValueError, "PHASE1_PREPARATION_INELIGIBLE"):
            build_phase1_normality_request(ineligible, self.config)

    def test_ineligible_does_not_call_transport(self) -> None:
        ineligible = Phase1NormalityPreparation(
            "INELIGIBLE",
            self.preparation.gate_evidence,
            Phase1EligibilityResult("INELIGIBLE", False, ("CHAIN_SET_MISMATCH",), (), ()),
            None,
            (),
        )
        transport = FakeTransport(provider_response(self.preparation))
        with self.assertRaises(ValueError):
            execute_phase1_normality_review(ineligible, self.config, transport, API_KEY)
        self.assertEqual(transport.calls, [])

    def test_normal_response_validates_and_early_stops(self) -> None:
        result, _ = self.execute(provider_response(self.preparation, "NORMAL"))
        self.assertEqual(result.response_status, "VALIDATED")
        self.assertEqual(result.routing_decision.route, "EARLY_NORMAL")

    def test_suspicious_response_proceeds_phase2(self) -> None:
        result, _ = self.execute(provider_response(self.preparation, "SUSPICIOUS"))
        self.assertEqual(result.response_status, "VALIDATED")
        self.assertEqual(result.routing_decision.route, "PROCEED_PHASE2")
        self.assertIn("LLM_SUSPICIOUS", result.routing_decision.reason_codes)

    def test_uncertain_response_proceeds_phase2(self) -> None:
        result, _ = self.execute(provider_response(self.preparation, "UNCERTAIN"))
        self.assertEqual(result.response_status, "VALIDATED")
        self.assertEqual(result.routing_decision.route, "PROCEED_PHASE2")
        self.assertIn("LLM_UNCERTAIN", result.routing_decision.reason_codes)

    def test_extra_response_field_is_rejected(self) -> None:
        response = chain_response(self.preparation)
        response["transaction_decision"] = "NORMAL"
        provider = provider_response(self.preparation)
        provider["output"][0]["content"][0]["text"] = json.dumps(response)
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")
        self.assertIsNone(result.validated_response)

    def test_unknown_evidence_id_is_rejected(self) -> None:
        response = chain_response(self.preparation)
        response["chain_decisions"][0]["evidence_ids"] = ["E999999"]
        provider = provider_response(self.preparation)
        provider["output"][0]["content"][0]["text"] = json.dumps(response)
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")

    def test_refusal_is_conservative(self) -> None:
        provider = provider_response(self.preparation)
        provider["output"][0]["content"] = [{"type": "refusal", "refusal": PROVIDER_SECRET_ERROR}]
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "REFUSED")
        self.assertEqual(result.fallback_action, "PROCEED_PHASE2")
        self.assertIsNone(result.routing_decision)

    def test_incomplete_is_conservative(self) -> None:
        result, _ = self.execute(provider_response(self.preparation, status="incomplete"))
        self.assertEqual(result.response_status, "INCOMPLETE")
        self.assertEqual(result.fallback_action, "PROCEED_PHASE2")

    def test_provider_failed_and_cancelled_are_conservative(self) -> None:
        for status in ("failed", "cancelled"):
            with self.subTest(status=status):
                result, _ = self.execute(provider_response(self.preparation, status=status))
                self.assertEqual(result.response_status, "PROVIDER_FAILED")
                self.assertEqual(result.fallback_action, "PROCEED_PHASE2")

    def test_multiple_messages_are_rejected(self) -> None:
        provider = provider_response(self.preparation)
        provider["output"].append(provider["output"][0])
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")

    def test_multiple_output_text_items_are_rejected(self) -> None:
        provider = provider_response(self.preparation)
        provider["output"][0]["content"].append(provider["output"][0]["content"][0])
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")

    def test_non_json_output_is_rejected(self) -> None:
        provider = provider_response(self.preparation)
        provider["output"][0]["content"][0]["text"] = "not-json"
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")

    def test_missing_model_is_rejected(self) -> None:
        provider = provider_response(self.preparation)
        provider.pop("model")
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")
        self.assertIn("PROVIDER_MODEL_MISSING", result.reason_codes)

    def test_invalid_usage_is_rejected(self) -> None:
        for invalid in (None, {"input_tokens": "120", "output_tokens": 30, "total_tokens": 150}):
            with self.subTest(invalid=invalid):
                result, _ = self.execute(provider_response(self.preparation, usage=invalid))
                self.assertEqual(result.response_status, "INVALID_RESPONSE")
                self.assertIn("PROVIDER_USAGE_INVALID", result.reason_codes)

    def test_fake_transport_is_called_once(self) -> None:
        _, transport = self.execute(provider_response(self.preparation))
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][2], self.config.timeout_seconds)

    def test_transport_failure_is_not_retried(self) -> None:
        transport = FakeTransport(error=RuntimeError(PROVIDER_SECRET_ERROR))
        result = execute_phase1_normality_review(
            self.preparation, self.config, transport, API_KEY
        )
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(result.response_status, "TRANSPORT_ERROR")
        self.assertEqual(result.fallback_action, "PROCEED_PHASE2")

    def test_api_key_is_not_serialized(self) -> None:
        result, transport = self.execute(provider_response(self.preparation))
        self.assertEqual(transport.calls[0][1], API_KEY)
        self.assertNotIn(API_KEY, serialize_phase1_normality_request(self.request))
        self.assertNotIn(API_KEY, serialize_phase1_normality_result(result))

    def test_provider_error_text_is_not_retained(self) -> None:
        transport = FakeTransport(error=RuntimeError(PROVIDER_SECRET_ERROR))
        result = execute_phase1_normality_review(
            self.preparation, self.config, transport, API_KEY
        )
        serialized = serialize_phase1_normality_result(result)
        self.assertNotIn(PROVIDER_SECRET_ERROR, serialized)
        self.assertEqual(result.reason_codes, ("TRANSPORT_FAILURE",))

    def test_result_does_not_save_raw_provider_response(self) -> None:
        provider = provider_response(self.preparation)
        provider["provider_private_metadata"] = PROVIDER_SECRET_ERROR
        result, _ = self.execute(provider)
        self.assertNotIn(PROVIDER_SECRET_ERROR, serialize_phase1_normality_result(result))
        self.assertNotIn("provider_private_metadata", serialize_phase1_normality_result(result))

    def test_provider_model_and_usage_are_recorded(self) -> None:
        result, _ = self.execute(provider_response(self.preparation))
        self.assertEqual(result.provider_reported_model, "gpt-5.6-terra-2026-08-01")
        self.assertEqual(
            result.usage.to_dict(),
            {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        )

    def test_full_flow_reads_no_environment_config_key_or_label(self) -> None:
        transport = FakeTransport(provider_response(self.preparation))
        with patch.object(os, "getenv", side_effect=AssertionError("environment access forbidden")), patch(
            "builtins.open", side_effect=AssertionError("file access forbidden")
        ):
            result = execute_phase1_normality_review(
                self.preparation, self.config, transport, API_KEY
            )
        self.assertEqual(result.response_status, "VALIDATED")
        self.assertEqual(len(transport.calls), 1)

    def test_request_contains_no_raw_identity_or_record_id(self) -> None:
        serialized = serialize_phase1_normality_request(self.request)
        for forbidden in (ADDRESS, TOKEN, RECIPIENT, *NORMAL_TX_IDS, *NORMAL_IDS, "hidden-query-record"):
            self.assertNotIn(forbidden, serialized)
        self.assertIn(SELECTOR, serialized)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src.cloc_v2.normal_baseline import build_normal_baseline
from src.cloc_v2.normal_distance import AssetBehavior, CallStep, ChainFeatureRecord, MainAnchorKey
from src.cloc_v2.normal_features import ChainFeatureExtractionStatus, NormalFeatureExtractionResult
from src.cloc_v2.phase1_normality_bridge import Phase1NormalityPreparation, prepare_phase1_normality_review
from src.cloc_v2.phase1_normality_deepseek import (
    Phase1DeepSeekConfig,
    build_phase1_deepseek_request,
    execute_phase1_deepseek_review,
    serialize_phase1_deepseek_request,
    serialize_phase1_deepseek_result,
)
from src.cloc_v2.phase1_normality_orchestration import Phase1EligibilityResult
from src.cloc_v2.phase1_normality_prompt import RESPONSE_SCHEMA_VERSION
from src.cloc_v2.strict_result_chains import StrictResultChain, StrictResultChainsResult


ADDRESS = "0x" + "a" * 40
TOKEN = "0x" + "c" * 40
RECIPIENT = "0x" + "f" * 40
SELECTOR = "0x11111111"
NORMAL_IDS = tuple(f"deepseek-hidden-normal-{index}" for index in range(1, 4))
NORMAL_TX_IDS = tuple("0x" + str(index) * 64 for index in range(1, 4))
API_KEY = "deepseek-key-sentinel-never-serialize"
SECRET_REASONING = "private-hidden-reasoning-must-disappear"
SECRET_ERROR = "provider-private-error-must-disappear"


def feature(record_id: str) -> ChainFeatureRecord:
    anchor = MainAnchorKey(1, ADDRESS, SELECTOR)
    normal_index = NORMAL_IDS.index(record_id) + 1 if record_id in NORMAL_IDS else None
    call_steps = (CallStep(ADDRESS, SELECTOR),)
    if normal_index is not None:
        call_steps += (CallStep(ADDRESS, f"0x{normal_index + 1:08x}"),)
    return ChainFeatureRecord(
        record_id,
        anchor,
        call_steps,
        frozenset({AssetBehavior(TOKEN, "OUTFLOW")}),
        (),
    )


def ready_preparation() -> Phase1NormalityPreparation:
    normals = tuple(feature(record_id) for record_id in NORMAL_IDS)
    baseline = build_normal_baseline(normals, min_group_size=3)
    query = feature("deepseek-hidden-query")
    strict_id = "strict_result_chain_001"
    strict = StrictResultChainsResult(
        blockchain_chain_id=1,
        edges=(),
        chains=(
            StrictResultChain(
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
            ),
        ),
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
        transaction_record_id="deepseek-hidden-query-transaction",
        records=(query,),
        chain_statuses=(
            ChainFeatureExtractionStatus(
                strict_id, query.record_id, "EXTRACTED_CLEAN", True, (), ()
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


def decision_response(preparation: Phase1NormalityPreparation, decision: str = "NORMAL") -> dict:
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
    else:
        assessment = "INSUFFICIENT_EVIDENCE"
        evidence = [scope.query_call_evidence_ids[0]]
        reasons = ["CRITICAL_EVIDENCE_INCOMPLETE"]
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


def provider_response(preparation: Phase1NormalityPreparation, decision: str = "NORMAL", **overrides) -> dict:
    value = {
        "id": "chatcmpl-sanitized",
        "object": "chat.completion",
        "model": "deepseek-v4-pro-2026-08",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(decision_response(preparation, decision)),
                    "reasoning_content": SECRET_REASONING,
                },
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
    }
    value.update(overrides)
    return value


class FakeTransport:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.calls = []

    def send(self, request_payload, api_key, timeout_seconds):
        self.calls.append((request_payload, api_key, timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.response


class Phase1NormalityDeepSeekTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Phase1DeepSeekConfig()
        self.preparation = ready_preparation()
        self.request = build_phase1_deepseek_request(self.preparation, self.config)

    def execute(self, response):
        transport = FakeTransport(response=response)
        result = execute_phase1_deepseek_review(
            self.preparation, self.config, transport, API_KEY
        )
        return result, transport

    def test_default_config_is_frozen(self) -> None:
        self.assertEqual(self.config.provider, "DEEPSEEK")
        self.assertEqual(self.config.endpoint_path, "/chat/completions")
        self.assertEqual(self.config.model, "deepseek-v4-pro")
        self.assertEqual(self.config.max_request_attempts, 1)

    def test_request_uses_chat_messages(self) -> None:
        payload = self.request.payload
        self.assertEqual([item["role"] for item in payload["messages"]], ["system", "user"])
        self.assertEqual(len(payload["messages"]), 2)

    def test_thinking_is_enabled(self) -> None:
        self.assertEqual(self.request.payload["thinking"], {"type": "enabled"})

    def test_reasoning_effort_is_high(self) -> None:
        self.assertEqual(self.request.payload["reasoning_effort"], "high")

    def test_response_format_is_json_object(self) -> None:
        self.assertEqual(self.request.payload["response_format"], {"type": "json_object"})

    def test_max_tokens_is_8192(self) -> None:
        self.assertEqual(self.request.payload["max_tokens"], 8192)

    def test_temperature_and_tools_are_not_sent(self) -> None:
        self.assertNotIn("temperature", self.request.payload)
        self.assertNotIn("tools", self.request.payload)

    def test_request_hash_and_serialization_are_stable(self) -> None:
        other = build_phase1_deepseek_request(ready_preparation(), Phase1DeepSeekConfig())
        self.assertEqual(self.request.prompt_sha256, other.prompt_sha256)
        self.assertEqual(self.request.request_sha256, other.request_sha256)
        self.assertEqual(
            serialize_phase1_deepseek_request(self.request),
            serialize_phase1_deepseek_request(other),
        )

    def test_ineligible_cannot_build_or_call_transport(self) -> None:
        ineligible = Phase1NormalityPreparation(
            "INELIGIBLE",
            self.preparation.gate_evidence,
            Phase1EligibilityResult("INELIGIBLE", False, ("CHAIN_SET_MISMATCH",), (), ()),
            None,
            (),
        )
        transport = FakeTransport(provider_response(self.preparation))
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_PREPARATION_INELIGIBLE"):
            execute_phase1_deepseek_review(ineligible, self.config, transport, API_KEY)
        self.assertEqual(transport.calls, [])

    def test_normal_response_uses_existing_router(self) -> None:
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

    def test_extra_field_is_locally_rejected(self) -> None:
        response = decision_response(self.preparation)
        response["transaction_decision"] = "NORMAL"
        provider = provider_response(self.preparation)
        provider["choices"][0]["message"]["content"] = json.dumps(response)
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")
        self.assertIn("LOCAL_RESPONSE_VALIDATION_FAILED", result.reason_codes)

    def test_unknown_evidence_is_locally_rejected(self) -> None:
        response = decision_response(self.preparation)
        response["chain_decisions"][0]["evidence_ids"] = ["E999999"]
        provider = provider_response(self.preparation)
        provider["choices"][0]["message"]["content"] = json.dumps(response)
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")

    def test_reasoning_content_is_discarded(self) -> None:
        result, _ = self.execute(provider_response(self.preparation))
        serialized = serialize_phase1_deepseek_result(result)
        self.assertNotIn("reasoning_content", serialized)
        self.assertNotIn(SECRET_REASONING, serialized)
        self.assertNotIn(SECRET_REASONING, json.dumps(result.validated_response))

    def test_refusal_is_conservative(self) -> None:
        provider = provider_response(self.preparation)
        provider["choices"][0]["message"]["refusal"] = SECRET_ERROR
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "REFUSED")
        self.assertEqual(result.fallback_action, "PROCEED_PHASE2")
        self.assertIsNone(result.routing_decision)

    def test_length_finish_is_incomplete(self) -> None:
        provider = provider_response(self.preparation)
        provider["choices"][0]["finish_reason"] = "length"
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INCOMPLETE")
        self.assertEqual(result.fallback_action, "PROCEED_PHASE2")

    def test_provider_error_is_conservative_and_sanitized(self) -> None:
        result, _ = self.execute({"error": {"message": SECRET_ERROR}})
        self.assertEqual(result.response_status, "PROVIDER_FAILED")
        self.assertEqual(result.fallback_action, "PROCEED_PHASE2")
        self.assertNotIn(SECRET_ERROR, serialize_phase1_deepseek_result(result))

    def test_multiple_choices_are_rejected(self) -> None:
        provider = provider_response(self.preparation)
        provider["choices"].append(provider["choices"][0])
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")

    def test_non_json_content_is_rejected(self) -> None:
        provider = provider_response(self.preparation)
        provider["choices"][0]["message"]["content"] = "not-json"
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")

    def test_missing_model_is_rejected(self) -> None:
        provider = provider_response(self.preparation)
        provider.pop("model")
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")
        self.assertIn("PROVIDER_MODEL_MISSING", result.reason_codes)

    def test_invalid_usage_is_rejected(self) -> None:
        provider = provider_response(self.preparation, usage={"prompt_tokens": "200"})
        result, _ = self.execute(provider)
        self.assertEqual(result.response_status, "INVALID_RESPONSE")
        self.assertIn("PROVIDER_USAGE_INVALID", result.reason_codes)

    def test_fake_transport_is_called_once(self) -> None:
        _, transport = self.execute(provider_response(self.preparation))
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][2], self.config.timeout_seconds)

    def test_transport_error_is_not_retried(self) -> None:
        transport = FakeTransport(error=RuntimeError(SECRET_ERROR))
        result = execute_phase1_deepseek_review(
            self.preparation, self.config, transport, API_KEY
        )
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(result.response_status, "TRANSPORT_ERROR")
        self.assertNotIn(SECRET_ERROR, serialize_phase1_deepseek_result(result))

    def test_key_only_reaches_transport_and_is_never_serialized(self) -> None:
        result, transport = self.execute(provider_response(self.preparation))
        self.assertEqual(transport.calls[0][1], API_KEY)
        self.assertNotIn(API_KEY, serialize_phase1_deepseek_request(self.request))
        self.assertNotIn(API_KEY, serialize_phase1_deepseek_result(result))

    def test_raw_provider_response_is_not_saved(self) -> None:
        provider = provider_response(self.preparation)
        provider["private_provider_field"] = SECRET_ERROR
        result, _ = self.execute(provider)
        serialized = serialize_phase1_deepseek_result(result)
        self.assertNotIn("private_provider_field", serialized)
        self.assertNotIn(SECRET_ERROR, serialized)

    def test_provider_model_and_usage_are_recorded(self) -> None:
        result, _ = self.execute(provider_response(self.preparation))
        self.assertEqual(result.provider_reported_model, "deepseek-v4-pro-2026-08")
        self.assertEqual(
            result.usage.to_dict(),
            {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
        )

    def test_request_contains_no_raw_identity(self) -> None:
        serialized = serialize_phase1_deepseek_request(self.request)
        for forbidden in (ADDRESS, TOKEN, RECIPIENT, *NORMAL_TX_IDS, *NORMAL_IDS, "deepseek-hidden-query"):
            self.assertNotIn(forbidden, serialized)
        self.assertIn(SELECTOR, serialized)

    def test_full_flow_reads_no_environment_or_files(self) -> None:
        transport = FakeTransport(provider_response(self.preparation))
        with patch.object(os, "getenv", side_effect=AssertionError("env forbidden")), patch(
            "builtins.open", side_effect=AssertionError("file forbidden")
        ):
            result = execute_phase1_deepseek_review(
                self.preparation, self.config, transport, API_KEY
            )
        self.assertEqual(result.response_status, "VALIDATED")
        self.assertEqual(len(transport.calls), 1)


if __name__ == "__main__":
    unittest.main()

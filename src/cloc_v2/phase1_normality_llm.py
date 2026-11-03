"""Formal phase-1 normality client contract for the OpenAI Responses API.

This module has no built-in HTTP implementation and never reads environment
variables, files, configuration, labels, or secrets.  Callers must explicitly
inject a transport and an API key.  Every execution makes at most one transport
call and all provider output is locally validated before routing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .phase1_normality_bridge import Phase1NormalityPreparation
from .phase1_normality_orchestration import Phase1RoutingDecision, decide_phase1_route
from .phase1_normality_prompt import (
    serialize_phase1_normality_prompt,
    validate_phase1_normality_response,
)


PROVIDER = "OPENAI"
ENDPOINT_PATH = "/v1/responses"
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
TEXT_VERBOSITY = "low"
MAX_OUTPUT_TOKENS = 8192
MAX_REQUEST_ATTEMPTS = 1

RESULT_STATUSES = frozenset(
    {"VALIDATED", "REFUSED", "INCOMPLETE", "PROVIDER_FAILED", "INVALID_RESPONSE", "TRANSPORT_ERROR"}
)
RESULT_REASON_CODES = frozenset(
    {
        "RESPONSE_VALIDATED",
        "PROVIDER_REFUSAL",
        "PROVIDER_RESPONSE_INCOMPLETE",
        "PROVIDER_RESPONSE_FAILED",
        "PROVIDER_RESPONSE_CANCELLED",
        "PROVIDER_STATUS_UNSUPPORTED",
        "PROVIDER_MODEL_MISSING",
        "PROVIDER_USAGE_INVALID",
        "PROVIDER_OUTPUT_INVALID",
        "PROVIDER_ASSISTANT_MESSAGE_COUNT_INVALID",
        "PROVIDER_OUTPUT_TEXT_COUNT_INVALID",
        "PROVIDER_OUTPUT_TEXT_JSON_INVALID",
        "LOCAL_RESPONSE_VALIDATION_FAILED",
        "REQUEST_PREPARATION_MISMATCH",
        "TRANSPORT_FAILURE",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_TRANSACTION_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")


class Phase1NormalityTransport(Protocol):
    """Caller-injected one-shot transport; no implementation is provided here."""

    def send(
        self,
        request_payload: Mapping[str, Any],
        api_key: str,
        timeout_seconds: int,
    ) -> Mapping[str, Any]: ...


class Phase1TransportError(Exception):
    """Sanitized transport failure that never retains provider error text."""

    def __init__(self) -> None:
        super().__init__("PHASE1_TRANSPORT_ERROR")


@dataclass(frozen=True)
class Phase1NormalityLLMConfig:
    provider: str = PROVIDER
    endpoint_path: str = ENDPOINT_PATH
    model: str = MODEL
    reasoning_effort: str = REASONING_EFFORT
    text_verbosity: str = TEXT_VERBOSITY
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    store: bool = False
    timeout_seconds: int = 60
    max_request_attempts: int = MAX_REQUEST_ATTEMPTS

    def __post_init__(self) -> None:
        if self.provider != PROVIDER:
            raise ValueError("PHASE1_PROVIDER_CONFIG_INVALID")
        if self.endpoint_path != ENDPOINT_PATH:
            raise ValueError("PHASE1_ENDPOINT_CONFIG_INVALID")
        if self.model != MODEL:
            raise ValueError("PHASE1_MODEL_CONFIG_INVALID")
        if self.reasoning_effort != REASONING_EFFORT:
            raise ValueError("PHASE1_REASONING_CONFIG_INVALID")
        if self.text_verbosity != TEXT_VERBOSITY:
            raise ValueError("PHASE1_VERBOSITY_CONFIG_INVALID")
        if self.max_output_tokens != MAX_OUTPUT_TOKENS:
            raise ValueError("PHASE1_MAX_OUTPUT_TOKENS_CONFIG_INVALID")
        if self.store is not False:
            raise ValueError("PHASE1_STORE_CONFIG_INVALID")
        if (
            not isinstance(self.timeout_seconds, int)
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("PHASE1_TIMEOUT_CONFIG_INVALID")
        if self.max_request_attempts != MAX_REQUEST_ATTEMPTS:
            raise ValueError("PHASE1_MAX_ATTEMPTS_CONFIG_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint_path": self.endpoint_path,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "text_verbosity": self.text_verbosity,
            "max_output_tokens": self.max_output_tokens,
            "store": self.store,
            "timeout_seconds": self.timeout_seconds,
            "max_request_attempts": self.max_request_attempts,
        }


@dataclass(frozen=True)
class Phase1NormalityUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("PHASE1_USAGE_INVALID")

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class Phase1NormalityRequest:
    provider: str
    endpoint_path: str
    requested_model: str
    reasoning_effort: str
    max_output_tokens: int
    prompt_sha256: str
    request_sha256: str
    _payload_json: str

    def __post_init__(self) -> None:
        if self.provider != PROVIDER or self.endpoint_path != ENDPOINT_PATH:
            raise ValueError("PHASE1_REQUEST_PROVIDER_INVALID")
        if self.requested_model != MODEL or self.reasoning_effort != REASONING_EFFORT:
            raise ValueError("PHASE1_REQUEST_MODEL_CONFIG_INVALID")
        if self.max_output_tokens != MAX_OUTPUT_TOKENS:
            raise ValueError("PHASE1_REQUEST_TOKEN_LIMIT_INVALID")
        if _SHA256.fullmatch(self.prompt_sha256) is None or _SHA256.fullmatch(self.request_sha256) is None:
            raise ValueError("PHASE1_REQUEST_HASH_INVALID")
        try:
            payload = json.loads(self._payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("PHASE1_REQUEST_PAYLOAD_INVALID") from exc
        if not isinstance(payload, dict):
            raise ValueError("PHASE1_REQUEST_PAYLOAD_INVALID")
        if _sha256(_canonical_json(payload)) != self.request_sha256:
            raise ValueError("PHASE1_REQUEST_HASH_MISMATCH")

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._payload_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint_path": self.endpoint_path,
            "requested_model": self.requested_model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "prompt_sha256": self.prompt_sha256,
            "request_sha256": self.request_sha256,
            "request_payload": self.payload,
        }


@dataclass(frozen=True)
class Phase1NormalityLLMResult:
    requested_model: str
    provider_reported_model: str | None
    reasoning_effort: str
    max_output_tokens: int
    prompt_sha256: str
    request_sha256: str
    response_status: str
    usage: Phase1NormalityUsage | None
    _validated_response_json: str | None
    routing_decision: Phase1RoutingDecision | None
    fallback_action: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.requested_model != MODEL or self.reasoning_effort != REASONING_EFFORT:
            raise ValueError("PHASE1_RESULT_MODEL_CONFIG_INVALID")
        if self.max_output_tokens != MAX_OUTPUT_TOKENS:
            raise ValueError("PHASE1_RESULT_TOKEN_LIMIT_INVALID")
        if _SHA256.fullmatch(self.prompt_sha256) is None or _SHA256.fullmatch(self.request_sha256) is None:
            raise ValueError("PHASE1_RESULT_HASH_INVALID")
        if self.provider_reported_model is not None and (
            not isinstance(self.provider_reported_model, str) or not self.provider_reported_model.strip()
        ):
            raise ValueError("PHASE1_PROVIDER_MODEL_INVALID")
        if self.response_status not in RESULT_STATUSES:
            raise ValueError("PHASE1_RESULT_STATUS_INVALID")
        if self.usage is not None and not isinstance(self.usage, Phase1NormalityUsage):
            raise ValueError("PHASE1_RESULT_USAGE_INVALID")
        reasons = tuple(sorted(set(self.reason_codes)))
        if not reasons or not set(reasons) <= RESULT_REASON_CODES:
            raise ValueError("PHASE1_RESULT_REASON_CODE_INVALID")
        object.__setattr__(self, "reason_codes", reasons)
        if self.response_status == "VALIDATED":
            if self._validated_response_json is None or self.routing_decision is None:
                raise ValueError("VALIDATED_RESULT_EVIDENCE_MISSING")
            if self.fallback_action is not None:
                raise ValueError("VALIDATED_RESULT_FALLBACK_FORBIDDEN")
            try:
                parsed = json.loads(self._validated_response_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("VALIDATED_RESPONSE_INVALID") from exc
            if not isinstance(parsed, dict):
                raise ValueError("VALIDATED_RESPONSE_INVALID")
        else:
            if self._validated_response_json is not None or self.routing_decision is not None:
                raise ValueError("UNVALIDATED_RESULT_MUST_NOT_RETAIN_RESPONSE")
            if self.fallback_action != "PROCEED_PHASE2":
                raise ValueError("UNVALIDATED_RESULT_FALLBACK_INVALID")

    @property
    def validated_response(self) -> dict[str, Any] | None:
        return json.loads(self._validated_response_json) if self._validated_response_json is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "provider_reported_model": self.provider_reported_model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "prompt_sha256": self.prompt_sha256,
            "request_sha256": self.request_sha256,
            "response_status": self.response_status,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "validated_response": self.validated_response,
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision is not None else None,
            "fallback_action": self.fallback_action,
            "reason_codes": list(self.reason_codes),
        }


def build_phase1_normality_request(
    preparation: Phase1NormalityPreparation,
    config: Phase1NormalityLLMConfig,
) -> Phase1NormalityRequest:
    """Construct one stable request only for a program-approved preparation."""
    if not isinstance(preparation, Phase1NormalityPreparation):
        raise ValueError("PHASE1_PREPARATION_INVALID")
    if not isinstance(config, Phase1NormalityLLMConfig):
        raise ValueError("PHASE1_LLM_CONFIG_INVALID")
    if (
        preparation.preparation_status != "READY_FOR_LLM"
        or preparation.prompt is None
        or preparation.eligibility.eligibility != "ELIGIBLE"
        or preparation.eligibility.llm_allowed is not True
    ):
        raise ValueError("PHASE1_PREPARATION_INELIGIBLE")

    prompt = preparation.prompt
    schema = prompt.response_schema
    schema.pop("$schema", None)
    payload = {
        "model": config.model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt.system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt.user_message}],
            },
        ],
        "reasoning": {"effort": config.reasoning_effort},
        "text": {
            "verbosity": config.text_verbosity,
            "format": {
                "type": "json_schema",
                "name": "phase1_normality_response",
                "strict": True,
                "schema": schema,
            },
        },
        "max_output_tokens": config.max_output_tokens,
        "store": config.store,
    }
    payload_json = _canonical_json(payload)
    serialized_for_safety = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if _RAW_ADDRESS.search(serialized_for_safety) or _RAW_TRANSACTION_HASH.search(serialized_for_safety):
        raise ValueError("PHASE1_REQUEST_CONTAINS_RAW_CHAIN_IDENTIFIER")
    prompt_sha256 = _sha256(serialize_phase1_normality_prompt(prompt))
    return Phase1NormalityRequest(
        provider=config.provider,
        endpoint_path=config.endpoint_path,
        requested_model=config.model,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=config.max_output_tokens,
        prompt_sha256=prompt_sha256,
        request_sha256=_sha256(payload_json),
        _payload_json=payload_json,
    )


def parse_phase1_normality_provider_response(
    provider_response: Mapping[str, Any],
    preparation: Phase1NormalityPreparation,
    request: Phase1NormalityRequest,
) -> Phase1NormalityLLMResult:
    """Parse one Responses response, locally validate it, then route it."""
    if not isinstance(preparation, Phase1NormalityPreparation) or preparation.prompt is None or preparation.gate_evidence is None:
        raise ValueError("PHASE1_PREPARATION_INVALID")
    if not isinstance(request, Phase1NormalityRequest):
        raise ValueError("PHASE1_REQUEST_INVALID")
    current_prompt_hash = _sha256(serialize_phase1_normality_prompt(preparation.prompt))
    if current_prompt_hash != request.prompt_sha256:
        return _failure_result("INVALID_RESPONSE", request, None, None, "REQUEST_PREPARATION_MISMATCH")
    if not isinstance(provider_response, Mapping):
        return _failure_result("INVALID_RESPONSE", request, None, None, "PROVIDER_OUTPUT_INVALID")

    status = provider_response.get("status")
    model_value = provider_response.get("model")
    provider_model = model_value if isinstance(model_value, str) and model_value.strip() else None
    if status == "incomplete":
        return _failure_result("INCOMPLETE", request, provider_model, None, "PROVIDER_RESPONSE_INCOMPLETE")
    if status in {"failed", "cancelled"}:
        reason = "PROVIDER_RESPONSE_FAILED" if status == "failed" else "PROVIDER_RESPONSE_CANCELLED"
        return _failure_result("PROVIDER_FAILED", request, provider_model, None, reason)
    if status != "completed":
        return _failure_result("PROVIDER_FAILED", request, provider_model, None, "PROVIDER_STATUS_UNSUPPORTED")
    if provider_model is None:
        return _failure_result("INVALID_RESPONSE", request, None, None, "PROVIDER_MODEL_MISSING")

    usage = _usage(provider_response.get("usage"))
    if usage is None:
        return _failure_result("INVALID_RESPONSE", request, provider_model, None, "PROVIDER_USAGE_INVALID")
    output = provider_response.get("output")
    if not isinstance(output, list):
        return _failure_result("INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_OUTPUT_INVALID")
    messages = [
        item
        for item in output
        if isinstance(item, Mapping) and item.get("type") == "message" and item.get("role") == "assistant"
    ]
    if len(messages) != 1:
        return _failure_result(
            "INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_ASSISTANT_MESSAGE_COUNT_INVALID"
        )
    content = messages[0].get("content")
    if not isinstance(content, list):
        return _failure_result("INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_OUTPUT_INVALID")
    if any(isinstance(item, Mapping) and item.get("type") == "refusal" for item in content):
        return _failure_result("REFUSED", request, provider_model, usage, "PROVIDER_REFUSAL")
    output_texts = [item for item in content if isinstance(item, Mapping) and item.get("type") == "output_text"]
    if len(output_texts) != 1 or len(content) != 1:
        return _failure_result(
            "INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_OUTPUT_TEXT_COUNT_INVALID"
        )
    raw_text = output_texts[0].get("text")
    if not isinstance(raw_text, str):
        return _failure_result("INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_OUTPUT_TEXT_JSON_INVALID")
    try:
        decoded = json.loads(raw_text)
    except json.JSONDecodeError:
        return _failure_result("INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_OUTPUT_TEXT_JSON_INVALID")
    try:
        validated = validate_phase1_normality_response(decoded, preparation.prompt)
    except (TypeError, ValueError):
        return _failure_result(
            "INVALID_RESPONSE", request, provider_model, usage, "LOCAL_RESPONSE_VALIDATION_FAILED"
        )
    routing = decide_phase1_route(preparation.gate_evidence, preparation.prompt, validated)
    return Phase1NormalityLLMResult(
        requested_model=request.requested_model,
        provider_reported_model=provider_model,
        reasoning_effort=request.reasoning_effort,
        max_output_tokens=request.max_output_tokens,
        prompt_sha256=request.prompt_sha256,
        request_sha256=request.request_sha256,
        response_status="VALIDATED",
        usage=usage,
        _validated_response_json=_canonical_json(validated),
        routing_decision=routing,
        fallback_action=None,
        reason_codes=("RESPONSE_VALIDATED",),
    )


def execute_phase1_normality_review(
    preparation: Phase1NormalityPreparation,
    config: Phase1NormalityLLMConfig,
    transport: Phase1NormalityTransport,
    api_key: str,
) -> Phase1NormalityLLMResult:
    """Make exactly one injected transport call; never retry on failure."""
    request = build_phase1_normality_request(preparation, config)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("PHASE1_API_KEY_REQUIRED")
    if transport is None or not callable(getattr(transport, "send", None)):
        raise ValueError("PHASE1_TRANSPORT_INVALID")
    try:
        response = transport.send(request.payload, api_key, config.timeout_seconds)
    except Exception:
        return _failure_result("TRANSPORT_ERROR", request, None, None, "TRANSPORT_FAILURE")
    return parse_phase1_normality_provider_response(response, preparation, request)


def serialize_phase1_normality_request(request: Phase1NormalityRequest) -> str:
    if not isinstance(request, Phase1NormalityRequest):
        raise ValueError("PHASE1_REQUEST_INVALID")
    return json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_phase1_normality_result(result: Phase1NormalityLLMResult) -> str:
    if not isinstance(result, Phase1NormalityLLMResult):
        raise ValueError("PHASE1_RESULT_INVALID")
    return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _failure_result(
    status: str,
    request: Phase1NormalityRequest,
    provider_model: str | None,
    usage: Phase1NormalityUsage | None,
    reason: str,
) -> Phase1NormalityLLMResult:
    return Phase1NormalityLLMResult(
        requested_model=request.requested_model,
        provider_reported_model=provider_model,
        reasoning_effort=request.reasoning_effort,
        max_output_tokens=request.max_output_tokens,
        prompt_sha256=request.prompt_sha256,
        request_sha256=request.request_sha256,
        response_status=status,
        usage=usage,
        _validated_response_json=None,
        routing_decision=None,
        fallback_action="PROCEED_PHASE2",
        reason_codes=(reason,),
    )


def _usage(value: Any) -> Phase1NormalityUsage | None:
    if not isinstance(value, Mapping):
        return None
    required = ("input_tokens", "output_tokens", "total_tokens")
    if any(
        key not in value
        or not isinstance(value[key], int)
        or isinstance(value[key], bool)
        or value[key] < 0
        for key in required
    ):
        return None
    return Phase1NormalityUsage(*(value[key] for key in required))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ENDPOINT_PATH",
    "MAX_OUTPUT_TOKENS",
    "MAX_REQUEST_ATTEMPTS",
    "MODEL",
    "PROVIDER",
    "Phase1NormalityLLMConfig",
    "Phase1NormalityLLMResult",
    "Phase1NormalityRequest",
    "Phase1NormalityTransport",
    "Phase1NormalityUsage",
    "Phase1TransportError",
    "build_phase1_normality_request",
    "execute_phase1_normality_review",
    "parse_phase1_normality_provider_response",
    "serialize_phase1_normality_request",
    "serialize_phase1_normality_result",
]

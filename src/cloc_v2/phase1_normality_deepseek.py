"""Independent DeepSeek client contract for formal phase-1 normality review.

The client targets DeepSeek Chat Completions and deliberately shares no client
implementation with the OpenAI Responses client.  It has no built-in network
transport, reads no configuration or secrets, makes at most one injected
transport call, and discards ``reasoning_content`` before local validation.
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


PROVIDER = "DEEPSEEK"
ENDPOINT_PATH = "/chat/completions"
MODEL = "deepseek-v4-pro"
REASONING_EFFORT = "high"
MAX_TOKENS = 8192
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
        "PROVIDER_MODEL_MISSING",
        "PROVIDER_USAGE_INVALID",
        "PROVIDER_CHOICES_INVALID",
        "PROVIDER_MESSAGE_INVALID",
        "PROVIDER_CONTENT_JSON_INVALID",
        "LOCAL_RESPONSE_VALIDATION_FAILED",
        "REQUEST_PREPARATION_MISMATCH",
        "TRANSPORT_FAILURE",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_TRANSACTION_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")


class Phase1DeepSeekTransport(Protocol):
    """Caller-injected one-shot transport; this module provides no HTTP client."""

    def send(
        self,
        request_payload: Mapping[str, Any],
        api_key: str,
        timeout_seconds: int,
    ) -> Mapping[str, Any]: ...


class Phase1DeepSeekTransportError(Exception):
    """Sanitized transport error that cannot retain provider text or secrets."""

    def __init__(self) -> None:
        super().__init__("PHASE1_DEEPSEEK_TRANSPORT_ERROR")


@dataclass(frozen=True)
class Phase1DeepSeekConfig:
    provider: str = PROVIDER
    endpoint_path: str = ENDPOINT_PATH
    model: str = MODEL
    thinking_enabled: bool = True
    reasoning_effort: str = REASONING_EFFORT
    max_tokens: int = MAX_TOKENS
    timeout_seconds: int = 60
    max_request_attempts: int = MAX_REQUEST_ATTEMPTS

    def __post_init__(self) -> None:
        if self.provider != PROVIDER:
            raise ValueError("DEEPSEEK_PROVIDER_CONFIG_INVALID")
        if self.endpoint_path != ENDPOINT_PATH:
            raise ValueError("DEEPSEEK_ENDPOINT_CONFIG_INVALID")
        if self.model != MODEL:
            raise ValueError("DEEPSEEK_MODEL_CONFIG_INVALID")
        if self.thinking_enabled is not True:
            raise ValueError("DEEPSEEK_THINKING_CONFIG_INVALID")
        if self.reasoning_effort != REASONING_EFFORT:
            raise ValueError("DEEPSEEK_REASONING_CONFIG_INVALID")
        if self.max_tokens != MAX_TOKENS:
            raise ValueError("DEEPSEEK_MAX_TOKENS_CONFIG_INVALID")
        if (
            not isinstance(self.timeout_seconds, int)
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("DEEPSEEK_TIMEOUT_CONFIG_INVALID")
        if self.max_request_attempts != MAX_REQUEST_ATTEMPTS:
            raise ValueError("DEEPSEEK_MAX_ATTEMPTS_CONFIG_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint_path": self.endpoint_path,
            "model": self.model,
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_request_attempts": self.max_request_attempts,
        }


@dataclass(frozen=True)
class Phase1DeepSeekUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("DEEPSEEK_USAGE_INVALID")

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class Phase1DeepSeekRequest:
    provider: str
    endpoint_path: str
    requested_model: str
    thinking_enabled: bool
    reasoning_effort: str
    max_tokens: int
    prompt_sha256: str
    request_sha256: str
    _payload_json: str

    def __post_init__(self) -> None:
        if self.provider != PROVIDER or self.endpoint_path != ENDPOINT_PATH:
            raise ValueError("DEEPSEEK_REQUEST_PROVIDER_INVALID")
        if (
            self.requested_model != MODEL
            or self.thinking_enabled is not True
            or self.reasoning_effort != REASONING_EFFORT
            or self.max_tokens != MAX_TOKENS
        ):
            raise ValueError("DEEPSEEK_REQUEST_CONFIG_INVALID")
        if _SHA256.fullmatch(self.prompt_sha256) is None or _SHA256.fullmatch(self.request_sha256) is None:
            raise ValueError("DEEPSEEK_REQUEST_HASH_INVALID")
        try:
            payload = json.loads(self._payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("DEEPSEEK_REQUEST_PAYLOAD_INVALID") from exc
        if not isinstance(payload, dict) or _sha256(_canonical_json(payload)) != self.request_sha256:
            raise ValueError("DEEPSEEK_REQUEST_HASH_MISMATCH")

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._payload_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint_path": self.endpoint_path,
            "requested_model": self.requested_model,
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "prompt_sha256": self.prompt_sha256,
            "request_sha256": self.request_sha256,
            "request_payload": self.payload,
        }


@dataclass(frozen=True)
class Phase1DeepSeekResult:
    requested_model: str
    provider_reported_model: str | None
    thinking_enabled: bool
    reasoning_effort: str
    max_tokens: int
    prompt_sha256: str
    request_sha256: str
    response_status: str
    usage: Phase1DeepSeekUsage | None
    _validated_response_json: str | None
    routing_decision: Phase1RoutingDecision | None
    fallback_action: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.requested_model != MODEL
            or self.thinking_enabled is not True
            or self.reasoning_effort != REASONING_EFFORT
            or self.max_tokens != MAX_TOKENS
        ):
            raise ValueError("DEEPSEEK_RESULT_CONFIG_INVALID")
        if _SHA256.fullmatch(self.prompt_sha256) is None or _SHA256.fullmatch(self.request_sha256) is None:
            raise ValueError("DEEPSEEK_RESULT_HASH_INVALID")
        if self.provider_reported_model is not None and (
            not isinstance(self.provider_reported_model, str) or not self.provider_reported_model.strip()
        ):
            raise ValueError("DEEPSEEK_PROVIDER_MODEL_INVALID")
        if self.response_status not in RESULT_STATUSES:
            raise ValueError("DEEPSEEK_RESULT_STATUS_INVALID")
        if self.usage is not None and not isinstance(self.usage, Phase1DeepSeekUsage):
            raise ValueError("DEEPSEEK_RESULT_USAGE_INVALID")
        reasons = tuple(sorted(set(self.reason_codes)))
        if not reasons or not set(reasons) <= RESULT_REASON_CODES:
            raise ValueError("DEEPSEEK_RESULT_REASON_INVALID")
        object.__setattr__(self, "reason_codes", reasons)
        if self.response_status == "VALIDATED":
            if self._validated_response_json is None or self.routing_decision is None:
                raise ValueError("DEEPSEEK_VALIDATED_RESULT_MISSING")
            if self.fallback_action is not None:
                raise ValueError("DEEPSEEK_VALIDATED_FALLBACK_FORBIDDEN")
            try:
                validated = json.loads(self._validated_response_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("DEEPSEEK_VALIDATED_RESPONSE_INVALID") from exc
            if not isinstance(validated, dict):
                raise ValueError("DEEPSEEK_VALIDATED_RESPONSE_INVALID")
        else:
            if self._validated_response_json is not None or self.routing_decision is not None:
                raise ValueError("DEEPSEEK_UNVALIDATED_DATA_FORBIDDEN")
            if self.fallback_action != "PROCEED_PHASE2":
                raise ValueError("DEEPSEEK_FALLBACK_INVALID")

    @property
    def validated_response(self) -> dict[str, Any] | None:
        return json.loads(self._validated_response_json) if self._validated_response_json is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "provider_reported_model": self.provider_reported_model,
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "prompt_sha256": self.prompt_sha256,
            "request_sha256": self.request_sha256,
            "response_status": self.response_status,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "validated_response": self.validated_response,
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision is not None else None,
            "fallback_action": self.fallback_action,
            "reason_codes": list(self.reason_codes),
        }


def build_phase1_deepseek_request(
    preparation: Phase1NormalityPreparation,
    config: Phase1DeepSeekConfig,
) -> Phase1DeepSeekRequest:
    """Build one DeepSeek request only after the existing hard gate approves it."""
    if not isinstance(preparation, Phase1NormalityPreparation):
        raise ValueError("DEEPSEEK_PREPARATION_INVALID")
    if not isinstance(config, Phase1DeepSeekConfig):
        raise ValueError("DEEPSEEK_CONFIG_INVALID")
    if (
        preparation.preparation_status != "READY_FOR_LLM"
        or preparation.prompt is None
        or preparation.eligibility.eligibility != "ELIGIBLE"
        or preparation.eligibility.llm_allowed is not True
    ):
        raise ValueError("DEEPSEEK_PREPARATION_INELIGIBLE")

    prompt = preparation.prompt
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": prompt.system_message},
            {"role": "user", "content": prompt.user_message},
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": config.reasoning_effort,
        "response_format": {"type": "json_object"},
        "max_tokens": config.max_tokens,
    }
    payload_json = _canonical_json(payload)
    if _RAW_ADDRESS.search(payload_json) or _RAW_TRANSACTION_HASH.search(payload_json):
        raise ValueError("DEEPSEEK_REQUEST_CONTAINS_RAW_CHAIN_IDENTIFIER")
    return Phase1DeepSeekRequest(
        provider=config.provider,
        endpoint_path=config.endpoint_path,
        requested_model=config.model,
        thinking_enabled=config.thinking_enabled,
        reasoning_effort=config.reasoning_effort,
        max_tokens=config.max_tokens,
        prompt_sha256=_sha256(serialize_phase1_normality_prompt(prompt)),
        request_sha256=_sha256(payload_json),
        _payload_json=payload_json,
    )


def parse_phase1_deepseek_response(
    provider_response: Mapping[str, Any],
    preparation: Phase1NormalityPreparation,
    request: Phase1DeepSeekRequest,
) -> Phase1DeepSeekResult:
    """Validate one Chat Completion without retaining hidden reasoning content."""
    if not isinstance(preparation, Phase1NormalityPreparation) or preparation.prompt is None or preparation.gate_evidence is None:
        raise ValueError("DEEPSEEK_PREPARATION_INVALID")
    if not isinstance(request, Phase1DeepSeekRequest):
        raise ValueError("DEEPSEEK_REQUEST_INVALID")
    if _sha256(serialize_phase1_normality_prompt(preparation.prompt)) != request.prompt_sha256:
        return _failure_result("INVALID_RESPONSE", request, None, None, "REQUEST_PREPARATION_MISMATCH")
    if not isinstance(provider_response, Mapping):
        return _failure_result("INVALID_RESPONSE", request, None, None, "PROVIDER_CHOICES_INVALID")

    model_value = provider_response.get("model")
    provider_model = model_value if isinstance(model_value, str) and model_value.strip() else None
    if "error" in provider_response:
        return _failure_result("PROVIDER_FAILED", request, provider_model, None, "PROVIDER_RESPONSE_FAILED")
    if provider_model is None:
        return _failure_result("INVALID_RESPONSE", request, None, None, "PROVIDER_MODEL_MISSING")
    usage = _usage(provider_response.get("usage"))
    if usage is None:
        return _failure_result("INVALID_RESPONSE", request, provider_model, None, "PROVIDER_USAGE_INVALID")
    choices = provider_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return _failure_result("INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_CHOICES_INVALID")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        return _failure_result("INCOMPLETE", request, provider_model, usage, "PROVIDER_RESPONSE_INCOMPLETE")
    if finish_reason == "content_filter":
        return _failure_result("REFUSED", request, provider_model, usage, "PROVIDER_REFUSAL")
    if finish_reason != "stop":
        return _failure_result("PROVIDER_FAILED", request, provider_model, usage, "PROVIDER_RESPONSE_FAILED")
    message = choice.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return _failure_result("INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_MESSAGE_INVALID")
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return _failure_result("REFUSED", request, provider_model, usage, "PROVIDER_REFUSAL")

    # Deliberately read only content. reasoning_content is neither copied nor
    # forwarded into validation, routing, results, serialization, or another turn.
    content = message.get("content")
    if not isinstance(content, str):
        return _failure_result("INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_CONTENT_JSON_INVALID")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        return _failure_result("INVALID_RESPONSE", request, provider_model, usage, "PROVIDER_CONTENT_JSON_INVALID")
    try:
        validated = validate_phase1_normality_response(decoded, preparation.prompt)
    except (TypeError, ValueError):
        return _failure_result("INVALID_RESPONSE", request, provider_model, usage, "LOCAL_RESPONSE_VALIDATION_FAILED")
    routing = decide_phase1_route(preparation.gate_evidence, preparation.prompt, validated)
    return Phase1DeepSeekResult(
        requested_model=request.requested_model,
        provider_reported_model=provider_model,
        thinking_enabled=request.thinking_enabled,
        reasoning_effort=request.reasoning_effort,
        max_tokens=request.max_tokens,
        prompt_sha256=request.prompt_sha256,
        request_sha256=request.request_sha256,
        response_status="VALIDATED",
        usage=usage,
        _validated_response_json=_canonical_json(validated),
        routing_decision=routing,
        fallback_action=None,
        reason_codes=("RESPONSE_VALIDATED",),
    )


def execute_phase1_deepseek_review(
    preparation: Phase1NormalityPreparation,
    config: Phase1DeepSeekConfig,
    transport: Phase1DeepSeekTransport,
    api_key: str,
) -> Phase1DeepSeekResult:
    """Perform exactly one caller-injected transport call and never retry."""
    request = build_phase1_deepseek_request(preparation, config)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("DEEPSEEK_API_KEY_REQUIRED")
    if transport is None or not callable(getattr(transport, "send", None)):
        raise ValueError("DEEPSEEK_TRANSPORT_INVALID")
    try:
        response = transport.send(request.payload, api_key, config.timeout_seconds)
    except Exception:
        return _failure_result("TRANSPORT_ERROR", request, None, None, "TRANSPORT_FAILURE")
    return parse_phase1_deepseek_response(response, preparation, request)


def serialize_phase1_deepseek_request(request: Phase1DeepSeekRequest) -> str:
    if not isinstance(request, Phase1DeepSeekRequest):
        raise ValueError("DEEPSEEK_REQUEST_INVALID")
    return json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_phase1_deepseek_result(result: Phase1DeepSeekResult) -> str:
    if not isinstance(result, Phase1DeepSeekResult):
        raise ValueError("DEEPSEEK_RESULT_INVALID")
    return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _failure_result(
    status: str,
    request: Phase1DeepSeekRequest,
    provider_model: str | None,
    usage: Phase1DeepSeekUsage | None,
    reason: str,
) -> Phase1DeepSeekResult:
    return Phase1DeepSeekResult(
        requested_model=request.requested_model,
        provider_reported_model=provider_model,
        thinking_enabled=request.thinking_enabled,
        reasoning_effort=request.reasoning_effort,
        max_tokens=request.max_tokens,
        prompt_sha256=request.prompt_sha256,
        request_sha256=request.request_sha256,
        response_status=status,
        usage=usage,
        _validated_response_json=None,
        routing_decision=None,
        fallback_action="PROCEED_PHASE2",
        reason_codes=(reason,),
    )


def _usage(value: Any) -> Phase1DeepSeekUsage | None:
    if not isinstance(value, Mapping):
        return None
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(
        field not in value
        or not isinstance(value[field], int)
        or isinstance(value[field], bool)
        or value[field] < 0
        for field in fields
    ):
        return None
    return Phase1DeepSeekUsage(*(value[field] for field in fields))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ENDPOINT_PATH",
    "MAX_REQUEST_ATTEMPTS",
    "MAX_TOKENS",
    "MODEL",
    "PROVIDER",
    "Phase1DeepSeekConfig",
    "Phase1DeepSeekRequest",
    "Phase1DeepSeekResult",
    "Phase1DeepSeekTransport",
    "Phase1DeepSeekTransportError",
    "Phase1DeepSeekUsage",
    "build_phase1_deepseek_request",
    "execute_phase1_deepseek_review",
    "parse_phase1_deepseek_response",
    "serialize_phase1_deepseek_request",
    "serialize_phase1_deepseek_result",
]

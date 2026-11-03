"""One-shot DeepSeek Chat Completions client for phase-2 localization."""
from __future__ import annotations

import hashlib
import json
import re
from urllib import error as urllib_error
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .phase2_candidate_shortlist_prompt import (
    Phase2ShortlistPrompt,
    Phase2ShortlistValidation,
    validate_phase2_candidate_shortlist_response,
)
from .phase2_localization_bridge import Phase2LocalizationPreparation
from .phase2_localization_prompt import (
    DECISION_V2_RESPONSE_SCHEMA_VERSION,
    TOPK_V3_RESPONSE_SCHEMA_VERSION,
    TOPK_V4_RESPONSE_SCHEMA_VERSION,
    materialize_phase2_localization_result_v2,
    materialize_phase2_localization_result_topk_v3,
    materialize_phase2_localization_result_topk_v4,
    validate_phase2_localization_decision_v2,
    validate_phase2_localization_topk_v3,
    validate_phase2_localization_topk_v4,
    validate_phase2_localization_response,
)

PROVIDER = "DEEPSEEK"
ENDPOINT_PATH = "/chat/completions"
MODEL = "deepseek-v4-pro"
MAX_TOKENS = 8192
REASONING_EFFORT = "high"
RESULT_STATUSES = frozenset({"VALIDATED", "VALIDATED_WITH_NORMALIZATION", "INVALID_RESPONSE", "PROVIDER_FAILED", "TRANSPORT_ERROR", "INCOMPLETE"})
TRANSPORT_ERROR_CATEGORIES = frozenset({"NETWORK_ERROR", "TIMEOUT", "HTTP_429", "HTTP_5XX", "HTTP_OTHER_4XX", "OTHER_TRANSPORT_ERROR"})
RETRYABLE_TRANSPORT_ERROR_CATEGORIES = frozenset({"NETWORK_ERROR", "TIMEOUT", "HTTP_429", "HTTP_5XX"})
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ROOT_FIELDS = frozenset({"schema_version", "status", "findings", "summary"})
_CODE_FENCE = re.compile(r"\A```json[ \t]*\r?\n(?P<body>.*?)\r?\n?```[ \t]*\Z", re.DOTALL)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Phase2DeepSeekTransport(Protocol):
    def send(self, request_payload: Mapping[str, Any], api_key: str, timeout_seconds: int) -> Mapping[str, Any]: ...


class Phase2ClassifiedTransportError(Exception):
    """Sanitized error category for transports that can classify failures."""

    def __init__(self, category: str) -> None:
        if category not in TRANSPORT_ERROR_CATEGORIES:
            raise ValueError("PHASE2_TRANSPORT_ERROR_CATEGORY_INVALID")
        self.category = category
        super().__init__(category)


@dataclass(frozen=True)
class Phase2LocalizationResponseNormalization:
    """Only the explicitly allowed provider-root compatibility edits."""

    _normalized_response_json: str | None = field(default=None, repr=False)
    actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(sorted(set(self.actions))))

    @property
    def normalized_response(self) -> dict[str, Any] | None:
        return json.loads(self._normalized_response_json) if self._normalized_response_json is not None else None

    @property
    def applied(self) -> bool:
        return bool(self.actions)


def normalize_phase2_localization_response(content: str) -> Phase2LocalizationResponseNormalization:
    """Parse one JSON object and apply no more than the documented root edits.

    This intentionally never scans prose for JSON and never retains values from
    provider-added root fields.
    """
    if not isinstance(content, str):
        return Phase2LocalizationResponseNormalization()
    actions: list[str] = []
    matched = _CODE_FENCE.fullmatch(content)
    candidate = content
    if matched is not None:
        candidate = matched.group("body")
        actions.append("JSON_CODE_FENCE_REMOVED")
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return Phase2LocalizationResponseNormalization()
    if not isinstance(parsed, Mapping):
        return Phase2LocalizationResponseNormalization(_canonical(parsed), tuple(actions))
    # Summary is presentation-only and is intentionally never retained from a
    # provider response, even when it happens to satisfy the prompt contract.
    normalized = {field: parsed[field] for field in _ROOT_FIELDS if field in parsed and field != "summary"}
    if "schema_version" not in normalized:
        normalized["schema_version"] = "phase2_localization_response_v1"
        actions.append("SCHEMA_VERSION_INSERTED_LOCALLY")
    normalized["summary"] = "SUMMARY_OMITTED_FROM_EVALUATION"
    actions.append("SUMMARY_REPLACED_LOCALLY")
    if any(field not in _ROOT_FIELDS for field in parsed):
        actions.append("EXTRA_ROOT_FIELDS_DROPPED")
    return Phase2LocalizationResponseNormalization(_canonical(normalized), tuple(actions))


@dataclass(frozen=True)
class Phase2DeepSeekConfig:
    provider: str = PROVIDER
    endpoint_path: str = ENDPOINT_PATH
    model: str = MODEL
    thinking_enabled: bool = True
    reasoning_effort: str = REASONING_EFFORT
    max_tokens: int = MAX_TOKENS
    timeout_seconds: int = 60
    max_request_attempts: int = 1

    def __post_init__(self) -> None:
        if (self.provider, self.endpoint_path, self.model, self.thinking_enabled, self.reasoning_effort, self.max_tokens, self.max_request_attempts) != (PROVIDER, ENDPOINT_PATH, MODEL, True, REASONING_EFFORT, MAX_TOKENS, 1):
            raise ValueError("PHASE2_DEEPSEEK_CONFIG_INVALID")
        if not isinstance(self.timeout_seconds, int) or isinstance(self.timeout_seconds, bool) or self.timeout_seconds <= 0:
            raise ValueError("PHASE2_DEEPSEEK_TIMEOUT_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "endpoint_path": self.endpoint_path, "model": self.model, "thinking_enabled": self.thinking_enabled, "reasoning_effort": self.reasoning_effort, "max_tokens": self.max_tokens, "timeout_seconds": self.timeout_seconds, "max_request_attempts": self.max_request_attempts}


@dataclass(frozen=True)
class Phase2DeepSeekUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (self.prompt_tokens, self.completion_tokens, self.total_tokens)):
            raise ValueError("PHASE2_DEEPSEEK_USAGE_INVALID")
    def to_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens}


@dataclass(frozen=True)
class Phase2DeepSeekRequest:
    prompt_sha256: str
    request_sha256: str
    _payload_json: str = field(repr=False)
    @property
    def payload(self) -> dict[str, Any]: return json.loads(self._payload_json)
    def __post_init__(self) -> None:
        if _SHA.fullmatch(self.prompt_sha256) is None or _SHA.fullmatch(self.request_sha256) is None or _sha(self._payload_json) != self.request_sha256:
            raise ValueError("PHASE2_DEEPSEEK_REQUEST_INVALID")
    def to_dict(self) -> dict[str, Any]:
        return {"provider": PROVIDER, "endpoint_path": ENDPOINT_PATH, "requested_model": MODEL, "thinking_enabled": True, "reasoning_effort": REASONING_EFFORT, "max_tokens": MAX_TOKENS, "prompt_sha256": self.prompt_sha256, "request_sha256": self.request_sha256, "request_payload": self.payload}


@dataclass(frozen=True)
class Phase2TransportAudit:
    request_count: int
    retry_count: int
    final_error_category: str | None = None

    def __post_init__(self) -> None:
        if self.request_count not in {1, 2} or self.retry_count not in {0, 1} or self.retry_count != self.request_count - 1:
            raise ValueError("PHASE2_TRANSPORT_AUDIT_INVALID")
        if self.final_error_category is not None and self.final_error_category not in TRANSPORT_ERROR_CATEGORIES:
            raise ValueError("PHASE2_TRANSPORT_AUDIT_CATEGORY_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {"request_count": self.request_count, "retry_count": self.retry_count, "final_error_category": self.final_error_category}


@dataclass(frozen=True)
class Phase2RetriedLocalizationResult:
    result: Phase2DeepSeekResult
    transport_audit: Phase2TransportAudit

    def to_dict(self) -> dict[str, Any]:
        return {"result": self.result.to_dict(), "transport_audit": self.transport_audit.to_dict()}


@dataclass(frozen=True)
class Phase2ShortlistDeepSeekResult:
    response_status: str
    prompt_sha256: str
    request_sha256: str
    reason_codes: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    transport_audit: Phase2TransportAudit

    def __post_init__(self) -> None:
        if self.response_status not in {"VALIDATED", "INVALID_RESPONSE", "PROVIDER_FAILED", "TRANSPORT_ERROR", "INCOMPLETE"}:
            raise ValueError("PHASE2_SHORTLIST_DEEPSEEK_STATUS_INVALID")
        if _SHA.fullmatch(self.prompt_sha256) is None or _SHA.fullmatch(self.request_sha256) is None:
            raise ValueError("PHASE2_SHORTLIST_DEEPSEEK_HASH_INVALID")
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))
        if (self.response_status == "VALIDATED") != bool(self.selected_candidate_ids):
            raise ValueError("PHASE2_SHORTLIST_DEEPSEEK_SELECTION_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_status": self.response_status,
            "prompt_sha256": self.prompt_sha256,
            "request_sha256": self.request_sha256,
            "reason_codes": list(self.reason_codes),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "transport_audit": self.transport_audit.to_dict(),
        }


@dataclass(frozen=True)
class Phase2DeepSeekResult:
    response_status: str
    prompt_sha256: str
    request_sha256: str
    provider_reported_model: str | None
    usage: Phase2DeepSeekUsage | None
    reason_codes: tuple[str, ...]
    _validated_response_json: str | None = field(default=None, repr=False)
    normalization_applied: bool = False
    normalization_actions: tuple[str, ...] = ()
    def __post_init__(self) -> None:
        if self.response_status not in RESULT_STATUSES or _SHA.fullmatch(self.prompt_sha256) is None or _SHA.fullmatch(self.request_sha256) is None:
            raise ValueError("PHASE2_DEEPSEEK_RESULT_INVALID")
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))
        object.__setattr__(self, "normalization_actions", tuple(sorted(set(self.normalization_actions))))
        if self.response_status in {"VALIDATED", "VALIDATED_WITH_NORMALIZATION"} and self._validated_response_json is None:
            raise ValueError("PHASE2_DEEPSEEK_VALIDATED_RESPONSE_MISSING")
        if self.response_status not in {"VALIDATED", "VALIDATED_WITH_NORMALIZATION"} and self._validated_response_json is not None:
            raise ValueError("PHASE2_DEEPSEEK_FAILED_RESPONSE_FORBIDDEN")
        if self.response_status == "VALIDATED" and (self.normalization_applied or self.normalization_actions):
            raise ValueError("PHASE2_DEEPSEEK_UNEXPECTED_NORMALIZATION")
        if self.response_status == "VALIDATED_WITH_NORMALIZATION" and (not self.normalization_applied or not self.normalization_actions):
            raise ValueError("PHASE2_DEEPSEEK_NORMALIZATION_MISSING")
    @property
    def validated_response(self) -> dict[str, Any] | None:
        return json.loads(self._validated_response_json) if self._validated_response_json is not None else None
    def to_dict(self) -> dict[str, Any]:
        return {"response_status": self.response_status, "prompt_sha256": self.prompt_sha256, "request_sha256": self.request_sha256, "provider_reported_model": self.provider_reported_model, "usage": self.usage.to_dict() if self.usage else None, "reason_codes": list(self.reason_codes), "normalization_applied": self.normalization_applied, "normalization_actions": list(self.normalization_actions), "validated_response": self.validated_response}


def build_phase2_deepseek_request(preparation: Phase2LocalizationPreparation, config: Phase2DeepSeekConfig) -> Phase2DeepSeekRequest:
    if not isinstance(preparation, Phase2LocalizationPreparation) or preparation.preparation_status != "READY_FOR_LLM" or preparation.prompt is None:
        raise ValueError("PHASE2_DEEPSEEK_PREPARATION_INELIGIBLE")
    if not isinstance(config, Phase2DeepSeekConfig):
        raise ValueError("PHASE2_DEEPSEEK_CONFIG_INVALID")
    payload = {"model": config.model, "messages": [{"role": "system", "content": preparation.prompt.system_message}, {"role": "user", "content": preparation.prompt.user_message}], "thinking": {"type": "enabled"}, "reasoning_effort": config.reasoning_effort, "response_format": {"type": "json_object"}, "max_tokens": config.max_tokens}
    payload_json = _canonical(payload)
    return Phase2DeepSeekRequest(preparation.prompt_sha256, _sha(payload_json), payload_json)


def build_phase2_shortlist_deepseek_request(prompt: Phase2ShortlistPrompt, config: Phase2DeepSeekConfig) -> Phase2DeepSeekRequest:
    if not isinstance(prompt, Phase2ShortlistPrompt) or not isinstance(config, Phase2DeepSeekConfig):
        raise ValueError("PHASE2_SHORTLIST_DEEPSEEK_INPUT_INVALID")
    payload = {"model": config.model, "messages": [{"role": "system", "content": prompt.system_message}, {"role": "user", "content": prompt.user_message}], "thinking": {"type": "enabled"}, "reasoning_effort": config.reasoning_effort, "response_format": {"type": "json_object"}, "max_tokens": config.max_tokens}
    payload_json = _canonical(payload)
    return Phase2DeepSeekRequest(prompt.prompt_sha256, _sha(payload_json), payload_json)


def _usage(value: Any) -> Phase2DeepSeekUsage | None:
    if not isinstance(value, Mapping): return None
    try: return Phase2DeepSeekUsage(value["prompt_tokens"], value["completion_tokens"], value["total_tokens"])
    except (KeyError, TypeError, ValueError): return None


def _failure(status: str, request: Phase2DeepSeekRequest, code: str, model: str | None = None, usage: Phase2DeepSeekUsage | None = None) -> Phase2DeepSeekResult:
    return Phase2DeepSeekResult(status, request.prompt_sha256, request.request_sha256, model, usage, (code,), None)


def parse_phase2_deepseek_response(provider_response: Mapping[str, Any], preparation: Phase2LocalizationPreparation, request: Phase2DeepSeekRequest) -> Phase2DeepSeekResult:
    if not isinstance(preparation, Phase2LocalizationPreparation) or preparation.prompt is None or request.prompt_sha256 != preparation.prompt_sha256:
        return _failure("INVALID_RESPONSE", request, "REQUEST_PREPARATION_MISMATCH")
    if not isinstance(provider_response, Mapping): return _failure("INVALID_RESPONSE", request, "PROVIDER_RESPONSE_INVALID")
    model = provider_response.get("model") if isinstance(provider_response.get("model"), str) else None
    if provider_response.get("error") is not None: return _failure("PROVIDER_FAILED", request, "PROVIDER_RESPONSE_FAILED", model)
    usage = _usage(provider_response.get("usage"))
    if model is None or usage is None: return _failure("INVALID_RESPONSE", request, "PROVIDER_METADATA_INVALID", model, usage)
    choices = provider_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping): return _failure("INVALID_RESPONSE", request, "PROVIDER_CHOICES_INVALID", model, usage)
    choice = choices[0]
    if choice.get("finish_reason") == "length": return _failure("INCOMPLETE", request, "PROVIDER_RESPONSE_INCOMPLETE", model, usage)
    if choice.get("finish_reason") != "stop": return _failure("PROVIDER_FAILED", request, "PROVIDER_FINISH_REASON_INVALID", model, usage)
    message = choice.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant": return _failure("INVALID_RESPONSE", request, "PROVIDER_MESSAGE_INVALID", model, usage)
    # Only content is read. reasoning_content is intentionally ignored and discarded.
    content = message.get("content")
    if not isinstance(content, str): return _failure("INVALID_RESPONSE", request, "PROVIDER_CONTENT_JSON_INVALID", model, usage)
    is_v4 = preparation.prompt.response_schema.get("title") == TOPK_V4_RESPONSE_SCHEMA_VERSION
    is_v3 = preparation.prompt.response_schema.get("title") == TOPK_V3_RESPONSE_SCHEMA_VERSION
    is_v2 = preparation.prompt.response_schema.get("title") == DECISION_V2_RESPONSE_SCHEMA_VERSION
    if is_v4:
        try: parsed = json.loads(content)
        except (json.JSONDecodeError, UnicodeError): return _failure("INVALID_RESPONSE", request, "PROVIDER_CONTENT_JSON_INVALID", model, usage)
        validation = validate_phase2_localization_topk_v4(parsed, preparation.prompt)
        if not validation.valid: return _failure("INVALID_RESPONSE", request, validation.reason_codes[0], model, usage)
        materialized = materialize_phase2_localization_result_topk_v4(validation, preparation.prompt)
        if validation.normalization_actions:
            return Phase2DeepSeekResult("VALIDATED_WITH_NORMALIZATION", request.prompt_sha256, request.request_sha256, model, usage, ("RESPONSE_VALIDATED_TOPK_V4", *validation.normalization_actions), _canonical(materialized), True, validation.normalization_actions)
        return Phase2DeepSeekResult("VALIDATED", request.prompt_sha256, request.request_sha256, model, usage, ("RESPONSE_VALIDATED_TOPK_V4",), _canonical(materialized))
    if is_v3:
        try: parsed = json.loads(content)
        except (json.JSONDecodeError, UnicodeError): return _failure("INVALID_RESPONSE", request, "PROVIDER_CONTENT_JSON_INVALID", model, usage)
        validation = validate_phase2_localization_topk_v3(parsed, preparation.prompt)
        if not validation.valid: return _failure("INVALID_RESPONSE", request, validation.reason_codes[0], model, usage)
        materialized = materialize_phase2_localization_result_topk_v3(validation, preparation.prompt)
        return Phase2DeepSeekResult("VALIDATED", request.prompt_sha256, request.request_sha256, model, usage, ("RESPONSE_VALIDATED_TOPK_V3",), _canonical(materialized))
    if is_v2:
        try: parsed = json.loads(content)
        except (json.JSONDecodeError, UnicodeError): return _failure("INVALID_RESPONSE", request, "PROVIDER_CONTENT_JSON_INVALID", model, usage)
        validation = validate_phase2_localization_decision_v2(parsed, preparation.prompt)
        if not validation.valid: return _failure("INVALID_RESPONSE", request, validation.reason_codes[0], model, usage)
        materialized = materialize_phase2_localization_result_v2(validation, preparation.prompt)
        return Phase2DeepSeekResult("VALIDATED", request.prompt_sha256, request.request_sha256, model, usage, ("RESPONSE_VALIDATED_V2",), _canonical(materialized))
    normalization = normalize_phase2_localization_response(content)
    if normalization.normalized_response is None:
        return _failure("INVALID_RESPONSE", request, "PROVIDER_CONTENT_JSON_INVALID", model, usage)
    validation = validate_phase2_localization_response(normalization.normalized_response, preparation.prompt)
    if not validation.valid: return _failure("INVALID_RESPONSE", request, validation.reason_codes[0], model, usage)
    status = "VALIDATED_WITH_NORMALIZATION" if normalization.applied else "VALIDATED"
    return Phase2DeepSeekResult(status, request.prompt_sha256, request.request_sha256, model, usage, ("RESPONSE_VALIDATED", *normalization.actions), _canonical(dict(validation.response)), normalization.applied, normalization.actions)


def execute_phase2_localization(preparation: Phase2LocalizationPreparation, config: Phase2DeepSeekConfig, transport: Phase2DeepSeekTransport, api_key: str) -> Phase2DeepSeekResult:
    request = build_phase2_deepseek_request(preparation, config)
    if not isinstance(api_key, str) or not api_key: raise ValueError("PHASE2_DEEPSEEK_API_KEY_REQUIRED")
    if transport is None or not callable(getattr(transport, "send", None)): raise ValueError("PHASE2_DEEPSEEK_TRANSPORT_INVALID")
    try: response = transport.send(request.payload, api_key, config.timeout_seconds)
    except Exception: return _failure("TRANSPORT_ERROR", request, "TRANSPORT_FAILURE")
    return parse_phase2_deepseek_response(response, preparation, request)


def classify_phase2_transport_error(error: BaseException) -> str:
    """Map a transport exception to a fixed category without retaining text."""
    category = getattr(error, "category", None)
    if category in TRANSPORT_ERROR_CATEGORIES:
        return category
    code = getattr(error, "code", None)
    if not isinstance(code, int) or isinstance(code, bool):
        code = getattr(error, "status_code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        if code == 429:
            return "HTTP_429"
        if 500 <= code <= 599:
            return "HTTP_5XX"
        if 400 <= code <= 499:
            return "HTTP_OTHER_4XX"
    if isinstance(error, (TimeoutError,)):
        return "TIMEOUT"
    if isinstance(error, (urllib_error.URLError, ConnectionError)):
        return "NETWORK_ERROR"
    if isinstance(error, urllib_error.HTTPError):
        return "OTHER_TRANSPORT_ERROR"
    return "OTHER_TRANSPORT_ERROR"


def _send_with_single_retry(
    request: Phase2DeepSeekRequest,
    config: Phase2DeepSeekConfig,
    transport_factory: Callable[[], Phase2DeepSeekTransport],
    api_key: str,
) -> tuple[Mapping[str, Any] | None, Phase2TransportAudit]:
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("PHASE2_DEEPSEEK_API_KEY_REQUIRED")
    if not callable(transport_factory):
        raise ValueError("PHASE2_DEEPSEEK_TRANSPORT_FACTORY_INVALID")
    for attempt in (1, 2):
        transport = transport_factory()
        if transport is None or not callable(getattr(transport, "send", None)):
            raise ValueError("PHASE2_DEEPSEEK_TRANSPORT_INVALID")
        try:
            response = transport.send(request.payload, api_key, config.timeout_seconds)
            return response, Phase2TransportAudit(attempt, attempt - 1)
        except Exception as error:
            category = classify_phase2_transport_error(error)
            if attempt == 1 and category in RETRYABLE_TRANSPORT_ERROR_CATEGORIES:
                continue
            return None, Phase2TransportAudit(attempt, attempt - 1, category)
    raise AssertionError("unreachable")


def execute_phase2_localization_with_single_retry(
    preparation: Phase2LocalizationPreparation,
    config: Phase2DeepSeekConfig,
    transport_factory: Callable[[], Phase2DeepSeekTransport],
    api_key: str,
) -> Phase2RetriedLocalizationResult:
    """Explicit v5 transport policy; legacy one-shot entry points remain unchanged."""
    request = build_phase2_deepseek_request(preparation, config)
    response, audit = _send_with_single_retry(request, config, transport_factory, api_key)
    if response is None:
        result = _failure("TRANSPORT_ERROR", request, audit.final_error_category or "OTHER_TRANSPORT_ERROR")
    else:
        result = parse_phase2_deepseek_response(response, preparation, request)
    return Phase2RetriedLocalizationResult(result, audit)


def _shortlist_provider_validation(
    provider_response: Mapping[str, Any],
    prompt: Phase2ShortlistPrompt,
) -> tuple[str, tuple[str, ...], Phase2ShortlistValidation | None]:
    if not isinstance(provider_response, Mapping):
        return "INVALID_RESPONSE", ("PROVIDER_RESPONSE_INVALID",), None
    if provider_response.get("error") is not None:
        return "PROVIDER_FAILED", ("PROVIDER_RESPONSE_FAILED",), None
    choices = provider_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return "INVALID_RESPONSE", ("PROVIDER_CHOICES_INVALID",), None
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        return "INCOMPLETE", ("PROVIDER_RESPONSE_INCOMPLETE",), None
    if choice.get("finish_reason") != "stop":
        return "PROVIDER_FAILED", ("PROVIDER_FINISH_REASON_INVALID",), None
    message = choice.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant" or not isinstance(message.get("content"), str):
        return "INVALID_RESPONSE", ("PROVIDER_MESSAGE_INVALID",), None
    validation = validate_phase2_candidate_shortlist_response(message["content"], prompt)
    if not validation.valid:
        return "INVALID_RESPONSE", validation.reason_codes, None
    return "VALIDATED", ("COARSE_SHORTLIST_RESPONSE_VALIDATED",), validation


def execute_phase2_candidate_shortlist_with_single_retry(
    prompt: Phase2ShortlistPrompt,
    config: Phase2DeepSeekConfig,
    transport_factory: Callable[[], Phase2DeepSeekTransport],
    api_key: str,
) -> Phase2ShortlistDeepSeekResult:
    request = build_phase2_shortlist_deepseek_request(prompt, config)
    response, audit = _send_with_single_retry(request, config, transport_factory, api_key)
    if response is None:
        return Phase2ShortlistDeepSeekResult("TRANSPORT_ERROR", request.prompt_sha256, request.request_sha256, (audit.final_error_category or "OTHER_TRANSPORT_ERROR",), (), audit)
    status, codes, validation = _shortlist_provider_validation(response, prompt)
    selected = validation.candidate_ids if validation is not None else ()
    return Phase2ShortlistDeepSeekResult(status, request.prompt_sha256, request.request_sha256, codes, selected, audit)


def serialize_phase2_deepseek_request(request: Phase2DeepSeekRequest) -> str:
    return json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def serialize_phase2_deepseek_result(result: Phase2DeepSeekResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

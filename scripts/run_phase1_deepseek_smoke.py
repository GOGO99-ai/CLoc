"""One-shot live DeepSeek compatibility smoke using synthetic evidence only.

Importing this module performs no environment read or network access.  The API
key is read only by ``main`` and is passed directly to the one-shot transport.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dry_run_phase1_normality import build_synthetic_phase1_preparation
from src.cloc_v2.phase1_normality_deepseek import (
    Phase1DeepSeekConfig,
    Phase1DeepSeekResult,
    Phase1DeepSeekTransportError,
    build_phase1_deepseek_request,
    execute_phase1_deepseek_review,
)


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
SCHEMA_VERSION = "phase1_deepseek_smoke_v1"
SMOKE_KIND = "SYNTHETIC_DEEPSEEK_API_COMPATIBILITY_CHECK"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "prototypes"
    / "phase1_deepseek_smoke"
    / "phase1_deepseek_smoke.json"
)
REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "smoke_kind",
        "scientific_experiment_result",
        "synthetic_input",
        "requested_model",
        "provider_reported_model",
        "response_status",
        "usage",
        "validated_response",
        "routing_decision",
        "fallback_action",
        "reason_codes",
        "prompt_sha256",
        "request_sha256",
        "request_attempt_count",
        "api_request_count",
        "rpc_request_count",
        "sanitization_check",
    }
)

_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_TRANSACTION_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_FORBIDDEN_KEYS = frozenset(
    {
        "authorization",
        "reasoning_content",
        "system_message",
        "user_message",
        "messages",
        "payload",
        "request_payload",
        "headers",
        "calldata",
        "input",
        "output",
        "error",
        "revertreason",
        "record_id",
        "transaction_hash",
    }
)


class _NoRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Phase1DeepSeekTransportError()


class DeepSeekHTTPSOneShotTransport:
    """Strict HTTPS POST transport whose instance can be used only once."""

    def __init__(self, url: str = DEEPSEEK_URL) -> None:
        if url != DEEPSEEK_URL or not url.startswith("https://"):
            raise ValueError("DEEPSEEK_SMOKE_URL_INVALID")
        self._url = url
        self.request_attempt_count = 0
        self.api_request_count = 0

    def send(
        self,
        request_payload: Mapping[str, Any],
        api_key: str,
        timeout_seconds: int,
    ) -> Mapping[str, Any]:
        if self.request_attempt_count != 0:
            raise Phase1DeepSeekTransportError()
        self.request_attempt_count += 1
        if not isinstance(api_key, str) or not api_key or not api_key.strip():
            raise Phase1DeepSeekTransportError()
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise Phase1DeepSeekTransportError()
        try:
            body = json.dumps(
                dict(request_payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            request = urllib_request.Request(
                self._url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            opener = urllib_request.build_opener(
                _NoRedirectHandler(),
                urllib_request.HTTPSHandler(context=ssl.create_default_context()),
            )
            self.api_request_count += 1
            with opener.open(request, timeout=timeout_seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise Phase1DeepSeekTransportError()
            decoded = raw.decode("utf-8", errors="strict")
            parsed = json.loads(decoded)
            if not isinstance(parsed, Mapping):
                raise Phase1DeepSeekTransportError()
            return dict(parsed)
        except Phase1DeepSeekTransportError:
            raise
        except (
            urllib_error.HTTPError,
            urllib_error.URLError,
            TimeoutError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            raise Phase1DeepSeekTransportError() from None


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key).lower() for key in value}
        for item in value.values():
            found.update(_all_keys(item))
        return found
    if isinstance(value, (list, tuple)):
        found: set[str] = set()
        for item in value:
            found.update(_all_keys(item))
        return found
    return set()


def _scan_report(
    report: Mapping[str, Any],
    *,
    api_key: str | None = None,
    forbidden_internal_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    text = json.dumps(report, ensure_ascii=False, sort_keys=True)
    keys = _all_keys(report)
    checks = {
        "credential_value_absent": api_key is None or api_key not in text,
        "credential_material_absent": "authorization" not in text.lower() and "bearer " not in text.lower(),
        "hidden_reasoning_absent": "reasoning_content" not in text.lower(),
        "request_material_absent": not bool(keys & _FORBIDDEN_KEYS),
        "raw_20_byte_address_absent": _RAW_ADDRESS.search(text) is None,
        "raw_32_byte_transaction_hash_absent": _RAW_TRANSACTION_HASH.search(text) is None,
        "internal_identity_absent": all(value not in text for value in forbidden_internal_values),
        "request_attempt_count_bounded": isinstance(report.get("request_attempt_count"), int)
        and 0 <= report["request_attempt_count"] <= 1,
        "api_request_count_bounded": isinstance(report.get("api_request_count"), int)
        and 0 <= report["api_request_count"] <= 1,
        "rpc_request_count_zero": report.get("rpc_request_count") == 0,
        "scientific_experiment_result_false": report.get("scientific_experiment_result") is False,
        "synthetic_input_true": report.get("synthetic_input") is True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def build_phase1_deepseek_smoke_report(
    result: Phase1DeepSeekResult,
    *,
    request_attempt_count: int,
    api_request_count: int,
    api_key: str | None = None,
    forbidden_internal_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the fixed, sanitized smoke report from a completed client result."""
    if not isinstance(result, Phase1DeepSeekResult):
        raise ValueError("DEEPSEEK_SMOKE_RESULT_INVALID")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "smoke_kind": SMOKE_KIND,
        "scientific_experiment_result": False,
        "synthetic_input": True,
        "requested_model": result.requested_model,
        "provider_reported_model": result.provider_reported_model,
        "response_status": result.response_status,
        "usage": result.usage.to_dict() if result.usage is not None else None,
        "validated_response": result.validated_response,
        "routing_decision": result.routing_decision.to_dict() if result.routing_decision is not None else None,
        "fallback_action": result.fallback_action,
        "reason_codes": list(result.reason_codes),
        "prompt_sha256": result.prompt_sha256,
        "request_sha256": result.request_sha256,
        "request_attempt_count": request_attempt_count,
        "api_request_count": api_request_count,
        "rpc_request_count": 0,
    }
    report["sanitization_check"] = _scan_report(
        report,
        api_key=api_key,
        forbidden_internal_values=forbidden_internal_values,
    )
    if not report["sanitization_check"]["passed"]:
        raise ValueError("DEEPSEEK_SMOKE_SANITIZATION_FAILED")
    return report


def serialize_phase1_deepseek_smoke_report(report: Mapping[str, Any]) -> str:
    if not isinstance(report, Mapping) or set(report) != REPORT_FIELDS:
        raise ValueError("DEEPSEEK_SMOKE_REPORT_FIELDS_INVALID")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("smoke_kind") != SMOKE_KIND:
        raise ValueError("DEEPSEEK_SMOKE_REPORT_VERSION_INVALID")
    fresh = _scan_report({key: value for key, value in report.items() if key != "sanitization_check"})
    if not fresh["passed"] or report.get("sanitization_check") != fresh:
        raise ValueError("DEEPSEEK_SMOKE_SANITIZATION_FAILED")
    return json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_phase1_deepseek_smoke_report(
    report: Mapping[str, Any],
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_PATH,
) -> Path:
    serialized = serialize_phase1_deepseek_smoke_report(report)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def main() -> int:
    try:
        preparation = build_synthetic_phase1_preparation()
        if (
            preparation.preparation_status != "READY_FOR_LLM"
            or preparation.eligibility.eligibility != "ELIGIBLE"
            or preparation.eligibility.llm_allowed is not True
            or preparation.prompt is None
            or preparation.gate_evidence is None
        ):
            print("DEEPSEEK_SMOKE_PREPARATION_FAILED")
            return 1
    except (TypeError, ValueError):
        print("DEEPSEEK_SMOKE_PREPARATION_FAILED")
        return 1

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not isinstance(api_key, str) or not api_key or not api_key.strip():
        print("DEEPSEEK_API_KEY_MISSING")
        return 1

    print("即将发送1次脱敏合成证据请求，可能产生少量费用")
    config = Phase1DeepSeekConfig()
    transport = DeepSeekHTTPSOneShotTransport()
    try:
        result = execute_phase1_deepseek_review(preparation, config, transport, api_key)
        report = build_phase1_deepseek_smoke_report(
            result,
            request_attempt_count=transport.request_attempt_count,
            api_request_count=transport.api_request_count,
            api_key=api_key,
        )
        destination = write_phase1_deepseek_smoke_report(report)
    except (OSError, TypeError, ValueError):
        print("DEEPSEEK_SMOKE_LOCAL_FAILURE")
        return 1

    print(f"response_status={result.response_status}")
    print(f"report_path={destination.relative_to(PROJECT_ROOT)}")
    print(f"request_attempt_count={transport.request_attempt_count}")
    print(f"api_request_count={transport.api_request_count}")
    return 0 if result.response_status == "VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEEPSEEK_URL",
    "DEFAULT_OUTPUT_PATH",
    "DeepSeekHTTPSOneShotTransport",
    "MAX_RESPONSE_BYTES",
    "build_phase1_deepseek_smoke_report",
    "main",
    "serialize_phase1_deepseek_smoke_report",
    "write_phase1_deepseek_smoke_report",
]

"""Two-request synthetic API contract smoke for source tournament Top-K v6.

Importing this module performs no environment read, file write, or network
request.  ``main --live`` is the only entry point that reads the API key and
uses the one-shot HTTPS transport.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_localization15_topk_v6_tournament import (
    build_synthetic_candidates,
    build_synthetic_context,
    run_synthetic_offline_readiness,
)
from scripts.run_phase1_deepseek_smoke import DeepSeekHTTPSOneShotTransport
from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekConfig
from src.cloc_v2.phase2_source_tournament import (
    READY,
    TournamentFinalist,
    TournamentGroup,
    TournamentPrompt,
    _provider_content,
    _request_payload,
    build_tournament_final_prompt,
    build_tournament_group_prompt,
    validate_tournament_response,
)


EXPERIMENT_TYPE = "SYNTHETIC_TOURNAMENT_V6_API_CONTRACT_SMOKE"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "phase2_tournament_v6_synthetic_smoke_20260901"
REPORT_FILENAME = "phase2_tournament_v6_synthetic_smoke_report.json"
_RAW_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_RAW_HASH = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_FORBIDDEN_KEYS = frozenset({
    "api_key", "authorization", "endpoint", "url", "messages", "request_payload",
    "provider_response", "reasoning_content", "source_text", "prompt", "location", "cause",
})


def _safe_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(not isinstance(value.get(key), int) or isinstance(value.get(key), bool) or value[key] < 0 for key in keys):
        return None
    return {key: int(value[key]) for key in keys}


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key).lower() for key in value}
        for child in value.values():
            found.update(_all_keys(child))
        return found
    if isinstance(value, (list, tuple)):
        found: set[str] = set()
        for child in value:
            found.update(_all_keys(child))
        return found
    return set()


def validate_public_report(report: Mapping[str, Any]) -> None:
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if _RAW_ADDRESS.search(rendered) or _RAW_HASH.search(rendered):
        raise ValueError("TOURNAMENT_V6_SMOKE_SENSITIVE_REPORT")
    if _FORBIDDEN_KEYS.intersection(_all_keys(report)):
        raise ValueError("TOURNAMENT_V6_SMOKE_SENSITIVE_REPORT")
    lowered = rendered.lower()
    if "http://" in lowered or "https://" in lowered or "function synthetic_" in lowered:
        raise ValueError("TOURNAMENT_V6_SMOKE_SENSITIVE_REPORT")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    validate_public_report(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class OneShotResult:
    status: str
    output_count: int
    request_count: int
    retry_count: int
    reason_codes: tuple[str, ...]
    usage: Mapping[str, int] | None
    validation: Any = None

    def public(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_count": self.output_count,
            "request_count": self.request_count,
            "retry_count": self.retry_count,
            "reason_codes": list(self.reason_codes),
            "usage": dict(self.usage) if self.usage is not None else None,
        }


def execute_one_shot(
    prompt: TournamentPrompt,
    api_key: str,
    transport: Any,
) -> OneShotResult:
    """Send exactly once; no retry path exists in this smoke adapter."""
    if not isinstance(prompt, TournamentPrompt) or not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("TOURNAMENT_V6_SMOKE_INPUT_INVALID")
    payload, _ = _request_payload(prompt, Phase2DeepSeekConfig())
    try:
        response = transport.send(payload, api_key, Phase2DeepSeekConfig().timeout_seconds)
    except Exception:
        return OneShotResult("TRANSPORT_ERROR", 0, 1, 0, ("TOURNAMENT_V6_SMOKE_TRANSPORT_ERROR",), None)
    usage = _safe_usage(response.get("usage") if isinstance(response, Mapping) else None)
    content, provider_code = _provider_content(response if isinstance(response, Mapping) else {})
    if content is None:
        return OneShotResult("INVALID_RESPONSE", 0, 1, 0, (provider_code,), usage)
    validation = validate_tournament_response(content, prompt)
    if not validation.valid:
        return OneShotResult("INVALID_RESPONSE", 0, 1, 0, validation.reason_codes, usage)
    return OneShotResult("VALIDATED", len(validation.findings), 1, 0, ("TOURNAMENT_V6_SMOKE_RESPONSE_VALIDATED",), usage, validation)


def _build_finalists(group_validation: Any) -> tuple[TournamentFinalist, ...]:
    candidates = build_synthetic_candidates(40)
    by_slice = {item.source_slice_id: item for item in candidates}
    finalists: list[TournamentFinalist] = []
    for finding in group_validation.findings:
        finalists.append(TournamentFinalist("G1", finding.rank, by_slice[finding.source_slice_id]))
    for group_index in range(2, 6):
        start = 20 + (group_index - 2) * 5 + 1
        for group_rank, candidate_index in enumerate(range(start, start + 5), start=1):
            finalists.append(TournamentFinalist(f"G{group_index}", group_rank, candidates[candidate_index - 1]))
    if len(finalists) != 25 or len({item.candidate.candidate_id for item in finalists}) != 25:
        raise ValueError("TOURNAMENT_V6_SMOKE_FINALISTS_INVALID")
    return tuple(finalists)


def run_synthetic_api_smoke(
    api_key: str,
    transport_factory: Callable[[], Any] = DeepSeekHTTPSOneShotTransport,
) -> dict[str, Any]:
    readiness = run_synthetic_offline_readiness(100)["readiness"]
    if readiness["readiness_status"] != READY:
        raise ValueError("TOURNAMENT_V6_OFFLINE_READINESS_REQUIRED")
    context = build_synthetic_context()
    group = TournamentGroup("G1", build_synthetic_candidates(20))
    group_built = build_tournament_group_prompt(group, context)
    if group_built.status != "READY_FOR_LLM" or group_built.prompt is None:
        raise ValueError("TOURNAMENT_V6_GROUP_PROMPT_NOT_READY")
    group_result = execute_one_shot(group_built.prompt, api_key, transport_factory())
    base: dict[str, Any] = {
        "schema_version": "phase2_tournament_v6_synthetic_smoke_report_v1",
        "experiment_type": EXPERIMENT_TYPE,
        "scientific_experiment_result": False,
        "group_input_count": 20,
        "group_output_count": group_result.output_count,
        "group_request_status": group_result.status,
        "group_prompt_bytes": group_built.prompt.prompt_bytes,
        "group_usage": dict(group_result.usage) if group_result.usage is not None else None,
        "final_input_count": 0,
        "final_output_count": 0,
        "final_request_status": "NOT_ATTEMPTED",
        "final_prompt_bytes": 0,
        "final_usage": None,
        "deepseek_request_count": group_result.request_count,
        "retry_count": 0,
        "rpc_request_count": 0,
        "source_network_request_count": 0,
        "stage_a_v5_request_count": 0,
        "provider_free_text_used": False,
        "reason_codes": list(group_result.reason_codes),
        "gate2_success": False,
        "sanitization_check": "PASSED",
    }
    if group_result.status != "VALIDATED" or group_result.output_count != 5 or group_result.validation is None:
        validate_public_report(base)
        return base
    finalists = _build_finalists(group_result.validation)
    final_built = build_tournament_final_prompt(finalists, context)
    if final_built.status != "READY_FOR_LLM" or final_built.prompt is None:
        base["reason_codes"] = sorted(set(base["reason_codes"] + list(final_built.reason_codes)))
        base["final_request_status"] = "PROMPT_NOT_READY"
        validate_public_report(base)
        return base
    final_result = execute_one_shot(final_built.prompt, api_key, transport_factory())
    base.update(
        final_input_count=len(final_built.retained_candidates),
        final_output_count=final_result.output_count,
        final_request_status=final_result.status,
        final_prompt_bytes=final_built.prompt.prompt_bytes,
        final_usage=dict(final_result.usage) if final_result.usage is not None else None,
        deepseek_request_count=group_result.request_count + final_result.request_count,
        reason_codes=sorted(set(group_result.reason_codes + final_result.reason_codes)),
    )
    base["gate2_success"] = (
        group_result.status == "VALIDATED"
        and group_result.output_count == 5
        and final_result.status == "VALIDATED"
        and final_result.output_count == 5
        and base["deepseek_request_count"] == 2
        and base["retry_count"] == 0
        and not base["provider_free_text_used"]
    )
    validate_public_report(base)
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two-request synthetic DeepSeek smoke for tournament v6")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if not args.live:
        print("LIVE_MODE_REQUIRED")
        return 2
    if args.output_dir.exists():
        print("OUTPUT_DIR_ALREADY_EXISTS")
        return 2
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        print("DEEPSEEK_API_KEY_MISSING")
        return 2
    report = run_synthetic_api_smoke(api_key)
    _atomic_json(args.output_dir / REPORT_FILENAME, report)
    print(f"group_request_status={report['group_request_status']}")
    print(f"final_request_status={report['final_request_status']}")
    print(f"deepseek_request_count={report['deepseek_request_count']}")
    print(f"gate2_success={str(report['gate2_success']).lower()}")
    return 0 if report["gate2_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


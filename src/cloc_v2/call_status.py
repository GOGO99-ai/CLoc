"""Pure, conservative interpretation of one raw ``callTracer`` frame."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


_HEX_BYTES = re.compile(r"^0x(?:[0-9a-fA-F]{2})*$")


@dataclass(frozen=True)
class CallStatusResolution:
    """Sanitized, serializable status evidence for one call frame."""

    status: str
    decision_source: str
    output_present: bool
    output_valid_hex: bool
    error_failure_evidence: bool
    revert_reason_failure_evidence: bool
    usable_as_success_evidence: bool
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision_source": self.decision_source,
            "output_present": self.output_present,
            "output_valid_hex": self.output_valid_hex,
            "error_failure_evidence": self.error_failure_evidence,
            "revert_reason_failure_evidence": self.revert_reason_failure_evidence,
            "usable_as_success_evidence": self.usable_as_success_evidence,
            "uncertainties": list(self.uncertainties),
        }


def resolve_call_status(frame: Any) -> CallStatusResolution:
    """Classify one in-memory callTracer frame without reading external state."""
    if not isinstance(frame, Mapping):
        return _result(
            status="UNKNOWN",
            decision_source="UNRESOLVED",
            output_present=False,
            output_valid_hex=False,
            error_failure_evidence=False,
            revert_reason_failure_evidence=False,
            uncertainties=("CALLTRACER_FRAME_INVALID",),
        )

    output_present = "output" in frame
    output = frame.get("output")
    output_valid_hex = isinstance(output, str) and _HEX_BYTES.fullmatch(output) is not None

    error_evidence, error_type_invalid = _failure_evidence(frame.get("error"), "error" in frame)
    revert_evidence, revert_type_invalid = _failure_evidence(
        frame.get("revertReason"),
        "revertReason" in frame,
    )
    if error_evidence or revert_evidence:
        source = "CALLTRACER_ERROR" if error_evidence else "CALLTRACER_REVERT_REASON"
        return _result(
            status="FAILED",
            decision_source=source,
            output_present=output_present,
            output_valid_hex=output_valid_hex,
            error_failure_evidence=error_evidence,
            revert_reason_failure_evidence=revert_evidence,
            uncertainties=(),
        )

    reasons: list[str] = []
    if error_type_invalid:
        reasons.append("CALLTRACER_ERROR_FIELD_INVALID")
    if revert_type_invalid:
        reasons.append("CALLTRACER_REVERT_REASON_FIELD_INVALID")
    if not output_present:
        reasons.append("CALLTRACER_OUTPUT_MISSING")
    elif output is None:
        reasons.append("CALLTRACER_OUTPUT_NULL")
    elif output == "":
        reasons.append("CALLTRACER_OUTPUT_EMPTY_STRING")
    elif not isinstance(output, str):
        reasons.append("CALLTRACER_OUTPUT_INVALID_TYPE")
    elif not output_valid_hex:
        reasons.append("CALLTRACER_OUTPUT_INVALID_HEX")

    if not reasons:
        return _result(
            status="SUCCESS",
            decision_source="CALLTRACER_OUTPUT",
            output_present=True,
            output_valid_hex=True,
            error_failure_evidence=False,
            revert_reason_failure_evidence=False,
            uncertainties=(),
        )
    return _result(
        status="UNKNOWN",
        decision_source="UNRESOLVED",
        output_present=output_present,
        output_valid_hex=output_valid_hex,
        error_failure_evidence=False,
        revert_reason_failure_evidence=False,
        uncertainties=tuple(sorted(set(reasons))),
    )


def serialize_call_status(result: CallStatusResolution) -> str:
    """Return stable JSON without writing an artifact."""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def _failure_evidence(value: Any, field_present: bool) -> tuple[bool, bool]:
    if not field_present or value is None:
        return False, False
    if isinstance(value, str):
        return bool(value.strip()), False
    return False, True


def _result(
    *,
    status: str,
    decision_source: str,
    output_present: bool,
    output_valid_hex: bool,
    error_failure_evidence: bool,
    revert_reason_failure_evidence: bool,
    uncertainties: tuple[str, ...],
) -> CallStatusResolution:
    return CallStatusResolution(
        status=status,
        decision_source=decision_source,
        output_present=output_present,
        output_valid_hex=output_valid_hex,
        error_failure_evidence=error_failure_evidence,
        revert_reason_failure_evidence=revert_reason_failure_evidence,
        usable_as_success_evidence=status == "SUCCESS",
        uncertainties=uncertainties,
    )


__all__ = [
    "CallStatusResolution",
    "resolve_call_status",
    "serialize_call_status",
]

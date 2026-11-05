"""Adapt callTracer frames to the normalized call-tree input."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from cloc.evidence.call_status import resolve_call_status


_HEX_BYTES = re.compile(r"^0x(?:[0-9a-fA-F]{2})*$")


def adapt_calltracer_call_tree(root_frame: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one raw callTracer root frame without retaining raw status text."""
    if not isinstance(root_frame, Mapping):
        raise ValueError("callTracer root frame must be an object")
    return _adapt_frame(root_frame, path="0")


def serialize_calltracer_call_tree(call_tree: Mapping[str, Any]) -> str:
    """Return stable JSON for an already adapted tree without writing a file."""
    return json.dumps(call_tree, ensure_ascii=False, indent=2) + "\n"


def _adapt_frame(frame: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    children = frame.get("calls", [])
    if not isinstance(children, list):
        raise ValueError(f"callTracer calls at {path} must be a list")
    if any(not isinstance(child, Mapping) for child in children):
        raise ValueError(f"callTracer child at {path} must be an object")

    raw_input = frame.get("input")
    selector = _selector_from_input(raw_input)
    node = {
        "type": frame.get("type"),
        "from": frame.get("from"),
        "address": frame.get("to"),
        "function": selector,
        "params": raw_input,
        "call_type": frame.get("type"),
        "value": frame.get("value"),
        "gas": frame.get("gas"),
        "gas_used": frame.get("gasUsed"),
        "call_status": resolve_call_status(frame).to_dict(),
        "children": [
            _adapt_frame(child, path=f"{path}/{index}")
            for index, child in enumerate(children)
        ],
    }
    return node


def _selector_from_input(value: Any) -> str | None:
    if not isinstance(value, str) or _HEX_BYTES.fullmatch(value) is None or len(value) < 10:
        return None
    return value[:10].lower()


__all__ = [
    "adapt_calltracer_call_tree",
    "serialize_calltracer_call_tree",
]

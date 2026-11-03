"""Small conservative Solidity function-range parser used by the baseline package."""

from __future__ import annotations

import re
from typing import Any


def _mask_comments_and_strings(text: str) -> str:
    result = list(text)
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for position in range(index, end):
                result[position] = " "
            index = end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = len(text) - 2 if end < 0 else end
            for position in range(index, min(end + 2, len(text))):
                if result[position] != "\n":
                    result[position] = " "
            index = end + 2
        elif text[index] in "\"'":
            quote = text[index]
            result[index] = " "
            index += 1
            while index < len(text):
                current = text[index]
                if current != "\n":
                    result[index] = " "
                if current == "\\":
                    index += 1
                    if index < len(text) and text[index] != "\n":
                        result[index] = " "
                elif current == quote:
                    index += 1
                    break
                index += 1
        else:
            index += 1
    return "".join(result)


def _brace_pairs(masked: str) -> tuple[dict[int, int], list[int]]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    depths = [0] * len(masked)
    depth = 0
    for index, char in enumerate(masked):
        depths[index] = depth
        if char == "{":
            stack.append(index)
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0 or not stack:
                raise ValueError("unbalanced closing brace")
            pairs[stack.pop()] = index
    if stack:
        raise ValueError("unbalanced opening brace")
    return pairs, depths


def _find_header_end(masked: str, start: int) -> tuple[int, str] | None:
    parens = brackets = 0
    for index in range(start, len(masked)):
        char = masked[index]
        if char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif char in "{;" and parens == 0 and brackets == 0:
            return index, char
    return None


def _split_params(text: str) -> list[str]:
    parts: list[str] = []
    start = depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _canonical_type(parameter: str) -> str | None:
    cleaned = " ".join(parameter.replace("indexed", " ").split())
    if "mapping" in cleaned or "function" in cleaned or cleaned.startswith("("):
        return None
    tokens = [
        value
        for value in cleaned.replace("payable", " ")
        .replace("memory", " ")
        .replace("calldata", " ")
        .replace("storage", " ")
        .split()
        if value
    ]
    if not tokens:
        return None
    value = tokens[0]
    return "uint256" if value == "uint" else "int256" if value == "int" else value


def _signature(kind: str, name: str | None, header: str) -> str | None:
    if kind != "FUNCTION" or not name:
        return None
    begin = header.find("(")
    end = header.rfind(")")
    if begin < 0 or end < begin:
        return None
    types = [_canonical_type(part) for part in _split_params(header[begin + 1 : end])]
    return None if any(value is None for value in types) else f"{name}({','.join(types)})"


def parse_solidity_source(text: str) -> dict[str, Any]:
    """Return lexical function ranges without inventing ranges on parse failure."""
    try:
        masked = _mask_comments_and_strings(text)
        pairs, depths = _brace_pairs(masked)
    except ValueError as exc:
        return {"parse_status": "PARSE_FAILED", "uncertainties": [str(exc)], "functions": []}
    contracts: list[dict[str, Any]] = []
    pattern = r"\b(?:(abstract)\s+)?(contract|library|interface)\s+([A-Za-z_$][\w$]*)[^;{]*\{"
    for match in re.finditer(pattern, masked):
        opening = match.end() - 1
        if opening in pairs:
            contracts.append(
                {
                    "contract_name": match.group(3),
                    "opening": opening,
                    "closing": pairs[opening],
                    "depth": depths[opening],
                }
            )
    functions: list[dict[str, Any]] = []
    kinds = {
        "function": "FUNCTION",
        "constructor": "CONSTRUCTOR",
        "fallback": "FALLBACK",
        "receive": "RECEIVE",
        "modifier": "MODIFIER",
    }
    for match in re.finditer(r"\b(function|constructor|fallback|receive|modifier)\b", masked):
        start = match.start()
        owner = next(
            (
                item
                for item in reversed(contracts)
                if item["opening"] < start < item["closing"] and depths[start] == item["depth"] + 1
            ),
            None,
        )
        if owner is None:
            continue
        end = _find_header_end(masked, match.end())
        if end is None:
            continue
        terminator, terminal = end
        if terminal == "{" and terminator not in pairs:
            continue
        kind = kinds[match.group(1)]
        name_match = (
            re.match(r"\s*([A-Za-z_$][\w$]*)", masked[match.end() : terminator])
            if kind in {"FUNCTION", "MODIFIER"}
            else None
        )
        name = name_match.group(1) if name_match else (
            "constructor" if kind == "CONSTRUCTOR" else kind.lower()
        )
        header = text[start:terminator]
        functions.append(
            {
                "contract_name": owner["contract_name"],
                "function_kind": kind,
                "function_name": name,
                "canonical_signature": _signature(kind, name, header),
                "_start": start,
                "_end": pairs[terminator] if terminal == "{" else terminator,
            }
        )
    return {"parse_status": "PARSED", "uncertainties": [], "functions": functions}


_INTERNAL_CALL = re.compile(r"(?<![.$A-Za-z0-9_])([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
_NON_CALL_KEYWORDS = frozenset(
    {
        "assert", "catch", "delete", "emit", "for", "if", "keccak256", "new",
        "require", "return", "revert", "selfdestruct", "super", "while",
    }
)


def internal_call_names(source_code: str) -> tuple[str, ...]:
    """Return conservative bare call names, excluding member calls and keywords."""
    masked = _mask_comments_and_strings(source_code)
    return tuple(
        dict.fromkeys(
            match.group(1)
            for match in _INTERNAL_CALL.finditer(masked)
            if match.group(1) not in _NON_CALL_KEYWORDS
        )
    )

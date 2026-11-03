"""Offline Solidity source indexing with conservative lexical parsing."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .source_cache import atomic_json, norm_address, sha

_MASK = (1 << 64) - 1
_RC = [0x0000000000000001,0x0000000000008082,0x800000000000808A,0x8000000080008000,0x000000000000808B,0x0000000080000001,0x8000000080008081,0x8000000000008009,0x000000000000008A,0x0000000000000088,0x0000000080008009,0x000000008000000A,0x000000008000808B,0x800000000000008B,0x8000000000008089,0x8000000000008003,0x8000000000008002,0x8000000000000080,0x000000000000800A,0x800000008000000A,0x8000000080008081,0x8000000000008080,0x0000000080000001,0x8000000080008008]
_ROT = [[0,36,3,41,18],[1,44,10,45,2],[62,6,43,15,61],[28,55,25,21,56],[27,20,39,8,14]]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _rol(value: int, amount: int) -> int:
    return ((value << amount) | (value >> (64 - amount))) & _MASK if amount else value


def ethereum_keccak256(data: bytes) -> bytes:
    """Keccak-256 with Ethereum padding (0x01), deliberately not NIST SHA3-256."""
    state = [0] * 25
    rate = 136
    padded = bytearray(data)
    padded.append(0x01)
    padded.extend(b"\0" * ((rate - len(padded) % rate - 1) % rate))
    padded.append(0x80)
    for offset in range(0, len(padded), rate):
        block = padded[offset:offset + rate]
        for lane in range(rate // 8):
            state[lane] ^= int.from_bytes(block[lane * 8:lane * 8 + 8], "little")
        for rc in _RC:
            columns = [state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20] for x in range(5)]
            deltas = [columns[(x - 1) % 5] ^ _rol(columns[(x + 1) % 5], 1) for x in range(5)]
            for x in range(5):
                for y in range(5):
                    state[x + 5 * y] ^= deltas[x]
            rotated = [0] * 25
            for x in range(5):
                for y in range(5):
                    rotated[y + 5 * ((2 * x + 3 * y) % 5)] = _rol(state[x + 5 * y], _ROT[x][y])
            for x in range(5):
                for y in range(5):
                    state[x + 5 * y] = rotated[x + 5 * y] ^ ((~rotated[(x + 1) % 5 + 5 * y]) & rotated[(x + 2) % 5 + 5 * y])
                    state[x + 5 * y] &= _MASK
            state[0] ^= rc
    return b"".join(word.to_bytes(8, "little") for word in state)[:32]


def selector_for_signature(signature: str | None) -> str:
    return "0x" + ethereum_keccak256(signature.encode("utf-8"))[:4].hex() if signature else "UNKNOWN"


def _mask_comments_and_strings(text: str) -> str:
    result = list(text)
    i = 0
    while i < len(text):
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = len(text) if j < 0 else j
            for k in range(i, j): result[k] = " "
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = len(text) - 2 if j < 0 else j
            for k in range(i, min(j + 2, len(text))):
                if result[k] != "\n": result[k] = " "
            i = j + 2
        elif text[i] in "\"'":
            quote = text[i]; result[i] = " "; i += 1
            while i < len(text):
                current = text[i]
                if current != "\n": result[i] = " "
                if current == "\\":
                    i += 1
                    if i < len(text) and text[i] != "\n": result[i] = " "
                elif current == quote:
                    i += 1; break
                i += 1
        else: i += 1
    return "".join(result)


def _line(text: str, position: int) -> int: return text.count("\n", 0, position) + 1

def _brace_pairs(masked: str) -> tuple[dict[int, int], list[int]]:
    stack: list[int] = []; pairs: dict[int, int] = {}; depths = [0] * len(masked); depth = 0
    for i, char in enumerate(masked):
        depths[i] = depth
        if char == "{": stack.append(i); depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0: raise ValueError("unbalanced closing brace")
            pairs[stack.pop()] = i
    if stack: raise ValueError("unbalanced opening brace")
    return pairs, depths

def _find_header_end(masked: str, start: int) -> tuple[int, str] | None:
    parens = brackets = 0
    for i in range(start, len(masked)):
        char = masked[i]
        if char == "(": parens += 1
        elif char == ")": parens -= 1
        elif char == "[": brackets += 1
        elif char == "]": brackets -= 1
        elif char in "{;" and parens == 0 and brackets == 0:
            return i, char
    return None

def _split_params(text: str) -> list[str]:
    parts: list[str] = []; start = depth = 0
    for i, char in enumerate(text):
        if char in "([{": depth += 1
        elif char in ")]}": depth -= 1
        elif char == "," and depth == 0: parts.append(text[start:i]); start = i + 1
    parts.append(text[start:]); return [part.strip() for part in parts if part.strip()]

def _canonical_type(param: str) -> str | None:
    cleaned = " ".join(param.replace("indexed", " ").split())
    if "mapping" in cleaned or "function" in cleaned or cleaned.startswith("("): return None
    tokens = [x for x in cleaned.replace("payable", " ").replace("memory", " ").replace("calldata", " ").replace("storage", " ").split() if x]
    if not tokens: return None
    value = tokens[0]
    if value == "uint": value = "uint256"
    if value == "int": value = "int256"
    return value

def _signature(kind: str, name: str | None, header: str) -> str | None:
    if kind != "FUNCTION" or not name: return None
    begin = header.find("("); end = header.rfind(")")
    if begin < 0 or end < begin: return None
    types = [_canonical_type(part) for part in _split_params(header[begin + 1:end])]
    return None if any(item is None for item in types) else f"{name}({','.join(types)})"

def parse_solidity_source(text: str) -> dict[str, Any]:
    """Return conservative contract/function records; failures do not invent line ranges."""
    try:
        masked = _mask_comments_and_strings(text); pairs, depths = _brace_pairs(masked)
    except ValueError as exc:
        return {"parse_status": "PARSE_FAILED", "uncertainties": [str(exc)], "contracts": [], "functions": []}
    contracts = []
    for match in re.finditer(r"\b(?:(abstract)\s+)?(contract|library|interface)\s+([A-Za-z_$][\w$]*)[^;{]*\{", masked):
        opening = match.end() - 1
        if opening not in pairs: continue
        contracts.append({"contract_name": match.group(3), "contract_type": "abstract" if match.group(1) else match.group(2), "start_line": _line(text, match.start()), "end_line": _line(text, pairs[opening]), "opening": opening, "closing": pairs[opening], "depth": depths[opening]})
    functions = []
    keyword_map = {"function": "FUNCTION", "constructor": "CONSTRUCTOR", "fallback": "FALLBACK", "receive": "RECEIVE", "modifier": "MODIFIER"}
    for match in re.finditer(r"\b(function|constructor|fallback|receive|modifier)\b", masked):
        start = match.start(); owner = next((item for item in reversed(contracts) if item["opening"] < start < item["closing"] and depths[start] == item["depth"] + 1), None)
        if not owner: continue
        end = _find_header_end(masked, match.end())
        if not end: continue
        terminator, terminal = end
        if terminal == "{" and terminator not in pairs: continue
        kind = keyword_map[match.group(1)]; header = text[start:terminator]
        name_match = re.match(r"\s*([A-Za-z_$][\w$]*)", masked[match.end():terminator]) if kind in {"FUNCTION", "MODIFIER"} else None
        name = name_match.group(1) if name_match else ("constructor" if kind == "CONSTRUCTOR" else kind.lower())
        if kind == "MODIFIER":
            modifier_name = re.match(r"\s*([A-Za-z_$][\w$]*)", masked[match.end():terminator]); name = modifier_name.group(1) if modifier_name else None
        signature = _signature(kind, name, header)
        function_id = "fn_" + hashlib.sha256(f"{owner['contract_name']}\0{kind}\0{start}\0{signature or name}".encode()).hexdigest()[:16]
        functions.append({"function_id": function_id, "contract_name": owner["contract_name"], "contract_type": owner["contract_type"], "function_kind": kind, "function_name": name, "canonical_signature": signature, "selector": selector_for_signature(signature), "visibility": _visibility(header), "state_mutability": _mutability(header), "signature_start_line": _line(text, start), "body_start_line": _line(text, terminator) if terminal == "{" else None, "end_line": _line(text, pairs[terminator]) if terminal == "{" else _line(text, terminator), "parse_status": "PARSED", "uncertainties": (["INTERFACE_OR_ABSTRACT_DECLARATION"] if terminal == ";" else []), "_start": start, "_end": pairs[terminator] if terminal == "{" else terminator})
    for item in contracts: item.pop("opening"); item.pop("closing"); item.pop("depth")
    return {"parse_status": "PARSED", "uncertainties": [], "contracts": contracts, "functions": functions}

def _visibility(header: str) -> str | None:
    return next((value for value in ("external", "public", "internal", "private") if value in header.split()), None)
def _mutability(header: str) -> str | None:
    return next((value for value in ("pure", "view", "payable") if value in header.split()), "nonpayable")

def build_source_index(acquisition: dict, cache_root: str | Path) -> dict:
    if acquisition.get("schema_version") != "source_acquisition_v1": raise ValueError("invalid source acquisition schema")
    root = Path(cache_root); chain = acquisition.get("chain"); seen: set[str] = set(); addresses = []
    def visit(address: str, path: list[str]) -> None:
        normalized = norm_address(address)
        if normalized in seen: return
        seen.add(normalized); directory = root / chain / normalized; manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            addresses.append({"address": normalized, "proxy_resolution_path": path + [normalized], "status": "SOURCE_MISSING", "files": [], "functions": [], "uncertainties": ["CACHE_MANIFEST_MISSING"]}); return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")); record = {"address": normalized, "proxy_resolution_path": path + [normalized], "status": manifest.get("status"), "verified": manifest.get("verified", False), "source_manifest_sha256": sha(manifest_path), "proxy": manifest.get("proxy", {}), "files": [], "functions": [], "uncertainties": []}
        if manifest.get("status") != "verified": record["uncertainties"].append("SOURCE_NOT_VERIFIED")
        for file_info in manifest.get("files", []):
            relative = file_info.get("path"); file_path = directory / str(relative)
            if not relative or not file_path.exists() or sha(file_path) != file_info.get("sha256"):
                record["uncertainties"].append(f"FILE_HASH_MISMATCH:{relative}"); continue
            parsed = parse_solidity_source(file_path.read_text(encoding="utf-8"))
            file_record = {"file_path": str(relative).replace("\\", "/"), "file_sha256": file_info["sha256"], "parse_method": "CONSERVATIVE_LEXICAL_V1", "parse_status": parsed["parse_status"], "uncertainties": parsed["uncertainties"], "contracts": parsed["contracts"], "functions": []}
            for function in parsed["functions"]:
                function = dict(function); function.pop("_start", None); function.pop("_end", None)
                function.update({"function_id": "fn_" + hashlib.sha256(f"{normalized}\0{file_record['file_path']}\0{function['function_id']}".encode()).hexdigest()[:16], "address": normalized, "file_path": file_record["file_path"], "parse_method": "CONSERVATIVE_LEXICAL_V1"})
                file_record["functions"].append(function); record["functions"].append(function)
            record["files"].append(file_record)
        addresses.append(record)
        implementation = record["proxy"].get("implementation_address")
        if implementation: visit(implementation, path + [normalized])
    for item in acquisition.get("addresses", []):
        if item.get("requested_address"): visit(item["requested_address"], [])
    addresses.sort(key=lambda item: item["address"])
    return {"schema_version": "source_index_v1", "chain": chain, "source_acquisition_sha256": canonical_sha256(acquisition), "addresses": addresses, "metadata": {"address_count": len(addresses), "function_count": sum(len(item["functions"]) for item in addresses), "parse_method": "CONSERVATIVE_LEXICAL_V1"}}


_INTERNAL_CALL = re.compile(r"(?<![.$A-Za-z0-9_])([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
_NON_CALL_KEYWORDS = frozenset({
    "assert", "catch", "delete", "emit", "for", "if", "keccak256", "new",
    "require", "return", "revert", "selfdestruct", "super", "while",
})


def resolve_unique_internal_function_calls(
    index: Mapping[str, Any],
    cache_root: str | Path,
    source_address: str,
    caller_function_id: str,
) -> dict[str, Any]:
    """Resolve only unique, bare, same-contract Solidity function calls.

    The helper deliberately ignores member calls, overloads, external-only
    functions, and any source whose indexed hash no longer matches the cache.
    It returns fixed issue codes and never guesses argument types.
    """
    if index.get("schema_version") != "source_index_v1":
        raise ValueError("SOURCE_INTERNAL_CALL_INDEX_INVALID")
    try:
        address = norm_address(source_address)
    except ValueError as exc:
        raise ValueError("SOURCE_INTERNAL_CALL_ADDRESS_INVALID") from exc
    records = [item for item in index.get("addresses", ()) if isinstance(item, Mapping) and item.get("address") == address]
    if len(records) != 1 or records[0].get("status") != "verified":
        return {"schema_version": "source_internal_calls_v1", "calls": [], "issue_codes": ["INTERNAL_CALL_SOURCE_UNAVAILABLE"]}
    record = records[0]
    callers = [item for item in record.get("functions", ()) if isinstance(item, Mapping) and item.get("function_id") == caller_function_id]
    if len(callers) != 1:
        return {"schema_version": "source_internal_calls_v1", "calls": [], "issue_codes": ["INTERNAL_CALLER_FUNCTION_UNRESOLVED"]}
    caller = callers[0]
    file_rows = [item for item in record.get("files", ()) if isinstance(item, Mapping) and item.get("file_path") == caller.get("file_path")]
    if len(file_rows) != 1 or file_rows[0].get("parse_status") != "PARSED":
        return {"schema_version": "source_internal_calls_v1", "calls": [], "issue_codes": ["INTERNAL_CALL_SOURCE_UNAVAILABLE"]}
    relative = Path(str(caller.get("file_path", "")))
    root = Path(cache_root).resolve()
    path = (root / str(index.get("chain")) / address / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return {"schema_version": "source_internal_calls_v1", "calls": [], "issue_codes": ["INTERNAL_CALL_SOURCE_PATH_INVALID"]}
    if not path.is_file() or sha(path) != file_rows[0].get("file_sha256"):
        return {"schema_version": "source_internal_calls_v1", "calls": [], "issue_codes": ["INTERNAL_CALL_SOURCE_UNAVAILABLE"]}
    body_start, end_line = caller.get("body_start_line"), caller.get("end_line")
    if not isinstance(body_start, int) or not isinstance(end_line, int) or body_start > end_line:
        return {"schema_version": "source_internal_calls_v1", "calls": [], "issue_codes": ["INTERNAL_CALLER_BODY_UNAVAILABLE"]}
    try:
        text = path.read_text(encoding="utf-8")
        masked = _mask_comments_and_strings(text)
    except (OSError, UnicodeError, ValueError):
        return {"schema_version": "source_internal_calls_v1", "calls": [], "issue_codes": ["INTERNAL_CALL_SOURCE_UNAVAILABLE"]}
    lines = masked.splitlines()
    if body_start < 1 or end_line > len(lines):
        return {"schema_version": "source_internal_calls_v1", "calls": [], "issue_codes": ["INTERNAL_CALLER_BODY_UNAVAILABLE"]}
    body = "\n".join(lines[body_start - 1:end_line])
    opening = body.find("{")
    if opening < 0:
        return {"schema_version": "source_internal_calls_v1", "calls": [], "issue_codes": ["INTERNAL_CALLER_BODY_UNAVAILABLE"]}
    names = tuple(dict.fromkeys(match.group(1) for match in _INTERNAL_CALL.finditer(body[opening + 1:]) if match.group(1) not in _NON_CALL_KEYWORDS))
    calls = []
    issue_codes: set[str] = set()
    for name in names:
        matches = [
            item
            for item in record.get("functions", ())
            if isinstance(item, Mapping)
            and item.get("contract_name") == caller.get("contract_name")
            and item.get("function_kind") == "FUNCTION"
            and item.get("function_name") == name
            and item.get("visibility") != "external"
            and isinstance(item.get("body_start_line"), int)
        ]
        if len(matches) == 1:
            function = matches[0]
            calls.append({key: function.get(key) for key in ("function_id", "file_path", "contract_name", "function_kind", "function_name", "canonical_signature", "selector", "visibility", "signature_start_line", "body_start_line", "end_line")})
        elif len(matches) > 1:
            issue_codes.add("INTERNAL_FUNCTION_CALL_AMBIGUOUS_EXCLUDED")
    calls.sort(key=lambda item: (item.get("file_path", ""), item.get("canonical_signature") or item.get("function_name") or "", item.get("signature_start_line", 0), item.get("end_line", 0), item.get("function_id", "")))
    return {"schema_version": "source_internal_calls_v1", "calls": calls, "issue_codes": sorted(issue_codes)}

def serialize_source_index(index: dict) -> str: return json.dumps(index, ensure_ascii=False, indent=2) + "\n"
def write_source_index(index: dict, path: str | Path) -> None: atomic_json(path, index)

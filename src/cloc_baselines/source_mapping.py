"""Map EVM program counters to Solidity functions from Sourcify artifacts."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schema import FunctionRecord, LineRecord
from .solidity_parser import internal_call_names, parse_solidity_source


_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")
_CALL_OPS = frozenset({"CALL", "CALLCODE", "STATICCALL", "DELEGATECALL"})


@dataclass(frozen=True)
class ContractSourceMap:
    address: str
    functions: tuple[FunctionRecord, ...]
    pc_to_function_id: Mapping[int, str]
    lines: tuple[LineRecord, ...]
    pc_to_line_id: Mapping[int, str]
    call_edges: tuple[tuple[str, str], ...]
    compiler_version: str | None
    source: str = "SOURCIFY_V2"


def instruction_program_counters(bytecode: str) -> tuple[int, ...]:
    value = bytecode[2:] if bytecode.startswith("0x") else bytecode
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("Runtime bytecode is not valid hex") from exc
    counters: list[int] = []
    pc = 0
    while pc < len(raw):
        counters.append(pc)
        opcode = raw[pc]
        pc += 1 + (opcode - 0x5F if 0x60 <= opcode <= 0x7F else 0)
    return tuple(counters)


def expand_source_map(source_map: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(source_map, str):
        raise ValueError("Source map must be a string")
    previous = ["-1", "-1", "-1", "", ""]
    output: list[tuple[int, int, int]] = []
    for token in source_map.split(";"):
        fields = token.split(":")
        for index, field in enumerate(fields[:5]):
            if field:
                previous[index] = field
        try:
            output.append((int(previous[0]), int(previous[1]), int(previous[2])))
        except ValueError:
            output.append((-1, -1, -1))
    return tuple(output)


def _runtime_bytecode(raw: Mapping[str, Any]) -> str:
    value = raw.get("runtimeBytecode")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("onchainBytecode", "recompiledBytecode"):
            if isinstance(value.get(key), str):
                return value[key]
    raise ValueError("Sourcify runtime bytecode is missing")


def _compiled_contract(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    compilation = raw.get("compilation")
    output = raw.get("stdJsonOutput")
    if not isinstance(compilation, Mapping) or not isinstance(output, Mapping):
        raise ValueError("Sourcify compilation output is missing")
    qualified = compilation.get("fullyQualifiedName")
    if not isinstance(qualified, str) or ":" not in qualified:
        raise ValueError("Sourcify fully-qualified contract name is missing")
    source_file, contract_name = qualified.rsplit(":", 1)
    contracts = output.get("contracts")
    try:
        artifact = contracts[source_file][contract_name]
    except (KeyError, TypeError) as exc:
        raise ValueError("Compiled contract artifact is missing") from exc
    if not isinstance(artifact, Mapping):
        raise ValueError("Compiled contract artifact is invalid")
    return artifact


def build_contract_source_map(address: str, raw: Mapping[str, Any]) -> ContractSourceMap:
    normalized_address = address.lower()
    if not _ADDRESS.fullmatch(normalized_address):
        raise ValueError(f"Invalid contract address: {address}")
    artifact = _compiled_contract(raw)
    try:
        deployed = artifact["evm"]["deployedBytecode"]
        source_map_text = deployed["sourceMap"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Deployed bytecode source map is missing") from exc
    counters = instruction_program_counters(_runtime_bytecode(raw))
    locations = expand_source_map(source_map_text)
    pc_locations = dict(zip(counters, locations))
    sources = raw.get("sources")
    source_ids = raw.get("sourceIds")
    if not isinstance(sources, Mapping) or not isinstance(source_ids, Mapping):
        raise ValueError("Sourcify sources or source IDs are missing")
    functions: list[FunctionRecord] = []
    ranges_by_file_id: dict[int, list[tuple[int, int, str]]] = {}
    source_by_file_id: dict[int, tuple[str, str, tuple[int, ...]]] = {}
    parsed_functions: list[tuple[str, str, str | None, str]] = []
    for source_file, source_value in sources.items():
        if not isinstance(source_file, str) or not isinstance(source_value, Mapping):
            continue
        text = source_value.get("content")
        source_id_value = source_ids.get(source_file)
        if not isinstance(text, str) or not isinstance(source_id_value, Mapping):
            continue
        file_id = source_id_value.get("id")
        if not isinstance(file_id, int):
            continue
        parsed = parse_solidity_source(text)
        encoded = text.encode("utf-8")
        line_starts = tuple([0] + [index + 1 for index, value in enumerate(encoded) if value == 10])
        source_by_file_id[file_id] = (source_file, text, line_starts)
        file_ranges: list[tuple[int, int, str]] = []
        for function in parsed.get("functions", []):
            start_character = function.get("_start")
            end_character = function.get("_end")
            if not isinstance(start_character, int) or not isinstance(end_character, int):
                continue
            start = len(text[:start_character].encode("utf-8"))
            end = len(text[: end_character + 1].encode("utf-8"))
            if end <= start:
                continue
            function_id = f"{normalized_address}#{source_file}#{start}:{end - start}"
            source_code = text[start_character : end_character + 1]
            functions.append(
                FunctionRecord(
                    function_id=function_id,
                    address=normalized_address,
                    source_file=source_file,
                    source_offset=start,
                    source_length=end - start,
                    function_name=function.get("function_name"),
                    function_signature=function.get("canonical_signature"),
                    source_code=source_code,
                )
            )
            parsed_functions.append(
                (
                    function_id,
                    str(function.get("contract_name") or ""),
                    function.get("function_name") if isinstance(function.get("function_name"), str) else None,
                    source_code,
                )
            )
            file_ranges.append((start, end, function_id))
        ranges_by_file_id[file_id] = file_ranges
    pc_to_function: dict[int, str] = {}
    pc_to_line: dict[int, str] = {}
    lines: dict[str, LineRecord] = {}
    for pc, (start, length, file_id) in pc_locations.items():
        if start < 0 or length < 0 or file_id < 0:
            continue
        candidates = [
            item
            for item in ranges_by_file_id.get(file_id, [])
            if item[0] <= start < item[1]
        ]
        if candidates:
            function_id = min(candidates, key=lambda item: item[1] - item[0])[2]
            pc_to_function[pc] = function_id
            source_detail = source_by_file_id.get(file_id)
            if source_detail is not None:
                source_file, text, line_starts = source_detail
                line_number = bisect_right(line_starts, start)
                source_lines = text.splitlines()
                source_text = source_lines[line_number - 1] if line_number <= len(source_lines) else ""
                # A minified Solidity file can place multiple functions on one physical
                # line.  Keep the enclosing function in the identity so Grace retains
                # an unambiguous function-line edge.
                line_id = f"{function_id}#L{line_number}"
                pc_to_line[pc] = line_id
                lines.setdefault(
                    line_id,
                    LineRecord(
                        line_id=line_id,
                        function_id=function_id,
                        address=normalized_address,
                        source_file=source_file,
                        line_number=line_number,
                        source_text=source_text,
                    ),
                )
    compilation = raw.get("compilation")
    compiler_version = compilation.get("compilerVersion") if isinstance(compilation, Mapping) else None
    call_edges: list[tuple[str, str]] = []
    for caller_id, contract_name, _caller_name, source_code in parsed_functions:
        for called_name in internal_call_names(source_code):
            matches = [
                function_id
                for function_id, owner, function_name, _source in parsed_functions
                if owner == contract_name and function_name == called_name
            ]
            if len(matches) == 1 and matches[0] != caller_id:
                call_edges.append((caller_id, matches[0]))
    return ContractSourceMap(
        address=normalized_address,
        functions=tuple(functions),
        pc_to_function_id=pc_to_function,
        lines=tuple(lines.values()),
        pc_to_line_id=pc_to_line,
        call_edges=tuple(dict.fromkeys(call_edges)),
        compiler_version=compiler_version if isinstance(compiler_version, str) else None,
    )


def _stack_address(stack: Any) -> str | None:
    if not isinstance(stack, list) or len(stack) < 2 or not isinstance(stack[-2], str):
        return None
    word = stack[-2].lower()
    if word.startswith("0x"):
        word = word[2:]
    if not re.fullmatch(r"[0-9a-f]{40,64}", word):
        return None
    return "0x" + word[-40:]


def reconstruct_trace_addresses(
    transaction_to: str,
    struct_logs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    root = transaction_to.lower()
    if not _ADDRESS.fullmatch(root):
        raise ValueError("Contract-creation transactions are not supported as trace roots")
    if not struct_logs:
        raise ValueError("Trace contains no struct logs")
    first_depth = struct_logs[0].get("depth")
    if not isinstance(first_depth, int):
        raise ValueError("Trace depth is missing")
    address_by_depth: dict[int, str] = {first_depth: root}
    output: list[dict[str, Any]] = []
    for index, log in enumerate(struct_logs):
        depth = log.get("depth")
        pc = log.get("pc")
        opcode = log.get("op")
        if not isinstance(depth, int) or not isinstance(pc, int) or not isinstance(opcode, str):
            raise ValueError(f"Invalid struct log at index {index}")
        if index:
            previous = struct_logs[index - 1]
            previous_depth = previous.get("depth")
            previous_opcode = previous.get("op")
            if isinstance(previous_depth, int) and depth == previous_depth + 1 and previous_opcode in _CALL_OPS:
                target = _stack_address(previous.get("stack"))
                if target is not None:
                    address_by_depth[depth] = target
        for known_depth in tuple(address_by_depth):
            if known_depth > depth:
                del address_by_depth[known_depth]
        explicit = log.get("address") or log.get("contract")
        address = explicit.lower() if isinstance(explicit, str) and _ADDRESS.fullmatch(explicit.lower()) else address_by_depth.get(depth)
        output.append(
            {
                "index": index,
                "pc": pc,
                "opcode": opcode,
                "depth": depth,
                "address": address,
            }
        )
    return tuple(output)

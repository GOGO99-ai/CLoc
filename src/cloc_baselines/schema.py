"""Validated interchange format shared by all CLoc comparison baselines."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "cloc_function_coverage_v1"
_TX_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, field)


@dataclass(frozen=True)
class FunctionRecord:
    function_id: str
    address: str
    source_file: str | None = None
    source_offset: int | None = None
    source_length: int | None = None
    function_name: str | None = None
    function_signature: str | None = None
    source_code: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FunctionRecord":
        function_id = _nonempty(raw.get("function_id"), "function_id")
        address = _nonempty(raw.get("address"), "address").lower()
        if not _ADDRESS.fullmatch(address):
            raise ValueError(f"Invalid function address: {address}")
        offset = raw.get("source_offset")
        length = raw.get("source_length")
        if offset is not None and (not isinstance(offset, int) or isinstance(offset, bool) or offset < 0):
            raise ValueError(f"Invalid source_offset for {function_id}")
        if length is not None and (not isinstance(length, int) or isinstance(length, bool) or length <= 0):
            raise ValueError(f"Invalid source_length for {function_id}")
        if (offset is None) != (length is None):
            raise ValueError(f"source_offset and source_length must be supplied together for {function_id}")
        return cls(
            function_id=function_id,
            address=address,
            source_file=_optional_text(raw.get("source_file"), "source_file"),
            source_offset=offset,
            source_length=length,
            function_name=_optional_text(raw.get("function_name"), "function_name"),
            function_signature=_optional_text(raw.get("function_signature"), "function_signature"),
            source_code=raw.get("source_code") if isinstance(raw.get("source_code"), str) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_id": self.function_id,
            "address": self.address,
            "source_file": self.source_file,
            "source_offset": self.source_offset,
            "source_length": self.source_length,
            "function_name": self.function_name,
            "function_signature": self.function_signature,
            "source_code": self.source_code,
        }


@dataclass(frozen=True)
class LineRecord:
    line_id: str
    function_id: str
    address: str
    source_file: str | None
    line_number: int | None
    source_text: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LineRecord":
        line_id = _nonempty(raw.get("line_id"), "line_id")
        function_id = _nonempty(raw.get("function_id"), "function_id")
        address = _nonempty(raw.get("address"), "address").lower()
        if not _ADDRESS.fullmatch(address):
            raise ValueError(f"Invalid line address: {address}")
        line_number = raw.get("line_number")
        if line_number is not None and (
            not isinstance(line_number, int) or isinstance(line_number, bool) or line_number <= 0
        ):
            raise ValueError(f"Invalid line_number for {line_id}")
        return cls(
            line_id=line_id,
            function_id=function_id,
            address=address,
            source_file=_optional_text(raw.get("source_file"), "source_file"),
            line_number=line_number,
            source_text=raw.get("source_text") if isinstance(raw.get("source_text"), str) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "function_id": self.function_id,
            "address": self.address,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "source_text": self.source_text,
        }


@dataclass(frozen=True)
class TestRecord:
    transaction_hash: str
    outcome: str
    covered_function_ids: tuple[str, ...]
    covered_line_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TestRecord":
        transaction_hash = _nonempty(raw.get("transaction_hash"), "transaction_hash").lower()
        if not _TX_HASH.fullmatch(transaction_hash):
            raise ValueError(f"Invalid transaction hash: {transaction_hash}")
        outcome = _nonempty(raw.get("outcome"), "outcome").lower()
        if outcome not in {"fail", "pass"}:
            raise ValueError(f"Invalid test outcome for {transaction_hash}: {outcome}")
        covered = raw.get("covered_function_ids")
        if not isinstance(covered, list):
            raise ValueError(f"covered_function_ids must be a list for {transaction_hash}")
        normalized = tuple(dict.fromkeys(_nonempty(item, "covered_function_id") for item in covered))
        if not normalized:
            raise ValueError(f"Transaction has no function coverage: {transaction_hash}")
        covered_lines = raw.get("covered_line_ids", [])
        if not isinstance(covered_lines, list):
            raise ValueError(f"covered_line_ids must be a list for {transaction_hash}")
        normalized_lines = tuple(dict.fromkeys(_nonempty(item, "covered_line_id") for item in covered_lines))
        return cls(transaction_hash, outcome, normalized, normalized_lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_hash": self.transaction_hash,
            "outcome": self.outcome,
            "covered_function_ids": list(self.covered_function_ids),
            "covered_line_ids": list(self.covered_line_ids),
        }


@dataclass(frozen=True)
class CoverageCase:
    case_id: str
    case_name: str
    chain_id: int
    chain_key: str
    functions: tuple[FunctionRecord, ...]
    tests: tuple[TestRecord, ...]
    fault_function_ids: tuple[str, ...]
    call_edges: tuple[tuple[str, str], ...] = ()
    lines: tuple[LineRecord, ...] = ()
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CoverageCase":
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported coverage schema: {raw.get('schema_version')!r}")
        case_id = _nonempty(raw.get("case_id"), "case_id")
        case_name = _nonempty(raw.get("case_name"), "case_name")
        chain_id = raw.get("chain_id")
        if not isinstance(chain_id, int) or isinstance(chain_id, bool) or chain_id <= 0:
            raise ValueError(f"Invalid chain_id for {case_id}")
        chain_key = _nonempty(raw.get("chain_key"), "chain_key").lower()
        function_rows = raw.get("functions")
        test_rows = raw.get("tests")
        fault_rows = raw.get("fault_function_ids")
        if not isinstance(function_rows, list) or not function_rows:
            raise ValueError(f"No functions supplied for {case_id}")
        if not isinstance(test_rows, list) or not test_rows:
            raise ValueError(f"No tests supplied for {case_id}")
        if not isinstance(fault_rows, list) or not fault_rows:
            raise ValueError(f"No fault functions supplied for {case_id}")
        functions = tuple(FunctionRecord.from_dict(item) for item in function_rows)
        function_ids = [item.function_id for item in functions]
        if len(function_ids) != len(set(function_ids)):
            raise ValueError(f"Duplicate function_id in {case_id}")
        function_set = set(function_ids)
        tests = tuple(TestRecord.from_dict(item) for item in test_rows)
        tx_hashes = [item.transaction_hash for item in tests]
        if len(tx_hashes) != len(set(tx_hashes)):
            raise ValueError(f"Duplicate transaction in {case_id}")
        for test in tests:
            unknown = set(test.covered_function_ids) - function_set
            if unknown:
                raise ValueError(f"Unknown covered functions in {case_id}: {sorted(unknown)}")
        line_rows = raw.get("lines", [])
        if not isinstance(line_rows, list):
            raise ValueError(f"lines must be a list in {case_id}")
        lines = tuple(LineRecord.from_dict(item) for item in line_rows)
        line_ids = [item.line_id for item in lines]
        if len(line_ids) != len(set(line_ids)):
            raise ValueError(f"Duplicate line_id in {case_id}")
        line_set = set(line_ids)
        for line in lines:
            if line.function_id not in function_set:
                raise ValueError(f"Line references an unknown function in {case_id}: {line.line_id}")
        for test in tests:
            unknown_lines = set(test.covered_line_ids) - line_set
            if unknown_lines:
                raise ValueError(f"Unknown covered lines in {case_id}: {sorted(unknown_lines)}")
        if not any(item.outcome == "fail" for item in tests):
            raise ValueError(f"No failing transaction in {case_id}")
        if not any(item.outcome == "pass" for item in tests):
            raise ValueError(f"No passing transaction in {case_id}")
        fault_function_ids = tuple(dict.fromkeys(_nonempty(item, "fault_function_id") for item in fault_rows))
        unknown_faults = set(fault_function_ids) - function_set
        if unknown_faults:
            raise ValueError(f"Unknown fault functions in {case_id}: {sorted(unknown_faults)}")
        edge_rows = raw.get("call_edges", [])
        if not isinstance(edge_rows, list):
            raise ValueError(f"call_edges must be a list in {case_id}")
        edges: list[tuple[str, str]] = []
        for edge in edge_rows:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError(f"Invalid call edge in {case_id}: {edge!r}")
            source, target = (_nonempty(edge[0], "call_edge source"), _nonempty(edge[1], "call_edge target"))
            if source not in function_set or target not in function_set:
                raise ValueError(f"Call edge references an unknown function in {case_id}")
            if source != target:
                edges.append((source, target))
        metadata = raw.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(f"metadata must be an object in {case_id}")
        case = cls(
            case_id=case_id,
            case_name=case_name,
            chain_id=chain_id,
            chain_key=chain_key,
            functions=functions,
            tests=tests,
            fault_function_ids=fault_function_ids,
            call_edges=tuple(dict.fromkeys(edges)),
            lines=lines,
            metadata=dict(metadata) if metadata is not None else None,
        )
        uncovered_faults = set(case.fault_function_ids) - set(case.candidate_function_ids)
        if uncovered_faults:
            raise ValueError(
                f"Fault functions are not covered by a failing transaction in {case_id}: {sorted(uncovered_faults)}"
            )
        return case

    @classmethod
    def load(cls, path: str | Path) -> "CoverageCase":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"Coverage artifact must be an object: {path}")
        return cls.from_dict(value)

    @property
    def candidate_function_ids(self) -> tuple[str, ...]:
        covered = {
            function_id
            for test in self.tests
            if test.outcome == "fail"
            for function_id in test.covered_function_ids
        }
        return tuple(item.function_id for item in self.functions if item.function_id in covered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": self.case_id,
            "case_name": self.case_name,
            "chain_id": self.chain_id,
            "chain_key": self.chain_key,
            "functions": [item.to_dict() for item in self.functions],
            "tests": [item.to_dict() for item in self.tests],
            "fault_function_ids": list(self.fault_function_ids),
            "call_edges": [list(item) for item in self.call_edges],
            "lines": [item.to_dict() for item in self.lines],
            "metadata": dict(self.metadata or {}),
        }


def dump_coverage_case(case: CoverageCase, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(case.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

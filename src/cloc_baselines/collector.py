"""Checkpointed RPC/Sourcify collector for CLoc function coverage."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import requests

from .schema import CoverageCase, FunctionRecord, LineRecord, TestRecord, dump_coverage_case
from .source_mapping import ContractSourceMap, build_contract_source_map, reconstruct_trace_addresses


_LOCATION = re.compile(r"^(0x[0-9a-fA-F]{40})#([^#]*)#([0-9]+):([0-9]+)$")


class CollectionError(RuntimeError):
    pass


class JsonRpcClient:
    def __init__(self, url: str, *, timeout: float = 180.0):
        if not isinstance(url, str) or not url:
            raise ValueError("RPC URL is missing")
        self._url = url
        self._timeout = timeout
        self.request_count = 0

    def call(self, method: str, params: Sequence[Any]) -> Any:
        self.request_count += 1
        response = requests.post(
            self._url,
            json={"jsonrpc": "2.0", "id": self.request_count, "method": method, "params": list(params)},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise CollectionError("RPC_RESPONSE_INVALID")
        if payload.get("error"):
            error = payload["error"]
            code = error.get("code") if isinstance(error, Mapping) else None
            raise CollectionError(f"RPC_ERROR:{code}")
        return payload.get("result")


class SourcifyClient:
    def __init__(self, cache_root: Path, *, timeout: float = 120.0):
        self.cache_root = cache_root
        self.timeout = timeout
        self.request_count = 0

    def load(self, chain_id: int, address: str) -> Mapping[str, Any] | None:
        normalized = address.lower()
        path = self.cache_root / str(chain_id) / f"{normalized}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, Mapping) and "error" not in payload else None
        self.request_count += 1
        response = requests.get(
            f"https://sourcify.dev/server/v2/contract/{chain_id}/{normalized}",
            params={"fields": "all"},
            timeout=self.timeout,
        )
        if response.status_code == 404:
            payload: Mapping[str, Any] = {"error": "SOURCE_NOT_FOUND"}
        else:
            response.raise_for_status()
            value = response.json()
            payload = value if isinstance(value, Mapping) else {"error": "SOURCE_RESPONSE_INVALID"}
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        return payload if "error" not in payload and "customCode" not in payload else None


@dataclass
class TraceCache:
    root: Path

    def load_or_fetch(
        self,
        *,
        chain_key: str,
        transaction_hash: str,
        rpc: JsonRpcClient,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        path = self.root / chain_key / f"{transaction_hash}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, Mapping) and isinstance(value.get("transaction"), Mapping) and isinstance(value.get("trace"), Mapping):
                transaction = value["transaction"]
                trace = value["trace"]
                call_trace = value.get("call_trace")
                if not isinstance(call_trace, Mapping):
                    call_trace = rpc.call("debug_traceTransaction", [transaction_hash, {"tracer": "callTracer"}])
                    if not isinstance(call_trace, Mapping):
                        raise CollectionError("CALL_TRACE_RESPONSE_INVALID")
                    with gzip.open(path, "wt", encoding="utf-8") as handle:
                        json.dump({"transaction": transaction, "trace": trace, "call_trace": call_trace}, handle, separators=(",", ":"))
                return transaction, trace, call_trace
        transaction = rpc.call("eth_getTransactionByHash", [transaction_hash])
        trace = rpc.call(
            "debug_traceTransaction",
            [transaction_hash, {"disableStorage": True, "disableMemory": True, "enableReturnData": False}],
        )
        call_trace = rpc.call("debug_traceTransaction", [transaction_hash, {"tracer": "callTracer"}])
        if not isinstance(transaction, Mapping) or not isinstance(trace, Mapping) or not isinstance(call_trace, Mapping):
            raise CollectionError("TRACE_RESPONSE_INVALID")
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"transaction": transaction, "trace": trace, "call_trace": call_trace}, handle, separators=(",", ":"))
        return transaction, trace, call_trace


def _fallback_function(address: str, selector: str) -> FunctionRecord:
    return FunctionRecord(
        function_id=f"{address}###{selector}",
        address=address,
        function_name=selector,
        function_signature=None,
    )


def _fallback_line(function: FunctionRecord) -> LineRecord:
    return LineRecord(
        line_id=f"{function.function_id}#ENTRY",
        function_id=function.function_id,
        address=function.address,
        source_file=None,
        line_number=None,
        source_text=function.function_name,
    )


def _call_frames(root: Mapping[str, Any]) -> tuple[tuple[str, str, str | None], ...]:
    output: list[tuple[str, str, str | None]] = []

    def visit(frame: Mapping[str, Any], parent_key: str | None) -> None:
        address = frame.get("to")
        input_data = frame.get("input")
        current_key: str | None = None
        if isinstance(address, str) and re.fullmatch(r"0x[0-9a-fA-F]{40}", address):
            normalized = address.lower()
            selector = input_data[:10].lower() if isinstance(input_data, str) and len(input_data) >= 10 else "0x00000000"
            current_key = f"{normalized}###{selector}"
            output.append((normalized, selector, parent_key))
        children = frame.get("calls")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, Mapping):
                    visit(child, current_key or parent_key)

    visit(root, None)
    return tuple(output)


def _source_file_matches(candidate: str | None, reported: str) -> bool:
    if not reported:
        return True
    if candidate is None:
        return False
    normalized_candidate = PurePosixPath(candidate.replace("\\", "/")).as_posix()
    normalized_reported = PurePosixPath(reported.replace("\\", "/")).as_posix()
    return normalized_candidate == normalized_reported or normalized_candidate.endswith("/" + normalized_reported) or PurePosixPath(normalized_candidate).name == PurePosixPath(normalized_reported).name


def resolve_fault_function_ids(
    locations: Sequence[str],
    functions: Sequence[FunctionRecord],
) -> tuple[str, ...]:
    resolved: list[str] = []
    for location in locations:
        match = _LOCATION.fullmatch(location.strip()) if isinstance(location, str) else None
        if match is None:
            continue
        address, source_file, offset_text, length_text = match.groups()
        address = address.lower()
        offset, length = int(offset_text), int(length_text)
        candidates = [
            function
            for function in functions
            if function.address == address
            and function.source_offset is not None
            and function.source_length is not None
            and function.source_offset <= offset
            and offset + length <= function.source_offset + function.source_length
            and _source_file_matches(function.source_file, source_file)
        ]
        if len(candidates) == 1:
            resolved.append(candidates[0].function_id)
    return tuple(dict.fromkeys(resolved))


class CaseCollector:
    def __init__(
        self,
        *,
        rpc: JsonRpcClient,
        sourcify: SourcifyClient,
        trace_cache: TraceCache,
    ):
        self.rpc = rpc
        self.sourcify = sourcify
        self.trace_cache = trace_cache
        self._source_maps: dict[tuple[int, str], ContractSourceMap | None] = {}

    def _source_map(self, chain_id: int, address: str) -> ContractSourceMap | None:
        key = (chain_id, address)
        if key not in self._source_maps:
            raw = self.sourcify.load(chain_id, address)
            if raw is None:
                self._source_maps[key] = None
            else:
                try:
                    self._source_maps[key] = build_contract_source_map(address, raw)
                except ValueError:
                    self._source_maps[key] = None
        return self._source_maps[key]

    def collect(self, manifest_case: Mapping[str, Any]) -> CoverageCase:
        chain_id = manifest_case.get("chain_id")
        chain_key = manifest_case.get("chain_key")
        if not isinstance(chain_id, int) or not isinstance(chain_key, str):
            raise CollectionError("MANIFEST_CHAIN_INVALID")
        failing = tuple(manifest_case.get("selected_fault_transactions", []))
        passing = tuple(manifest_case.get("normal_transactions", []))
        function_by_id: dict[str, FunctionRecord] = {}
        line_by_id: dict[str, LineRecord] = {}
        tests: list[TestRecord] = []
        call_edges: list[tuple[str, str]] = []
        source_mapped_addresses: set[str] = set()
        source_missing_addresses: set[str] = set()
        for outcome, transaction_hashes in (("fail", failing), ("pass", passing)):
            for transaction_hash in transaction_hashes:
                transaction, trace, call_trace = self.trace_cache.load_or_fetch(
                    chain_key=chain_key,
                    transaction_hash=transaction_hash,
                    rpc=self.rpc,
                )
                transaction_to = transaction.get("to")
                struct_logs = trace.get("structLogs")
                if not isinstance(transaction_to, str) or not isinstance(struct_logs, list):
                    raise CollectionError(f"TRACE_FIELDS_MISSING:{transaction_hash}")
                reconstructed = reconstruct_trace_addresses(transaction_to, struct_logs)
                coverage: list[str] = []
                line_coverage: list[str] = []
                sequence: list[tuple[str, str, int]] = []
                mapped_addresses_in_transaction: set[str] = set()
                addresses = sorted({item["address"] for item in reconstructed if isinstance(item.get("address"), str)})
                maps: dict[str, ContractSourceMap] = {}
                for address in addresses:
                    source_map = self._source_map(chain_id, address)
                    if source_map is None:
                        source_missing_addresses.add(address)
                        continue
                    maps[address] = source_map
                    source_mapped_addresses.add(address)
                    for function in source_map.functions:
                        function_by_id.setdefault(function.function_id, function)
                    for line in source_map.lines:
                        line_by_id.setdefault(line.line_id, line)
                    call_edges.extend(source_map.call_edges)
                for item in reconstructed:
                    address = item.get("address")
                    source_map = maps.get(address) if isinstance(address, str) else None
                    function_id = source_map.pc_to_function_id.get(item["pc"]) if source_map is not None else None
                    if function_id is None:
                        continue
                    if isinstance(address, str):
                        mapped_addresses_in_transaction.add(address)
                    coverage.append(function_id)
                    line_id = source_map.pc_to_line_id.get(item["pc"]) if source_map is not None else None
                    if line_id is not None:
                        line_coverage.append(line_id)
                    if not sequence or sequence[-1][0] != function_id:
                        sequence.append((function_id, address, item["depth"]))
                for source, target in zip(sequence, sequence[1:]):
                    if source[0] != target[0] and target[2] > source[2]:
                        call_edges.append((source[0], target[0]))
                fallback_ids: dict[str, str] = {}
                for address, selector, parent_key in _call_frames(call_trace):
                    if address in mapped_addresses_in_transaction:
                        continue
                    fallback = _fallback_function(address, selector)
                    function_by_id.setdefault(fallback.function_id, fallback)
                    fallback_line = _fallback_line(fallback)
                    line_by_id.setdefault(fallback_line.line_id, fallback_line)
                    fallback_ids[fallback.function_id] = fallback.function_id
                    coverage.append(fallback.function_id)
                    line_coverage.append(fallback_line.line_id)
                    if parent_key in fallback_ids:
                        call_edges.append((fallback_ids[parent_key], fallback.function_id))
                if not coverage:
                    input_data = transaction.get("input")
                    selector = input_data[:10].lower() if isinstance(input_data, str) and len(input_data) >= 10 else "0x00000000"
                    fallback = _fallback_function(transaction_to.lower(), selector)
                    function_by_id.setdefault(fallback.function_id, fallback)
                    fallback_line = _fallback_line(fallback)
                    line_by_id.setdefault(fallback_line.line_id, fallback_line)
                    coverage.append(fallback.function_id)
                    line_coverage.append(fallback_line.line_id)
                tests.append(
                    TestRecord(
                        transaction_hash,
                        outcome,
                        tuple(dict.fromkeys(coverage)),
                        tuple(dict.fromkeys(line_coverage)),
                    )
                )
        functions = tuple(function_by_id.values())
        fault_function_ids = resolve_fault_function_ids(
            tuple(manifest_case.get("fault_locations", [])), functions
        )
        if not fault_function_ids:
            raise CollectionError("GROUND_TRUTH_FUNCTION_NOT_COVERED_OR_SOURCE_UNAVAILABLE")
        candidate_ids = {
            function_id
            for test in tests
            if test.outcome == "fail"
            for function_id in test.covered_function_ids
        }
        if not set(fault_function_ids) <= candidate_ids:
            raise CollectionError("GROUND_TRUTH_FUNCTION_NOT_EXECUTED_BY_REPRESENTATIVE_FAULT_TRANSACTION")
        edge_set = tuple(
            edge
            for edge in dict.fromkeys(call_edges)
            if edge[0] in function_by_id and edge[1] in function_by_id
        )
        return CoverageCase.from_dict(
            {
                "schema_version": "cloc_function_coverage_v1",
                "case_id": manifest_case["case_id"],
                "case_name": manifest_case["case_name"],
                "chain_id": chain_id,
                "chain_key": chain_key,
                "functions": [item.to_dict() for item in functions],
                "tests": [item.to_dict() for item in tests],
                "fault_function_ids": list(fault_function_ids),
                "call_edges": [list(item) for item in edge_set],
                "lines": [item.to_dict() for item in line_by_id.values()],
                "metadata": {
                    "collector": "RPC_STRUCTLOG_PLUS_SOURCIFY_V2",
                    "call_graph": "STATIC_UNIQUE_INTERNAL_CALLS_PLUS_DYNAMIC_CROSS_CONTRACT_CALLS",
                    "source_mapped_addresses": sorted(source_mapped_addresses),
                    "source_missing_addresses": sorted(source_missing_addresses),
                    "rpc_request_count": self.rpc.request_count,
                    "sourcify_request_count": self.sourcify.request_count,
                },
            }
        )


def write_case(case: CoverageCase, path: Path) -> None:
    dump_coverage_case(case, path)

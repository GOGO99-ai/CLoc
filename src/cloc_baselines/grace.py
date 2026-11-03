"""Leakage-free CLoc adaptation of the public Grace heterogeneous graph model."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ranking import RankedFunction, rank_scores
from .schema import CoverageCase, LineRecord, TestRecord


GRACE_SCHEMA_VERSION = "cloc_grace_graph_v1"


@dataclass(frozen=True)
class GraceGraphCase:
    case_id: str
    case_name: str
    chain_id: int
    chain_key: str
    function_ids: tuple[str, ...]
    tests: tuple[TestRecord, ...]
    lines: tuple[LineRecord, ...]
    fault_function_ids: tuple[str, ...]
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_coverage(cls, case: CoverageCase) -> "GraceGraphCase":
        candidates = case.candidate_function_ids
        candidate_set = set(candidates)
        lines = tuple(line for line in case.lines if line.function_id in candidate_set)
        line_set = {line.line_id for line in lines}
        tests = tuple(
            TestRecord(
                transaction_hash=test.transaction_hash,
                outcome=test.outcome,
                covered_function_ids=test.covered_function_ids,
                covered_line_ids=tuple(line_id for line_id in test.covered_line_ids if line_id in line_set),
            )
            for test in case.tests
        )
        graph = cls(
            case_id=case.case_id,
            case_name=case.case_name,
            chain_id=case.chain_id,
            chain_key=case.chain_key,
            function_ids=candidates,
            tests=tests,
            lines=lines,
            fault_function_ids=case.fault_function_ids,
            metadata={
                "adapter": "CLOC_GRACE_SOLIDITY_LINE_GRAPH",
                "test_selection_policy": "FIXED_EPOCH_NO_HELD_OUT_LABEL_SELECTION",
                "source_coverage_schema": "cloc_function_coverage_v1",
            },
        )
        graph.validate()
        return graph

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraceGraphCase":
        if raw.get("schema_version") != GRACE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Grace graph schema: {raw.get('schema_version')!r}")
        function_rows = raw.get("function_ids")
        test_rows = raw.get("tests")
        line_rows = raw.get("lines")
        fault_rows = raw.get("fault_function_ids")
        if not isinstance(function_rows, list) or not all(isinstance(item, str) for item in function_rows):
            raise ValueError("Grace function_ids must be a list of strings")
        if not isinstance(test_rows, list) or not isinstance(line_rows, list) or not isinstance(fault_rows, list):
            raise ValueError("Grace graph tests, lines and fault_function_ids must be lists")
        metadata = raw.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("Grace graph metadata must be an object")
        graph = cls(
            case_id=str(raw.get("case_id", "")),
            case_name=str(raw.get("case_name", "")),
            chain_id=raw.get("chain_id"),
            chain_key=str(raw.get("chain_key", "")),
            function_ids=tuple(dict.fromkeys(function_rows)),
            tests=tuple(TestRecord.from_dict(item) for item in test_rows),
            lines=tuple(LineRecord.from_dict(item) for item in line_rows),
            fault_function_ids=tuple(dict.fromkeys(item for item in fault_rows if isinstance(item, str))),
            metadata=dict(metadata) if metadata is not None else None,
        )
        graph.validate()
        return graph

    @classmethod
    def load(cls, path: str | Path) -> "GraceGraphCase":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"Grace graph must be an object: {path}")
        return cls.from_dict(raw)

    def validate(self) -> None:
        if not self.case_id or not self.case_name or not self.chain_key:
            raise ValueError("Grace graph identity fields must be non-empty")
        if not isinstance(self.chain_id, int) or isinstance(self.chain_id, bool) or self.chain_id <= 0:
            raise ValueError(f"Invalid Grace graph chain_id for {self.case_id}")
        if not self.function_ids or len(self.function_ids) != len(set(self.function_ids)):
            raise ValueError(f"Grace graph needs unique candidate functions for {self.case_id}")
        function_set = set(self.function_ids)
        if not set(self.fault_function_ids) or not set(self.fault_function_ids) <= function_set:
            raise ValueError(f"Grace graph fault functions are invalid for {self.case_id}")
        line_ids = [line.line_id for line in self.lines]
        if not line_ids or len(line_ids) != len(set(line_ids)):
            raise ValueError(f"Grace graph needs unique source lines for {self.case_id}")
        line_set = set(line_ids)
        functions_with_lines = {line.function_id for line in self.lines}
        if not function_set <= functions_with_lines:
            missing = sorted(function_set - functions_with_lines)
            raise ValueError(f"Grace graph functions without source lines in {self.case_id}: {missing}")
        if any(line.function_id not in function_set for line in self.lines):
            raise ValueError(f"Grace graph line references a non-candidate function in {self.case_id}")
        if not any(test.outcome == "fail" for test in self.tests) or not any(
            test.outcome == "pass" for test in self.tests
        ):
            raise ValueError(f"Grace graph requires failing and passing tests for {self.case_id}")
        for test in self.tests:
            if set(test.covered_line_ids) - line_set:
                raise ValueError(f"Grace test references an unknown line in {self.case_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRACE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "case_name": self.case_name,
            "chain_id": self.chain_id,
            "chain_key": self.chain_key,
            "function_ids": list(self.function_ids),
            "tests": [test.to_dict() for test in self.tests],
            "lines": [line.to_dict() for line in self.lines],
            "fault_function_ids": list(self.fault_function_ids),
            "metadata": dict(self.metadata or {}),
        }


def dump_grace_graph(graph: GraceGraphCase, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _graph_tensors(graph: GraceGraphCase, *, device: Any) -> tuple[Any, Any, int]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("Grace requires PyTorch") from exc
    functions = graph.function_ids
    failing = tuple(test for test in graph.tests if test.outcome == "fail")
    passing = tuple(test for test in graph.tests if test.outcome == "pass")
    lines = graph.lines
    function_index = {value: index for index, value in enumerate(functions)}
    fail_index = {
        test.transaction_hash: len(functions) + index for index, test in enumerate(failing)
    }
    pass_index = {
        test.transaction_hash: len(functions) + len(failing) + index for index, test in enumerate(passing)
    }
    line_start = len(functions) + len(failing) + len(passing)
    line_index = {line.line_id: line_start + index for index, line in enumerate(lines)}
    node_count = line_start + len(lines)
    adjacency = torch.zeros((node_count, node_count), dtype=torch.float32, device=device)
    for line in lines:
        source = function_index[line.function_id]
        target = line_index[line.line_id]
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    for test in graph.tests:
        source = (fail_index if test.outcome == "fail" else pass_index)[test.transaction_hash]
        for line_id in test.covered_line_ids:
            target = line_index[line_id]
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
    degree = adjacency.sum(dim=1).clamp(min=1.0)
    inverse_sqrt = degree.rsqrt()
    adjacency = inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    type_ids = (
        [2] * len(functions)
        + [3] * len(failing)
        + [5] * len(passing)
        + [4] * len(lines)
    )
    node_types = torch.tensor(type_ids, dtype=torch.long, device=device)
    return node_types, adjacency, len(functions)


def rank_grace_leave_one_out(
    graphs: Sequence[GraceGraphCase],
    coverage_cases: Mapping[str, CoverageCase],
    *,
    seed: int = 0,
    epochs: int = 15,
    learning_rate: float = 1.0e-3,
    embedding_size: int = 32,
    layers: int = 5,
    device: str | None = None,
) -> dict[str, tuple[RankedFunction, ...]]:
    """Train one model per held-out CLoc case without using its labels for epoch selection."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("Grace requires PyTorch") from exc
    if len(graphs) < 2:
        raise ValueError("Grace leave-one-out evaluation requires at least two cases")
    if epochs <= 0 or embedding_size < 2 or layers <= 0:
        raise ValueError("Invalid Grace training hyperparameters")
    execution_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    class GraphBlock(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.outer_norm = nn.LayerNorm(width)
            self.linear = nn.Linear(width, width)
            self.inner_norm = nn.LayerNorm(width)
            self.lstm = nn.LSTMCell(width, width)
            self.linear_second = nn.Linear(width, width)
            self.dropout = nn.Dropout(0.1)
            self.final_norm = nn.LayerNorm(width)

        def forward(self, value: Any, adjacency: Any) -> Any:
            state = self.linear(self.outer_norm(value))
            normalized = self.inner_norm(state)
            aggregate = adjacency @ state
            zeros = torch.zeros_like(normalized)
            _hidden, cell = self.lstm(aggregate, (zeros, normalized))
            state = state + self.dropout(cell)
            state = self.linear_second(state)
            return self.dropout(self.final_norm(value + self.dropout(state)))

    class GraceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.type_embedding = nn.Embedding(6, embedding_size - 1)
            self.blocks = nn.ModuleList(GraphBlock(embedding_size) for _ in range(layers))
            self.output = nn.Linear(embedding_size, 1)

        def forward(self, graph: GraceGraphCase) -> Any:
            node_types, adjacency, function_count = _graph_tensors(graph, device=execution_device)
            type_embedding = self.type_embedding(node_types)
            # Grace's lexical-overlap feature is 1 when no method/test-name token overlaps.
            # CLoc test identities are transaction hashes, so this faithful mapping is constant.
            overlap = torch.zeros((len(node_types), 1), dtype=torch.float32, device=execution_device)
            overlap[:function_count] = 1.0
            state = torch.cat((type_embedding, overlap), dim=1)
            for block in self.blocks:
                state = block(state, adjacency)
            return torch.softmax(self.output(state[:function_count]).squeeze(-1), dim=0)

    graph_by_id = {graph.case_id: graph for graph in graphs}
    if len(graph_by_id) != len(graphs):
        raise ValueError("Duplicate Grace case_id")
    missing_coverage = set(graph_by_id) - set(coverage_cases)
    if missing_coverage:
        raise ValueError(f"Coverage missing for Grace cases: {sorted(missing_coverage)}")
    rankings: dict[str, tuple[RankedFunction, ...]] = {}
    ordered_ids = tuple(graph_by_id)
    for fold_index, held_out_id in enumerate(ordered_ids):
        fold_seed = seed + fold_index
        random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)
        model = GraceModel().to(execution_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        training_ids = [case_id for case_id in ordered_ids if case_id != held_out_id]
        model.train()
        for epoch in range(epochs):
            order = list(training_ids)
            random.Random(fold_seed * 1000 + epoch).shuffle(order)
            for case_id in order:
                graph = graph_by_id[case_id]
                scores = model(graph)
                fault_indices = [
                    graph.function_ids.index(function_id) for function_id in graph.fault_function_ids
                ]
                loss = -torch.log(scores[fault_indices].clamp(min=1.0e-10)).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        held_out = graph_by_id[held_out_id]
        with torch.no_grad():
            values = model(held_out).detach().cpu().tolist()
        score_map = dict(zip(held_out.function_ids, (float(value) for value in values)))
        rankings[held_out_id] = rank_scores(coverage_cases[held_out_id], score_map)
    return rankings

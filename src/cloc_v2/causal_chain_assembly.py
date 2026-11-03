"""Assemble candidate backbones into bounded evidence-branch chains."""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .backbone_models import CausalBackbonesResult
from .causal_backbones import LEVEL_PRIORITY, VIEWS, _validate_graph, normalized_sha256
from .causal_chain_models import (
    CandidateCausalChain,
    CandidateCausalChainsResult,
    ChainAssemblyConfig,
    EvidenceBranch,
)


CANONICAL_EDGE_PRIORITY = {
    "RESULT": 0,
    "ASSET": 1,
    "STATE": 2,
    "PARAM": 3,
    "CALL": 4,
    "ORDER": 5,
}
SEMANTIC_TYPES = ("ASSET", "STATE", "PARAM")
NO_BRANCH_REASONS = (
    "NO_EXTERNAL_SEMANTIC_PREDECESSOR",
    "ALL_CANDIDATES_REJECTED_BY_LIMITS_OR_CONFLICTS",
)


@dataclass(frozen=True)
class _BranchCandidate:
    attachment_node_id: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    semantic_edge: Mapping[str, Any]
    execution_span: int
    attachment_distance: int

    @property
    def external_node_ids(self) -> tuple[str, ...]:
        return self.node_ids[:-1]


def build_candidate_causal_chains(
    graph: Mapping[str, Any],
    backbones_document: Mapping[str, Any],
    config: ChainAssemblyConfig | None = None,
) -> CandidateCausalChainsResult:
    """Pure, deterministic stage-3B assembly from graph and backbone JSON data."""
    cfg = config or ChainAssemblyConfig()
    _validate_config(cfg)
    nodes, edges, seeds, root_ids = _validate_graph(graph)
    edge_by_id = {edge["edge_id"]: edge for edge in edges}
    incoming = defaultdict(list)
    parallel = defaultdict(list)
    for edge in edges:
        incoming[edge["dst"]].append(edge)
        parallel[(edge["src"], edge["dst"])].append(edge)
    for edge_list in incoming.values():
        edge_list.sort(key=lambda edge: edge["edge_id"])
    for edge_list in parallel.values():
        edge_list.sort(key=lambda edge: edge["edge_id"])

    graph_sha256 = normalized_sha256(graph)
    backbones = _validate_backbones_document(
        backbones_document, graph_sha256, edge_by_id, nodes, seeds, root_ids
    )
    seed_order = {seed["result_id"]: index for index, seed in enumerate(seeds)}
    groups = _group_backbones(backbones, seed_order)
    audit = Counter(
        {
            "input_backbone_count": len(backbones),
            "grouped_chain_count": len(groups),
            "merged_backbone_count": len(backbones) - len(groups),
            "branch_candidates_considered": 0,
            "branch_candidates_truncated": 0,
            "branch_rejected_node_limit": 0,
            "branch_rejected_hypothesis_limit": 0,
            "branch_rejected_state_object_limit": 0,
            "branch_rejected_external_node_conflict": 0,
            "branch_rejected_edge_conflict": 0,
        }
    )
    chains: list[CandidateCausalChain] = []
    for chain_index, group in enumerate(groups, start=1):
        chain = _assemble_chain(
            chain_id=f"chain_c{chain_index:03d}",
            group=group,
            nodes=nodes,
            edge_by_id=edge_by_id,
            parallel=parallel,
            incoming=incoming,
            root_ids=root_ids,
            config=cfg,
            audit=audit,
        )
        chains.append(chain)

    _validate_output(chains, edge_by_id, nodes, seeds, root_ids)
    coverage = _result_coverage(chains, seeds)
    no_branch_reasons = Counter(
        chain.structural_features["no_branch_reason"]
        for chain in chains
        if chain.structural_features["no_branch_reason"] is not None
    )
    support_types = Counter(
        edge_by_id[edge_id]["edge_type"]
        for chain in chains
        for edge_id in chain.supporting_edge_ids
    )
    branch_types = Counter(
        branch.semantic_edge_type for chain in chains for branch in chain.branches
    )
    assembly_audit = {
        **dict(sorted(audit.items())),
        "supporting_edge_count": sum(len(chain.supporting_edge_ids) for chain in chains),
        "supporting_edge_type_counts": dict(sorted(support_types.items())),
        "branch_count": sum(len(chain.branches) for chain in chains),
        "branch_semantic_type_counts": dict(sorted(branch_types.items())),
        "chains_without_branches": sum(not chain.branches for chain in chains),
        "no_branch_reason_counts": dict(sorted(no_branch_reasons.items())),
        "structural_invariant_failures": 0,
    }
    return CandidateCausalChainsResult(
        candidate_causal_graph_sha256=graph_sha256,
        causal_backbones_sha256=normalized_sha256(backbones_document),
        assembly_config=cfg,
        chains=tuple(chains),
        result_coverage=coverage,
        assembly_audit=assembly_audit,
    )


def _group_backbones(
    backbones: list[Mapping[str, Any]], seed_order: Mapping[str, int]
) -> list[list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]] = defaultdict(list)
    for backbone in backbones:
        grouped[(backbone["result_id"], tuple(backbone["node_ids"]))].append(backbone)
    return [
        grouped[key]
        for key in sorted(
            grouped,
            key=lambda item: (seed_order[item[0]], item[1]),
        )
    ]


def _assemble_chain(
    chain_id: str,
    group: list[Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
    edge_by_id: Mapping[str, Mapping[str, Any]],
    parallel: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    incoming: Mapping[str, list[Mapping[str, Any]]],
    root_ids: set[str],
    config: ChainAssemblyConfig,
    audit: Counter[str],
) -> CandidateCausalChain:
    first = group[0]
    result_id = first["result_id"]
    result_node_id = first["result_node_id"]
    backbone_nodes = tuple(first["node_ids"])
    semantic_views = tuple(
        view for view in VIEWS if any(view in backbone["semantic_views"] for backbone in group)
    )
    stop_reason = _canonical_stop_reason(group)
    canonical_edges, supporting_edges = _canonicalize_backbone(
        backbone_nodes,
        result_id,
        result_node_id,
        parallel,
        root_ids,
    )
    candidates = _branch_candidates(
        backbone_nodes, result_node_id, nodes, incoming, config
    )
    candidates.sort(key=lambda candidate: _branch_rank(candidate, nodes))
    if len(candidates) > config.max_branch_candidates_per_chain:
        audit["branch_candidates_truncated"] += len(candidates) - config.max_branch_candidates_per_chain
        candidates = candidates[: config.max_branch_candidates_per_chain]
    audit["branch_candidates_considered"] += len(candidates)
    selected, rejection_counts = _select_branches(
        candidates, backbone_nodes, nodes, config
    )
    for key, value in rejection_counts.items():
        audit[f"branch_rejected_{key}"] += value
    branches = tuple(
        EvidenceBranch(
            branch_id=f"{chain_id}_branch_b{index:03d}",
            attachment_node_id=candidate.attachment_node_id,
            node_ids=candidate.node_ids,
            edge_ids=candidate.edge_ids,
            semantic_edge_id=candidate.semantic_edge["edge_id"],
            semantic_edge_type=candidate.semantic_edge["edge_type"],
            evidence_level=candidate.semantic_edge["evidence_level"],
        )
        for index, candidate in enumerate(selected, start=1)
    )
    all_node_ids = _all_node_ids(backbone_nodes, branches)
    all_edge_ids = _all_edge_ids(canonical_edges, supporting_edges, branches)
    no_branch_reason = None
    if not branches:
        no_branch_reason = (
            "NO_EXTERNAL_SEMANTIC_PREDECESSOR"
            if not candidates
            else "ALL_CANDIDATES_REJECTED_BY_LIMITS_OR_CONFLICTS"
        )
    structure = {
        "backbone_node_count": len(backbone_nodes),
        "backbone_edge_count": len(canonical_edges),
        "supporting_edge_count": len(supporting_edges),
        "branch_count": len(branches),
        "branch_candidate_count": len(candidates),
        "branch_rejection_counts": dict(sorted(rejection_counts.items())),
        "branch_semantic_type_counts": dict(
            sorted(Counter(branch.semantic_edge_type for branch in branches).items())
        ),
        "total_node_count": len(all_node_ids),
        "total_edge_count": len(all_edge_ids),
        "no_branch_reason": no_branch_reason,
    }
    return CandidateCausalChain(
        chain_id=chain_id,
        result_id=result_id,
        result_node_id=result_node_id,
        semantic_views=semantic_views,
        backbone_node_ids=backbone_nodes,
        backbone_edge_ids=tuple(edge["edge_id"] for edge in canonical_edges),
        supporting_edge_ids=tuple(edge["edge_id"] for edge in supporting_edges),
        stop_reason=stop_reason,
        branches=branches,
        all_node_ids=all_node_ids,
        all_edge_ids=all_edge_ids,
        structural_features=structure,
        uncertainties=(
            "candidate graph edges and assembled branches do not prove causality",
            "branch selection is bounded and deterministic",
        ),
    )


def _canonicalize_backbone(
    node_ids: tuple[str, ...],
    result_id: str,
    result_node_id: str,
    parallel: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    root_ids: set[str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    canonical: list[Mapping[str, Any]] = []
    supporting: list[Mapping[str, Any]] = []
    for index, (src, dst) in enumerate(zip(node_ids, node_ids[1:])):
        candidates = parallel[(src, dst)]
        is_final_hop = index == len(node_ids) - 2 and result_node_id not in root_ids
        if is_final_hop:
            result_edges = [
                edge
                for edge in candidates
                if edge["edge_type"] == "RESULT"
                and result_id in edge.get("source_refs", ())
                and edge.get("attributes", {}).get("result_id") == result_id
            ]
            if len(result_edges) != 1:
                raise ValueError(f"Missing unique final RESULT edge for {result_id}")
            selected = result_edges[0]
        else:
            candidates = [edge for edge in candidates if edge["edge_type"] != "RESULT"]
            if not candidates:
                raise ValueError(f"No non-RESULT canonical edge for backbone hop {src}->{dst}")
            selected = min(candidates, key=_canonical_edge_rank)
        canonical.append(selected)
        supporting.extend(edge for edge in candidates if edge["edge_id"] != selected["edge_id"])
        if is_final_hop:
            supporting.extend(
                edge
                for edge in parallel[(src, dst)]
                if edge["edge_id"] != selected["edge_id"] and edge not in supporting
            )
    supporting.sort(key=lambda edge: edge["edge_id"])
    return canonical, supporting


def _branch_candidates(
    backbone_nodes: tuple[str, ...],
    result_node_id: str,
    nodes: Mapping[str, Mapping[str, Any]],
    incoming: Mapping[str, list[Mapping[str, Any]]],
    config: ChainAssemblyConfig,
) -> list[_BranchCandidate]:
    main_nodes = set(backbone_nodes)
    candidates: list[_BranchCandidate] = []
    for attachment in backbone_nodes:
        for semantic in incoming.get(attachment, ()):
            if semantic["edge_type"] not in SEMANTIC_TYPES or semantic["src"] in main_nodes:
                continue
            branch_edges = _with_call_ancestors(
                semantic, main_nodes, incoming, config.max_branch_depth
            )
            node_ids = (branch_edges[0]["src"],) + tuple(
                edge["dst"] for edge in branch_edges
            )
            if len(node_ids) != len(set(node_ids)):
                continue
            if any(node_id in main_nodes for node_id in node_ids[:-1]):
                continue
            span = nodes[node_ids[-1]]["trace_index"] - nodes[node_ids[0]]["trace_index"]
            distance = (
                nodes[result_node_id]["trace_index"] - nodes[attachment]["trace_index"]
            )
            candidates.append(
                _BranchCandidate(
                    attachment_node_id=attachment,
                    node_ids=node_ids,
                    edge_ids=tuple(edge["edge_id"] for edge in branch_edges),
                    semantic_edge=semantic,
                    execution_span=span,
                    attachment_distance=distance,
                )
            )
    return candidates


def _with_call_ancestors(
    semantic: Mapping[str, Any],
    main_nodes: set[str],
    incoming: Mapping[str, list[Mapping[str, Any]]],
    max_branch_depth: int,
) -> tuple[Mapping[str, Any], ...]:
    reverse_calls: list[Mapping[str, Any]] = []
    current = semantic["src"]
    maximum_calls = min(2, max_branch_depth - 1)
    while len(reverse_calls) < maximum_calls:
        call_edges = [edge for edge in incoming.get(current, ()) if edge["edge_type"] == "CALL"]
        if len(call_edges) != 1:
            break
        call_edge = call_edges[0]
        if call_edge["src"] in main_nodes or call_edge["src"] == current:
            break
        if any(call_edge["src"] == edge["src"] for edge in reverse_calls):
            break
        reverse_calls.append(call_edge)
        current = call_edge["src"]
    return tuple(reversed(reverse_calls)) + (semantic,)


def _select_branches(
    candidates: list[_BranchCandidate],
    backbone_nodes: tuple[str, ...],
    nodes: Mapping[str, Mapping[str, Any]],
    config: ChainAssemblyConfig,
) -> tuple[list[_BranchCandidate], Counter[str]]:
    selected: list[_BranchCandidate] = []
    used_external_nodes: set[str] = set()
    used_edges: set[str] = set()
    state_object_counts: Counter[str] = Counter()
    hypothesis_count = 0
    rejections: Counter[str] = Counter()

    def select(candidate: _BranchCandidate) -> bool:
        nonlocal hypothesis_count
        allowed, reason = _branch_allowed(
            candidate,
            selected,
            backbone_nodes,
            used_external_nodes,
            used_edges,
            state_object_counts,
            hypothesis_count,
            config,
        )
        if not allowed:
            rejections[reason] += 1
            return False
        selected.append(candidate)
        used_external_nodes.update(candidate.external_node_ids)
        used_edges.update(candidate.edge_ids)
        if candidate.semantic_edge["evidence_level"] == "HYPOTHESIS":
            hypothesis_count += 1
        if candidate.semantic_edge["edge_type"] == "STATE":
            state_object_counts[_state_object_key(candidate.semantic_edge)] += 1
        return True

    selected_ids: set[str] = set()
    for semantic_type in SEMANTIC_TYPES:
        for candidate in candidates:
            if candidate.semantic_edge["edge_type"] != semantic_type:
                continue
            if select(candidate):
                selected_ids.add(candidate.semantic_edge["edge_id"])
                break
    for candidate in candidates:
        if candidate.semantic_edge["edge_id"] in selected_ids:
            continue
        if len(selected) >= config.max_branches_per_chain:
            break
        if select(candidate):
            selected_ids.add(candidate.semantic_edge["edge_id"])
    return selected, rejections


def _branch_allowed(
    candidate: _BranchCandidate,
    selected: list[_BranchCandidate],
    backbone_nodes: tuple[str, ...],
    used_external_nodes: set[str],
    used_edges: set[str],
    state_object_counts: Counter[str],
    hypothesis_count: int,
    config: ChainAssemblyConfig,
) -> tuple[bool, str]:
    if len(selected) >= config.max_branches_per_chain:
        return False, "branch_limit"
    if set(candidate.external_node_ids) & used_external_nodes:
        return False, "external_node_conflict"
    if set(candidate.edge_ids) & used_edges:
        return False, "edge_conflict"
    if len(set(backbone_nodes) | used_external_nodes | set(candidate.external_node_ids)) > config.max_total_nodes_per_chain:
        return False, "node_limit"
    if (
        candidate.semantic_edge["evidence_level"] == "HYPOTHESIS"
        and hypothesis_count >= config.max_hypothesis_branches
    ):
        return False, "hypothesis_limit"
    if candidate.semantic_edge["edge_type"] == "STATE":
        key = _state_object_key(candidate.semantic_edge)
        if state_object_counts[key] >= config.max_edges_per_state_object_per_chain:
            return False, "state_object_limit"
    return True, ""


def _canonical_edge_rank(edge: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        CANONICAL_EDGE_PRIORITY[edge["edge_type"]],
        LEVEL_PRIORITY[edge["evidence_level"]],
        edge["edge_id"],
    )


def _branch_rank(
    candidate: _BranchCandidate, nodes: Mapping[str, Mapping[str, Any]]
) -> tuple[int, int, int, int, str]:
    return (
        LEVEL_PRIORITY[candidate.semantic_edge["evidence_level"]],
        candidate.attachment_distance,
        candidate.execution_span,
        len(candidate.edge_ids),
        candidate.semantic_edge["edge_id"],
    )


def _canonical_stop_reason(group: list[Mapping[str, Any]]) -> str:
    order = {
        "ROOT_REACHED": 0,
        "NO_ELIGIBLE_PREDECESSOR": 1,
        "MAX_DEPTH_REACHED": 2,
        "EXPANSION_BUDGET_REACHED": 3,
    }
    return min((backbone["stop_reason"] for backbone in group), key=lambda value: (order[value], value))


def _state_object_key(edge: Mapping[str, Any]) -> str:
    obj = edge.get("attributes", {}).get("state_object", {})
    contract = (obj.get("contract_address") or {}).get("normalized", "")
    account = (obj.get("account_address") or {}).get("normalized", "")
    return f"{obj.get('state_object_type', '')}|{contract}|{account}"


def _all_node_ids(
    backbone_nodes: tuple[str, ...], branches: tuple[EvidenceBranch, ...]
) -> tuple[str, ...]:
    node_ids = list(backbone_nodes)
    seen = set(node_ids)
    for branch in branches:
        for node_id in branch.node_ids[:-1]:
            if node_id not in seen:
                node_ids.append(node_id)
                seen.add(node_id)
    return tuple(node_ids)


def _all_edge_ids(
    canonical_edges: list[Mapping[str, Any]],
    supporting_edges: list[Mapping[str, Any]],
    branches: tuple[EvidenceBranch, ...],
) -> tuple[str, ...]:
    edge_ids = [edge["edge_id"] for edge in canonical_edges]
    edge_ids.extend(edge["edge_id"] for edge in supporting_edges)
    edge_ids.extend(edge_id for branch in branches for edge_id in branch.edge_ids)
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Assembled chain contains duplicate edge IDs")
    return tuple(edge_ids)


def _result_coverage(
    chains: list[CandidateCausalChain], seeds: list[Mapping[str, Any]]
) -> dict[str, Any]:
    chain_counts = Counter(chain.result_id for chain in chains)
    seed_ids = [seed["result_id"] for seed in seeds]
    return {
        "result_seed_count": len(seed_ids),
        "covered_result_count": len(chain_counts),
        "missing_result_ids": [result_id for result_id in seed_ids if result_id not in chain_counts],
        "chain_counts_by_result": dict(sorted(chain_counts.items())),
    }


def _validate_backbones_document(
    document: Mapping[str, Any],
    graph_sha256: str,
    edge_by_id: Mapping[str, Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
    seeds: list[Mapping[str, Any]],
    root_ids: set[str],
) -> list[Mapping[str, Any]]:
    if document.get("schema_version") != CausalBackbonesResult.SCHEMA_VERSION:
        raise ValueError("Input schema_version must be causal_backbones_v1")
    if document.get("candidate_causal_graph_v2_sha256") != graph_sha256:
        raise ValueError("causal_backbones_v1 graph hash does not match input graph")
    backbones = list(document.get("backbones", ()))
    seed_by_id = {seed["result_id"]: seed for seed in seeds}
    ids: set[str] = set()
    for backbone in backbones:
        backbone_id = backbone.get("backbone_id")
        if not isinstance(backbone_id, str) or backbone_id in ids:
            raise ValueError("Duplicate or invalid backbone_id")
        ids.add(backbone_id)
        result_id = backbone.get("result_id")
        if result_id not in seed_by_id:
            raise ValueError(f"Unknown ResultSeed in backbone: {backbone_id}")
        if backbone.get("result_node_id") != seed_by_id[result_id]["node_id"]:
            raise ValueError(f"Backbone ResultSeed node mismatch: {backbone_id}")
        node_ids = tuple(backbone.get("node_ids", ()))
        edge_ids = tuple(backbone.get("edge_ids", ()))
        if not node_ids or len(node_ids) != len(edge_ids) + 1:
            raise ValueError(f"Invalid backbone path shape: {backbone_id}")
        if len(node_ids) != len(set(node_ids)) or any(node_id not in nodes for node_id in node_ids):
            raise ValueError(f"Invalid backbone node references: {backbone_id}")
        if any(edge_id not in edge_by_id for edge_id in edge_ids):
            raise ValueError(f"Invalid backbone edge references: {backbone_id}")
        for index, edge_id in enumerate(edge_ids):
            edge = edge_by_id[edge_id]
            if edge["src"] != node_ids[index] or edge["dst"] != node_ids[index + 1]:
                raise ValueError(f"Broken backbone path: {backbone_id}")
        if result_id in seed_by_id and node_ids[-1] != seed_by_id[result_id]["node_id"]:
            raise ValueError(f"Backbone does not end at its ResultSeed: {backbone_id}")
        if node_ids[-1] not in root_ids:
            if not edge_ids or edge_by_id[edge_ids[-1]]["edge_type"] != "RESULT":
                raise ValueError(f"Backbone lacks final RESULT edge: {backbone_id}")
        if any(edge_by_id[edge_id]["edge_type"] == "RESULT" for edge_id in edge_ids[:-1]):
            raise ValueError(f"Backbone contains intermediate RESULT edge: {backbone_id}")
        if any(view not in VIEWS for view in backbone.get("semantic_views", ())):
            raise ValueError(f"Invalid semantic view: {backbone_id}")
    return backbones


def _validate_config(config: ChainAssemblyConfig) -> None:
    if min(
        config.max_branches_per_chain,
        config.max_branch_depth,
        config.max_total_nodes_per_chain,
        config.max_branch_candidates_per_chain,
    ) < 1:
        raise ValueError("Branch and node limits must be positive")
    if min(
        config.max_hypothesis_branches,
        config.max_edges_per_state_object_per_chain,
    ) < 0:
        raise ValueError("Evidence limits must be non-negative")


def _validate_output(
    chains: list[CandidateCausalChain],
    edge_by_id: Mapping[str, Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
    seeds: list[Mapping[str, Any]],
    root_ids: set[str],
) -> None:
    seed_by_id = {seed["result_id"]: seed for seed in seeds}
    chain_ids: set[str] = set()
    for chain in chains:
        if chain.chain_id in chain_ids:
            raise ValueError("Duplicate chain_id")
        chain_ids.add(chain.chain_id)
        if chain.result_id not in seed_by_id or chain.result_node_id != seed_by_id[chain.result_id]["node_id"]:
            raise ValueError(f"Invalid ResultSeed binding: {chain.chain_id}")
        backbone_nodes = chain.backbone_node_ids
        backbone_edges = chain.backbone_edge_ids
        if len(backbone_nodes) != len(backbone_edges) + 1:
            raise ValueError(f"Invalid backbone shape: {chain.chain_id}")
        for index, edge_id in enumerate(backbone_edges):
            edge = edge_by_id[edge_id]
            if edge["src"] != backbone_nodes[index] or edge["dst"] != backbone_nodes[index + 1]:
                raise ValueError(f"Broken canonical backbone: {chain.chain_id}")
            if nodes[edge["src"]]["trace_index"] >= nodes[edge["dst"]]["trace_index"]:
                raise ValueError(f"Reverse-time canonical backbone edge: {chain.chain_id}")
        if chain.result_node_id not in root_ids:
            if not backbone_edges or edge_by_id[backbone_edges[-1]]["edge_type"] != "RESULT":
                raise ValueError(f"Missing final RESULT edge: {chain.chain_id}")
        if any(edge_by_id[edge_id]["edge_type"] == "RESULT" for edge_id in backbone_edges[:-1]):
            raise ValueError(f"Intermediate RESULT edge: {chain.chain_id}")
        main_set = set(backbone_nodes)
        if any(
            edge_by_id[edge_id]["src"] not in main_set or edge_by_id[edge_id]["dst"] not in main_set
            for edge_id in chain.supporting_edge_ids
        ):
            raise ValueError(f"Supporting edge leaves backbone: {chain.chain_id}")
        external_nodes: set[str] = set()
        branch_edges: set[str] = set()
        for branch in chain.branches:
            if branch.attachment_node_id not in main_set:
                raise ValueError(f"Invalid branch attachment: {chain.chain_id}")
            if not branch.edge_ids or branch.node_ids[-1] != branch.attachment_node_id:
                raise ValueError(f"Invalid branch shape: {chain.chain_id}")
            if len(branch.node_ids) != len(set(branch.node_ids)):
                raise ValueError(f"Repeated branch node: {chain.chain_id}")
            if set(branch.node_ids[:-1]) & main_set or set(branch.node_ids[:-1]) & external_nodes:
                raise ValueError(f"Branch external node overlap: {chain.chain_id}")
            if set(branch.edge_ids) & branch_edges:
                raise ValueError(f"Branch edge overlap: {chain.chain_id}")
            for index, edge_id in enumerate(branch.edge_ids):
                edge = edge_by_id[edge_id]
                if edge["src"] != branch.node_ids[index] or edge["dst"] != branch.node_ids[index + 1]:
                    raise ValueError(f"Broken branch path: {chain.chain_id}")
                if edge["edge_type"] == "RESULT":
                    raise ValueError(f"RESULT edge in branch: {chain.chain_id}")
            semantic = edge_by_id[branch.semantic_edge_id]
            if semantic["edge_type"] not in SEMANTIC_TYPES or semantic["edge_id"] not in branch.edge_ids:
                raise ValueError(f"Invalid branch semantic edge: {chain.chain_id}")
            if (
                branch.edge_ids[-1] != branch.semantic_edge_id
                or semantic["src"] != branch.node_ids[-2]
                or semantic["dst"] != branch.attachment_node_id
            ):
                raise ValueError(f"Branch semantic edge does not attach exactly once: {chain.chain_id}")
            external_nodes.update(branch.node_ids[:-1])
            branch_edges.update(branch.edge_ids)
        expected_nodes = set(backbone_nodes) | external_nodes
        expected_edges = set(backbone_edges) | set(chain.supporting_edge_ids) | branch_edges
        if set(chain.all_node_ids) != expected_nodes or len(chain.all_node_ids) != len(expected_nodes):
            raise ValueError(f"all_node_ids mismatch: {chain.chain_id}")
        if set(chain.all_edge_ids) != expected_edges or len(chain.all_edge_ids) != len(expected_edges):
            raise ValueError(f"all_edge_ids mismatch: {chain.chain_id}")


def load_json_document(path: str | Path, expected_schema: str) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Input does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {error.msg}") from error
    if not isinstance(document, dict) or document.get("schema_version") != expected_schema:
        raise ValueError(f"Input schema_version must be {expected_schema}")
    return document


def serialize_candidate_causal_chains(result: CandidateCausalChainsResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def write_candidate_causal_chains(result: CandidateCausalChainsResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    )
    try:
        with handle:
            handle.write(serialize_candidate_causal_chains(result))
        os.replace(handle.name, target)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target

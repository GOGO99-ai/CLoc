"""Offline replay for CALL-multiplicity to unique-internal-callee support."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_v2.asset_continuity import build_asset_continuity
from src.cloc_v2.augmented_causal_graph import build_augmented_candidate_causal_graph
from src.cloc_v2.candidate_causal_graph import build_candidate_causal_graph
from src.cloc_v2.causal_backbones import build_causal_backbones
from src.cloc_v2.causal_chain_assembly import build_candidate_causal_chains
from src.cloc_v2.formal_normal_baseline_loader import _parse_record
from src.cloc_v2.function_identity import OfflineFunctionCandidate, resolve_function_identity
from src.cloc_v2.normal_distance import CallStep, ChainFeatureRecord, MainAnchorKey, abnormal_observed_call_identities
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.parameter_evidence import build_parameter_evidence
from src.cloc_v2.phase2_call_parent_support import map_abnormal_calls_to_unique_parent_keys
from src.cloc_v2.phase2_localization_bridge import prepare_phase2_localization
from src.cloc_v2.phase2_source_graph_fallback import build_phase2_source_graph_fallback
from src.cloc_v2.result_detection import detect_result_candidates
from src.cloc_v2.runtime_semantics import build_runtime_annotations
from src.cloc_v2.source_cache import atomic_json
from src.cloc_v2.source_index import build_source_index
from src.cloc_v2.state_evidence import build_state_evidence
from src.cloc_v2.strict_result_chains import build_strict_result_chains
from src.cloc_v2.transaction_graph import build_transaction_graph
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index, lookup_verified_source_cache


DEFAULT_MANIFEST = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "locked_inference_manifest.json"
DEFAULT_NORMAL = ROOT / "data" / "prototypes" / "formal_normal_baseline_full"
DEFAULT_CACHE = ROOT / "data" / "source_cache"
DEFAULT_TREES = ROOT / "data" / "cache" / "all_tree"
DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "week10_full_candidate_chain_replay_20260910" / "report.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_normal_records(root: Path) -> tuple[ChainFeatureRecord, ...]:
    checkpoint = _load(root / "checkpoint.json")
    records = []
    for sample_key, state in checkpoint["samples"].items():
        if not sample_key.startswith("eth:") or state.get("status") != "SUCCEEDED":
            continue
        artifact = _load(root / state["artifact"])
        records.extend(_parse_record(value) for value in artifact["feature_records"])
    return tuple(records)


def _offline_identity_evidence(graph: Any, cache_index: Any, cache_root: Path):
    requested = set()
    effective = {}
    addresses = sorted({node.callee_address.lower() for node in graph.nodes if isinstance(node.callee_address, str)})
    for address in addresses:
        lookup = lookup_verified_source_cache(cache_index, runtime_address=address, requested_address=address)
        if lookup.binding is not None:
            effective[address] = lookup.binding.effective_address
            requested.add(lookup.binding.requested_address)
    acquisition = {
        "schema_version": "source_acquisition_v1",
        "chain": "eth",
        "addresses": [{"requested_address": value} for value in sorted(requested)],
    }
    source_index = build_source_index(acquisition, cache_root)
    candidates = tuple(
        OfflineFunctionCandidate(
            1,
            function["address"],
            function["function_name"],
            function["canonical_signature"],
            function["selector"],
            "VERIFIED_SOURCE_CACHE",
        )
        for address in source_index["addresses"]
        for function in address["functions"]
        if function.get("function_kind") == "FUNCTION"
        and function.get("canonical_signature")
        and function.get("selector") != "UNKNOWN"
    )
    return candidates, effective


def _candidate_chain_query_records(graph: Any, strict: Any, chains: dict[str, Any], candidates: tuple[Any, ...], effective: dict[str, str]):
    anchors: dict[str, set[MainAnchorKey]] = defaultdict(set)
    for chain in strict.chains:
        if chain.anchor is not None:
            anchors[chain.result_id].add(
                MainAnchorKey(chain.anchor.blockchain_chain_id, chain.anchor.effective_address, chain.anchor.selector)
            )
    nodes = {node.node_id: node for node in graph.nodes}
    records = []
    for chain in chains.get("chains", ()):
        steps = []
        unresolved = False
        for node_id in chain.get("all_node_ids", ()):
            node = nodes[node_id]
            runtime = node.callee_address.lower() if isinstance(node.callee_address, str) else None
            identity = resolve_function_identity(
                blockchain_chain_id=1,
                runtime_address=runtime,
                effective_address=effective.get(runtime) if runtime else None,
                trace_selector=node.function_selector,
                function_name=node.function_name,
                offline_candidates=candidates,
            )
            if identity.usable_for_anchor and identity.effective_address and identity.selector:
                steps.append(CallStep(identity.effective_address, identity.selector))
            else:
                unresolved = True
        for position, anchor in enumerate(sorted(anchors.get(chain.get("result_id"), ()), key=lambda item: (item.effective_address, item.selector)), start=1):
            records.append(
                ChainFeatureRecord(
                    f"{chain.get('chain_id')}:{position}",
                    anchor,
                    tuple(steps),
                    frozenset(),
                    ("PARTIAL_CALL_IDENTITY",) if unresolved else (),
                    result_scoring_context=getattr(strict, "result_scoring_context", "legacy_independent_weights"),
                )
            )
    return tuple(records)


def _prompt_keys(preparation: Any) -> set[tuple[str, str, int, int]]:
    return {
        (row["relative_file"], row["function_signature"], row["start_line"], row["end_line"])
        for row in preparation.pre_llm_candidates
        if row["included_in_prompt"]
    }


def _run_sample(sample: dict[str, Any], tree_path: Path, normal_records: tuple[ChainFeatureRecord, ...], cache_index: Any, cache_root: Path):
    graph = build_transaction_graph(_load(tree_path))
    annotations = build_runtime_annotations(graph)
    ledger = build_observed_asset_ledger(graph, annotations)
    continuity = build_asset_continuity(graph, annotations, ledger)
    results = detect_result_candidates(graph, annotations, ledger, continuity.to_dict())
    candidates, effective = _offline_identity_evidence(graph, cache_index, cache_root)
    strict = build_strict_result_chains(
        graph,
        continuity.to_dict(),
        results.to_dict(),
        blockchain_chain_id=1,
        offline_function_candidates=candidates,
        offline_effective_address_map=effective,
    )
    causal = build_candidate_causal_graph(graph, annotations, ledger, continuity.to_dict(), results.to_dict()).to_dict()
    parameter = build_parameter_evidence(graph, annotations, ledger, causal).to_dict()
    state = build_state_evidence(graph, annotations, ledger, causal).to_dict()
    augmented = build_augmented_candidate_causal_graph(causal, parameter, state).to_dict()
    backbones = build_causal_backbones(augmented).to_dict()
    chain_document = build_candidate_causal_chains(augmented, backbones).to_dict()
    queries = _candidate_chain_query_records(graph, strict, chain_document, candidates, effective)
    query_evidence_mode = "REBUILT_CANDIDATE_CAUSAL_CHAINS"
    query_chain_count = len(chain_document.get("chains", ()))
    abnormal_calls = abnormal_observed_call_identities(queries, normal_records)

    empty_slices = {"schema_version": "source_request_slices_v1", "chain": "eth", "requests": [], "metadata": {}}
    slices = build_phase2_source_graph_fallback(
        graph,
        strict,
        empty_slices,
        cache_index,
        cache_root,
        max_candidates=50,
    )
    parent_keys = map_abnormal_calls_to_unique_parent_keys(slices, abnormal_calls)
    common = dict(max_pre_llm_candidates=50, max_prompt_candidates=20, prompt_contract="topk_v4")
    baseline = prepare_phase2_localization(strict, causal, graph, slices, **common)
    enhanced = prepare_phase2_localization(
        strict,
        causal,
        graph,
        slices,
        abnormal_parent_function_keys=parent_keys,
        **common,
    )
    protected = [
        {
            "candidate_alias": row["candidate_alias"],
            "contract_alias": row["contract_alias"],
            "relative_file": row["relative_file"],
            "function_signature": row["function_signature"],
            "selection_stage": row["selection_stage"],
        }
        for row in enhanced.pre_llm_candidates
        if row.get("abnormal_parent_support")
    ]
    return {
        "sample_alias": sample["sample_alias"],
        "graph_node_count": len(graph.nodes),
        "raw_param_edge_count": len(parameter["parameter_edges"]),
        "raw_state_edge_count": len(state["state_edges"]),
        "query_evidence_mode": query_evidence_mode,
        "query_chain_count": query_chain_count,
        "query_record_count": len(queries),
        "abnormal_call_identity_count": len(abnormal_calls),
        "abnormal_parent_key_count": len(parent_keys),
        "protected_candidates": protected,
        "baseline_status": baseline.preparation_status,
        "enhanced_status": enhanced.preparation_status,
        "prompt_selection_changed": _prompt_keys(baseline) != _prompt_keys(enhanced),
        "unchanged_without_parent_support": not parent_keys and _prompt_keys(baseline) == _prompt_keys(enhanced),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--normal-root", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--tree-root", type=Path, default=DEFAULT_TREES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest = _load(args.manifest)
    trees = {path.stem.lower(): path for path in args.tree_root.rglob("*.json")}
    selected = [(sample, trees[sample["tx_hash"].lower()]) for sample in manifest["samples"] if sample["tx_hash"].lower() in trees]
    normal_records = _load_normal_records(args.normal_root)
    cache_index = build_verified_source_cache_index(args.cache_root, blockchain_chain_id=1, chain_key="eth")
    rows = [_run_sample(sample, path, normal_records, cache_index, args.cache_root) for sample, path in selected]
    report = {
        "schema_version": "week10_full_candidate_chain_replay_v1",
        "mode": "OFFLINE_NO_LLM_NO_NETWORK",
        "available_sample_count": len(rows),
        "normal_record_count": len(normal_records),
        "samples": rows,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Connect method components without importing experiment scripts or labels."""
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from cloc.config import CHAIN_IDS, load_chain_config
from cloc.evidence.graph import build_transaction_graph
from cloc.evidence.semantics import build_runtime_annotations
from cloc.evidence.ledger import build_observed_asset_ledger
from cloc.evidence.continuity import build_asset_continuity
from cloc.evidence.outcomes import detect_result_candidates
from cloc.evidence.causal_graph import build_candidate_causal_graph
from cloc.evidence.chains import build_strict_result_chains
from cloc.reference.features import extract_normal_features
from cloc.reference.storage import load_reference_library
from cloc.reference.observations import extract_amount_observations
from cloc.review.preparation import prepare_reference_review
from cloc.localization.preparation import (
    MAX_PROMPT_BYTES_RANKING,
    prepare_candidate_ranking,
)
from cloc.localization.candidates import (
    MAX_SOURCE_FUNCTION_CANDIDATES_RANKING,
    build_ranking_source_graph_fallback,
)
from cloc.localization.prompt import MAX_SOURCE_SLICES_RANKING, set_ranking_limit
from cloc.source.fetcher import SourceFetcher
from cloc.source.index import build_source_index
from cloc.source.slices import build_source_request_slices
from cloc.source.windows import build_verified_source_window_fallback
from cloc.source.verified_cache import (
    build_verified_source_cache_index,
    lookup_verified_source_cache,
)
from cloc.runtime.policy import ExecutionPolicy
from cloc.trace.calltracer import adapt_calltracer_call_tree


@dataclass(frozen=True)
class EvidenceBundle:
    """Original execution artifacts shared by review and localization."""
    chain: str
    graph: Any
    annotations: Any
    ledger: Any
    continuity: Any
    outcomes: Any
    chains: Any
    causal_graph: Any
    raw_trace: Mapping[str, Any] = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "chain": self.chain,
            "transaction_graph": self.graph.to_dict(),
            "runtime_annotations": self.annotations.to_dict(),
            "asset_ledger": self.ledger.to_dict(),
            "asset_continuity": self.continuity.to_dict(),
            "result_candidates": self.outcomes.to_dict(),
            "result_chains": self.chains.to_dict(),
            "causal_graph": self.causal_graph.to_dict(),
        }


def build_evidence(raw_trace: Mapping[str, Any], chain: str) -> EvidenceBundle:
    """Construct Stage I evidence from a callTracer frame or RPC result."""
    if chain not in CHAIN_IDS:
        raise ValueError("chain must be eth or bsc")
    if not isinstance(raw_trace, Mapping):
        raise ValueError("expected a successful callTracer frame")
    if "jsonrpc" in raw_trace and "error" in raw_trace:
        raise ValueError("expected a successful callTracer frame")
    frame = raw_trace.get("result", raw_trace)
    tree = adapt_calltracer_call_tree(frame)
    graph = build_transaction_graph(tree)
    annotations = build_runtime_annotations(graph)
    ledger = build_observed_asset_ledger(graph, annotations)
    continuity = build_asset_continuity(graph, annotations, ledger)
    outcomes = detect_result_candidates(graph, annotations, ledger, continuity.to_dict())
    chains = build_strict_result_chains(
        graph, continuity.to_dict(), outcomes.to_dict(),
        blockchain_chain_id=CHAIN_IDS[chain],
        offline_function_candidates=(), offline_effective_address_map={},
    )
    causal = build_candidate_causal_graph(
        graph, annotations, ledger, continuity.to_dict(), outcomes.to_dict()
    )
    return EvidenceBundle(chain, graph, annotations, ledger, continuity, outcomes, chains, causal, raw_trace)


def prepare_review(bundle: EvidenceBundle, baseline_dir: Path, record_id: str):
    """Prepare Stage II; observed amounts never enter the retrieval distance."""
    loaded = load_reference_library(baseline_dir, query_chain=bundle.chain, query_trace=bundle.raw_trace)
    features = extract_normal_features(
        bundle.graph, bundle.chains, bundle.ledger, transaction_record_id="query::" + record_id
    )
    if any(record.record_id in loaded.normal_record_transaction_ids for record in features.records):
        raise ValueError("QUERY_REFERENCE_ID_COLLISION")
    amounts = extract_amount_observations(bundle.graph, bundle.chains, bundle.ledger, features)
    return prepare_reference_review(
        bundle.chains, features, loaded.baseline, loaded.normal_record_transaction_ids,
        amount_observations={**loaded.amount_observations, **amounts},
    )


def prepare_localization(
    bundle: EvidenceBundle,
    cache_root: Path,
    *,
    fetch_sources: bool = False,
    max_source_addresses: int = 3,
    top_k: int = 5,
):
    """Prepare Stage III from original evidence and verified source bindings."""
    if max_source_addresses < 1:
        raise ValueError("max_source_addresses must be positive")
    graph = bundle.graph
    nodes = {node.node_id: node for node in graph.nodes}
    anchors = [chain.anchor.node_id for chain in bundle.chains.chains if chain.anchor]
    chain_nodes = [node for chain in bundle.chains.chains for node in chain.node_ids]
    ordered_nodes = tuple(dict.fromkeys(anchors + chain_nodes))
    addresses = list(dict.fromkeys(
        nodes[node_id].callee_address.lower()
        for node_id in ordered_nodes
        if node_id in nodes and nodes[node_id].callee_address
    ))[:max_source_addresses]
    cache_index = build_verified_source_cache_index(
        cache_root, blockchain_chain_id=CHAIN_IDS[bundle.chain], chain_key=bundle.chain
    )
    fetcher = None
    if fetch_sources:
        fetcher = SourceFetcher(
            bundle.chain, load_chain_config(bundle.chain), cache_root=cache_root,
            policy=ExecutionPolicy(allow_explorer=True), max_retries=1, max_proxy_depth=3,
        )
    acquired = []
    statuses = []
    for address in addresses:
        lookup = lookup_verified_source_cache(
            cache_index, runtime_address=address, requested_address=address
        )
        if lookup.status == "VERIFIED_SOURCE_CACHE_HIT" and lookup.binding:
            acquired.append({"requested_address": lookup.binding.artifact_address})
            statuses.append(lookup.status)
        elif lookup.status in {"VERIFIED_SOURCE_CACHE_CONFLICT", "VERIFIED_SOURCE_CACHE_INVALID"}:
            raise ValueError(lookup.status)
        elif fetcher is not None:
            result = fetcher.acquire(address, refresh=False, origins=())
            acquired.append(result.to_dict())
            statuses.append(result.status)
        else:
            statuses.append(lookup.status)
    acquisition = {"schema_version": "source_acquisition_v1", "chain": bundle.chain, "addresses": acquired}
    index = build_source_index(acquisition, cache_root)
    requests = [
        {"chain_id": "Q", "node_id": node, "focus": "ranking_localization", "priority": 1 if i == 0 else 2}
        for i, node in enumerate(ordered_nodes[:12])
    ]
    slices = build_source_request_slices(index, graph.to_dict(), cache_root, requests)
    slices = build_verified_source_window_fallback(index, graph.to_dict(), slices, cache_root)
    # Refresh after acquisition so newly fetched implementations are visible.
    if fetch_sources:
        cache_index = build_verified_source_cache_index(
            cache_root, blockchain_chain_id=CHAIN_IDS[bundle.chain], chain_key=bundle.chain
        )
    slices = build_ranking_source_graph_fallback(
        graph, bundle.chains, slices, cache_index, cache_root,
        max_candidates=MAX_SOURCE_FUNCTION_CANDIDATES_RANKING,
    )
    preparation = prepare_candidate_ranking(
        bundle.chains, bundle.causal_graph, graph, slices,
        max_pre_llm_candidates=MAX_SOURCE_FUNCTION_CANDIDATES_RANKING,
        max_prompt_candidates=MAX_SOURCE_SLICES_RANKING,
        max_prompt_bytes=MAX_PROMPT_BYTES_RANKING, prompt_contract="ranking",
    )
    if preparation.prompt is not None:
        prompt = set_ranking_limit(preparation.prompt, top_k)
        encoded = json.dumps(prompt.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        preparation = replace(preparation, prompt=prompt, prompt_sha256=hashlib.sha256(encoded).hexdigest(), prompt_bytes=len(encoded))
    return preparation, {"source_statuses": statuses, "source_address_count": len(addresses)}


def analyze_transaction(raw_trace, chain, baseline_path, cache_root, client, *, record_id="query",
                        fetch_sources=True, max_source_addresses=3, top_k=5, review_tokens=4096,
                        ranking_tokens=8192):
    """Route Stage II first, then rank from the untouched Stage I evidence."""
    from cloc.review.client import execute_review
    from cloc.localization.client import execute_ranking
    from cloc.reporting.report import build_report

    bundle = build_evidence(raw_trace, chain)
    review_preparation = prepare_review(bundle, baseline_path, record_id)
    review = execute_review(review_preparation, client, output_tokens=review_tokens)
    artifacts = {"evidence": bundle.to_dict(), "review_preparation": review_preparation.to_dict(), "review": review}
    if review["route"] == "EARLY_NORMAL":
        return build_report(chain, review), artifacts
    ranking_preparation, acquisition = prepare_localization(bundle, cache_root,
        fetch_sources=fetch_sources, max_source_addresses=max_source_addresses, top_k=top_k)
    ranking = execute_ranking(ranking_preparation, client, output_tokens=ranking_tokens)
    artifacts.update(ranking_preparation=ranking_preparation.to_dict(), acquisition=acquisition, ranking=ranking)
    return build_report(chain, review, ranking), artifacts


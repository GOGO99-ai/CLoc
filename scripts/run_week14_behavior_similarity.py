"""Offline validation of preregistered source-behavior similarity."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_week9_call_parent_replay as replay
from src.cloc_v2.source_cache import atomic_json
from src.cloc_v2.source_index import _mask_comments_and_strings
from src.cloc_v2.transaction_graph import load_call_tree


DEFAULT_TRUTH = ROOT / "data" / "prototypes" / "localization15_topk_v4_20260901" / "evaluation_ground_truth.json"
DEFAULT_OUTPUT = ROOT / "data" / "prototypes" / "week14_behavior_similarity_20260910" / "report.json"

TOKEN_WEIGHTS = {"member": 3, "call": 2, "id": 1}
DIRECT_IDENTITY_STATUSES = frozenset({"SELECTOR_EXACT_MATCH", "UNIQUE_SIGNATURE_MATCH", "UNIQUE_NAME_MATCH"})
PROTECTED_STAGES = frozenset({"PROTECTED_EVIDENCE", "SEQUENCE_EXACT_SUPPORT", "ABNORMAL_PARENT_UNIQUE_CALLEE"})
STOP_WORDS = frozenset(
    "address anonymous as assembly bool break bytes calldata case catch constant constructor continue contract "
    "default delete do else emit enum error event external fallback false fixed for function if immutable import "
    "indexed int interface internal is library mapping memory modifier new override payable pragma private public "
    "pure receive return returns revert selfdestruct solidity storage string struct super switch true try type "
    "ufixed uint unchecked using var view virtual while wei gwei ether years weeks days hours minutes seconds "
    "this require assert block msg tx abi now".split()
)
IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
MEMBER_CALL = re.compile(r"\.\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
BARE_CALL = re.compile(r"(?<![.$A-Za-z0-9_])([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
LINE_PREFIX = re.compile(r"(?m)^\s*\d+:\s?")
BUILTIN_TYPE = re.compile(r"(?:u?int(?:[0-9]{0,3})?|bytes(?:[0-9]{1,2})?)$")


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("relative_file", row.get("file_path", ""))).replace("\\", "/"),
        str(row.get("function_signature", row.get("signature") or row.get("function_name") or "")),
        int(row.get("start_line", 0)),
        int(row.get("end_line", 0)),
    )


def _ignored_identifier(value: str) -> bool:
    lowered = value.lower()
    return len(lowered) < 3 or lowered in STOP_WORDS or BUILTIN_TYPE.fullmatch(lowered) is not None


def _behavior_tokens(source: str) -> frozenset[str]:
    text = LINE_PREFIX.sub("", source)
    masked = _mask_comments_and_strings(text)
    opening, closing = masked.find("{"), masked.rfind("}")
    if opening < 0 or closing <= opening:
        return frozenset()
    body = masked[opening + 1 : closing]
    member_names = {match.group(1).lower() for match in MEMBER_CALL.finditer(body) if not _ignored_identifier(match.group(1))}
    call_names = {
        match.group(1).lower()
        for match in BARE_CALL.finditer(body)
        if not _ignored_identifier(match.group(1))
    }
    identifiers = {
        match.group(0).lower()
        for match in IDENTIFIER.finditer(body)
        if not _ignored_identifier(match.group(0))
    }
    identifier_only = identifiers - member_names - call_names
    return frozenset(
        {f"member:{value}" for value in member_names}
        | {f"call:{value}" for value in call_names}
        | {f"id:{value}" for value in identifier_only}
    )


def _score(candidate: frozenset[str], reference: frozenset[str], idf: Mapping[str, float]) -> float:
    def term(token: str) -> float:
        kind = token.split(":", 1)[0]
        return TOKEN_WEIGHTS[kind] * idf[token] ** 2

    numerator = sum(term(token) for token in candidate & reference)
    candidate_norm = sum(term(token) for token in candidate)
    reference_norm = sum(term(token) for token in reference)
    if not numerator or not candidate_norm or not reference_norm:
        return 0.0
    return numerator / math.sqrt(candidate_norm * reference_norm)


def _truth_match(row: Mapping[str, Any], truths: list[dict[str, Any]]) -> bool:
    path, signature, start, end = _candidate_key(row)
    return any(
        path == str(truth.get("relative_file", "")).replace("\\", "/")
        and signature == truth.get("function_signature")
        and start <= int(truth.get("truth_end_line", -1))
        and end >= int(truth.get("truth_start_line", 1 << 30))
        for truth in truths
    )


def _source_by_key(slices: Mapping[str, Any]) -> dict[tuple[str, str, int, int], str]:
    result: dict[tuple[str, str, int, int], str] = {}
    for request in slices.get("requests", ()):
        if not isinstance(request, Mapping) or not isinstance(request.get("source"), str):
            continue
        key = _candidate_key(request)
        current = result.get(key)
        if current is None or request["source"] < current:
            result[key] = request["source"]
    return result


def _build_sample(
    sample: dict[str, Any],
    tree_path: Path,
    normal_records: tuple[Any, ...],
    cache_index: Any,
    cache_root: Path,
    *,
    max_prompt_candidates: int = 20,
    max_prompt_bytes: int | None = None,
):
    graph = replay.build_transaction_graph(load_call_tree(tree_path))
    annotations = replay.build_runtime_annotations(graph)
    ledger = replay.build_observed_asset_ledger(graph, annotations)
    continuity = replay.build_asset_continuity(graph, annotations, ledger)
    results = replay.detect_result_candidates(graph, annotations, ledger, continuity.to_dict())
    identities, effective = replay._offline_identity_evidence(graph, cache_index, cache_root)
    strict = replay.build_strict_result_chains(
        graph,
        continuity.to_dict(),
        results.to_dict(),
        blockchain_chain_id=1,
        offline_function_candidates=identities,
        offline_effective_address_map=effective,
    )
    causal = replay.build_candidate_causal_graph(graph, annotations, ledger, continuity.to_dict(), results.to_dict()).to_dict()
    parameter = replay.build_parameter_evidence(graph, annotations, ledger, causal).to_dict()
    state = replay.build_state_evidence(graph, annotations, ledger, causal).to_dict()
    augmented = replay.build_augmented_candidate_causal_graph(causal, parameter, state).to_dict()
    backbones = replay.build_causal_backbones(augmented).to_dict()
    chains = replay.build_candidate_causal_chains(augmented, backbones).to_dict()
    queries = replay._candidate_chain_query_records(graph, strict, chains, identities, effective)
    abnormal_calls = replay.abnormal_observed_call_identities(queries, normal_records)
    empty_slices = {"schema_version": "source_request_slices_v1", "chain": "eth", "requests": [], "metadata": {}}
    slices = replay.build_phase2_source_graph_fallback(graph, strict, empty_slices, cache_index, cache_root, max_candidates=50)
    parent_keys = replay.map_abnormal_calls_to_unique_parent_keys(slices, abnormal_calls)
    preparation = replay.prepare_phase2_localization(
        strict,
        causal,
        graph,
        slices,
        abnormal_parent_function_keys=parent_keys,
        max_pre_llm_candidates=50,
        max_prompt_candidates=max_prompt_candidates,
        max_prompt_bytes=max_prompt_bytes,
        prompt_contract="topk_v4",
    )
    return slices, preparation


def _evaluate_sample(sample: dict[str, Any], truths: list[dict[str, Any]], tree_path: Path, normal_records: tuple[Any, ...], cache_index: Any, cache_root: Path) -> dict[str, Any]:
    slices, preparation = _build_sample(sample, tree_path, normal_records, cache_index, cache_root)
    rows = preparation.pre_llm_candidates
    sources = _source_by_key(slices)
    tokens = {_candidate_key(row): _behavior_tokens(sources.get(_candidate_key(row), "")) for row in rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("contract_alias", ""))].append(row)

    scored: list[tuple[float, int, dict[str, Any], list[str]]] = []
    for contract_rows in grouped.values():
        documents = [tokens[_candidate_key(row)] for row in contract_rows if tokens[_candidate_key(row)]]
        if not documents:
            continue
        counts = Counter(token for document in documents for token in document)
        idf = {token: 1.0 + math.log((len(documents) + 1) / (count + 1)) for token, count in counts.items()}
        reference = frozenset(
            token
            for row in contract_rows
            if row.get("selector_match_status") in DIRECT_IDENTITY_STATUSES
            for token in tokens[_candidate_key(row)]
        )
        if not reference:
            continue
        for position, row in enumerate(rows, start=1):
            if row not in contract_rows or row.get("selector_match_status") != "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK":
                continue
            candidate_tokens = tokens[_candidate_key(row)]
            value = _score(candidate_tokens, reference, idf)
            shared = sorted(candidate_tokens & reference, key=lambda token: (-TOKEN_WEIGHTS[token.split(":", 1)[0]] * idf[token], token))
            scored.append((value, position, row, shared[:10]))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    winner = ranked[0] if ranked and ranked[0][0] > 0 else None
    selected = sorted(
        (row for row in rows if row.get("included_in_prompt")),
        key=lambda row: int(str(row.get("source_slice_id", "S999"))[1:]),
    )
    selected_keys = {_candidate_key(row) for row in selected}
    simulated = list(selected)
    if winner is not None and _candidate_key(winner[2]) not in selected_keys:
        protected = [row for row in selected if row.get("selection_stage") in PROTECTED_STAGES]
        remaining = [row for row in selected if row.get("selection_stage") not in PROTECTED_STAGES]
        simulated = (protected + [winner[2]] + remaining)[:20]

    truth_rows = [row for row in rows if _truth_match(row, truths)]
    truth_keys = {_candidate_key(row) for row in truth_rows}
    current_hit = bool(truth_keys & {_candidate_key(row) for row in selected})
    simulated_hit = bool(truth_keys & {_candidate_key(row) for row in simulated})
    truth_behavior_ranks = [index for index, item in enumerate(ranked, start=1) if _candidate_key(item[2]) in truth_keys]
    truth_positions = [index for index, row in enumerate(rows, start=1) if _candidate_key(row) in truth_keys]
    return {
        "sample_alias": sample["sample_alias"],
        "preparation_status": preparation.preparation_status,
        "candidate_count": len(rows),
        "truth_candidate_positions": truth_positions,
        "current_prompt_truth_hit": current_hit,
        "simulated_prompt_truth_hit": simulated_hit,
        "truth_behavior_ranks": truth_behavior_ranks,
        "eligible_behavior_candidate_count": len(ranked),
        "behavior_winner": None if winner is None else {
            "candidate_alias": winner[2].get("candidate_alias"),
            "relative_file": winner[2].get("relative_file"),
            "function_signature": winner[2].get("function_signature"),
            "score": round(winner[0], 8),
            "is_truth": _candidate_key(winner[2]) in truth_keys,
            "already_in_prompt": _candidate_key(winner[2]) in selected_keys,
            "shared_tokens": winner[3],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=replay.DEFAULT_MANIFEST)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--normal-root", type=Path, default=replay.DEFAULT_NORMAL)
    parser.add_argument("--cache-root", type=Path, default=replay.DEFAULT_CACHE)
    parser.add_argument("--tree-root", type=Path, default=replay.DEFAULT_TREES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    manifest = replay._load(args.manifest)
    truth_document = replay._load(args.truth)
    truth_by_sample = {row["sample_alias"]: row["truths"] for row in truth_document["labels"]}
    trees = {path.stem.lower(): path for path in args.tree_root.rglob("*.json")}
    selected = [(sample, trees[sample["tx_hash"].lower()]) for sample in manifest["samples"] if sample["tx_hash"].lower() in trees]
    normal_records = replay._load_normal_records(args.normal_root)
    cache_index = replay.build_verified_source_cache_index(args.cache_root, blockchain_chain_id=1, chain_key="eth")
    rows = [
        _evaluate_sample(sample, truth_by_sample[sample["sample_alias"]], tree, normal_records, cache_index, args.cache_root)
        for sample, tree in selected
    ]
    improved = [row["sample_alias"] for row in rows if not row["current_prompt_truth_hit"] and row["simulated_prompt_truth_hit"]]
    regressed = [row["sample_alias"] for row in rows if row["current_prompt_truth_hit"] and not row["simulated_prompt_truth_hit"]]
    report = {
        "schema_version": "week14_behavior_similarity_offline_v1",
        "mode": "OFFLINE_NO_LLM_NO_NETWORK",
        "preregistration": "docs/week14_behavior_similarity_preregistration.md",
        "available_sample_count": len(rows),
        "current_prompt_truth_hits": sum(row["current_prompt_truth_hit"] for row in rows),
        "simulated_prompt_truth_hits": sum(row["simulated_prompt_truth_hit"] for row in rows),
        "improved_samples": improved,
        "regressed_samples": regressed,
        "mechanical_acceptance_gate_passed": bool(improved) and not regressed,
        "samples": rows,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

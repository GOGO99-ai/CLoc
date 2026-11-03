from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


HEX_FUNCTION_RE = re.compile(r"^(0x)?[0-9a-fA-F]{8}$")
ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")
INT_RE = re.compile(r"\b\d{6,}\b")

CALLBACK_EXACT = {
    "callback",
    "tokensreceived",
    "uniswapv2call",
    "pancakecall",
    "dvmflashloancall",
    "executeoperation",
    "callfunction",
}
FLASHLOAN_NAMES = {
    "flashloan",
    "borrowflashloan",
    "loan",
    "operate",
}
PRICE_READ_EXACT = {
    "getunderlyingprice",
    "latestanswer",
    "getprice",
    "getassetprice",
    "consult",
}
ORACLE_WRITE_EXACT = {
    "setoracledata",
    "setunderlyingprice",
    "setprice",
    "setassetprice",
    "updateprice",
    "updateanswer",
    "setlatestanswer",
    "postprices",
}
CONFIG_WRITE_EXACT = {
    "setconfig",
    "setparameter",
    "setcollateralfactor",
    "setborrowcap",
    "setreservefactor",
    "setgovernance",
    "setowner",
    "setcontroller",
    "upgrade",
    "initialize",
}
PERMISSION_NAMES = {
    "borrowallowed",
    "mintallowed",
    "redeemallowed",
    "transferallowed",
    "permissioncheck",
    "checkrole",
    "hasrole",
    "onlyowner",
    "seizeallowed",
}
REPEATED_SENSITIVE_TAGS = {
    "deposit",
    "deposit_internal",
    "withdraw",
    "borrow",
    "mint",
    "liquidate",
    "transfer",
    "transferFrom",
    "approve",
    "swap",
    "callback",
    "flashloan_callback",
    "oracle_write",
    "config_write",
}
COMMON_TAGS = {"deposit", "withdraw", "borrow", "transfer", "approve", "price_read"}
RESULT_TAGS = {"borrow", "withdraw", "liquidate", "mint", "remove_liquidity", "oracle_write", "callback", "fallback"}
PRIORITY_TAGS = {
    "flashloan",
    "flashloan_callback",
    "callback",
    "fallback",
    "oracle_write",
    "config_write",
    "price_read",
    "permission_check",
    "approve",
    "max_approval",
    "deposit",
    "deposit_internal",
    "withdraw",
    "swap",
    "add_liquidity",
    "remove_liquidity",
    "borrow",
    "repay",
    "liquidate",
}


@dataclass
class FlatNode:
    node_id: str
    parent_id: str | None
    depth: int
    order_index: int
    raw: Dict[str, Any]
    semantic: Dict[str, Any]


def load_call_tree(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("call_tree") or data


def build_semantic_artifacts(call_tree: Dict[str, Any]) -> Dict[str, Any]:
    flat_nodes = _flatten_call_tree(call_tree)
    id_map = {node.node_id: node for node in flat_nodes}
    _apply_risk_tags(flat_nodes, id_map)
    semantic_nodes = [node.semantic for node in flat_nodes]
    raw_candidates = _build_raw_candidate_chains(flat_nodes, id_map)
    candidate_chains = _deduplicate_candidate_chains(raw_candidates)
    representative_chains, cluster_count, suppressed_count = _compress_clusters(candidate_chains)
    return {
        "semantic_nodes": semantic_nodes,
        "candidate_chains": candidate_chains,
        "representative_chains": representative_chains,
        "raw_candidate_count": len(raw_candidates),
        "dedup_candidate_count": len(candidate_chains),
        "cluster_count": cluster_count,
        "suppressed_similar_chains_count": suppressed_count,
    }


def write_semantic_artifacts(call_tree_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    call_tree = load_call_tree(call_tree_path)
    artifacts = build_semantic_artifacts(call_tree)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    semantic_path = out_dir / "semantic_nodes.json"
    chains_path = out_dir / "candidate_chains.json"
    semantic_path.write_text(json.dumps(artifacts["semantic_nodes"], indent=2, ensure_ascii=False), encoding="utf-8")
    chains_path.write_text(
        json.dumps(
            {
                "raw_candidate_count": artifacts["raw_candidate_count"],
                "dedup_candidate_count": artifacts["dedup_candidate_count"],
                "cluster_count": artifacts["cluster_count"],
                "suppressed_similar_chains_count": artifacts["suppressed_similar_chains_count"],
                "representative_chains": artifacts["representative_chains"],
                "chains": artifacts["candidate_chains"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "semantic_nodes_path": str(semantic_path),
        "candidate_chains_path": str(chains_path),
        "semantic_node_count": len(artifacts["semantic_nodes"]),
        "raw_candidate_count": artifacts["raw_candidate_count"],
        "dedup_candidate_count": artifacts["dedup_candidate_count"],
        "cluster_count": artifacts["cluster_count"],
        "suppressed_similar_chains_count": artifacts["suppressed_similar_chains_count"],
    }


def _flatten_call_tree(call_tree: Dict[str, Any]) -> List[FlatNode]:
    nodes: List[FlatNode] = []

    def visit(node: Dict[str, Any], parent_id: str | None, depth: int) -> str:
        node_id = f"n{len(nodes) + 1}"
        semantic = _build_semantic_node(node_id, parent_id, depth, len(nodes), node)
        flat = FlatNode(node_id=node_id, parent_id=parent_id, depth=depth, order_index=len(nodes), raw=node, semantic=semantic)
        nodes.append(flat)
        children_ids: List[str] = []
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                children_ids.append(visit(child, node_id, depth + 1))
        semantic["children_ids"] = children_ids
        return node_id

    visit(call_tree, None, 0)
    return nodes


def _build_semantic_node(node_id: str, parent_id: str | None, depth: int, order_index: int, node: Dict[str, Any]) -> Dict[str, Any]:
    address = str(node.get("address") or "")
    function = str(node.get("function") or "")
    params_summary = _summarize_params(node.get("params"), node.get("decoded_params"))
    call_type = str(node.get("call_type") or node.get("type") or "")
    value = node.get("value")
    semantic_tags = _infer_semantic_tags(function, params_summary, call_type, value)
    return {
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "order_index": order_index,
        "address": address,
        "function": function,
        "params_summary": params_summary,
        "call_type": call_type,
        "value": value,
        "semantic_tags": semantic_tags,
        "risk_tags": [],
        "asset_effect": _infer_asset_effect(semantic_tags, params_summary, value),
        "state_effect": _infer_state_effect(semantic_tags),
        "children_ids": [],
    }


def _summarize_params(params: Any, decoded_params: Any) -> str:
    if decoded_params not in (None, "", [], {}):
        text = json.dumps(decoded_params, ensure_ascii=False) if not isinstance(decoded_params, str) else decoded_params
    else:
        text = params if isinstance(params, str) else json.dumps(params, ensure_ascii=False)
    return str(text).replace("\n", " ").strip()[:400]


def _normalized_function(function: str) -> str:
    return function.lower().split("{", 1)[0].strip()


def _infer_semantic_tags(function: str, params_summary: str, call_type: str, value: Any) -> List[str]:
    fn = _normalized_function(function)
    call = call_type.lower()
    tags: List[str] = []

    if "delegate" in call:
        tags.append("delegatecall")
    if "static" in call:
        tags.append("staticcall")
    if fn == "fallback" or function.lower().startswith("fallback{"):
        tags.append("fallback")
        tags.append("callback")
    if fn == "receive" or function.lower().startswith("receive{"):
        tags.append("fallback")
        tags.append("callback")
    if fn in CALLBACK_EXACT:
        tags.append("callback")
    if fn in CALLBACK_EXACT - {"callback"}:
        tags.append("flashloan_callback")
    if any(name in fn for name in FLASHLOAN_NAMES):
        tags.append("flashloan")
    if fn == "transfer":
        tags.append("transfer")
    if fn == "transferfrom":
        tags.append("transferFrom")
    if fn == "approve":
        tags.append("approve")
        if _looks_like_max_approval(params_summary):
            tags.append("max_approval")
    if fn.startswith("mint"):
        tags.append("mint")
    if fn.startswith("burn"):
        tags.append("burn")
    if "depositinternal" in fn:
        tags.append("deposit_internal")
        tags.append("deposit")
    elif "deposit" in fn:
        tags.append("deposit")
    if "withdraw" in fn or "redeemunderlying" in fn or fn == "redeem" or "emergencywithdraw" in fn:
        tags.append("withdraw")
    if "borrow" in fn:
        tags.append("borrow")
    if "repay" in fn:
        tags.append("repay")
    if "liquidat" in fn or "seize" in fn:
        tags.append("liquidate")
    if "swap" in fn or "exchange" in fn:
        tags.append("swap")
    if "addliquidity" in fn or "joinpool" in fn:
        tags.append("add_liquidity")
    if "removeliquidity" in fn or "exitpool" in fn:
        tags.append("remove_liquidity")
    if "balanceof" in fn or "getcash" in fn:
        tags.append("balance_read")
    if fn in PRICE_READ_EXACT or any(token in fn for token in ("price", "oracle")) and not any(token in fn for token in ("set", "update", "post")):
        tags.append("price_read")
    if fn in ORACLE_WRITE_EXACT:
        tags.append("oracle_write")
    elif any(token in fn for token in ("oracle", "price")) and any(token in fn for token in ("set", "update", "post")):
        tags.append("oracle_write")
    if fn in CONFIG_WRITE_EXACT:
        tags.append("config_write")
    elif any(token in fn for token in ("config", "parameter", "collateralfactor", "borrowcap", "reservefactor")):
        tags.append("config_write")
    if fn in PERMISSION_NAMES or fn.endswith("allowed"):
        tags.append("permission_check")
    if HEX_FUNCTION_RE.fullmatch(function or "") or not function:
        tags.append("unknown_selector")
    if value not in (None, "", "0", 0, "0x0", "0x00") and "transfer" not in tags and "withdraw" not in tags:
        tags.append("transfer")

    return _dedupe(tags)


def _looks_like_max_approval(params_summary: str) -> bool:
    return "115792089237316195423570985008687907853269984665640564039457584007913129639935" in params_summary.lower()


def _infer_asset_effect(tags: List[str], params_summary: str, value: Any) -> List[str]:
    effects: List[str] = []
    if any(tag in tags for tag in ("transfer", "transferFrom", "withdraw", "remove_liquidity", "borrow", "liquidate")):
        effects.append("asset_out")
    if any(tag in tags for tag in ("deposit", "deposit_internal", "repay", "add_liquidity")):
        effects.append("asset_in")
    if "approve" in tags:
        effects.append("approval")
    if "mint" in tags:
        effects.append("supply_increase")
    if "burn" in tags:
        effects.append("supply_decrease")
    if value not in (None, "", "0", 0, "0x0", "0x00"):
        effects.append("native_value")
    if not effects and ADDRESS_RE.search(params_summary):
        effects.append("address_touch")
    return _dedupe(effects)


def _infer_state_effect(tags: List[str]) -> List[str]:
    effects: List[str] = []
    if "oracle_write" in tags:
        effects.append("oracle_mutation")
    if "config_write" in tags:
        effects.append("config_mutation")
    if "permission_check" in tags:
        effects.append("permission_gate")
    if any(tag in tags for tag in ("price_read", "balance_read")):
        effects.append("read_only")
    if "delegatecall" in tags:
        effects.append("foreign_context")
    return _dedupe(effects)


def _apply_risk_tags(flat_nodes: List[FlatNode], id_map: Dict[str, FlatNode]) -> None:
    sensitive_counter: Dict[tuple[str, str], int] = {}
    for node in flat_nodes:
        tags = set(node.semantic["semantic_tags"])
        if tags & REPEATED_SENSITIVE_TAGS:
            key = (node.semantic["address"].lower(), _normalized_function(node.semantic["function"]))
            sensitive_counter[key] = sensitive_counter.get(key, 0) + 1

    for node in flat_nodes:
        risks = node.semantic["risk_tags"]
        params_summary = node.semantic["params_summary"]
        tags = set(node.semantic["semantic_tags"])
        ancestors = _ancestor_nodes(node, id_map)

        if _extract_max_amount(params_summary) >= 10**21:
            risks.append("large_amount")
        if any("flashloan" in anc.semantic["semantic_tags"] for anc in ancestors):
            risks.append("after_flashloan")
        if any(set(anc.semantic["semantic_tags"]) & {"callback", "flashloan_callback", "fallback"} for anc in ancestors):
            risks.append("after_callback")
        if any(set(anc.semantic["semantic_tags"]) & {"callback", "flashloan_callback", "fallback"} for anc in ancestors) and tags & {"borrow", "withdraw", "transfer", "liquidate"}:
            risks.append("reentrancy_risk")
        key = (node.semantic["address"].lower(), _normalized_function(node.semantic["function"]))
        if sensitive_counter.get(key, 0) > 1 and tags & REPEATED_SENSITIVE_TAGS:
            risks.append("repeated_sensitive_call")
        if any("oracle_write" in anc.semantic["semantic_tags"] or "config_write" in anc.semantic["semantic_tags"] for anc in ancestors) and tags & {"borrow", "mint", "withdraw"}:
            risks.append("oracle_write_before_borrow")
        if "max_approval" in tags and node.parent_id is not None:
            risks.append("max_approval_to_external")
        if "large_amount" in risks and tags & {"transfer", "transferFrom", "withdraw", "remove_liquidity", "borrow", "mint"}:
            risks.append("large_asset_out")
        if tags & {"oracle_write", "config_write"}:
            risks.append("sensitive_config_write")
        if tags and tags.issubset(COMMON_TAGS) and not {"large_amount", "after_flashloan", "after_callback", "oracle_write_before_borrow"} & set(risks):
            risks.append("normal_behavior_common")
        node.semantic["risk_tags"] = _dedupe(risks)


def _build_raw_candidate_chains(flat_nodes: List[FlatNode], id_map: Dict[str, FlatNode]) -> List[Dict[str, Any]]:
    chains: List[Dict[str, Any]] = []
    for node in flat_nodes:
        if not _is_result_node(node):
            continue
        chain_nodes = _select_chain_nodes(node, id_map)
        if chain_nodes:
            chains.append(_build_chain_record(f"chain_{len(chains) + 1}", chain_nodes, node, "path"))
        chains.extend(_build_temporal_candidates(node, flat_nodes))
    chains.sort(key=lambda item: item["total_score"], reverse=True)
    return chains


def _deduplicate_candidate_chains(chains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for chain in chains:
        key = "|".join(chain["semantic_pattern"]) + "|" + chain["result_address"].lower() + "|" + _normalized_function(chain["result_function"])
        current = best.get(key)
        if current is None or chain["total_score"] > current["total_score"]:
            best[key] = chain
    deduped = list(best.values())
    deduped.sort(key=lambda item: item["total_score"], reverse=True)
    return deduped


def _is_result_node(node: FlatNode) -> bool:
    tags = set(node.semantic["semantic_tags"])
    if tags & RESULT_TAGS:
        return True
    return bool(tags & {"transfer", "transferFrom"} and "large_amount" in node.semantic["risk_tags"])


def _select_chain_nodes(node: FlatNode, id_map: Dict[str, FlatNode]) -> List[FlatNode]:
    ancestors = list(reversed(_ancestor_nodes(node, id_map)))
    selected: List[FlatNode] = []
    for anc in ancestors:
        tags = set(anc.semantic["semantic_tags"])
        if anc.parent_id is None or tags & PRIORITY_TAGS:
            selected.append(anc)
    if not selected or selected[-1].node_id != node.node_id:
        selected.append(node)
    expanded: List[FlatNode] = []
    selected_ids = {item.node_id for item in selected}
    cursor = node
    while cursor:
        if cursor.node_id in selected_ids or cursor.parent_id is None:
            expanded.append(cursor)
        cursor = id_map.get(cursor.parent_id) if cursor.parent_id else None
    expanded.reverse()
    return _dedupe_nodes(expanded)


def _build_temporal_candidates(result_node: FlatNode, flat_nodes: List[FlatNode]) -> List[Dict[str, Any]]:
    result_tags = set(result_node.semantic["semantic_tags"])
    if not (result_tags & {"borrow", "mint", "withdraw", "liquidate"}):
        return []
    price_reads = [
        node for node in flat_nodes
        if node.order_index < result_node.order_index and "price_read" in node.semantic["semantic_tags"]
    ]
    writes = [
        node for node in flat_nodes
        if node.order_index < result_node.order_index and set(node.semantic["semantic_tags"]) & {"oracle_write", "config_write"}
    ]
    candidates: List[Dict[str, Any]] = []
    for write in writes:
        nearby_reads = [node for node in price_reads if write.order_index <= node.order_index <= result_node.order_index]
        if not nearby_reads and not (result_tags & {"borrow", "mint", "withdraw"}):
            continue
        chain_nodes = [write]
        if nearby_reads:
            chain_nodes.append(nearby_reads[0])
        chain_nodes.append(result_node)
        candidates.append(_build_chain_record(f"temporal_{write.node_id}_{result_node.node_id}", chain_nodes, result_node, "transaction_order_edge"))
    return candidates


def _build_chain_record(chain_id: str, chain_nodes: List[FlatNode], result_node: FlatNode, edge_type: str) -> Dict[str, Any]:
    trigger_node = chain_nodes[0]
    semantic_pattern = [",".join(node.semantic["semantic_tags"]) or "untagged" for node in chain_nodes]
    score_detail = _score_chain(chain_nodes, result_node, edge_type)
    rootish = [node for node in chain_nodes[:-1] if set(node.semantic["semantic_tags"]) & PRIORITY_TAGS]
    return {
        "chain_id": chain_id,
        "node_ids": [node.node_id for node in chain_nodes],
        "chain_summary": _build_chain_summary(chain_nodes),
        "trigger_node": trigger_node.node_id,
        "result_node": result_node.node_id,
        "result_address": result_node.semantic["address"],
        "result_function": result_node.semantic["function"],
        "semantic_pattern": semantic_pattern,
        "edge_type": edge_type,
        "score_detail": score_detail,
        "total_score": round(score_detail["total_score"], 3),
        "suspected_root_addresses": _dedupe([node.semantic["address"] for node in rootish if node.semantic["address"]]),
        "suspected_root_functions": _dedupe([node.semantic["function"] for node in rootish if node.semantic["function"]]),
    }


def _score_chain(chain_nodes: List[FlatNode], result_node: FlatNode, edge_type: str) -> Dict[str, Any]:
    result_severity = _result_severity(result_node)
    causal_continuity = min(3.0, max(1.0, len(chain_nodes) / 2.0))
    if edge_type != "path":
        causal_continuity = max(1.0, causal_continuity - 0.5)
    abnormal_combination = _abnormal_combination(chain_nodes)
    source_verifiability = round(sum(_function_verifiability(node.semantic["function"]) for node in chain_nodes) / len(chain_nodes), 3)
    negative_commonness = _negative_commonness(chain_nodes)
    infrastructure_penalty = 1.0 if _unknown_ratio(chain_nodes) > 0.6 else 0.0
    total_score = (
        2 * result_severity
        + 2 * causal_continuity
        + 1.5 * abnormal_combination
        + source_verifiability
        - 1.5 * negative_commonness
        - infrastructure_penalty
    )
    return {
        "result_severity": result_severity,
        "causal_continuity": causal_continuity,
        "abnormal_combination": abnormal_combination,
        "source_verifiability": source_verifiability,
        "negative_commonness": negative_commonness,
        "infrastructure_penalty": infrastructure_penalty,
        "total_score": total_score,
    }


def _result_severity(node: FlatNode) -> float:
    tags = set(node.semantic["semantic_tags"])
    if "oracle_write" in tags or "config_write" in tags:
        return 2.0
    if "liquidate" in tags or "remove_liquidity" in tags or "mint" in tags:
        return 2.8
    if "borrow" in tags or "withdraw" in tags:
        return 2.6
    if tags & {"transfer", "transferFrom"} and "large_amount" in node.semantic["risk_tags"]:
        return 2.5
    if tags & {"callback", "fallback"}:
        return 2.2
    return 1.5


def _abnormal_combination(chain_nodes: List[FlatNode]) -> float:
    tags = {tag for node in chain_nodes for tag in node.semantic["semantic_tags"]}
    functions = {_normalized_function(node.semantic["function"]) for node in chain_nodes}
    score = 0.0
    if "flashloan" in tags and "swap" in tags:
        score += 1.0
    if ("oracle_write" in tags or "config_write" in tags) and ("price_read" in tags or tags & {"borrow", "mint", "withdraw"}):
        score += 1.2
    if ("oracle_write" in tags or "config_write" in tags) and "price_read" in tags and tags & {"borrow", "mint", "withdraw"}:
        score += 1.0
    if "max_approval" in tags and ("transferFrom" in tags or "transfer" in tags):
        score += 1.0
    if ("callback" in tags or "flashloan_callback" in tags or "fallback" in tags) and (tags & {"withdraw", "borrow", "mint", "liquidate"}):
        score += 1.0
    if "delegatecall" in tags and ("config_write" in tags or "oracle_write" in tags):
        score += 0.7
    if "add_liquidity" in tags and (("remove_liquidity" in tags) or ("remove_liquidity_one_coin" in functions)) and (tags & {"withdraw", "transfer", "transferFrom"}):
        score += 1.0
    return min(score, 3.0)


def _negative_commonness(chain_nodes: List[FlatNode]) -> float:
    tags = {tag for node in chain_nodes for tag in node.semantic["semantic_tags"]}
    if _is_boring_borrow_chain(chain_nodes):
        return 2.0
    if tags and tags.issubset(COMMON_TAGS | {"unknown_selector", "balance_read", "permission_check", "staticcall", "delegatecall"}):
        return 1.5
    return 0.0


def _is_boring_borrow_chain(chain_nodes: List[FlatNode]) -> bool:
    tags = {tag for node in chain_nodes for tag in node.semantic["semantic_tags"]}
    functions = {_normalized_function(node.semantic["function"]) for node in chain_nodes}
    has_core = "borrow" in tags and "permission_check" in tags and ("borrowindex" in functions or "totalborrows" in functions)
    has_unusual = bool(tags & {"oracle_write", "config_write", "callback", "flashloan_callback", "fallback"}) or any(
        "large_asset_out" in node.semantic["risk_tags"] for node in chain_nodes
    )
    return has_core and not has_unusual


def _function_verifiability(function: str) -> float:
    if not function:
        return 0.0
    if HEX_FUNCTION_RE.fullmatch(function):
        return 0.3
    return 1.0


def _unknown_ratio(chain_nodes: List[FlatNode]) -> float:
    unknown = sum(1 for node in chain_nodes if "unknown_selector" in node.semantic["semantic_tags"])
    return unknown / len(chain_nodes)


def _build_chain_summary(chain_nodes: List[FlatNode]) -> str:
    pieces = []
    for node in chain_nodes[:6]:
        fn = node.semantic["function"] or "unknown"
        tags = "/".join(node.semantic["semantic_tags"][:2]) or "untagged"
        pieces.append(f"{fn}<{tags}>")
    return " -> ".join(pieces)


def _ancestor_nodes(node: FlatNode, id_map: Dict[str, FlatNode]) -> List[FlatNode]:
    ancestors: List[FlatNode] = []
    cursor = id_map.get(node.parent_id) if node.parent_id else None
    while cursor:
        ancestors.append(cursor)
        cursor = id_map.get(cursor.parent_id) if cursor.parent_id else None
    return ancestors


def _extract_max_amount(text: str) -> int:
    values = [int(match.group(0)) for match in INT_RE.finditer(text)]
    return max(values) if values else 0


def _dedupe(values: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for value in values:
        key = value if isinstance(value, (str, int, float, bool, type(None))) else json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            ordered.append(value)
    return ordered


def _dedupe_nodes(nodes: List[FlatNode]) -> List[FlatNode]:
    seen = set()
    ordered: List[FlatNode] = []
    for node in nodes:
        if node.node_id not in seen:
            seen.add(node.node_id)
            ordered.append(node)
    return ordered


KEY_ACTIONS = {
    "deposit",
    "withdraw",
    "emergencywithdraw",
    "borrow",
    "liquidateborrow",
    "mint",
    "burn",
    "transfer",
    "transferfrom",
    "add_liquidity",
    "remove_liquidity",
    "swap",
    "oracle_write",
    "config_write",
    "callback",
}


def _compress_clusters(chains: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int, int]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for chain in chains:
        key = f"{chain['trigger_node']}|{'|'.join(chain['semantic_pattern'])}"
        grouped.setdefault(key, []).append(chain)

    representatives: List[Dict[str, Any]] = []
    suppressed_count = 0
    for cluster in grouped.values():
        cluster_sorted = sorted(cluster, key=lambda item: item["total_score"], reverse=True)
        kept: List[Dict[str, Any]] = []
        seen_actions = set()
        for chain in cluster_sorted:
            result_fn = _normalized_function(chain.get("result_function", ""))
            result_pattern = ",".join(chain.get("semantic_pattern", []))
            action = _result_action(result_fn, result_pattern)
            if not kept:
                kept.append(chain)
                if action:
                    seen_actions.add(action)
                continue
            if action and action not in seen_actions:
                kept.append(chain)
                seen_actions.add(action)
            else:
                suppressed_count += 1
        representatives.extend(kept)

    representatives.sort(key=lambda item: item["total_score"], reverse=True)
    return representatives, len(grouped), suppressed_count


def _result_action(result_function: str, semantic_pattern: str) -> str:
    if result_function in KEY_ACTIONS:
        return result_function
    lowered_pattern = semantic_pattern.lower()
    for action in KEY_ACTIONS:
        if action in lowered_pattern:
            return action
    return ""

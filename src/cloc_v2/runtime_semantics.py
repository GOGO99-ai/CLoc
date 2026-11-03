"""Pure phase-2A runtime semantics for a phase-1 transaction graph.

Rules describe exact observed call names and standard argument layouts only.
They deliberately do not infer causal relationships, exploit types, or source
locations.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .graph_models import NormalizedCallNode, TransactionGraph, TransactionGraphEdge
from .runtime_annotation_models import AddressProfile, AnnotationEvidence, NodeRuntimeAnnotation, RuntimeAnnotations
from .transaction_graph import serialize_transaction_graph


ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
INTEGER_WITH_DISPLAY_RE = re.compile(r"^\s*(\d+)(?:\s+\[[^\]]+\])?\s*$")
HEX_RE = re.compile(r"^[0-9a-fA-F]*$")
ERC20_TRANSFER_SELECTOR = "0xa9059cbb"
ERC20_TRANSFER_FROM_SELECTOR = "0x23b872dd"

ACTION_TOKEN_TRANSFER = "TOKEN_TRANSFER"
ACTION_TOKEN_TRANSFER_FROM = "TOKEN_TRANSFER_FROM"
ACTION_TOKEN_APPROVAL = "TOKEN_APPROVAL"
ACTION_BALANCE_READ = "BALANCE_READ"
ACTION_RESERVE_READ = "RESERVE_READ"
ACTION_PRICE_READ = "PRICE_READ"
ACTION_SWAP = "SWAP"
ACTION_FLASH_LOAN = "FLASH_LOAN"
ACTION_CALLBACK = "CALLBACK"
ACTION_REWARD_CLAIM = "REWARD_CLAIM"
ACTION_DEPOSIT = "DEPOSIT"
ACTION_WITHDRAW = "WITHDRAW"
ACTION_BORROW = "BORROW"
ACTION_REPAYMENT_CANDIDATE = "REPAYMENT_CANDIDATE"

ALLOWED_ACTION_TYPES = frozenset(
    {
        ACTION_TOKEN_TRANSFER,
        ACTION_TOKEN_TRANSFER_FROM,
        ACTION_TOKEN_APPROVAL,
        ACTION_BALANCE_READ,
        ACTION_RESERVE_READ,
        ACTION_PRICE_READ,
        ACTION_SWAP,
        ACTION_FLASH_LOAN,
        ACTION_CALLBACK,
        ACTION_REWARD_CLAIM,
        ACTION_DEPOSIT,
        ACTION_WITHDRAW,
        ACTION_BORROW,
        ACTION_REPAYMENT_CANDIDATE,
    }
)


@dataclass(frozen=True)
class FunctionRule:
    rule_id: str
    action_types: tuple[str, ...]
    reason: str
    operand_parser: Callable[[NormalizedCallNode], tuple[dict[str, Any], tuple[str, ...]]] | None = None


def _rule(rule_id: str, action_types: tuple[str, ...], reason: str, parser: Callable[[NormalizedCallNode], tuple[dict[str, Any], tuple[str, ...]]] | None = None) -> FunctionRule:
    return FunctionRule(rule_id=rule_id, action_types=action_types, reason=reason, operand_parser=parser)


# Exact, case-insensitive function-name registry.  Adding a new supported
# function is a local registry edit, not a scattered keyword heuristic.
FUNCTION_RULES: dict[str, FunctionRule] = {
    "transfer": _rule("FN_EXACT_TRANSFER", (ACTION_TOKEN_TRANSFER,), "function name exactly matches transfer", lambda node: _parse_transfer(node)),
    "transferfrom": _rule(
        "FN_EXACT_TRANSFER_FROM",
        (ACTION_TOKEN_TRANSFER, ACTION_TOKEN_TRANSFER_FROM),
        "function name exactly matches transferFrom",
        lambda node: _parse_transfer_from(node),
    ),
    "safetransferfrom": _rule(
        "FN_EXACT_SAFE_TRANSFER_FROM",
        (ACTION_TOKEN_TRANSFER, ACTION_TOKEN_TRANSFER_FROM),
        "function name exactly matches safeTransferFrom",
        lambda node: _parse_safe_transfer_from(node),
    ),
    "approve": _rule("FN_EXACT_APPROVE", (ACTION_TOKEN_APPROVAL,), "function name exactly matches approve", lambda node: _parse_approve(node)),
    "balanceof": _rule("FN_EXACT_BALANCE_OF", (ACTION_BALANCE_READ,), "function name exactly matches balanceOf", lambda node: _parse_balance_of(node)),
    "getreserves": _rule("FN_EXACT_GET_RESERVES", (ACTION_RESERVE_READ,), "function name exactly matches getReserves", lambda node: _parse_raw_arguments(node)),
    "getreserve": _rule("FN_EXACT_GET_RESERVE", (ACTION_RESERVE_READ,), "function name exactly matches getReserve", lambda node: _parse_raw_arguments(node)),
    "getreservedata": _rule("FN_EXACT_GET_RESERVE_DATA", (ACTION_RESERVE_READ,), "function name exactly matches getReserveData", lambda node: _parse_raw_arguments(node)),
    "queryrate": _rule("FN_EXACT_QUERY_RATE", (ACTION_PRICE_READ,), "function name exactly matches queryRate", lambda node: _parse_raw_arguments(node)),
    "latestanswer": _rule("FN_EXACT_LATEST_ANSWER", (ACTION_PRICE_READ,), "function name exactly matches latestAnswer", lambda node: _parse_raw_arguments(node)),
    "getamountsout": _rule("FN_EXACT_GET_AMOUNTS_OUT", (ACTION_PRICE_READ,), "function name exactly matches getAmountsOut", lambda node: _parse_raw_arguments(node)),
    "getamountout": _rule("FN_EXACT_GET_AMOUNT_OUT", (ACTION_PRICE_READ,), "function name exactly matches getAmountOut", lambda node: _parse_raw_arguments(node)),
    "getprice": _rule("FN_EXACT_GET_PRICE", (ACTION_PRICE_READ,), "function name exactly matches getPrice", lambda node: _parse_raw_arguments(node)),
    "getunderlyingprice": _rule("FN_EXACT_GET_UNDERLYING_PRICE", (ACTION_PRICE_READ,), "function name exactly matches getUnderlyingPrice", lambda node: _parse_raw_arguments(node)),
    "getassetprice": _rule("FN_EXACT_GET_ASSET_PRICE", (ACTION_PRICE_READ,), "function name exactly matches getAssetPrice", lambda node: _parse_raw_arguments(node)),
    "consult": _rule("FN_EXACT_CONSULT", (ACTION_PRICE_READ,), "function name exactly matches consult", lambda node: _parse_raw_arguments(node)),
    "swap": _rule("FN_EXACT_SWAP", (ACTION_SWAP,), "function name exactly matches swap", lambda node: _parse_raw_arguments(node)),
    "swapexacttokensfortokens": _rule("FN_EXACT_SWAP_EXACT_TOKENS_FOR_TOKENS", (ACTION_SWAP,), "function name exactly matches swapExactTokensForTokens", lambda node: _parse_raw_arguments(node)),
    "swapexactethfortokens": _rule("FN_EXACT_SWAP_EXACT_ETH_FOR_TOKENS", (ACTION_SWAP,), "function name exactly matches swapExactETHForTokens", lambda node: _parse_raw_arguments(node)),
    "swapexacttokensforeth": _rule("FN_EXACT_SWAP_EXACT_TOKENS_FOR_ETH", (ACTION_SWAP,), "function name exactly matches swapExactTokensForETH", lambda node: _parse_raw_arguments(node)),
    "swaptokensforexacttokens": _rule("FN_EXACT_SWAP_TOKENS_FOR_EXACT_TOKENS", (ACTION_SWAP,), "function name exactly matches swapTokensForExactTokens", lambda node: _parse_raw_arguments(node)),
    "flashloan": _rule("FN_EXACT_FLASH_LOAN", (ACTION_FLASH_LOAN,), "function name exactly matches flashLoan", lambda node: _parse_raw_arguments(node)),
    "getreward": _rule("FN_EXACT_GET_REWARD", (ACTION_REWARD_CLAIM,), "function name exactly matches getReward", lambda node: _parse_raw_arguments(node)),
    "claimreward": _rule("FN_EXACT_CLAIM_REWARD", (ACTION_REWARD_CLAIM,), "function name exactly matches claimReward", lambda node: _parse_raw_arguments(node)),
    "claimrewards": _rule("FN_EXACT_CLAIM_REWARDS", (ACTION_REWARD_CLAIM,), "function name exactly matches claimRewards", lambda node: _parse_raw_arguments(node)),
    "deposit": _rule("FN_EXACT_DEPOSIT", (ACTION_DEPOSIT,), "function name exactly matches deposit", lambda node: _parse_raw_arguments(node)),
    "withdraw": _rule("FN_EXACT_WITHDRAW", (ACTION_WITHDRAW,), "function name exactly matches withdraw", lambda node: _parse_raw_arguments(node)),
    "borrow": _rule("FN_EXACT_BORROW", (ACTION_BORROW,), "function name exactly matches borrow", lambda node: _parse_raw_arguments(node)),
}

# Exact selector registry for raw callTracer calldata.  These are the only
# selector identities decoded here; no length- or content-based inference is
# permitted.
SELECTOR_RULES: dict[str, FunctionRule] = {
    ERC20_TRANSFER_SELECTOR: _rule(
        "SELECTOR_EXACT_ERC20_TRANSFER",
        (ACTION_TOKEN_TRANSFER,),
        "selector exactly matches ERC-20 transfer(address,uint256)",
        lambda node: _parse_transfer_calldata(node),
    ),
    ERC20_TRANSFER_FROM_SELECTOR: _rule(
        "SELECTOR_EXACT_ERC20_TRANSFER_FROM",
        (ACTION_TOKEN_TRANSFER, ACTION_TOKEN_TRANSFER_FROM),
        "selector exactly matches ERC-20 transferFrom(address,address,uint256)",
        lambda node: _parse_transfer_from_calldata(node),
    ),
}
SELECTOR_FUNCTION_NAMES = {
    ERC20_TRANSFER_SELECTOR: "transfer",
    ERC20_TRANSFER_FROM_SELECTOR: "transferfrom",
}

CALLBACK_RULE = _rule(
    "FN_EXPLICIT_CALLBACK",
    (ACTION_CALLBACK,),
    "function name is in the explicit callback registry",
    lambda node: _parse_raw_arguments(node),
)
CALLBACK_FUNCTION_NAMES = frozenset(
    {
        "uniswapv2call",
        "pancakecall",
        "biswapcall",
        "dppflashloancall",
        "executeoperation",
    }
)

CAPABILITY_ACTIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("TOKEN_LIKE", frozenset({ACTION_TOKEN_TRANSFER, ACTION_TOKEN_TRANSFER_FROM, ACTION_TOKEN_APPROVAL, ACTION_BALANCE_READ})),
    ("SWAP_LIKE", frozenset({ACTION_SWAP})),
    ("PRICE_SOURCE_LIKE", frozenset({ACTION_PRICE_READ, ACTION_RESERVE_READ})),
    ("REWARD_LIKE", frozenset({ACTION_REWARD_CLAIM})),
    ("FLASH_LENDER_LIKE", frozenset({ACTION_FLASH_LOAN})),
    ("CALLBACK_RECEIVER_LIKE", frozenset({ACTION_CALLBACK})),
)


def build_runtime_annotations(transaction_graph: TransactionGraph) -> RuntimeAnnotations:
    """Build deterministic annotations without mutating or enriching the graph."""
    annotations = tuple(_annotate_node(node) for node in transaction_graph.nodes)
    profiles = _build_address_profiles(transaction_graph.nodes, annotations)
    return RuntimeAnnotations(
        transaction_graph_sha256=transaction_graph_sha256(transaction_graph),
        node_annotations=annotations,
        address_profiles=profiles,
    )


def load_transaction_graph(path: str | Path) -> TransactionGraph:
    """Load only a valid phase-1 ``transaction_graph_v1`` JSON artifact."""
    graph_path = Path(path)
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Transaction graph does not exist: {graph_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Transaction graph is not valid JSON: {graph_path} ({exc.msg})") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != TransactionGraph.SCHEMA_VERSION:
        raise ValueError("Input schema_version must be transaction_graph_v1")

    try:
        nodes = tuple(_node_from_dict(item) for item in payload["nodes"])
        call_edges = tuple(_edge_from_dict(item) for item in payload["call_edges"])
        order_edges = tuple(_edge_from_dict(item) for item in payload["order_edges"])
        graph = TransactionGraph(
            root_id=payload["root_id"],
            source_tree_sha256=payload["source_tree_sha256"],
            nodes=nodes,
            call_edges=call_edges,
            order_edges=order_edges,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid transaction_graph_v1 structure: {exc}") from exc
    return graph


def serialize_runtime_annotations(annotations: RuntimeAnnotations) -> str:
    return json.dumps(annotations.to_dict(), indent=2, ensure_ascii=False) + "\n"


def write_runtime_annotations(annotations: RuntimeAnnotations, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_runtime_annotations(annotations), encoding="utf-8")
    return path


def transaction_graph_sha256(transaction_graph: TransactionGraph) -> str:
    return hashlib.sha256(serialize_transaction_graph(transaction_graph).encode("utf-8")).hexdigest()


def split_top_level_arguments(params_raw: Any) -> list[str]:
    """Split comma-separated arguments while preserving nested arrays/brackets."""
    if params_raw is None:
        return []
    text = str(params_raw).strip()
    if not text:
        return []

    arguments: list[str] = []
    current: list[str] = []
    closing_for = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for character in text:
        if quote is not None:
            current.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            current.append(character)
        elif character in closing_for:
            stack.append(closing_for[character])
            current.append(character)
        elif stack and character == stack[-1]:
            stack.pop()
            current.append(character)
        elif character == "," and not stack:
            arguments.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    arguments.append("".join(current).strip())
    return arguments


def _annotate_node(node: NormalizedCallNode) -> NodeRuntimeAnnotation:
    normalized_name = (node.function_name or "").strip().lower()
    selector = (node.function_selector or "").strip().lower()
    name_rule = FUNCTION_RULES.get(normalized_name)
    if name_rule is None and normalized_name in CALLBACK_FUNCTION_NAMES:
        name_rule = CALLBACK_RULE
    selector_rule = SELECTOR_RULES.get(selector)

    uncertainties: list[str] = []
    if node.caller_source == "DERIVED_FROM_PARENT":
        uncertainties.append("caller address is derived from parent call")
    if normalized_name and selector:
        expected_name = SELECTOR_FUNCTION_NAMES.get(selector)
        if expected_name != normalized_name:
            uncertainties.append("FUNCTION_NAME_SELECTOR_CONFLICT")
            return NodeRuntimeAnnotation(
                node_id=node.node_id,
                action_types=(),
                observed_operands={},
                annotation_evidence=(),
                uncertainties=tuple(_dedupe_strings(uncertainties)),
            )

    rule = selector_rule if selector else name_rule
    if selector and selector_rule is None and not normalized_name:
        uncertainties.append(f"function selector unresolved: {node.function_selector}")
    elif selector and selector_rule is None and normalized_name:
        uncertainties.append("FUNCTION_NAME_SELECTOR_CONFLICT")
        rule = None
    if rule is None:
        return NodeRuntimeAnnotation(
            node_id=node.node_id,
            action_types=(),
            observed_operands={},
            annotation_evidence=(),
            uncertainties=tuple(uncertainties),
        )

    operands, parser_uncertainties = rule.operand_parser(node) if rule.operand_parser else ({}, ())
    uncertainties.extend(parser_uncertainties)
    evidence = tuple(
        AnnotationEvidence(
            action_type=action_type,
            evidence_level="OBSERVED",
            rule_id=rule.rule_id,
            reason=rule.reason,
            trace_refs=(node.trace_path,),
        )
        for action_type in rule.action_types
    )
    return NodeRuntimeAnnotation(
        node_id=node.node_id,
        action_types=rule.action_types,
        observed_operands=operands,
        annotation_evidence=evidence,
        uncertainties=tuple(_dedupe_strings(uncertainties)),
    )


def _parse_transfer(node: NormalizedCallNode) -> tuple[dict[str, Any], tuple[str, ...]]:
    args = split_top_level_arguments(node.params_raw)
    operands = {
        "token_address": _address_operand(node.callee_address),
        "from_address": _address_operand(node.caller_address),
        "to_address": None,
        "amount_raw": None,
    }
    if len(args) == 2:
        operands["to_address"] = _address_operand(args[0])
        operands["amount_raw"] = _integer_string(args[1])
    if operands["to_address"] is not None and operands["amount_raw"] is not None:
        return operands, ()
    operands["arguments_raw"] = args
    return operands, ("transfer arguments do not exactly match expected (address, amount) format",)


def _parse_transfer_from(node: NormalizedCallNode) -> tuple[dict[str, Any], tuple[str, ...]]:
    args = split_top_level_arguments(node.params_raw)
    operands = {
        "token_address": _address_operand(node.callee_address),
        "from_address": None,
        "to_address": None,
        "amount_raw": None,
    }
    if len(args) == 3:
        operands["from_address"] = _address_operand(args[0])
        operands["to_address"] = _address_operand(args[1])
        operands["amount_raw"] = _integer_string(args[2])
    if operands["from_address"] is not None and operands["to_address"] is not None and operands["amount_raw"] is not None:
        return operands, ()
    operands["arguments_raw"] = args
    return operands, ("transferFrom arguments do not exactly match expected (address, address, amount) format",)


def _parse_safe_transfer_from(node: NormalizedCallNode) -> tuple[dict[str, Any], tuple[str, ...]]:
    args = split_top_level_arguments(node.params_raw)
    operands = {
        "token_address": _address_operand(node.callee_address),
        "from_address": None,
        "to_address": None,
        "amount_raw": None,
    }
    if len(args) in {3, 4, 5}:
        operands["from_address"] = _address_operand(args[0])
        operands["to_address"] = _address_operand(args[1])
        # ERC-721 uses tokenId as the observed quantity; ERC-1155 uses amount.
        amount_index = 3 if len(args) == 5 else 2
        operands["amount_raw"] = _integer_string(args[amount_index])
    if operands["from_address"] is not None and operands["to_address"] is not None and operands["amount_raw"] is not None:
        return operands, ()
    operands["arguments_raw"] = args
    return operands, ("safeTransferFrom arguments do not match a supported standard overload",)


def _parse_transfer_calldata(node: NormalizedCallNode) -> tuple[dict[str, Any], tuple[str, ...]]:
    operands = {
        "token_address": _address_operand(node.callee_address),
        "from_address": _address_operand(node.caller_address),
        "to_address": None,
        "amount_raw": None,
    }
    words, uncertainties = _decode_exact_static_calldata(
        node.params_raw,
        expected_selector=ERC20_TRANSFER_SELECTOR,
        parameter_types=("address", "uint256"),
    )
    if words is not None:
        operands["to_address"] = _abi_address_word(words[0])
        operands["amount_raw"] = str(int(words[1], 16))
    return operands, uncertainties


def _parse_transfer_from_calldata(node: NormalizedCallNode) -> tuple[dict[str, Any], tuple[str, ...]]:
    operands = {
        "token_address": _address_operand(node.callee_address),
        "from_address": None,
        "to_address": None,
        "amount_raw": None,
    }
    words, uncertainties = _decode_exact_static_calldata(
        node.params_raw,
        expected_selector=ERC20_TRANSFER_FROM_SELECTOR,
        parameter_types=("address", "address", "uint256"),
    )
    if words is not None:
        operands["from_address"] = _abi_address_word(words[0])
        operands["to_address"] = _abi_address_word(words[1])
        operands["amount_raw"] = str(int(words[2], 16))
    return operands, uncertainties


def _decode_exact_static_calldata(
    value: Any,
    *,
    expected_selector: str,
    parameter_types: tuple[str, ...],
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    """Decode only an exact selector and exact-width static ABI payload."""
    if not isinstance(value, str) or not value.startswith("0x"):
        return None, ("ERC20_CALLDATA_INVALID_HEX",)
    hex_body = value[2:]
    if HEX_RE.fullmatch(hex_body) is None:
        return None, ("ERC20_CALLDATA_INVALID_HEX",)
    if len(hex_body) % 2:
        return None, ("ERC20_CALLDATA_ODD_HEX_LENGTH",)

    reasons: list[str] = []
    expected_hex_length = 8 + 64 * len(parameter_types)
    if len(hex_body) != expected_hex_length:
        reasons.append("ERC20_CALLDATA_LENGTH_INVALID")
    if len(hex_body) >= 8 and f"0x{hex_body[:8].lower()}" != expected_selector:
        reasons.append("ERC20_CALLDATA_SELECTOR_MISMATCH")
    if reasons:
        return None, tuple(_dedupe_strings(reasons))

    words = tuple(
        hex_body[8 + index * 64 : 8 + (index + 1) * 64]
        for index in range(len(parameter_types))
    )
    for word, parameter_type in zip(words, parameter_types):
        if parameter_type == "address" and word[:24] != "0" * 24:
            return None, ("ERC20_CALLDATA_ADDRESS_PADDING_INVALID",)
    return words, ()


def _abi_address_word(word: str) -> dict[str, str]:
    normalized = f"0x{word[24:].lower()}"
    return {"original": normalized, "normalized": normalized}


def _parse_approve(node: NormalizedCallNode) -> tuple[dict[str, Any], tuple[str, ...]]:
    args = split_top_level_arguments(node.params_raw)
    operands = {"token_address": _address_operand(node.callee_address), "spender_address": None, "amount_raw": None}
    if len(args) == 2:
        operands["spender_address"] = _address_operand(args[0])
        operands["amount_raw"] = _integer_string(args[1])
    if operands["spender_address"] is not None and operands["amount_raw"] is not None:
        return operands, ()
    operands["arguments_raw"] = args
    return operands, ("approve arguments do not exactly match expected (address, amount) format",)


def _parse_balance_of(node: NormalizedCallNode) -> tuple[dict[str, Any], tuple[str, ...]]:
    args = split_top_level_arguments(node.params_raw)
    operands = {"token_address": _address_operand(node.callee_address), "account_address": None}
    if len(args) == 1:
        operands["account_address"] = _address_operand(args[0])
    if operands["account_address"] is not None:
        return operands, ()
    operands["arguments_raw"] = args
    return operands, ("balanceOf arguments do not exactly match expected (address) format",)


def _parse_raw_arguments(node: NormalizedCallNode) -> tuple[dict[str, Any], tuple[str, ...]]:
    return {"arguments_raw": split_top_level_arguments(node.params_raw)}, ()


def _address_operand(value: Any) -> dict[str, str] | None:
    if not isinstance(value, str):
        return None
    original = value.strip()
    if not ADDRESS_RE.fullmatch(original):
        return None
    return {"original": original, "normalized": original.lower()}


def _integer_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    matched = INTEGER_WITH_DISPLAY_RE.fullmatch(value)
    return matched.group(1) if matched else None


def _build_address_profiles(
    nodes: tuple[NormalizedCallNode, ...], annotations: tuple[NodeRuntimeAnnotation, ...]
) -> tuple[AddressProfile, ...]:
    annotations_by_id = {item.node_id: item for item in annotations}
    grouped: dict[str, dict[str, Any]] = {}
    for node in nodes:
        annotation = annotations_by_id[node.node_id]
        if not annotation.action_types or not isinstance(node.callee_address, str):
            continue
        address = _address_operand(node.callee_address)
        if address is None:
            continue
        normalized = address["normalized"]
        entry = grouped.setdefault(
            normalized,
            {
                "address": address["original"],
                "functions": [],
                "actions": set(),
                "node_ids": [],
            },
        )
        if node.function_name and node.function_name not in entry["functions"]:
            entry["functions"].append(node.function_name)
        entry["actions"].update(annotation.action_types)
        entry["node_ids"].append(node.node_id)

    profiles: list[AddressProfile] = []
    for normalized, entry in grouped.items():
        capabilities = tuple(
            capability
            for capability, actions in CAPABILITY_ACTIONS
            if entry["actions"] & actions
        )
        profiles.append(
            AddressProfile(
                address=entry["address"],
                normalized_address=normalized,
                observed_functions=tuple(entry["functions"]),
                observed_capabilities=capabilities,
                evidence_node_ids=tuple(entry["node_ids"]),
            )
        )
    return tuple(sorted(profiles, key=lambda item: item.normalized_address))


def _node_from_dict(payload: dict[str, Any]) -> NormalizedCallNode:
    return NormalizedCallNode(
        node_id=payload["node_id"],
        trace_index=payload["trace_index"],
        trace_path=payload["trace_path"],
        depth=payload["depth"],
        parent_id=payload.get("parent_id"),
        sibling_order=payload["sibling_order"],
        caller_address=payload.get("caller_address"),
        caller_source=payload["caller_source"],
        callee_address=payload.get("callee_address"),
        node_type=payload.get("node_type"),
        function_name=payload.get("function_name"),
        function_selector=payload.get("function_selector"),
        call_type=payload.get("call_type"),
        params_raw=payload.get("params_raw"),
        decoded_params=payload.get("decoded_params"),
        value=payload.get("value"),
        gas=payload.get("gas"),
        contract_role=payload.get("contract_role", "UNKNOWN"),
        semantic_flags=tuple(payload.get("semantic_flags", [])),
        source_ref=payload.get("source_ref"),
    )


def _edge_from_dict(payload: dict[str, Any]) -> TransactionGraphEdge:
    return TransactionGraphEdge(
        edge_id=payload["edge_id"],
        edge_type=payload["edge_type"],
        src=payload["src"],
        dst=payload["dst"],
        evidence_level=payload["evidence_level"],
        reason=payload["reason"],
        trace_refs=tuple(payload["trace_refs"]),
    )


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered

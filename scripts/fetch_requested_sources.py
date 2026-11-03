"""Acquire verified source artefacts for explicit addresses or validated phase-1 requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloc.runtime.execution_policy import ExecutionPolicy
from src.cloc_v2.config_loader import ENV_OVERRIDES, load_chain_config
from src.cloc_v2.source_cache import norm_address
from src.cloc_v2.source_fetcher import SourceFetcher
from src.cloc_v2.source_manifest import write_run_manifest


def canonical_file_sha256(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def requested_addresses_from_ranked(ranked_path: Path, transaction_graph_path: Path) -> tuple[list[tuple[str, list[dict]]], dict[str, str]]:
    ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
    graph = json.loads(transaction_graph_path.read_text(encoding="utf-8"))
    if ranked.get("schema_version") != "phase1_ranked_chains_v1":
        raise ValueError("ranked input must be a formal phase1_ranked_chains_v1 document")
    if ranked.get("selection_audit", {}).get("all_chains_scored_exactly_once") is not True:
        raise ValueError("ranked input does not contain a fully validated phase-1 response set")
    if graph.get("schema_version") != "transaction_graph_v1":
        raise ValueError("transaction graph must be transaction_graph_v1")
    nodes = {node["node_id"]: node for node in graph.get("nodes", [])}
    collected: dict[str, list[dict]] = {}
    for selected in ranked.get("selected_topk", []):
        chain_id = selected.get("chain_id")
        for request in selected.get("source_requests", []):
            node_id = request.get("node_id")
            node = nodes.get(node_id)
            if not chain_id or not node or not node.get("callee_address"):
                raise ValueError(f"invalid ranked source_request node_id: {node_id}")
            address = norm_address(node["callee_address"])
            origin = {"chain_id": chain_id, "node_id": node_id, "focus": request.get("focus"), "priority": request.get("priority"), "request_reason": request.get("request_reason", "")}
            if origin not in collected.setdefault(address, []):
                collected[address].append(origin)
    return sorted(collected.items()), {"phase1_ranked_chains_v1": canonical_file_sha256(ranked_path), "transaction_graph_v1": canonical_file_sha256(transaction_graph_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch only requested verified sources; network is opt-in.")
    parser.add_argument("--chain", required=True, help="Cache/network label; eth and bsc use existing CLoc settings")
    parser.add_argument("--chainid", type=int, help="Etherscan V2 chainid for a non-default chain")
    parser.add_argument("--explorer-base-url", help="Etherscan V2 API endpoint for a non-default chain")
    parser.add_argument("--tx", help="Transaction hash used to select the run output directory")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--addresses", help="Comma-separated 0x addresses")
    inputs.add_argument("--addresses-file", type=Path, help="JSON array or newline-separated address file")
    inputs.add_argument("--ranked-input", type=Path, help="Formal phase1_ranked_chains_v1 JSON")
    parser.add_argument("--transaction-graph", type=Path, help="Required with --ranked-input")
    parser.add_argument("--output", type=Path, help="Default: data/runs/<chain>/<tx>/source_acquisition_v1.json")
    parser.add_argument("--cache-root", type=Path, default=Path("data/source_cache"))
    parser.add_argument("--api-key-env", help="Environment variable containing the Explorer key")
    parser.add_argument("--allow-network", action="store_true", help="Permit Explorer HTTP requests (default is offline)")
    parser.add_argument("--refresh", action="store_true", help="Ignore the seven-day failed/unverified cache TTL")
    return parser.parse_args()


def explicit_addresses(args: argparse.Namespace) -> list[tuple[str, list[dict]]]:
    if args.addresses:
        raw = [x.strip() for x in args.addresses.split(",") if x.strip()]
    else:
        text = args.addresses_file.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
            raw = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            raw = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw:
        raise ValueError("no addresses supplied")
    return [(address, []) for address in sorted({norm_address(x) for x in raw})]


def main() -> int:
    args = parse_args()
    if args.ranked_input and not args.transaction_graph:
        raise SystemExit("--transaction-graph is required with --ranked-input")
    if not args.ranked_input and args.transaction_graph:
        raise SystemExit("--transaction-graph is only valid with --ranked-input")
    if not args.output and not args.tx:
        raise SystemExit("--tx is required unless --output is supplied")
    if args.ranked_input:
        pairs, input_hashes = requested_addresses_from_ranked(args.ranked_input, args.transaction_graph)
    else:
        pairs = explicit_addresses(args)
        canonical_addresses = [address for address, _ in pairs]
        input_hashes = {"explicit_addresses": hashlib.sha256(json.dumps(canonical_addresses, separators=(",", ":")).encode("utf-8")).hexdigest()}
    chain = args.chain.lower()
    # Base URL and Etherscan V2 chainid come from the existing chain loader, but its legacy key is never used.
    try:
        config = load_chain_config(chain)
    except ValueError:
        if not args.chainid or not args.explorer_base_url:
            raise SystemExit("non-default chains require --chainid and --explorer-base-url")
        config = {"chain": chain, "chainid": args.chainid, "base_url": args.explorer_base_url}
    if args.chainid:
        config["chainid"] = args.chainid
    if args.explorer_base_url:
        config["base_url"] = args.explorer_base_url
    env_name = args.api_key_env or ENV_OVERRIDES.get(chain, {}).get("key", "ETHERSCAN_API_KEY")
    config["key"] = os.getenv(env_name, "")
    config["api_key_source"] = "environment" if config["key"] else "missing"
    output = args.output or Path("data/runs") / chain / args.tx.lower() / "source_acquisition_v1.json"
    policy = ExecutionPolicy(allow_explorer=args.allow_network, allow_artifact_cache_write=True)
    fetcher = SourceFetcher(chain, config, cache_root=args.cache_root, policy=policy)
    results = [fetcher.acquire(address, refresh=args.refresh, origins=origins) for address, origins in pairs]
    write_run_manifest(output, chain, args.tx.lower() if args.tx else None, results, input_hashes)
    print(json.dumps({"ok": True, "output": str(output), "address_count": len(results), "network_allowed": args.allow_network, "status_counts": {status: sum(x.status == status for x in results) for status in sorted({x.status for x in results})}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

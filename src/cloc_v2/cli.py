from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloc.runtime.execution_policy import (
    ExecutionPolicy,
    ExternalAccessDenied,
    RequiredArtifactUnavailable,
)
from src.cloc_v2 import (
    build_run_layout,
    find_case_by_tx,
    load_chain_config,
    scan_dataset_cases,
    summarize_chain_config,
)
from src.data_collection.trace_parser import TraceParser
from src.data_collection.txn_replayer import TraceReplayFailed, TransactionReplayer


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal CLOC v2 entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser("dry-run", help="Validate dataset lookup and config mapping.")
    dry_run.add_argument("--tx", required=True, help="Transaction hash to locate in dataset")
    dry_run.add_argument("--dataset-root", default="dataset", help="Dataset root directory")
    dry_run.set_defaults(handler=_handle_dry_run)

    replay = subparsers.add_parser("replay-trace", help="Replay a transaction into raw trace text.")
    replay.add_argument("--tx", required=True, help="Transaction hash")
    replay.add_argument("--chain", required=True, choices=["eth", "bsc"], help="Chain key")
    replay.add_argument(
        "--mode",
        choices=["cache-only", "live"],
        default="cache-only",
        help="Use cached trace only, or allow cast/RPC replay.",
    )
    replay.set_defaults(handler=_handle_replay_trace)

    call_tree = subparsers.add_parser(
        "build-call-tree",
        help="Build a raw call tree JSON from a local trace text file.",
    )
    call_tree.add_argument("--trace-file", required=True, help="Path to a local raw trace text file")
    call_tree.add_argument(
        "--output",
        help="Optional output path for the call tree JSON. Defaults to <trace-file stem>.call_tree.json",
    )
    call_tree.set_defaults(handler=_handle_build_call_tree)

    run_call_tree = subparsers.add_parser(
        "run-call-tree",
        help="Replay raw trace text and build a raw call tree in one step.",
    )
    run_call_tree.add_argument("--tx", required=True, help="Transaction hash")
    run_call_tree.add_argument("--chain", required=True, choices=["eth", "bsc"], help="Chain key")
    run_call_tree.add_argument(
        "--mode",
        choices=["cache-only", "live"],
        default="cache-only",
        help="Use cached trace only, or allow cast/RPC replay.",
    )
    run_call_tree.add_argument(
        "--output-root",
        default="data/runs",
        help="Root directory for trace and call tree outputs.",
    )
    run_call_tree.set_defaults(handler=_handle_run_call_tree)

    args = parser.parse_args()
    return args.handler(args)


def _handle_dry_run(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    cases = scan_dataset_cases(dataset_root)
    match = find_case_by_tx(args.tx, dataset_root)

    if match is None:
        print(
            json.dumps(
                {
                    "dataset_root": str(dataset_root),
                    "case_count": len(cases),
                    "matched": False,
                    "transaction_hash": args.tx.lower(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1

    chain_config = load_chain_config(match["mapped_chain"])
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "case_count": len(cases),
                "matched": True,
                "case_name": match["case_name"],
                "cause": match["cause"],
                "platform": match["platform"],
                "mapped_chain": match["mapped_chain"],
                "transaction_hash": args.tx.lower(),
                "has_location": match["has_location"],
                "location_count": match["location_count"],
                "case_file": match["path"],
                "config_summary": summarize_chain_config(chain_config),
                "next_action": "dry-run only; no network access performed",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _handle_replay_trace(args: argparse.Namespace) -> int:
    chain_config = load_chain_config(args.chain)
    policy = _build_policy(args.mode)
    replayer = TransactionReplayer(
        rpc_url=chain_config["rpc_url"],
        chain=args.chain,
        policy=policy,
    )

    cache_file = Path(replayer.cache_dir) / f"{args.chain}_{args.tx.lower()}.txt"

    try:
        trace_text = replayer.run(args.tx)
    except RequiredArtifactUnavailable as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": args.mode,
                    "chain": args.chain,
                    "transaction_hash": args.tx.lower(),
                    "cache_file": str(cache_file),
                    "reason": exc.reason,
                    "next_action": "switch to --mode live after confirming RPC/foundry access",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2
    except ExternalAccessDenied as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": args.mode,
                    "chain": args.chain,
                    "transaction_hash": args.tx.lower(),
                    "denied_capability": exc.capability,
                    "required_flag": exc.flag,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 3
    except TraceReplayFailed as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": args.mode,
                    "chain": args.chain,
                    "transaction_hash": args.tx.lower(),
                    "cache_file": str(cache_file),
                    "cast_run_error": exc.cast_error,
                    "rpc_fallback_error": exc.rpc_error,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 4

    print(
        json.dumps(
            {
                "ok": True,
                "mode": args.mode,
                "chain": args.chain,
                "transaction_hash": args.tx.lower(),
                "cache_file": str(cache_file),
                "trace_length": len(trace_text),
                "trace_preview": trace_text[:500],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _handle_build_call_tree(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace_file)
    output_path = Path(args.output) if args.output else trace_path.with_suffix(".call_tree.json")

    trace_text = trace_path.read_text(encoding="utf-8")
    parse_result = TraceParser().parse(trace_text)
    call_tree = parse_result.get("call_tree")
    serialized = call_tree.to_dict() if call_tree and hasattr(call_tree, "to_dict") else None

    payload = {"call_tree": serialized}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output_path), "call_tree_generated": serialized is not None},
            ensure_ascii=False,
        )
    )
    return 0 if serialized is not None else 1


def _handle_run_call_tree(args: argparse.Namespace) -> int:
    chain_config = load_chain_config(args.chain)
    policy = _build_policy(args.mode)
    replayer = TransactionReplayer(
        rpc_url=chain_config["rpc_url"],
        chain=args.chain,
        policy=policy,
    )
    layout = build_run_layout(args.tx, args.chain, args.output_root)
    run_root = Path(layout["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)

    try:
        trace_text = replayer.run(args.tx)
    except RequiredArtifactUnavailable as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": args.mode,
                    "chain": args.chain,
                    "transaction_hash": args.tx.lower(),
                    "planned_outputs": layout,
                    "reason": exc.reason,
                    "next_action": "switch to --mode live after confirming RPC/foundry access",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2
    except ExternalAccessDenied as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": args.mode,
                    "chain": args.chain,
                    "transaction_hash": args.tx.lower(),
                    "planned_outputs": layout,
                    "denied_capability": exc.capability,
                    "required_flag": exc.flag,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 3
    except TraceReplayFailed as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": args.mode,
                    "chain": args.chain,
                    "transaction_hash": args.tx.lower(),
                    "planned_outputs": layout,
                    "cast_run_error": exc.cast_error,
                    "rpc_fallback_error": exc.rpc_error,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 4

    trace_path = Path(layout["trace_file"])
    trace_path.write_text(trace_text, encoding="utf-8")

    parse_result = TraceParser().parse(trace_text)
    call_tree = parse_result.get("call_tree")
    serialized = call_tree.to_dict() if call_tree and hasattr(call_tree, "to_dict") else None

    call_tree_path = Path(layout["call_tree_file"])
    call_tree_path.write_text(
        json.dumps({"call_tree": serialized}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": serialized is not None,
                "mode": args.mode,
                "chain": args.chain,
                "transaction_hash": args.tx.lower(),
                "trace_file": str(trace_path),
                "call_tree_file": str(call_tree_path),
                "trace_length": len(trace_text),
                "call_tree_generated": serialized is not None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if serialized is not None else 1


def _build_policy(mode: str) -> ExecutionPolicy:
    if mode == "live":
        return ExecutionPolicy(
            allow_foundry=True,
            allow_rpc=True,
            allow_explorer=False,
            allow_artifact_cache_write=True,
        )
    return ExecutionPolicy.current_live_default()

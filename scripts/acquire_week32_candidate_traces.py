"""Acquire callTracer trees for the Week 32 source-evaluable remainder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_localization15_offline_rebuild as offline
from scripts.audit_full_fault_localization_evaluability import DEV15_MANIFEST, _dataset_locations, _manifest_transactions
from scripts.validate_real_normal_baseline import JsonRpcClient, RpcValidationError
from src.cloc_v2.calltracer_adapter import adapt_calltracer_call_tree
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.source_cache import atomic_json
from src.cloc_v2.transaction_graph import build_transaction_graph, load_call_tree
from src.data_collection.trace_parser import TraceParser


SCREEN_ROOT = ROOT / "data" / "prototypes" / "week32_remaining_dataset_screen_20260911"
SOURCE_AUDIT = SCREEN_ROOT / "post_sourcify_source_evaluability.json"
TRACE_ROOT = ROOT / "data" / "trace_acquisition" / "week32_remaining_dataset_20260911"
LOCKED = TRACE_ROOT / "locked_manifest.json"
CHECKPOINT = TRACE_ROOT / "checkpoint.json"
REPORT = SCREEN_ROOT / "trace_acquisition_report.json"
STOP_ERRORS = frozenset({"RPC_UNAUTHORIZED", "RPC_QUOTA_EXCEEDED", "RPC_CALLTRACER_UNSUPPORTED"})
CAST = Path.home() / ".foundry" / "bin" / "cast.exe"
CAST_TIMEOUT_SECONDS = 45


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_manifest() -> dict[str, Any]:
    source_rows = {
        row["sample_alias"]: row
        for row in _load(SOURCE_AUDIT)["samples"]
        if row["group"] != "HISTORICAL_DEV15"
        and row["post_safe_repair_level"] == "FILE_FUNCTION_LINE_EVALUABLE"
    }
    transactions = sorted(_dataset_locations(ROOT / "dataset"))
    aliases = {transaction: f"fault_audit_eth_{index:03d}" for index, transaction in enumerate(transactions, 1)}
    dev = set(_manifest_transactions(DEV15_MANIFEST))
    trees = offline._tree_index()
    samples = []
    for transaction in transactions:
        alias = aliases[transaction]
        if transaction in dev or alias not in source_rows:
            continue
        samples.append({
            "sample_alias": alias,
            "transaction_hash": transaction,
            "trace_status_at_lock": "LOCAL_TRACE_PRESENT" if transaction in trees else "TRACE_REQUEST_REQUIRED",
        })
    return {
        "schema_version": "week32_candidate_trace_lock_v1",
        "selection_rule": "NON_DEV15_AND_FUNCTION_FILE_LINE_EVALUABLE_WITHOUT_PREDICTION",
        "prediction_used_for_selection": False,
        "source_evaluable_sample_count": len(samples),
        "existing_local_trace_count": sum(row["trace_status_at_lock"] == "LOCAL_TRACE_PRESENT" for row in samples),
        "trace_request_sample_count": sum(row["trace_status_at_lock"] == "TRACE_REQUEST_REQUIRED" for row in samples),
        "samples": samples,
    }


def _load_or_lock() -> dict[str, Any]:
    current = _build_manifest()
    if LOCKED.exists():
        locked = _load(LOCKED)
        if locked != current:
            raise ValueError("WEEK32_TRACE_SELECTION_CHANGED")
        return locked
    atomic_json(LOCKED, current)
    return current


def _tree_path(transaction: str) -> Path:
    return TRACE_ROOT / "eth" / transaction / "call_tree.json"


def _valid_tree(path: Path) -> tuple[bool, int]:
    if not path.is_file():
        return False, 0
    try:
        graph = build_transaction_graph(load_call_tree(path))
    except (OSError, ValueError, TypeError):
        return False, 0
    return True, len(graph.nodes)


def _public_report(manifest: Mapping[str, Any], rows: list[dict[str, Any]], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in rows)
    ready = [row["sample_alias"] for row in rows if row["status"] in {"EXISTING_LOCAL_TRACE", "TRACE_ACQUIRED"}]
    report = {
        "schema_version": "week32_candidate_trace_acquisition_v1",
        "selection_rule": manifest["selection_rule"],
        "source_evaluable_sample_count": manifest["source_evaluable_sample_count"],
        "status_counts": dict(sorted(counts.items())),
        "ready_for_offline_preparation_count": len(ready),
        "ready_sample_aliases": ready,
        "chain_id_verified": bool(checkpoint["chain_id_verified"]),
        "external_call_counts": {
            "direct_rpc": int(checkpoint["rpc_request_count"]),
            "foundry_trace_replay": int(checkpoint.get("foundry_replay_count", 0)),
            "source": 0,
            "llm": 0,
        },
        "rows": rows,
    }
    atomic_json(REPORT, report)
    return report


def _cast_tree(transaction: str, rpc_url: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(CAST), "run", "--quick", "--disable-labels", "--color", "never", "--rpc-url", rpc_url, transaction],
        capture_output=True,
        text=True,
        timeout=CAST_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("CAST_REPLAY_FAILED")
    parsed = TraceParser().parse(completed.stdout).get("call_tree")
    tree = parsed.to_dict() if parsed is not None else None
    if not isinstance(tree, dict):
        raise ValueError("CAST_TRACE_PARSE_FAILED")
    return tree


def run(*, cast_fallback: bool = False) -> dict[str, Any]:
    manifest = _load_or_lock()
    checkpoint = _load(CHECKPOINT) if CHECKPOINT.exists() else {
        "schema_version": "week32_candidate_trace_checkpoint_v1",
        "completed": [],
        "failures": {},
        "rpc_request_count": 0,
        "foundry_replay_count": 0,
        "chain_id_verified": False,
    }
    config = load_chain_config("eth")
    rpc_url = str(config.get("rpc_url") or "")
    client = JsonRpcClient(rpc_url, max_requests=1 + manifest["trace_request_sample_count"])
    if not cast_fallback and not checkpoint["chain_id_verified"]:
        chain_id = client.call("eth_chainId", [])
        checkpoint["rpc_request_count"] += 1
        checkpoint["chain_id_verified"] = chain_id in {1, "0x1", "0X1"}
        atomic_json(CHECKPOINT, checkpoint)
        if not checkpoint["chain_id_verified"]:
            raise RuntimeError("RPC_CHAIN_ID_MISMATCH")

    for sample in manifest["samples"]:
        if sample["trace_status_at_lock"] != "TRACE_REQUEST_REQUIRED":
            continue
        alias = sample["sample_alias"]
        transaction = sample["transaction_hash"]
        valid, _node_count = _valid_tree(_tree_path(transaction))
        if valid:
            if alias not in checkpoint["completed"]:
                checkpoint["completed"].append(alias)
            continue
        try:
            if cast_fallback:
                checkpoint["foundry_replay_count"] = int(checkpoint.get("foundry_replay_count", 0)) + 1
                tree = _cast_tree(transaction, rpc_url)
            else:
                raw_frame = client.call("debug_traceTransaction", [transaction, {"tracer": "callTracer"}])
                checkpoint["rpc_request_count"] += 1
                tree = adapt_calltracer_call_tree(raw_frame)
            graph = build_transaction_graph(tree)
            atomic_json(_tree_path(transaction), {"call_tree": tree})
            checkpoint["completed"].append(alias)
            checkpoint["failures"].pop(alias, None)
            checkpoint.setdefault("node_counts", {})[alias] = len(graph.nodes)
        except RpcValidationError as error:
            checkpoint["rpc_request_count"] += 1
            checkpoint["failures"][alias] = error.category
            atomic_json(CHECKPOINT, checkpoint)
            if error.category in STOP_ERRORS:
                break
            continue
        except (TypeError, ValueError):
            checkpoint["failures"][alias] = "CALL_TREE_INVALID"
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            checkpoint["failures"][alias] = "CAST_REPLAY_FAILED"
        atomic_json(CHECKPOINT, checkpoint)

    existing_trees = offline._tree_index()
    rows: list[dict[str, Any]] = []
    for sample in manifest["samples"]:
        alias = sample["sample_alias"]
        transaction = sample["transaction_hash"]
        if sample["trace_status_at_lock"] == "LOCAL_TRACE_PRESENT":
            path = existing_trees[transaction][0]
            valid, node_count = _valid_tree(path)
            status = "EXISTING_LOCAL_TRACE" if valid else "EXISTING_TRACE_INVALID"
        else:
            valid, node_count = _valid_tree(_tree_path(transaction))
            status = "TRACE_ACQUIRED" if valid else checkpoint["failures"].get(alias, "TRACE_NOT_ACQUIRED")
        rows.append({"sample_alias": alias, "status": status, "trace_node_count": node_count})
    return _public_report(manifest, rows, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cast-fallback", action="store_true")
    args = parser.parse_args()
    result = run(cast_fallback=args.cast_fallback)
    print(json.dumps({key: result[key] for key in (
        "source_evaluable_sample_count",
        "status_counts",
        "ready_for_offline_preparation_count",
        "ready_sample_aliases",
        "external_call_counts",
    )}, ensure_ascii=False, indent=2))

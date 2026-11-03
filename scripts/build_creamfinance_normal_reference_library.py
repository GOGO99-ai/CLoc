"""Build the small CreamFinance normal-reference JSON prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.normal_reference_library import (  # noqa: E402
    build_query_features,
    build_reference_record,
    load_cached_or_replay_call_tree,
    load_faultless_candidates,
    retrieve_similar_references,
)
from src.data_collection.trace_parser import TraceParser  # noqa: E402
from src.data_collection.txn_replayer import TransactionReplayer  # noqa: E402
from cloc.runtime.execution_policy import ExecutionPolicy  # noqa: E402


CASE_PATH = next((PROJECT_ROOT / "dataset").rglob("CreamFinance_copy_4644_5055.json"))
PARSED_CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "all_tree"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prototypes" / "normal_reference_library" / "creamfinance"
FAULT_TX_HASH = "0xa9a1b8ea288eb9ad315088f17f7c7386b9989c95b4d13c81b69d5ddad7ffe61e"


def main() -> None:
    platform, candidates = load_faultless_candidates(CASE_PATH)
    config = load_chain_config("eth")
    replayer = TransactionReplayer(
        rpc_url=config["rpc_url"],
        cache_dir=str(PROJECT_ROOT / "data" / "cache" / "traces"),
        chain="eth",
        policy=ExecutionPolicy(allow_foundry=False, allow_rpc=True, allow_artifact_cache_write=True),
    )
    parser = TraceParser()
    attempts, library = [], []

    for tx_hash in candidates[:20]:
        if len(library) == 3:
            break
        try:
            tree, cache_used, trace_path = load_cached_or_replay_call_tree(tx_hash, parsed_cache_root=PARSED_CACHE_ROOT, replayer=replayer, trace_parser=parser)
            library.append(build_reference_record(f"creamfinance_normal_{len(library) + 1:03d}", tx_hash, platform, tree, cache_used=cache_used, trace_path=trace_path))
            attempts.append({"tx_hash": tx_hash, "status": "SUCCESS", "cache_used": cache_used, "failure_reason": None})
        except Exception as exc:  # each trace is intentionally independent
            attempts.append({"tx_hash": tx_hash, "status": "FAILED", "cache_used": False, "failure_reason": f"{type(exc).__name__}: {str(exc)[:240]}"})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write("trace_attempt_manifest.json", {"candidate_count": len(candidates), "attempts": attempts})
    _write("normal_reference_library.json", {"platform": platform, "references": library})

    demo: dict[str, object] = {"fault_tx_hash": FAULT_TX_HASH, "query_status": "FAILED", "failure_reason": None, "results": []}
    try:
        tree, cache_used, trace_path = load_cached_or_replay_call_tree(FAULT_TX_HASH, parsed_cache_root=PARSED_CACHE_ROOT, replayer=replayer, trace_parser=parser)
        query = build_query_features(tree)
        demo.update({"query_status": "SUCCESS", "trace_source": {"cache_used": cache_used, "trace_path": trace_path}, "query_features": query["retrieval_features"], "results": retrieve_similar_references(query, library)})
    except Exception as exc:
        demo["failure_reason"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    _write("retrieval_demo.json", demo)


def _write(name: str, payload: object) -> None:
    (OUTPUT_DIR / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

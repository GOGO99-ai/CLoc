"""Collect complete internal-function coverage for locked CLoc baseline cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloc_baselines.collector import CaseCollector, JsonRpcClient, SourcifyClient, TraceCache, write_case
from src.cloc_baselines.grace import GraceGraphCase, dump_grace_graph


def _safe_error(error: BaseException) -> str:
    text = str(error).splitlines()[0][:200]
    return f"{type(error).__name__}:{text}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "coverage")
    parser.add_argument("--grace-dir", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "grace")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "comparison" / "cloc_baselines" / "collector_cache")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest.get("cases") if isinstance(manifest, Mapping) else None
    if not isinstance(cases, list):
        raise ValueError("Invalid CLoc baseline manifest")
    if args.case_id:
        selected = [item for item in cases if item.get("case_id") in set(args.case_id)]
    else:
        selected = list(cases)
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    clients: dict[str, JsonRpcClient] = {}
    trace_cache = TraceCache(args.cache_dir / "traces")
    sourcify = SourcifyClient(args.cache_dir / "sourcify")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.grace_dir.mkdir(parents=True, exist_ok=True)
    completed = skipped = failed = 0
    for item in selected:
        case_id = item["case_id"]
        output = args.output_dir / f"{case_id}.json"
        grace_output = args.grace_dir / f"{case_id}.json"
        if output.exists() and grace_output.exists() and not args.overwrite:
            skipped += 1
            print(f"{case_id}: SKIPPED")
            continue
        chain_key = item["chain_key"]
        environment_key = "ETH_RPC_URL" if chain_key == "eth" else "BSC_RPC_URL"
        rpc_url = os.getenv(environment_key)
        if not rpc_url:
            print(f"{case_id}: FAILED:{environment_key}_MISSING")
            failed += 1
            continue
        client = clients.setdefault(chain_key, JsonRpcClient(rpc_url))
        collector = CaseCollector(rpc=client, sourcify=sourcify, trace_cache=trace_cache)
        try:
            case = collector.collect(item)
            write_case(case, output)
            dump_grace_graph(GraceGraphCase.from_coverage(case), grace_output)
        except Exception as exc:
            failed += 1
            error_path = args.output_dir / f"{case_id}.error.json"
            error_path.write_text(
                json.dumps(
                    {"case_id": case_id, "status": "FAILED", "reason": _safe_error(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"{case_id}: FAILED:{_safe_error(exc)}")
            continue
        completed += 1
        print(
            f"{case_id}: COMPLETED tests={len(case.tests)} functions={len(case.functions)} "
            f"lines={len(case.lines)}"
        )
    summary = {
        "selected": len(selected),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "rpc_request_count": sum(client.request_count for client in clients.values()),
        "sourcify_request_count": sourcify.request_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloc.runtime.execution_policy import ExecutionPolicy
from src.cloc_v2.address_extractor import extract_addresses_from_call_tree
from src.cloc_v2.config_loader import load_chain_config
from src.cloc_v2.output_layout import build_run_layout
from src.data_collection.contract_downloader import ContractDownloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch verified contract sources for an existing call tree.")
    parser.add_argument("--tx", required=True, help="Transaction hash")
    parser.add_argument("--chain", required=True, choices=["eth", "bsc"], help="Chain key")
    parser.add_argument("--use-proxy", action="store_true", help="Route explorer requests through a local proxy")
    parser.add_argument("--proxy-port", default="7897", help="Local proxy port")
    return parser.parse_args()


def find_call_tree_path(tx_hash: str, chain: str) -> Path:
    layout = build_run_layout(tx_hash, chain)
    primary = Path(layout["call_tree_file"])
    if primary.exists():
        return primary

    fallback_root = Path("data/cache/all_tree")
    fallback = list(fallback_root.rglob(f"{tx_hash.lower()}.json"))
    if fallback:
        return fallback[0]

    raise FileNotFoundError(
        f"call_tree.json not found for tx {tx_hash.lower()} under {primary} or {fallback_root}"
    )


def load_call_tree(call_tree_path: Path) -> dict:
    payload = json.loads(call_tree_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "call_tree" in payload and isinstance(payload["call_tree"], dict):
        return payload["call_tree"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported call tree payload in {call_tree_path}")


def main() -> int:
    args = parse_args()
    tx_hash = args.tx.lower()
    chain = args.chain.lower()

    call_tree_path = find_call_tree_path(tx_hash, chain)
    call_tree = load_call_tree(call_tree_path)

    layout = build_run_layout(tx_hash, chain)
    run_root = Path(layout["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    sources_root = run_root / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    legacy_cache_root = run_root / "legacy_contracts"
    legacy_cache_root.mkdir(parents=True, exist_ok=True)

    addresses = extract_addresses_from_call_tree(call_tree)
    address_list_path = Path(layout["address_list_file"])
    address_list_path.write_text(json.dumps(addresses, indent=2), encoding="utf-8")

    chain_config = load_chain_config(chain)
    trace_path = Path(layout["trace_file"])
    policy = ExecutionPolicy(
        allow_foundry=False,
        allow_rpc=False,
        allow_explorer=True,
        allow_artifact_cache_write=True,
    )

    report = {
        "tx_hash": tx_hash,
        "chain": chain,
        "trace_file": str(trace_path) if trace_path.exists() else None,
        "call_tree_file": str(call_tree_path),
        "address_count": len(addresses),
        "verified_count": 0,
        "unverified_count": 0,
        "failed_count": 0,
        "proxy_count": 0,
        "config_summary": {
            "base_url": chain_config.get("base_url"),
            "chainid": chain_config.get("chainid"),
            "rpc_url_present": bool(chain_config.get("rpc_url")),
            "api_key_present": bool(chain_config.get("key")),
        },
        "addresses": [],
    }

    for address in addresses:
        target_dir = sources_root / address
        downloader = ContractDownloader(
            chain=chain,
            address=[address],
            chain_config=chain_config,
            txn_hash=tx_hash,
            run_source_root=target_dir,
            legacy_cache_root=legacy_cache_root,
            policy=policy,
            use_proxy=args.use_proxy,
            proxy_port=args.proxy_port,
        )
        result = downloader.get_source_code_details(address, target_dir)
        status = result.get("status", "failed")
        if result.get("is_proxy"):
            report["proxy_count"] += 1
        if status == "verified":
            report["verified_count"] += 1
        elif status == "unverified":
            report["unverified_count"] += 1
        else:
            report["failed_count"] += 1

        report["addresses"].append(
            {
                "address": address,
                "status": status,
                "contract_name": result.get("contract_name"),
                "source_format": result.get("source_format", "unknown"),
                "is_proxy": result.get("is_proxy", False),
                "implementation": result.get("implementation"),
                "saved_dir": result.get("saved_dir"),
                "error": result.get("error"),
            }
        )

    report_path = run_root / "source_download_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(report_path), "address_count": len(addresses)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

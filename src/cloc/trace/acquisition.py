"""Acquire JSON callTracer evidence without converting it to lossy text."""
import json
import re
from pathlib import Path
import requests
from cloc.config import load_chain_config


def acquire_trace(transaction_hash, chain, cache_root):
    """Read the trace cache or request debug_traceTransaction using a local endpoint."""
    if not isinstance(transaction_hash, str) or re.fullmatch(r"0x[0-9a-fA-F]{64}", transaction_hash) is None:
        raise ValueError("TRANSACTION_HASH_INVALID")
    config = load_chain_config(chain)
    transaction_hash = transaction_hash.lower()
    path = Path(cache_root) / chain / (transaction_hash + ".json")
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(cached, dict) or set(cached) != {"schema_version", "chain", "transaction_hash", "trace"} or cached["schema_version"] != "trace_cache_v1" or cached["chain"] != chain or cached["transaction_hash"] != transaction_hash:
            raise ValueError("TRACE_CACHE_IDENTITY_INVALID")
        return cached["trace"]
    if not config["rpc_url"]:
        raise ValueError("RPC_URL_MISSING")
    try:
        response = requests.post(config["rpc_url"], timeout=120, allow_redirects=False,
            json={"jsonrpc": "2.0", "id": 1, "method": "debug_traceTransaction",
                  "params": [transaction_hash, {"tracer": "callTracer"}]})
        if response.status_code != 200:
            raise ValueError("RPC_HTTP_FAILURE")
        payload = response.json()
        if not isinstance(payload, dict) or "error" in payload or not isinstance(payload.get("result"), dict):
            raise ValueError("RPC_TRACE_UNAVAILABLE")
        frame = payload["result"]
        if not frame.get("type"):
            raise ValueError("RPC_TRACE_INVALID")
    except (requests.RequestException, OSError):
        raise ValueError("RPC_TRANSPORT_FAILED") from None
    except (TypeError, json.JSONDecodeError):
        raise ValueError("RPC_RESPONSE_INVALID") from None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "trace_cache_v1", "chain": chain,
                              "transaction_hash": transaction_hash, "trace": frame}, indent=2) + "\n", encoding="utf-8")
    return frame

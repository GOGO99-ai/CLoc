from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict

from config import SUPPORTED_CHAINS


PLATFORM_TO_CHAIN = {
    "BNBChain": "bsc",
    "Ethereum": "eth",
}

ENV_OVERRIDES = {
    "eth": {
        "rpc_url": "ETH_RPC_URL",
        "key": "ETHERSCAN_API_KEY",
    },
    "bsc": {
        "rpc_url": "BSC_RPC_URL",
        "key": "BSCSCAN_API_KEY",
    },
}


def map_platform_to_chain(platform: str) -> str:
    try:
        return PLATFORM_TO_CHAIN[platform]
    except KeyError as exc:
        raise ValueError(f"Unsupported platform: {platform}") from exc


def load_chain_config(chain: str) -> Dict[str, Any]:
    chain_key = chain.lower()
    if chain_key not in SUPPORTED_CHAINS:
        raise ValueError(f"Unsupported chain: {chain}")

    config = deepcopy(SUPPORTED_CHAINS[chain_key])
    overrides = ENV_OVERRIDES.get(chain_key, {})

    for field, env_name in overrides.items():
        env_value = os.getenv(env_name)
        if env_value:
            config[field] = env_value

    config["chain"] = chain_key
    config["rpc_source"] = "env" if os.getenv(overrides.get("rpc_url", "")) else "config.py"
    config["api_key_source"] = "env" if os.getenv(overrides.get("key", "")) else "config.py"
    return config


def summarize_chain_config(chain_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chain": chain_config["chain"],
        "chainid": chain_config.get("chainid"),
        "base_url": chain_config.get("base_url"),
        "rpc_url_present": bool(chain_config.get("rpc_url")),
        "rpc_source": chain_config.get("rpc_source"),
        "api_key_present": bool(chain_config.get("key")),
        "api_key_source": chain_config.get("api_key_source"),
        "api_key_masked": _mask_secret(chain_config.get("key", "")),
    }


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"

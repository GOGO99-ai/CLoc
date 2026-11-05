"""Load chain endpoints and credentials from the process environment."""
import os

CHAIN_IDS = {"eth": 1, "bsc": 56}


def load_chain_config(chain: str) -> dict:
    """Return the acquisition configuration without reading private files."""
    if chain not in CHAIN_IDS:
        raise ValueError("chain must be eth or bsc")
    key_name = "ETHERSCAN_API_KEY" if chain == "eth" else "BSCSCAN_API_KEY"
    return {
        "chain": chain,
        "chainid": CHAIN_IDS[chain],
        "base_url": "https://api.etherscan.io/v2/api",
        "rpc_url": os.environ.get(f"{chain.upper()}_RPC_URL", ""),
        "key": os.environ.get(key_name) or os.environ.get("ETHERSCAN_API_KEY", ""),
    }


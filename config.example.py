"""Portable configuration template. Copy to config.py; supply secrets via environment."""
import os

SUPPORTED_CHAINS = {
    "eth": {"base_url": "https://api.etherscan.io/v2/api", "key": os.getenv("ETHERSCAN_API_KEY", ""), "rpc_url": os.getenv("ETH_RPC_URL", ""), "chainid": "1"},
    "bsc": {"base_url": "https://api.etherscan.io/v2/api", "key": os.getenv("BSCSCAN_API_KEY", ""), "rpc_url": os.getenv("BSC_RPC_URL", ""), "chainid": "56"},
}
PROXIES = {k:v for k,v in {"http": os.getenv("HTTP_PROXY"), "https": os.getenv("HTTPS_PROXY")}.items() if v}
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "180"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

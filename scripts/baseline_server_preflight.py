"""Check server prerequisites without printing credentials or secret values."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path


DEFAULT_ENVIRONMENT_KEYS = (
    "ETH_RPC_URL",
    "BSC_RPC_URL",
    "ETHERSCAN_API_KEY",
    "BSCSCAN_API_KEY",
)


def build_preflight(environment_keys: tuple[str, ...] = DEFAULT_ENVIRONMENT_KEYS) -> dict[str, object]:
    commands = {
        command: shutil.which(command) is not None
        for command in ("git", "java", "node", "npm", "solc")
    }
    packages = {
        package: importlib.util.find_spec(package) is not None
        for package in ("numpy", "pandas", "torch", "torch_geometric", "networkx", "requests")
    }
    cuda = {"torch_available": False, "cuda_available": False, "device_count": 0}
    if packages["torch"]:
        import torch

        cuda = {
            "torch_available": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    environment = {
        key: {"present": bool(os.getenv(key)), "value_disclosed": False}
        for key in environment_keys
    }
    return {
        "schema_version": "cloc_baseline_server_preflight_v1",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "commands": commands,
        "packages": packages,
        "accelerator": cuda,
        "environment": environment,
        "secret_values_logged": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = build_preflight()
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

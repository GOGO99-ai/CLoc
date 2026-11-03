"""Validate that the baseline environment can execute a small CUDA workload."""

from __future__ import annotations

import json

import torch


def main() -> int:
    payload: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        result = left @ left.T
        torch.cuda.synchronize(device)
        payload.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "matmul_checksum": float(result.sum().cpu()),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["cuda_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

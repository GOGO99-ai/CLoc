"""Run the public DAppFL implementation on the shared evaluable CLoc subset.

The upstream repository ships processed PyG graphs, but its CLI references two
undefined argparse attributes and only prints fold-averaged metrics.  This
adapter keeps the upstream model, loss, data, split seed, and hyperparameters,
while emitting case-level rankings and metrics over the union of held-out
cases.  It intentionally does not rebuild graphs or contact chain/explorer
services.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


DAPPFL_CASES = (
    "RikkeiFinance",
    "SpaceGodzilla",
    "bZx0215",
    "SaddleFinance",
    "Cover",
    "MerlinLab",
    "Bearn",
    "FormationFi",
    "IndexedFinance",
    "MonoX",
    "Nmbplatform",
    "InverseFinance",
    "PunkProtocol",
    "FEGtoken",
    "FortressProtocol",
    "XCarnival",
    "QubitFinance",
    "SushiSwap",
    "ForceDAO",
    "CreamFinance",
    "RevestFinance",
    "LiFi",
)


def _link_once(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Refusing to replace unrelated subset artifact: {destination}")
        return
    destination.symlink_to(source.resolve())


def prepare_subset(upstream_data: Path, subset_data: Path) -> None:
    raw_dir = subset_data / "raw"
    processed_dir = subset_data / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name in DAPPFL_CASES:
        raw_source = upstream_data / "raw" / f"{name}.json"
        processed_source = upstream_data / "processed" / f"{name}.pt"
        if not raw_source.is_file() or not processed_source.is_file():
            raise FileNotFoundError(f"DAppFL artifact missing for {name}")
        _link_once(raw_source, raw_dir / raw_source.name)
        _link_once(processed_source, processed_dir / processed_source.name)
    for name in ("pre_filter.pt", "pre_transform.pt"):
        source = upstream_data / "processed" / name
        if source.is_file():
            _link_once(source, processed_dir / name)


class PublishedGraphDataset:
    """Load the published ``.pt`` files without importing the crawler stack."""

    def __init__(self, root: Path, torch: Any):
        self.root = root
        self.torch = torch
        self.processed_file_names = [
            str(root / "processed" / f"{name}.pt") for name in DAPPFL_CASES
        ]
        missing = [path for path in self.processed_file_names if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Published DAppFL graph missing: {missing[0]}")
        self._metadata: tuple[list[str], list[tuple[str, str, str]]] | None = None

    def __len__(self) -> int:
        return len(self.processed_file_names)

    def __getitem__(self, index: Any):
        if isinstance(index, (list, tuple)) or getattr(index, "ndim", 0) > 0:
            return [self[int(item)] for item in index]
        data = self.torch.load(
            self.processed_file_names[int(index)], map_location="cpu", weights_only=False
        )
        for node_type in data.node_types:
            data[node_type].x = data[node_type].x.float()
            if node_type != "Case":
                data[node_type].mask = data[node_type].x[:, 1] > 0
        for edge_type in data.edge_types:
            data[edge_type].edge_index = data[edge_type].edge_index.long()
            if data[edge_type].get("edge_attr") is not None:
                data[edge_type].edge_attr = data[edge_type].edge_attr.float()
        return data

    @property
    def metadata(self) -> tuple[list[str], list[tuple[str, str, str]]]:
        if self._metadata is None:
            node_types: set[str] = set()
            edge_types: set[tuple[str, str, str]] = set()
            for index in range(len(self)):
                graph = self[index]
                node_types.update(graph.node_types)
                edge_types.update(graph.edge_types)
            self._metadata = (sorted(node_types), sorted(edge_types))
        return self._metadata


def _case_metrics(scores: list[float], labels: list[int]) -> dict[str, Any]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("Invalid DAppFL prediction vectors")
    fault_indices = [index for index, value in enumerate(labels) if int(value) == 1]
    if not fault_indices:
        raise ValueError("Held-out DAppFL case has no fault label")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
    fault_ranks = [rank_by_index[index] for index in fault_indices]
    return {
        "first_fault_rank": float(min(fault_ranks)),
        "average_fault_rank": float(sum(fault_ranks) / len(fault_ranks)),
        "fault_count": len(fault_indices),
        "candidate_count": len(scores),
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    if not count:
        raise ValueError("No DAppFL cases were evaluated")
    return {
        "recall": {
            "recall_at_1": sum(row["first_fault_rank"] <= 1 for row in cases) / count,
            "recall_at_3": sum(row["first_fault_rank"] <= 3 for row in cases) / count,
            "recall_at_5": sum(row["first_fault_rank"] <= 5 for row in cases) / count,
        },
        "hit_counts": {
            "top_1": sum(row["first_fault_rank"] <= 1 for row in cases),
            "top_3": sum(row["first_fault_rank"] <= 3 for row in cases),
            "top_5": sum(row["first_fault_rank"] <= 5 for row in cases),
        },
        "mfr": sum(row["first_fault_rank"] for row in cases) / count,
        "mar": sum(row["average_fault_rank"] for row in cases) / count,
    }


def _kfold_indices(count: int, folds: int, seed: int, np: Any):
    if folds < 2 or folds > count:
        raise ValueError("fold count must be between 2 and the dataset size")
    shuffled = np.arange(count)
    np.random.RandomState(seed).shuffle(shuffled)
    sizes = np.full(folds, count // folds, dtype=int)
    sizes[: count % folds] += 1
    offset = 0
    for size in sizes:
        test_ids = shuffled[offset : offset + size].tolist()
        held_out = set(test_ids)
        train_ids = [index for index in range(count) if index not in held_out]
        yield train_ids, test_ids
        offset += int(size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dappfl-root", type=Path, required=True)
    parser.add_argument("--subset-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    upstream_root = args.dappfl_root.resolve()
    subset_data = args.subset_data.resolve()
    output = args.output.resolve()
    upstream_data = upstream_root / "data"
    prepare_subset(upstream_data, subset_data)
    sys.path.insert(0, str(upstream_root))
    os.chdir(upstream_root)

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from algos.focal_loss import FocalLoss
    import models.HGT as hgt_module

    # The public code targets an older PyG whose HGTConv accepted
    # ``group='sum'``.  Current PyG always uses additive aggregation here and
    # rejects that removed keyword, so discard only that no-op compatibility
    # argument while keeping the upstream HGT implementation intact.
    upstream_hgt_conv = hgt_module.HGTConv

    def compatible_hgt_conv(*conv_args: Any, **conv_kwargs: Any):
        conv_kwargs.pop("group", None)
        return upstream_hgt_conv(*conv_args, **conv_kwargs)

    hgt_module.HGTConv = compatible_hgt_conv
    HGT = hgt_module.HGT

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = PublishedGraphDataset(subset_data, torch)
    if len(dataset) != len(DAPPFL_CASES):
        raise RuntimeError(f"Expected {len(DAPPFL_CASES)} DAppFL cases, found {len(dataset)}")

    rows: list[dict[str, Any]] = []
    splitter = _kfold_indices(len(dataset), args.folds, args.seed, np)
    for fold, (train_ids, test_ids) in enumerate(splitter):
        model = HGT(
            hidden_channels=args.hidden_channels,
            out_channels=2,
            metadata=dataset.metadata,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        criterion = FocalLoss(alpha=1, gamma=2)
        train_loader = DataLoader(
            dataset[train_ids],
            batch_size=args.batch_size,
            drop_last=True,
            pin_memory=device.type == "cuda",
            shuffle=True,
        )
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                mask = batch["FunctionDefinition"].mask
                logits = model(batch)["FunctionDefinition"][mask]
                loss = criterion(logits, batch["FunctionDefinition"].y[mask])
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            print(f"dappfl fold={fold + 1}/{args.folds} epoch={epoch + 1}/{args.epochs} loss={total_loss:.6f}", flush=True)

        model.eval()
        with torch.no_grad():
            for case_index in test_ids:
                batch = dataset[case_index].to(device)
                mask = batch["FunctionDefinition"].mask
                logits = model(batch)["FunctionDefinition"][mask]
                scores = functional.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
                labels = batch["FunctionDefinition"].y[mask].detach().cpu().tolist()
                case_name = Path(dataset.processed_file_names[case_index]).stem
                rows.append({"case_id": case_name, "fold": fold, **_case_metrics(scores, labels)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: row["case_id"])
    metrics = _aggregate(rows)
    payload = {
        "schema_version": "cloc_dappfl_result_v1",
        "method": "dappfl_public_code_adaptation",
        "upstream_commit": "66acd9c",
        "protocol": "OFFICIAL_SHUFFLED_5_FOLD_CASE_LEVEL_WEIGHTED",
        "expected_case_count": 40,
        "evaluated_case_count": len(rows),
        "partial": True,
        "seed": args.seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "hyperparameters": {
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "metrics": metrics,
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    recall = metrics["recall"]
    print(
        "dappfl: "
        f"Recall@1={recall['recall_at_1']:.4f} "
        f"Recall@3={recall['recall_at_3']:.4f} "
        f"Recall@5={recall['recall_at_5']:.4f} "
        f"MFR={metrics['mfr']:.4f} MAR={metrics['mar']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

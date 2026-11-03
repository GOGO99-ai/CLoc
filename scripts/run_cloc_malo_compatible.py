"""Run the public MALO localization network on the common CLoc subset.

The public MALO repository contains the localization model, but omits the
downloaders/settings and all processed graph artifacts.  This adapter therefore
uses the public ``models/ours.py`` network with the already published DAppFL
graphs for the same cases.  DAppFL's money-flow features are aggregated into a
contract graph and CLoc function-call edges connect contracts.  The result is a
transparent MALO-compatible reimplementation, not an official MALO rerun.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


CASE_MAP = {
    "RikkeiFinance": "001_rikkei_finance",
    "SpaceGodzilla": "002_spacegodzilla",
    "bZx0215": "003_bzx_0215",
    "SaddleFinance": "004_saddle_finance",
    "Cover": "007_cover_protocol",
    "MerlinLab": "008_merlin_labs",
    "Bearn": "011_bearn",
    "FormationFi": "012_formation_fi",
    "IndexedFinance": "013_indexed_finance",
    "MonoX": "014_monox",
    "Nmbplatform": "015_nmbplatform",
    "InverseFinance": "017_inversefinance",
    "PunkProtocol": "018_punk_protocol",
    "FEGtoken": "020_fegtoken",
    "FortressProtocol": "021_fortress_protocol",
    "XCarnival": "023_xcarnival",
    "QubitFinance": "025_qubit_finance",
    "SushiSwap": "030_sushiswap",
    "ForceDAO": "031_force_dao",
    "CreamFinance": "033_creamfinance",
    "RevestFinance": "035_revest_finance",
    "LiFi": "036_li_fi",
}


class PublishedGraphDataset:
    """Load DAppFL's published graphs without importing its crawler stack."""

    def __init__(self, root: Path, torch: Any):
        self.root = root
        self.torch = torch
        self.processed_file_names = [
            str(root / "processed" / f"{name}.pt") for name in CASE_MAP
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


def _load_malo_model(model_path: Path):
    spec = importlib.util.spec_from_file_location("malo_public_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MALO model: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Ours


def _function_positions(graph: Any) -> list[str]:
    positions = graph["FunctionDefinition"].position
    return [str(value) for value in positions]


def _contract_graph(graph: Any, coverage: dict[str, Any], torch: Any, Data: Any):
    positions = _function_positions(graph)
    addresses = sorted({position.split("#", 1)[0].lower() for position in positions})
    if not addresses:
        raise ValueError("MALO-compatible graph has no contract address")
    address_to_index = {address: index for index, address in enumerate(addresses)}
    features = [[0.0, 0.0, 0.0] for _ in addresses]
    function_x = graph["FunctionDefinition"].x.detach().cpu()
    for index, position in enumerate(positions):
        address = position.split("#", 1)[0].lower()
        row = features[address_to_index[address]]
        row[0] += 1.0
        if function_x.shape[1] >= 2:
            row[1] += float(function_x[index, -2])
            row[2] = max(row[2], abs(float(function_x[index, -1])))

    edge_counts: dict[tuple[str, str], int] = {}
    for source, target in coverage.get("call_edges", []):
        source_address = str(source).split("#", 1)[0].lower()
        target_address = str(target).split("#", 1)[0].lower()
        if source_address in address_to_index and target_address in address_to_index:
            key = (source_address, target_address)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    for address in addresses:
        edge_counts.setdefault((address, address), 1)

    ordered_edges = sorted(edge_counts.items())
    edge_index = torch.tensor(
        [[address_to_index[source], address_to_index[target]] for (source, target), _ in ordered_edges],
        dtype=torch.long,
    ).t().contiguous()
    edge_attr = torch.tensor(
        [[math.log1p(count), float(source != target)] for (source, target), count in ordered_edges],
        dtype=torch.float32,
    )
    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        addresses=addresses,
    )


def _case_metrics(scores: list[float], labels: list[int]) -> dict[str, Any]:
    fault_indices = [index for index, value in enumerate(labels) if int(value) == 1]
    if not fault_indices:
        raise ValueError("MALO-compatible held-out case has no fault label")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
    ranks = [rank_by_index[index] for index in fault_indices]
    return {
        "first_fault_rank": float(min(ranks)),
        "average_fault_rank": float(sum(ranks) / len(ranks)),
        "fault_count": len(ranks),
        "candidate_count": len(scores),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise ValueError("No MALO-compatible cases were evaluated")
    return {
        "recall": {
            "recall_at_1": sum(row["first_fault_rank"] <= 1 for row in rows) / count,
            "recall_at_3": sum(row["first_fault_rank"] <= 3 for row in rows) / count,
            "recall_at_5": sum(row["first_fault_rank"] <= 5 for row in rows) / count,
        },
        "hit_counts": {
            "top_1": sum(row["first_fault_rank"] <= 1 for row in rows),
            "top_3": sum(row["first_fault_rank"] <= 3 for row in rows),
            "top_5": sum(row["first_fault_rank"] <= 5 for row in rows),
        },
        "mfr": sum(row["first_fault_rank"] for row in rows) / count,
        "mar": sum(row["average_fault_rank"] for row in rows) / count,
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


def _criterion(logits: Any, labels: Any, functional: Any):
    log_probabilities = functional.log_softmax(logits.flatten(), dim=0)
    base_loss = -(labels.float() @ log_probabilities)
    return base_loss * ((1.0 - (-base_loss).exp()) ** 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dappfl-root", type=Path, required=True)
    parser.add_argument("--malo-root", type=Path, required=True)
    parser.add_argument("--subset-data", type=Path, required=True)
    parser.add_argument("--coverage-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--edge-dims", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    dappfl_root = args.dappfl_root.resolve()
    malo_root = args.malo_root.resolve()
    subset_data = args.subset_data.resolve()
    coverage_dir = args.coverage_dir.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(dappfl_root))
    os.chdir(dappfl_root)

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from torch_geometric.data import Data

    Ours = _load_malo_model(malo_root / "models" / "ours.py")
    original_selector = Ours.get_ytype2contract

    def device_selector(function_graph: Any, transfer_graph: Any):
        return original_selector(function_graph, transfer_graph).to(
            function_graph["FunctionDefinition"].x.device
        )

    Ours.get_ytype2contract = staticmethod(device_selector)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = PublishedGraphDataset(subset_data, torch)
    if len(dataset) != len(CASE_MAP):
        raise RuntimeError(f"Expected {len(CASE_MAP)} MALO-compatible cases, found {len(dataset)}")

    case_names = [Path(path).stem for path in dataset.processed_file_names]
    coverage_by_name = {
        name: json.loads((coverage_dir / f"{CASE_MAP[name]}.json").read_text(encoding="utf-8"))
        for name in case_names
    }
    transfer_by_name = {
        name: _contract_graph(dataset[index], coverage_by_name[name], torch, Data)
        for index, name in enumerate(case_names)
    }

    rows: list[dict[str, Any]] = []
    splitter = _kfold_indices(len(dataset), args.folds, args.seed, np)
    for fold, (train_ids, test_ids) in enumerate(splitter):
        model = Ours(
            hidden_channels=args.hidden_channels,
            edge_dims=args.edge_dims,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            metadata=dataset.metadata,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        model.train()
        for epoch in range(args.epochs):
            order = train_ids[:]
            random.Random(args.seed + fold * 1000 + epoch).shuffle(order)
            total_loss = 0.0
            for case_index in order:
                name = case_names[case_index]
                function_graph = copy.deepcopy(dataset[case_index]).to(device)
                function_graph.y_ids = _function_positions(function_graph)
                transfer_graph = copy.deepcopy(transfer_by_name[name]).to(device)
                mask = function_graph["FunctionDefinition"].mask
                labels = function_graph["FunctionDefinition"].y[mask]
                optimizer.zero_grad(set_to_none=True)
                logits = model(function_graph, transfer_graph).flatten()[mask]
                loss = _criterion(logits, labels, functional)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            print(f"malo fold={fold + 1}/{args.folds} epoch={epoch + 1}/{args.epochs} loss={total_loss:.6f}", flush=True)

        model.eval()
        with torch.no_grad():
            for case_index in test_ids:
                name = case_names[case_index]
                function_graph = copy.deepcopy(dataset[case_index]).to(device)
                function_graph.y_ids = _function_positions(function_graph)
                transfer_graph = copy.deepcopy(transfer_by_name[name]).to(device)
                mask = function_graph["FunctionDefinition"].mask
                scores = model(function_graph, transfer_graph).flatten()[mask].detach().cpu().tolist()
                labels = function_graph["FunctionDefinition"].y[mask].detach().cpu().tolist()
                rows.append({"case_id": name, "fold": fold, **_case_metrics(scores, labels)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: row["case_id"])
    metrics = _aggregate(rows)
    payload = {
        "schema_version": "cloc_malo_compatible_result_v1",
        "method": "malo_public_model_cloc_adaptation",
        "upstream_commit": "bab0d1b",
        "protocol": "SHUFFLED_5_FOLD_CASE_LEVEL_WEIGHTED",
        "expected_case_count": 40,
        "evaluated_case_count": len(rows),
        "partial": True,
        "official_malo_rerun": False,
        "adaptation_limitations": [
            "PUBLIC_REPOSITORY_OMITS_PROCESSED_MALO_GRAPHS",
            "PUBLIC_REPOSITORY_OMITS_DOWNLOADER_AND_SETTINGS_MODULES",
            "DAPPFL_PUBLISHED_FUNCTION_GRAPHS_REUSED_FOR_COMMON_CASES",
            "DAPPFL_MONEY_FLOW_FEATURES_AGGREGATED_TO_CONTRACT_GRAPH",
            "CODEBERT_TRANSFORM_NOT_REAPPLIED",
        ],
        "seed": args.seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "hyperparameters": {
            "hidden_channels": args.hidden_channels,
            "edge_dims": args.edge_dims,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
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
        "malo-compatible: "
        f"Recall@1={recall['recall_at_1']:.4f} "
        f"Recall@3={recall['recall_at_3']:.4f} "
        f"Recall@5={recall['recall_at_5']:.4f} "
        f"MFR={metrics['mfr']:.4f} MAR={metrics['mar']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

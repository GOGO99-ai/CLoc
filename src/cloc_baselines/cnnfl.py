"""Faithful per-case CNNFL architecture over the normalized coverage matrix."""

from __future__ import annotations

from typing import Any

from .ranking import RankedFunction, rank_scores
from .schema import CoverageCase


def rank_cnnfl_case(
    case: CoverageCase,
    *,
    seed: int = 0,
    epochs: int = 20,
    batch_size: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 5.0e-4,
    device: str | None = None,
    malo_label_leakage_compatibility: bool = False,
) -> tuple[RankedFunction, ...]:
    """Train CNNFL exactly once for one DApp and rank virtual one-hot tests.

    The default input is ordinary binary function coverage.  The public MALO
    repository writes ``-1`` into cells where a failing test covers a labeled
    fault function.  That directly exposes evaluation labels during training,
    so it is available only behind an explicit compatibility flag.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - depends on server environment
        raise RuntimeError("CNNFL requires PyTorch") from exc

    candidates = case.candidate_function_ids
    candidate_count = len(candidates)
    if candidate_count < 4:
        raise ValueError("CNNFL requires at least four candidate functions for two pooling layers")
    index = {function_id: position for position, function_id in enumerate(candidates)}
    coverage_rows: list[list[float]] = []
    outcomes: list[float] = []
    faults = set(case.fault_function_ids)
    for test in case.tests:
        row = [0.0] * candidate_count
        for function_id in test.covered_function_ids:
            if function_id not in index:
                continue
            value = 1.0
            if malo_label_leakage_compatibility and test.outcome == "fail" and function_id in faults:
                value = -1.0
            row[index[function_id]] = value
        coverage_rows.append(row)
        outcomes.append(1.0 if test.outcome == "fail" else 0.0)

    class CNNFL(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 3), stride=1, padding=(0, 1))
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
            self.fc1 = nn.Linear(64 * (width // 4), 1024)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(1024, 1024)
            self.relu4 = nn.ReLU()
            self.fc3 = nn.Linear(1024, 1)

        def forward(self, value: Any) -> Any:
            value = self.pool1(self.relu1(self.conv1(value)))
            value = self.pool2(self.relu2(self.conv2(value)))
            value = value.reshape(-1, 64 * value.size(3))
            value = self.relu3(self.fc1(value))
            value = self.relu4(self.fc2(value))
            return self.fc3(value)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    execution_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    features = torch.tensor(coverage_rows, dtype=torch.float32)
    labels = torch.tensor(outcomes, dtype=torch.float32)
    loader = DataLoader(list(zip(features, labels)), batch_size=batch_size, shuffle=False)
    model = CNNFL(candidate_count).to(execution_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(execution_device)
            target_batch = target_batch.to(execution_device)
            optimizer.zero_grad()
            logits = model(feature_batch.unsqueeze(1).unsqueeze(1))
            loss = criterion(torch.sigmoid(logits), target_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        virtual_tests = torch.eye(candidate_count, dtype=torch.float32, device=execution_device)
        scores_tensor = torch.sigmoid(model(virtual_tests.unsqueeze(1).unsqueeze(1))[:, 0])
        scores = {
            function_id: float(scores_tensor[position].detach().cpu())
            for function_id, position in index.items()
        }
    return rank_scores(case, scores)

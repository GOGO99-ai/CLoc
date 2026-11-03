"""Reproducible function-level baseline harness for the CLoc dataset."""

from .manifest import build_cloc_baseline_manifest
from .metrics import evaluate_rankings
from .ranking import rank_case
from .schema import CoverageCase, FunctionRecord, LineRecord, TestRecord

__all__ = [
    "CoverageCase",
    "FunctionRecord",
    "LineRecord",
    "TestRecord",
    "build_cloc_baseline_manifest",
    "evaluate_rankings",
    "rank_case",
]

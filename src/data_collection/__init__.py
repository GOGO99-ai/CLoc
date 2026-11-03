"""
Data collection module for CLoc.

Imports are resolved lazily so the package remains importable while
Increment 1 focuses on dry-run setup instead of the full legacy pipeline.
"""

__all__ = [
    "TransactionSequencer",
    "TransactionInfoCollector",
    "ContractDownloader",
    "ContractInfoCollector",
    "TraceParser",
]


def __getattr__(name):
    if name == "TransactionSequencer":
        from .txn_sequencer import TransactionSequencer

        return TransactionSequencer
    if name == "TransactionInfoCollector":
        from .txn_info_collector import TransactionInfoCollector

        return TransactionInfoCollector
    if name == "ContractDownloader":
        from .contract_downloader import ContractDownloader

        return ContractDownloader
    if name == "ContractInfoCollector":
        from .contract_info_collector import ContractInfoCollector

        return ContractInfoCollector
    if name == "TraceParser":
        from .trace_parser import TraceParser

        return TraceParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

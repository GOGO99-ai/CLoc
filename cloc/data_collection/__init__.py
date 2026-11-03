"""
Compatibility re-exports for legacy data_collection imports.
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
        from src.data_collection import TransactionSequencer

        return TransactionSequencer
    if name == "TransactionInfoCollector":
        from src.data_collection import TransactionInfoCollector

        return TransactionInfoCollector
    if name == "ContractDownloader":
        from src.data_collection import ContractDownloader

        return ContractDownloader
    if name == "ContractInfoCollector":
        from src.data_collection import ContractInfoCollector

        return ContractInfoCollector
    if name == "TraceParser":
        from src.data_collection import TraceParser

        return TraceParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

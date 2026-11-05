"""Control explicit external access during artifact acquisition."""
from __future__ import annotations

from dataclasses import dataclass


class ExternalAccessDenied(RuntimeError):
    def __init__(self, capability: str, flag: str, artifact: str):
        super().__init__(f"External access denied for {capability} while resolving {artifact}; enable {flag}.")
        self.capability = capability
        self.flag = flag
        self.artifact = artifact


class RequiredArtifactUnavailable(RuntimeError):
    def __init__(self, artifact: str, name: str, reason: str):
        super().__init__(f"Required artifact unavailable: {artifact} ({name}) due to {reason}.")
        self.artifact = artifact
        self.name = name
        self.reason = reason


@dataclass(frozen=True)
class ExecutionPolicy:
    allow_foundry: bool = False
    allow_rpc: bool = False
    allow_explorer: bool = False
    allow_artifact_cache_write: bool = True

    @classmethod
    def current_live_default(cls) -> "ExecutionPolicy":
        return cls()

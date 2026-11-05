"""Index verified cache bindings and reject conflicting identities."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = frozenset({"VERIFIED_SOURCE_CACHE_HIT", "VERIFIED_SOURCE_CACHE_MISS", "VERIFIED_SOURCE_CACHE_CONFLICT", "VERIFIED_SOURCE_CACHE_INVALID"})


def _address(value: Any) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value.lower()) is None:
        raise ValueError("SOURCE_CACHE_ADDRESS_INVALID")
    return value.lower()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class VerifiedSourceCacheBinding:
    blockchain_chain_id: int
    runtime_address: str = field(repr=False)
    effective_address: str = field(repr=False)
    requested_address: str = field(repr=False)
    artifact_address: str = field(repr=False)
    artifact_relative_path: str = field(repr=False)
    source_sha256: str = field(repr=False)
    verification_status: str
    proxy_resolution_source: str
    solidity_file_count: int
    cache_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.blockchain_chain_id, int) or isinstance(self.blockchain_chain_id, bool) or self.blockchain_chain_id <= 0:
            raise ValueError("SOURCE_CACHE_CHAIN_ID_INVALID")
        object.__setattr__(self, "runtime_address", _address(self.runtime_address))
        object.__setattr__(self, "effective_address", _address(self.effective_address))
        object.__setattr__(self, "requested_address", _address(self.requested_address))
        object.__setattr__(self, "artifact_address", _address(self.artifact_address))
        if not isinstance(self.artifact_relative_path, str) or not self.artifact_relative_path or Path(self.artifact_relative_path).is_absolute() or ".." in Path(self.artifact_relative_path).parts:
            raise ValueError("SOURCE_CACHE_ARTIFACT_PATH_INVALID")
        if not isinstance(self.source_sha256, str) or _SHA.fullmatch(self.source_sha256) is None:
            raise ValueError("SOURCE_CACHE_SHA_INVALID")
        if self.verification_status != "verified" or not isinstance(self.proxy_resolution_source, str) or not self.proxy_resolution_source:
            raise ValueError("SOURCE_CACHE_BINDING_STATUS_INVALID")
        if not isinstance(self.solidity_file_count, int) or self.solidity_file_count < 1 or not isinstance(self.cache_version, str) or not self.cache_version:
            raise ValueError("SOURCE_CACHE_BINDING_METADATA_INVALID")


@dataclass(frozen=True)
class VerifiedSourceCacheLookup:
    status: str
    binding: VerifiedSourceCacheBinding | None = field(default=None, repr=False)
    lookup_kind: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATUSES or (self.status == "VERIFIED_SOURCE_CACHE_HIT") != (self.binding is not None):
            raise ValueError("SOURCE_CACHE_LOOKUP_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "lookup_kind": self.lookup_kind, "source_sha_alias": self.binding.source_sha256[:12] if self.binding else None, "solidity_file_count": self.binding.solidity_file_count if self.binding else 0}


@dataclass(frozen=True)
class VerifiedSourceCacheIndex:
    blockchain_chain_id: int
    chain_key: str
    cache_root: Path = field(repr=False)
    bindings: tuple[VerifiedSourceCacheBinding, ...] = field(repr=False)
    invalid_count: int
    conflict_count: int
    _by_address: Mapping[str, tuple[VerifiedSourceCacheBinding, ...]] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.blockchain_chain_id, int) or self.blockchain_chain_id <= 0 or not isinstance(self.chain_key, str) or not self.chain_key:
            raise ValueError("SOURCE_CACHE_INDEX_INVALID")
        object.__setattr__(self, "cache_root", self.cache_root.resolve())
        object.__setattr__(self, "bindings", tuple(sorted(self.bindings, key=lambda item: (item.source_sha256, item.artifact_relative_path, item.runtime_address, item.effective_address))))
        object.__setattr__(self, "_by_address", MappingProxyType({key: tuple(value) for key, value in sorted(self._by_address.items())}))

    def summary(self) -> dict[str, Any]:
        runtime = len({item.runtime_address for item in self.bindings})
        effective = len({item.effective_address for item in self.bindings})
        proxies = sum(item.proxy_resolution_source == "VERIFIED_PROXY_IMPLEMENTATION" for item in self.bindings)
        return {"schema_version": "verified_source_cache_index_summary_v1", "blockchain_chain_id": self.blockchain_chain_id, "chain_key": self.chain_key, "verified_binding_count": len(self.bindings), "runtime_address_binding_count": runtime, "effective_address_binding_count": effective, "proxy_binding_count": proxies, "conflict_count": self.conflict_count, "invalid_count": self.invalid_count}


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve()); return True
    except ValueError:
        return False


def _manifest_entry(root: Path, manifest_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not _inside(root, manifest_path) or manifest_path.name != "manifest.json": return None, "PATH_ESCAPE"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError): return None, "MANIFEST_INVALID"
    if not isinstance(value, Mapping) or value.get("status") != "verified" or value.get("verified") is not True: return None, "NOT_VERIFIED"
    try:
        requested = _address(value.get("requested_address")); resolved = _address(value.get("resolved_address") or requested)
    except ValueError: return None, "ADDRESS_INVALID"
    files = value.get("files")
    if not isinstance(files, list) or not files: return None, "FILES_MISSING"
    entries = []
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str) or _SHA.fullmatch(item["sha256"]) is None: return None, "FILE_METADATA_INVALID"
        if Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts: return None, "FILE_PATH_INVALID"
        file_path = manifest_path.parent / item["path"]
        if not _inside(manifest_path.parent, file_path) or not file_path.is_file(): return None, "FILE_PATH_INVALID"
        try:
            if _sha256(file_path) != item["sha256"]: return None, "FILE_HASH_MISMATCH"
        except OSError: return None, "FILE_UNREADABLE"
        if file_path.suffix.lower() == ".sol": entries.append(item["sha256"])
    if not entries: return None, "SOLIDITY_FILES_MISSING"
    digest = hashlib.sha256("\n".join(sorted(entries)).encode("ascii")).hexdigest()
    return {"requested": requested, "resolved": resolved, "artifact": manifest_path.parent.relative_to(root).as_posix(), "source_sha": digest, "solidity_file_count": len(entries), "cache_version": str(value.get("cached_at") or value.get("schema_version") or "source_manifest_v1"), "proxy": value.get("proxy") if isinstance(value.get("proxy"), Mapping) else {}, "manifest_sha": _sha256(manifest_path)}, None


def build_verified_source_cache_index(cache_root: str | Path, *, blockchain_chain_id: int, chain_key: str) -> VerifiedSourceCacheIndex:
    root = Path(cache_root).resolve(); chain_dir = root / chain_key.lower()
    entries: dict[str, dict[str, Any]] = {}; invalid = 0
    if chain_dir.exists() and _inside(root, chain_dir):
        for manifest_path in sorted(chain_dir.rglob("manifest.json")):
            entry, error = _manifest_entry(root, manifest_path)
            if entry is None:
                invalid += 1; continue
            entries[entry["requested"]] = entry
    bindings: list[VerifiedSourceCacheBinding] = []
    for entry in entries.values():
        bindings.append(VerifiedSourceCacheBinding(blockchain_chain_id, entry["requested"], entry["resolved"], entry["requested"], entry["requested"], entry["artifact"], entry["source_sha"], "verified", "DIRECT_VERIFIED_CACHE", entry["solidity_file_count"], entry["cache_version"]))
    for parent in entries.values():
        impl = parent["proxy"].get("implementation_address")
        expected = parent["proxy"].get("implementation_manifest_sha256")
        try: impl = _address(impl)
        except ValueError: continue
        child = entries.get(impl)
        if child is None or expected != child["manifest_sha"]: continue
        bindings.append(VerifiedSourceCacheBinding(blockchain_chain_id, parent["requested"], impl, parent["requested"], child["requested"], child["artifact"], child["source_sha"], "verified", "VERIFIED_PROXY_IMPLEMENTATION", child["solidity_file_count"], child["cache_version"]))
    by_address: dict[str, list[VerifiedSourceCacheBinding]] = {}
    for binding in bindings:
        for address in (binding.runtime_address, binding.effective_address, binding.requested_address): by_address.setdefault(address, []).append(binding)
    conflicts = sum(len({item.source_sha256 for item in values}) > 1 for values in by_address.values())
    return VerifiedSourceCacheIndex(blockchain_chain_id, chain_key.lower(), root, tuple(bindings), invalid, conflicts, by_address)


def lookup_verified_source_cache(index: VerifiedSourceCacheIndex, *, runtime_address: str | None = None, effective_address: str | None = None, proxy_implementation_addresses: tuple[str, ...] = (), requested_address: str | None = None) -> VerifiedSourceCacheLookup:
    if not isinstance(index, VerifiedSourceCacheIndex): return VerifiedSourceCacheLookup("VERIFIED_SOURCE_CACHE_INVALID")
    queries = (("effective_address", effective_address), ("runtime_address", runtime_address), *(("proxy_implementation", item) for item in proxy_implementation_addresses), ("requested_address", requested_address))
    for kind, value in queries:
        if value is None: continue
        try: address = _address(value)
        except ValueError: return VerifiedSourceCacheLookup("VERIFIED_SOURCE_CACHE_INVALID")
        candidates = index._by_address.get(address, ())
        if not candidates: continue
        unique = {(item.source_sha256, item.artifact_relative_path): item for item in candidates}
        if len({item.source_sha256 for item in unique.values()}) > 1: return VerifiedSourceCacheLookup("VERIFIED_SOURCE_CACHE_CONFLICT", None, kind)
        binding = sorted(unique.values(), key=lambda item: (item.artifact_relative_path, item.source_sha256))[0]
        return VerifiedSourceCacheLookup("VERIFIED_SOURCE_CACHE_HIT", binding, kind)
    return VerifiedSourceCacheLookup("VERIFIED_SOURCE_CACHE_MISS")


def serialize_verified_source_cache_summary(index: VerifiedSourceCacheIndex) -> str:
    if not isinstance(index, VerifiedSourceCacheIndex): raise ValueError("SOURCE_CACHE_INDEX_INVALID")
    return json.dumps(index.summary(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index, lookup_verified_source_cache, serialize_verified_source_cache_summary


def _address(number: int) -> str:
    return "0x" + f"{number:040x}"


def _write(root: Path, address: str, *, source: str = "contract C {}", verified: bool = True, proxy: dict | None = None, path: str = "sources/C.sol") -> str:
    directory = root / "eth" / address
    file = directory / path; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(source, encoding="utf-8")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest = {"schema_version": "source_manifest_v1", "cached_at": "v1", "requested_address": address, "resolved_address": address, "status": "verified" if verified else "unverified", "verified": verified, "files": [{"path": path, "sha256": digest}], "proxy": proxy or {}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()


class VerifiedSourceCacheIndexTests(unittest.TestCase):
    def test_runtime_and_effective_hits_are_chain_isolated_and_normalized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); runtime, effective = _address(1), _address(2)
            _write(root, effective)
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            self.assertEqual(lookup_verified_source_cache(index, effective_address=effective.upper()).status, "VERIFIED_SOURCE_CACHE_HIT")
            self.assertEqual(lookup_verified_source_cache(index, runtime_address=runtime).status, "VERIFIED_SOURCE_CACHE_MISS")
            self.assertEqual(index.blockchain_chain_id, 1)

    def test_verified_proxy_implementation_is_resolved_to_child_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); runtime, implementation = _address(3), _address(4)
            child_manifest_sha = _write(root, implementation, source="contract Impl {}")
            _write(root, runtime, source="contract Proxy {}", proxy={"implementation_address": implementation, "implementation_manifest_sha256": child_manifest_sha})
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            hit = lookup_verified_source_cache(index, runtime_address=runtime)
            self.assertEqual(hit.status, "VERIFIED_SOURCE_CACHE_CONFLICT")
            hit = lookup_verified_source_cache(index, effective_address=implementation)
            self.assertEqual(hit.status, "VERIFIED_SOURCE_CACHE_HIT")
            self.assertEqual(hit.binding.artifact_address, implementation)

    def test_duplicate_sha_deduplicates_but_distinct_aliases_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); parent, child = _address(5), _address(6)
            child_manifest_sha = _write(root, child, source="contract Same {}")
            _write(root, parent, source="contract Same {}", proxy={"implementation_address": child, "implementation_manifest_sha256": child_manifest_sha})
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            self.assertEqual(lookup_verified_source_cache(index, runtime_address=parent).status, "VERIFIED_SOURCE_CACHE_HIT")
            self.assertGreaterEqual(index.summary()["proxy_binding_count"], 1)

    def test_invalid_unverified_missing_and_escape_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); unverified = _address(7); _write(root, unverified, verified=False)
            escape = _address(8); _write(root, escape, path="sources/../escape.sol")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            self.assertEqual(lookup_verified_source_cache(index, runtime_address="bad").status, "VERIFIED_SOURCE_CACHE_INVALID")
            self.assertEqual(lookup_verified_source_cache(index, runtime_address=unverified).status, "VERIFIED_SOURCE_CACHE_MISS")
            self.assertGreaterEqual(index.invalid_count, 2)

    def test_summary_never_contains_addresses_or_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); address = _address(9); _write(root, address)
            rendered = serialize_verified_source_cache_summary(build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth"))
            self.assertNotIn(address, rendered); self.assertNotIn(str(root), rendered)


if __name__ == "__main__":
    unittest.main()

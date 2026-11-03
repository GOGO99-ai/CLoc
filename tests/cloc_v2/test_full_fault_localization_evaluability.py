from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import audit_full_fault_localization_evaluability as audit


def _tx(number: int) -> str:
    return "0x" + f"{number:064x}"


def _address(number: int) -> str:
    return "0x" + f"{number:040x}"


def _cache(root: Path, address: str, source: str, *, proxy: dict | None = None, name: str = "sources/C.sol") -> str:
    directory = root / "eth" / address
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_manifest_v1",
        "cached_at": "v1",
        "requested_address": address,
        "resolved_address": address,
        "status": "verified",
        "verified": True,
        "files": [{"path": name, "sha256": digest}],
        "proxy": proxy or {},
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"samples": [{"sample_alias": alias, "tx_hash": transaction} for alias, transaction in rows]}), encoding="utf-8")


def _dataset(path: Path, transaction: str, location: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"platform": "Ethereum", "fault": {"transaction_hash": transaction, "location": location, "cause": "MUST_NOT_BE_USED"}}), encoding="utf-8")


class FullFaultLocalizationEvaluabilityTests(unittest.TestCase):
    def test_full_file_function_line_missing_invalid_and_line_without_function(self):
        source = "contract C {\n function f() external {\n  uint x = 1;\n }\n}\n"
        address = _address(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = root / "dataset"; cache = root / "cache"
            _cache(cache, address, source)
            good_offset = source.encode().index(b"uint x")
            outside_offset = source.encode().index(b"contract")
            _dataset(dataset / "a.json", _tx(1), f"{address}#sources/C.sol#{good_offset}:6")
            _dataset(dataset / "b.json", _tx(2), f"{_address(2)}#sources/C.sol#0:1")
            _dataset(dataset / "c.json", _tx(3), "not-a-location")
            _dataset(dataset / "d.json", _tx(4), f"{address}#sources/C.sol#{outside_offset}:8")
            dev = root / "dev.json"; blind = root / "blind.json"; frozen = root / "frozen.json"
            _manifest(dev, [("dev_001", _tx(1))]); _manifest(blind, [("blind_001", _tx(4))])
            frozen.write_text(json.dumps({"predictions": []}), encoding="utf-8")
            result = audit.build_full_fault_evaluability_audit(dataset_root=dataset, cache_root=cache, dev15_manifest=dev, blind10_manifest=blind, blind10_frozen=frozen, expected_transaction_count=4)
            self.assertEqual(result["overall"]["file_evaluable_count"], 2)
            self.assertEqual(result["overall"]["function_evaluable_count"], 1)
            self.assertEqual(result["overall"]["line_range_evaluable_count"], 2)
            self.assertEqual(result["overall"]["cache_missing_count"], 1)
            self.assertEqual(result["overall"]["location_invalid_count"], 1)
            detail = result["latest_blind10"]["samples"][0]
            self.assertEqual(detail["missing_layer"], "FUNCTION_INDEX")
            self.assertIn("LOCATION_OUTSIDE_ALL_PARSED_FUNCTIONS", detail["function_resolution_gaps"])

    def test_safe_conflict_resolution_uses_truth_file_without_prediction(self):
        proxy = _address(10); implementation = _address(11)
        proxy_source = "contract Proxy { fallback() external {} }\n"
        impl_source = "contract Impl { function target() external { uint x = 1; } }\n"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cache = root / "cache"
            child_sha = _cache(cache, implementation, impl_source, name="sources/Impl.sol")
            _cache(cache, proxy, proxy_source, proxy={"implementation_address": implementation, "implementation_manifest_sha256": child_sha}, name="sources/Proxy.sol")
            index = audit.build_verified_source_cache_index(cache, blockchain_chain_id=1, chain_key="eth")
            offset = impl_source.encode().index(b"uint x")
            label = audit.parse_localization_ground_truth(f"{proxy}#sources/Impl.sol#{offset}:6")
            self.assertEqual(audit.lookup_verified_source_cache(index, runtime_address=proxy).status, "VERIFIED_SOURCE_CACHE_CONFLICT")
            resolved, detail = audit._resolve_conflict_without_guessing(label, index)
            self.assertIsNotNone(resolved)
            self.assertTrue(resolved.function_available)
            self.assertEqual(detail["distinct_source_sha_count"], 2)
            self.assertTrue(detail["direct_proxy_mixed"])

    def test_absolute_path_is_only_normalized_to_safe_suffix(self):
        address = _address(20)
        label = audit._safe_absolute_path_label(f"{address}#C:\\unsafe\\contracts\\A.sol#1:2")
        self.assertIsNotNone(label)
        self.assertEqual(label.source_file, "contracts/A.sol")
        self.assertIsNone(audit._safe_absolute_path_label(f"{address}#C:\\unsafe\\A.sol#1:2"))

    def test_groups_are_disjoint_and_remainder_is_stable(self):
        source = "contract C { function f() external {} }"
        address = _address(30)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = root / "dataset"; cache = root / "cache"; _cache(cache, address, source)
            for number in range(1, 4): _dataset(dataset / f"{number}.json", _tx(number), f"{address}#sources/C.sol#13:8")
            dev = root / "dev.json"; blind = root / "blind.json"; frozen = root / "frozen.json"
            _manifest(dev, [("dev", _tx(1))]); _manifest(blind, [("blind", _tx(2))]); frozen.write_text('{"predictions":[]}', encoding="utf-8")
            result = audit.build_full_fault_evaluability_audit(dataset_root=dataset, cache_root=cache, dev15_manifest=dev, blind10_manifest=blind, blind10_frozen=frozen, expected_transaction_count=3)
            self.assertEqual(result["groups"]["HISTORICAL_DEV15"]["transaction_count"], 1)
            self.assertEqual(result["groups"]["LATEST_BLIND10"]["transaction_count"], 1)
            self.assertEqual(result["groups"]["UNRUN_REMAINDER"]["transaction_count"], 1)
            self.assertEqual(result["group_overlap_count"], 0)

    def test_overlapping_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = root / "dataset"; dataset.mkdir(); dev = root / "dev"; blind = root / "blind"; frozen = root / "frozen"
            _dataset(dataset / "a.json", _tx(1), "bad")
            _manifest(dev, [("a", _tx(1))]); _manifest(blind, [("b", _tx(1))]); frozen.write_text('{"predictions":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "AUDIT_GROUP_OVERLAP"):
                audit.build_full_fault_evaluability_audit(dataset_root=dataset, cache_root=root / "cache", dev15_manifest=dev, blind10_manifest=blind, blind10_frozen=frozen, expected_transaction_count=1)

    def test_blind_conflict_report_is_sanitized_and_does_not_guess(self):
        address = _address(40); implementation = _address(41)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cache = root / "cache"; child = _cache(cache, implementation, "contract I {}")
            _cache(cache, address, "contract P {}", proxy={"implementation_address": implementation, "implementation_manifest_sha256": child})
            index = audit.build_verified_source_cache_index(cache, blockchain_chain_id=1, chain_key="eth")
            frozen = root / "frozen.json"
            frozen.write_text(json.dumps({"predictions": [{"sample_alias": "blind_001", "status": "SOURCE_FAILURE", "reason_codes": ["VERIFIED_SOURCE_CACHE_CONFLICT"]}]}), encoding="utf-8")
            result = audit._blind_conflicts(frozen, index)
            self.assertEqual(result["affected_sample_count"], 1)
            self.assertEqual(result["samples"][0]["distinct_source_sha_count"], 2)
            self.assertFalse(result["samples"][0]["safe_offline_resolution"])
            self.assertNotIn(address, json.dumps(result))

    def test_public_audit_rejects_identity_path_and_source(self):
        for value in ({"address": _address(1)}, {"x": _tx(1)}, {"location": "x"}, {"source_text": "contract C {}"}, {"artifact_relative_path": "eth/x"}):
            with self.assertRaises(ValueError): audit._validate_public(value)
        audit._validate_public({"sample_alias": "fault_audit_eth_001", "count": 1})

    def test_atomic_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audit.json"; value = {"schema_version": "x", "sample_alias": "safe_001"}
            audit.write_audit(path, value)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), value)

    def test_module_has_no_inference_or_external_client_import(self):
        source = Path(audit.__file__).read_text(encoding="utf-8")
        self.assertNotIn("phase2_localization_bridge", source)
        self.assertNotIn("phase2_localization_deepseek", source)
        self.assertNotIn("JsonRpcClient", source)
        self.assertNotIn("DeepSeek", source)
        self.assertNotIn('.get("cause")', source)


if __name__ == "__main__":
    unittest.main()

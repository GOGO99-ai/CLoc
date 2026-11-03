from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import audit_missing_verified_sources as audit


def _tx(number: int) -> str:
    return "0x" + f"{number:064x}"


def _address(number: int) -> str:
    return "0x" + f"{number:040x}"


def _dataset(path: Path, transaction: str, address: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"platform": "Ethereum", "fault": {"transaction_hash": transaction, "location": f"{address}#sources/C.sol#31:6", "cause": "MUST_NOT_BE_USED"}}), encoding="utf-8")


def _response(source: str, *, proxy: str = "0", implementation: str = "") -> dict:
    return {"status": "1", "result": [{"SourceCode": source, "ContractName": "C", "ABI": "[]", "Proxy": proxy, "Implementation": implementation}]}


class Response:
    status_code = 200

    def __init__(self, value: dict):
        self.value = value

    def json(self) -> dict:
        return self.value


class Session:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return Response(value)


class MissingVerifiedSourceAuditTests(unittest.TestCase):
    def test_lock_is_deterministic_deduplicated_and_does_not_read_cause(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, cache = root / "dataset", root / "cache"
            _dataset(dataset / "b.json", _tx(2), _address(1))
            _dataset(dataset / "a.json", _tx(1), _address(1))
            first = audit.build_locked_manifest(dataset, cache, expected_faults=2, expected_missing=2)
            second = audit.build_locked_manifest(dataset, cache, expected_faults=2, expected_missing=2)
            self.assertEqual(first, second)
            self.assertEqual(first["transaction_count"], 2)
            self.assertEqual(first["deduplicated_contract_address_count"], 1)
            self.assertFalse(first["cause_read"])
            self.assertNotIn("MUST_NOT_BE_USED", json.dumps(first))
            audit.validate_locked_manifest(first, expected_missing=2)

    def test_direct_and_proxy_are_classified_and_requests_counted_once(self):
        source = "contract C { function f() external { uint x = 1; } }"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, cache, output = root / "dataset", root / "cache", root / "out"
            _dataset(dataset / "a.json", _tx(1), _address(1))
            _dataset(dataset / "b.json", _tx(2), _address(2))
            session = Session([_response(source), _response(source, proxy="1", implementation=_address(3)), _response(source)])
            result = audit.run_backfill(output, dataset_root=dataset, cache_root=cache, config={"base_url": "https://invalid", "chainid": 1, "key": "secret"}, session=session, expected_faults=2, expected_missing=2)
            self.assertEqual(result["audit_status"], "COMPLETED")
            self.assertEqual(session.calls, 3)
            self.assertEqual(result["physical_source_request_count"], 3)
            self.assertEqual(result["classification_counts"][audit.DIRECT], 1)
            self.assertEqual(result["classification_counts"][audit.PROXY], 1)
            self.assertEqual(result["retry_count"], 0)
            self.assertNotIn("secret", (output / "verified_source_backfill_audit.json").read_text())

    def test_unverified_continues_but_network_error_stops_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, cache, output = root / "dataset", root / "cache", root / "out"
            for number in range(1, 4):
                _dataset(dataset / f"{number}.json", _tx(number), _address(number))
            session = Session([_response(""), TimeoutError("secret network detail"), _response("contract Never {}")])
            result = audit.run_backfill(output, dataset_root=dataset, cache_root=cache, config={"base_url": "https://invalid", "chainid": 1, "key": "secret"}, session=session, expected_faults=3, expected_missing=3)
            self.assertEqual(result["audit_status"], "STOPPED_NETWORK_ERROR")
            self.assertEqual(session.calls, 2)
            self.assertEqual(result["classification_counts"][audit.UNVERIFIED], 1)
            self.assertEqual(result["classification_counts"][audit.FAILED], 1)
            checkpoint = json.loads((output / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["physical_source_request_count"], 2)
            self.assertEqual(len(checkpoint["completed_address_aliases"]), 2)
            self.assertNotIn("secret network detail", json.dumps(result))

    def test_budget_is_recorded_before_request(self):
        calls = []
        session = audit.CountingSession(Session([_response("contract C{}")]), lambda: calls.append(1) or 91, limit=90)
        with self.assertRaisesRegex(audit.SourceRequestBudgetExceeded, "SOURCE_REQUEST_BUDGET_EXCEEDED"):
            session.get("https://invalid", {}, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(session.delegate.calls, 0)

    def test_missing_key_stops_before_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, output = root / "dataset", root / "out"
            _dataset(dataset / "a.json", _tx(1), _address(1))
            session = Session([])
            with self.assertRaisesRegex(RuntimeError, "SOURCE_API_KEY_MISSING"):
                audit.run_backfill(output, dataset_root=dataset, cache_root=root / "cache", config={"base_url": "https://invalid", "chainid": 1, "key": ""}, session=session, expected_faults=1, expected_missing=1)
            self.assertEqual(session.calls, 0)

    def test_manifest_sha_detects_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            _dataset(dataset / "a.json", _tx(1), _address(1))
            manifest = audit.build_locked_manifest(dataset, root / "cache", expected_faults=1, expected_missing=1)
            manifest["transaction_count"] = 2
            with self.assertRaisesRegex(ValueError, "TRANSACTION_COUNT_INVALID|SHA_INVALID"):
                audit.validate_locked_manifest(manifest, expected_missing=1)

    def test_module_does_not_import_inference_rpc_or_llm(self):
        text = Path(audit.__file__).read_text(encoding="utf-8")
        for forbidden in ("phase2_localization_bridge", "phase2_localization_deepseek", "JsonRpcClient", "DeepSeekClient", '.get("cause")'):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()

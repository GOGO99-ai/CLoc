from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error as urllib_error

from scripts.validate_real_normal_baseline import (
    JsonRpcClient,
    NormalSample,
    RpcValidationError,
    build_baseline_sensitivity,
    build_leave_one_out_queries,
    main,
    process_calltracer_sample,
    run_live_validation,
    run_validation,
    select_normal_samples,
)
from src.cloc_v2.normal_distance import CallStep, ChainFeatureRecord, MainAnchorKey


def addr(character: str) -> str:
    return "0x" + character * 40


ANCHOR = MainAnchorKey(1, addr("a"), "0x11111111")
STEP = CallStep(addr("a"), "0x11111111")


def record(record_id: str, *, uncertain: bool = False) -> ChainFeatureRecord:
    return ChainFeatureRecord(
        record_id=record_id,
        main_anchor=ANCHOR,
        call_steps=(STEP,),
        asset_behaviors=frozenset(),
        uncertainties=("TEST_UNCERTAINTY",) if uncertain else (),
    )


def make_trace_files(directory: Path, count: int = 10) -> list[str]:
    hashes = ["0x" + f"{index:064x}" for index in range(count, 0, -1)]
    for tx_hash in hashes:
        (directory / f"eth_{tx_hash}.txt").write_text("not-read", encoding="utf-8")
    return sorted(hashes)


def successful_frame() -> dict:
    recipient = addr("c")
    transfer_input = (
        "0xa9059cbb"
        + "0" * 24
        + recipient[2:]
        + f"{123456789:064x}"
    )
    return {
        "type": "CALL",
        "from": addr("f"),
        "to": addr("a"),
        "input": "0x11111111deadbeef",
        "output": "0x736563726574",
        "calls": [
            {
                "type": "CALL",
                "from": addr("a"),
                "to": addr("b"),
                "input": transfer_input,
                "output": "0x7261776368696c646f7574707574",
                "error": "RAW_ERROR_MUST_NOT_APPEAR",
                "revertReason": "RAW_REVERT_REASON_MUST_NOT_APPEAR",
                "calls": [],
            }
        ],
        "privateMarker": "RAW_TRACE_MUST_NOT_APPEAR",
    }


class FakeRpcClient:
    def __init__(self) -> None:
        self.request_count = 0

    def call(self, method: str, params: list[object]):
        self.request_count += 1
        if method == "eth_chainId":
            return "0x1"
        return successful_frame()


class FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.body


class RealNormalBaselineValidationTests(unittest.TestCase):
    def test_default_mode_does_not_invoke_live_runner(self) -> None:
        invoked = False

        def live_runner(**kwargs):
            nonlocal invoked
            invoked = True
            return {}

        result = run_validation(live=False, live_runner=live_runner)
        self.assertFalse(invoked)
        self.assertEqual(result, {"status": "LIVE_MODE_REQUIRED", "rpc_request_count": 0})

    def test_cli_requires_explicit_live_flag(self) -> None:
        with patch("scripts.validate_real_normal_baseline.run_validation") as runner:
            self.assertEqual(main([]), 0)
        runner.assert_not_called()

    def test_explicit_live_flag_invokes_live_runner(self) -> None:
        result = run_validation(live=True, live_runner=lambda **kwargs: {"status": "DONE"})
        self.assertEqual(result["status"], "DONE")

    def test_rpc_client_hard_stops_after_nine_requests(self) -> None:
        transport = lambda url, payload, timeout: {"jsonrpc": "2.0", "id": payload["id"], "result": "0x1"}
        client = JsonRpcClient("memory://rpc", transport=transport)
        for _ in range(9):
            client.call("eth_chainId", [])
        with self.assertRaisesRegex(RpcValidationError, "RPC_REQUEST_LIMIT_EXCEEDED"):
            client.call("eth_chainId", [])
        self.assertEqual(client.request_count, 9)

    def test_default_client_uses_standard_library_without_requests_module(self) -> None:
        captured: dict[str, object] = {}

        def urlopen(request, timeout):
            captured["method"] = request.get_method()
            captured["content_type"] = request.get_header("Content-type")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeHttpResponse(b'{"jsonrpc":"2.0","id":1,"result":"0x1"}')

        with patch.dict(sys.modules, {"requests": None}), patch(
            "scripts.validate_real_normal_baseline.urllib_request.urlopen", side_effect=urlopen
        ):
            client = JsonRpcClient("https://redacted.invalid")
            self.assertEqual(client.call("eth_chainId", []), "0x1")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["content_type"], "application/json")
        self.assertEqual(captured["body"], {"jsonrpc": "2.0", "id": 1, "method": "eth_chainId", "params": []})
        self.assertEqual(captured["timeout"], 55.0)

    def test_standard_library_http_errors_are_safely_classified(self) -> None:
        for code, expected in ((401, "RPC_UNAUTHORIZED"), (403, "RPC_UNAUTHORIZED"), (429, "RPC_QUOTA_EXCEEDED"), (503, "RPC_HTTP_ERROR")):
            with self.subTest(status_code=code), patch(
                "scripts.validate_real_normal_baseline.urllib_request.urlopen",
                side_effect=urllib_error.HTTPError("https://redacted.invalid/secret", code, "PRIVATE_RESPONSE_BODY", {}, None),
            ):
                with self.assertRaisesRegex(RpcValidationError, expected) as raised:
                    JsonRpcClient("https://redacted.invalid/secret").call("eth_chainId", [])
                self.assertNotIn("redacted.invalid", str(raised.exception))
                self.assertNotIn("PRIVATE_RESPONSE_BODY", str(raised.exception))

    def test_standard_library_network_and_invalid_json_are_sanitized(self) -> None:
        cases = (
            (socket.timeout("secret endpoint"), "RPC_TIMEOUT"),
            (urllib_error.URLError("secret endpoint"), "RPC_NETWORK_ERROR"),
            (None, "RPC_INVALID_RESPONSE"),
        )
        for failure, expected in cases:
            with self.subTest(expected=expected):
                side_effect = failure
                if failure is None:
                    side_effect = lambda request, timeout: FakeHttpResponse(b"PRIVATE_NON_JSON_RESPONSE")
                with patch("scripts.validate_real_normal_baseline.urllib_request.urlopen", side_effect=side_effect):
                    with self.assertRaisesRegex(RpcValidationError, expected) as raised:
                        JsonRpcClient("https://redacted.invalid/secret").call("eth_chainId", [])
                self.assertNotIn("redacted.invalid", str(raised.exception))
                self.assertNotIn("PRIVATE_NON_JSON_RESPONSE", str(raised.exception))

    def test_sample_selection_is_hash_sorted_and_does_not_read_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            expected = make_trace_files(directory)
            with patch.object(Path, "read_text", side_effect=AssertionError("trace content read")):
                selected = select_normal_samples(directory)
        self.assertEqual([item.tx_hash for item in selected], expected[:8])
        self.assertEqual([item.sample_id for item in selected], [f"normal_sample_{index:03d}" for index in range(1, 9)])
        self.assertTrue(all(item.is_normal for item in selected))

    def test_existing_manifest_prevents_sample_drift_before_rpc(self) -> None:
        with tempfile.TemporaryDirectory() as trace_temp, tempfile.TemporaryDirectory() as output_temp:
            trace_dir = Path(trace_temp)
            output_dir = Path(output_temp)
            make_trace_files(trace_dir)
            (output_dir / "normal_input_manifest.json").write_text(
                json.dumps({"samples": [{"sample_id": "different", "tx_hash": "0x" + "f" * 64}]}),
                encoding="utf-8",
            )
            client = FakeRpcClient()
            with self.assertRaisesRegex(ValueError, "do not match"):
                run_live_validation(trace_dir=trace_dir, output_dir=output_dir, rpc_client=client)
        self.assertEqual(client.request_count, 0)

    def test_stage_statistics_include_transfer_asset_and_chain_counts(self) -> None:
        sample = NormalSample("normal_sample_001", 1, "0x" + "1" * 64)
        summary, records = process_calltracer_sample(sample, successful_frame())
        self.assertEqual(summary["call_node_count"], 2)
        self.assertEqual(summary["recognized_transfer_node_count"], 1)
        self.assertEqual(summary["asset_fact_count"], 1)
        self.assertEqual(summary["complete_asset_fact_count"], 1)
        self.assertEqual(summary["incomplete_asset_fact_count"], 0)
        self.assertEqual(summary["result_candidate_count"], 1)
        self.assertEqual(summary["strict_chain_count"], 1)
        self.assertEqual(summary["strict_ready_count"], 1)
        self.assertEqual(summary["strict_uncertain_count"], 0)
        self.assertEqual(summary["clean_record_count"], 1)
        self.assertEqual(summary["uncertain_record_count"], 0)
        self.assertEqual(summary["rejected_record_count"], 0)
        self.assertTrue(records)
        self.assertNotIn("feature_extraction", summary)

    def test_only_clean_records_enter_formal_sensitivity(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only clean records"):
            build_baseline_sensitivity((record("clean"), record("uncertain", uncertain=True)))
        result = build_baseline_sensitivity((record("clean-a"), record("clean-b")))
        self.assertEqual(result["clean_record_count"], 2)

    def test_minimum_group_sizes_two_three_and_five_all_execute(self) -> None:
        result = build_baseline_sensitivity((record("a"), record("b"), record("c"), record("d"), record("e")))
        self.assertEqual([item["min_group_size"] for item in result["settings"]], [2, 3, 5])

    def test_leave_one_out_never_matches_query_record_itself(self) -> None:
        records = (record("a"), record("b"), record("c"))
        result = build_leave_one_out_queries(records)
        for setting in result["settings"]:
            for query in setting["queries"]:
                self.assertNotIn(query["query_record_id"], [item["normal_record_id"] for item in query["top_matches"]])

    def test_live_validation_writes_only_sanitized_allowed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as trace_temp, tempfile.TemporaryDirectory() as output_temp:
            trace_dir = Path(trace_temp)
            output_dir = Path(output_temp)
            make_trace_files(trace_dir)
            client = FakeRpcClient()
            report = run_live_validation(trace_dir=trace_dir, output_dir=output_dir, rpc_client=client)
            names = sorted(path.name for path in output_dir.iterdir())
            payload = "".join(path.read_text(encoding="utf-8") for path in sorted(output_dir.iterdir()))
        self.assertEqual(
            names,
            sorted(
                [
                    "normal_input_manifest.json",
                    "normal_feature_audit.json",
                    "baseline_sensitivity.json",
                    "leave_one_out_queries.json",
                    "validation_report.json",
                ]
            ),
        )
        self.assertEqual(client.request_count, 9)
        self.assertEqual(report["successful_sample_count"], 8)
        self.assertTrue(all("call_node_count" in item for item in report["sample_results"]))
        for forbidden_value in (
            "0x11111111deadbeef",
            "0x736563726574",
            "0x22222222cafebabe",
            "0x7261776368696c646f7574707574",
            "RAW_ERROR_MUST_NOT_APPEAR",
            "RAW_REVERT_REASON_MUST_NOT_APPEAR",
            "RAW_TRACE_MUST_NOT_APPEAR",
        ):
            self.assertNotIn(forbidden_value, payload)
        lowered = payload.lower()
        for forbidden_field in ('"rpc_url"', '"api_key"', '"fault"', '"cause"', '"location"', '"ground_truth"', '"paired_transaction"'):
            self.assertNotIn(forbidden_field, lowered)


if __name__ == "__main__":
    unittest.main()

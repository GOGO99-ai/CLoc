from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib import error as urllib_error

import scripts.run_phase1_deepseek_smoke as smoke
from scripts.dry_run_phase1_normality import build_synthetic_phase1_preparation
from src.cloc_v2.phase1_normality_deepseek import (
    Phase1DeepSeekConfig,
    Phase1DeepSeekTransportError,
    build_phase1_deepseek_request,
    execute_phase1_deepseek_review,
)
from src.cloc_v2.phase1_normality_prompt import RESPONSE_SCHEMA_VERSION


API_KEY = "mock-deepseek-key-never-persist"
SECRET_REASONING = "private-reasoning-never-persist"
SECRET_BODY = "private-provider-body-never-persist"


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.read_sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.body


class FakeClientTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = []

    def send(self, payload, api_key, timeout_seconds):
        self.calls.append((payload, api_key, timeout_seconds))
        return self.response


def validated_chain_response(preparation) -> dict:
    scope = preparation.prompt.citation_whitelist.chain_scopes[0]
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "chain_decisions": [
            {
                "chain_alias": scope.chain_alias,
                "decision": "NORMAL",
                "reference_assessment": "CONSISTENT",
                "supporting_reference_aliases": [scope.reference_aliases[0]],
                "evidence_ids": [scope.query_call_evidence_ids[0], scope.reference_call_evidence_ids[0]],
                "reason_codes": ["CONSISTENT_CALL_SEQUENCE", "WITHIN_CALIBRATED_THRESHOLD"],
                "brief_rationale": (
                    f"{scope.chain_alias} matches {scope.reference_aliases[0]} using "
                    f"{scope.query_call_evidence_ids[0]} and {scope.reference_call_evidence_ids[0]}."
                ),
            }
        ],
    }


def provider_response(preparation) -> dict:
    return {
        "model": "deepseek-v4-pro-snapshot",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(validated_chain_response(preparation)),
                    "reasoning_content": SECRET_REASONING,
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def validated_result():
    preparation = build_synthetic_phase1_preparation()
    transport = FakeClientTransport(provider_response(preparation))
    result = execute_phase1_deepseek_review(
        preparation, Phase1DeepSeekConfig(), transport, API_KEY
    )
    return result


class DeepSeekSmokeTransportTests(unittest.TestCase):
    def invoke_transport(self, body: bytes):
        response = FakeHTTPResponse(body)
        opener = MagicMock()
        opener.open.return_value = response
        transport = smoke.DeepSeekHTTPSOneShotTransport()
        with patch.object(smoke.urllib_request, "build_opener", return_value=opener), patch.object(
            smoke.ssl, "create_default_context", return_value=MagicMock()
        ):
            result = transport.send({"model": "deepseek-v4-pro"}, API_KEY, 17)
        return result, transport, opener, response

    def test_fixed_https_url(self) -> None:
        self.assertEqual(smoke.DEEPSEEK_URL, "https://api.deepseek.com/chat/completions")
        with self.assertRaises(ValueError):
            smoke.DeepSeekHTTPSOneShotTransport("http://api.deepseek.com/chat/completions")

    def test_request_method_is_post(self) -> None:
        _, _, opener, _ = self.invoke_transport(b"{}")
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")

    def test_request_body_is_json(self) -> None:
        _, _, opener, _ = self.invoke_transport(b"{}")
        request = opener.open.call_args.args[0]
        self.assertEqual(json.loads(request.data.decode("utf-8")), {"model": "deepseek-v4-pro"})

    def test_explicit_headers_are_content_type_and_authorization_only(self) -> None:
        _, _, opener, _ = self.invoke_transport(b"{}")
        request = opener.open.call_args.args[0]
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(set(headers), {"content-type", "authorization"})
        self.assertEqual(headers["content-type"], "application/json")

    def test_authorization_is_created_only_inside_transport(self) -> None:
        transport = smoke.DeepSeekHTTPSOneShotTransport()
        self.assertNotIn(API_KEY, repr(transport.__dict__))
        _, _, opener, _ = self.invoke_transport(b"{}")
        headers = dict(opener.open.call_args.args[0].header_items())
        self.assertIn(API_KEY, next(value for key, value in headers.items() if key.lower() == "authorization"))

    def test_tls_uses_default_context(self) -> None:
        response = FakeHTTPResponse(b"{}")
        opener = MagicMock()
        opener.open.return_value = response
        with patch.object(smoke.urllib_request, "build_opener", return_value=opener), patch.object(
            smoke.ssl, "create_default_context", return_value=MagicMock()
        ) as context:
            smoke.DeepSeekHTTPSOneShotTransport().send({}, API_KEY, 10)
        context.assert_called_once_with()

    def test_redirect_handler_rejects_redirect(self) -> None:
        with self.assertRaises(Phase1DeepSeekTransportError):
            smoke._NoRedirectHandler().redirect_request(None, None, 302, "secret", {}, "https://other")

    def test_response_read_is_bounded_to_two_mib_plus_one(self) -> None:
        _, _, _, response = self.invoke_transport(b"{}")
        self.assertEqual(response.read_sizes, [smoke.MAX_RESPONSE_BYTES + 1])

    def test_oversized_response_is_rejected(self) -> None:
        body = b"x" * (smoke.MAX_RESPONSE_BYTES + 1)
        with self.assertRaises(Phase1DeepSeekTransportError):
            self.invoke_transport(body)

    def test_transport_instance_allows_only_one_send(self) -> None:
        transport = smoke.DeepSeekHTTPSOneShotTransport()
        response = FakeHTTPResponse(b"{}")
        opener = MagicMock()
        opener.open.return_value = response
        with patch.object(smoke.urllib_request, "build_opener", return_value=opener), patch.object(
            smoke.ssl, "create_default_context", return_value=MagicMock()
        ):
            transport.send({}, API_KEY, 10)
            with self.assertRaises(Phase1DeepSeekTransportError):
                transport.send({}, API_KEY, 10)
        self.assertEqual(opener.open.call_count, 1)

    def test_no_automatic_retry_on_network_error(self) -> None:
        opener = MagicMock()
        opener.open.side_effect = urllib_error.URLError(SECRET_BODY)
        transport = smoke.DeepSeekHTTPSOneShotTransport()
        with patch.object(smoke.urllib_request, "build_opener", return_value=opener), patch.object(
            smoke.ssl, "create_default_context", return_value=MagicMock()
        ):
            with self.assertRaises(Phase1DeepSeekTransportError):
                transport.send({}, API_KEY, 10)
        self.assertEqual(opener.open.call_count, 1)

    def test_http_error_body_is_not_read(self) -> None:
        body = MagicMock()
        error = urllib_error.HTTPError(smoke.DEEPSEEK_URL, 401, SECRET_BODY, {}, body)
        opener = MagicMock()
        opener.open.side_effect = error
        with patch.object(smoke.urllib_request, "build_opener", return_value=opener), patch.object(
            smoke.ssl, "create_default_context", return_value=MagicMock()
        ):
            with self.assertRaises(Phase1DeepSeekTransportError):
                smoke.DeepSeekHTTPSOneShotTransport().send({}, API_KEY, 10)
        body.read.assert_not_called()

    def test_http_error_counts_one_started_api_request(self) -> None:
        error = urllib_error.HTTPError(smoke.DEEPSEEK_URL, 429, SECRET_BODY, {}, None)
        opener = MagicMock()
        opener.open.side_effect = error
        transport = smoke.DeepSeekHTTPSOneShotTransport()
        with patch.object(smoke.urllib_request, "build_opener", return_value=opener), patch.object(
            smoke.ssl, "create_default_context", return_value=MagicMock()
        ):
            with self.assertRaises(Phase1DeepSeekTransportError):
                transport.send({}, API_KEY, 10)
        self.assertEqual(transport.request_attempt_count, 1)
        self.assertEqual(transport.api_request_count, 1)

    def test_invalid_utf8_is_rejected(self) -> None:
        with self.assertRaises(Phase1DeepSeekTransportError):
            self.invoke_transport(b"\xff")

    def test_invalid_json_is_rejected(self) -> None:
        with self.assertRaises(Phase1DeepSeekTransportError):
            self.invoke_transport(b"not-json")

    def test_non_object_json_is_rejected(self) -> None:
        with self.assertRaises(Phase1DeepSeekTransportError):
            self.invoke_transport(b"[]")


class DeepSeekSmokeReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = validated_result()
        self.report = smoke.build_phase1_deepseek_smoke_report(
            self.result, request_attempt_count=1, api_request_count=1, api_key=API_KEY
        )

    def test_valid_provider_response_builds_validated_report(self) -> None:
        self.assertEqual(self.report["response_status"], "VALIDATED")

    def test_normal_is_routed_by_formal_router(self) -> None:
        self.assertEqual(self.report["routing_decision"]["route"], "EARLY_NORMAL")
        self.assertIn(
            "ALL_USABLE_CLEAN_CHAINS_LLM_NORMAL",
            self.report["routing_decision"]["reason_codes"],
        )

    def test_reasoning_content_is_absent(self) -> None:
        text = smoke.serialize_phase1_deepseek_smoke_report(self.report)
        self.assertNotIn("reasoning_content", text)
        self.assertNotIn(SECRET_REASONING, text)

    def test_key_absent_from_client_request_result_and_report(self) -> None:
        preparation = build_synthetic_phase1_preparation()
        request = build_phase1_deepseek_request(preparation, Phase1DeepSeekConfig())
        self.assertNotIn(API_KEY, json.dumps(request.to_dict()))
        self.assertNotIn(API_KEY, json.dumps(self.result.to_dict()))
        self.assertNotIn(API_KEY, smoke.serialize_phase1_deepseek_smoke_report(self.report))

    def test_nonvalidated_result_does_not_fabricate_routing(self) -> None:
        preparation = build_synthetic_phase1_preparation()
        transport = FakeClientTransport({"model": "x", "choices": [], "usage": {}})
        result = execute_phase1_deepseek_review(
            preparation, Phase1DeepSeekConfig(), transport, API_KEY
        )
        report = smoke.build_phase1_deepseek_smoke_report(
            result, request_attempt_count=1, api_request_count=1, api_key=API_KEY
        )
        self.assertNotEqual(report["response_status"], "VALIDATED")
        self.assertIsNone(report["routing_decision"])
        self.assertEqual(report["fallback_action"], "PROCEED_PHASE2")

    def test_report_excludes_prompt_payload_and_headers(self) -> None:
        keys = smoke._all_keys(self.report)
        self.assertTrue(keys.isdisjoint({"system_message", "user_message", "messages", "payload", "headers"}))

    def test_serialized_report_never_names_hidden_reasoning_field(self) -> None:
        text = smoke.serialize_phase1_deepseek_smoke_report(self.report).lower()
        self.assertNotIn("reasoning_content", text)
        self.assertNotIn("authorization", text)
        self.assertNotIn("bearer ", text)

    def test_report_excludes_raw_identities(self) -> None:
        text = smoke.serialize_phase1_deepseek_smoke_report(self.report)
        self.assertIsNone(smoke._RAW_ADDRESS.search(text))
        self.assertIsNone(smoke._RAW_TRANSACTION_HASH.search(text))
        self.assertNotIn("synthetic-normal-record", text)
        self.assertNotIn("synthetic-query-record", text)

    def test_report_fixed_scientific_flags(self) -> None:
        self.assertIs(self.report["scientific_experiment_result"], False)
        self.assertIs(self.report["synthetic_input"], True)
        self.assertEqual(self.report["smoke_kind"], "SYNTHETIC_DEEPSEEK_API_COMPATIBILITY_CHECK")

    def test_rpc_count_is_zero(self) -> None:
        self.assertEqual(self.report["rpc_request_count"], 0)

    def test_request_counts_are_at_most_one(self) -> None:
        self.assertEqual(self.report["request_attempt_count"], 1)
        self.assertEqual(self.report["api_request_count"], 1)
        with self.assertRaises(ValueError):
            smoke.build_phase1_deepseek_smoke_report(
                self.result, request_attempt_count=2, api_request_count=1
            )

    def test_report_root_has_only_fixed_fields(self) -> None:
        self.assertEqual(set(self.report), smoke.REPORT_FIELDS)

    def test_report_serialization_is_stable(self) -> None:
        left = smoke.serialize_phase1_deepseek_smoke_report(self.report)
        right = smoke.serialize_phase1_deepseek_smoke_report(dict(reversed(tuple(self.report.items()))))
        self.assertEqual(left, right)

    def test_report_is_atomically_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            with patch.object(smoke.os, "replace", wraps=os.replace) as replace:
                written = smoke.write_phase1_deepseek_smoke_report(self.report, path)
            self.assertEqual(written, path)
            replace.assert_called_once()
            self.assertEqual(path.read_text(encoding="utf-8"), smoke.serialize_phase1_deepseek_smoke_report(self.report))
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_failed_scan_does_not_overwrite_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("existing-complete-report", encoding="utf-8")
            unsafe = dict(self.report)
            unsafe["request_attempt_count"] = 2
            with self.assertRaises(ValueError):
                smoke.write_phase1_deepseek_smoke_report(unsafe, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "existing-complete-report")

    def test_network_error_text_is_not_in_diagnostic_report(self) -> None:
        preparation = build_synthetic_phase1_preparation()
        class ErrorTransport:
            def send(self, payload, api_key, timeout):
                raise RuntimeError(SECRET_BODY)
        result = execute_phase1_deepseek_review(
            preparation, Phase1DeepSeekConfig(), ErrorTransport(), API_KEY
        )
        report = smoke.build_phase1_deepseek_smoke_report(
            result, request_attempt_count=1, api_request_count=1, api_key=API_KEY
        )
        self.assertNotIn(SECRET_BODY, smoke.serialize_phase1_deepseek_smoke_report(report))


class DeepSeekSmokeMainTests(unittest.TestCase):
    def test_import_does_not_read_environment(self) -> None:
        with patch.object(os.environ, "get", side_effect=AssertionError("env read on import")):
            importlib.reload(smoke)

    def test_only_main_reads_deepseek_api_key(self) -> None:
        source = Path(smoke.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count('os.environ.get("DEEPSEEK_API_KEY")'), 1)
        self.assertNotIn("load_dotenv", source)

    def test_missing_key_makes_zero_requests(self) -> None:
        transport = MagicMock()
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch.object(
            smoke, "DeepSeekHTTPSOneShotTransport", return_value=transport
        ), contextlib.redirect_stdout(output):
            code = smoke.main()
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue().strip(), "DEEPSEEK_API_KEY_MISSING")
        transport.send.assert_not_called()

    def test_empty_or_whitespace_key_makes_zero_requests(self) -> None:
        for value in ("", "   "):
            with self.subTest(value=value):
                transport = MagicMock()
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": value}, clear=True), patch.object(
                    smoke, "DeepSeekHTTPSOneShotTransport", return_value=transport
                ), contextlib.redirect_stdout(io.StringIO()):
                    code = smoke.main()
                self.assertEqual(code, 1)
                transport.send.assert_not_called()

    def test_main_does_not_read_other_environment_names(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, {"OTHER_API_KEY": API_KEY}, clear=True), contextlib.redirect_stdout(output):
            code = smoke.main()
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue().strip(), "DEEPSEEK_API_KEY_MISSING")

    def test_main_passes_unmodified_key_once(self) -> None:
        result = validated_result()
        fake_transport = MagicMock(request_attempt_count=1, api_request_count=1)
        captured = {}
        def fake_execute(preparation, config, transport, api_key):
            captured["api_key"] = api_key
            return result
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "  exact-key-with-spaces  "}, clear=True
        ), patch.object(smoke, "DeepSeekHTTPSOneShotTransport", return_value=fake_transport), patch.object(
            smoke, "execute_phase1_deepseek_review", side_effect=fake_execute
        ), patch.object(
            smoke, "DEFAULT_OUTPUT_PATH", Path(directory) / "report.json"
        ), patch.object(
            smoke, "write_phase1_deepseek_smoke_report", return_value=smoke.PROJECT_ROOT / "report.json"
        ), contextlib.redirect_stdout(io.StringIO()):
            code = smoke.main()
        self.assertEqual(code, 0)
        self.assertEqual(captured["api_key"], "  exact-key-with-spaces  ")

    def test_mock_environment_is_restored_after_test_scope(self) -> None:
        original = os.environ.get("DEEPSEEK_API_KEY")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "mock-only"}, clear=True):
            self.assertEqual(os.environ.get("DEEPSEEK_API_KEY"), "mock-only")
        self.assertEqual(os.environ.get("DEEPSEEK_API_KEY"), original)

    def test_all_transport_tests_mock_real_network(self) -> None:
        with patch.object(smoke.urllib_request, "build_opener", side_effect=AssertionError("real network forbidden")):
            transport = smoke.DeepSeekHTTPSOneShotTransport()
            with self.assertRaises(AssertionError):
                transport.send({}, API_KEY, 10)
        self.assertEqual(transport.api_request_count, 0)


if __name__ == "__main__":
    unittest.main()

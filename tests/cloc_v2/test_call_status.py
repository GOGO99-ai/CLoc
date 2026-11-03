from __future__ import annotations

import unittest

from src.cloc_v2.call_status import resolve_call_status, serialize_call_status


class CallStatusTests(unittest.TestCase):
    def test_nonempty_valid_output_is_success(self) -> None:
        built = resolve_call_status({"output": "0xdeadbeef"})
        self.assertEqual(built.status, "SUCCESS")
        self.assertEqual(built.decision_source, "CALLTRACER_OUTPUT")
        self.assertTrue(built.output_present)
        self.assertTrue(built.output_valid_hex)
        self.assertTrue(built.usable_as_success_evidence)

    def test_zero_length_hex_output_is_success(self) -> None:
        built = resolve_call_status({"output": "0x"})
        self.assertEqual(built.status, "SUCCESS")
        self.assertTrue(built.output_valid_hex)

    def test_missing_output_is_unknown(self) -> None:
        built = resolve_call_status({"type": "CALL"})
        self.assertEqual(built.status, "UNKNOWN")
        self.assertIn("CALLTRACER_OUTPUT_MISSING", built.uncertainties)

    def test_empty_and_null_output_are_unknown(self) -> None:
        for output, reason in (("", "CALLTRACER_OUTPUT_EMPTY_STRING"), (None, "CALLTRACER_OUTPUT_NULL")):
            with self.subTest(output=output):
                built = resolve_call_status({"output": output})
                self.assertEqual(built.status, "UNKNOWN")
                self.assertFalse(built.output_valid_hex)
                self.assertIn(reason, built.uncertainties)

    def test_non_hex_and_odd_length_output_are_unknown(self) -> None:
        for output in ("not-hex", "0x0", "0xz0"):
            with self.subTest(output=output):
                built = resolve_call_status({"output": output})
                self.assertEqual(built.status, "UNKNOWN")
                self.assertIn("CALLTRACER_OUTPUT_INVALID_HEX", built.uncertainties)

    def test_error_overrides_valid_output(self) -> None:
        built = resolve_call_status({"output": "0xdeadbeef", "error": "execution reverted"})
        self.assertEqual(built.status, "FAILED")
        self.assertEqual(built.decision_source, "CALLTRACER_ERROR")
        self.assertTrue(built.error_failure_evidence)
        self.assertFalse(built.usable_as_success_evidence)

    def test_revert_reason_overrides_valid_output(self) -> None:
        built = resolve_call_status({"output": "0xdeadbeef", "revertReason": "denied"})
        self.assertEqual(built.status, "FAILED")
        self.assertEqual(built.decision_source, "CALLTRACER_REVERT_REASON")
        self.assertTrue(built.revert_reason_failure_evidence)

    def test_empty_error_does_not_by_itself_mean_failure(self) -> None:
        built = resolve_call_status({"error": "   "})
        self.assertEqual(built.status, "UNKNOWN")
        self.assertFalse(built.error_failure_evidence)
        self.assertIn("CALLTRACER_OUTPUT_MISSING", built.uncertainties)

    def test_type_anomalies_are_conservative_unknown(self) -> None:
        cases = (
            ({"output": 7}, "CALLTRACER_OUTPUT_INVALID_TYPE"),
            ({"output": "0x", "error": 1}, "CALLTRACER_ERROR_FIELD_INVALID"),
            ({"output": "0x", "revertReason": []}, "CALLTRACER_REVERT_REASON_FIELD_INVALID"),
            ([], "CALLTRACER_FRAME_INVALID"),
        )
        for frame, reason in cases:
            with self.subTest(reason=reason):
                built = resolve_call_status(frame)
                self.assertEqual(built.status, "UNKNOWN")
                self.assertIn(reason, built.uncertainties)

    def test_dictionary_order_does_not_change_serialization(self) -> None:
        first = resolve_call_status({"output": "0xdeadbeef", "type": "CALL", "from": "x"})
        second = resolve_call_status({"from": "x", "type": "CALL", "output": "0xdeadbeef"})
        self.assertEqual(serialize_call_status(first), serialize_call_status(second))

    def test_serialization_excludes_raw_trace_values(self) -> None:
        secret_output = "0xdeadbeefcafebabe"
        secret_error = "sensitive-error-text"
        secret_input = "0x11111111cafebabe"
        serialized = serialize_call_status(
            resolve_call_status(
                {"input": secret_input, "output": secret_output, "error": secret_error}
            )
        )
        self.assertNotIn(secret_output, serialized)
        self.assertNotIn(secret_error, serialized)
        self.assertNotIn(secret_input, serialized)
        self.assertNotIn('"input"', serialized)
        self.assertNotIn('"output"', serialized)
        self.assertNotIn('"error"', serialized)


if __name__ == "__main__":
    unittest.main()

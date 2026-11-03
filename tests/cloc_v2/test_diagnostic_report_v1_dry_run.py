from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts.dry_run_diagnostic_report_v1 import (
    build_synthetic_diagnostic_report_v1,
    write_synthetic_diagnostic_report_v1,
)


class DiagnosticReportV1DryRunTests(unittest.TestCase):
    def test_synthetic_report_writes_strict_json_and_display_markdown(self):
        report, markdown = build_synthetic_diagnostic_report_v1()
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_synthetic_diagnostic_report_v1(Path(directory), report, markdown)
            document = json.loads(json_path.read_text(encoding="utf-8"))
            rendered = markdown_path.read_text(encoding="utf-8")
        self.assertEqual(document["schema_version"], "diagnostic_report_v1")
        self.assertEqual(document["status"], "DIAGNOSED")
        self.assertEqual(len(document["top3_source_locations"]), 3)
        self.assertLessEqual(len(document["root_cause_explanations"]), 3)
        root_cause = document["root_cause_explanations"][0]
        for field in ("trigger_condition", "erroneous_behavior", "causal_propagation", "final_impact"):
            self.assertIn(field, root_cause)
            self.assertNotRegex(root_cause[field], r"\b[QESF][1-9][0-9]*\b")
        self.assertIn("# CLoc 智能合约交易诊断报告", rendered)
        self.assertIn("contracts/SyntheticVault.sol", rendered)
        self.assertIn("execute(uint256)", rendered)
        self.assertIn("10–15", rendered)
        self.assertIn("**触发条件**", rendered)
        self.assertIn("**错误行为**", rendered)
        self.assertIn("**因果传播**", rendered)
        self.assertIn("**最终影响**", rendered)
        self.assertIn("**证据引用**", rendered)

    def test_synthetic_artifacts_are_stable_and_sanitized(self):
        first_report, first_markdown = build_synthetic_diagnostic_report_v1()
        second_report, second_markdown = build_synthetic_diagnostic_report_v1()
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_markdown, second_markdown)
        text = json.dumps(first_report, ensure_ascii=False, sort_keys=True) + first_markdown
        self.assertIsNone(re.search(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])", text))
        self.assertIsNone(re.search(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])", text))
        self.assertNotIn("reasoning_content", text)
        self.assertNotIn("provider_response", text)
        self.assertEqual(first_report["provenance"]["external_request_counts"], {"rpc": 0, "source_network": 0, "llm": 0})
        self.assertFalse(first_report["provenance"]["cause_used"])
        self.assertFalse(first_report["provenance"]["location_used"])

    def test_markdown_keeps_fact_and_inference_visibly_separate(self):
        _, markdown = build_synthetic_diagnostic_report_v1()
        self.assertIn("[已确认事实]", markdown)
        self.assertIn("[LLM推断", markdown)
        self.assertIn("文件、函数和行范围来自已确认源码元数据", markdown)
        self.assertIn("没有重新排名", markdown)


if __name__ == "__main__":
    unittest.main()

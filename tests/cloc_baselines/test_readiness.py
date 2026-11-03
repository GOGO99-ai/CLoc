from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_cloc_baseline_readiness import audit_readiness


class ReadinessTests(unittest.TestCase):
    def test_missing_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "001_demo",
                                "case_name": "Demo",
                                "chain_id": 1,
                                "chain_key": "eth",
                                "selected_fault_transactions": ["0x" + "1" * 64],
                                "normal_transactions": ["0x" + "2" * 64],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = audit_readiness(
                manifest_path=manifest,
                coverage_dir=root / "coverage",
                grace_dir=root / "grace",
            )
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["coverage_ready_case_count"], 0)
            self.assertEqual(result["grace_ready_case_count"], 0)
            self.assertEqual(result["reason_code_counts"]["FUNCTION_COVERAGE_MISSING"], 1)
            self.assertEqual(result["reason_code_counts"]["GRACE_GRAPH_MISSING"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cloc_baselines.manifest import build_cloc_baseline_manifest


class ManifestTests(unittest.TestCase):
    def test_representative_tree_transaction_is_locked_and_artifacts_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset" / "Logic bug"
            trees = root / "trees" / "Logic bug"
            samples = root / "samples" / "eth"
            dataset.mkdir(parents=True)
            trees.mkdir(parents=True)
            samples.mkdir(parents=True)
            fault_one = "0x" + "1" * 64
            fault_two = "0x" + "2" * 64
            normal = "0x" + "3" * 64
            (dataset / "demo.json").write_text(
                json.dumps(
                    {
                        "name": "Demo",
                        "cause": "logic",
                        "platform": "Ethereum",
                        "fault": {
                            "transaction_hash": [[fault_one], [fault_two]],
                            "location": ["0x" + "a" * 40 + "#C.sol#1:2"],
                        },
                        "faultless": {"transaction_hash": [normal]},
                    }
                ),
                encoding="utf-8",
            )
            (trees / f"{fault_two}.json").write_text("{}", encoding="utf-8")
            (samples / f"{normal}.json").write_text(
                json.dumps({"tx_hash": normal, "feature_records": [{"call_steps": []}]}),
                encoding="utf-8",
            )
            result = build_cloc_baseline_manifest(
                dataset_root=root / "dataset",
                call_tree_root=root / "trees",
                normal_samples_root=root / "samples",
            )
            self.assertEqual(result["summary"]["case_count"], 1)
            case = result["cases"][0]
            self.assertEqual(case["selected_fault_transactions"], [fault_two])
            self.assertEqual(case["all_fault_transaction_groups"], [[fault_one], [fault_two]])
            self.assertEqual(case["local_artifacts"]["selected_fault_call_tree_count"], 1)
            self.assertEqual(case["local_artifacts"]["usable_normal_call_feature_count"], 1)
            self.assertFalse(case["readiness"]["function_coverage_ready"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import importlib.util
import unittest

from src.cloc_baselines.metrics import evaluate_rankings
from src.cloc_baselines.ranking import ochiai_score, rank_case, tarantula_score
from src.cloc_baselines.schema import CoverageCase
from src.cloc_baselines.cnnfl import rank_cnnfl_case


def _case() -> CoverageCase:
    return CoverageCase.from_dict(
        {
            "schema_version": "cloc_function_coverage_v1",
            "case_id": "001_demo",
            "case_name": "Demo",
            "chain_id": 1,
            "chain_key": "eth",
            "functions": [
                {"function_id": "f1", "address": "0x" + "1" * 40},
                {"function_id": "f2", "address": "0x" + "2" * 40},
                {"function_id": "f3", "address": "0x" + "3" * 40},
            ],
            "tests": [
                {"transaction_hash": "0x" + "a" * 64, "outcome": "fail", "covered_function_ids": ["f1", "f2"]},
                {"transaction_hash": "0x" + "b" * 64, "outcome": "fail", "covered_function_ids": ["f1"]},
                {"transaction_hash": "0x" + "c" * 64, "outcome": "pass", "covered_function_ids": ["f2", "f3"]},
                {"transaction_hash": "0x" + "d" * 64, "outcome": "pass", "covered_function_ids": ["f3"]},
            ],
            "fault_function_ids": ["f1"],
            "call_edges": [["f1", "f2"], ["f2", "f3"]],
        }
    )


class RankingTests(unittest.TestCase):
    def test_reference_equations(self) -> None:
        self.assertEqual(tarantula_score(0, 0, 1, 1), 0.0)
        self.assertAlmostEqual(tarantula_score(2, 0, 0, 2), 1.0)
        self.assertAlmostEqual(ochiai_score(2, 0, 0, 2), 1.0)
        self.assertAlmostEqual(ochiai_score(1, 1, 1, 1), 0.5)

    def test_tarantula_and_ochiai_rank_fault_first(self) -> None:
        case = _case()
        for method in ("tarantula", "ochiai"):
            rows = rank_case(case, method)
            self.assertEqual(rows[0].function_id, "f1")
            self.assertTrue(rows[0].is_fault)
            self.assertEqual((rows[0].ef, rows[0].ep, rows[0].nf, rows[0].np), (2, 0, 0, 2))

    def test_prfl_variants_are_finite_and_deterministic(self) -> None:
        case = _case()
        for method in ("prfl", "prfl_malo"):
            first = rank_case(case, method)
            second = rank_case(case, method)
            self.assertEqual(first, second)
            self.assertTrue(all(math.isfinite(item.score) for item in first))

    def test_metrics(self) -> None:
        rows = rank_case(_case(), "ochiai")
        metrics = evaluate_rankings({"001_demo": rows})
        self.assertEqual(metrics["hit_counts"], {"top_1": 1, "top_3": 1, "top_5": 1})
        self.assertEqual(metrics["mfr"], 1.0)
        self.assertEqual(metrics["mar"], 1.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_cnnfl_one_epoch_smoke(self) -> None:
        raw = _case().to_dict()
        raw["functions"].append({"function_id": "f4", "address": "0x" + "4" * 40})
        raw["tests"][0]["covered_function_ids"].append("f3")
        raw["tests"][0]["covered_function_ids"].append("f4")
        raw["tests"][2]["covered_function_ids"].append("f4")
        case = CoverageCase.from_dict(raw)
        rows = rank_cnnfl_case(case, seed=7, epochs=1, device="cpu")
        self.assertEqual(len(rows), 4)
        self.assertEqual({item.function_id for item in rows}, {"f1", "f2", "f3", "f4"})
        self.assertTrue(all(0.0 <= item.score <= 1.0 for item in rows))


if __name__ == "__main__":
    unittest.main()

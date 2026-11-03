from __future__ import annotations

import importlib.util
import unittest

from src.cloc_baselines.grace import GraceGraphCase, rank_grace_leave_one_out
from src.cloc_baselines.schema import CoverageCase


def _case(index: int) -> CoverageCase:
    address = "0x" + str(index) * 40
    prefix = f"c{index}"
    return CoverageCase.from_dict(
        {
            "schema_version": "cloc_function_coverage_v1",
            "case_id": f"00{index}_demo",
            "case_name": f"Demo {index}",
            "chain_id": 1,
            "chain_key": "eth",
            "functions": [
                {"function_id": f"{prefix}_f1", "address": address},
                {"function_id": f"{prefix}_f2", "address": address},
            ],
            "lines": [
                {
                    "line_id": f"{prefix}_l1",
                    "function_id": f"{prefix}_f1",
                    "address": address,
                    "line_number": 1,
                },
                {
                    "line_id": f"{prefix}_l2",
                    "function_id": f"{prefix}_f2",
                    "address": address,
                    "line_number": 2,
                },
            ],
            "tests": [
                {
                    "transaction_hash": "0x" + f"{index}" * 64,
                    "outcome": "fail",
                    "covered_function_ids": [f"{prefix}_f1", f"{prefix}_f2"],
                    "covered_line_ids": [f"{prefix}_l1", f"{prefix}_l2"],
                },
                {
                    "transaction_hash": "0x" + f"{index + 2}" * 64,
                    "outcome": "pass",
                    "covered_function_ids": [f"{prefix}_f2"],
                    "covered_line_ids": [f"{prefix}_l2"],
                },
            ],
            "fault_function_ids": [f"{prefix}_f1"],
        }
    )


class GraceTests(unittest.TestCase):
    def test_graph_is_derived_without_extra_labels(self) -> None:
        graph = GraceGraphCase.from_coverage(_case(1))
        self.assertEqual(graph.function_ids, ("c1_f1", "c1_f2"))
        self.assertEqual(len(graph.lines), 2)
        self.assertEqual(graph.metadata["test_selection_policy"], "FIXED_EPOCH_NO_HELD_OUT_LABEL_SELECTION")
        self.assertEqual(GraceGraphCase.from_dict(graph.to_dict()), graph)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_one_epoch_leave_one_out_smoke(self) -> None:
        cases = [_case(1), _case(2)]
        graphs = [GraceGraphCase.from_coverage(case) for case in cases]
        rankings = rank_grace_leave_one_out(
            graphs,
            {case.case_id: case for case in cases},
            seed=3,
            epochs=1,
            layers=1,
            device="cpu",
        )
        self.assertEqual(set(rankings), {"001_demo", "002_demo"})
        self.assertTrue(all(len(rows) == 2 for rows in rankings.values()))


if __name__ == "__main__":
    unittest.main()

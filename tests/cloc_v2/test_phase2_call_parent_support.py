from __future__ import annotations

import unittest

from src.cloc_v2.normal_distance import CallStep
from src.cloc_v2.phase2_call_parent_support import map_abnormal_calls_to_unique_parent_keys
from src.cloc_v2.source_index import selector_for_signature


ADDRESS = "0x" + "a" * 40
PARENT = ("contracts/C.sol", "entry()", 2, 4)


def slices(*, callee_count: int = 1, contract_alias: str = "K1"):
    parent = {
        "relative_file": PARENT[0],
        "function_signature": PARENT[1],
        "start_line": PARENT[2],
        "end_line": PARENT[3],
        "selector_match_status": "SELECTOR_EXACT_MATCH",
        "direct_internal_callee_count": callee_count,
    }
    public_parent = {**parent, "contract_alias": contract_alias}
    return {
        "schema_version": "source_request_slices_v1",
        "requests": [{
            "source_address": ADDRESS,
            "file_path": "contracts/C.sol",
            "signature": "helper()",
            "start_line": 5,
            "end_line": 6,
            "unique_parent_function_identity": parent,
        }],
        "metadata": {"candidate_union": {"candidates": [{
            "contract_alias": contract_alias,
            "relative_file": "contracts/C.sol",
            "function_signature": "helper()",
            "start_line": 5,
            "end_line": 6,
            "unique_parent_function_identity": public_parent,
        }]}},
    }


class Phase2CallParentSupportTests(unittest.TestCase):
    def test_exact_abnormal_parent_maps_to_contract_scoped_key(self):
        call = CallStep(ADDRESS, selector_for_signature(PARENT[1]))
        self.assertEqual(
            map_abnormal_calls_to_unique_parent_keys(slices(), (call,)),
            (("K1", *PARENT),),
        )

    def test_different_call_identity_does_not_map(self):
        other = CallStep(ADDRESS, selector_for_signature("other()"))
        self.assertEqual(map_abnormal_calls_to_unique_parent_keys(slices(), (other,)), ())

    def test_parent_with_multiple_internal_callees_does_not_map(self):
        call = CallStep(ADDRESS, selector_for_signature(PARENT[1]))
        self.assertEqual(map_abnormal_calls_to_unique_parent_keys(slices(callee_count=2), (call,)), ())


if __name__ == "__main__":
    unittest.main()

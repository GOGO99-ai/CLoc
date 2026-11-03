from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import src.cloc_v2.phase2_legacy_v4_profile as legacy
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index


def _row(
    alias: str,
    *,
    path: str = "contracts/A.sol",
    signature: str = "f()",
    direct: bool = False,
    selector: bool = False,
    asset: bool = False,
    relation: int = 3,
    distance: int = 3,
    depth: int = 0,
    start: int = 10,
) -> dict:
    return {
        "candidate_alias": alias,
        "contract_alias": "K1",
        "relative_file": path,
        "function_signature": signature,
        "visibility": "external",
        "start_line": start,
        "end_line": start + 2,
        "selector_match_status": "SELECTOR_EXACT_MATCH" if selector else "FUNCTION_NAME_UNIQUE_MATCH",
        "candidate_sources": ["DIRECT_STRICT_CHAIN_SOURCE" if direct else "TRANSACTION_CALL_GRAPH_SOURCE"],
        "call_relation_modes": ["STRICT_CHAIN_NODE" if direct else "UNDIRECTED_CALL_CORRIDOR"],
        "shortest_call_path_length": distance,
        "internal_expansion_depth": depth,
        "supporting_evidence_ids": ["E1"],
        "ranking_features": {
            "direct_strict_evidence": direct,
            "selector_priority": 0 if selector else 1,
            "asset_result_endpoint": asset,
            "asset_related": asset,
            "call_asset_joint_support": asset,
            "call_relation_priority": relation,
            "shortest_call_path_length": distance,
            "internal_expansion_depth": depth,
            "address_match_source": "RUNTIME_ADDRESS_MATCH",
            "address_match_priority": 1,
            "executed_call_node": True,
            "verified_source_conflict_free": True,
        },
    }


def _source(rows: list[dict], *, issue_codes: list[str] | None = None) -> dict:
    requests = [
        {
            "file_path": row["relative_file"],
            "signature": row["function_signature"],
            "function_name": row["function_signature"].split("(", 1)[0],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "source": "function body is internal-only test data",
            "function_match_status": "RESOLVED",
        }
        for row in rows
    ]
    union = {
        "schema_version": "phase2_source_candidate_union_v1",
        "candidate_count": len(rows),
        "max_candidates": 20,
        "total_candidate_count_before_truncation": len(rows),
        "candidate_count_before_truncation": len(rows),
        "candidate_count_after_truncation": len(rows),
        "candidate_limit_reached": len(rows) == 20,
        "candidate_limit_exceeded": False,
        "fallback_candidate_count": 0,
        "fallback_candidate_count_before_truncation": 0,
        "fallback_truncated_candidate_count": 0,
        "coarse_source_function_supplement_count_before_truncation": 0,
        "coarse_source_function_supplement_count_after_truncation": 0,
        "coarse_source_function_supplement_truncated_count": 0,
        "candidate_source_counts_before_truncation": {},
        "candidate_source_counts_after_truncation": {},
        "truncated_candidate_count": 0,
        "issue_codes": issue_codes or [],
        "candidates": deepcopy(rows),
    }
    return {"schema_version": "source_request_slices_v1", "requests": requests, "metadata": {"candidate_union": union}}


def _build(rows: list[dict]) -> dict:
    value = _source(rows)
    with patch.object(legacy, "build_phase2_source_graph_fallback", return_value=value) as mocked:
        result = legacy.build_legacy_v4_candidates({}, {}, {}, object(), Path("cache"))
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["max_candidates"] == 20
    return result


class LegacyV4ProfileTests(unittest.TestCase):
    def test_locked_constants_and_immutability(self):
        profile = legacy.LOCKED_LEGACY_V4_PROFILE
        self.assertEqual(profile.profile_id, "CLOC_PHASE2_V4_LEGACY_STYLE_LOCKED_V1")
        self.assertEqual(profile.status, "RECONSTRUCTED_AND_LOCKED")
        self.assertEqual((profile.candidate_pool_limit, profile.prompt_candidate_limit, profile.topk_limit), (20, 12, 5))
        self.assertEqual((profile.max_deepseek_requests_per_sample, profile.retry_enabled), (1, False))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            profile.topk_limit = 3
        with self.assertRaises(ValueError):
            legacy.LegacyV4Profile(candidate_pool_limit=21)

    def test_profile_sha_and_serialization_are_stable(self):
        first = legacy.serialize_legacy_v4_profile()
        second = legacy.serialize_legacy_v4_profile()
        self.assertEqual(first, second)
        value = json.loads(first)
        self.assertEqual(value["configuration_sha256"], legacy.legacy_v4_profile_sha256())
        self.assertFalse(value["reconstruction_notice"]["historical_byte_identity_proven"])
        self.assertFalse(value["deepseek_request_policy"]["retry_enabled"])

    def test_direct_then_selector_then_asset_priority(self):
        rows = [
            _row("C1", signature="plain()"),
            _row("C2", signature="asset()", asset=True),
            _row("C3", signature="selector()", selector=True),
            _row("C4", signature="direct()", direct=True),
        ]
        result = _build(rows)
        ordered = result["metadata"]["candidate_union"]["candidates"]
        self.assertEqual([row["function_signature"] for row in ordered], ["direct()", "selector()", "asset()", "plain()"])

    def test_call_relation_distance_and_internal_depth_priority(self):
        rows = [
            _row("C1", signature="deep()", relation=2, distance=1, depth=2),
            _row("C2", signature="far()", relation=2, distance=3, depth=0),
            _row("C3", signature="near()", relation=2, distance=1, depth=0),
            _row("C4", signature="strongRelation()", relation=1, distance=9, depth=2),
        ]
        result = _build(rows)
        ordered = result["metadata"]["candidate_union"]["candidates"]
        self.assertEqual([row["function_signature"] for row in ordered], ["strongRelation()", "near()", "deep()", "far()"])

    def test_final_file_signature_line_and_stable_id_order(self):
        rows = [
            _row("C4", path="contracts/B.sol", signature="a()", start=1),
            _row("C3", path="contracts/A.sol", signature="b()", start=1),
            _row("C2", path="contracts/A.sol", signature="a()", start=20),
            _row("C1", path="contracts/A.sol", signature="a()", start=10),
        ]
        result = _build(rows)
        ordered = result["metadata"]["candidate_union"]["candidates"]
        self.assertEqual(
            [(row["file_alias"], row["function_signature"], row["start_line"]) for row in ordered],
            [("F1", "a()", 10), ("F1", "a()", 20), ("F1", "b()", 1), ("F2", "a()", 1)],
        )

    def test_pool_is_twenty_and_prompt_prefix_is_twelve(self):
        rows = [_row(f"C{i}", path=f"contracts/F{i:02d}.sol", signature=f"f{i:02d}()", start=i) for i in range(1, 21)]
        result = _build(list(reversed(rows)))
        selected = legacy.select_legacy_v4_prompt_candidates(result)
        self.assertEqual(len(result["metadata"]["candidate_union"]["candidates"]), 20)
        self.assertEqual(len(selected), 12)
        self.assertEqual([row["candidate_alias"] for row in selected], [f"C{i}" for i in range(1, 13)])
        self.assertEqual(result["metadata"]["candidate_union"]["legacy_prompt_candidate_ids"], [f"C{i}" for i in range(1, 13)])

    def test_candidate_pool_drift_above_twenty_is_rejected(self):
        rows = [_row(f"C{i}", path=f"contracts/F{i:02d}.sol", signature=f"f{i:02d}()", start=i) for i in range(1, 22)]
        value = _source(rows)
        value["metadata"]["candidate_union"]["max_candidates"] = 20
        with patch.object(legacy, "build_phase2_source_graph_fallback", return_value=value):
            with self.assertRaisesRegex(ValueError, "LEGACY_V4_CANDIDATE_LIMIT_DRIFT"):
                legacy.build_legacy_v4_candidates({}, {}, {}, object(), Path("cache"))

    def test_same_input_is_deterministic(self):
        rows = [_row("C2", signature="b()"), _row("C1", signature="a()")]
        first = _build(rows)
        second = _build(deepcopy(rows))
        self.assertEqual(first, second)
        self.assertEqual(legacy.select_legacy_v4_prompt_candidates(first), legacy.select_legacy_v4_prompt_candidates(second))

    def test_real_candidate_builder_path_retains_verified_call_graph_union(self):
        from tests.cloc_v2.test_phase2_source_graph_fallback import _cache, _inputs

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            _cache(root, addresses[1], source="contract C { function f() external {} }")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = legacy.build_legacy_v4_candidates(graph, strict, existing, index, root)
            union = result["metadata"]["candidate_union"]
            self.assertEqual(union["legacy_profile_id"], legacy.PROFILE_ID)
            self.assertLessEqual(union["candidate_count"], 20)
            self.assertNotIn("COARSE_SOURCE_FUNCTION_SUPPLEMENT_USED", union["issue_codes"])
            self.assertEqual(
                [row["candidate_alias"] for row in union["candidates"]],
                [f"C{i}" for i in range(1, union["candidate_count"] + 1)],
            )

    def test_coexistence_supplement_fails_closed(self):
        rows = [_row("C1")]
        rows[0]["candidate_sources"].append("COARSE_SOURCE_FUNCTION_SUPPLEMENT")
        value = _source(rows, issue_codes=["COARSE_SOURCE_FUNCTION_SUPPLEMENT_USED"])
        with patch.object(legacy, "build_phase2_source_graph_fallback", return_value=value):
            with self.assertRaisesRegex(ValueError, "LEGACY_V4_COEXISTENCE_FALLBACK_DRIFT"):
                legacy.build_legacy_v4_candidates({}, {}, {}, object(), Path("cache"))

    def test_profile_disables_multiview_v5_v6_and_label_guidance(self):
        disabled = set(legacy.LOCKED_LEGACY_V4_PROFILE.disabled_capabilities)
        self.assertIn("MULTI_VIEW_CANDIDATE_RANK_FUSION", disabled)
        self.assertIn("PHASE2_COARSE_TO_FINE_TOPK_V5", disabled)
        self.assertIn("PHASE2_SOURCE_TOURNAMENT_TOPK_V6", disabled)
        self.assertIn("LOCATION_OR_CAUSE_GUIDED_SELECTION", disabled)
        source = Path(legacy.__file__).read_text(encoding="utf-8")
        self.assertNotIn("phase2_localization_bridge", source)
        self.assertNotIn("phase2_source_tournament import", source)
        self.assertNotIn("phase2_candidate_shortlist", source)

    def test_locked_artifact_exactly_matches_profile_serializer(self):
        path = Path(__file__).resolve().parents[2] / "data" / "prototypes" / "phase2_v4_legacy_style_profile_lock_20260824" / "locked_profile.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), json.loads(legacy.serialize_legacy_v4_profile()))


if __name__ == "__main__":
    unittest.main()

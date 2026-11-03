from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cloc_v2.phase2_source_graph_fallback import (
    MAX_INTERNAL_EXPANSION_DEPTH,
    MAX_SOURCE_FUNCTION_CANDIDATES,
    MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
    MAX_SOURCE_SLICES,
    MODE,
    build_phase2_source_candidate_union,
    build_phase2_source_graph_fallback,
    serialize_phase2_source_candidate_union,
)
from src.cloc_v2.source_index import selector_for_signature
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index


def _address(value: int) -> str:
    return "0x" + f"{value:040x}"


def _cache(root: Path, address: str, *, source: str | None = None, proxy: dict | None = None) -> str:
    source = source or "contract C { function run() external {} }"
    directory = root / "eth" / address / "contracts"
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / "C.sol"
    file.write_text(source, encoding="utf-8")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    (directory.parent / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "source_manifest_v1",
                "cached_at": "v1",
                "requested_address": address,
                "resolved_address": address,
                "status": "verified",
                "verified": True,
                "files": [{"path": "contracts/C.sol", "sha256": digest}],
                "proxy": proxy or {},
            }
        ),
        encoding="utf-8",
    )
    return hashlib.sha256((directory.parent / "manifest.json").read_bytes()).hexdigest()


def _inputs(*, path: bool = True, existing_resolved: bool = False):
    addresses = (_address(1), _address(2), _address(3))
    graph = {
        "schema_version": "transaction_graph_v1",
        "nodes": [
            {"node_id": "n1", "trace_index": 1, "callee_address": addresses[0], "function_name": "run", "function_selector": "0x00000000"},
            {"node_id": "n2", "trace_index": 2, "callee_address": addresses[1], "function_name": "middle", "function_selector": "0x00000000"},
            {"node_id": "n3", "trace_index": 3, "callee_address": addresses[2], "function_name": "finish", "function_selector": "0x00000000"},
        ],
        "call_edges": [
            {"edge_id": "call_e001", "edge_type": "CALL", "src": "n1", "dst": "n2"},
            {"edge_id": "call_e002", "edge_type": "CALL", "src": "n2", "dst": "n3"},
        ] if path else [],
    }
    strict = {
        "schema_version": "strict_result_chains_v1",
        "chains": [{"strict_chain_id": "strict_1", "node_ids": ["n2", "n3"]}],
    }
    existing = {
        "schema_version": "source_request_slices_v1",
        "chain": "eth",
        "requests": [{"node_id": "n2", "function_match_status": "RESOLVED"}] if existing_resolved else [],
        "metadata": {"resolution_mode": "NONE"},
    }
    return graph, strict, existing, addresses


def _resolved_request(node_id: str, address: str, *, function: str = "run", evidence_ids=()):
    return {
        "chain_id": "Q",
        "node_id": node_id,
        "focus": "phase2_localization",
        "priority": 0,
        "requested_address": address,
        "source_address": address,
        "status": "RESOLVED",
        "function_match_status": "RESOLVED",
        "file_path": "contracts/C.sol",
        "contract_name": "C",
        "function_name": function,
        "signature": f"{function}()",
        "selector": selector_for_signature(f"{function}()"),
        "start_line": 1,
        "end_line": 1,
        "source": f"1: contract C {{ function {function}() external {{}} }}",
        "candidate_functions": [],
        "proxy_resolution_path": [address],
        "uncertainties": [],
        "evidence_ids": list(evidence_ids),
        "resolution_mode": "EXACT_FUNCTION_SLICE",
    }


class Phase2SourceGraphFallbackTests(unittest.TestCase):
    def test_direct_and_call_graph_verified_functions_are_merged(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            existing["requests"] = [_resolved_request("n2", addresses[1], function="middle", evidence_ids=("strict_evidence",))]
            _cache(root, addresses[0])
            _cache(root, addresses[1], source="contract C { function middle() external {} }")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            signatures = [item.get("signature") for item in result["requests"]]
            self.assertIn("middle()", signatures)
            self.assertIn("run()", signatures)
            self.assertEqual(result["metadata"]["candidate_union"]["candidate_count"], 2)

    def test_existing_verified_window_is_preserved_when_no_exact_function_exists(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, _addresses = _inputs()
            existing["requests"] = [{"node_id": "n2", "function_match_status": "WINDOW_FALLBACK", "file_path": "contracts/C.sol", "start_line": 1, "end_line": 2}]
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            self.assertEqual(result["requests"], existing["requests"])
            self.assertEqual(result["metadata"]["candidate_union"]["candidate_count"], 0)

    def test_cache_hit_with_real_call_path_builds_coarse_fallback(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            _cache(root, addresses[0])
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            self.assertEqual(result["metadata"]["source_selection_mode"], MODE)
            self.assertEqual(len(result["requests"]), 1)
            association = result["requests"][0]["fallback_association"]
            self.assertEqual(association["strict_chain_id"], "strict_1")
            self.assertEqual(association["call_edge_ids"], ["call_e001", "call_e002"])
            self.assertEqual(association["call_path_length"], 2)
            self.assertEqual(association["relation_kind"], "DIRECTED_TO_RESULT")
            self.assertEqual(association["path_directionality"], "FORWARD")
            self.assertIn("SOURCE_TO_RESULT_CALL_PATH_COARSE", result["requests"][0]["uncertainties"])

    def test_strict_chain_node_cache_hit_has_identity_relation(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            _cache(root, addresses[1])
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            association = result["requests"][0]["fallback_association"]
            self.assertEqual(association["relation_kind"], "STRICT_CHAIN_NODE")
            self.assertEqual(association["path_directionality"], "IDENTITY")
            self.assertEqual(association["call_edge_ids"], [])
            self.assertEqual(association["call_path_length"], 0)

    def test_strict_chain_to_candidate_forward_path_is_reverse_relation(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            descendant = _address(10)
            graph["nodes"].append({"node_id": "n4", "trace_index": 4, "callee_address": descendant, "function_name": "run", "function_selector": "0x00000000"})
            graph["call_edges"].append({"edge_id": "call_e003", "edge_type": "CALL", "src": "n3", "dst": "n4"})
            _cache(root, descendant)
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            association = result["requests"][0]["fallback_association"]
            self.assertEqual(association["relation_kind"], "DIRECTED_FROM_STRICT_CHAIN")
            self.assertEqual(association["path_directionality"], "REVERSE")
            self.assertEqual(association["call_edge_ids"], ["call_e003"])

    def test_sibling_branch_uses_undirected_real_call_corridor(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            sibling = _address(11)
            graph["nodes"].append({"node_id": "n4", "trace_index": 4, "callee_address": sibling, "function_name": "run", "function_selector": "0x00000000"})
            graph["call_edges"].append({"edge_id": "call_e003", "edge_type": "CALL", "src": "n1", "dst": "n4"})
            _cache(root, sibling)
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            association = result["requests"][0]["fallback_association"]
            self.assertEqual(association["relation_kind"], "UNDIRECTED_CALL_CORRIDOR")
            self.assertEqual(association["path_directionality"], "UNDIRECTED")
            self.assertEqual(association["call_edge_ids"], ["call_e001", "call_e003"])
            self.assertIn("UNDIRECTED_CALL_CORRIDOR_USED", result["requests"][0]["uncertainties"])
            self.assertIn("UNDIRECTED_CALL_CORRIDOR_USED", result["metadata"]["fallback_issue_codes"])

    def test_cache_node_without_a_real_call_path_is_excluded(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs(path=False)
            _cache(root, addresses[0])
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            self.assertEqual(result["requests"], [])

    def test_effective_and_proxy_identity_are_normalized_before_source_slicing(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            implementation = _address(40)
            child_manifest = _cache(root, implementation)
            _cache(root, addresses[0], proxy={"implementation_address": implementation, "implementation_manifest_sha256": child_manifest})
            graph["nodes"][0]["effective_address"] = implementation
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            self.assertEqual(len(result["requests"]), 1)
            self.assertEqual(result["requests"][0]["source_address"], implementation)
            self.assertEqual(result["requests"][0]["fallback_association"]["address_match_source"], "EFFECTIVE_ADDRESS_MATCH")

    def test_conflicting_cache_binding_is_never_used(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            implementation = _address(41)
            child_manifest = _cache(root, implementation, source="contract Impl { function run() external {} }")
            _cache(root, addresses[0], source="contract Proxy { function run() external {} }", proxy={"implementation_address": implementation, "implementation_manifest_sha256": child_manifest})
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            self.assertEqual(result["requests"], [])
            self.assertIn("VERIFIED_SOURCE_CACHE_CONFLICT", result["metadata"]["candidate_union"]["issue_codes"])

    def test_output_is_stable_and_slice_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            for index_value in range(1, 16):
                address = _address(index_value)
                _cache(root, address)
                node_id = f"c{index_value:02d}"
                graph["nodes"].append({"node_id": node_id, "trace_index": index_value + 3, "callee_address": address, "function_name": "run", "function_selector": "0x00000000"})
                graph["call_edges"].append({"edge_id": f"edge_{index_value:02d}", "edge_type": "CALL", "src": node_id, "dst": "n3"})
            cache_index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            first = build_phase2_source_graph_fallback(graph, strict, existing, cache_index, root)
            second = build_phase2_source_graph_fallback(graph, strict, existing, cache_index, root)
            self.assertEqual(first, second)
            self.assertLessEqual(len(first["requests"]), MAX_SOURCE_FUNCTION_CANDIDATES)
            self.assertEqual(MAX_SOURCE_SLICES, 12)

    def test_tiers_then_distance_define_stable_candidate_order(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            descendant = _address(10)
            sibling_near = _address(11)
            sibling_far = _address(12)
            graph["nodes"].extend([
                {"node_id": "n4", "trace_index": 9, "callee_address": descendant, "function_name": "run", "function_selector": "0x00000000"},
                {"node_id": "n5", "trace_index": 7, "callee_address": sibling_near, "function_name": "run", "function_selector": "0x00000000"},
                {"node_id": "n6", "trace_index": 6, "callee_address": sibling_far, "function_name": "run", "function_selector": "0x00000000"},
                {"node_id": "n7", "trace_index": 5, "callee_address": _address(13), "function_name": "none", "function_selector": "0x00000000"},
            ])
            graph["call_edges"].extend([
                {"edge_id": "call_e003", "edge_type": "CALL", "src": "n3", "dst": "n4"},
                {"edge_id": "call_e004", "edge_type": "CALL", "src": "n1", "dst": "n5"},
                {"edge_id": "call_e005", "edge_type": "CALL", "src": "n7", "dst": "n6"},
                {"edge_id": "call_e006", "edge_type": "CALL", "src": "n1", "dst": "n7"},
            ])
            _cache(root, addresses[1], source="contract C { function middle() external {} }")
            _cache(root, addresses[0])
            for address in (descendant, sibling_near, sibling_far):
                _cache(root, address)
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            relations = [item["fallback_association"]["relation_kind"] for item in result["requests"]]
            self.assertEqual(relations[:3], ["STRICT_CHAIN_NODE", "DIRECTED_TO_RESULT", "DIRECTED_FROM_STRICT_CHAIN"])
            corridor_lengths = [item["fallback_association"]["call_path_length"] for item in result["requests"] if item["fallback_association"]["relation_kind"] == "UNDIRECTED_CALL_CORRIDOR"]
            self.assertEqual(corridor_lengths, sorted(corridor_lengths))

    def test_duplicate_node_binding_and_function_slice_is_removed(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            _cache(root, addresses[0])
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            duplicate = {
                "chain_id": "Q", "node_id": "n1", "focus": "phase2_localization", "priority": 1,
                "function_match_status": "RESOLVED", "file_path": "contracts/C.sol", "function_name": "run",
                "signature": "run()", "start_line": 1, "end_line": 1, "uncertainties": [],
            }
            sliced = {"schema_version": "source_request_slices_v1", "chain": "eth", "requests": [duplicate, dict(duplicate)], "metadata": {}}
            with patch("src.cloc_v2.phase2_source_graph_fallback.build_source_request_slices", return_value=sliced), patch("src.cloc_v2.phase2_source_graph_fallback.build_verified_source_window_fallback", side_effect=lambda _i, _g, exact, _r: exact):
                result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            self.assertEqual(len(result["requests"]), 1)

    def test_same_function_from_direct_and_graph_sources_merges_evidence(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][1]["function_name"] = "run"
            strict["chains"][0]["edge_ids"] = ["strict_call_e001"]
            existing["requests"] = [_resolved_request("n2", addresses[1], evidence_ids=("direct_evidence",))]
            _cache(root, addresses[1])
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual(union["candidate_count"], 1)
            candidate = union["candidates"][0]
            self.assertEqual(candidate["candidate_sources"], ["DIRECT_STRICT_CHAIN_SOURCE", "TRANSACTION_CALL_GRAPH_EXECUTED_FUNCTION"])
            self.assertIn("direct_evidence", candidate["supporting_evidence_ids"])
            self.assertIn("strict_call_e001", candidate["supporting_evidence_ids"])

    def test_identical_source_function_from_cloned_deployments_is_merged(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][1]["function_name"] = "run"
            strict["chains"][0]["node_ids"] = ["n1", "n2"]
            _cache(root, addresses[0])
            _cache(root, addresses[1])
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual(union["candidate_count"], 1)
            self.assertEqual(union["candidates"][0]["function_signature"], "run()")

    def test_directed_candidate_ranks_before_weak_corridor_candidate(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            weak = _address(9)
            graph["nodes"].append({"node_id": "n4", "trace_index": 4, "callee_address": weak, "function_name": "weak", "function_selector": selector_for_signature("weak()")})
            graph["call_edges"].append({"edge_id": "call_e003", "edge_type": "CALL", "src": "n1", "dst": "n4"})
            graph["nodes"][0]["function_selector"] = selector_for_signature("run()")
            _cache(root, addresses[0])
            _cache(root, weak, source="contract C { function weak() external {} }")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual([item["function_signature"] for item in union["candidates"]], ["run()", "weak()"])
            self.assertEqual(union["candidates"][1]["call_relation_modes"], ["UNDIRECTED_CALL_CORRIDOR"])
            self.assertIn("WEAK_CALL_COMPONENT_CANDIDATE_USED", union["issue_codes"])

    def test_internal_function_expansion_is_unique_and_bounded_to_two_layers(self):
        source = """contract C {
 function entry() external { first(); }
 function first() internal { second(); }
 function second() internal { third(); }
 function third() internal {}
}"""
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][1].update(function_name="entry", function_selector=selector_for_signature("entry()"))
            _cache(root, addresses[1], source=source)
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            by_function = {item["function_signature"]: item for item in union["candidates"]}
            self.assertEqual(set(by_function), {"entry()", "first()", "second()"})
            self.assertNotIn("third()", by_function)
            self.assertEqual(by_function["second()"]["internal_expansion_depth"], MAX_INTERNAL_EXPANSION_DEPTH)
            self.assertEqual(
                by_function["first()"]["unique_parent_function_identity"],
                {
                    "contract_alias": "K1",
                    "relative_file": "contracts/C.sol",
                    "function_signature": "entry()",
                    "start_line": 2,
                    "end_line": 2,
                    "selector_match_status": "SELECTOR_EXACT_MATCH",
                    "direct_internal_callee_count": 1,
                },
            )
            self.assertEqual(by_function["second()"]["unique_parent_function_identity"]["function_signature"], "first()")
            self.assertIn("INTERNAL_FUNCTION_EXPANSION_USED", union["issue_codes"])

    def test_ambiguous_overloaded_internal_call_is_not_guessed(self):
        source = """contract C {
 function entry() external { overloaded(1); }
 function overloaded(uint256 x) internal {}
 function overloaded(address x) internal {}
}"""
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][1].update(function_name="entry", function_selector=selector_for_signature("entry()"))
            _cache(root, addresses[1], source=source)
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual([item["function_signature"] for item in union["candidates"]], ["entry()"])
            self.assertIn("INTERNAL_FUNCTION_CALL_AMBIGUOUS_EXCLUDED", union["issue_codes"])

    def test_candidate_union_cap_is_twenty_and_truncation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, _addresses = _inputs()
            graph["nodes"] = graph["nodes"][1:]
            graph["call_edges"] = []
            strict["chains"][0]["node_ids"] = ["n2", "n3"]
            for position in range(25):
                address = _address(100 + position)
                node_id = f"c{position:02d}"
                function = f"f{position:02d}"
                graph["nodes"].append({"node_id": node_id, "trace_index": position + 10, "callee_address": address, "function_name": function, "function_selector": selector_for_signature(f"{function}()")})
                graph["call_edges"].append({"edge_id": f"call_cap_{position:02d}", "edge_type": "CALL", "src": node_id, "dst": "n3"})
                _cache(root, address, source=f"contract C {{ function {function}() external {{}} }}")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            first = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            second = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual(first, second)
            self.assertEqual(first["candidate_count"], MAX_SOURCE_FUNCTION_CANDIDATES)
            self.assertEqual(first["truncated_candidate_count"], 5)
            self.assertIn("PHASE2_SOURCE_FUNCTION_CANDIDATES_TRUNCATED", first["issue_codes"])

    def test_redacted_candidate_serialization_is_stable(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            _cache(root, addresses[0])
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            serialized = serialize_phase2_source_candidate_union(union)
            self.assertEqual(serialized, serialize_phase2_source_candidate_union(union))
            self.assertRegex(union["candidates"][0]["contract_alias"], r"^K[1-9][0-9]*$")
            self.assertIn("ranking_features", union["candidates"][0])
            self.assertIn("address_match_priority", union["candidates"][0]["ranking_features"])
            self.assertNotIn(addresses[0], serialized.lower())
            self.assertNotIn("contract C", serialized)
            self.assertNotIn("source_address", serialized)
            unsafe = dict(union); unsafe["source_address"] = addresses[0]
            with self.assertRaisesRegex(ValueError, "PRIVATE_FIELD"):
                serialize_phase2_source_candidate_union(unsafe)

    def test_existing_exact_slice_no_longer_prevents_reliable_graph_union(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            existing["requests"] = [_resolved_request("n2", addresses[1], function="middle")]
            _cache(root, addresses[0])
            _cache(root, addresses[1], source="contract C { function middle() external {} }")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            with patch("src.cloc_v2.phase2_source_graph_fallback.lookup_verified_source_cache", wraps=__import__("src.cloc_v2.phase2_source_graph_fallback", fromlist=["lookup_verified_source_cache"]).lookup_verified_source_cache) as lookup:
                result = build_phase2_source_graph_fallback(graph, strict, existing, index, root)
            self.assertGreater(lookup.call_count, 0)
            self.assertEqual({item["function_signature"] for item in result["metadata"]["candidate_union"]["candidates"]}, {"middle()", "run()"})

    def test_graph_linked_unresolved_node_adds_only_public_external_functions(self):
        source = """contract C {
 function alpha() external {}
 function beta(uint256 value) public {}
 function hidden() private {}
 function helper() internal {}
 constructor() {}
 fallback() external {}
 receive() external payable {}
}"""
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][0].update(function_name="UNKNOWN", function_selector="UNKNOWN")
            _cache(root, addresses[0], source=source)
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual({row["function_signature"] for row in union["candidates"]}, {"alpha()", "beta(uint256)"})
            self.assertTrue(all(row["selector_match_status"] == "FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK" for row in union["candidates"]))
            self.assertTrue(all("GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK" in row["candidate_sources"] for row in union["candidates"]))
            self.assertIn("GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK_USED", union["issue_codes"])
            self.assertIn("FUNCTION_IDENTITY_COARSE_SOURCE_FALLBACK", union["issue_codes"])

    def test_topk_v4_resolved_identity_coexists_with_other_public_external_supplements(self):
        source = """contract C {
 function exact() external {}
 function alpha() external {}
 function beta(uint256 value) public {}
 function hidden() private {}
 function helper() internal {}
 constructor() {}
 fallback() external {}
 receive() external payable {}
}"""
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][1].update(function_name="exact", function_selector=selector_for_signature("exact()"))
            existing["requests"] = [_resolved_request("n2", addresses[1], function="exact")]
            existing["requests"][0].update(start_line=2, end_line=2)
            _cache(root, addresses[1], source=source)
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(
                graph,
                strict,
                existing,
                index,
                root,
                max_candidates=MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
            )
            signatures = [row["function_signature"] for row in union["candidates"]]
            self.assertEqual(signatures, ["exact()", "alpha()", "beta(uint256)"])
            self.assertEqual(union["candidates"][0]["selector_match_status"], "SELECTOR_EXACT_MATCH")
            self.assertEqual(signatures.count("exact()"), 1)
            self.assertIn("DIRECT_STRICT_CHAIN_SOURCE", union["candidates"][0]["candidate_sources"])
            self.assertIn("COARSE_SOURCE_FUNCTION_SUPPLEMENT", union["candidates"][0]["candidate_sources"])
            self.assertIn("GRAPH_LINKED_SOURCE_FALLBACK_COEXISTS_WITH_RESOLVED_IDENTITY", union["issue_codes"])
            self.assertIn("COARSE_SOURCE_FUNCTION_SUPPLEMENT_USED", union["issue_codes"])
            self.assertNotIn("hidden()", signatures)
            self.assertNotIn("helper()", signatures)
            self.assertTrue(all(value not in signatures for value in ("constructor", "fallback", "receive")))

    def test_legacy_candidate_limit_keeps_resolved_node_fallback_suppressed(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][1].update(function_name="exact", function_selector=selector_for_signature("exact()"))
            existing["requests"] = [_resolved_request("n2", addresses[1], function="exact")]
            _cache(root, addresses[1], source="contract C { function exact() external {} function alpha() external {} }")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual([row["function_signature"] for row in union["candidates"]], ["exact()"])
            self.assertNotIn("GRAPH_LINKED_SOURCE_FALLBACK_COEXISTS_WITH_RESOLVED_IDENTITY", union["issue_codes"])

    def test_topk_v4_coexisting_supplements_report_deterministic_truncation(self):
        supplements = "\n".join(f"function f{index:03d}() external {{}}" for index in range(105))
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][1].update(function_name="exact", function_selector=selector_for_signature("exact()"))
            existing["requests"] = [_resolved_request("n2", addresses[1], function="exact")]
            _cache(root, addresses[1], source=f"contract C {{ function exact() external {{}} {supplements} }}")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(
                graph,
                strict,
                existing,
                index,
                root,
                max_candidates=MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
            )
            self.assertEqual(union["candidate_count_before_truncation"], 106)
            self.assertEqual(union["candidate_count_after_truncation"], 100)
            self.assertEqual(union["coarse_source_function_supplement_truncated_count"], 6)
            self.assertIn("COARSE_SOURCE_FUNCTION_SUPPLEMENT_TRUNCATED", union["issue_codes"])
            self.assertEqual(union, build_phase2_source_candidate_union(graph, strict, existing, index, root, max_candidates=MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4))

    def test_unlinked_verified_source_never_adds_public_function_fallback(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs(path=False)
            graph["nodes"][0].update(function_name="UNKNOWN", function_selector="UNKNOWN")
            _cache(root, addresses[0], source="contract C { function alpha() external {} }")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual(union["candidate_count"], 0)

    def test_exact_and_selector_candidates_rank_before_coarse_function_fallback(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][0].update(function_name="UNKNOWN", function_selector="UNKNOWN")
            graph["nodes"][1].update(function_name="exact", function_selector=selector_for_signature("exact()"))
            existing["requests"] = [_resolved_request("n2", addresses[1], function="exact")]
            _cache(root, addresses[0], source="contract C { function coarse() external {} }")
            _cache(root, addresses[1], source="contract C { function exact() external {} }")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual([row["function_signature"] for row in union["candidates"]], ["exact()", "coarse()"])
            self.assertEqual(union["candidates"][0]["selector_match_status"], "SELECTOR_EXACT_MATCH")

    def test_coarse_fallback_preserves_overloads_by_full_signature(self):
        source = """contract C {
 function overloaded(uint256 value) external {}
 function overloaded(address value) external {}
}"""
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][0].update(function_name="UNKNOWN", function_selector="UNKNOWN")
            _cache(root, addresses[0], source=source)
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(graph, strict, existing, index, root)
            self.assertEqual([row["function_signature"] for row in union["candidates"]], ["overloaded(address)", "overloaded(uint256)"])

    def test_topk_v4_candidate_cap_is_hundred_and_fallback_truncation_is_audited(self):
        functions = "\n".join(f"function f{index:03d}() external {{}}" for index in range(105))
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            graph, strict, existing, addresses = _inputs()
            graph["nodes"][0].update(function_name="UNKNOWN", function_selector="UNKNOWN")
            _cache(root, addresses[0], source=f"contract C {{\n{functions}\n}}")
            index = build_verified_source_cache_index(root, blockchain_chain_id=1, chain_key="eth")
            union = build_phase2_source_candidate_union(
                graph,
                strict,
                existing,
                index,
                root,
                max_candidates=MAX_SOURCE_FUNCTION_CANDIDATES_TOPK_V4,
            )
            self.assertEqual(union["candidate_count"], 100)
            self.assertEqual(union["candidate_count_before_truncation"], 105)
            self.assertEqual(union["candidate_count_after_truncation"], 100)
            self.assertTrue(union["candidate_limit_reached"])
            self.assertTrue(union["candidate_limit_exceeded"])
            self.assertEqual(union["truncated_candidate_count"], 5)
            self.assertEqual(union["fallback_candidate_count_before_truncation"], 105)
            self.assertEqual(union["fallback_truncated_candidate_count"], 5)
            self.assertIn("GRAPH_LINKED_FUNCTION_FALLBACK_TRUNCATED", union["issue_codes"])
            self.assertIn("PHASE2_V4_SOURCE_FUNCTION_CANDIDATES_TRUNCATED_AT_100", union["issue_codes"])
            self.assertIn("f087()", [row["function_signature"] for row in union["candidates"]])
            self.assertNotIn("f100()", [row["function_signature"] for row in union["candidates"]])
            self.assertEqual(union["candidate_source_counts_before_truncation"]["GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK"], 105)
            self.assertEqual(union["candidate_source_counts_after_truncation"]["GRAPH_LINKED_VERIFIED_SOURCE_FUNCTION_FALLBACK"], 100)
            self.assertEqual([row["function_signature"] for row in union["candidates"][:3]], ["f000()", "f001()", "f002()"])


if __name__ == "__main__":
    unittest.main()

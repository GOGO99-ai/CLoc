from __future__ import annotations

import copy
import unittest
from pathlib import Path

from src.cloc_v2.causal_chain_assembly import (
    build_candidate_causal_chains,
    load_json_document,
    normalized_sha256,
    serialize_candidate_causal_chains,
)
from src.cloc_v2.causal_chain_models import ChainAssemblyConfig


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data" / "runs" / "eth" / "0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299"


def _edge(edge_id, edge_type, src, dst, level="STRONG_ASSOCIATION", attributes=None):
    return {
        "edge_id": edge_id,
        "edge_type": edge_type,
        "src": src,
        "dst": dst,
        "evidence_level": level,
        "reason": "test",
        "source_stage": "test",
        "source_refs": [edge_id],
        "attributes": attributes or {},
    }


def fixture_graph():
    nodes = [
        {"node_id": "n001", "trace_index": 1, "trace_path": "0", "parent_id": None},
        {"node_id": "n006", "trace_index": 2, "trace_path": "0/0", "parent_id": "n001"},
        {"node_id": "n005", "trace_index": 3, "trace_path": "0/0/0", "parent_id": "n006"},
        {"node_id": "n007", "trace_index": 4, "trace_path": "0/1", "parent_id": "n001"},
        {"node_id": "n002", "trace_index": 5, "trace_path": "0/2", "parent_id": "n001"},
        {"node_id": "n008", "trace_index": 6, "trace_path": "0/3", "parent_id": "n001"},
        {"node_id": "n009", "trace_index": 7, "trace_path": "0/4", "parent_id": "n001"},
        {"node_id": "n003", "trace_index": 10, "trace_path": "0/2/0", "parent_id": "n002"},
        {"node_id": "n004", "trace_index": 20, "trace_path": "0/2/0/0", "parent_id": "n003"},
    ]
    state_object = {
        "state_object_type": "CONTRACT_STATE",
        "contract_address": {"normalized": "0x" + "a" * 40},
        "account_address": None,
    }
    match = [{"source_edge_id": "raw", "value_type": "UINT", "normalized_value": "7"}]
    edges = [
        _edge("call_main_1", "CALL", "n001", "n002", "OBSERVED"),
        _edge("param_main", "PARAM", "n001", "n002", attributes={"match_count": 1, "matches": match}),
        _edge("call_main_2", "CALL", "n002", "n003", "OBSERVED"),
        _edge("state_main", "STATE", "n002", "n003", "HYPOTHESIS", {"state_object": state_object}),
        _edge("result", "RESULT", "n003", "n004", attributes={"result_id": "result_r001"}),
        _edge("call_ancestor", "CALL", "n006", "n005", "OBSERVED"),
        _edge("asset_branch", "ASSET", "n005", "n003"),
        _edge("state_branch", "STATE", "n007", "n002", "HYPOTHESIS", {"state_object": state_object}),
        _edge("param_branch", "PARAM", "n008", "n003", "OBSERVED", {"match_count": 1, "matches": match}),
        _edge("pure_call", "CALL", "n009", "n003", "OBSERVED"),
    ]
    edges[4]["source_refs"] = ["result_r001"]
    return {
        "schema_version": "candidate_causal_graph_v2",
        "candidate_causal_graph_v1_sha256": "a" * 64,
        "parameter_evidence_sha256": "b" * 64,
        "state_evidence_sha256": "c" * 64,
        "upstream_bindings": {"transaction_graph_sha256": "d" * 64},
        "nodes": nodes,
        "edges": edges,
        "result_seeds": [{"result_id": "result_r001", "node_id": "n004"}],
        "root_results_without_edge": [],
    }


def fixture_backbones(graph):
    return {
        "schema_version": "causal_backbones_v1",
        "candidate_causal_graph_v2_sha256": normalized_sha256(graph),
        "backbones": [
            {
                "backbone_id": "backbone_b001",
                "result_id": "result_r001",
                "result_node_id": "n004",
                "semantic_views": ["CONTROL_CONTEXT"],
                "node_ids": ["n001", "n002", "n003", "n004"],
                "edge_ids": ["call_main_1", "call_main_2", "result"],
                "stop_reason": "ROOT_REACHED",
            },
            {
                "backbone_id": "backbone_b002",
                "result_id": "result_r001",
                "result_node_id": "n004",
                "semantic_views": ["ASSET_FLOW", "STATE_DEPENDENCY", "PARAM_PROPAGATION"],
                "node_ids": ["n001", "n002", "n003", "n004"],
                "edge_ids": ["param_main", "state_main", "result"],
                "stop_reason": "ROOT_REACHED",
            },
        ],
    }


class CandidateCausalChainTests(unittest.TestCase):
    def test_merges_identical_backbones_canonicalizes_and_extracts_branches(self):
        graph = fixture_graph()
        result = build_candidate_causal_chains(graph, fixture_backbones(graph))
        self.assertEqual(1, len(result.chains))
        chain = result.chains[0]
        self.assertEqual(
            ("ASSET_FLOW", "STATE_DEPENDENCY", "PARAM_PROPAGATION", "CONTROL_CONTEXT"),
            chain.semantic_views,
        )
        self.assertEqual(("param_main", "state_main", "result"), chain.backbone_edge_ids)
        self.assertEqual(("call_main_1", "call_main_2"), chain.supporting_edge_ids)
        self.assertEqual({"ASSET", "STATE", "PARAM"}, {branch.semantic_edge_type for branch in chain.branches})
        asset = next(branch for branch in chain.branches if branch.semantic_edge_type == "ASSET")
        self.assertEqual(("n006", "n005", "n003"), asset.node_ids)
        self.assertEqual(("call_ancestor", "asset_branch"), asset.edge_ids)
        self.assertNotIn("pure_call", chain.all_edge_ids)
        self.assertEqual(1, result.assembly_audit["merged_backbone_count"])

    def test_limits_branch_sharing_and_determinism(self):
        graph = fixture_graph()
        state_object = graph["edges"][3]["attributes"]["state_object"]
        graph["nodes"].append({"node_id": "n010", "trace_index": 8, "trace_path": "0/5", "parent_id": "n001"})
        graph["edges"].append(_edge("state_branch_2", "STATE", "n010", "n003", "HYPOTHESIS", {"state_object": state_object}))
        document = fixture_backbones(graph)
        before_graph, before_document = copy.deepcopy(graph), copy.deepcopy(document)
        config = ChainAssemblyConfig(
            max_branches_per_chain=2,
            max_branch_depth=1,
            max_total_nodes_per_chain=6,
            max_hypothesis_branches=0,
            max_edges_per_state_object_per_chain=1,
        )
        first = build_candidate_causal_chains(graph, document, config)
        second = build_candidate_causal_chains(graph, document, config)
        self.assertEqual(serialize_candidate_causal_chains(first), serialize_candidate_causal_chains(second))
        self.assertEqual(before_graph, graph)
        self.assertEqual(before_document, document)
        chain = first.chains[0]
        self.assertLessEqual(len(chain.branches), 2)
        self.assertLessEqual(len(chain.all_node_ids), 6)
        self.assertFalse(any(branch.evidence_level == "HYPOTHESIS" for branch in chain.branches))
        self.assertTrue(all(len(branch.edge_ids) == 1 for branch in chain.branches))
        main_nodes = set(chain.backbone_node_ids)
        external_nodes = set()
        for branch in chain.branches:
            self.assertEqual(branch.attachment_node_id, branch.node_ids[-1])
            self.assertFalse(set(branch.node_ids[:-1]) & main_nodes)
            self.assertFalse(set(branch.node_ids[:-1]) & external_nodes)
            external_nodes.update(branch.node_ids[:-1])
        state_limited = build_candidate_causal_chains(
            graph,
            document,
            ChainAssemblyConfig(max_edges_per_state_object_per_chain=1),
        )
        self.assertEqual(
            1,
            sum(
                branch.semantic_edge_type == "STATE"
                for branch in state_limited.chains[0].branches
            ),
        )
        self.assertGreater(state_limited.assembly_audit["branch_rejected_state_object_limit"], 0)

    def test_rejects_bad_hash_and_reports_no_branch_reason(self):
        graph = fixture_graph()
        no_branch_graph = copy.deepcopy(graph)
        no_branch_graph["edges"] = no_branch_graph["edges"][:5]
        document = fixture_backbones(no_branch_graph)
        result = build_candidate_causal_chains(no_branch_graph, document)
        self.assertEqual((), result.chains[0].branches)
        self.assertEqual(
            "NO_EXTERNAL_SEMANTIC_PREDECESSOR",
            result.chains[0].structural_features["no_branch_reason"],
        )
        bad = copy.deepcopy(document)
        bad["candidate_causal_graph_v2_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "graph hash"):
            build_candidate_causal_chains(no_branch_graph, bad)

    def test_real_inputs_are_compatible_and_structural_invariants_hold(self):
        if not RUN.exists():
            self.skipTest("real run is unavailable")
        graph = load_json_document(RUN / "candidate_causal_graph_v2.json", "candidate_causal_graph_v2")
        backbones = load_json_document(RUN / "causal_backbones_v1.json", "causal_backbones_v1")
        result = build_candidate_causal_chains(graph, backbones)
        self.assertEqual(36, result.assembly_audit["input_backbone_count"])
        self.assertEqual(19, len(result.chains))
        self.assertEqual([], result.result_coverage["missing_result_ids"])
        edge_map = {edge["edge_id"]: edge for edge in graph["edges"]}
        for chain in result.chains:
            self.assertEqual(len(chain.all_node_ids), len(set(chain.all_node_ids)))
            self.assertEqual(len(chain.all_edge_ids), len(set(chain.all_edge_ids)))
            self.assertEqual("RESULT", edge_map[chain.backbone_edge_ids[-1]]["edge_type"])
            self.assertFalse(any(edge_map[edge_id]["edge_type"] == "RESULT" for edge_id in chain.backbone_edge_ids[:-1]))
            self.assertLessEqual(chain.structural_features["branch_count"], 4)
            self.assertLessEqual(chain.structural_features["total_node_count"], 32)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations
import copy,unittest
from pathlib import Path
from src.cloc_v2.asset_continuity import load_observed_asset_ledger
from src.cloc_v2.observed_asset_ledger import load_runtime_annotations
from src.cloc_v2.parameter_evidence import build_parameter_evidence,load_candidate_causal_graph,serialize_parameter_evidence
from src.cloc_v2.runtime_semantics import load_transaction_graph
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
class ParameterEvidenceTests(unittest.TestCase):
 def setUp(self):
  if not RUN.exists():self.skipTest('artifacts unavailable')
  self.g=load_transaction_graph(RUN/'transaction_graph_v1.json');self.a=load_runtime_annotations(RUN/'runtime_annotations_v1.json');self.l=load_observed_asset_ledger(RUN/'observed_asset_ledger_v1.json');self.c=load_candidate_causal_graph(RUN/'candidate_causal_graph_v1.json')
 def test_invariants_determinism_and_immutability(self):
  before=copy.deepcopy(self.c);x=build_parameter_evidence(self.g,self.a,self.l,self.c);y=build_parameter_evidence(self.g,self.a,self.l,self.c)
  self.assertEqual(serialize_parameter_evidence(x),serialize_parameter_evidence(y));self.assertEqual(before,self.c)
  nodes={n.node_id:n for n in self.g.nodes};self.assertTrue(all(e.src!=e.dst and nodes[e.src].trace_index<nodes[e.dst].trace_index for e in x.parameter_edges));self.assertTrue(all(e.edge_id==f'param_e{i:03d}' for i,e in enumerate(x.parameter_edges,1)));self.assertFalse(any(e.evidence_level=='STRONG_ASSOCIATION' and e.relation_type=='SIBLING_VALUE_REUSE' for e in x.parameter_edges))
 def test_hash_mismatch_fails(self):
  bad=copy.deepcopy(self.c);bad['transaction_graph_sha256']='bad'
  with self.assertRaisesRegex(ValueError,'hash binding'):build_parameter_evidence(self.g,self.a,self.l,bad)
if __name__=='__main__':unittest.main()

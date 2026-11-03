from __future__ import annotations
import copy,unittest
from pathlib import Path
from src.cloc_v2.asset_continuity import load_observed_asset_ledger
from src.cloc_v2.observed_asset_ledger import load_runtime_annotations
from src.cloc_v2.runtime_semantics import load_transaction_graph
from src.cloc_v2.state_evidence import build_state_evidence,load_candidate_causal_graph,serialize_state_evidence
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
class StateEvidenceTests(unittest.TestCase):
 def setUp(self):
  if not RUN.exists():self.skipTest('artifacts unavailable')
  self.g=load_transaction_graph(RUN/'transaction_graph_v1.json');self.a=load_runtime_annotations(RUN/'runtime_annotations_v1.json');self.l=load_observed_asset_ledger(RUN/'observed_asset_ledger_v1.json');self.c=load_candidate_causal_graph(RUN/'candidate_causal_graph_v1.json')
 def test_invariants_objects_levels_and_determinism(self):
  before=copy.deepcopy(self.c);x=build_state_evidence(self.g,self.a,self.l,self.c);y=build_state_evidence(self.g,self.a,self.l,self.c);nodes={n.node_id:n for n in self.g.nodes}
  self.assertEqual(serialize_state_evidence(x),serialize_state_evidence(y));self.assertEqual(before,self.c);self.assertTrue(all(e.src!=e.dst and nodes[e.src].trace_index<nodes[e.dst].trace_index for e in x.state_edges));self.assertTrue(all(e.evidence_level!='OBSERVED' for e in x.state_edges));self.assertTrue(all(e.evidence_level=='HYPOTHESIS' for e in x.state_edges if e.relation_type.startswith('SAME_CONTRACT')));self.assertFalse(any(e.to_dict()['edge_type'] in {'PARAM','RESULT'} for e in x.state_edges))
  for e in x.state_edges:
   if e.state_object.state_object_type=='TOKEN_BALANCE':self.assertIsNotNone(e.state_object.account_address)
 def test_specific_object_consistency_and_stable_ids(self):
  x=build_state_evidence(self.g,self.a,self.l,self.c)
  self.assertTrue(all(e.edge_id==f'state_e{i:03d}' for i,e in enumerate(x.state_edges,1)))
  for e in x.state_edges:
   if e.relation_type=='SWAP_TO_RESERVE_READ':self.assertEqual('POOL_RESERVES',e.state_object.state_object_type)
   if e.relation_type in {'TRANSFER_TO_BALANCE_READ','BALANCE_READ_TO_TRANSFER'}:self.assertEqual('TOKEN_BALANCE',e.state_object.state_object_type)
 def test_hash_mismatch_fails(self):
  bad=copy.deepcopy(self.c);bad['runtime_annotations_sha256']='bad'
  with self.assertRaisesRegex(ValueError,'hash binding'):build_state_evidence(self.g,self.a,self.l,bad)
if __name__=='__main__':unittest.main()

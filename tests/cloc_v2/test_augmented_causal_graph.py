from __future__ import annotations
import copy,unittest
from pathlib import Path
from src.cloc_v2.augmented_causal_graph import build_augmented_candidate_causal_graph,load_json_artifact,serialize_augmented_graph
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
class AugmentedCausalGraphTests(unittest.TestCase):
 def setUp(self):
  if not RUN.exists():self.skipTest('artifacts unavailable')
  self.base=load_json_artifact(RUN/'candidate_causal_graph_v1.json','candidate_causal_graph_v1');self.param=load_json_artifact(RUN/'parameter_evidence_v1.json','parameter_evidence_v1');self.state=load_json_artifact(RUN/'state_evidence_v1.json','state_evidence_v1')
 def test_lossless_base_and_complete_evidence_aggregation(self):
  g=build_augmented_candidate_causal_graph(self.base,self.param,self.state);d=g.to_dict()
  self.assertEqual(d['nodes'],self.base['nodes']);self.assertEqual(d['result_seeds'],self.base['result_seeds']);self.assertEqual(d['root_results_without_edge'],self.base['root_results_without_edge'])
  for kind in ('CALL','ORDER','ASSET','RESULT'):
   self.assertEqual([e for e in d['edges'] if e['edge_type']==kind],[e for e in self.base['edges'] if e['edge_type']==kind])
  bundles=[e for e in d['edges'] if e['edge_type']=='PARAM'];refs=[r for e in bundles for r in e['source_refs']];self.assertEqual(sorted(refs),sorted(e['edge_id'] for e in self.param['parameter_edges']));self.assertEqual(len(refs),len(set(refs)))
  self.assertTrue(all(e['attributes']['match_count']==len(e['attributes']['matches']) and 0<e['attributes']['match_count']<=50 for e in bundles))
  state_refs=[r for e in d['edges'] if e['edge_type']=='STATE' for r in e['source_refs']];self.assertEqual(sorted(state_refs),sorted(e['edge_id'] for e in self.state['state_edges']))
  self.assertEqual({k:'SUPPORTED' for k in ('CALL','ORDER','ASSET','PARAM','STATE','RESULT')},d['edge_type_support'])
 def test_determinism_input_immutability_and_graph_invariants(self):
  before=(copy.deepcopy(self.base),copy.deepcopy(self.param),copy.deepcopy(self.state));a=build_augmented_candidate_causal_graph(self.base,self.param,self.state);b=build_augmented_candidate_causal_graph(self.base,self.param,self.state);d=a.to_dict();nodes={n['node_id']:n for n in d['nodes']}
  self.assertEqual(serialize_augmented_graph(a),serialize_augmented_graph(b));self.assertEqual(before,(self.base,self.param,self.state));self.assertTrue(all(nodes[e['src']]['trace_index']<nodes[e['dst']]['trace_index'] for e in d['edges']));self.assertTrue(d['metadata']['dag_valid']);self.assertEqual(0,d['metadata']['dangling_edge_count']);self.assertLessEqual(d['metadata']['total_graph_edge_count'],15*d['metadata']['node_count'])
  pairs=lambda kind:{(e['src'],e['dst']) for e in d['edges'] if e['edge_type']==kind};self.assertTrue(pairs('CALL')&pairs('RESULT'));self.assertTrue(pairs('CALL')&pairs('PARAM'))
  variant=copy.deepcopy(self.state);call_pair=next(iter(pairs('CALL')));variant['state_edges'][0]['src'],variant['state_edges'][0]['dst']=call_pair
  augmented=build_augmented_candidate_causal_graph(self.base,self.param,variant).to_dict();call_pairs={(e['src'],e['dst']) for e in augmented['edges'] if e['edge_type']=='CALL'};state_pairs={(e['src'],e['dst']) for e in augmented['edges'] if e['edge_type']=='STATE'};self.assertIn(call_pair,call_pairs&state_pairs)
 def test_invalid_hash_relation_level_and_direction_fail(self):
  bad=copy.deepcopy(self.param);bad['candidate_causal_graph_sha256']='bad'
  with self.assertRaisesRegex(ValueError,'not bound'):build_augmented_candidate_causal_graph(self.base,bad,self.state)
  bad=copy.deepcopy(self.param);bad['parameter_edges'][0]['relation_type']='UNKNOWN'
  with self.assertRaisesRegex(ValueError,'Unknown PARAM'):build_augmented_candidate_causal_graph(self.base,bad,self.state)
  bad=copy.deepcopy(self.state);bad['state_edges'][0]['evidence_level']='OBSERVED'
  with self.assertRaisesRegex(ValueError,'Invalid evidence_level'):build_augmented_candidate_causal_graph(self.base,self.param,bad)
  bad=copy.deepcopy(self.param);bad['parameter_edges'][0]['src'],bad['parameter_edges'][0]['dst']=bad['parameter_edges'][0]['dst'],bad['parameter_edges'][0]['src']
  with self.assertRaisesRegex(ValueError,'Reverse-time'):build_augmented_candidate_causal_graph(self.base,bad,self.state)
 def test_nonhistorical_evidence_counts_are_accepted_when_metadata_matches(self):
  param=copy.deepcopy(self.param);removed=next(e for e in param['parameter_edges'] if e['relation_type']=='PARENT_CHILD_FORWARD')
  param['parameter_edges'].remove(removed);param['metadata']['parent_child_edge_count']-=1;param['metadata']['strong_association_count']-=1
  state=copy.deepcopy(self.state);removed=next(e for e in state['state_edges'] if e['relation_type']=='TRANSFER_TO_BALANCE_READ')
  state['state_edges'].remove(removed);state['metadata']['transfer_to_balance_read_count']-=1;state['metadata']['strong_association_count']-=1
  result=build_augmented_candidate_causal_graph(self.base,param,state).to_dict()
  self.assertEqual(len(self.param['parameter_edges'])-1,result['metadata']['raw_param_edge_count'])
  self.assertEqual(len(self.state['state_edges'])-1,result['metadata']['state_edge_count'])
 def test_evidence_metadata_count_drift_fails(self):
  bad=copy.deepcopy(self.param);bad['parameter_edges'].pop()
  with self.assertRaisesRegex(ValueError,'Raw PARAM metadata count mismatch'):build_augmented_candidate_causal_graph(self.base,bad,self.state)
if __name__=='__main__':unittest.main()

from __future__ import annotations
import copy,unittest
from pathlib import Path
from src.cloc_v2.backbone_models import BackboneSearchConfig
from src.cloc_v2.causal_backbones import build_causal_backbones,load_candidate_graph_v2,serialize_causal_backbones
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
def synthetic():
 nodes=[{'node_id':f'n{i:03d}','trace_index':i,'trace_path':str(i),'parent_id':None if i==1 else f'n{i-1:03d}'} for i in range(1,4)]
 def e(i,t,src,dst,level='STRONG_ASSOCIATION',attrs=None):return {'edge_id':i,'edge_type':t,'src':src,'dst':dst,'evidence_level':level,'reason':'x','source_stage':'test','source_refs':[i], 'attributes':attrs or {}}
 matches=[{'source_edge_id':'param_raw_1','value_type':'UINT','normalized_value':'2'}]
 edges=[e('call1','CALL','n001','n002','OBSERVED'),e('asset1','ASSET','n001','n002'),e('param1','PARAM','n001','n002',attrs={'relation_type':'PARENT_CHILD_FORWARD','match_count':1,'matches':matches}),e('state1','STATE','n001','n002',attrs={'relation_type':'TRANSFER_TO_BALANCE_READ','state_object':{'state_object_type':'TOKEN_BALANCE','contract_address':{'normalized':'0x'+'a'*40},'account_address':{'normalized':'0x'+'b'*40}},'execution_distance':1}),e('result1','RESULT','n002','n003',attrs={'result_id':'result_r001'})]
 edges[-1]['source_refs']=['result_r001']
 return {'schema_version':'candidate_causal_graph_v2','candidate_causal_graph_v1_sha256':'a'*64,'parameter_evidence_sha256':'b'*64,'state_evidence_sha256':'c'*64,'upstream_bindings':{'x':'d'*64},'nodes':nodes,'edges':edges,'result_seeds':[{'result_id':'result_r001','node_id':'n003'}],'root_results_without_edge':[]}
def chain_graph():
 nodes=[{'node_id':f'n{i:03d}','trace_index':i,'trace_path':str(i),'parent_id':None if i==1 else f'n{i-1:03d}'} for i in range(1,7)]
 state={'state_object_type':'TOKEN_BALANCE','contract_address':{'normalized':'0x'+'a'*40},'account_address':{'normalized':'0x'+'b'*40}}
 match=[{'source_edge_id':'raw_param','value_type':'UINT','normalized_value':'2'}]
 edges=[
  {'edge_id':'call1','edge_type':'CALL','src':'n001','dst':'n002','evidence_level':'OBSERVED','reason':'x','source_stage':'test','source_refs':['call1'],'attributes':{}},
  {'edge_id':'asset1','edge_type':'ASSET','src':'n002','dst':'n003','evidence_level':'STRONG_ASSOCIATION','reason':'x','source_stage':'test','source_refs':['asset1'],'attributes':{}},
  {'edge_id':'state1','edge_type':'STATE','src':'n003','dst':'n004','evidence_level':'HYPOTHESIS','reason':'x','source_stage':'test','source_refs':['state1'],'attributes':{'state_object':state}},
  {'edge_id':'param1','edge_type':'PARAM','src':'n004','dst':'n005','evidence_level':'HYPOTHESIS','reason':'x','source_stage':'test','source_refs':['raw_param'],'attributes':{'match_count':1,'matches':match}},
  {'edge_id':'result1','edge_type':'RESULT','src':'n005','dst':'n006','evidence_level':'STRONG_ASSOCIATION','reason':'x','source_stage':'test','source_refs':['result_r001'],'attributes':{'result_id':'result_r001'}},
 ]
 return {'schema_version':'candidate_causal_graph_v2','candidate_causal_graph_v1_sha256':'a'*64,'parameter_evidence_sha256':'b'*64,'state_evidence_sha256':'c'*64,'upstream_bindings':{'x':'d'*64},'nodes':nodes,'edges':edges,'result_seeds':[{'result_id':'result_r001','node_id':'n006'}],'root_results_without_edge':[]}
def limits_graph():
 graph=chain_graph();graph['nodes']=graph['nodes'][:4];graph['result_seeds'][0]['node_id']='n004';state={'state_object_type':'CONTRACT_STATE','contract_address':{'normalized':'0x'+'c'*40},'account_address':None};match=[{'source_edge_id':'raw','value_type':'UINT','normalized_value':'3'}];edges=[]
 for src,dst,suffix in [('n001','n002','1'),('n002','n003','2')]:
  edges.extend([
   {'edge_id':'call'+suffix,'edge_type':'CALL','src':src,'dst':dst,'evidence_level':'OBSERVED','reason':'x','source_stage':'test','source_refs':['call'+suffix],'attributes':{}},
   {'edge_id':'order'+suffix,'edge_type':'ORDER','src':src,'dst':dst,'evidence_level':'OBSERVED','reason':'x','source_stage':'test','source_refs':['order'+suffix],'attributes':{}},
   {'edge_id':'asset'+suffix,'edge_type':'ASSET','src':src,'dst':dst,'evidence_level':'STRONG_ASSOCIATION','reason':'x','source_stage':'test','source_refs':['asset'+suffix],'attributes':{}},
   {'edge_id':'param'+suffix,'edge_type':'PARAM','src':src,'dst':dst,'evidence_level':'HYPOTHESIS','reason':'x','source_stage':'test','source_refs':['raw'],'attributes':{'match_count':1,'matches':match}},
   {'edge_id':'state'+suffix,'edge_type':'STATE','src':src,'dst':dst,'evidence_level':'HYPOTHESIS','reason':'x','source_stage':'test','source_refs':['state'+suffix],'attributes':{'state_object':state}},
  ])
 edges.append({'edge_id':'result1','edge_type':'RESULT','src':'n003','dst':'n004','evidence_level':'STRONG_ASSOCIATION','reason':'x','source_stage':'test','source_refs':['result_r001'],'attributes':{'result_id':'result_r001'}});graph['edges']=edges;return graph
class CausalBackboneTests(unittest.TestCase):
 def test_views_anchor_forward_order_and_param_bundle_single_adjacency(self):
  graph=synthetic();result=build_causal_backbones(graph);by_view={v:b for b in result.backbones for v in b.semantic_views}
  self.assertEqual({'ASSET_FLOW','STATE_DEPENDENCY','PARAM_PROPAGATION','CONTROL_CONTEXT'},set(by_view))
  required={'ASSET_FLOW':'ASSET','STATE_DEPENDENCY':'STATE','PARAM_PROPAGATION':'PARAM','CONTROL_CONTEXT':'CALL'}
  edge_map={e['edge_id']:e for e in graph['edges']}
  for view,kind in required.items():
   b=by_view[view];self.assertEqual('RESULT',edge_map[b.edge_ids[-1]]['edge_type']);self.assertIn(kind,[edge_map[x]['edge_type'] for x in b.edge_ids]);self.assertEqual(('n001','n002','n003'),b.node_ids);self.assertEqual(len(b.edge_ids),2)
 def test_limits_input_immutability_and_byte_determinism(self):
  graph=synthetic();graph['nodes'][0]['parent_id']='n000';graph['nodes'].insert(0,{'node_id':'n000','trace_index':0,'trace_path':'0','parent_id':None});graph['edges'].insert(0,{'edge_id':'call0','edge_type':'CALL','src':'n000','dst':'n001','evidence_level':'OBSERVED','reason':'x','source_stage':'test','source_refs':['call0'],'attributes':{}});before=copy.deepcopy(graph);cfg=BackboneSearchConfig(max_depth=2);a=build_causal_backbones(graph,cfg);b=build_causal_backbones(graph,cfg);self.assertEqual(serialize_causal_backbones(a),serialize_causal_backbones(b));self.assertEqual(before,graph);self.assertTrue(all(x.stop_reason=='MAX_DEPTH_REACHED' for x in a.backbones));self.assertTrue(any('CONTROL_CONTEXT' in x.semantic_views for x in a.backbones))
 def test_invalid_result_param_state_and_direction_fail(self):
  graph=synthetic();bad=copy.deepcopy(graph);bad['edges'][-1]['source_refs']=['other']
  with self.assertRaisesRegex(ValueError,'exactly one'):build_causal_backbones(bad)
  bad=copy.deepcopy(graph);bad['edges'][2]['attributes']['match_count']=2
  with self.assertRaisesRegex(ValueError,'PARAM bundle'):build_causal_backbones(bad)
  bad=copy.deepcopy(graph);bad['edges'][3]['attributes']['state_object']={}
  with self.assertRaisesRegex(ValueError,'STATE object'):build_causal_backbones(bad)
  bad=copy.deepcopy(graph);bad['edges'][0]['src'],bad['edges'][0]['dst']=bad['edges'][0]['dst'],bad['edges'][0]['src']
  with self.assertRaisesRegex(ValueError,'Reverse-time'):build_causal_backbones(bad)
  bad=copy.deepcopy(graph);bad['parameter_evidence_sha256']='not-a-hash'
  with self.assertRaisesRegex(ValueError,'upstream SHA-256'):build_causal_backbones(bad)
 def test_root_seed_limits_and_complete_pruning_counters(self):
  root=synthetic();root['edges']=root['edges'][:-1];root['result_seeds'][0]['node_id']='n001';result=build_causal_backbones(root)
  self.assertEqual(1,len(result.backbones));self.assertEqual(('CONTROL_CONTEXT',),result.backbones[0].semantic_views);self.assertEqual((),result.backbones[0].edge_ids);self.assertIn('ROOT_RESULT_SEED',result.backbones[0].selection_basis)
  limited=build_causal_backbones(limits_graph(),BackboneSearchConfig(max_order_edges=1,max_hypothesis_edges=1,max_state_edges_per_object=3))
  for reason in ('ORDER_LIMIT','HYPOTHESIS_LIMIT'):self.assertGreater(limited.pruning_counts[reason],0)
  state_limited=build_causal_backbones(limits_graph(),BackboneSearchConfig(max_state_edges_per_object=1));self.assertGreater(state_limited.pruning_counts['STATE_OBJECT_LIMIT'],0)
  branched=build_causal_backbones(limits_graph(),BackboneSearchConfig(max_branching_per_node=2));self.assertGreater(branched.pruning_counts['BRANCHING_LIMIT'],0)
  budgeted=build_causal_backbones(limits_graph(),BackboneSearchConfig(max_expanded_states_per_result_per_view=1));self.assertGreater(budgeted.pruning_counts['EXPANSION_BUDGET_REACHED'],0);self.assertTrue(all(b.stop_reason=='EXPANSION_BUDGET_REACHED' for b in budgeted.backbones))
  self.assertEqual(set(result.pruning_counts),{'CYCLE_OR_REPEATED_NODE','BRANCHING_LIMIT','BEAM_WIDTH_LIMIT','ORDER_LIMIT','HYPOTHESIS_LIMIT','STATE_OBJECT_LIMIT','OTHER_RESULT_EDGE','DUPLICATE_PATH','DIVERSITY_FILTER','EXPANSION_BUDGET_REACHED','CONTROL_CALL_FALLBACK'})
 def test_cross_view_dedup_and_diversity_rule(self):
  result=build_causal_backbones(chain_graph());self.assertEqual(1,len(result.backbones));self.assertEqual(('ASSET_FLOW','STATE_DEPENDENCY','PARAM_PROPAGATION','CONTROL_CONTEXT'),result.backbones[0].semantic_views)
 def test_real_graph_all_results_have_control_and_paths_are_valid(self):
  if not RUN.exists():self.skipTest('real graph unavailable')
  graph=load_candidate_graph_v2(RUN/'candidate_causal_graph_v2.json');result=build_causal_backbones(graph);edge_map={e['edge_id']:e for e in graph['edges']};by_result={s['result_id'] for s in graph['result_seeds']}
  self.assertEqual(by_result,{b.result_id for b in result.backbones});self.assertEqual(by_result,{b.result_id for b in result.backbones if 'CONTROL_CONTEXT' in b.semantic_views})
  for b in result.backbones:
   self.assertEqual(len(set(b.node_ids)),len(b.node_ids));self.assertEqual(len(b.edge_ids)+1,len(b.node_ids));self.assertEqual('RESULT',edge_map[b.edge_ids[-1]]['edge_type']);self.assertFalse(any(edge_map[x]['edge_type']=='RESULT' for x in b.edge_ids[:-1]));self.assertLessEqual(b.order_edge_count,2);self.assertLessEqual(b.hypothesis_edge_count,4)
  for result_id in by_result:
   for view in ('ASSET_FLOW','STATE_DEPENDENCY','PARAM_PROPAGATION','CONTROL_CONTEXT'):
    paths=[b for b in result.backbones if b.result_id==result_id and view in b.semantic_views];self.assertLessEqual(len(paths),2)
    if len(paths)==2:
     first,second=paths;a=set(first.edge_ids);b=set(second.edge_ids);jaccard=len(a&b)/len(a|b);semantic=lambda path:{eid for eid in path.edge_ids if edge_map[eid]['edge_type'] in {'ASSET','STATE','PARAM'}}
     self.assertTrue(jaccard<=0.85 or bool(semantic(second)-semantic(first)))
  direct_root=[b for b in result.backbones if b.result_id=='result_r006' and 'CONTROL_CONTEXT' in b.semantic_views]
  self.assertEqual(1,len(direct_root));self.assertIn('DIRECT_ROOT_RESULT_CONTEXT',direct_root[0].selection_basis)
if __name__=='__main__':unittest.main()

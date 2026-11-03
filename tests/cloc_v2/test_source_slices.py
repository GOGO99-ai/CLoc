import json,tempfile,unittest
from pathlib import Path
from src.cloc_v2.source_index import build_source_index,selector_for_signature
from src.cloc_v2.source_slices import build_source_request_slices,requests_from_ranked,write_source_request_slices
from tests.cloc_v2.test_source_index import A,B,acquisition,cache_contract
C='0x'+'3'*40

def graph(*nodes):return {'schema_version':'transaction_graph_v1','nodes':list(nodes)}
class SourceSliceTests(unittest.TestCase):
 def test_selector_unique_name_ambiguous_special_and_real_lines(self):
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A);index=build_source_index(acquisition(),td)
   g=graph({'node_id':'n1','callee_address':A,'function_name':'foo','function_selector':selector_for_signature('foo(address)')},{'node_id':'n2','callee_address':A,'function_name':'foo','function_selector':None},{'node_id':'n3','callee_address':A,'function_name':'receive','function_selector':None},{'node_id':'n4','callee_address':A,'function_name':'help','function_selector':None})
   payload=build_source_request_slices(index,g,td,[{'chain_id':'c','node_id':f'n{i}','focus':'validation','priority':1} for i in range(1,5)])
   one,two,three,four=payload['requests'];self.assertEqual('RESOLVED',one['function_match_status']);self.assertIn('function foo(address who)',one['source']);self.assertTrue(one['source'].startswith('9:'))
   self.assertEqual('AMBIGUOUS',two['function_match_status']);self.assertEqual(2,len(two['candidate_functions']));self.assertEqual('RESOLVED',three['function_match_status']);self.assertEqual('RESOLVED',four['function_match_status'])
 def test_interface_declaration_does_not_hide_unique_implementation(self):
  unique='''interface I { function run(uint256 amount) external; }
contract C { function run(uint256 amount) external { helper(); } function helper() internal {} }'''
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A,unique);index=build_source_index(acquisition(),td)
   item=build_source_request_slices(index,graph({'node_id':'n','callee_address':A,'function_name':'run','function_selector':None}),td,[{'node_id':'n'}])['requests'][0]
   self.assertEqual('RESOLVED',item['function_match_status']);self.assertEqual('C',item['contract_name']);self.assertIn('UNIQUE_IMPLEMENTATION_BODY_MATCH',item['uncertainties'])
  multiple='''contract C { function run(uint256 amount) external {} }
contract D { function run(uint256 amount) external {} }'''
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A,multiple);index=build_source_index(acquisition(),td)
   item=build_source_request_slices(index,graph({'node_id':'n','callee_address':A,'function_name':'run','function_selector':None}),td,[{'node_id':'n'}])['requests'][0]
   self.assertEqual('AMBIGUOUS',item['function_match_status']);self.assertEqual(2,len(item['candidate_functions']))
 def test_proxy_multi_hop_and_failure_statuses(self):
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A,'contract P {}',{'implementation_address':B});cache_contract(td,B,'contract I {}',{'implementation_address':C});cache_contract(td,C,'contract Final { function ping(address x) external {} }')
   index=build_source_index(acquisition(),td);g=graph({'node_id':'n','callee_address':A,'function_name':'ping','function_selector':None})
   item=build_source_request_slices(index,g,td,[{'node_id':'n','focus':'x','priority':1}])['requests'][0];self.assertEqual('RESOLVED',item['function_match_status']);self.assertEqual(C,item['source_address']);self.assertEqual([A,B,C],item['proxy_resolution_path'])
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A,'contract Bad { function nope() {');index=build_source_index(acquisition(),td);g=graph({'node_id':'n','callee_address':A,'function_name':'nope','function_selector':None})
   self.assertEqual('PARSE_FAILED',build_source_request_slices(index,g,td,[{'node_id':'n'}])['requests'][0]['function_match_status'])
  index={'schema_version':'source_index_v1','chain':'eth','addresses':[]};g=graph({'node_id':'n','callee_address':A,'function_name':'x','function_selector':None})
  self.assertEqual('SOURCE_MISSING',build_source_request_slices(index,g,'.',[{'node_id':'n'}])['requests'][0]['function_match_status'])
 def test_ranked_contract_and_atomic_determinism(self):
  ranked={'schema_version':'phase1_ranked_chains_v1','selection_audit':{'all_chains_scored_exactly_once':True},'selected_topk':[{'chain_id':'c2','source_requests':[{'node_id':'n2','focus':'validation','priority':2}]},{'chain_id':'c1','source_requests':[{'node_id':'n1','focus':'arithmetic','priority':1}]}]}
  self.assertEqual(['n1','n2'],[x['node_id'] for x in requests_from_ranked(ranked)])
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A);index=build_source_index(acquisition(),td);g=graph({'node_id':'n1','callee_address':A,'function_name':'help','function_selector':None})
   payload=build_source_request_slices(index,g,td,[{'node_id':'n1'}]);p=Path(td)/'o.json';write_source_request_slices(payload,p);first=p.read_bytes();write_source_request_slices(payload,p);self.assertEqual(first,p.read_bytes())
  with self.assertRaises(ValueError):requests_from_ranked({'schema_version':'phase1_ranked_chains_v1'})

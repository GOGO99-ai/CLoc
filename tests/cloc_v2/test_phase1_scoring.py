import copy,json,unittest
from pathlib import Path
from src.cloc_v2.phase1_scoring import build_phase1_scoring_packets,build_phase1_scoring_packets_v2,validate_phase1_response
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
class Phase1Tests(unittest.TestCase):
 def test_packets_and_response_contract(self):
  docs=[json.loads((RUN/x).read_text()) for x in ('candidate_causal_graph_v2.json','candidate_causal_chains_v1.json','runtime_annotations_v1.json','result_candidates_v1.json','transaction_graph_v1.json')]
  before=copy.deepcopy(docs);p=build_phase1_scoring_packets(*docs);q=build_phase1_scoring_packets(*docs)
  self.assertEqual(json.dumps(p,sort_keys=True),json.dumps(q,sort_keys=True));self.assertEqual(before,docs);self.assertTrue(p['address_dictionary']);self.assertTrue(all(b['character_count']<=60000 and len(b['chain_ids'])<=6 for b in p['batches']))
  c=json.loads(p['batches'][0]['prompt_body'].split('\nDATA:\n',1)[1])['chains'][0];self.assertTrue(all(len(e.get('matches',[]))<=3 for e in c['backbone']['edges'] if e['edge_type']=='PARAM'))
  r={'responses':[{'chain_id':c['chain_id'],'result_relevance':1,'asset_anomaly':2,'state_dependency':3,'parameter_influence':4,'control_interaction':5,'causal_coherence':0,'total_score':46.0,'concise_rationale':'x','key_node_ids':[c['backbone']['nodes'][0]['node_id']],'key_edge_ids':[c['backbone']['edges'][0]['edge_id']],'counter_evidence':[],'uncertainties':[],'source_requests':[]}]}
  self.assertTrue(validate_phase1_response(r,p));r['responses'][0]['total_score']=0
  with self.assertRaises(ValueError):validate_phase1_response(r,p)
 def test_v2_compact_payload_has_exact_chain_coverage(self):
  docs=[json.loads((RUN/x).read_text()) for x in ('candidate_causal_graph_v2.json','candidate_causal_chains_v1.json','runtime_annotations_v1.json','result_candidates_v1.json','transaction_graph_v1.json')]
  p=build_phase1_scoring_packets_v2(*docs);ids=[i for b in p['batches'] for i in b['chain_ids']]
  self.assertEqual(19,len(ids));self.assertEqual(19,len(set(ids)));self.assertTrue(all(b['character_count']<=60000 for b in p['batches']));self.assertIn('candidate_causal_chains_v1',p['input_sha256'])
if __name__=='__main__':unittest.main()

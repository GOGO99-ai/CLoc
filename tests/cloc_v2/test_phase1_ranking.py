import copy,json,unittest
from pathlib import Path
from src.cloc_v2.phase1_ranking import validate_and_rank
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
class RankingTests(unittest.TestCase):
 def test_fixture_validation_ranking_and_determinism(self):
  p=json.loads((RUN/'phase1_scoring_packets_v2.json').read_text());c=json.loads((RUN/'candidate_causal_chains_v1.json').read_text());docs=[]
  for b in p['batches']:
   rs=[]
   for i,x in enumerate(b['chain_ids']):
    rs.append({'chain_id':x,'result_relevance':5-i%3,'asset_anomaly':3,'state_dependency':2,'parameter_influence':1,'control_interaction':4,'causal_coherence':5-i%2,'total_score':73.0-(i%3)*4,'concise_rationale':'fixture','key_node_ids':[],'key_edge_ids':[],'counter_evidence':[],'uncertainties':[],'source_requests':[]})
   docs.append({'batch_id':b['batch_id'],'responses':rs})
  # Correct fixture totals according to the contract.
  for d in docs:
   d['batch_id']=d['batch_id'].replace('phase1_batch_','')
   for r in d['responses']:
    r['total_score']=round((r['result_relevance']*20+r['asset_anomaly']*15+r['state_dependency']*15+r['parameter_influence']*15+r['control_interaction']*15+r['causal_coherence']*20)/5,2);r['scores']={k:r.pop(k) for k in ('result_relevance','asset_anomaly','state_dependency','parameter_influence','control_interaction','causal_coherence')};r['total_score']=f"{r['total_score']:.2f}"
  a=validate_and_rank(p,c,docs);z=validate_and_rank(p,c,copy.deepcopy(docs));self.assertEqual(json.dumps(a,sort_keys=True),json.dumps(z,sort_keys=True));self.assertEqual(19,len(a['raw_ranking']));self.assertEqual(a['raw_ranking'][0]['chain_id'],a['selected_topk'][0]['chain_id']);self.assertLessEqual(len(a['selected_topk']),5)
  bad=copy.deepcopy(docs);bad[0]['responses'][0]['chain_id']='unknown'
  with self.assertRaises(ValueError):validate_and_rank(p,c,bad)
  with self.assertRaises(ValueError):validate_and_rank(p,c,docs[:-1])
if __name__=='__main__':unittest.main()

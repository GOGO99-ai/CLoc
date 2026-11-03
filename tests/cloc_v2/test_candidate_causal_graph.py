from __future__ import annotations
import copy, unittest
from pathlib import Path
from src.cloc_v2.asset_continuity import load_observed_asset_ledger
from src.cloc_v2.candidate_causal_graph import build_candidate_causal_graph, load_result_candidates, serialize_candidate_causal_graph
from src.cloc_v2.observed_asset_ledger import load_runtime_annotations
from src.cloc_v2.result_detection import load_asset_continuity
from src.cloc_v2.runtime_semantics import load_transaction_graph
from src.cloc_v2.result_detection import detect_result_candidates
from src.cloc_v2.result_models import ResultDetectionConfig
from src.cloc_v2.outflow_scoring import validate_serialized_scoring

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'

class CandidateCausalGraphTests(unittest.TestCase):
 def test_grouped_recompute_and_tamper_rejection(self):
  fresh=detect_result_candidates(self.graph,self.annotations,self.ledger,self.continuity).to_dict()
  self.assertEqual(validate_serialized_scoring(self.results['detection_config']),'legacy_independent_weights')
  build_candidate_causal_graph(self.graph,self.annotations,self.ledger,self.continuity,fresh)
  for field in ('lambda','confidence','signal','audit','binding','mixed','unknown','missing'):
   with self.subTest(field=field):
    bad=copy.deepcopy(fresh)
    if field=='lambda':bad['detection_config']['outflow_lambda']='0'
    elif field=='confidence':bad['result_candidates'][0]['confidence']='0.0000'
    elif field=='signal':bad['result_candidates'][0]['signals'][0]['value']='0.1234'
    elif field=='audit':bad['result_candidates'][0]['outflow_score_audit']['raw_signals']['magnitude']='0.1234'
    elif field=='binding':bad['transaction_graph_sha256']='bad'
    elif field=='mixed':bad['detection_config']['outflow_weights']=self.results['detection_config']['outflow_weights']
    elif field=='unknown':bad['detection_config']['outflow_scoring_version']='future_v9'
    elif field=='missing':del bad['detection_config']['outflow_lambda']
    with self.assertRaises(ValueError):build_candidate_causal_graph(self.graph,self.annotations,self.ledger,self.continuity,bad)

 def test_lambda_provenance_prevents_normal_reference_mixing(self):
  from dataclasses import replace
  from src.cloc_v2.normal_distance import ChainFeatureRecord,MainAnchorKey,call_distance,calibrate_group_threshold
  from src.cloc_v2.outflow_scoring import scoring_context_id
  from src.cloc_v2.strict_result_chains import build_strict_result_chains
  from src.cloc_v2.normal_features import extract_normal_features
  fresh=detect_result_candidates(self.graph,self.annotations,self.ledger,self.continuity).to_dict()
  strict=build_strict_result_chains(self.graph,self.continuity,fresh,blockchain_chain_id=1)
  features=extract_normal_features(self.graph,strict,self.ledger,transaction_record_id='sample')
  self.assertTrue(features.records)
  self.assertTrue(all(r.result_scoring_context==scoring_context_id(fresh['detection_config']) for r in features.records))
  record=features.records[0]; legacy=replace(record,result_scoring_context='legacy_independent_weights',record_id='old')
  with self.assertRaisesRegex(ValueError,'SCORING_CONTEXT_MISMATCH'):call_distance(record,legacy)
  with self.assertRaisesRegex(ValueError,'SCORING_CONTEXT_MISMATCH'):calibrate_group_threshold((record,legacy),min_group_size=2)
  zero=replace(record,result_scoring_context=scoring_context_id(ResultDetectionConfig(outflow_lambda='0').to_dict()))
  with self.assertRaisesRegex(ValueError,'SCORING_CONTEXT_MISMATCH'):call_distance(record,zero)

 def setUp(self):
  if not RUN.exists(): self.skipTest('real phase artifacts unavailable')
  self.graph=load_transaction_graph(RUN/'transaction_graph_v1.json'); self.annotations=load_runtime_annotations(RUN/'runtime_annotations_v1.json'); self.ledger=load_observed_asset_ledger(RUN/'observed_asset_ledger_v1.json'); self.continuity=load_asset_continuity(RUN/'asset_continuity_v1.json'); self.results=load_result_candidates(RUN/'result_candidates_v1.json')
 def test_preserves_nodes_edges_and_result_seeds(self):
  built=build_candidate_causal_graph(self.graph,self.annotations,self.ledger,self.continuity,self.results)
  counts={k:sum(e.edge_type==k for e in built.edges) for k in ('CALL','ORDER','ASSET','RESULT')}
  self.assertEqual((len(built.nodes),counts['CALL'],counts['ORDER'],counts['ASSET'],counts['RESULT'],len(built.result_seeds)),(980,979,598,67,17,17))
  self.assertEqual({seed.result_id for seed in built.result_seeds},{x['result_id'] for x in self.results['result_candidates']})
  pairs={(e.src,e.dst) for e in built.edges if e.edge_type=='CALL'} & {(e.src,e.dst) for e in built.edges if e.edge_type=='RESULT'}
  self.assertTrue(pairs)
  self.assertTrue(all(e.source_refs for e in built.edges))
 def test_direct_parent_determinism_immutability_and_bad_reference(self):
  before=copy.deepcopy(self.results); first=build_candidate_causal_graph(self.graph,self.annotations,self.ledger,self.continuity,self.results); second=build_candidate_causal_graph(self.graph,self.annotations,self.ledger,self.continuity,self.results)
  self.assertEqual(serialize_candidate_causal_graph(first),serialize_candidate_causal_graph(second)); self.assertEqual(self.results,before)
  nodes={n.node_id:n for n in self.graph.nodes}
  for edge in first.edges:
   if edge.edge_type=='RESULT': self.assertEqual(nodes[edge.dst].parent_id,edge.src)
  bad=copy.deepcopy(self.results); bad['result_candidates'][0]['node_id']='n_missing'
  with self.assertRaisesRegex(ValueError,'missing node'): build_candidate_causal_graph(self.graph,self.annotations,self.ledger,self.continuity,bad)

if __name__=='__main__': unittest.main()

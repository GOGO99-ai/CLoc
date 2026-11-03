import importlib.util,json,tempfile,unittest
from pathlib import Path
from src.cloc_v2.source_manifest import write_run_manifest
from src.cloc_v2.source_models import SourceResult

SCRIPT=Path(__file__).resolve().parents[2]/'scripts'/'fetch_requested_sources.py'
spec=importlib.util.spec_from_file_location('fetch_requested_sources',SCRIPT);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
A='0x'+'1'*40
class SourceManifestTests(unittest.TestCase):
 def test_ranked_node_mapping_and_run_manifest_is_deterministic(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);ranked=root/'ranked.json';graph=root/'graph.json'
   ranked.write_text(json.dumps({'schema_version':'phase1_ranked_chains_v1','selection_audit':{'all_chains_scored_exactly_once':True},'selected_topk':[{'chain_id':'chain_c001','source_requests':[{'node_id':'n001','focus':'validation','priority':1,'request_reason':'test'},{'node_id':'n001','focus':'validation','priority':1,'request_reason':'test'}]}]}))
   graph.write_text(json.dumps({'schema_version':'transaction_graph_v1','nodes':[{'node_id':'n001','callee_address':A}]}))
   pairs,hashes=mod.requested_addresses_from_ranked(ranked,graph);self.assertEqual([(A,pairs[0][1])],pairs);self.assertEqual(2,len(hashes))
   out=root/'run.json';r=SourceResult('eth',A,A,'offline_missing',origins=pairs[0][1]);first=write_run_manifest(out,'eth','0xtx',[r],hashes);b1=out.read_bytes();write_run_manifest(out,'eth','0xtx',[r],hashes);self.assertEqual(b1,out.read_bytes());self.assertEqual(first['addresses'][0]['origins'][0]['node_id'],'n001')
 def test_invalid_ranked_request_is_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p=root/'r.json';g=root/'g.json';p.write_text(json.dumps({'schema_version':'phase1_ranked_chains_v1','selection_audit':{'all_chains_scored_exactly_once':True},'selected_topk':[{'chain_id':'c','source_requests':[{'node_id':'missing'}]}]}));g.write_text(json.dumps({'schema_version':'transaction_graph_v1','nodes':[]}))
   with self.assertRaises(ValueError):mod.requested_addresses_from_ranked(p,g)

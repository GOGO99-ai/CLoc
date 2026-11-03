from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.cloc_v2.source_slices import build_source_request_slices,requests_from_ranked,write_source_request_slices
def main():
 p=argparse.ArgumentParser(description='Build offline full-function source slices for validated phase-1 requests.')
 p.add_argument('--index',required=True,type=Path);p.add_argument('--transaction-graph',required=True,type=Path);p.add_argument('--cache-root',required=True,type=Path);p.add_argument('--ranked-input',type=Path);p.add_argument('--node-ids',help='Comma-separated explicit test node ids');p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if bool(a.ranked_input)==bool(a.node_ids):raise SystemExit('provide exactly one of --ranked-input or --node-ids')
 input_paths=(a.index,a.transaction_graph)+((a.ranked_input,) if a.ranked_input else ())
 if any(x.resolve()==a.output.resolve() for x in input_paths if x.exists()):raise SystemExit('output must differ from inputs')
 index=json.loads(a.index.read_text(encoding='utf-8'));graph=json.loads(a.transaction_graph.read_text(encoding='utf-8'));ranked=None
 if a.ranked_input: ranked=json.loads(a.ranked_input.read_text(encoding='utf-8'));requests=requests_from_ranked(ranked)
 else: requests=[{'chain_id':None,'node_id':item.strip(),'focus':'EXPLICIT_TEST','priority':3} for item in a.node_ids.split(',') if item.strip()]
 payload=build_source_request_slices(index,graph,a.cache_root,requests,ranked);write_source_request_slices(payload,a.output);print(json.dumps({'ok':True,'output':str(a.output),'request_count':payload['metadata']['request_count'],'resolved_count':payload['metadata']['resolved_count']},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())

"""Validate complete phase-1 response batches and atomically write rankings."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.cloc_v2.phase1_ranking import validate_and_rank,write_ranked
def load(path,schema):
 data=json.loads(Path(path).read_text(encoding='utf8'))
 if data.get('schema_version')!=schema:raise ValueError(f'expected {schema}')
 return data
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--packets',required=True);p.add_argument('--chains',required=True);p.add_argument('--responses-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--k',type=int,default=5);p.add_argument('--lambda-value',default='0.90');a=p.parse_args(argv)
 try:
  docs=[json.loads(x.read_text(encoding='utf8')) for x in sorted(Path(a.responses_dir).glob('*.json'))]
  if not docs:raise ValueError('responses directory contains no JSON files')
  result=validate_and_rank(load(a.packets,'phase1_scoring_packets_v2'),load(a.chains,'candidate_causal_chains_v1'),docs,a.k,a.lambda_value)
  write_ranked(result,a.output)
 except (ValueError,json.JSONDecodeError) as e:print(f'error: {e}',file=sys.stderr);return 2
 print(json.dumps({'ok':True,'output':str(Path(a.output).resolve()),'topk_count':len(result['selected_topk'])}));return 0
if __name__=='__main__': raise SystemExit(main())

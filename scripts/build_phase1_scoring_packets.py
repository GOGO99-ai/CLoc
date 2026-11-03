from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cloc_v2.phase1_scoring import PROMPT_TEMPLATE,build_phase1_scoring_packets,write_text
def load(path,schema):
 d=json.loads(Path(path).read_text(encoding='utf8'))
 if d.get('schema_version')!=schema:raise ValueError(f'expected {schema}')
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--chains',required=True);p.add_argument('--annotations',required=True);p.add_argument('--results',required=True);p.add_argument('--transaction-graph',required=True);p.add_argument('--packets-output',required=True);p.add_argument('--prompt-output',required=True);a=p.parse_args()
 try:
  r=build_phase1_scoring_packets(load(a.graph,'candidate_causal_graph_v2'),load(a.chains,'candidate_causal_chains_v1'),load(a.annotations,'runtime_annotations_v1'),load(a.results,'result_candidates_v1'),load(a.transaction_graph,'transaction_graph_v1'))
  write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',a.packets_output);write_text(PROMPT_TEMPLATE,a.prompt_output)
 except (ValueError,json.JSONDecodeError) as e:print(f'error: {e}',file=sys.stderr);return 2
 print(json.dumps({'ok':True,'batch_count':len(r['batches'])}));return 0
if __name__=='__main__':raise SystemExit(main())

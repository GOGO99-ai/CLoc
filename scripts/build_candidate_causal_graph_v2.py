from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.cloc_v2.augmented_causal_graph import build_augmented_candidate_causal_graph,load_json_artifact,write_augmented_graph
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--base-graph',required=True);p.add_argument('--parameter-evidence',required=True);p.add_argument('--state-evidence',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
 ins=[Path(x).resolve() for x in (a.base_graph,a.parameter_evidence,a.state_evidence)];out=Path(a.output).resolve()
 if out in ins or len(set(ins))!=3:p.error('all input and output paths must be distinct')
 try:g=build_augmented_candidate_causal_graph(load_json_artifact(ins[0],'candidate_causal_graph_v1'),load_json_artifact(ins[1],'parameter_evidence_v1'),load_json_artifact(ins[2],'state_evidence_v1'));write_augmented_graph(g,out)
 except ValueError as e:print(f'error: {e}',file=sys.stderr);return 2
 print(json.dumps({'ok':True,'output':str(out),**g.to_dict()['metadata']}));return 0
if __name__=='__main__':raise SystemExit(main())

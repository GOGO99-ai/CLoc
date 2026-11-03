from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.cloc_v2.asset_continuity import load_observed_asset_ledger
from src.cloc_v2.observed_asset_ledger import load_runtime_annotations
from src.cloc_v2.runtime_semantics import load_transaction_graph
from src.cloc_v2.state_evidence import build_state_evidence,load_candidate_causal_graph,write_state_evidence
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--annotations',required=True);p.add_argument('--ledger',required=True);p.add_argument('--causal-graph',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
 ins=[Path(x).resolve() for x in (a.graph,a.annotations,a.ledger,a.causal_graph)];out=Path(a.output).resolve()
 if out in ins or len(set(ins))!=4:p.error('all input and output paths must be distinct')
 try:r=build_state_evidence(load_transaction_graph(ins[0]),load_runtime_annotations(ins[1]),load_observed_asset_ledger(ins[2]),load_candidate_causal_graph(ins[3]));write_state_evidence(r,out)
 except ValueError as e:print(f'error: {e}',file=sys.stderr);return 2
 print(json.dumps({'ok':True,'output':str(out),**r.to_dict()['metadata']}));return 0
if __name__=='__main__':raise SystemExit(main())

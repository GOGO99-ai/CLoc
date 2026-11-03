from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.cloc_v2.causal_backbones import build_causal_backbones,load_candidate_graph_v2,write_causal_backbones
def main(argv=None):
 p=argparse.ArgumentParser(description='Extract bounded deterministic candidate causal backbones.');p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv);source=Path(a.input).resolve();target=Path(a.output).resolve()
 if source==target:p.error('input and output paths must be distinct')
 try:r=build_causal_backbones(load_candidate_graph_v2(source));write_causal_backbones(r,target)
 except ValueError as e:print(f'error: {e}',file=sys.stderr);return 2
 print(json.dumps({'ok':True,'output':str(target),**r.to_dict()['metadata']}));return 0
if __name__=='__main__':raise SystemExit(main())

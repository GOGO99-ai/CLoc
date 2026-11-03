from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.cloc_v2.asset_continuity import load_observed_asset_ledger
from src.cloc_v2.candidate_causal_graph import build_candidate_causal_graph, load_result_candidates, write_candidate_causal_graph
from src.cloc_v2.result_detection import load_asset_continuity
from src.cloc_v2.observed_asset_ledger import load_runtime_annotations
from src.cloc_v2.runtime_semantics import load_transaction_graph
def main(argv=None):
 p=argparse.ArgumentParser(); [p.add_argument(x,required=True) for x in ("--graph","--annotations","--ledger","--continuity","--results","--output")]; a=p.parse_args(argv)
 paths=[Path(getattr(a,k[2:])).resolve() for k in ("--graph","--annotations","--ledger","--continuity","--results")]; out=Path(a.output).resolve()
 if out in paths or len(set(paths))!=5: p.error("all input and output paths must be distinct")
 try:
  built=build_candidate_causal_graph(load_transaction_graph(paths[0]),load_runtime_annotations(paths[1]),load_observed_asset_ledger(paths[2]),load_asset_continuity(paths[3]),load_result_candidates(paths[4])); write_candidate_causal_graph(built,out)
 except ValueError as e: print(f"error: {e}",file=sys.stderr); return 2
 print(json.dumps({"ok":True,"output":str(out),**built.to_dict()["metadata"]})); return 0
if __name__=="__main__": raise SystemExit(main())

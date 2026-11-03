from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cloc_v2.phase1_scoring import PROMPT_TEMPLATE_V2,build_phase1_scoring_packets_v2,write_text
def load(p):return json.loads(Path(p).read_text())
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();[a.add_argument(x,required=True) for x in ('--graph','--chains','--annotations','--results','--transaction-graph','--packets-output','--prompt-output')];v=a.parse_args()
 r=build_phase1_scoring_packets_v2(load(v.graph),load(v.chains),load(v.annotations),load(v.results),load(v.transaction_graph));write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',v.packets_output);write_text(PROMPT_TEMPLATE_V2,v.prompt_output);print(json.dumps({'ok':True,'batches':len(r['batches'])}))

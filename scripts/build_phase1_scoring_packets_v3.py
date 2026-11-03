from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.cloc_v2.phase1_payload_v3 import audit_llm_payload,build_phase1_scoring_packets_v3
from src.cloc_v2.phase1_scoring import write_text
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--packets-v2',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args(argv)
 if a.packets_v2.resolve()==a.output.resolve():raise SystemExit('output must differ from v2 input')
 result=build_phase1_scoring_packets_v3(json.loads(a.packets_v2.read_text(encoding='utf8')));write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',a.output);print(json.dumps({'ok':True,'output':str(a.output),**audit_llm_payload(result)},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())

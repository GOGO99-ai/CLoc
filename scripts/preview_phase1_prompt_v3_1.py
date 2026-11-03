from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.cloc_v2.phase1_prompting import build_preview_v3_1,prompt_document_v3_1,preview_metrics_v3_1,write_preview
from src.cloc_v2.phase1_scoring import write_text
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--packets',required=True,type=Path);p.add_argument('--batch-id',default='phase1_batch_b001');p.add_argument('--preview-output',required=True,type=Path);p.add_argument('--prompt-output',required=True,type=Path);a=p.parse_args(argv)
 if a.packets.resolve() in {a.preview_output.resolve(),a.prompt_output.resolve()}:raise SystemExit('outputs must differ from packets input')
 preview=build_preview_v3_1(json.loads(a.packets.read_text(encoding='utf8')),a.batch_id);write_preview(preview,a.preview_output);write_text(prompt_document_v3_1(),a.prompt_output);print(json.dumps({'ok':True,'preview':str(a.preview_output),'prompt':str(a.prompt_output),**preview_metrics_v3_1(preview)},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())

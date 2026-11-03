from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.cloc_v2.phase1_llm_client import ClientConfig,OpenAICompatibleTransport,Phase1Runner
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--packets',required=True,type=Path);p.add_argument('--prompt',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--provider',default='UNSPECIFIED');p.add_argument('--base-url');p.add_argument('--api-key-env',default='OPENAI_API_KEY');p.add_argument('--model',default='UNSPECIFIED');p.add_argument('--temperature',default='UNSPECIFIED');p.add_argument('--max-output-tokens',default='UNSPECIFIED');p.add_argument('--timeout-seconds',default='UNSPECIFIED');p.add_argument('--max-attempts',type=int,default=3);p.add_argument('--concurrency',type=int,default=1);p.add_argument('--omit-temperature',action='store_true');mode=p.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--batch-id');a=p.parse_args(argv)
 config=ClientConfig(a.provider,a.model,a.temperature,a.max_output_tokens,a.timeout_seconds,a.max_attempts,a.concurrency,a.omit_temperature);runner=Phase1Runner(json.loads(a.packets.read_text(encoding='utf8')),a.prompt,a.output_dir,config,a.packets,a.batch_id)
 if not a.execute:
  manifest=runner.dry_run();print(json.dumps({'ok':True,'mode':'dry_run','batch_count':len(manifest['batches']),'output_dir':str(a.output_dir)},ensure_ascii=False));return 0
 config.validate_execute()
 if not a.base_url:raise SystemExit('--execute requires --base-url')
 api_key=os.getenv(a.api_key_env)
 if not api_key:raise SystemExit(f'--execute requires environment variable {a.api_key_env}')
 manifest=runner.execute(OpenAICompatibleTransport(a.base_url),api_key,a.resume);print(json.dumps({'ok':manifest['complete'],'mode':'execute','output_dir':str(a.output_dir)},ensure_ascii=False));return 0 if manifest['complete'] else 2
if __name__=='__main__':raise SystemExit(main())

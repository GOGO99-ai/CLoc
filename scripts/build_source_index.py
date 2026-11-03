from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.cloc_v2.source_index import build_source_index, write_source_index
def main():
 p=argparse.ArgumentParser(description='Build an offline Solidity source index from 5A cache artifacts.')
 p.add_argument('--acquisition',required=True,type=Path);p.add_argument('--cache-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if a.acquisition.resolve()==a.output.resolve(): raise SystemExit('output must differ from acquisition input')
 acquisition=json.loads(a.acquisition.read_text(encoding='utf-8'));index=build_source_index(acquisition,a.cache_root);write_source_index(index,a.output);print(json.dumps({'ok':True,'output':str(a.output),'address_count':index['metadata']['address_count'],'function_count':index['metadata']['function_count']},indent=2));return 0
if __name__=='__main__': raise SystemExit(main())

"""Offline, version-isolated endpoint/chain replay and lambda scan.

Each worker imports either the saved prechange source or the current source.
No raw trace, endpoint or normal-feature artifact is shared across formulas.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/experiments/outflow_lambda_20261015"
LAMBDAS = ("0.00", "0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.65", "0.70", "0.80", "0.90", "1.00")
TZ = timezone(timedelta(hours=8))
FILES = ("transaction_graph_v1.json", "runtime_annotations_v1.json", "observed_asset_ledger_v1.json", "asset_continuity_v1.json")


def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def worker(args):
    sys.path.insert(0, str(ROOT))  # Unchanged root config/module dependencies.
    sys.path.insert(0, str(args.source_root.resolve()))
    from src.cloc_v2 import result_detection as rd
    from src.cloc_v2.result_models import ResultDetectionConfig
    from src.cloc_v2.candidate_causal_graph import build_candidate_causal_graph
    from src.cloc_v2.strict_result_chains import build_strict_result_chains
    from src.cloc_v2.normal_features import extract_normal_features
    if not Path(rd.__file__).resolve().is_relative_to(args.source_root.resolve()):
        raise ValueError('WORKER_SOURCE_ISOLATION_FAILED')
    graph = rd.load_transaction_graph(args.worker_input / FILES[0])
    annotations = rd.load_runtime_annotations(args.worker_input / FILES[1])
    ledger = rd.load_observed_asset_ledger(args.worker_input / FILES[2])
    continuity = rd.load_asset_continuity(args.worker_input / FILES[3])
    config = ResultDetectionConfig() if args.outflow_lambda is None else ResultDetectionConfig(outflow_lambda=args.outflow_lambda)
    cfg = config.to_dict()
    collected = []
    select = rd._select_outflows

    def capture(*pos, **kw):
        # Score every eligible fact via the same original implementation;
        # selection for the actual artifact still uses its unchanged defaults.
        all_cfg = replace(pos[-1], outflow_threshold="0", max_candidates_per_asset=max(1, len(ledger.transfer_facts)))
        collected.extend(select(*pos[:-1], all_cfg, **kw))
        return select(*pos, **kw)

    rd._select_outflows = capture
    try:
        result = rd.detect_result_candidates(graph, annotations, ledger, continuity, config).to_dict()
    finally:
        rd._select_outflows = select
    save(args.worker_output / "result_candidates_v1.json", result)
    causal = build_candidate_causal_graph(graph, annotations, ledger, continuity, result).to_dict()
    save(args.worker_output / "candidate_causal_graph_v1.json", causal)
    strict = build_strict_result_chains(graph, continuity, result, blockchain_chain_id=1)
    save(args.worker_output / "strict_result_chains_v1.json", strict.to_dict())
    features = extract_normal_features(graph, strict, ledger, transaction_record_id=args.worker_input.name)
    save(args.worker_output / "normal_features_v1.json", features.to_dict())
    selected = {r['source_fact_ids'][0]: r for r in result['result_candidates'] if r['result_type']=='R_OUTFLOW'}
    remaining = rd._remaining_amounts(continuity)
    facts = {f.fact_id: f for f in ledger.transfer_facts}
    grouped = defaultdict(list)
    for c in collected:
        grouped[rd._asset_key(facts[c.source_fact_ids[0]])].append(facts[c.source_fact_ids[0]])
    magnitudes = {k:v for group in grouped.values() for k,v in rd._magnitude_scores(group).items()}
    root = next(n for n in graph.nodes if n.node_id == graph.root_id)
    callees = {n.callee_address.lower() for n in graph.nodes if isinstance(n.callee_address, str)}
    ranks = {}
    for key in grouped:
        rows = sorted((c for c in collected if rd._asset_key(facts[c.source_fact_ids[0]])==key), key=lambda c:(-Decimal(c.confidence),c.trace_index,c.node_id,c.source_fact_ids))
        ranks.update({c.source_fact_ids[0]: i for i,c in enumerate(rows,1)})
    audit = []
    for c in collected:
        fact_id=c.source_fact_ids[0]; fact=facts[fact_id]
        raw = c.to_dict().get('outflow_score_audit',{}).get('raw_signals') or {
            'magnitude':str(magnitudes[fact_id]),'terminal_retention':str(rd._retention(fact,remaining)),
            'late_execution':str(Decimal(fact.trace_index)/len(graph.nodes)),
            'root_receiver':str(int(rd._same_address(fact.to_address,root.callee_address))),
            'source_activity':str(int(fact.from_address['normalized'] in callees)),
        }
        with localcontext(Context(prec=50,rounding=ROUND_HALF_UP)):
            d=(Decimal(raw['magnitude'])+Decimal(raw['terminal_retention']))/2
            ctx=(Decimal(raw['late_execution'])+Decimal(raw['root_receiver'])+Decimal(raw['source_activity']))/3
        keep=selected.get(fact_id)
        reason=keep['selection_reason'] if keep else 'BELOW_THRESHOLD' if Decimal(c.confidence)<Decimal(config.outflow_threshold) else 'ASSET_CAP_OR_NODE_TYPE_DEDUPLICATION'
        audit.append({'source_fact_id':fact_id,'asset_key':list(rd._asset_key(fact)), 'node_id':fact.node_id,
                      'raw_signals':raw,'asset_group_mean':str(d),'context_group_mean':str(ctx),
                      'confidence':c.confidence,'kept':keep is not None,'selection_reason':reason,'asset_rank':ranks[fact_id]})
    save(args.worker_output/'all_eligible_audit.json',audit)
    summary={'status':'SUCCEEDED','time':datetime.now(TZ).isoformat(),'config':cfg,'eligible_count':len(audit),'scoring_module_path':rd.__file__,'scoring_module_sha256':sha(rd.__file__),
             'selected_count':len(selected),'top1_fallback_count':result['metadata']['top1_fallback_count'],
             'strict_chain_count':len(strict.chains),'ready_chain_count':sum(c.status=='READY' for c in strict.chains),
             'feature_record_count':len(features.records),'dangling_edge_count':causal['metadata']['dangling_edge_count'],
             'input_hashes':{name:sha(args.worker_input/name) for name in FILES},
             'artifact_hashes':{p.name:sha(p) for p in sorted(args.worker_output.glob('*.json')) if p.name not in {'run_summary.json','process_log.json'}}}
    save(args.worker_output/'run_summary.json',summary)
    return 0


def prepare_samples(output):
    sys.path.insert(0,str(ROOT))
    from src.cloc_v2.transaction_graph import build_transaction_graph, load_call_tree
    from src.cloc_v2.runtime_semantics import build_runtime_annotations
    from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
    from src.cloc_v2.asset_continuity import build_asset_continuity
    from src.data_collection.trace_parser import TraceParser
    mono=ROOT/'data/runs/eth/0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
    samples=[{'alias':'monox','kind':'FAULT','transaction':mono.name,'bundle':str(mono),'input_files':{n:sha(mono/n) for n in FILES}}]
    normals=ROOT/'data/prototypes/formal_normal_baseline_full/samples/eth'
    traces=sorted(p for p in (ROOT/'data/cache/traces').glob('eth_*.txt') if (normals/(p.stem[4:]+'.json')).exists())[:2]
    fault_manifest=ROOT/'data/prototypes/localization15_topk_v4_20260901/locked_inference_manifest.json'
    allowed={r['tx_hash'].lower() for r in load(fault_manifest)['samples']}-{mono.name}
    fault_trees=sorted((p for base in [ROOT/'data/cache/all_tree',ROOT/'data/trace_acquisition/localization15_extension_20260911/eth'] for p in base.rglob('*.json') if (p.stem.lower() if p.stem.startswith('0x') else p.parent.name.lower()) in allowed),key=lambda p:((p.stem if p.stem.startswith('0x') else p.parent.name),str(p)))[:1]
    selected=[('FAULT',p) for p in fault_trees]+[('NORMAL',p) for p in traces]
    for i,(kind,path) in enumerate(selected,1):
        alias=f'{kind.lower()}_{i:02d}'; bundle=output/'inputs'/alias
        item={'alias':alias,'kind':kind,'source_path':str(path.relative_to(ROOT)), 'source_sha256':sha(path),'bundle':str(bundle),'transaction':path.stem[4:] if kind=='NORMAL' else path.stem if path.stem.startswith('0x') else path.parent.name}
        if kind=='NORMAL':
            membership=normals/(item['transaction']+'.json'); item['normal_membership_path']=str(membership.relative_to(ROOT));item['normal_membership_sha256']=sha(membership)
        else:
            item['fault_membership_path']=str(fault_manifest.relative_to(ROOT));item['fault_membership_sha256']=sha(fault_manifest)
        try:
            tree=TraceParser().parse(path.read_text(encoding='utf-8'))['call_tree'] if kind=='NORMAL' else load_call_tree(path)
            if hasattr(tree, 'to_dict'): tree=tree.to_dict()
            graph=build_transaction_graph(tree); annotations=build_runtime_annotations(graph);ledger=build_observed_asset_ledger(graph,annotations);continuity=build_asset_continuity(graph,annotations,ledger)
            for name,obj in zip(FILES,(graph,annotations,ledger,continuity)):save(bundle/name,obj.to_dict())
            item['input_files']={name:sha(bundle/name) for name in FILES};item['preparation_status']='SUCCEEDED'
        except Exception as exc:
            item['preparation_status']='FAILED';item['error']=f'{type(exc).__name__}: {exc}'
        samples.append(item)
    return samples


def scan(args):
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'scan_manifest.json'
    source_hashes={str(p.relative_to(ROOT)):sha(p) for base in ['src/cloc_v2','src/data_collection','scripts','tests/cloc_v2'] for p in sorted((ROOT/base).glob('*.py'))}
    source_hashes['config.py']=sha(ROOT/'config.py')
    for entry in load(output/'prechange_manifest.json')['source_files']:
        if sha(output/'baseline_repo'/entry['path'])!=entry['sha256']:raise ValueError('BASELINE_SNAPSHOT_HASH_DRIFT')
    if manifest_path.exists():
        manifest=load(manifest_path)
        if manifest['source_hashes']!=source_hashes or manifest['lambdas']!=list(args.lambdas):raise ValueError('SCAN_CODE_OR_CONFIGURATION_DRIFT: use a new output directory')
    else:
        manifest={'run_id':'outflow_lambda_20261015','time':datetime.now(TZ).isoformat(),'lambdas':list(args.lambdas),
                  'default_lambda':'0.65','selection_rule':'MONOX_FIXED; FIRST_OTHER_FROZEN15_FAULT_BY_TX_AND_PATH; FIRST_TWO_CACHED_TEXT_TRACES_WITH_FORMAL_NORMAL_MEMBERSHIP_BY_FILENAME',
                  'samples':prepare_samples(output),'source_hashes':source_hashes,'baseline_manifest_sha256':sha(output/'prechange_manifest.json'),
                  'external_calls':{'rpc':0,'source':0,'llm':0},'llm_experiment_executed':False}
        save(manifest_path,manifest)
    rows=[]
    for mode in ['legacy',*args.lambdas]:
        for sample in manifest['samples']:
            dest=output/('legacy' if mode=='legacy' else f'lambda_{mode}')/sample['alias']
            command=[sys.executable,str(Path(__file__).resolve()),'--worker-input',sample['bundle'],'--worker-output',str(dest),'--source-root',str(output/'baseline_repo' if mode=='legacy' else ROOT)]
            if mode!='legacy':command+=['--outflow-lambda',mode]
            record={'sample_alias':sample['alias'],'kind':sample['kind'],'mode':mode,'command':command,'time':datetime.now(TZ).isoformat()}
            try:
                if sample.get('preparation_status')=='FAILED':raise ValueError(sample['error'])
                for name,digest in sample['input_files'].items():
                    if sha(Path(sample['bundle'])/name)!=digest:raise ValueError('INPUT_HASH_DRIFT')
                result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=180)
                save(dest/'process_log.json',{'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
                if result.returncode:raise ValueError(result.stderr[-3000:])
                record.update(load(dest/'run_summary.json'))
            except Exception as exc:record.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
            rows.append(record);save(output/'runs.json',rows)
            print(f"mode={mode} sample={sample['alias']} status={record['status']}",flush=True)
    summaries=[]; comparisons=[]
    for mode in args.lambdas:
        selected=[r for r in rows if r['mode']==mode];success=[r for r in selected if r['status']=='SUCCEEDED']
        summaries.append({'lambda':mode,'success_numerator':len(success),'planned_denominator':len(manifest['samples']),
                          'nonempty_chain_numerator':sum(r['strict_chain_count']>0 for r in success),
                          'outflow_count':sum(r['selected_count'] for r in success),'top1_fallback_count':sum(r['top1_fallback_count'] for r in success),
                          'ready_chain_count':sum(r['ready_chain_count'] for r in success),'built_chain_count':sum(r['strict_chain_count'] for r in success)})
        for sample in manifest['samples']:
            oldpath=output/'legacy'/sample['alias']; newpath=output/f'lambda_{mode}'/sample['alias']
            if not (oldpath/'all_eligible_audit.json').exists() or not (newpath/'all_eligible_audit.json').exists():continue
            old={r['source_fact_id']:r for r in load(oldpath/'all_eligible_audit.json')};new={r['source_fact_id']:r for r in load(newpath/'all_eligible_audit.json')}
            if set(old)!=set(new):raise ValueError('ELIGIBLE_FACT_SET_CHANGED')
            for fact in old:
                comparisons.append({'lambda':mode,'sample_alias':sample['alias'],'transaction':sample['transaction'],'asset_key':old[fact]['asset_key'],'source_fact_id':fact,'legacy':old[fact],'grouped':new[fact]})
            def profits(path):return sorted(({k:v for k,v in r.items() if k!='result_id'} for r in load(path/'result_candidates_v1.json')['result_candidates'] if r['result_type']=='R_PROFIT'),key=lambda r:json.dumps(r,sort_keys=True))
            if profits(oldpath)!=profits(newpath):raise ValueError('R_PROFIT_SEMANTICS_CHANGED')
        mode_rows=[r for r in comparisons if r['lambda']==mode]
        summaries[-1]['identity_membership_changes']=sum(r['legacy']['kept']!=r['grouped']['kept'] for r in mode_rows)
        summaries[-1]['identity_comparison_denominator']=len(mode_rows)
    save(output/'candidate_comparison.json',comparisons)
    save(output/'scan_summary.json',{'summary':summaries,'run_count':len(rows),'failure_count':sum(r['status']!='SUCCEEDED' for r in rows),'function_recall_measured':False})
    print(json.dumps(summaries,indent=2))
    return 1 if any(r['status']!='SUCCEEDED' for r in rows) else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--lambdas',nargs='+',default=LAMBDAS)
    p.add_argument('--worker-input',type=Path);p.add_argument('--worker-output',type=Path);p.add_argument('--source-root',type=Path);p.add_argument('--outflow-lambda')
    args=p.parse_args()
    return worker(args) if args.worker_input else scan(args)


if __name__=='__main__':raise SystemExit(main())

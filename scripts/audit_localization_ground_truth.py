"""Offline audit of localization ground-truth availability and frozen Top-K metrics."""
from __future__ import annotations
import argparse,hashlib,json,sys,tempfile
from collections import Counter
from pathlib import Path
from typing import Any,Mapping

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.cloc_v2.localization_ground_truth import parse_localization_ground_truth,resolve_localization_ground_truth,evaluate_localization_topk
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index

DEFAULT_EXPERIMENT=ROOT/"data"/"prototypes"/"final_small_experiment_eth_seed_20260901"
DEFAULT_OUTPUT=ROOT/"data"/"prototypes"/"localization_ground_truth_audit_20260901"/"ground_truth_audit.json"

def _load(path:Path)->Any:
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError):return None
def _canonical(v:Any)->str:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def _flatten(value:Any)->list[str]:
    out=[]; stack=[value]
    while stack:
        item=stack.pop()
        if isinstance(item,list):stack.extend(reversed(item))
        elif isinstance(item,str):out.append(item)
    return out
def _frozen_valid(blob:Mapping[str,Any])->bool:
    payload={k:blob[k] for k in ("schema_version","kind","locked_manifest_sha256","phase2_queue","predictions") if k in blob}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()==blob.get("payload_sha256")

def _dataset_records(dataset_root:Path)->list[tuple[tuple[str,...],tuple[str,...]]]:
    rows=[]
    for path in sorted(dataset_root.rglob("*.json")):
        value=_load(path)
        if not isinstance(value,Mapping) or value.get("platform")!="Ethereum":continue
        fault=value.get("fault")
        if not isinstance(fault,Mapping):continue
        txs=tuple(sorted(set(x.lower() for x in _flatten(fault.get("transaction_hash")) if x.startswith("0x") and len(x)==66)))
        locations=tuple(_flatten(fault.get("location")))
        if txs:rows.append((txs,locations))
    return rows

def build_ground_truth_audit(*,dataset_root:Path,cache_root:Path,experiment_root:Path)->dict[str,Any]:
    index=build_verified_source_cache_index(cache_root,blockchain_chain_id=1,chain_key="eth")
    records=_dataset_records(dataset_root); transactions=sorted({tx for txs,_ in records for tx in txs}); reason_counts=Counter(); label_total=label_parsed=label_explicit=label_cache_hit=label_file=label_function=label_lines=0
    tx_with_location=tx_offset=tx_parse=tx_explicit=tx_parse_explicit=tx_cache_hit=tx_file=tx_function=tx_lines=0; label_offset=0; resolved_by_tx={}
    for txs,locations in records:
        if locations:tx_with_location+=len(txs)
        offset_shaped=[x for x in locations if isinstance(x,str) and len(x.split("#"))==3 and re_full_offset(x.split("#")[-1])]
        if offset_shaped:tx_offset+=len(txs)
        raw_explicit=any(isinstance(x,str) and len(x.split("#"))==3 and bool(x.split("#")[1]) for x in locations)
        tx_explicit+=len(txs)*int(raw_explicit);label_offset+=len(offset_shaped)
        parsed=[]
        for location in locations:
            label_total+=1
            try:label=parse_localization_ground_truth(location)
            except ValueError:reason_counts["GROUND_TRUTH_LOCATION_INVALID"]+=1;continue
            label_parsed+=1; label_explicit+=int(label.source_file is not None)
            resolved=resolve_localization_ground_truth(label,index); parsed.append(resolved)
            reason_counts.update(resolved.reason_codes)
            label_cache_hit+=int("GROUND_TRUTH_SOURCE_CACHE_MISSING" not in resolved.reason_codes and "GROUND_TRUTH_SOURCE_CACHE_CONFLICT" not in resolved.reason_codes)
            label_file+=int(resolved.source_file_available);label_function+=int(resolved.function_available);label_lines+=int(resolved.line_range_available)
        for tx in txs:resolved_by_tx.setdefault(tx,[]).extend(parsed)
        if parsed:
            tx_parse+=len(txs);tx_parse_explicit+=len(txs)*int(any(label.source_file is not None for label in [parse_localization_ground_truth(x) for x in locations if isinstance(x,str) and _safe_parse(x)]));tx_cache_hit+=len(txs)*int(any("GROUND_TRUTH_SOURCE_CACHE_MISSING" not in x.reason_codes and "GROUND_TRUTH_SOURCE_CACHE_CONFLICT" not in x.reason_codes for x in parsed));tx_file+=len(txs)*int(any(x.source_file_available for x in parsed));tx_function+=len(txs)*int(any(x.function_available for x in parsed));tx_lines+=len(txs)*int(any(x.line_range_available for x in parsed))

    p1=_load(experiment_root/"frozen_phase1_predictions.json");p2=_load(experiment_root/"frozen_phase2_predictions.json");truth=_load(experiment_root/"evaluation_truth.json");manifest=_load(experiment_root/"locked_inference_manifest.json");checkpoint=_load(experiment_root/"checkpoint.json")
    if not all(isinstance(x,Mapping) for x in (p1,p2,truth,manifest,checkpoint)) or not _frozen_valid(p1) or not _frozen_valid(p2):raise ValueError("FROZEN_PREDICTIONS_INVALID")
    tx_by_alias={x["sample_alias"]:x["tx_hash"].lower() for x in manifest.get("samples",[]) if isinstance(x,Mapping)}; labels=truth.get("labels",{}); phase2_rows={x["sample_alias"]:x for x in p2.get("predictions",[])}
    topk_total=sum(len(x.get("top3",[])) for x in phase2_rows.values());candidate_fields={"file":0,"function":0,"line_range":0}
    for row in phase2_rows.values():
        for item in row.get("top3",[]):candidate_fields["file"]+=int(bool(item.get("relative_file")));candidate_fields["function"]+=int(bool(item.get("function")));candidate_fields["line_range"]+=int(isinstance(item.get("start_line"),int) and isinstance(item.get("end_line"),int))
    fault_queue=[a for a in p2.get("phase2_queue",[]) if labels.get(a)=="FAULT"]; experiment_levels=Counter(); metric_counts=Counter(); condition_den=0
    for alias in fault_queue:
        resolutions=resolved_by_tx.get(tx_by_alias.get(alias,""),[]);best=next((x for x in resolutions if x.function_available),next((x for x in resolutions if x.line_range_available),next((x for x in resolutions if x.source_file_available),None)))
        if best is None:experiment_levels["unresolved"]+=1;continue
        experiment_levels["file"]+=int(best.source_file_available);experiment_levels["function"]+=int(best.function_available);experiment_levels["line_range"]+=int(best.line_range_available)
        findings=phase2_rows.get(alias,{}).get("top3",[]);metrics=evaluate_localization_topk(best,findings)
        if findings:condition_den+=1
        for key,value in metrics.to_dict().items():
            if key!="topk_generated" and value is True:metric_counts[key]+=1
    names=("file_recall_at_1","file_recall_at_3","function_recall_at_1","function_recall_at_3","line_range_recall_at_1","line_range_recall_at_3")
    conditional={name:(metric_counts[name]/condition_den if condition_den else None) for name in names};end_to_end={name:(metric_counts[name]/len(fault_queue) if fault_queue else None) for name in names}
    return {"schema_version":"localization_ground_truth_audit_v1","scientific_generalization_result":False,"exact_line_localization_supported":False,"provider_free_text_used":False,"dataset":{"ethereum_fault_transaction_count":len(transactions),"transaction_with_location_count":tx_with_location,"transaction_offset_length_parseable_count":tx_offset,"transaction_strict_location_parseable_count":tx_parse,"transaction_explicit_file_count":tx_explicit,"transaction_strict_parseable_explicit_file_count":tx_parse_explicit,"transaction_verified_source_cache_hit_count":tx_cache_hit,"transaction_file_truth_resolved_count":tx_file,"transaction_function_truth_resolved_count":tx_function,"transaction_line_range_truth_resolved_count":tx_lines,"location_label_count":label_total,"offset_length_parseable_label_count":label_offset,"strict_parseable_location_label_count":label_parsed,"explicit_file_location_label_count":label_explicit,"verified_source_cache_hit_label_count":label_cache_hit,"file_truth_resolved_label_count":label_file,"function_truth_resolved_label_count":label_function,"line_range_truth_resolved_label_count":label_lines,"reason_code_counts":dict(sorted(reason_counts.items()))},"frozen_experiment":{"phase2_queue_count":len(p2.get("phase2_queue",[])),"valid_topk_sample_count":sum(bool(x.get("topk_frozen")) for x in phase2_rows.values()),"topk_candidate_count":topk_total,"candidate_field_counts":candidate_fields,"fault_queue_count":len(fault_queue),"fault_truth_resolution_levels":dict(experiment_levels),"conditional_metric_denominator":condition_den,"end_to_end_metric_denominator":len(fault_queue),"confirmed_hit_counts":{name:metric_counts[name] for name in names},"conditional_metrics":conditional,"end_to_end_metrics":end_to_end},"coverage":{"ground_truth_resolution_coverage":tx_lines/len(transactions) if transactions else None,"topk_generation_coverage":sum(bool(x.get("topk_frozen")) for x in phase2_rows.values())/len(p2.get("phase2_queue",[])) if p2.get("phase2_queue") else None,"evaluation_coverage":condition_den/len(fault_queue) if fault_queue else None},"top5_supported":False,"top5_note":"Top-K v3最多3项；如需与Top-5直接比较，必须在查看本轮评价结果前预先冻结扩展契约并重新盲测。"}

def _safe_parse(value:str)->bool:
    try:parse_localization_ground_truth(value);return True
    except ValueError:return False

def re_full_offset(value:str)->bool:
    import re
    match=re.fullmatch(r"([0-9]+):([0-9]+)",value)
    return bool(match and int(match.group(2))>0)

def write_audit(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n"
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,delete=False) as h:h.write(data);tmp=Path(h.name)
    tmp.replace(path)

def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args(argv);audit=build_ground_truth_audit(dataset_root=ROOT/"dataset",cache_root=ROOT/"data"/"source_cache",experiment_root=DEFAULT_EXPERIMENT);write_audit(a.output,audit);print("audit_status=COMPLETED");print("rpc_request_count=0");print("source_request_count=0");print("llm_request_count=0");return 0
if __name__=="__main__":raise SystemExit(main())

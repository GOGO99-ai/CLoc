"""Re-evaluate frozen localization predictions with truth repair v1, offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.audit_full_fault_localization_evaluability import _dataset_locations
from src.cloc_v2.localization_ground_truth import parse_localization_ground_truth_repair_v1,resolve_localization_ground_truth_repair_v1
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index


DATASET_ROOT=ROOT/"dataset"
CACHE_ROOT=ROOT/"data"/"source_cache"
HISTORICAL15_ROOT=ROOT/"data"/"prototypes"/"localization15_topk_v4_20260901"
FAULT10_ROOT=ROOT/"data"/"prototypes"/"localization_fault10_blind_legacy_v4_seed_20260824"
DEFAULT_OUTPUT=ROOT/"data"/"prototypes"/"localization_truth_repair_v1_20260824"
EXPECTED_PROFILE_SHA="617bee01e72f6c71a5e3e42b5f8912b53c325516e494f3ff5e8e66f56b3353c9"
_TX=re.compile(r"^0x[0-9a-fA-F]{64}$")
_RAW_IDENTITY=re.compile(r"0x[0-9a-fA-F]{40,64}")


def _canonical(value:Any)->str:return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def _sha(value:Any)->str:return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
def _file_sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))


def _manifest_core(value:Mapping[str,Any],kind:str)->dict[str,Any]:
    if kind=="historical15":keys=("schema_version","selection_namespace","selection_seed","samples")
    elif kind=="fault10":keys=("schema_version","experiment_type","selection_namespace","selection_seed","blockchain_chain_id","profile_id","profile_sha256","samples","selection_audit")
    else:raise ValueError("TRUTH_REPAIR_INPUT_KIND_INVALID")
    if any(key not in value for key in keys):raise ValueError("TRUTH_REPAIR_MANIFEST_SCHEMA_INVALID")
    return {key:value[key] for key in keys}


def _frozen_core(value:Mapping[str,Any],kind:str)->dict[str,Any]:
    if kind=="historical15":keys=("schema_version","kind","locked_manifest_sha256","predictions","external_call_counts")
    elif kind=="fault10":keys=("schema_version","kind","locked_manifest_sha256","profile_sha256","predictions","physical_request_counts")
    else:raise ValueError("TRUTH_REPAIR_INPUT_KIND_INVALID")
    if any(key not in value for key in keys):raise ValueError("TRUTH_REPAIR_FROZEN_SCHEMA_INVALID")
    return {key:value[key] for key in keys}


def validate_frozen_group(root:Path,kind:str,expected_count:int)->tuple[Mapping[str,Any],Mapping[str,Any],dict[str,str],dict[str,str]]:
    manifest=_load(root/"locked_inference_manifest.json");frozen=_load(root/"frozen_predictions.json");checkpoint=_load(root/"checkpoint.json")
    if not all(isinstance(item,Mapping) for item in (manifest,frozen,checkpoint)):raise ValueError("TRUTH_REPAIR_INPUT_INVALID")
    manifest_sha=_sha(_manifest_core(manifest,kind))
    if manifest.get("manifest_sha256")!=manifest_sha:raise ValueError("TRUTH_REPAIR_MANIFEST_SHA_INVALID")
    frozen_sha=_sha(_frozen_core(frozen,kind))
    if frozen.get("payload_sha256")!=frozen_sha or frozen.get("locked_manifest_sha256")!=manifest_sha:raise ValueError("TRUTH_REPAIR_FROZEN_SHA_INVALID")
    if checkpoint.get("status")!="COMPLETED" or checkpoint.get("frozen_predictions_sha256")!=frozen_sha:raise ValueError("TRUTH_REPAIR_CHECKPOINT_INVALID")
    if kind=="fault10" and (manifest.get("profile_sha256")!=EXPECTED_PROFILE_SHA or frozen.get("profile_sha256")!=EXPECTED_PROFILE_SHA):raise ValueError("TRUTH_REPAIR_PROFILE_SHA_INVALID")
    samples=manifest.get("samples");predictions=frozen.get("predictions")
    if not isinstance(samples,list) or not isinstance(predictions,list) or len(samples)!=expected_count or len(predictions)!=expected_count:raise ValueError("TRUTH_REPAIR_SAMPLE_COUNT_INVALID")
    tx_by_alias={}
    for row in samples:
        if not isinstance(row,Mapping) or not isinstance(row.get("sample_alias"),str) or not isinstance(row.get("tx_hash"),str) or not _TX.fullmatch(row["tx_hash"]):raise ValueError("TRUTH_REPAIR_MANIFEST_SAMPLE_INVALID")
        tx_by_alias[row["sample_alias"]]=row["tx_hash"].lower()
    prediction_by_alias={row["sample_alias"]:row for row in predictions if isinstance(row,Mapping) and isinstance(row.get("sample_alias"),str)}
    if set(prediction_by_alias)!=set(tx_by_alias):raise ValueError("TRUTH_REPAIR_PREDICTION_SET_INVALID")
    digests={name:_file_sha(root/name) for name in ("locked_inference_manifest.json","frozen_predictions.json","checkpoint.json")}
    return manifest,frozen,tx_by_alias,digests


def _truths_for_transactions(transactions:Mapping[str,str],dataset_root:Path,cache_root:Path)->dict[str,list[dict[str,Any]]]:
    locations=_dataset_locations(dataset_root);index=build_verified_source_cache_index(cache_root,blockchain_chain_id=1,chain_key="eth");result={}
    for alias,transaction in sorted(transactions.items()):
        truths=[];issues=[]
        for raw in locations.get(transaction,()):
            try:label=parse_localization_ground_truth_repair_v1(raw);resolved=resolve_localization_ground_truth_repair_v1(label,index)
            except ValueError as exc:issues.append(str(exc));continue
            issues.extend(resolved.reason_codes)
            if resolved.source_file_available and resolved.normalized_source_file and resolved.line_range_available:
                truths.append({"relative_file":resolved.normalized_source_file,"function_signature":resolved.function_signature,"function_name":resolved.function_name,"function_name_unique_in_file":resolved.function_name_unique_in_file,"truth_start_line":resolved.truth_start_line,"truth_end_line":resolved.truth_end_line,"function_available":resolved.function_available,"reason_codes":list(resolved.reason_codes),"binding_role":resolved.source_binding_role,"path_recovery_applied":resolved.path_recovery_applied})
        unique={_canonical(item):item for item in truths}
        result[alias]=[{**item} for _,item in sorted(unique.items())]
        if not result[alias]:result[alias]=[{"function_available":False,"reason_codes":sorted(set(issues))}]
    return result


def _normal_file(value:Any)->str|None:
    if not isinstance(value,str):return None
    normalized=PurePosixPath(value.replace("\\","/")).as_posix().lstrip("./")
    if not normalized or PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:return None
    return normalized


def _file_hit(predicted:Any,truth:Any)->bool:
    left=_normal_file(predicted);right=_normal_file(truth)
    return bool(left and right and (left==right or left.endswith("/"+right) or right.endswith("/"+left)))


def _truth_hit(finding:Mapping[str,Any],truth:Mapping[str,Any],level:str)->bool:
    if not _file_hit(finding.get("relative_file"),truth.get("relative_file")):return False
    if level=="file":return True
    if level=="function":
        predicted=finding.get("function") or finding.get("function_signature")
        return bool(predicted==truth.get("function_signature") or (isinstance(predicted,str) and "(" not in predicted and truth.get("function_name_unique_in_file") and predicted==truth.get("function_name")))
    start,end=finding.get("predicted_line_start"),finding.get("predicted_line_end")
    return bool(isinstance(start,int) and not isinstance(start,bool) and isinstance(end,int) and not isinstance(end,bool) and start<=truth.get("truth_end_line",-1) and end>=truth.get("truth_start_line",10**18))


def evaluate_frozen_group(kind:str,manifest:Mapping[str,Any],frozen:Mapping[str,Any],truths:Mapping[str,Sequence[Mapping[str,Any]]],input_digests:Mapping[str,str])->dict[str,Any]:
    predictions={item["sample_alias"]:item for item in frozen["predictions"]};rows=[];hit_counts=Counter();denominators=Counter();valid_topk=0
    for sample in manifest["samples"]:
        alias=sample["sample_alias"];prediction=predictions[alias];topk=prediction.get("topk",()) if prediction.get("topk_valid") else ();valid_topk+=int(bool(prediction.get("topk_valid")))
        truth_rows=[item for item in truths.get(alias,()) if item.get("relative_file")]
        available={"file":bool(truth_rows),"function":any(item.get("function_available") for item in truth_rows),"line_range":bool(truth_rows)}
        hits={}
        for level in ("file","function","line_range"):
            denominators[level]+=int(available[level])
            eligible_truths=[item for item in truth_rows if level!="function" or item.get("function_available")]
            for k in (1,3,5):
                hit=available[level] and any(_truth_hit(finding,truth,"line" if level=="line_range" else level) for finding in sorted((item for item in topk if isinstance(item,Mapping)),key=lambda item:item.get("rank",999))[:k] for truth in eligible_truths)
                hits[f"{level}_hit_at_{k}"]=bool(hit);hit_counts[f"{level}_recall_at_{k}"]+=int(hit)
        reasons=sorted({reason for item in truths.get(alias,()) for reason in item.get("reason_codes",())})
        rows.append({"sample_alias":alias,"topk_valid":bool(prediction.get("topk_valid")),"topk_count":len(topk),"file_evaluable":available["file"],"function_evaluable":available["function"],"line_range_evaluable":available["line_range"],"evaluation_status":"FILE_FUNCTION_LINE_EVALUABLE" if all(available.values()) else "EVALUATION_UNAVAILABLE","reason_codes":reasons,**hits})
    total=len(rows);metrics={}
    for level in ("file","function","line_range"):
        for k in (1,3,5):
            key=f"{level}_recall_at_{k}";metrics[key]=hit_counts[key]/total if total else None;metrics["conditional_"+key]=hit_counts[key]/denominators[level] if denominators[level] else None
    return {"schema_version":"localization_truth_repair_v1_re_evaluation","dataset_kind":kind,"scientific_generalization_result":False,"evaluation_sample_count":total,"valid_topk_count":valid_topk,"fully_evaluable_count":sum(row["evaluation_status"]=="FILE_FUNCTION_LINE_EVALUABLE" for row in rows),"evaluation_unavailable_count":sum(row["evaluation_status"]=="EVALUATION_UNAVAILABLE" for row in rows),"evaluable_denominators":dict(denominators),"metrics":metrics,"input_integrity":{"manifest_sha_valid":True,"frozen_payload_sha_valid":True,"checkpoint_complete":True,"profile_sha_status":"VALIDATED" if kind=="fault10" else "NOT_APPLICABLE_HISTORICAL_V4_SCHEMA","input_file_sha256":dict(input_digests)},"samples":rows,"rpc_request_count":0,"source_network_request_count":0,"deepseek_request_count":0,"cause_read":False,"prediction_used_for_truth_resolution":False}


def combine_reports(historical:Mapping[str,Any],fault10:Mapping[str,Any])->dict[str,Any]:
    rows=[*historical["samples"],*fault10["samples"]];den=Counter();hits=Counter()
    for row in rows:
        for level in ("file","function","line_range"):
            den[level]+=int(row[f"{level}_evaluable"])
            for k in (1,3,5):hits[f"{level}_recall_at_{k}"]+=int(row[f"{level}_hit_at_{k}"])
    total=len(rows);metrics={}
    for level in ("file","function","line_range"):
        for k in (1,3,5):
            key=f"{level}_recall_at_{k}";metrics[key]=hits[key]/total if total else None;metrics["conditional_"+key]=hits[key]/den[level] if den[level] else None
    return {"schema_version":"localization_truth_repair_v1_combined25","scientific_generalization_result":False,"evaluation_sample_count":total,"fully_evaluable_count":sum(row["evaluation_status"]=="FILE_FUNCTION_LINE_EVALUABLE" for row in rows),"evaluation_unavailable_count":sum(row["evaluation_status"]=="EVALUATION_UNAVAILABLE" for row in rows),"evaluable_denominators":dict(den),"metrics":metrics,"samples":rows,"rpc_request_count":0,"source_network_request_count":0,"deepseek_request_count":0,"cause_read":False,"prediction_used_for_truth_resolution":False}


def _validate_public(value:Mapping[str,Any])->None:
    text=_canonical(value)
    if _RAW_IDENTITY.search(text):raise ValueError("TRUTH_REPAIR_OUTPUT_SENSITIVE")
    forbidden={"tx_hash","transaction_hash","address","location","cause","source_text","provider_response","prompt","rpc_url","api_key"};stack=[value]
    while stack:
        current=stack.pop()
        if isinstance(current,Mapping):
            if any(str(key).lower() in forbidden for key in current):raise ValueError("TRUTH_REPAIR_OUTPUT_SENSITIVE")
            stack.extend(current.values())
        elif isinstance(current,(list,tuple)):stack.extend(current)


def _atomic(path:Path,value:Mapping[str,Any])->None:
    _validate_public(value);path.parent.mkdir(parents=True,exist_ok=True);handle=tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,prefix=f".{path.name}.",suffix=".tmp",delete=False)
    try:
        with handle:json.dump(dict(value),handle,ensure_ascii=False,sort_keys=True,indent=2);handle.write("\n")
        Path(handle.name).replace(path)
    except BaseException:Path(handle.name).unlink(missing_ok=True);raise


def run_truth_repair(output_dir:Path=DEFAULT_OUTPUT,*,dataset_root:Path=DATASET_ROOT,cache_root:Path=CACHE_ROOT,historical_root:Path=HISTORICAL15_ROOT,fault10_root:Path=FAULT10_ROOT)->dict[str,Any]:
    if output_dir.exists():raise ValueError("TRUTH_REPAIR_OUTPUT_EXISTS")
    h_manifest,h_frozen,h_transactions,h_digests=validate_frozen_group(historical_root,"historical15",15)
    f_manifest,f_frozen,f_transactions,f_digests=validate_frozen_group(fault10_root,"fault10",10)
    # Location is first read below, after both frozen inputs and profile binding pass.
    h_truths=_truths_for_transactions(h_transactions,dataset_root,cache_root);f_truths=_truths_for_transactions(f_transactions,dataset_root,cache_root)
    historical=evaluate_frozen_group("historical15",h_manifest,h_frozen,h_truths,h_digests);fault10=evaluate_frozen_group("fault10",f_manifest,f_frozen,f_truths,f_digests);combined=combine_reports(historical,fault10)
    for root,digests in ((historical_root,h_digests),(fault10_root,f_digests)):
        if any(_file_sha(root/name)!=digest for name,digest in digests.items()):raise ValueError("TRUTH_REPAIR_INPUT_MODIFIED")
    audit={"schema_version":"localization_truth_repair_v1_audit","repair_id":"LOCALIZATION_GROUND_TRUTH_EVALUATION_REPAIR_V1","scientific_generalization_result":False,"location_read_phase":"AFTER_BOTH_FROZEN_PREDICTIONS_VALIDATED","cause_read":False,"prediction_used_for_truth_resolution":False,"historical15_fully_evaluable_before":15,"fault10_fully_evaluable_before":0,"combined25_fully_evaluable_before":15,"historical15_fully_evaluable_after":historical["fully_evaluable_count"],"fault10_fully_evaluable_after":fault10["fully_evaluable_count"],"combined25_fully_evaluable_after":combined["fully_evaluable_count"],"newly_recovered_fully_evaluable_count":combined["fully_evaluable_count"]-15,"rpc_request_count":0,"source_network_request_count":0,"deepseek_request_count":0,"historical_inputs_modified":False}
    output_dir.mkdir(parents=True,exist_ok=False);_atomic(output_dir/"truth_repair_audit.json",audit);_atomic(output_dir/"historical15_re_evaluation.json",historical);_atomic(output_dir/"fault10_re_evaluation.json",fault10);_atomic(output_dir/"combined25_re_evaluation.json",combined)
    return {"audit":audit,"historical15":historical,"fault10":fault10,"combined25":combined}


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args(argv)
    result=run_truth_repair(args.output_dir);print("truth_repair_status=COMPLETED");print("combined25_fully_evaluable="+str(result["combined25"]["fully_evaluable_count"]));print("rpc_request_count=0");print("source_network_request_count=0");print("deepseek_request_count=0");return 0


if __name__=="__main__":raise SystemExit(main())

from __future__ import annotations
import hashlib,json,os,tempfile
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any,Mapping
from .asset_continuity import observed_asset_ledger_sha256
from .asset_ledger_models import ObservedAssetLedger
from .graph_models import TransactionGraph
from .observed_asset_ledger import runtime_annotations_sha256
from .parameter_atoms import extract_parameter_atoms
from .parameter_evidence_models import ParamAtom,ParameterEvidenceConfig,ParameterEvidenceEdge,ParameterEvidenceResult
from .runtime_annotation_models import RuntimeAnnotations
from .runtime_semantics import transaction_graph_sha256
ZERO="0x"+"0"*40

def build_parameter_evidence(graph:TransactionGraph,annotations:RuntimeAnnotations,ledger:ObservedAssetLedger,causal:Mapping[str,Any],config:ParameterEvidenceConfig|None=None)->ParameterEvidenceResult:
    cfg=config or ParameterEvidenceConfig(); gh,ah,lh,ch=_validate(graph,annotations,ledger,causal)
    nodes={n.node_id:n for n in graph.nodes}; atoms=extract_parameter_atoms(graph); bynode=defaultdict(list)
    for a in atoms:bynode[a.node_id].append(a)
    freq=Counter((a.atom_type,a.normalized_value) for a in atoms); skipped=[]; totals=Counter(); raw=[]
    def skip(reason,**data):
        totals[reason]+=1
        if sum(x["reason"]==reason for x in skipped)<cfg.max_skipped_samples_per_reason: skipped.append({"reason":reason,**data})
    for call in graph.call_edges:_match_atoms(call.src,call.dst,"PARENT_CHILD_FORWARD","STRONG_ASSOCIATION",bynode,freq,nodes,cfg,raw,skip,allow_high=True)
    for order in graph.order_edges:_match_atoms(order.src,order.dst,"SIBLING_VALUE_REUSE","HYPOTHESIS",bynode,freq,nodes,cfg,raw,skip,allow_high=False)
    for fact in ledger.transfer_facts:
        if fact.status!="COMPLETE" or fact.to_address is None or fact.amount_raw is None:continue
        for dst in graph.nodes:
            distance=dst.trace_index-fact.trace_index
            if distance<=0 or not isinstance(dst.caller_address,str) or dst.caller_address.lower()!=fact.to_address["normalized"]:continue
            candidates=[]
            candidates.extend((a,{"source_kind":"ASSET_EFFECT","fact_id":fact.fact_id,"field":"amount_raw"},"STRONG_ASSOCIATION") for a in bynode[dst.node_id] if a.atom_type=="UINT" and a.normalized_value==fact.amount_raw and a.normalized_value not in cfg.ignored_uint_values)
            if fact.asset_address:candidates.extend((a,{"source_kind":"ASSET_EFFECT","fact_id":fact.fact_id,"field":"asset_address"},"HYPOTHESIS") for a in bynode[dst.node_id] if a.atom_type=="ADDRESS" and a.normalized_value==fact.asset_address["normalized"] and a.normalized_value!=ZERO)
            for atom,loc,level in candidates:
                if distance>cfg.max_execution_distance:skip("EXECUTION_DISTANCE_EXCEEDED",src=fact.node_id,dst=dst.node_id,value=atom.normalized_value);continue
                if atom.atom_type=="ADDRESS" and freq[(atom.atom_type,atom.normalized_value)]>cfg.max_global_occurrences:skip("HIGH_FREQUENCY_VALUE",src=fact.node_id,dst=dst.node_id,value=atom.normalized_value);continue
                raw.append(_raw(fact.node_id,dst.node_id,"ASSET_EFFECT_TO_LATER_INPUT",atom,loc,nodes,level,freq[(atom.atom_type,atom.normalized_value)]))
    merged=_merge(raw,nodes)
    return ParameterEvidenceResult(gh,ah,lh,ch,cfg,tuple(atoms),tuple(merged),tuple(skipped),dict(totals))

def _match_atoms(src,dst,relation,level,bynode,freq,nodes,cfg,raw,skip,allow_high):
    right=defaultdict(list)
    for a in bynode[dst]:right[(a.atom_type,a.normalized_value)].append(a)
    for a in bynode[src]:
        key=(a.atom_type,a.normalized_value)
        if _low(a,cfg):skip("ZERO_ADDRESS" if a.atom_type=="ADDRESS" else "LOW_INFORMATION_VALUE",src=src,dst=dst,value=a.normalized_value);continue
        if key not in right:continue
        if freq[key]>cfg.max_global_occurrences and not allow_high:skip("HIGH_FREQUENCY_VALUE",src=src,dst=dst,value=a.normalized_value);continue
        uncertainty=["trace does not prove source-level data dependency"]
        if freq[key]>cfg.max_global_occurrences:uncertainty.append("value has low specificity due to high global occurrence")
        raw.append(_raw(src,dst,relation,a,{"source_kind":"FUNCTION_ARGUMENT","argument_path":a.argument_path},nodes,level,freq[key],right[key],uncertainty))
def _low(a,cfg):return (a.atom_type=="UINT" and a.normalized_value in cfg.ignored_uint_values) or (a.atom_type=="ADDRESS" and cfg.ignore_zero_address and a.normalized_value==ZERO)
def _raw(src,dst,relation,atom,srcloc,nodes,level,count,dst_atoms=None,uncertainty=None):
    dst_atoms=dst_atoms or [atom]; reason={"PARENT_CHILD_FORWARD":"direct child call exactly reuses a typed parent argument","SIBLING_VALUE_REUSE":"adjacent sibling calls exactly reuse the same typed input value","ASSET_EFFECT_TO_LATER_INPUT":"asset receiver later invokes a function with the exact asset effect value"}[relation]
    basis=["EXACT_TYPED_VALUE","TEMPORAL_PRECEDENCE"]
    if relation=="ASSET_EFFECT_TO_LATER_INPUT":basis.insert(1,"ASSET_RECEIVER_EQUALS_LATER_CALLER")
    return {"src":src,"dst":dst,"relation":relation,"type":atom.atom_type,"value":atom.normalized_value,"src_locations":[srcloc],"dst_locations":[{"source_kind":"FUNCTION_ARGUMENT","argument_path":x.argument_path} for x in dst_atoms],"level":level,"reason":reason,"refs":[nodes[src].trace_path,nodes[dst].trace_path],"basis":basis,"count":count,"distance":nodes[dst].trace_index-nodes[src].trace_index,"uncertainties":uncertainty or ["trace does not prove source-level data dependency"]}
def _merge(raw,nodes):
    groups={}
    for x in raw:
        key=(x["src"],x["dst"],x["relation"],x["type"],x["value"])
        if key not in groups:groups[key]=x
        else:
            groups[key]["src_locations"]+=x["src_locations"];groups[key]["dst_locations"]+=x["dst_locations"]
    ordered=sorted(groups.values(),key=lambda x:(nodes[x["src"]].trace_index,nodes[x["dst"]].trace_index,x["relation"],x["type"],x["value"]))
    out=[]
    for i,x in enumerate(ordered,1):
        loc=lambda values:tuple(sorted({json.dumps(v,sort_keys=True):v for v in values}.values(),key=lambda v:json.dumps(v,sort_keys=True)))
        out.append(ParameterEvidenceEdge(f"param_e{i:03d}",x["relation"],x["src"],x["dst"],x["type"],x["value"],loc(x["src_locations"]),loc(x["dst_locations"]),x["level"],x["reason"],tuple(x["refs"]),tuple(x["basis"]),x["count"],x["distance"],tuple(x["uncertainties"])))
    return out
def load_candidate_causal_graph(path):
    try:p=json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:raise ValueError(f"Invalid candidate causal graph: {e}") from e
    if p.get("schema_version")!="candidate_causal_graph_v1":raise ValueError("Input schema_version must be candidate_causal_graph_v1")
    return p
def candidate_causal_graph_sha256(p):return hashlib.sha256(json.dumps(p,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _validate(g,a,l,c):
    gh,ah,lh=transaction_graph_sha256(g),runtime_annotations_sha256(a),observed_asset_ledger_sha256(l)
    if c.get("transaction_graph_sha256")!=gh or c.get("runtime_annotations_sha256")!=ah or c.get("observed_asset_ledger_sha256")!=lh:raise ValueError("Candidate causal graph hash binding mismatch")
    if c.get("edge_type_support",{}).get("PARAM")!="NOT_IMPLEMENTED":raise ValueError("Candidate causal graph PARAM support must remain NOT_IMPLEMENTED")
    return gh,ah,lh,candidate_causal_graph_sha256(c)
def serialize_parameter_evidence(r):return json.dumps(r.to_dict(),ensure_ascii=False,indent=2)+"\n"
def write_parameter_evidence(r,path):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);h=tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=target.parent,delete=False)
    try:
        with h:h.write(serialize_parameter_evidence(r))
        os.replace(h.name,target)
    except Exception:Path(h.name).unlink(missing_ok=True);raise
    return target

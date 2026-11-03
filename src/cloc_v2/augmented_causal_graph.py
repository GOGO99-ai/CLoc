"""Deterministic phase-2D-3 assembly of the complete candidate graph v2."""
from __future__ import annotations
import hashlib,json,os,tempfile
from collections import Counter,defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Mapping

EDGE_ORDER={"CALL":0,"ORDER":1,"ASSET":2,"PARAM":3,"STATE":4,"RESULT":5}
ALLOWED_LEVELS={"CALL":{"OBSERVED"},"ORDER":{"OBSERVED"},"ASSET":{"STRONG_ASSOCIATION"},"RESULT":{"STRONG_ASSOCIATION"},"PARAM":{"STRONG_ASSOCIATION","HYPOTHESIS"},"STATE":{"STRONG_ASSOCIATION","HYPOTHESIS"}}
PARAM_RELATIONS={"PARENT_CHILD_FORWARD","ASSET_EFFECT_TO_LATER_INPUT","SIBLING_VALUE_REUSE"}
STATE_RELATIONS={"SWAP_TO_RESERVE_READ","TRANSFER_TO_BALANCE_READ","BALANCE_READ_TO_TRANSFER","SAME_CONTRACT_MUTATION_TO_READ","SAME_CONTRACT_MUTATION_SEQUENCE"}
REVERSE_DEFAULTS={"edge_type_priority":["RESULT","ASSET","STATE","PARAM","CALL","ORDER"],"evidence_level_priority":["OBSERVED","STRONG_ASSOCIATION","HYPOTHESIS"],"treat_parameter_bundle_as_single_adjacency":True,"max_depth":20,"max_branching_per_node":8,"max_edges_per_state_object_per_chain":3}

@dataclass(frozen=True)
class CandidateCausalGraphV2:
    base_hash:str;parameter_hash:str;state_hash:str;upstream_bindings:dict[str,str];nodes:tuple[dict[str,Any],...];edges:tuple[dict[str,Any],...];result_seeds:tuple[dict[str,Any],...];root_results:tuple[str,...];raw_param_count:int
    SCHEMA_VERSION="candidate_causal_graph_v2"
    def to_dict(self):
        counts=Counter(e["edge_type"] for e in self.edges);levels=Counter(e["evidence_level"] for e in self.edges)
        return {"schema_version":self.SCHEMA_VERSION,"candidate_causal_graph_v1_sha256":self.base_hash,"parameter_evidence_sha256":self.parameter_hash,"state_evidence_sha256":self.state_hash,"upstream_bindings":self.upstream_bindings,"nodes":list(self.nodes),"edges":list(self.edges),"result_seeds":list(self.result_seeds),"root_results_without_edge":list(self.root_results),"edge_type_support":{k:"SUPPORTED" for k in ("CALL","ORDER","ASSET","PARAM","STATE","RESULT")},"reverse_search_defaults":REVERSE_DEFAULTS,"metadata":{"node_count":len(self.nodes),"call_edge_count":counts["CALL"],"order_edge_count":counts["ORDER"],"asset_edge_count":counts["ASSET"],"raw_param_edge_count":self.raw_param_count,"bundled_param_edge_count":counts["PARAM"],"state_edge_count":counts["STATE"],"result_edge_count":counts["RESULT"],"result_seed_count":len(self.result_seeds),"total_graph_edge_count":len(self.edges),"observed_edge_count":levels["OBSERVED"],"strong_association_edge_count":levels["STRONG_ASSOCIATION"],"hypothesis_edge_count":levels["HYPOTHESIS"],"dag_valid":True,"dangling_edge_count":0}}

def build_augmented_candidate_causal_graph(base:Mapping[str,Any],parameter:Mapping[str,Any],state:Mapping[str,Any])->CandidateCausalGraphV2:
    _schemas(base,parameter,state);base_hash=normalized_sha256(base);param_hash=normalized_sha256(parameter);state_hash=normalized_sha256(state)
    if parameter.get("candidate_causal_graph_sha256")!=base_hash or state.get("candidate_causal_graph_sha256")!=base_hash:raise ValueError("Evidence artifact is not bound to candidate_causal_graph_v1")
    for key in ("transaction_graph_sha256","runtime_annotations_sha256","observed_asset_ledger_sha256"):
        if parameter.get(key)!=base.get(key) or state.get(key)!=base.get(key):raise ValueError(f"Upstream hash mismatch: {key}")
    nodes=tuple(base.get("nodes",()));node_map={n["node_id"]:n for n in nodes}
    if len(node_map)!=len(nodes):raise ValueError("Duplicate node IDs in base graph")
    _validate_raw_evidence(parameter,state,node_map)
    param_edges=bundle_parameter_edges(parameter.get("parameter_edges",()),node_map)
    state_edges=convert_state_edges(state.get("state_edges",()),node_map)
    base_edges=[dict(e) for e in base.get("edges",())]
    all_edges=base_edges+param_edges+state_edges
    _validate_edges(all_edges,node_map)
    all_edges.sort(key=lambda e:(EDGE_ORDER[e["edge_type"]],node_map[e["src"]]["trace_index"],node_map[e["dst"]]["trace_index"],e.get("attributes",{}).get("relation_type",""),e["evidence_level"],e["edge_id"]))
    if len(all_edges)>15*len(nodes):raise ValueError("Total graph edge count exceeds 15 × node_count")
    if len(param_edges)>len(parameter.get("parameter_edges",())):raise ValueError("Bundled PARAM count exceeds raw count")
    if len(state_edges)>len(state.get("state_edges",())):raise ValueError("Converted STATE count exceeds raw count")
    _validate_dag(all_edges,node_map)
    expected={k:sum(e["edge_type"]==k for e in base_edges) for k in ("CALL","ORDER","ASSET","RESULT")}
    for kind,count in expected.items():
        if count!=base.get("metadata",{}).get(f"{kind.lower()}_edge_count"):raise ValueError(f"Base {kind} edge count mismatch")
    bindings={k:base[k] for k in ("transaction_graph_sha256","runtime_annotations_sha256","observed_asset_ledger_sha256","asset_continuity_sha256","result_candidates_sha256")}
    return CandidateCausalGraphV2(base_hash,param_hash,state_hash,bindings,nodes,tuple(all_edges),tuple(base.get("result_seeds",())),tuple(base.get("root_results_without_edge",())),len(parameter.get("parameter_edges",())))

def bundle_parameter_edges(raw_edges,node_map):
    groups=defaultdict(list)
    for e in raw_edges:
        relation=e.get("relation_type")
        if relation not in PARAM_RELATIONS:raise ValueError(f"Unknown PARAM relation_type: {relation}")
        groups[(e["src"],e["dst"],relation,e["evidence_level"])].append(e)
    ordered=sorted(groups.items(),key=lambda item:(node_map[item[0][0]]["trace_index"],node_map[item[0][1]]["trace_index"],item[0][2],item[0][3]))
    result=[]
    for index,(key,items) in enumerate(ordered,1):
        if len(items)>50:raise ValueError(f"PARAM bundle exceeds 50 matches: {key}")
        matches=[]
        for e in items:
            matches.append({"source_edge_id":e["edge_id"],"value_type":e["value_type"],"normalized_value":e["normalized_value"],"src_locations":e.get("src_locations",[]),"dst_locations":e.get("dst_locations",[]),"global_occurrence_count":e.get("global_occurrence_count"),"execution_distance":e.get("execution_distance"),"matching_basis":e.get("matching_basis",[]),"uncertainties":e.get("uncertainties",[])})
        matches.sort(key=lambda m:(m["value_type"],m["normalized_value"],m["source_edge_id"]))
        refs=sorted({m["source_edge_id"] for m in matches})
        if len(refs)!=len(matches):raise ValueError(f"Duplicate PARAM source edge in bundle: {key}")
        result.append({"edge_id":f"cg_param_e{index:03d}","edge_type":"PARAM","src":key[0],"dst":key[1],"evidence_level":key[3],"reason":"one or more exact typed values are reused in a constrained parameter context","source_stage":"parameter_evidence_v1","source_refs":refs,"attributes":{"relation_type":key[2],"match_count":len(matches),"matches":matches}})
    return result

def convert_state_edges(raw_edges,node_map):
    groups=defaultdict(list)
    for e in raw_edges:
        relation=e.get("relation_type")
        if relation not in STATE_RELATIONS:raise ValueError(f"Unknown STATE relation_type: {relation}")
        obj=e.get("state_object",{});contract=(obj.get("contract_address") or {}).get("normalized","");account=(obj.get("account_address") or {}).get("normalized","")
        groups[(e["src"],e["dst"],relation,obj.get("state_object_type"),contract,account,e["evidence_level"])].append(e)
    ordered=sorted(groups.items(),key=lambda item:(node_map[item[0][0]]["trace_index"],node_map[item[0][1]]["trace_index"],item[0][2],item[0][3] or "",item[0][4],item[0][5]))
    out=[]
    for index,(key,items) in enumerate(ordered,1):
        first=items[0];refs=sorted({e["edge_id"] for e in items})
        basis=sorted({x for e in items for x in e.get("matching_basis",())});uncertainties=sorted({x for e in items for x in e.get("uncertainties",())})
        out.append({"edge_id":f"cg_state_e{index:03d}","edge_type":"STATE","src":key[0],"dst":key[1],"evidence_level":key[6],"reason":first.get("reason",""),"source_stage":"state_evidence_v1","source_refs":refs,"attributes":{"relation_type":key[2],"state_object":first["state_object"],"src_access_mode":first.get("src_access_mode"),"dst_access_mode":first.get("dst_access_mode"),"execution_distance":first.get("execution_distance"),"matching_basis":basis,"uncertainties":uncertainties}})
    return out

def load_json_artifact(path,schema):
    try:p=json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as e:raise ValueError(f"Input does not exist: {path}") from e
    except json.JSONDecodeError as e:raise ValueError(f"Invalid JSON: {path} ({e.msg})") from e
    if not isinstance(p,dict) or p.get("schema_version")!=schema:raise ValueError(f"Input schema_version must be {schema}")
    return p
def normalized_sha256(p):return hashlib.sha256(json.dumps(p,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def serialize_augmented_graph(g):return json.dumps(g.to_dict(),ensure_ascii=False,indent=2)+"\n"
def write_augmented_graph(g,path):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);h=tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=target.parent,delete=False)
    try:
        with h:h.write(serialize_augmented_graph(g))
        os.replace(h.name,target)
    except Exception:Path(h.name).unlink(missing_ok=True);raise
    return target
def _schemas(b,p,s):
    if b.get("schema_version")!="candidate_causal_graph_v1" or p.get("schema_version")!="parameter_evidence_v1" or s.get("schema_version")!="state_evidence_v1":raise ValueError("Input schema_version mismatch")
    if b.get("edge_type_support",{}).get("PARAM")!="NOT_IMPLEMENTED" or b.get("edge_type_support",{}).get("STATE")!="NOT_IMPLEMENTED":raise ValueError("Base graph PARAM/STATE support must be NOT_IMPLEMENTED")
def _validate_raw_evidence(p,s,nodes):
    for artifact,kind,field,relations in (
        (p,"PARAM","parameter_edges",PARAM_RELATIONS),
        (s,"STATE","state_edges",STATE_RELATIONS),
    ):
        edges=artifact.get(field,())
        if not isinstance(edges,(list,tuple)):raise ValueError(f"Raw {kind} edges must be a list")
        edge_ids=set()
        for e in edges:
            if e.get("edge_type")!=kind:raise ValueError(f"Invalid raw {kind} edge_type: {e.get('edge_id')}")
            if e.get("relation_type") not in relations:raise ValueError(f"Unknown {kind} relation_type: {e.get('relation_type')}")
            if e.get("evidence_level") not in ALLOWED_LEVELS[kind]:raise ValueError(f"Invalid evidence_level for {e.get('edge_id')}")
            edge_id=e.get("edge_id")
            if not isinstance(edge_id,str) or not edge_id or edge_id in edge_ids:raise ValueError(f"Invalid or duplicate raw {kind} edge_id: {edge_id}")
            edge_ids.add(edge_id)
            if e.get("src") not in nodes or e.get("dst") not in nodes:raise ValueError(f"Dangling {kind} edge: {e.get('edge_id')}")
            if nodes[e["src"]]["trace_index"]>=nodes[e["dst"]]["trace_index"]:raise ValueError(f"Reverse-time {kind} edge: {e.get('edge_id')}")
        _validate_evidence_metadata(artifact,kind,edges)
def _validate_evidence_metadata(artifact,kind,edges):
    if kind=="PARAM":
        expected={
            "parent_child_edge_count":sum(e["relation_type"]=="PARENT_CHILD_FORWARD" for e in edges),
            "asset_amount_edge_count":sum(e["relation_type"]=="ASSET_EFFECT_TO_LATER_INPUT" and e.get("value_type")=="UINT" for e in edges),
            "asset_address_edge_count":sum(e["relation_type"]=="ASSET_EFFECT_TO_LATER_INPUT" and e.get("value_type")=="ADDRESS" for e in edges),
            "sibling_reuse_edge_count":sum(e["relation_type"]=="SIBLING_VALUE_REUSE" for e in edges),
        }
    else:
        expected={
            "swap_to_reserve_read_count":sum(e["relation_type"]=="SWAP_TO_RESERVE_READ" for e in edges),
            "transfer_to_balance_read_count":sum(e["relation_type"]=="TRANSFER_TO_BALANCE_READ" for e in edges),
            "balance_read_to_transfer_count":sum(e["relation_type"]=="BALANCE_READ_TO_TRANSFER" for e in edges),
            "same_contract_mutation_to_read_count":sum(e["relation_type"]=="SAME_CONTRACT_MUTATION_TO_READ" for e in edges),
            "same_contract_mutation_sequence_count":sum(e["relation_type"]=="SAME_CONTRACT_MUTATION_SEQUENCE" for e in edges),
        }
    expected.update({
        "strong_association_count":sum(e["evidence_level"]=="STRONG_ASSOCIATION" for e in edges),
        "hypothesis_count":sum(e["evidence_level"]=="HYPOTHESIS" for e in edges),
    })
    metadata=artifact.get("metadata")
    if not isinstance(metadata,Mapping):raise ValueError(f"Raw {kind} metadata must be an object")
    for key,value in expected.items():
        if metadata.get(key)!=value:raise ValueError(f"Raw {kind} metadata count mismatch: {key}")
def _validate_edges(edges,nodes):
    for e in edges:
        kind=e.get("edge_type")
        if kind not in EDGE_ORDER:raise ValueError(f"Unknown edge_type: {kind}")
        if e.get("evidence_level") not in ALLOWED_LEVELS[kind]:raise ValueError(f"Invalid evidence_level for {e.get('edge_id')}")
        if e.get("src")==e.get("dst"):raise ValueError(f"Self edge: {e.get('edge_id')}")
        if e.get("src") not in nodes or e.get("dst") not in nodes:raise ValueError(f"Dangling edge: {e.get('edge_id')}")
        if nodes[e["src"]]["trace_index"]>=nodes[e["dst"]]["trace_index"]:raise ValueError(f"Reverse-time edge: {e.get('edge_id')}")
def _validate_dag(edges,nodes):
    indegree={n:0 for n in nodes};adj=defaultdict(list)
    for e in edges:adj[e["src"]].append(e["dst"]);indegree[e["dst"]]+=1
    queue=[n for n,d in indegree.items() if d==0];seen=0
    while queue:
        current=queue.pop();seen+=1
        for dst in adj[current]:
            indegree[dst]-=1
            if indegree[dst]==0:queue.append(dst)
    if seen!=len(nodes):raise ValueError("Augmented graph contains a directed cycle")

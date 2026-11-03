"""Bounded deterministic reverse search from result seeds."""
from __future__ import annotations
import hashlib,json,os,tempfile
from collections import Counter,defaultdict
from dataclasses import dataclass,replace
from decimal import Decimal
from pathlib import Path
from typing import Any,Mapping
from .backbone_models import Backbone,BackboneSearchConfig,CausalBackbonesResult

VIEWS=("ASSET_FLOW","STATE_DEPENDENCY","PARAM_PROPAGATION","CONTROL_CONTEXT")
VIEW_PRIORITY={"ASSET_FLOW":("ASSET","STATE","PARAM","CALL","ORDER"),"STATE_DEPENDENCY":("STATE","PARAM","ASSET","CALL","ORDER"),"PARAM_PROPAGATION":("PARAM","STATE","ASSET","CALL","ORDER"),"CONTROL_CONTEXT":("CALL","PARAM","STATE","ASSET","ORDER")}
VIEW_REQUIRED={"ASSET_FLOW":"ASSET","STATE_DEPENDENCY":"STATE","PARAM_PROPAGATION":"PARAM","CONTROL_CONTEXT":"CALL"}
LEVEL_PRIORITY={"OBSERVED":0,"STRONG_ASSOCIATION":1,"HYPOTHESIS":2}
STOP_REASONS={"ROOT_REACHED","NO_ELIGIBLE_PREDECESSOR","MAX_DEPTH_REACHED","EXPANSION_BUDGET_REACHED"}
PRUNING_REASONS=("CYCLE_OR_REPEATED_NODE","BRANCHING_LIMIT","BEAM_WIDTH_LIMIT","ORDER_LIMIT","HYPOTHESIS_LIMIT","STATE_OBJECT_LIMIT","OTHER_RESULT_EDGE","DUPLICATE_PATH","DIVERSITY_FILTER","EXPANSION_BUDGET_REACHED","CONTROL_CALL_FALLBACK")
@dataclass(frozen=True)
class _State:
    current:str;reverse_edges:tuple[dict[str,Any],...];visited:frozenset[str];order_count:int;hypothesis_count:int;state_counts:tuple[tuple[str,int],...]

def build_causal_backbones(graph:Mapping[str,Any],config:BackboneSearchConfig|None=None)->CausalBackbonesResult:
    cfg=config or BackboneSearchConfig();_validate_config(cfg);nodes,edges,seeds,root_ids=_validate_graph(graph);incoming=defaultdict(list)
    for edge in edges:incoming[edge["dst"]].append(edge)
    pruning=Counter({reason:0 for reason in PRUNING_REASONS});expanded_counts={};selected_all=[]
    for seed in seeds:
        result_id,node_id=seed["result_id"],seed["node_id"]
        anchors=[e for e in incoming[node_id] if e["edge_type"]=="RESULT" and result_id in e.get("source_refs",())]
        root_result=node_id in root_ids
        if root_result:
            anchor=None
        elif len(anchors)!=1:raise ValueError(f"ResultSeed {result_id} must have exactly one binding RESULT edge")
        else:anchor=anchors[0]
        per_view=[]
        for view in VIEWS:
            paths,expanded=_search_view(view,seed,anchor,nodes,incoming,cfg,pruning,root_result)
            expanded_counts[f"{result_id}:{view}"]=expanded
            per_view.extend(paths)
        merged={}
        for item in per_view:
            key=item["edge_ids"]
            if key in merged:merged[key]["views"].update(item["views"])
            else:merged[key]=item
        items=sorted(merged.values(),key=lambda x:(min(VIEWS.index(v) for v in x["views"]),_final_rank(x),x["edge_ids"]))[:cfg.max_backbones_per_result]
        if not root_result and not any("CONTROL_CONTEXT" in x["views"] for x in items):raise ValueError(f"ResultSeed {result_id} has no CONTROL_CONTEXT backbone")
        selected_all.extend(items)
    backbones=[]
    for index,item in enumerate(selected_all,1):
        forward=item["edges"];node_ids=(forward[0]["src"],)+tuple(e["dst"] for e in forward) if forward else (item["result_node_id"],)
        type_counts=Counter(e["edge_type"] for e in forward);level_counts=Counter(e["evidence_level"] for e in forward)
        basis=["RESULT_EDGE_ANCHORED" if forward else "ROOT_RESULT_SEED","DETERMINISTIC_BOUNDED_REVERSE_SEARCH"]
        if any(view!="CONTROL_CONTEXT" for view in item["views"]) or type_counts["CALL"]:
            basis.append("VIEW_REQUIREMENT_SATISFIED")
        if "CONTROL_CONTEXT" in item["views"] and not type_counts["CALL"] and node_ids[0] in root_ids:
            basis.append("DIRECT_ROOT_RESULT_CONTEXT")
        backbones.append(Backbone(f"backbone_b{index:03d}",item["result_id"],item["result_node_id"],tuple(v for v in VIEWS if v in item["views"]),node_ids[0],node_ids[-1],node_ids,tuple(e["edge_id"] for e in forward),item["stop_reason"],dict(sorted(type_counts.items())),dict(sorted(level_counts.items())),nodes[node_ids[-1]]["trace_index"]-nodes[node_ids[0]]["trace_index"],level_counts["HYPOTHESIS"],type_counts["ORDER"],tuple(basis),("candidate graph edges do not prove causality",)))
    _validate_output(backbones,edges,nodes,root_ids)
    return CausalBackbonesResult(normalized_sha256(graph),cfg,tuple(backbones),dict(pruning),expanded_counts)

def _search_view(view,seed,anchor,nodes,incoming,cfg,pruning,root_result):
    if root_result:
        item={"result_id":seed["result_id"],"result_node_id":seed["node_id"],"views":{view},"edges":tuple(),"edge_ids":tuple(),"stop_reason":"ROOT_REACHED"}
        return ([item] if view=="CONTROL_CONTEXT" else []),0
    start=anchor["src"];initial=_State(start,tuple(),frozenset({seed["node_id"],start}),0,0,tuple());frontier=[initial];terminals=[];expanded=0
    while frontier:
        next_frontier=[]
        for state in frontier:
            if state.current in root_result_ids(nodes):terminals.append(_terminal(seed,view,state,anchor,"ROOT_REACHED"));continue
            if len(state.reverse_edges)+1>=cfg.max_depth:terminals.append(_terminal(seed,view,state,anchor,"MAX_DEPTH_REACHED"));continue
            if expanded>=cfg.max_expanded_states_per_result_per_view:
                terminals.append(_terminal(seed,view,state,anchor,"EXPANSION_BUDGET_REACHED"));pruning["EXPANSION_BUDGET_REACHED"]+=1;continue
            expanded+=1;candidates=[]
            for edge in incoming.get(state.current,()):
                if edge["edge_type"]=="RESULT":pruning["OTHER_RESULT_EDGE"]+=1;continue
                if edge["src"] in state.visited:pruning["CYCLE_OR_REPEATED_NODE"]+=1;continue
                if edge["edge_type"]=="ORDER" and state.order_count>=cfg.max_order_edges:pruning["ORDER_LIMIT"]+=1;continue
                if edge["evidence_level"]=="HYPOTHESIS" and state.hypothesis_count>=cfg.max_hypothesis_edges:pruning["HYPOTHESIS_LIMIT"]+=1;continue
                counts=dict(state.state_counts)
                if edge["edge_type"]=="STATE":
                    key=_state_key(edge);counts[key]=counts.get(key,0)+1
                    if counts[key]>cfg.max_state_edges_per_object:pruning["STATE_OBJECT_LIMIT"]+=1;continue
                candidates.append((edge,counts))
            candidates.sort(key=lambda pair:_edge_rank(pair[0],view,nodes))
            if len(candidates)>cfg.max_branching_per_node:pruning["BRANCHING_LIMIT"]+=len(candidates)-cfg.max_branching_per_node;candidates=candidates[:cfg.max_branching_per_node]
            if not candidates:terminals.append(_terminal(seed,view,state,anchor,"NO_ELIGIBLE_PREDECESSOR"));continue
            for edge,counts in candidates:next_frontier.append(_State(edge["src"],state.reverse_edges+(edge,),state.visited|{edge["src"]},state.order_count+(edge["edge_type"]=="ORDER"),state.hypothesis_count+(edge["evidence_level"]=="HYPOTHESIS"),tuple(sorted(counts.items()))))
        next_frontier.sort(key=lambda s:_partial_rank(s,view,nodes))
        if len(next_frontier)>cfg.beam_width:pruning["BEAM_WIDTH_LIMIT"]+=len(next_frontier)-cfg.beam_width;next_frontier=next_frontier[:cfg.beam_width]
        frontier=next_frontier
    required=VIEW_REQUIRED[view];valid=[t for t in terminals if required is None or any(e["edge_type"]==required for e in t["edges"])]
    valid.sort(key=_final_rank);selected=[]
    for candidate in valid:
        if any(candidate["edge_ids"]==x["edge_ids"] for x in selected):pruning["DUPLICATE_PATH"]+=1;continue
        if selected and not _diverse(selected[0],candidate,cfg):pruning["DIVERSITY_FILTER"]+=1;continue
        selected.append(candidate)
        if len(selected)>=cfg.max_backbones_per_view:break
    if view=="CONTROL_CONTEXT" and not selected:
        selected=[_control_fallback(seed,anchor,nodes,incoming,cfg,pruning)]
    return selected,expanded

def _control_fallback(seed,anchor,nodes,incoming,cfg,pruning):
    current=anchor["src"];visited={seed["node_id"],current};reverse=[];reason="NO_ELIGIBLE_PREDECESSOR"
    while True:
        if current in root_result_ids(nodes):reason="ROOT_REACHED";break
        if len(reverse)+1>=cfg.max_depth:reason="MAX_DEPTH_REACHED";break
        calls=sorted((e for e in incoming.get(current,()) if e["edge_type"]=="CALL"),key=lambda e:(nodes[e["dst"]]["trace_index"]-nodes[e["src"]]["trace_index"],e["edge_id"]))
        calls=[e for e in calls if e["src"] not in visited]
        if not calls:break
        edge=calls[0];reverse.append(edge);current=edge["src"];visited.add(current)
    pruning["CONTROL_CALL_FALLBACK"]+=1
    state=_State(current,tuple(reverse),frozenset(visited),0,0,tuple())
    return _terminal(seed,"CONTROL_CONTEXT",state,anchor,reason)

def _terminal(seed,view,state,anchor,reason):
    forward=tuple(reversed(state.reverse_edges))+(anchor,)
    return {"result_id":seed["result_id"],"result_node_id":seed["node_id"],"views":{view},"edges":forward,"edge_ids":tuple(e["edge_id"] for e in forward),"stop_reason":reason}
def _edge_rank(edge,view,nodes):
    priority=VIEW_PRIORITY[view].index(edge["edge_type"]);distance=edge.get("attributes",{}).get("execution_distance")
    if not isinstance(distance,int):distance=nodes[edge["dst"]]["trace_index"]-nodes[edge["src"]]["trace_index"]
    return priority,LEVEL_PRIORITY[edge["evidence_level"]],distance,edge["edge_id"]
def _partial_rank(state,view,nodes):
    types={e["edge_type"] for e in state.reverse_edges};required=VIEW_REQUIRED[view]
    return (0 if required is None or required in types else 1,-len(types&{"ASSET","STATE","PARAM"}),state.hypothesis_count,state.order_count,nodes[state.current]["trace_index"],tuple(e["edge_id"] for e in state.reverse_edges))
def _final_rank(item):
    edges=item["edges"];types={e["edge_type"] for e in edges};hyp=sum(e["evidence_level"]=="HYPOTHESIS" for e in edges);order=sum(e["edge_type"]=="ORDER" for e in edges)
    return (0 if item["stop_reason"]=="ROOT_REACHED" else 1,-len(types&{"ASSET","STATE","PARAM"}),hyp,order,-len(edges),item["edge_ids"])
def _diverse(first,second,cfg):
    a,b=set(first["edge_ids"]),set(second["edge_ids"]);j=Decimal(len(a&b))/Decimal(len(a|b)) if a|b else Decimal(1)
    semantic=lambda x:{e["edge_id"] for e in x["edges"] if e["edge_type"] in {"ASSET","STATE","PARAM"}}
    return j<=Decimal(cfg.diversity_jaccard_threshold) or bool(semantic(second)-semantic(first))
def _state_key(edge):
    obj=edge.get("attributes",{}).get("state_object",{});contract=(obj.get("contract_address") or {}).get("normalized","");account=(obj.get("account_address") or {}).get("normalized","")
    return f"{obj.get('state_object_type','')}|{contract}|{account}"
def root_result_ids(nodes):return {node_id for node_id,node in nodes.items() if node.get("parent_id") is None}

def _validate_graph(graph):
    if graph.get("schema_version")!="candidate_causal_graph_v2":raise ValueError("Input schema_version must be candidate_causal_graph_v2")
    for key in ("candidate_causal_graph_v1_sha256","parameter_evidence_sha256","state_evidence_sha256"):
        if not _is_sha256(graph.get(key)):raise ValueError(f"Invalid upstream SHA-256: {key}")
    bindings=graph.get("upstream_bindings")
    if not isinstance(bindings,dict) or not bindings:raise ValueError("Missing upstream binding: upstream_bindings")
    for key,value in bindings.items():
        if not _is_sha256(value):raise ValueError(f"Invalid upstream SHA-256: upstream_bindings.{key}")
    raw_nodes=list(graph.get("nodes",()));nodes={n["node_id"]:n for n in raw_nodes};edges=list(graph.get("edges",()));edge_ids=set()
    if len(nodes)!=len(raw_nodes):raise ValueError("Duplicate node ID")
    for e in edges:
        if e.get("edge_id") in edge_ids:raise ValueError(f"Duplicate edge ID: {e.get('edge_id')}")
        edge_ids.add(e.get("edge_id"))
        if e.get("edge_type") not in {"CALL","ORDER","ASSET","PARAM","STATE","RESULT"}:raise ValueError(f"Invalid edge type: {e.get('edge_id')}")
        if e.get("evidence_level") not in LEVEL_PRIORITY:raise ValueError(f"Invalid evidence level: {e.get('edge_id')}")
        if e.get("src") not in nodes or e.get("dst") not in nodes:raise ValueError(f"Invalid edge reference: {e.get('edge_id')}")
        if nodes[e["src"]]["trace_index"]>=nodes[e["dst"]]["trace_index"]:raise ValueError(f"Reverse-time edge: {e.get('edge_id')}")
        if e["edge_type"]=="PARAM":
            attrs=e.get("attributes",{});matches=attrs.get("matches",());refs=e.get("source_refs",())
            if attrs.get("match_count")!=len(matches) or len(matches)!=len(refs):raise ValueError(f"Invalid PARAM bundle: {e.get('edge_id')}")
        if e["edge_type"]=="STATE":_validate_state_object(e)
    _dag(edges,nodes)
    seeds=list(graph.get("result_seeds",()));root_ids=root_result_ids(nodes);incoming=defaultdict(list)
    if len({seed.get("result_id") for seed in seeds})!=len(seeds):raise ValueError("Duplicate ResultSeed ID")
    for e in edges:incoming[e["dst"]].append(e)
    for seed in seeds:
        if seed["node_id"] not in nodes:raise ValueError(f"Invalid ResultSeed node: {seed['result_id']}")
        result_incoming=[e for e in incoming[seed["node_id"]] if e["edge_type"]=="RESULT"]
        bound=[e for e in result_incoming if seed["result_id"] in e.get("source_refs",()) and e.get("attributes",{}).get("result_id")==seed["result_id"]]
        if seed["node_id"] not in root_ids and (len(result_incoming)!=1 or len(bound)!=1):raise ValueError(f"ResultSeed {seed['result_id']} must have exactly one binding RESULT edge")
    return nodes,edges,seeds,root_ids
def _validate_state_object(edge):
    obj=edge.get("attributes",{}).get("state_object")
    if not isinstance(obj,dict):raise ValueError(f"Empty STATE object: {edge.get('edge_id')}")
    typ=obj.get("state_object_type");contract=obj.get("contract_address");account=obj.get("account_address")
    if typ not in {"POOL_RESERVES","TOKEN_BALANCE","CONTRACT_STATE"} or not isinstance(contract,dict) or not contract.get("normalized"):raise ValueError(f"Invalid STATE object: {edge.get('edge_id')}")
    if typ=="TOKEN_BALANCE" and (not isinstance(account,dict) or not account.get("normalized")):raise ValueError(f"TOKEN_BALANCE lacks account: {edge.get('edge_id')}")
def _dag(edges,nodes):
    indegree={n:0 for n in nodes};adj=defaultdict(list)
    for e in edges:adj[e["src"]].append(e["dst"]);indegree[e["dst"]]+=1
    queue=[n for n,v in indegree.items() if v==0];seen=0
    while queue:
        node=queue.pop();seen+=1
        for dst in adj[node]:
            indegree[dst]-=1
            if indegree[dst]==0:queue.append(dst)
    if seen!=len(nodes):raise ValueError("Input graph is not a DAG")
def _is_sha256(value):return isinstance(value,str) and len(value)==64 and all(char in "0123456789abcdefABCDEF" for char in value)
def _validate_config(c):
    if min(c.max_depth,c.max_branching_per_node,c.beam_width,c.max_expanded_states_per_result_per_view,c.max_backbones_per_view,c.max_backbones_per_result)<1:raise ValueError("Search limits must be positive")
    if min(c.max_order_edges,c.max_hypothesis_edges,c.max_state_edges_per_object)<0:raise ValueError("Evidence limits must be non-negative")
    try:threshold=Decimal(c.diversity_jaccard_threshold)
    except Exception as e:raise ValueError("Invalid diversity threshold") from e
    if threshold<0 or threshold>1:raise ValueError("Diversity threshold must be between 0 and 1")
def _validate_output(backbones,edges,nodes,root_ids):
    by_id={e["edge_id"]:e for e in edges}
    for b in backbones:
        if len(set(b.node_ids))!=len(b.node_ids):raise ValueError(f"Repeated node in {b.backbone_id}")
        if len(b.node_ids)!=len(b.edge_ids)+1:raise ValueError(f"Invalid path shape: {b.backbone_id}")
        for i,edge_id in enumerate(b.edge_ids):
            e=by_id[edge_id]
            if e["src"]!=b.node_ids[i] or e["dst"]!=b.node_ids[i+1]:raise ValueError(f"Broken forward path: {b.backbone_id}")
        if b.result_node_id not in root_ids and (not b.edge_ids or by_id[b.edge_ids[-1]]["edge_type"]!="RESULT"):raise ValueError(f"Missing final RESULT edge: {b.backbone_id}")
        if any(by_id[e]["edge_type"]=="RESULT" for e in b.edge_ids[:-1]):raise ValueError(f"Intermediate RESULT edge: {b.backbone_id}")
        if b.stop_reason not in STOP_REASONS:raise ValueError(f"Invalid stop reason: {b.backbone_id}")
def normalized_sha256(p):return hashlib.sha256(json.dumps(p,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def load_candidate_graph_v2(path):
    try:p=json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as e:raise ValueError(f"Input does not exist: {path}") from e
    except json.JSONDecodeError as e:raise ValueError(f"Invalid JSON: {e.msg}") from e
    if not isinstance(p,dict) or p.get("schema_version")!="candidate_causal_graph_v2":raise ValueError("Input schema_version must be candidate_causal_graph_v2")
    return p
def serialize_causal_backbones(r):return json.dumps(r.to_dict(),ensure_ascii=False,indent=2)+"\n"
def write_causal_backbones(r,path):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);h=tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=target.parent,delete=False)
    try:
        with h:h.write(serialize_causal_backbones(r))
        os.replace(h.name,target)
    except Exception:Path(h.name).unlink(missing_ok=True);raise
    return target

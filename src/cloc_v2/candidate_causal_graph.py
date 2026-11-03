"""Pure phase-2D-1 assembly of already-observed and candidate evidence."""
from __future__ import annotations
import hashlib, json, os, tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping
from .asset_continuity import observed_asset_ledger_sha256
from .asset_ledger_models import ObservedAssetLedger
from .causal_graph_models import CandidateCausalGraph, EvidenceEdge, NodeRef, ResultSeed
from .graph_models import TransactionGraph
from .observed_asset_ledger import runtime_annotations_sha256
from .result_detection import asset_continuity_sha256
from .runtime_annotation_models import RuntimeAnnotations
from .runtime_semantics import transaction_graph_sha256
from .outflow_scoring import validate_serialized_scoring, PROFIT_KEYS
from .result_models import ResultDetectionConfig
from .result_detection import detect_result_candidates

ORDER = {"CALL":0,"ORDER":1,"ASSET":2,"RESULT":3}
Q = Decimal("0.0001")

def build_candidate_causal_graph(graph: TransactionGraph, annotations: RuntimeAnnotations, ledger: ObservedAssetLedger, continuity: Mapping[str,Any], results: Mapping[str,Any]) -> CandidateCausalGraph:
    gh, ah, lh, ch, rh = _validate_bindings(graph, annotations, ledger, continuity, results)
    nodes = {node.node_id:node for node in graph.nodes}; facts={fact.fact_id for fact in ledger.transfer_facts}
    _validate_transaction_graph(graph, nodes)
    _validate_results(results, nodes, facts, graph, annotations, ledger, continuity)
    refs=tuple(NodeRef(n.node_id,n.trace_index,n.trace_path,n.parent_id) for n in sorted(graph.nodes,key=lambda x:x.trace_index))
    edges=[]
    for raw in graph.call_edges + graph.order_edges:
        edges.append(EvidenceEdge("",raw.edge_type,raw.src,raw.dst,raw.evidence_level,raw.reason,raw.trace_refs,"transaction_graph_v1",(raw.edge_id,),{}))
    asset_ids=set()
    for raw in continuity.get("asset_edges",[]):
        edge_id=raw.get("edge_id"); asset_ids.add(edge_id)
        src,dst=raw.get("src_node_id"),raw.get("dst_node_id")
        if src not in nodes or dst not in nodes or nodes[src].trace_index>=nodes[dst].trace_index: raise ValueError(f"Invalid ASSET edge {edge_id}: endpoints must exist and be time-forward")
        attrs={key:raw.get(key) for key in ("src_fact_id","dst_fact_id","asset_type","asset_address","via_address","allocated_amount_raw","match_type","matching_basis","uncertainties")}
        edges.append(EvidenceEdge("","ASSET",src,dst,"STRONG_ASSOCIATION",raw.get("reason",""),tuple(raw.get("trace_refs",())),"asset_continuity_v1",(edge_id,),attrs))
    seeds=[]; roots=[]; seed_ids=set()
    for candidate in results.get("result_candidates",[]):
        seed=ResultSeed(candidate["result_id"],candidate["node_id"],candidate["result_type"],candidate["confidence"],candidate["evidence_level"],candidate["selection_reason"]); seeds.append(seed); seed_ids.add(seed.node_id)
        parent=nodes[seed.node_id].parent_id
        if parent is None: roots.append(seed.result_id)
        else: edges.append(EvidenceEdge("","RESULT",parent,seed.node_id,"STRONG_ASSOCIATION","direct parent call produced a call classified as a result candidate",(),"result_candidates_v1",(seed.result_id,),{"result_id":seed.result_id,"result_type":seed.result_type,"confidence":seed.confidence,"selection_reason":seed.selection_reason}))
    ordered=sorted(edges,key=lambda e:(ORDER[e.edge_type],nodes[e.src].trace_index,nodes[e.dst].trace_index,e.source_refs[0]))
    numbered=[]; counters={key:0 for key in ORDER}
    for edge in ordered:
        counters[edge.edge_type]+=1; numbered.append(EvidenceEdge(f"cg_{edge.edge_type.lower()}_e{counters[edge.edge_type]:03d}",edge.edge_type,edge.src,edge.dst,edge.evidence_level,edge.reason,edge.trace_refs,edge.source_stage,edge.source_refs,edge.attributes))
    _validate_assembled(tuple(numbered), nodes, seed_ids, roots)
    return CandidateCausalGraph(gh,ah,lh,ch,rh,refs,tuple(numbered),tuple(seeds),tuple(roots))

def load_result_candidates(path: str|Path)->dict[str,Any]:
    try: payload=json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc: raise ValueError(f"Result candidates do not exist: {path}") from exc
    except json.JSONDecodeError as exc: raise ValueError(f"Result candidates are not valid JSON: {exc.msg}") from exc
    if not isinstance(payload,dict) or payload.get("schema_version")!="result_candidates_v1": raise ValueError("Input schema_version must be result_candidates_v1")
    validate_serialized_scoring(payload.get("detection_config"))
    return payload

def result_candidates_sha256(payload:Mapping[str,Any])->str: return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def serialize_candidate_causal_graph(graph:CandidateCausalGraph)->str: return json.dumps(graph.to_dict(),ensure_ascii=False,indent=2)+"\n"
def write_candidate_causal_graph(graph:CandidateCausalGraph,path:str|Path)->Path:
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); handle=tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=target.parent,prefix=f".{target.name}.",suffix=".tmp",delete=False)
    try:
        with handle: handle.write(serialize_candidate_causal_graph(graph))
        os.replace(handle.name,target)
    except Exception:
        Path(handle.name).unlink(missing_ok=True); raise
    return target

def _validate_bindings(graph,annotations,ledger,continuity,results):
    if continuity.get("schema_version")!="asset_continuity_v1" or results.get("schema_version")!="result_candidates_v1": raise ValueError("Invalid input schema_version")
    gh,ah,lh=transaction_graph_sha256(graph),runtime_annotations_sha256(annotations),observed_asset_ledger_sha256(ledger); ch=asset_continuity_sha256(continuity)
    for payload,name in ((continuity,"Asset continuity"),(results,"Result candidates")):
        if payload.get("transaction_graph_sha256")!=gh or payload.get("runtime_annotations_sha256")!=ah or payload.get("observed_asset_ledger_sha256")!=lh: raise ValueError(f"{name} hash binding mismatch")
    if results.get("asset_continuity_sha256")!=ch: raise ValueError("Result candidates are not bound to supplied asset continuity")
    return gh,ah,lh,ch,result_candidates_sha256(results)

def _validate_results(results,nodes,facts,graph,annotations,ledger,continuity):
    config = results.get("detection_config")
    version = validate_serialized_scoring(config)
    if version == "legacy_independent_weights":
        return _validate_legacy_results(results,nodes,facts)
    weights = config.get("profit_weights")
    if not isinstance(weights, Mapping) or set(weights) != set(PROFIT_KEYS):
        raise ValueError("Missing complete profit weights")
    required = ("outflow_threshold", "max_candidates_per_asset", "ensure_top1_per_asset")
    if any(key not in config for key in required):
        raise ValueError("Missing grouped scoring selection configuration")
    parsed = ResultDetectionConfig(
        outflow_lambda=config["outflow_lambda"],
        **{f"profit_{key}_weight": weights[key] for key in PROFIT_KEYS},
        **{key: config[key] for key in required},
    )
    # Rebuild unrounded signals from hash-bound inputs at fixed precision.
    # This also verifies candidate selection, raw-feature audits and R_PROFIT.
    expected = detect_result_candidates(graph, annotations, ledger, continuity, parsed).to_dict()
    if results != expected:
        raise ValueError("Grouped result candidates differ from bound-input recomputation")


def _validate_legacy_results(results,nodes,facts):
    config=results.get("detection_config",{}); ow=config.get("outflow_weights",{}); pw=config.get("profit_weights",{})
    needed={"R_OUTFLOW":(["MAGNITUDE_RANK","TERMINAL_RETENTION","LATE_EXECUTION","ROOT_RECEIVER","SOURCE_ACTIVITY"],ow),"R_PROFIT":(["COMPLETE_FINANCING","AFTER_LAST_REPAYMENT","SAME_FINANCING_ASSET","TERMINAL_RETENTION","ROOT_RECEIVER"],pw)}
    for item in results.get("result_candidates",[]):
        if item.get("node_id") not in nodes: raise ValueError(f"Result candidate references missing node {item.get('node_id')}")
        if any(ref not in facts for ref in item.get("source_fact_ids",())): raise ValueError(f"Result candidate {item.get('result_id')} references missing source fact")
        typ=item.get("result_type")
        if typ not in needed: raise ValueError(f"Unsupported result type {typ}")
        names,weights=needed[typ]; signal={x.get("signal_type"):x.get("value") for x in item.get("signals",())}
        if any(name not in signal for name in names): raise ValueError(f"Result candidate {item.get('result_id')} has missing signals")
        keys=("magnitude","terminal_retention","late_execution","root_receiver","source_activity") if typ=="R_OUTFLOW" else ("complete_financing","after_last_repayment","same_financing_asset","terminal_retention","root_receiver")
        try:
            values=[Decimal(signal[name]) for name in names]
            if typ=="R_OUTFLOW": values[2]=Decimal(nodes[item["node_id"]].trace_index)/Decimal(len(nodes))
            recomputed=sum(Decimal(weights[k])*value for k,value in zip(keys,values)).quantize(Q,rounding=ROUND_HALF_UP)
        except Exception as exc: raise ValueError(f"Result candidate {item.get('result_id')} confidence cannot be recomputed") from exc
        try: reported=Decimal(item.get("confidence"))
        except Exception as exc: raise ValueError(f"Result candidate {item.get('result_id')} confidence cannot be recomputed") from exc
        if abs(recomputed-reported)>Q: raise ValueError(f"Result candidate {item.get('result_id')} confidence mismatch")

def _validate_transaction_graph(graph,nodes):
    if len(nodes)!=len(graph.nodes): raise ValueError("Transaction graph has duplicate node IDs")
    call_in={node_id:0 for node_id in nodes}
    for edge in graph.call_edges:
        if edge.src not in nodes or edge.dst not in nodes or nodes[edge.dst].parent_id!=edge.src: raise ValueError(f"Invalid CALL edge {edge.edge_id}")
        call_in[edge.dst]+=1
    if len(graph.call_edges)!=len(nodes)-1 or call_in.get(graph.root_id)!=0 or any(call_in[node_id]!=1 for node_id in nodes if node_id!=graph.root_id): raise ValueError("CALL edge invariant failed")
    for edge in graph.order_edges:
        if edge.src not in nodes or edge.dst not in nodes or nodes[edge.src].parent_id!=nodes[edge.dst].parent_id or nodes[edge.dst].sibling_order!=nodes[edge.src].sibling_order+1: raise ValueError(f"Invalid ORDER edge {edge.edge_id}")

def _validate_assembled(edges,nodes,seeds,roots):
    result_in={node:0 for node in seeds}
    for edge in edges:
        if edge.src==edge.dst or edge.src not in nodes or edge.dst not in nodes: raise ValueError(f"Dangling or self edge {edge.edge_id}")
        if edge.edge_type=="RESULT":
            if edge.dst not in seeds: raise ValueError(f"RESULT edge target is not a seed: {edge.edge_id}")
            result_in[edge.dst]+=1
        if edge.edge_type in {"PARAM","STATE"}: raise ValueError("PARAM/STATE edges are not implemented")
    if any(result_in[node]!=1 for node in seeds if node not in {nodes_id for nodes_id in []} and nodes[node].parent_id is not None): raise ValueError("Non-root ResultSeed must have one RESULT edge")

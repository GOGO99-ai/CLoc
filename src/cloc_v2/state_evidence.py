from __future__ import annotations
import json,os,tempfile
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any,Mapping
from .asset_continuity import observed_asset_ledger_sha256
from .asset_ledger_models import ObservedAssetLedger
from .graph_models import TransactionGraph
from .observed_asset_ledger import runtime_annotations_sha256
from .parameter_evidence import candidate_causal_graph_sha256
from .runtime_annotation_models import RuntimeAnnotations
from .runtime_semantics import transaction_graph_sha256
from .state_accesses import extract_state_accesses
from .state_evidence_models import StateEvidenceConfig,StateEvidenceEdge,StateEvidenceResult

def build_state_evidence(graph:TransactionGraph,annotations:RuntimeAnnotations,ledger:ObservedAssetLedger,causal:Mapping[str,Any],config:StateEvidenceConfig|None=None)->StateEvidenceResult:
 cfg=config or StateEvidenceConfig();gh,ah,lh,ch=_validate(graph,annotations,ledger,causal);nodes={n.node_id:n for n in graph.nodes};access=extract_state_accesses(graph,annotations,ledger);groups=defaultdict(list)
 for a in access:groups[a.state_object.key()].append(a)
 raw=[];skipped=[];totals=Counter()
 def skip(reason,src,dst):
  totals[reason]+=1
  if sum(x["reason"]==reason for x in skipped)<cfg.max_skipped_samples_per_reason:skipped.append({"reason":reason,"src":src,"dst":dst})
 # Pool: reads after each swap, stopping at next same-pool swap.
 for key,seq in groups.items():
  if key[0]!="POOL_RESERVES":continue
  seq=sorted(seq,key=lambda a:a.trace_index)
  for i,mut in enumerate(seq):
   if mut.action_type!="SWAP":continue
   used=0
   for later in seq[i+1:]:
    if later.action_type=="SWAP":break
    if later.action_type!="RESERVE_READ":continue
    if later.trace_index-mut.trace_index>cfg.max_execution_distance:skip("EXECUTION_DISTANCE_EXCEEDED",mut.node_id,later.node_id);continue
    if used>=cfg.max_reads_per_mutation:skip("MAX_READS_EXCEEDED",mut.node_id,later.node_id);continue
    raw.append(_edge("SWAP_TO_RESERVE_READ",mut,later,"STRONG_ASSOCIATION",("SAME_POOL_ADDRESS","MUTATION_BEFORE_READ","NO_INTERVENING_SWAP")));used+=1
 # Token/account: nearest later read and nearest prior read around each transfer mutation.
 for key,seq in groups.items():
  if key[0]!="TOKEN_BALANCE":continue
  seq=sorted(seq,key=lambda a:a.trace_index);reads=[a for a in seq if a.access_mode=="READ"]
  for mut in (a for a in seq if a.access_mode=="MUTATION_CANDIDATE"):
   after=next((r for r in reads if r.trace_index>mut.trace_index),None)
   before=next((r for r in reversed(reads) if r.trace_index<mut.trace_index),None)
   for relation,left,right,level,basis in (("TRANSFER_TO_BALANCE_READ",mut,after,"STRONG_ASSOCIATION",("SAME_TOKEN_ACCOUNT","TRANSFER_BEFORE_NEAREST_READ")),("BALANCE_READ_TO_TRANSFER",before,mut,"HYPOTHESIS",("SAME_TOKEN_ACCOUNT","NEAREST_READ_BEFORE_TRANSFER"))):
    if left is None or right is None:continue
    if right.trace_index-left.trace_index>cfg.max_execution_distance:skip("EXECUTION_DISTANCE_EXCEEDED",left.node_id,right.node_id);continue
    raw.append(_edge(relation,left,right,level,basis))
 # Contract state: adjacent accesses for mutation->read, adjacent mutations for sequence.
 specific={(x["src"].node_id,x["dst"].node_id) for x in raw if x["relation"] in {"SWAP_TO_RESERVE_READ","TRANSFER_TO_BALANCE_READ"}}
 for key,seq in groups.items():
  if key[0]!="CONTRACT_STATE":continue
  seq=sorted(seq,key=lambda a:a.trace_index)
  for left,right in zip(seq,seq[1:]):
   if left.access_mode=="MUTATION_CANDIDATE" and right.access_mode=="READ" and (left.node_id,right.node_id) not in specific:
    if right.trace_index-left.trace_index<=cfg.max_execution_distance:raw.append(_edge("SAME_CONTRACT_MUTATION_TO_READ",left,right,"HYPOTHESIS",("SAME_CONTRACT_ADDRESS","ADJACENT_CONTRACT_ACCESS")))
    else:skip("EXECUTION_DISTANCE_EXCEEDED",left.node_id,right.node_id)
  muts=[a for a in seq if a.access_mode=="MUTATION_CANDIDATE"]
  for left,right in zip(muts,muts[1:]):
   if right.trace_index-left.trace_index<=cfg.max_execution_distance:raw.append(_edge("SAME_CONTRACT_MUTATION_SEQUENCE",left,right,"HYPOTHESIS",("SAME_CONTRACT_ADDRESS","ADJACENT_MUTATION_ACCESS")))
   else:skip("EXECUTION_DISTANCE_EXCEEDED",left.node_id,right.node_id)
 edges=_merge(raw,nodes)
 if len(edges)>10*len(graph.nodes):raise ValueError("STATE edge count exceeds safety limit")
 if any(nodes[e.src].trace_index>=nodes[e.dst].trace_index or e.src==e.dst for e in edges):raise ValueError("STATE edge time/self invariant failed")
 return StateEvidenceResult(gh,ah,lh,ch,cfg,tuple(access),tuple(edges),tuple(skipped),dict(totals))
def _edge(relation,src,dst,level,basis):
 reason={"SWAP_TO_RESERVE_READ":"the same pool is swapped and its reserves are read later","TRANSFER_TO_BALANCE_READ":"a transferred token account is read afterward","BALANCE_READ_TO_TRANSFER":"a related token account is read before a later transfer","SAME_CONTRACT_MUTATION_TO_READ":"adjacent state-related actions occur on the same contract","SAME_CONTRACT_MUTATION_SEQUENCE":"adjacent mutation candidates occur on the same contract"}[relation]
 return {"relation":relation,"src":src,"dst":dst,"level":level,"reason":reason,"basis":basis}
def _merge(raw,nodes):
 groups={}
 for x in raw:
  obj=x["src"].state_object;key=(x["src"].node_id,x["dst"].node_id,x["relation"],*obj.key())
  groups[key]=x
 ordered=sorted(groups.values(),key=lambda x:(nodes[x["src"].node_id].trace_index,nodes[x["dst"].node_id].trace_index,x["relation"],x["src"].state_object.key()))
 return [StateEvidenceEdge(f"state_e{i:03d}",x["relation"],x["src"].node_id,x["dst"].node_id,x["src"].state_object,x["src"].access_mode,x["dst"].access_mode,x["level"],x["reason"],tuple(x["basis"]),x["dst"].trace_index-x["src"].trace_index,(x["src"].trace_refs[0],x["dst"].trace_refs[0]),("storage-level dependency is not available in trace",)) for i,x in enumerate(ordered,1)]
def _validate(g,a,l,c):
 if c.get("schema_version")!="candidate_causal_graph_v1":raise ValueError("Input schema_version must be candidate_causal_graph_v1")
 gh,ah,lh=transaction_graph_sha256(g),runtime_annotations_sha256(a),observed_asset_ledger_sha256(l)
 if c.get("transaction_graph_sha256")!=gh or c.get("runtime_annotations_sha256")!=ah or c.get("observed_asset_ledger_sha256")!=lh:raise ValueError("Candidate causal graph hash binding mismatch")
 if c.get("edge_type_support",{}).get("STATE")!="NOT_IMPLEMENTED":raise ValueError("Candidate causal graph STATE support must remain NOT_IMPLEMENTED")
 return gh,ah,lh,candidate_causal_graph_sha256(c)
def load_candidate_causal_graph(path):
 try:p=json.loads(Path(path).read_text(encoding="utf-8"))
 except Exception as e:raise ValueError(f"Invalid candidate causal graph: {e}") from e
 if p.get("schema_version")!="candidate_causal_graph_v1":raise ValueError("Input schema_version must be candidate_causal_graph_v1")
 return p
def serialize_state_evidence(r):return json.dumps(r.to_dict(),ensure_ascii=False,indent=2)+"\n"
def write_state_evidence(r,path):
 target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);h=tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=target.parent,delete=False)
 try:
  with h:h.write(serialize_state_evidence(r))
  os.replace(h.name,target)
 except Exception:Path(h.name).unlink(missing_ok=True);raise
 return target

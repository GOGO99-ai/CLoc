from __future__ import annotations
from .asset_ledger_models import ObservedAssetLedger
from .graph_models import TransactionGraph
from .runtime_annotation_models import RuntimeAnnotations
from .state_evidence_models import StateAccess,StateObject
MUT={"SWAP","DEPOSIT","WITHDRAW","BORROW","REWARD_CLAIM"};READ={"BALANCE_READ","RESERVE_READ","PRICE_READ"}
def extract_state_accesses(graph:TransactionGraph,annotations:RuntimeAnnotations,ledger:ObservedAssetLedger)->list[StateAccess]:
 nodes={n.node_id:n for n in graph.nodes};anns={a.node_id:a for a in annotations.node_annotations};raw=[]
 def add(node,obj,mode,action,rule,reason,fact=None,role=None,level="STRONG_ASSOCIATION"):
  raw.append(StateAccess("",node.node_id,node.trace_index,obj,mode,action,level,rule,reason,(node.trace_path,),("trace does not contain storage differences",),fact,role))
 for node in graph.nodes:
  ann=anns.get(node.node_id);actions=set(ann.action_types if ann else ())
  address=_addr(node.callee_address)
  if address and "SWAP" in actions:add(node,StateObject("POOL_RESERVES",address,None),"MUTATION_CANDIDATE","SWAP","SWAP_POOL_MUTATION_CANDIDATE","swap may affect reserves of the called pool")
  if address and "RESERVE_READ" in actions:add(node,StateObject("POOL_RESERVES",address,None),"READ","RESERVE_READ","RESERVE_READ_POOL","reserve read accesses the called pool")
  if address:
   for action in sorted(actions&MUT):add(node,StateObject("CONTRACT_STATE",address,None),"MUTATION_CANDIDATE",action,f"{action}_CONTRACT_MUTATION_CANDIDATE","runtime action may affect state of the called contract",level="HYPOTHESIS")
   for action in sorted(actions&READ):add(node,StateObject("CONTRACT_STATE",address,None),"READ",action,f"{action}_CONTRACT_READ","runtime action reads state associated with the called contract",level="HYPOTHESIS")
  if "BALANCE_READ" in actions and ann:
   token=ann.observed_operands.get("token_address");account=ann.observed_operands.get("account_address")
   if token and account:add(node,StateObject("TOKEN_BALANCE",token,account),"READ","BALANCE_READ","BALANCE_OF_TOKEN_ACCOUNT_READ","balanceOf reads an exact token and account pair")
 for fact in ledger.transfer_facts:
  if fact.status!="COMPLETE" or not fact.asset_address:continue
  node=nodes[fact.node_id]
  for account,role in ((fact.from_address,"FROM_ACCOUNT_BALANCE"),(fact.to_address,"TO_ACCOUNT_BALANCE")):
   if account:add(node,StateObject("TOKEN_BALANCE",fact.asset_address,account),"MUTATION_CANDIDATE",fact.transfer_kind,"TOKEN_TRANSFER_BALANCE_MUTATION_CANDIDATE","complete transfer may affect the exact token account balance",fact.fact_id,role)
 unique={}
 for a in raw:unique[(a.node_id,a.state_object.key(),a.access_mode,a.action_type,a.source_fact_id,a.account_role)]=a
 ordered=sorted(unique.values(),key=lambda a:(a.trace_index,a.state_object.key(),a.access_mode,a.action_type,a.account_role or ""))
 return [StateAccess(f"state_access_{i:03d}",**{k:v for k,v in a.__dict__.items() if k!="access_id"}) for i,a in enumerate(ordered,1)]
def _addr(value):return {"original":value,"normalized":value.lower()} if isinstance(value,str) and value.startswith("0x") and len(value)==42 else None

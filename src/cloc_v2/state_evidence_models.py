from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class StateObject:
    state_object_type:str; contract_address:dict[str,str]; account_address:dict[str,str]|None
    def to_dict(self):return {"state_object_type":self.state_object_type,"contract_address":self.contract_address,"account_address":self.account_address}
    def key(self):return self.state_object_type,self.contract_address["normalized"],self.account_address["normalized"] if self.account_address else ""
@dataclass(frozen=True)
class StateAccess:
    access_id:str;node_id:str;trace_index:int;state_object:StateObject;access_mode:str;action_type:str;evidence_level:str;rule_id:str;reason:str;trace_refs:tuple[str,...];uncertainties:tuple[str,...];source_fact_id:str|None=None;account_role:str|None=None
    def to_dict(self):
        d={"access_id":self.access_id,"node_id":self.node_id,"trace_index":self.trace_index,"state_object":self.state_object.to_dict(),"access_mode":self.access_mode,"action_type":self.action_type,"evidence_level":self.evidence_level,"rule_id":self.rule_id,"reason":self.reason,"trace_refs":list(self.trace_refs),"uncertainties":list(self.uncertainties)}
        if self.source_fact_id:d["source_fact_id"]=self.source_fact_id
        if self.account_role:d["account_role"]=self.account_role
        return d
@dataclass(frozen=True)
class StateEvidenceConfig:
    max_reads_per_mutation:int=10;max_execution_distance:int=300;connect_nearest_balance_read_only:bool=True;connect_adjacent_contract_access_only:bool=True;max_skipped_samples_per_reason:int=100
    def to_dict(self):return {"max_reads_per_mutation":self.max_reads_per_mutation,"max_execution_distance":self.max_execution_distance,"connect_nearest_balance_read_only":self.connect_nearest_balance_read_only,"connect_adjacent_contract_access_only":self.connect_adjacent_contract_access_only,"max_skipped_samples_per_reason":self.max_skipped_samples_per_reason}
@dataclass(frozen=True)
class StateEvidenceEdge:
    edge_id:str;relation_type:str;src:str;dst:str;state_object:StateObject;src_access_mode:str;dst_access_mode:str;evidence_level:str;reason:str;matching_basis:tuple[str,...];execution_distance:int;trace_refs:tuple[str,...];uncertainties:tuple[str,...]
    def to_dict(self):return {"edge_id":self.edge_id,"edge_type":"STATE","relation_type":self.relation_type,"src":self.src,"dst":self.dst,"state_object":self.state_object.to_dict(),"src_access_mode":self.src_access_mode,"dst_access_mode":self.dst_access_mode,"evidence_level":self.evidence_level,"reason":self.reason,"matching_basis":list(self.matching_basis),"execution_distance":self.execution_distance,"trace_refs":list(self.trace_refs),"uncertainties":list(self.uncertainties)}
@dataclass(frozen=True)
class StateEvidenceResult:
    transaction_graph_sha256:str;runtime_annotations_sha256:str;observed_asset_ledger_sha256:str;candidate_causal_graph_sha256:str;detection_config:StateEvidenceConfig;state_accesses:tuple[StateAccess,...];state_edges:tuple[StateEvidenceEdge,...];skipped_relations:tuple[dict[str,Any],...];skipped_totals:dict[str,int]
    SCHEMA_VERSION="state_evidence_v1"
    def to_dict(self):
        rel=lambda name:sum(e.relation_type==name for e in self.state_edges)
        return {"schema_version":self.SCHEMA_VERSION,"transaction_graph_sha256":self.transaction_graph_sha256,"runtime_annotations_sha256":self.runtime_annotations_sha256,"observed_asset_ledger_sha256":self.observed_asset_ledger_sha256,"candidate_causal_graph_sha256":self.candidate_causal_graph_sha256,"detection_config":self.detection_config.to_dict(),"state_accesses":[a.to_dict() for a in self.state_accesses],"state_edges":[e.to_dict() for e in self.state_edges],"skipped_relations":list(self.skipped_relations),"skipped_relation_totals":dict(sorted(self.skipped_totals.items())),"metadata":{"pool_reserve_access_count":sum(a.state_object.state_object_type=="POOL_RESERVES" for a in self.state_accesses),"token_balance_access_count":sum(a.state_object.state_object_type=="TOKEN_BALANCE" for a in self.state_accesses),"contract_state_access_count":sum(a.state_object.state_object_type=="CONTRACT_STATE" for a in self.state_accesses),"swap_to_reserve_read_count":rel("SWAP_TO_RESERVE_READ"),"transfer_to_balance_read_count":rel("TRANSFER_TO_BALANCE_READ"),"balance_read_to_transfer_count":rel("BALANCE_READ_TO_TRANSFER"),"same_contract_mutation_to_read_count":rel("SAME_CONTRACT_MUTATION_TO_READ"),"same_contract_mutation_sequence_count":rel("SAME_CONTRACT_MUTATION_SEQUENCE"),"strong_association_count":sum(e.evidence_level=="STRONG_ASSOCIATION" for e in self.state_edges),"hypothesis_count":sum(e.evidence_level=="HYPOTHESIS" for e in self.state_edges),"skipped_distance_count":self.skipped_totals.get("EXECUTION_DISTANCE_EXCEEDED",0),"skipped_different_object_count":self.skipped_totals.get("DIFFERENT_STATE_OBJECT",0)}}

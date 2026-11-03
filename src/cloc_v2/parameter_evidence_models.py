from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ParamAtom:
    node_id:str; argument_path:str; atom_type:str; raw_value:str; normalized_value:str; extraction_rule:str; trace_refs:tuple[str,...]; trace_index:int
    def to_dict(self): return {"node_id":self.node_id,"argument_path":self.argument_path,"atom_type":self.atom_type,"raw_value":self.raw_value,"normalized_value":self.normalized_value,"extraction_rule":self.extraction_rule,"trace_refs":list(self.trace_refs)}

@dataclass(frozen=True)
class ParameterEvidenceConfig:
    ignored_uint_values:tuple[str,...]=("0","1"); ignore_zero_address:bool=True; max_global_occurrences:int=20; max_execution_distance:int=200; max_skipped_samples_per_reason:int=100
    def to_dict(self): return {"ignored_uint_values":list(self.ignored_uint_values),"ignore_zero_address":self.ignore_zero_address,"max_global_occurrences":self.max_global_occurrences,"max_execution_distance":self.max_execution_distance,"max_skipped_samples_per_reason":self.max_skipped_samples_per_reason}

@dataclass(frozen=True)
class ParameterEvidenceEdge:
    edge_id:str; relation_type:str; src:str; dst:str; value_type:str; normalized_value:str; src_locations:tuple[dict[str,Any],...]; dst_locations:tuple[dict[str,Any],...]; evidence_level:str; reason:str; trace_refs:tuple[str,...]; matching_basis:tuple[str,...]; global_occurrence_count:int; execution_distance:int; uncertainties:tuple[str,...]
    def to_dict(self): return {"edge_id":self.edge_id,"edge_type":"PARAM","relation_type":self.relation_type,"src":self.src,"dst":self.dst,"value_type":self.value_type,"normalized_value":self.normalized_value,"src_locations":list(self.src_locations),"dst_locations":list(self.dst_locations),"evidence_level":self.evidence_level,"reason":self.reason,"trace_refs":list(self.trace_refs),"matching_basis":list(self.matching_basis),"global_occurrence_count":self.global_occurrence_count,"execution_distance":self.execution_distance,"uncertainties":list(self.uncertainties)}

@dataclass(frozen=True)
class ParameterEvidenceResult:
    transaction_graph_sha256:str; runtime_annotations_sha256:str; observed_asset_ledger_sha256:str; candidate_causal_graph_sha256:str; detection_config:ParameterEvidenceConfig; parameter_atoms:tuple[ParamAtom,...]; parameter_edges:tuple[ParameterEvidenceEdge,...]; skipped_matches:tuple[dict[str,Any],...]; skipped_totals:dict[str,int]
    SCHEMA_VERSION="parameter_evidence_v1"
    def to_dict(self):
        return {"schema_version":self.SCHEMA_VERSION,"transaction_graph_sha256":self.transaction_graph_sha256,"runtime_annotations_sha256":self.runtime_annotations_sha256,"observed_asset_ledger_sha256":self.observed_asset_ledger_sha256,"candidate_causal_graph_sha256":self.candidate_causal_graph_sha256,"detection_config":self.detection_config.to_dict(),"parameter_atoms":[x.to_dict() for x in self.parameter_atoms],"parameter_edges":[x.to_dict() for x in self.parameter_edges],"skipped_matches":list(self.skipped_matches),"skipped_match_totals":dict(sorted(self.skipped_totals.items())),"metadata":{"uint_atom_count":sum(x.atom_type=="UINT" for x in self.parameter_atoms),"address_atom_count":sum(x.atom_type=="ADDRESS" for x in self.parameter_atoms),"parent_child_edge_count":sum(x.relation_type=="PARENT_CHILD_FORWARD" for x in self.parameter_edges),"asset_amount_edge_count":sum(x.relation_type=="ASSET_EFFECT_TO_LATER_INPUT" and x.value_type=="UINT" for x in self.parameter_edges),"asset_address_edge_count":sum(x.relation_type=="ASSET_EFFECT_TO_LATER_INPUT" and x.value_type=="ADDRESS" for x in self.parameter_edges),"sibling_reuse_edge_count":sum(x.relation_type=="SIBLING_VALUE_REUSE" for x in self.parameter_edges),"strong_association_count":sum(x.evidence_level=="STRONG_ASSOCIATION" for x in self.parameter_edges),"hypothesis_count":sum(x.evidence_level=="HYPOTHESIS" for x in self.parameter_edges),"skipped_high_frequency_count":self.skipped_totals.get("HIGH_FREQUENCY_VALUE",0)}}

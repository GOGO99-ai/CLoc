"""Models for deterministic candidate causal backbones."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class BackboneSearchConfig:
    max_depth:int=20;max_branching_per_node:int=8;beam_width:int=32;max_expanded_states_per_result_per_view:int=500;max_backbones_per_view:int=2;max_backbones_per_result:int=8;max_order_edges:int=2;max_hypothesis_edges:int=4;max_state_edges_per_object:int=3;diversity_jaccard_threshold:str="0.85"
    def to_dict(self):return {"max_depth":self.max_depth,"max_branching_per_node":self.max_branching_per_node,"beam_width":self.beam_width,"max_expanded_states_per_result_per_view":self.max_expanded_states_per_result_per_view,"max_backbones_per_view":self.max_backbones_per_view,"max_backbones_per_result":self.max_backbones_per_result,"max_order_edges":self.max_order_edges,"max_hypothesis_edges":self.max_hypothesis_edges,"max_state_edges_per_object":self.max_state_edges_per_object,"diversity_jaccard_threshold":self.diversity_jaccard_threshold}
@dataclass(frozen=True)
class Backbone:
    backbone_id:str;result_id:str;result_node_id:str;semantic_views:tuple[str,...];start_node_id:str;end_node_id:str;node_ids:tuple[str,...];edge_ids:tuple[str,...];stop_reason:str;edge_type_counts:dict[str,int];evidence_level_counts:dict[str,int];execution_span:int;hypothesis_edge_count:int;order_edge_count:int;selection_basis:tuple[str,...];uncertainties:tuple[str,...]
    def to_dict(self):return {"backbone_id":self.backbone_id,"result_id":self.result_id,"result_node_id":self.result_node_id,"semantic_views":list(self.semantic_views),"start_node_id":self.start_node_id,"end_node_id":self.end_node_id,"node_ids":list(self.node_ids),"edge_ids":list(self.edge_ids),"stop_reason":self.stop_reason,"edge_type_counts":self.edge_type_counts,"evidence_level_counts":self.evidence_level_counts,"execution_span":self.execution_span,"hypothesis_edge_count":self.hypothesis_edge_count,"order_edge_count":self.order_edge_count,"selection_basis":list(self.selection_basis),"uncertainties":list(self.uncertainties)}
@dataclass(frozen=True)
class CausalBackbonesResult:
    candidate_causal_graph_v2_sha256:str;search_config:BackboneSearchConfig;backbones:tuple[Backbone,...];pruning_counts:dict[str,int];expanded_state_counts:dict[str,int]
    SCHEMA_VERSION="causal_backbones_v1"
    def to_dict(self):
        from collections import Counter
        views=Counter(v for b in self.backbones for v in b.semantic_views);stops=Counter(b.stop_reason for b in self.backbones);lengths=Counter(len(b.edge_ids) for b in self.backbones)
        return {"schema_version":self.SCHEMA_VERSION,"candidate_causal_graph_v2_sha256":self.candidate_causal_graph_v2_sha256,"search_config":self.search_config.to_dict(),"backbones":[b.to_dict() for b in self.backbones],"metadata":{"backbone_count":len(self.backbones),"result_count":len({b.result_id for b in self.backbones}),"semantic_view_counts":dict(sorted(views.items())),"stop_reason_counts":dict(sorted(stops.items())),"path_length_distribution":{str(k):v for k,v in sorted(lengths.items())},"pruning_counts":dict(sorted(self.pruning_counts.items())),"expanded_state_counts":dict(sorted(self.expanded_state_counts.items()))}}

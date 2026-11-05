"""Typed semantic annotations for observed calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnnotationEvidence:
    action_type: str
    evidence_level: str
    rule_id: str
    reason: str
    trace_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "evidence_level": self.evidence_level,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "trace_refs": list(self.trace_refs),
        }


@dataclass(frozen=True)
class NodeRuntimeAnnotation:
    node_id: str
    action_types: tuple[str, ...]
    observed_operands: dict[str, Any]
    annotation_evidence: tuple[AnnotationEvidence, ...]
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "action_types": list(self.action_types),
            "observed_operands": self.observed_operands,
            "annotation_evidence": [item.to_dict() for item in self.annotation_evidence],
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class AddressProfile:
    address: str
    normalized_address: str
    observed_functions: tuple[str, ...]
    observed_capabilities: tuple[str, ...]
    evidence_node_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "normalized_address": self.normalized_address,
            "observed_functions": list(self.observed_functions),
            "observed_capabilities": list(self.observed_capabilities),
            "evidence_node_ids": list(self.evidence_node_ids),
        }


@dataclass(frozen=True)
class RuntimeAnnotations:
    transaction_graph_sha256: str
    node_annotations: tuple[NodeRuntimeAnnotation, ...]
    address_profiles: tuple[AddressProfile, ...]

    SCHEMA_VERSION = "runtime_annotations_v1"

    def to_dict(self) -> dict[str, Any]:
        action_counts: dict[str, int] = {}
        for annotation in self.node_annotations:
            for action_type in annotation.action_types:
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
        sorted_action_counts = {key: action_counts[key] for key in sorted(action_counts)}
        annotated_count = sum(bool(item.action_types) for item in self.node_annotations)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "transaction_graph_sha256": self.transaction_graph_sha256,
            "node_annotations": [item.to_dict() for item in self.node_annotations],
            "address_profiles": [item.to_dict() for item in self.address_profiles],
            "metadata": {
                "annotated_node_count": annotated_count,
                "unannotated_node_count": len(self.node_annotations) - annotated_count,
                "action_type_counts": sorted_action_counts,
                "address_profile_count": len(self.address_profiles),
            },
        }

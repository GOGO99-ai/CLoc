"""Group normal references and calibrate structural distances."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable

from cloc.reference.distance import (
    ChainDistance,
    ChainFeatureRecord,
    GroupThresholdCalibration,
    MainAnchorKey,
    calibrate_group_threshold,
    compute_chain_distance,
    require_same_scoring_context,
)


@dataclass(frozen=True)
class NormalBaselineGroup:
    """One exact-main-anchor normal group and its independent calibration."""

    main_anchor: MainAnchorKey
    records: tuple[ChainFeatureRecord, ...]
    calibration: GroupThresholdCalibration

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_anchor": self.main_anchor.to_dict(),
            "records": [record.to_dict() for record in self.records],
            "calibration": self.calibration.to_dict(),
        }


@dataclass(frozen=True)
class NormalBaseline:
    """Immutable in-memory normal-reference library; persistence is external."""

    min_group_size: int
    groups: tuple[NormalBaselineGroup, ...]
    record_count: int
    group_count: int
    calibrated_group_count: int
    uncalibrated_group_count: int

    SCHEMA_VERSION = "normal_baseline_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "min_group_size": self.min_group_size,
            "groups": [group.to_dict() for group in self.groups],
            "metadata": {
                "record_count": self.record_count,
                "group_count": self.group_count,
                "calibrated_group_count": self.calibrated_group_count,
                "uncalibrated_group_count": self.uncalibrated_group_count,
            },
        }


@dataclass(frozen=True)
class TopNormalMatch:
    """One rank-stable match retained for first-stage explanation."""

    normal_record_id: str
    rank: int
    distance: ChainDistance

    def to_dict(self) -> dict[str, Any]:
        values = self.distance.to_dict()
        return {
            "normal_record_id": self.normal_record_id,
            "rank": self.rank,
            "call_distance": values["call_distance"],
            "asset_distance": values["asset_distance"],
            "chain_distance": values["chain_distance"],
        }


@dataclass(frozen=True)
class NormalBaselineQuery:
    """Single-chain quantitative result plus diagnostic Top-3 references."""

    query_record_id: str
    main_anchor: MainAnchorKey
    group_status: str
    group_threshold: Decimal | None
    top_matches: tuple[TopNormalMatch, ...]
    nearest_distance: Decimal | None
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_record_id": self.query_record_id,
            "main_anchor": self.main_anchor.to_dict(),
            "group_status": self.group_status,
            "group_threshold": _distance_text(self.group_threshold),
            "top_matches": [match.to_dict() for match in self.top_matches],
            "nearest_distance": _distance_text(self.nearest_distance),
            "status": self.status,
            "reasons": list(self.reasons),
        }


def build_normal_baseline(
    records: Iterable[ChainFeatureRecord],
    *,
    min_group_size: int,
) -> NormalBaseline:
    """Build exact-anchor normal groups, delegating each threshold to normal_distance."""
    if not isinstance(min_group_size, int) or isinstance(min_group_size, bool) or min_group_size <= 0:
        raise ValueError("min_group_size must be an explicitly supplied positive integer")
    supplied = tuple(records)
    if any(not isinstance(record, ChainFeatureRecord) for record in supplied):
        raise ValueError("records must contain only ChainFeatureRecord values")
    require_same_scoring_context(supplied)
    if len({record.record_id for record in supplied}) != len(supplied):
        raise ValueError("record_id must be globally unique in a normal baseline")

    grouped: dict[MainAnchorKey, list[ChainFeatureRecord]] = {}
    for record in supplied:
        grouped.setdefault(record.main_anchor, []).append(record)
    groups: list[NormalBaselineGroup] = []
    for anchor in sorted(grouped, key=_anchor_order):
        members = tuple(sorted(grouped[anchor], key=lambda item: item.record_id))
        groups.append(
            NormalBaselineGroup(
                main_anchor=anchor,
                records=members,
                calibration=calibrate_group_threshold(members, min_group_size=min_group_size),
            )
        )
    frozen_groups = tuple(groups)
    calibrated = sum(group.calibration.status == "CALIBRATED" for group in frozen_groups)
    return NormalBaseline(
        min_group_size=min_group_size,
        groups=frozen_groups,
        record_count=len(supplied),
        group_count=len(frozen_groups),
        calibrated_group_count=calibrated,
        uncalibrated_group_count=len(frozen_groups) - calibrated,
    )


def query_normal_baseline(
    baseline: NormalBaseline,
    query: ChainFeatureRecord,
) -> NormalBaselineQuery:
    """Compare one chain only to its exact-anchor normal group and return Top-3."""
    if not isinstance(baseline, NormalBaseline):
        raise ValueError("baseline must be a NormalBaseline")
    if not isinstance(query, ChainFeatureRecord):
        raise ValueError("query must be a ChainFeatureRecord")
    require_same_scoring_context((query, *(record for group in baseline.groups for record in group.records)))
    group = next((item for item in baseline.groups if item.main_anchor == query.main_anchor), None)
    if group is None:
        return NormalBaselineQuery(
            query_record_id=query.record_id,
            main_anchor=query.main_anchor,
            group_status="NOT_FOUND",
            group_threshold=None,
            top_matches=(),
            nearest_distance=None,
            status="UNCERTAIN",
            reasons=("NORMAL_GROUP_NOT_FOUND",),
        )

    distances = sorted(
        (compute_chain_distance(query, record) for record in group.records),
        key=lambda item: (
            item.chain_distance,
            item.call_distance,
            item.asset_distance,
            item.normal_record_id,
        ),
    )
    matches = tuple(
        TopNormalMatch(normal_record_id=distance.normal_record_id, rank=index, distance=distance)
        for index, distance in enumerate(distances[:3], start=1)
    )
    nearest = distances[0].chain_distance if distances else None
    reasons: list[str] = []
    if query.uncertainties:
        reasons.extend(query.uncertainties)
        reasons.append("QUERY_FEATURE_RECORD_UNCERTAIN")
    if group.calibration.threshold is None:
        reasons.extend(group.calibration.uncertainties)
        reasons.append("NORMAL_GROUP_THRESHOLD_UNAVAILABLE")

    if reasons:
        status = "UNCERTAIN"
    elif nearest is not None and nearest <= group.calibration.threshold:
        status = "PASS"
        reasons.append("NEAREST_DISTANCE_WITHIN_GROUP_THRESHOLD")
    else:
        status = "FAIL"
        reasons.append("NEAREST_DISTANCE_EXCEEDS_GROUP_THRESHOLD")
    return NormalBaselineQuery(
        query_record_id=query.record_id,
        main_anchor=query.main_anchor,
        group_status=group.calibration.status,
        group_threshold=group.calibration.threshold,
        top_matches=matches,
        nearest_distance=nearest,
        status=status,
        reasons=tuple(sorted(set(reasons))),
    )


def serialize_normal_baseline(baseline: NormalBaseline) -> str:
    """Return stable JSON for the normal baseline without writing a file."""
    return json.dumps(baseline.to_dict(), ensure_ascii=False, indent=2) + "\n"


def serialize_normal_baseline_query(result: NormalBaselineQuery) -> str:
    """Return stable JSON for one query result without writing a file."""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def _anchor_order(anchor: MainAnchorKey) -> tuple[int, str, str]:
    return anchor.blockchain_chain_id, anchor.effective_address, anchor.selector


def _distance_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return ChainDistance("", "", value, value, value).to_dict()["chain_distance"]


__all__ = [
    "NormalBaseline",
    "NormalBaselineGroup",
    "NormalBaselineQuery",
    "TopNormalMatch",
    "build_normal_baseline",
    "query_normal_baseline",
    "serialize_normal_baseline",
    "serialize_normal_baseline_query",
]

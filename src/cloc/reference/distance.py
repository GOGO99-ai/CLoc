"""Compare call sequences and asset behavior within business entries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from typing import Any, Iterable


_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_SELECTOR = re.compile(r"^0x[0-9a-fA-F]{8}$")
_NATIVE_ASSET = "NATIVE"
_DISTANCE_QUANTUM = Decimal("0.000000000000000001")
_QUANTILE = Decimal("0.95")
_QUANTILE_METHOD = "EMPIRICAL_NEAREST_RANK"
_CALL_SAFE_UNCERTAINTIES = frozenset({"ASSET_DIRECTION_RELATIVE_TO_ANCHOR_UNRESOLVED"})


@dataclass(frozen=True)
class MainAnchorKey:
    """The only permitted normal-group key: chain, effective address, selector."""

    blockchain_chain_id: int
    effective_address: str
    selector: str

    def __post_init__(self) -> None:
        if not isinstance(self.blockchain_chain_id, int) or isinstance(self.blockchain_chain_id, bool) or self.blockchain_chain_id <= 0:
            raise ValueError("blockchain_chain_id must be a positive integer")
        address = _normalized_address(self.effective_address)
        selector = _normalized_selector(self.selector)
        if address is None:
            raise ValueError("effective_address must be a valid address")
        if selector is None:
            raise ValueError("selector must be a valid selector")
        object.__setattr__(self, "effective_address", address)
        object.__setattr__(self, "selector", selector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockchain_chain_id": self.blockchain_chain_id,
            "effective_address": self.effective_address,
            "selector": self.selector,
        }


@dataclass(frozen=True)
class CallStep:
    """One ordered CALL identity; repeated values deliberately remain distinct."""

    effective_address: str
    selector: str

    def __post_init__(self) -> None:
        address = _normalized_address(self.effective_address)
        selector = _normalized_selector(self.selector)
        if address is None or selector is None:
            raise ValueError("CALL step requires a valid effective_address and selector")
        object.__setattr__(self, "effective_address", address)
        object.__setattr__(self, "selector", selector)

    def to_dict(self) -> dict[str, str]:
        return {"effective_address": self.effective_address, "selector": self.selector}


@dataclass(frozen=True)
class AssetBehavior:
    """An amount-free asset behavior relative to the business contract."""

    asset_identifier: str
    direction: str

    def __post_init__(self) -> None:
        asset = _normalized_asset_identifier(self.asset_identifier)
        if asset is None:
            raise ValueError("asset_identifier must be NATIVE or a valid asset address")
        if self.direction not in {"INFLOW", "OUTFLOW"}:
            raise ValueError("direction must be INFLOW or OUTFLOW")
        object.__setattr__(self, "asset_identifier", asset)

    def to_dict(self) -> dict[str, str]:
        return {"asset_identifier": self.asset_identifier, "direction": self.direction}


@dataclass(frozen=True)
class ChainFeatureRecord:
    """Prepared, label-free normal-chain features for one record."""

    record_id: str
    main_anchor: MainAnchorKey
    call_steps: tuple[CallStep, ...]
    asset_behaviors: frozenset[AssetBehavior]
    uncertainties: tuple[str, ...] = ()
    result_scoring_context: str = "legacy_independent_weights"

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("record_id must be a non-empty string")
        if not isinstance(self.main_anchor, MainAnchorKey):
            raise ValueError("main_anchor must be a MainAnchorKey")
        if not isinstance(self.result_scoring_context, str) or not self.result_scoring_context:
            raise ValueError("result_scoring_context must be a non-empty provenance identifier")
        if any(not isinstance(step, CallStep) for step in self.call_steps):
            raise ValueError("call_steps must contain only CallStep values")
        if any(not isinstance(item, AssetBehavior) for item in self.asset_behaviors):
            raise ValueError("asset_behaviors must contain only AssetBehavior values")
        if any(not isinstance(reason, str) or not reason for reason in self.uncertainties):
            raise ValueError("uncertainties must contain non-empty strings")
        object.__setattr__(self, "call_steps", tuple(self.call_steps))
        object.__setattr__(self, "asset_behaviors", frozenset(self.asset_behaviors))
        object.__setattr__(self, "uncertainties", tuple(sorted(set(self.uncertainties))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "main_anchor": self.main_anchor.to_dict(),
            "call_steps": [step.to_dict() for step in self.call_steps],
            "asset_behaviors": [item.to_dict() for item in sorted(self.asset_behaviors, key=_asset_order)],
            "uncertainties": list(self.uncertainties),
            **({"result_scoring_context": self.result_scoring_context} if self.result_scoring_context != "legacy_independent_weights" else {}),
        }


@dataclass(frozen=True)
class ChainDistance:
    """All components of the distance between two prepared chain records."""

    query_record_id: str
    normal_record_id: str
    call_distance: Decimal
    asset_distance: Decimal
    chain_distance: Decimal

    def to_dict(self) -> dict[str, str]:
        return {
            "query_record_id": self.query_record_id,
            "normal_record_id": self.normal_record_id,
            "call_distance": _distance_text(self.call_distance),
            "asset_distance": _distance_text(self.asset_distance),
            "chain_distance": _distance_text(self.chain_distance),
        }


@dataclass(frozen=True)
class NearestNeighborDistance:
    """The nearest other record within one main-anchor group."""

    record_id: str
    nearest_record_id: str
    chain_distance: Decimal

    def to_dict(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "nearest_record_id": self.nearest_record_id,
            "chain_distance": _distance_text(self.chain_distance),
        }


@dataclass(frozen=True)
class GroupThresholdCalibration:
    """One group-local nearest-neighbor 95% threshold, if safely available."""

    main_anchor: MainAnchorKey | None
    min_group_size: int
    record_count: int
    status: str
    quantile: Decimal
    quantile_method: str
    nearest_neighbor_distances: tuple[NearestNeighborDistance, ...]
    threshold: Decimal | None
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_anchor": self.main_anchor.to_dict() if self.main_anchor is not None else None,
            "min_group_size": self.min_group_size,
            "record_count": self.record_count,
            "status": self.status,
            "quantile": _distance_text(self.quantile),
            "quantile_method": self.quantile_method,
            "nearest_neighbor_distances": [item.to_dict() for item in self.nearest_neighbor_distances],
            "threshold": _distance_text(self.threshold) if self.threshold is not None else None,
            "uncertainties": list(self.uncertainties),
        }


def require_same_scoring_context(records) -> None:
    contexts = {r.result_scoring_context for r in records if isinstance(r, ChainFeatureRecord)}
    if len(contexts) > 1:
        raise ValueError("RESULT_SCORING_CONTEXT_MISMATCH: rebuild query and normal references with the same scoring configuration")


def call_distance(query: ChainFeatureRecord, normal: ChainFeatureRecord) -> Decimal:
    """Return ``1 - LCS / max(lengths)`` for complete, ordered CALL steps."""
    require_same_scoring_context((query, normal))
    query_steps = query.call_steps
    normal_steps = normal.call_steps
    if not query_steps and not normal_steps:
        return Decimal(0)
    if not query_steps or not normal_steps:
        return Decimal(1)
    return _ratio_distance(_lcs_length(query_steps, normal_steps), max(len(query_steps), len(normal_steps)))


def is_call_topology_reliable(record: ChainFeatureRecord) -> bool:
    """Allow CALL-only use when the sole uncertainty belongs to ASSET evidence."""
    if not isinstance(record, ChainFeatureRecord):
        raise ValueError("record must be a ChainFeatureRecord")
    return bool(record.call_steps) and set(record.uncertainties).issubset(_CALL_SAFE_UNCERTAINTIES)


def call_multiplicity_excess(
    query: ChainFeatureRecord,
    normal_records: Iterable[ChainFeatureRecord],
    identity: CallStep,
    *,
    min_group_size: int = 3,
) -> int | None:
    """Return occurrences above the reliable exact-anchor normal upper bound."""
    if not isinstance(query, ChainFeatureRecord) or not isinstance(identity, CallStep):
        raise ValueError("query and identity must be typed CALL features")
    if not isinstance(min_group_size, int) or isinstance(min_group_size, bool) or min_group_size < 1:
        raise ValueError("min_group_size must be a positive integer")
    if not is_call_topology_reliable(query):
        return None
    normal_records = tuple(normal_records)
    require_same_scoring_context((query, *normal_records))
    references = tuple(
        record
        for record in normal_records
        if isinstance(record, ChainFeatureRecord)
        and record.main_anchor == query.main_anchor
        and is_call_topology_reliable(record)
    )
    if len(references) < min_group_size:
        return None
    normal_upper_bound = max(sum(step == identity for step in record.call_steps) for record in references)
    observed = sum(step == identity for step in query.call_steps)
    return max(0, observed - normal_upper_bound)


def abnormal_observed_call_identities(
    query_records: Iterable[ChainFeatureRecord],
    normal_records: Iterable[ChainFeatureRecord],
    *,
    min_group_size: int = 3,
) -> tuple[CallStep, ...]:
    """Return exact CALL identities whose observed lower bound exceeds normal.

    Query records may represent bounded candidate causal chains.  Their
    unresolved calls can only lower an exact identity's observed count, so they
    are not used to reject an otherwise exact positive observation.  Reference
    records remain subject to :func:`is_call_topology_reliable`.
    """
    queries = tuple(query_records)
    references = tuple(normal_records)
    require_same_scoring_context((*queries, *references))
    if any(not isinstance(record, ChainFeatureRecord) for record in queries):
        raise ValueError("query_records must contain only ChainFeatureRecord values")
    if not isinstance(min_group_size, int) or isinstance(min_group_size, bool) or min_group_size < 1:
        raise ValueError("min_group_size must be a positive integer")

    abnormal: set[CallStep] = set()
    for query in queries:
        reliable_references = tuple(
            record
            for record in references
            if isinstance(record, ChainFeatureRecord)
            and record.main_anchor == query.main_anchor
            and is_call_topology_reliable(record)
        )
        if len(reliable_references) < min_group_size:
            continue
        for identity in set(query.call_steps):
            normal_upper_bound = max(
                sum(step == identity for step in record.call_steps)
                for record in reliable_references
            )
            if sum(step == identity for step in query.call_steps) > normal_upper_bound:
                abnormal.add(identity)
    return tuple(sorted(abnormal, key=lambda item: (item.effective_address, item.selector)))


def asset_distance(query: ChainFeatureRecord, normal: ChainFeatureRecord) -> Decimal:
    """Return Jaccard distance for amount-free, directional ASSET behavior sets."""
    require_same_scoring_context((query, normal))
    query_assets = query.asset_behaviors
    normal_assets = normal.asset_behaviors
    if not query_assets and not normal_assets:
        return Decimal(0)
    if not query_assets or not normal_assets:
        return Decimal(1)
    return _ratio_distance(len(query_assets & normal_assets), len(query_assets | normal_assets))


def compute_chain_distance(query: ChainFeatureRecord, normal: ChainFeatureRecord) -> ChainDistance:
    """Return all distance components; no feature uncertainty is discarded."""
    call_part = call_distance(query, normal)
    asset_part = asset_distance(query, normal)
    return ChainDistance(
        query_record_id=query.record_id,
        normal_record_id=normal.record_id,
        call_distance=call_part,
        asset_distance=asset_part,
        chain_distance=max(call_part, asset_part),
    )


def calibrate_group_threshold(
    records: Iterable[ChainFeatureRecord],
    *,
    min_group_size: int,
) -> GroupThresholdCalibration:
    """Calibrate one exact-anchor group using the empirical nearest-rank 95th percentile."""
    if not isinstance(min_group_size, int) or isinstance(min_group_size, bool) or min_group_size <= 0:
        raise ValueError("min_group_size must be an explicitly supplied positive integer")
    ordered = tuple(sorted(records, key=lambda item: item.record_id))
    require_same_scoring_context(ordered)
    if any(not isinstance(record, ChainFeatureRecord) for record in ordered):
        raise ValueError("records must contain only ChainFeatureRecord values")
    if len({record.record_id for record in ordered}) != len(ordered):
        raise ValueError("record_id values must be unique within a calibration group")
    anchors = {record.main_anchor for record in ordered}
    if len(anchors) > 1:
        raise ValueError("all records must share exactly one main_anchor")
    anchor = next(iter(anchors), None)
    if len(ordered) < 2 or len(ordered) < min_group_size:
        return _uncalibrated(anchor, min_group_size, ordered, "INSUFFICIENT_GROUP")
    uncertain = tuple(record for record in ordered if record.uncertainties)
    if uncertain:
        return _uncalibrated(
            anchor,
            min_group_size,
            ordered,
            "UNCERTAIN_INPUT",
            ("UNCERTAIN_FEATURE_RECORDS_PRESENT",),
        )

    nearest: list[NearestNeighborDistance] = []
    for record in ordered:
        candidates = [
            compute_chain_distance(record, other)
            for other in ordered
            if other.record_id != record.record_id
        ]
        best = min(candidates, key=lambda item: (item.chain_distance, item.normal_record_id))
        nearest.append(
            NearestNeighborDistance(
                record_id=record.record_id,
                nearest_record_id=best.normal_record_id,
                chain_distance=best.chain_distance,
            )
        )
    ordered_nearest = tuple(sorted(nearest, key=lambda item: item.record_id))
    values = sorted(item.chain_distance for item in ordered_nearest)
    rank = (95 * len(values) + 99) // 100  # ceil(0.95 * m), exact integer arithmetic.
    return GroupThresholdCalibration(
        main_anchor=anchor,
        min_group_size=min_group_size,
        record_count=len(ordered),
        status="CALIBRATED",
        quantile=_QUANTILE,
        quantile_method=_QUANTILE_METHOD,
        nearest_neighbor_distances=ordered_nearest,
        threshold=values[rank - 1],
        uncertainties=(),
    )


def serialize_group_threshold_calibration(result: GroupThresholdCalibration) -> str:
    """Return a stable JSON document without creating an artifact."""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def _uncalibrated(
    anchor: MainAnchorKey | None,
    min_group_size: int,
    records: tuple[ChainFeatureRecord, ...],
    status: str,
    extra_reasons: tuple[str, ...] = (),
) -> GroupThresholdCalibration:
    reasons = list(extra_reasons)
    if status == "INSUFFICIENT_GROUP":
        reasons.append("GROUP_SIZE_BELOW_REQUIRED_MINIMUM")
    return GroupThresholdCalibration(
        main_anchor=anchor,
        min_group_size=min_group_size,
        record_count=len(records),
        status=status,
        quantile=_QUANTILE,
        quantile_method=_QUANTILE_METHOD,
        nearest_neighbor_distances=(),
        threshold=None,
        uncertainties=tuple(sorted(set(reasons))),
    )


def _lcs_length(left: tuple[CallStep, ...], right: tuple[CallStep, ...]) -> int:
    """Dynamic-programming LCS over complete CallStep values, not sets."""
    previous = [0] * (len(right) + 1)
    for left_step in left:
        current = [0]
        for index, right_step in enumerate(right, start=1):
            if left_step == right_step:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _ratio_distance(common: int, total: int) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        value = Decimal(1) - (Decimal(common) / Decimal(total))
        return value.quantize(_DISTANCE_QUANTUM, rounding=ROUND_HALF_EVEN)


def _normalized_address(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and _ADDRESS.fullmatch(value) else None


def _normalized_selector(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and _SELECTOR.fullmatch(value) else None


def _normalized_asset_identifier(value: Any) -> str | None:
    if value == _NATIVE_ASSET:
        return _NATIVE_ASSET
    return _normalized_address(value)


def _asset_order(item: AssetBehavior) -> tuple[str, str]:
    return item.asset_identifier, item.direction


def _distance_text(value: Decimal) -> str:
    return format(value.quantize(_DISTANCE_QUANTUM, rounding=ROUND_HALF_EVEN), "f")


__all__ = [
    "AssetBehavior",
    "CallStep",
    "ChainDistance",
    "ChainFeatureRecord",
    "GroupThresholdCalibration",
    "MainAnchorKey",
    "NearestNeighborDistance",
    "asset_distance",
    "abnormal_observed_call_identities",
    "call_multiplicity_excess",
    "call_distance",
    "calibrate_group_threshold",
    "compute_chain_distance",
    "is_call_topology_reliable",
    "serialize_group_threshold_calibration",
]

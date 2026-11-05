"""Score and identify asset outflow candidates deterministically."""
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any, Mapping
import hashlib
import json

SCORING_CONTEXT = Context(prec=50, rounding=ROUND_HALF_UP)
OUTFLOW_KEYS = ("magnitude", "terminal_retention", "late_execution", "root_receiver", "source_activity")
PROFIT_KEYS = ("complete_financing", "after_last_repayment", "same_financing_asset", "terminal_retention", "root_receiver")


def unit_decimal(value: Any, name: str) -> Decimal:
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
            raise ValueError(name)
        result = Decimal(value)
        if not result.is_finite() or not 0 <= result <= 1:
            raise ValueError(name)
        return result
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number between 0 and 1") from exc


def grouped_outflow_score(values, outflow_lambda):
    lam = unit_decimal(outflow_lambda, "outflow_lambda")
    if len(values) != 5:
        raise ValueError("R_OUTFLOW requires five signals")
    with localcontext(SCORING_CONTEXT):
        values = tuple(unit_decimal(v, "outflow signal") for v in values)
        asset = (values[0] + values[1]) / 2
        context = (values[2] + values[3] + values[4]) / 3
        score = lam * asset + (1 - lam) * context
        audit = {
            "raw_signals": dict(zip(OUTFLOW_KEYS, map(str, values))),
            "asset_group_mean": str(asset), "context_group_mean": str(context),
            "unrounded_score": str(score),
        }
        return score, audit


def validate_serialized_scoring(config: Mapping[str, Any]) -> str:
    if not isinstance(config, Mapping):
        raise ValueError("Missing detection configuration")
    version = config.get("outflow_scoring_version")
    if version is None or version == "legacy_independent_weights":
        if any(k in config for k in ("outflow_lambda", "outflow_feature_semantics_version", "outflow_decimal_precision", "outflow_rounding")):
            raise ValueError("Conflicting legacy and grouped scoring configuration")
        weights = config.get("outflow_weights")
        if not isinstance(weights, Mapping) or set(weights) != set(OUTFLOW_KEYS):
            raise ValueError("Legacy scoring requires complete outflow_weights")
        parsed = [unit_decimal(weights[k], k) for k in OUTFLOW_KEYS]
        if sum(parsed, Decimal(0)) != 1:
            raise ValueError("Legacy outflow weights must sum to 1")
        return "legacy_independent_weights"
    if version != "grouped_mean_v1":
        raise ValueError(f"Unknown outflow scoring version: {version}")
    if "outflow_weights" in config:
        raise ValueError("Conflicting legacy and grouped scoring configuration")
    unit_decimal(config.get("outflow_lambda"), "outflow_lambda")
    if (config.get("outflow_feature_semantics_version"), config.get("outflow_decimal_precision"), config.get("outflow_rounding")) != ("runtime_signals_v1", 50, "ROUND_HALF_UP"):
        raise ValueError("Unsupported outflow feature semantics or precision contract")
    return version


def scoring_context_id(config) -> str:
    version = validate_serialized_scoring(config)
    # Canonical lambda text prevents 0.65 and 0.650 from becoming two modes.
    normalized = dict(config)
    if version == "grouped_mean_v1":
        with localcontext(SCORING_CONTEXT):
            normalized["outflow_lambda"] = str(unit_decimal(config["outflow_lambda"], "outflow_lambda").normalize())
    return version + ":" + hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

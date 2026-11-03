"""Privacy-preserving v3 view of unchanged phase-1 compact scoring packets."""
from __future__ import annotations

import copy
import re

from .phase1_scoring import canonical_sha256

ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")
_DROP_EXACT = {"arguments_raw", "params_raw", "raw_params", "raw_calldata", "calldata", "calldata_blob", "input_data"}
_RAW_ALLOWED = {"amount_raw", "allocated_amount_raw"}


def _key_is_dropped(key: str) -> bool:
    lowered = key.lower()
    return "original" in lowered or lowered in _DROP_EXACT or ("raw" in lowered and lowered not in _RAW_ALLOWED)


def _addresses(value) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_addresses(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_addresses(item) for item in value)) if value else set()
    return {match.group(0).lower() for match in ADDRESS.finditer(value)} if isinstance(value, str) else set()


def _aliases(packets_v2: dict) -> dict[str, str]:
    existing = {str(address).lower(): alias for alias, address in packets_v2.get("resolver_metadata", {}).get("address_dictionary", {}).items()}
    all_addresses = _addresses([batch["llm_payload"] for batch in packets_v2.get("batches", [])])
    next_index = max((int(alias[1:]) for alias in existing.values() if re.fullmatch(r"a\d+", alias)), default=0) + 1
    for address in sorted(all_addresses - set(existing)):
        existing[address] = f"a{next_index:03d}"; next_index += 1
    return existing


def _sanitize(value, aliases: dict[str, str]):
    if isinstance(value, dict):
        return {key: _sanitize(item, aliases) for key, item in value.items() if not _key_is_dropped(str(key))}
    if isinstance(value, list): return [_sanitize(item, aliases) for item in value]
    if isinstance(value, str): return ADDRESS.sub(lambda match: aliases[match.group(0).lower()], value)
    return value


def build_phase1_scoring_packets_v3(packets_v2: dict) -> dict:
    if packets_v2.get("schema_version") != "phase1_scoring_packets_v2": raise ValueError("expected phase1_scoring_packets_v2")
    aliases = _aliases(packets_v2)
    batches = []
    for batch in packets_v2["batches"]:
        sanitized = _sanitize(batch["llm_payload"], aliases)
        batches.append({**{key: copy.deepcopy(value) for key, value in batch.items() if key != "llm_payload"}, "llm_payload": sanitized})
    resolver = copy.deepcopy(packets_v2.get("resolver_metadata", {}))
    resolver["address_dictionary"] = {alias: address for address, alias in sorted(aliases.items(), key=lambda item: item[1])}
    return {"schema_version": "phase1_scoring_packets_v3", "source_phase1_scoring_packets_v2_sha256": canonical_sha256(packets_v2), "input_sha256": copy.deepcopy(packets_v2.get("input_sha256", {})), "resolver_metadata": resolver, "batch_config": copy.deepcopy(packets_v2.get("batch_config", {})), "batches": batches}


def audit_llm_payload(packets: dict) -> dict:
    rendered = __import__("json").dumps([batch["llm_payload"] for batch in packets.get("batches", [])], ensure_ascii=False, separators=(",", ":"))
    return {"evm_address_count": len(ADDRESS.findall(rendered)), "original_field_count": rendered.count("original"), "arguments_raw_count": rendered.count("arguments_raw"), "resolver_metadata_count": rendered.count("resolver_metadata")}

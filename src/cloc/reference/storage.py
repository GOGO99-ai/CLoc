"""Build and load a portable normal-reference library from curated traces."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from cloc.reference.features import extract_normal_features
from cloc.reference.library import build_normal_baseline
from cloc.reference.loader import load_formal_normal_baseline
from cloc.reference.observations import extract_amount_observations


SCHEMA_VERSION = "normal_reference_store_v2"


def trace_digest(trace):
    frame = trace.get("result", trace)
    return hashlib.sha256(json.dumps(frame, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ReferenceLibrary:
    baseline: object
    normal_record_transaction_ids: dict
    amount_observations: dict = field(default_factory=dict)
    excluded_query_samples: int = 0


def _assemble(samples, exclude_trace=None):
    # Import locally: stage assembly also uses this storage adapter.
    from cloc.pipeline import build_evidence

    records, identities, observations = [], {}, {}
    seen_ids, seen_traces = set(), set()
    excluded = 0
    if not isinstance(samples, list):
        raise ValueError("REFERENCE_SAMPLES_INVALID")
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"id", "chain", "trace", "trace_sha256", "normality_source"}:
            raise ValueError("REFERENCE_SAMPLE_FIELDS_INVALID")
        name, chain, trace = sample["id"], sample["chain"], sample["trace"]
        if not isinstance(name, str) or not name or name in seen_ids:
            raise ValueError("REFERENCE_SAMPLE_ID_INVALID")
        if chain not in {"eth", "bsc"} or not isinstance(trace, dict):
            raise ValueError("REFERENCE_SAMPLE_TRACE_INVALID")
        if not isinstance(sample["normality_source"], str) or not sample["normality_source"].strip():
            raise ValueError("REFERENCE_NORMALITY_SOURCE_REQUIRED")
        digest = trace_digest(trace)
        identity = (chain, digest)
        if digest != sample["trace_sha256"] or identity in seen_traces:
            raise ValueError("REFERENCE_TRACE_HASH_OR_DUPLICATE_INVALID")
        seen_ids.add(name)
        seen_traces.add(identity)
        if identity == exclude_trace:
            excluded += 1
            continue
        bundle = build_evidence(trace, chain)
        features = extract_normal_features(bundle.graph, bundle.chains, bundle.ledger, transaction_record_id=name)
        clean = tuple(record for record in features.records if not record.uncertainties)
        amounts = extract_amount_observations(bundle.graph, bundle.chains, bundle.ledger, features)
        for record in clean:
            if record.record_id in identities:
                raise ValueError("REFERENCE_RECORD_ID_DUPLICATE")
            identities[record.record_id] = chain + ":" + digest
            observations[record.record_id] = amounts.get(record.record_id, [])
        records.extend(clean)
    return ReferenceLibrary(build_normal_baseline(records, min_group_size=3), identities, observations, excluded)


def build_reference_library(manifest_path, output_path):
    """The manifest lists known-normal traces; transaction success alone is not a label."""
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or set(manifest) != {"samples"} or not isinstance(manifest["samples"], list):
        raise ValueError("NORMAL_MANIFEST_INVALID")
    samples = []
    for item in manifest["samples"]:
        if not isinstance(item, dict) or set(item) != {"id", "chain", "trace", "normality_source"}:
            raise ValueError("NORMAL_MANIFEST_SAMPLE_INVALID")
        trace_path = manifest_path.parent / item["trace"]
        trace = json.loads(trace_path.read_text(encoding="utf-8-sig"))
        if not isinstance(trace, dict):
            raise ValueError("NORMAL_TRACE_INVALID")
        samples.append({**item, "trace": trace, "trace_sha256": trace_digest(trace)})
    loaded = _assemble(samples)
    if not loaded.baseline.record_count:
        raise ValueError("NORMAL_LIBRARY_HAS_NO_CLEAN_RECORDS")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": SCHEMA_VERSION, "samples": samples}
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"sample_count": len(samples), "record_count": loaded.baseline.record_count,
            "group_count": loaded.baseline.group_count, "calibrated_group_count": loaded.baseline.calibrated_group_count}


def load_reference_library(path, *, query_chain=None, query_trace=None):
    path = Path(path)
    if path.is_dir():
        legacy = load_formal_normal_baseline(path)
        # Old stores contain structural features only; never fabricate amounts.
        return ReferenceLibrary(legacy.baseline, dict(legacy.normal_record_transaction_ids))
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "samples"} or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("NORMAL_LIBRARY_VERSION_INVALID")
    exclude = (query_chain, trace_digest(query_trace)) if query_trace is not None else None
    return _assemble(value["samples"], exclude)

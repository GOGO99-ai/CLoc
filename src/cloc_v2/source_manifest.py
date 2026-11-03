from __future__ import annotations
from .source_cache import atomic_json
def write_run_manifest(path,chain,tx_hash,results,inputs=None):
 payload={'schema_version':'source_acquisition_v1','chain':chain,'tx_hash':tx_hash,'inputs':inputs or {},'addresses':[r.to_dict() for r in results]}
 atomic_json(path,payload);return payload

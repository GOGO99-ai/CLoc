from __future__ import annotations
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];SPEC=importlib.util.spec_from_file_location("gt_audit",ROOT/"scripts"/"audit_localization_ground_truth.py");assert SPEC and SPEC.loader;mod=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=mod;SPEC.loader.exec_module(mod)

class AuditTests(unittest.TestCase):
    def test_frozen_sha_must_validate(self):
        self.assertFalse(mod._frozen_valid({"schema_version":"x","kind":"phase1","locked_manifest_sha256":"x","phase2_queue":[],"predictions":[],"payload_sha256":"bad"}))

    def test_dataset_reader_uses_fault_hash_and_location_shapes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"x.json").write_text(json.dumps({"platform":"Ethereum","cause":"ignored","fault":{"transaction_hash":[["0x"+"2"*64]],"location":["0x"+"1"*40+"#A.sol#0:1"]}}),encoding="utf-8")
            rows=mod._dataset_records(root);self.assertEqual(len(rows),1);self.assertEqual(len(rows[0][0]),1);self.assertEqual(len(rows[0][1]),1)

    def test_write_audit_is_atomic_and_aggregate_only(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"audit.json";value={"schema_version":"x","scientific_generalization_result":False,"exact_line_localization_supported":False,"provider_free_text_used":False,"dataset":{"count":1}};mod.write_audit(path,value);self.assertEqual(json.loads(path.read_text(encoding="utf-8")),value)


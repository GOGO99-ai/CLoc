from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import re_evaluate_localization_truth_repair_v1 as repair


class LocalizationTruthRepairV1Tests(unittest.TestCase):
    def test_file_function_and_line_metrics_use_separate_denominators(self):
        manifest={"samples":[{"sample_alias":"a"},{"sample_alias":"b"}]};frozen={"predictions":[{"sample_alias":"a","topk_valid":True,"topk":[{"rank":1,"relative_file":"src/A.sol","function":"f()","predicted_line_start":3,"predicted_line_end":3}]},{"sample_alias":"b","topk_valid":False,"topk":[]}]}
        truths={"a":[{"relative_file":"src/A.sol","function_signature":"f()","function_name":"f","function_name_unique_in_file":True,"truth_start_line":3,"truth_end_line":3,"function_available":True,"reason_codes":[]}],"b":[{"function_available":False,"reason_codes":["GROUND_TRUTH_SOURCE_CACHE_MISSING"]}]}
        result=repair.evaluate_frozen_group("x",manifest,frozen,truths,{})
        self.assertEqual(result["fully_evaluable_count"],1);self.assertEqual(result["evaluable_denominators"],{"file":1,"function":1,"line_range":1})
        self.assertEqual(result["metrics"]["file_recall_at_1"],.5);self.assertEqual(result["metrics"]["conditional_function_recall_at_5"],1.0)

    def test_line_hit_never_uses_whole_slice_as_prediction(self):
        manifest={"samples":[{"sample_alias":"a"}]};frozen={"predictions":[{"sample_alias":"a","topk_valid":True,"topk":[{"rank":1,"relative_file":"A.sol","function":"f()","slice_start_line":1,"slice_end_line":100}]}]};truths={"a":[{"relative_file":"A.sol","function_signature":"f()","function_name":"f","function_name_unique_in_file":True,"truth_start_line":20,"truth_end_line":20,"function_available":True,"reason_codes":[]}]}
        result=repair.evaluate_frozen_group("x",manifest,frozen,truths,{})
        self.assertEqual(result["metrics"]["file_recall_at_1"],1.0);self.assertEqual(result["metrics"]["line_range_recall_at_5"],0.0)

    def test_frozen_validation_occurs_before_truth_resolver(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"locked_inference_manifest.json").write_text("{}",encoding="utf-8");(root/"frozen_predictions.json").write_text("{}",encoding="utf-8");(root/"checkpoint.json").write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):repair.validate_frozen_group(root,"historical15",15)

    def test_combined_metrics_keep_fixed_25_style_denominator(self):
        base=lambda alias,hit:{"sample_alias":alias,"evaluation_status":"FILE_FUNCTION_LINE_EVALUABLE","file_evaluable":True,"function_evaluable":True,"line_range_evaluable":True,**{f"{level}_hit_at_{k}":hit for level in ("file","function","line_range") for k in (1,3,5)}}
        h={"samples":[base("a",True)]};f={"samples":[base("b",False)]};result=repair.combine_reports(h,f)
        self.assertEqual(result["evaluation_sample_count"],2);self.assertEqual(result["metrics"]["function_recall_at_5"],.5)

    def test_public_output_rejects_identity_truth_and_provider_fields(self):
        for value in ({"tx_hash":"0x"+"1"*64},{"location":"x"},{"address":"x"},{"provider_response":"x"},{"cause":"x"}):
            with self.assertRaises(ValueError):repair._validate_public(value)
        repair._validate_public({"sample_alias":"safe_001","payload_sha256":"a"*64})

    def test_atomic_output_is_sanitized(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"x.json";repair._atomic(path,{"schema_version":"x","sample_alias":"safe"});self.assertTrue(path.exists())

    def test_module_does_not_import_inference_or_network_clients(self):
        text=Path(repair.__file__).read_text(encoding="utf-8")
        self.assertNotIn("phase2_localization",text);self.assertNotIn("DeepSeek",text);self.assertNotIn("JsonRpcClient",text);self.assertNotIn('.get("cause")',text)


if __name__=="__main__":unittest.main()

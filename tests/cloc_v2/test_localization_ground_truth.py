from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from src.cloc_v2.localization_ground_truth import parse_localization_ground_truth,parse_localization_ground_truth_repair_v1,resolve_localization_ground_truth,resolve_localization_ground_truth_repair_v1,evaluate_localization_topk,serialize_localization_ground_truth_summary
from src.cloc_v2.verified_source_cache_index import build_verified_source_cache_index

ADDRESS="0x"+"1"*40
SOURCE="contract C {\n  function pay(uint256 x) external {\n    uint256 中文 = x;\n  }\n}\n"

def cache(root:Path,files:dict[str,str]):
    d=root/"eth"/ADDRESS;entries=[]
    for name,text in files.items():
        p=d/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding="utf-8");entries.append({"path":name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    (d/"manifest.json").write_text(json.dumps({"status":"verified","verified":True,"requested_address":ADDRESS,"resolved_address":ADDRESS,"files":entries,"schema_version":"source_manifest_v1","proxy":{}}),encoding="utf-8")
    return build_verified_source_cache_index(root,blockchain_chain_id=1,chain_key="eth")

class GroundTruthTests(unittest.TestCase):
    def test_parse_valid_empty_and_invalid(self):
        label=parse_localization_ground_truth(f"{ADDRESS}#src/A.sol#3:4");self.assertEqual(label.source_file,"src/A.sol")
        empty=parse_localization_ground_truth(f"{ADDRESS}##0:1");self.assertIsNone(empty.source_file);self.assertIn("GROUND_TRUTH_FILE_NOT_REPORTED",empty.reason_codes)
        for value in ("bad",f"{ADDRESS}#../A.sol#0:1",f"{ADDRESS}#A.sol#0:0"):
            with self.assertRaises(ValueError):parse_localization_ground_truth(value)

    def test_utf8_offset_and_function_resolution(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);index=cache(root,{"sources/A.sol":SOURCE});raw=SOURCE.encode();needle="uint256 中文".encode();offset=raw.index(needle)
            result=resolve_localization_ground_truth(parse_localization_ground_truth(f"{ADDRESS}#A.sol#{offset}:{len(needle)}"),index)
            self.assertTrue(result.source_file_available);self.assertTrue(result.function_available);self.assertEqual(result.function_signature,"pay(uint256)");self.assertEqual((result.truth_start_line,result.truth_end_line),(3,3))

    def test_multibyte_middle_and_out_of_range_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"sources/A.sol":SOURCE});raw=SOURCE.encode();middle=raw.index("中".encode())+1
            bad=resolve_localization_ground_truth(parse_localization_ground_truth(f"{ADDRESS}#A.sol#{middle}:1"),index);self.assertIn("GROUND_TRUTH_OFFSET_ENCODING_INVALID",bad.reason_codes)
            outside=resolve_localization_ground_truth(parse_localization_ground_truth(f"{ADDRESS}#A.sol#9999:1"),index);self.assertIn("GROUND_TRUTH_OFFSET_OUT_OF_RANGE",outside.reason_codes)

    def test_empty_file_unique_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"sources/A.sol":SOURCE});result=resolve_localization_ground_truth(parse_localization_ground_truth(f"{ADDRESS}##0:1"),index);self.assertTrue(result.source_file_available)
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"sources/A.sol":SOURCE,"sources/B.sol":SOURCE});result=resolve_localization_ground_truth(parse_localization_ground_truth(f"{ADDRESS}##0:1"),index);self.assertIn("GROUND_TRUTH_FILE_AMBIGUOUS",result.reason_codes)

    def test_three_level_recall_at_1_and_3(self):
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"sources/A.sol":SOURCE});raw=SOURCE.encode();needle=b"uint256 x";offset=raw.index(needle);truth=resolve_localization_ground_truth(parse_localization_ground_truth(f"{ADDRESS}#A.sol#{offset}:{len(needle)}"),index)
            rows=[{"rank":1,"relative_file":"X.sol","function":"x()","start_line":1,"end_line":2},{"rank":2,"relative_file":"sources/A.sol","function":"pay(uint256)","start_line":2,"end_line":3}]
            metrics=evaluate_localization_topk(truth,rows);self.assertFalse(metrics.file_recall_at_1);self.assertTrue(metrics.file_recall_at_3);self.assertTrue(metrics.function_recall_at_3);self.assertTrue(metrics.line_range_recall_at_3)
            none=evaluate_localization_topk(truth,[]);self.assertFalse(none.file_recall_at_1);self.assertFalse(none.topk_generated)

    def test_summary_is_sanitized(self):
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"sources/A.sol":SOURCE});result=resolve_localization_ground_truth(parse_localization_ground_truth(f"{ADDRESS}#A.sol#0:1"),index);text=serialize_localization_ground_truth_summary(result)
            self.assertNotIn(ADDRESS,text);self.assertNotIn("A.sol",text);self.assertNotIn(SOURCE,text)

    def test_overloaded_function_name_requires_full_signature(self):
        source="contract C {\n function pay(uint x) external { uint a=x; }\n function pay(address x) external { address a=x; }\n}\n"
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"sources/A.sol":source});raw=source.encode();needle=b"uint a=x";offset=raw.index(needle);truth=resolve_localization_ground_truth(parse_localization_ground_truth(f"{ADDRESS}#A.sol#{offset}:{len(needle)}"),index)
            name_only=evaluate_localization_topk(truth,[{"rank":1,"relative_file":"A.sol","function":"pay","start_line":2,"end_line":2}]);self.assertFalse(name_only.function_recall_at_1)
            full=evaluate_localization_topk(truth,[{"rank":1,"relative_file":"A.sol","function":"pay(uint256)","start_line":2,"end_line":2}]);self.assertTrue(full.function_recall_at_1)

    def test_evaluation_does_not_change_candidate_order(self):
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"sources/A.sol":SOURCE});raw=SOURCE.encode();needle=b"uint256 x";offset=raw.index(needle);truth=resolve_localization_ground_truth(parse_localization_ground_truth(f"{ADDRESS}#A.sol#{offset}:{len(needle)}"),index)
            rows=[{"rank":2,"relative_file":"A.sol","function":"pay(uint256)","start_line":2,"end_line":3},{"rank":1,"relative_file":"X.sol","function":"x()","start_line":1,"end_line":1}];before=json.dumps(rows,sort_keys=True)
            evaluate_localization_topk(truth,rows);self.assertEqual(json.dumps(rows,sort_keys=True),before)

    def test_repair_v1_recovers_absolute_marker_suffix_without_opening_label_path(self):
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"contracts/A.sol":SOURCE});raw=SOURCE.encode();needle=b"uint256 x";offset=raw.index(needle)
            label=parse_localization_ground_truth_repair_v1(f"{ADDRESS}#C:\\untrusted\\repo\\contracts\\A.sol#{offset}:{len(needle)}")
            result=resolve_localization_ground_truth_repair_v1(label,index)
            self.assertTrue(result.function_available);self.assertTrue(result.path_recovery_applied)
            self.assertEqual(result.source_binding_role,"GROUND_TRUTH_LOCATION_ADDRESS")
            self.assertIn("GROUND_TRUTH_ABSOLUTE_PATH_SUFFIX_RECOVERED",result.reason_codes)

    def test_repair_v1_rejects_traversal_and_disallowed_absolute_suffix(self):
        with self.assertRaisesRegex(ValueError,"GROUND_TRUTH_PATH_TRAVERSAL_REJECTED"):
            parse_localization_ground_truth_repair_v1(f"{ADDRESS}#C:\\repo\\contracts\\..\\A.sol#0:1")
        with self.assertRaisesRegex(ValueError,"GROUND_TRUTH_PATH_SUFFIX_NOT_ALLOWED"):
            parse_localization_ground_truth_repair_v1(f"{ADDRESS}#C:\\repo\\A.sol#0:1")

    def test_repair_v1_absolute_suffix_requires_unique_file_match(self):
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"a/contracts/A.sol":SOURCE,"b/contracts/A.sol":SOURCE});label=parse_localization_ground_truth_repair_v1(f"{ADDRESS}#C:\\repo\\contracts\\A.sol#0:1")
            result=resolve_localization_ground_truth_repair_v1(label,index)
            self.assertIn("GROUND_TRUTH_PATH_SUFFIX_AMBIGUOUS",result.reason_codes)
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"contracts/B.sol":SOURCE});label=parse_localization_ground_truth_repair_v1(f"{ADDRESS}#C:\\repo\\contracts\\A.sol#0:1")
            result=resolve_localization_ground_truth_repair_v1(label,index)
            self.assertIn("GROUND_TRUTH_PATH_SUFFIX_NO_MATCH",result.reason_codes)

    def test_repair_v1_keeps_function_outside_location_unavailable(self):
        with tempfile.TemporaryDirectory() as t:
            index=cache(Path(t),{"sources/A.sol":SOURCE});offset=SOURCE.encode().index(b"contract")
            result=resolve_localization_ground_truth_repair_v1(parse_localization_ground_truth_repair_v1(f"{ADDRESS}#sources/A.sol#{offset}:8"),index)
            self.assertTrue(result.source_file_available);self.assertTrue(result.line_range_available);self.assertFalse(result.function_available)
            self.assertIn("GROUND_TRUTH_FUNCTION_NOT_FOUND",result.reason_codes)

    def test_repair_v1_uses_proxy_role_without_mixing_direct_source(self):
        implementation="0x"+"2"*40;proxy_source="contract Proxy { fallback() external {} }\n";impl_source="contract Impl { function target() external { uint x=1; } }\n"
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);child=root/"eth"/implementation;child.mkdir(parents=True);impl=child/"sources/Impl.sol";impl.parent.mkdir();impl.write_text(impl_source,encoding="utf-8");impl_digest=hashlib.sha256(impl.read_bytes()).hexdigest();child_manifest={"status":"verified","verified":True,"requested_address":implementation,"resolved_address":implementation,"files":[{"path":"sources/Impl.sol","sha256":impl_digest}],"proxy":{}};(child/"manifest.json").write_text(json.dumps(child_manifest),encoding="utf-8");child_manifest_sha=hashlib.sha256((child/"manifest.json").read_bytes()).hexdigest()
            parent=root/"eth"/ADDRESS;parent.mkdir(parents=True);proxy=parent/"sources/Proxy.sol";proxy.parent.mkdir();proxy.write_text(proxy_source,encoding="utf-8");proxy_digest=hashlib.sha256(proxy.read_bytes()).hexdigest();(parent/"manifest.json").write_text(json.dumps({"status":"verified","verified":True,"requested_address":ADDRESS,"resolved_address":ADDRESS,"files":[{"path":"sources/Proxy.sol","sha256":proxy_digest}],"proxy":{"implementation_address":implementation,"implementation_manifest_sha256":child_manifest_sha}}),encoding="utf-8")
            index=build_verified_source_cache_index(root,blockchain_chain_id=1,chain_key="eth");offset=impl_source.encode().index(b"uint x")
            result=resolve_localization_ground_truth_repair_v1(parse_localization_ground_truth_repair_v1(f"{ADDRESS}#sources/Impl.sol#{offset}:6"),index)
            self.assertTrue(result.function_available);self.assertEqual(result.source_binding_role,"PROXY_IMPLEMENTATION_ADDRESS")

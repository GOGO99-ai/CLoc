from __future__ import annotations

import importlib.util, json, sys, tempfile, unittest
from pathlib import Path

_PATH=Path(__file__).resolve().parents[2]/"scripts"/"run_phase2_real_pilot.py"; _SPEC=importlib.util.spec_from_file_location("run_phase2_real_pilot",_PATH); pilot=importlib.util.module_from_spec(_SPEC); sys.modules[_SPEC.name]=pilot; _SPEC.loader.exec_module(pilot)

class Rpc:
 def __init__(self, values): self.values=list(values); self.calls=[]
 def call(self, method, params): self.calls.append(method); value=self.values.pop(0); (_ for _ in ()).throw(value) if isinstance(value,Exception) else None; return value

class DeepResult:
 pass

def bundle(): return pilot.Phase2EvidenceBundle(object(),object(),{}, {"result_candidate_count":2,"strict_chain_count":2}, (), ())
def source(stop=None): return ({"schema_version":"source_request_slices_v1","requests":[]},{"source_request_count":0,"attempted_address_count":0,"verified_source_count":0,"cache_hit_count":0,"resolved_slice_count":0,"stop_code":stop})

class Phase2RealPilotTests(unittest.TestCase):
 def test_manifest_fixed_and_stable(self):
  one=pilot.build_locked_manifest();two=pilot.build_locked_manifest();self.assertEqual(one.manifest_sha256,two.manifest_sha256);self.assertEqual(one.tx_hash,pilot.TX_HASH);self.assertEqual(one.sample_alias,"phase2_fault_007")
 def test_offline_zero_requests(self):
  report=pilot.run_phase2_pilot(pilot.build_locked_manifest(),live=False);self.assertEqual(report["external_call_counts"],{"rpc":0,"source":0,"deepseek":0})
 def test_chain_mismatch_stops_after_one(self):
  rpc=Rpc(["0x38"]);report=pilot.run_phase2_pilot(pilot.build_locked_manifest(),live=True,rpc_client=rpc,source_builder=lambda b:source(),deepseek_runner=lambda p:None);self.assertEqual(len(rpc.calls),1);self.assertEqual(report["run_status"],"STOPPED")
 def test_rpc_error_stops_without_source_or_deepseek(self):
  rpc=Rpc(["0x1",RuntimeError("RPC_NETWORK_ERROR")]);calls={"source":0,"deep":0}
  report=pilot.run_phase2_pilot(pilot.build_locked_manifest(),live=True,rpc_client=rpc,evidence_builder=lambda x:bundle(),source_builder=lambda b:calls.__setitem__("source",1),deepseek_runner=lambda p:calls.__setitem__("deep",1));self.assertEqual(len(rpc.calls),2);self.assertEqual(calls,{"source":0,"deep":0});self.assertEqual(report["external_call_counts"]["rpc"],2)
 def test_source_error_stops_before_deepseek(self):
  rpc=Rpc(["0x1",{}]);deep=[];report=pilot.run_phase2_pilot(pilot.build_locked_manifest(),live=True,rpc_client=rpc,evidence_builder=lambda x:bundle(),source_builder=lambda b:source("SOURCE_FETCH_EXTERNAL_ERROR"),deepseek_runner=lambda p:deep.append(1));self.assertFalse(deep);self.assertEqual(report["run_status"],"STOPPED")
 def test_request_limits(self):
  report=pilot.build_phase2_report(pilot.build_locked_manifest(),run_status="X");self.assertEqual(report["request_limits"],{"rpc":2,"source":3,"deepseek":1})
 def test_location_evaluation_rules(self):
  rows=[{"relative_file":"contracts/A.sol","function":"f(uint256)","start_line":10,"end_line":20},{"relative_file":"B.sol","function":"g()","start_line":1,"end_line":3}]
  self.assertEqual(pilot.evaluate_top3_locations(rows,{"file":"A.sol","line":15}),"TOP1_HIT");self.assertEqual(pilot.evaluate_top3_locations(rows,{"function":"g"}),"TOP3_HIT");self.assertEqual(pilot.evaluate_top3_locations(rows,{"function":"z"}),"MISS");self.assertEqual(pilot.evaluate_top3_locations(rows,"unknown"),"EVALUATION_UNAVAILABLE")
 def test_location_does_not_change_inference_manifest(self):
  manifest=pilot.build_locked_manifest();before=manifest.manifest_sha256;pilot.evaluate_top3_locations([],"one");pilot.evaluate_top3_locations([],"two");self.assertEqual(before,manifest.manifest_sha256)
 def test_report_rejects_sensitive_content(self):
  report=pilot.build_phase2_report(pilot.build_locked_manifest(),run_status="X");report["reason_codes"]=["0x"+"1"*40]
  with self.assertRaisesRegex(ValueError,"SENSITIVE"):pilot.validate_phase2_report(report)
 def test_atomic_write(self):
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/"x.json";pilot._atomic(path,{"a":1});self.assertEqual(json.loads(path.read_text()),{"a":1});self.assertEqual(len(list(Path(td).iterdir())),1)
 def test_checkpoint_counts_match_on_safe_stop(self):
  checkpoints=[];rpc=Rpc(["0x1",{}]);report=pilot.run_phase2_pilot(pilot.build_locked_manifest(),live=True,rpc_client=rpc,evidence_builder=lambda x:bundle(),source_builder=lambda b:source("SOURCE_FETCH_EXTERNAL_ERROR"),deepseek_runner=lambda p:None,checkpoint_callback=checkpoints.append);self.assertEqual(checkpoints[-1]["rpc_request_count"],report["external_call_counts"]["rpc"]);self.assertEqual(checkpoints[-1]["source_request_count"],report["external_call_counts"]["source"])
 def test_label_object_cannot_enter_bridge_or_client(self):
  import inspect
  self.assertNotIn("location",inspect.signature(pilot.prepare_phase2_localization).parameters);self.assertNotIn("location",inspect.signature(pilot.execute_phase2_localization).parameters)
 def test_fake_ready_end_to_end_calls_deepseek_once(self):
  from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekResult
  nodes=[{"node_id":"n1","callee_address":"0x"+"1"*40},{"node_id":"n2","callee_address":"0x"+"2"*40}]
  strict={"schema_version":"strict_result_chains_v1","edges":[{"edge_id":"e1","edge_type":"CALL","src_node_id":"n1","dst_node_id":"n2","uncertainties":[]}],"chains":[{"strict_chain_id":"c1","result_type":"R_OUTFLOW","node_ids":["n1","n2"],"edge_ids":["e1"],"anchor":{"node_id":"n1"},"uncertainties":[]}],"result_statuses":[],"critical_uncertainties":[],"coverage":{}}
  graph={"schema_version":"transaction_graph_v1","nodes":nodes};causal={"schema_version":"candidate_causal_graph_v1","edges":[]}
  ready_bundle=pilot.Phase2EvidenceBundle(graph,strict,causal,{"result_candidate_count":1,"strict_chain_count":1},("n1",),("0x"+"1"*40,))
  slices={"schema_version":"source_request_slices_v1","requests":[{"node_id":"n1","function_match_status":"RESOLVED","file_path":"C.sol","function_name":"f","signature":"f()","start_line":1,"end_line":1,"source":"1: function f(){}","uncertainties":[]}]}
  response={"schema_version":"phase2_localization_response_v1","status":"LOCATED","findings":[{"rank":1,"source_slice_id":"S1","role":"ROOT_CAUSE","confidence":"HIGH","evidence_ids":["E1"],"explanation":"S1 relates to E1"}],"summary":"localized"}
  calls=[]
  def deep(prep):
   calls.append(1);return Phase2DeepSeekResult("VALIDATED",prep.prompt_sha256,"b"*64,"deepseek-v4-pro",None,("RESPONSE_VALIDATED",),json.dumps(response))
  report=pilot.run_phase2_pilot(pilot.build_locked_manifest(),live=True,rpc_client=Rpc(["0x1",{}]),evidence_builder=lambda raw:ready_bundle,source_builder=lambda b:(slices,{"source_request_count":0,"attempted_address_count":1,"verified_source_count":1,"cache_hit_count":1,"resolved_slice_count":1,"stop_code":None}),deepseek_runner=deep)
  self.assertEqual(len(calls),1);self.assertEqual(report["run_status"],"COMPLETED");self.assertEqual(len(report["top3"]),1);self.assertEqual(report["external_call_counts"]["deepseek"],1)
 def test_normalized_valid_response_is_evaluated_and_reported(self):
  from src.cloc_v2.phase2_localization_deepseek import Phase2DeepSeekResult
  result=Phase2DeepSeekResult("VALIDATED_WITH_NORMALIZATION","a"*64,"b"*64,"deepseek-v4-pro",None,("RESPONSE_VALIDATED","SUMMARY_REPLACED_LOCALLY"),json.dumps({"schema_version":"phase2_localization_response_v1","status":"INSUFFICIENT_EVIDENCE","findings":[],"summary":"SUMMARY_OMITTED_FROM_EVALUATION"}),True,("SUMMARY_REPLACED_LOCALLY",))
  report=pilot.build_phase2_report(pilot.build_locked_manifest(),run_status="COMPLETED",deepseek_result=result)
  self.assertTrue(report["deepseek"]["valid"]);self.assertTrue(report["deepseek"]["normalization_applied"]);self.assertEqual(report["deepseek"]["normalization_actions"],["SUMMARY_REPLACED_LOCALLY"])

if __name__=="__main__":unittest.main()

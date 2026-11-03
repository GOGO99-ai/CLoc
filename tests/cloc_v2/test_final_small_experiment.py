"""Offline safety tests for the final small-experiment coordinator."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("final_small", ROOT / "scripts" / "run_final_small_experiment.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def tx(number: int) -> str:
    return "0x" + f"{number:064x}"


def fixture_inputs(tmp_path: Path):
    manifest = {"samples": [{"chain_key": "eth", "blockchain_chain_id": 1, "sample_id": f"n{i}", "tx_hash": tx(i)} for i in range(1, 31)]}
    checkpoint = {"samples": {f"eth:{tx(i)}": {"status": "SUCCEEDED"} for i in range(1, 31)}}
    dataset = tmp_path / "dataset"; dataset.mkdir()
    for i in range(101, 131):
        (dataset / f"{i}.json").write_text(json.dumps({"platform": "Ethereum", "fault": {"transaction_hash": [[tx(i)]], "cause": "ignored", "location": ["ignored"]}}), encoding="utf-8")
    (dataset / "bad.json").write_text("{", encoding="utf-8")
    prototypes = tmp_path / "prototypes"; prototypes.mkdir()
    (prototypes / "old_manifest.json").write_text(json.dumps({"samples": [{"tx_hash": tx(1)}]}), encoding="utf-8")
    return manifest, checkpoint, dataset, prototypes


class FakeRpc:
    def __init__(self): self.calls = []
    def call(self, method, params):
        self.calls.append((method, params))
        return "0x1" if method == "eth_chainId" else {"type": "CALL"}


class FinalSmallExperimentTests(unittest.TestCase):
    def test_selection_is_stable_isolated_and_excludes_history(self):
        with tempfile.TemporaryDirectory() as temp:
            normal, checkpoint, dataset, prototypes = fixture_inputs(Path(temp))
            first = mod.build_final_selection(normal, checkpoint, dataset, prototypes)
            second = mod.build_final_selection(normal, checkpoint, dataset, prototypes)
            self.assertEqual(first.manifest.to_dict(), second.manifest.to_dict())
            self.assertNotIn(tx(1), {x.tx_hash for x in first.manifest.samples})
            self.assertEqual(len(first.manifest.samples), 40)
            payload = json.dumps(first.manifest.to_dict())
            self.assertNotIn("NORMAL", payload); self.assertNotIn("FAULT", payload)
            self.assertNotIn("cause", payload.lower()); self.assertNotIn("location", payload.lower())
            self.assertEqual(list(first.truth.labels.values()).count("NORMAL"), 20)

    def test_queue_is_label_blind_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            normal, checkpoint, dataset, prototypes = fixture_inputs(Path(temp))
            selection = mod.build_final_selection(normal, checkpoint, dataset, prototypes)
            rows = [{"sample_alias": item.sample_alias, "final_route": "PROCEED_PHASE2"} for item in selection.manifest.samples]
            queue = mod.select_phase2_queue(rows, selection.manifest)
            self.assertEqual(len(queue), 10)
            self.assertEqual(queue, mod.select_phase2_queue(list(reversed(rows)), selection.manifest))

    def test_real_checkpoint_key_and_nested_fault_hashes_are_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            normal, checkpoint, dataset, _ = fixture_inputs(Path(temp))
            self.assertEqual(len(mod._normal_pool(normal, checkpoint)), 30)
            self.assertEqual(len(mod._fault_pool(dataset)), 30)

    def test_inference_is_offline_by_default_and_one_trace_per_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); normal, checkpoint, dataset, prototypes = fixture_inputs(root)
            selection = mod.build_final_selection(normal, checkpoint, dataset, prototypes)
            self.assertEqual(mod.infer_final_experiment(selection.manifest, root, live=False)["external_call_counts"]["rpc"], 0)
            rpc = FakeRpc()
            def p1(sample, raw): return {"final_route": "PROCEED_PHASE2", "reason_codes": [], "llm_called": False}
            def p2(sample, raw): return {"status": "INSUFFICIENT_SOURCE", "reason_codes": [], "llm_called": False, "topk_frozen": False}
            result = mod.infer_final_experiment(selection.manifest, root, live=True, rpc_client=rpc, phase1_processor=p1, phase2_processor=p2)
            self.assertEqual(result["external_call_counts"]["rpc"], 41)
            self.assertEqual(sum(method == "debug_traceTransaction" for method, _ in rpc.calls), 40)
            self.assertEqual(result["external_call_counts"]["phase1_deepseek"], 0)
            self.assertEqual(result["external_call_counts"]["phase2_deepseek"], 0)
            self.assertTrue((root / "frozen_phase1_predictions.json").exists())
            self.assertTrue((root / "frozen_phase2_predictions.json").exists())

    def test_freeze_is_immutable_and_evaluation_waits_for_frozen_blobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); normal, checkpoint, dataset, prototypes = fixture_inputs(root)
            selection = mod.build_final_selection(normal, checkpoint, dataset, prototypes)
            with self.assertRaises(ValueError): mod.evaluate_final_experiment(root, selection.manifest, selection.truth)
            rpc = FakeRpc()
            mod.infer_final_experiment(selection.manifest, root, live=True, rpc_client=rpc,
                phase1_processor=lambda sample, raw: {"final_route": "EARLY_NORMAL", "reason_codes": [], "llm_called": False},
                phase2_processor=lambda sample, raw: {"status": "NOT_QUEUED", "reason_codes": [], "llm_called": False, "topk_frozen": False})
            before = (root / "frozen_phase1_predictions.json").read_text(encoding="utf-8")
            report = mod.evaluate_final_experiment(root, selection.manifest, selection.truth, location_loader=lambda alias: "TOP1_HIT")
            self.assertEqual(report["phase1"]["normal_count"], 20)
            self.assertEqual(report["phase1"]["fault_count"], 20)
            self.assertEqual((root / "frozen_phase1_predictions.json").read_text(encoding="utf-8"), before)

    def test_prediction_sanitization_and_budget_constants(self):
        self.assertEqual((mod.MAX_RPC_REQUESTS, mod.MAX_SOURCE_REQUESTS, mod.MAX_PHASE1_DEEPSEEK_REQUESTS, mod.MAX_PHASE2_DEEPSEEK_REQUESTS, mod.MAX_DEEPSEEK_REQUESTS), (41, 30, 40, 10, 50))
        with self.assertRaises(ValueError): mod._sanitize({"provider_response": "secret"})
        with self.assertRaises(ValueError): mod._sanitize({"x": "0x" + "1" * 40})

    def test_live_placeholder_is_removed(self):
        text=(ROOT/"scripts"/"run_final_small_experiment.py").read_text(encoding="utf-8")
        self.assertNotIn("FINAL_EXPERIMENT_LIVE_WIRING_REQUIRED",text)
        self.assertIn("build_live_adapters",text)

    def test_phase1_and_phase2_share_same_trace_object(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); normal, checkpoint, dataset, prototypes=fixture_inputs(root); selection=mod.build_final_selection(normal,checkpoint,dataset,prototypes); rpc=FakeRpc(); seen={}
            def p1(sample,raw): seen.setdefault(sample.sample_alias,id(raw)); return {"final_route":"PROCEED_PHASE2","reason_codes":[],"llm_called":False}
            def p2(sample,raw): self.assertEqual(seen[sample.sample_alias],id(raw)); return {"status":"INSUFFICIENT_SOURCE","reason_codes":[],"llm_called":False,"topk_frozen":False}
            mod.infer_final_experiment(selection.manifest,root,live=True,rpc_client=rpc,phase1_processor=p1,phase2_processor=p2)
            self.assertEqual(sum(method=="debug_traceTransaction" for method,_ in rpc.calls),40)

    def test_restart_skips_completed_samples_and_keeps_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); normal, checkpoint, dataset, prototypes=fixture_inputs(root); selection=mod.build_final_selection(normal,checkpoint,dataset,prototypes); rpc=FakeRpc()
            p1=lambda sample,raw:{"final_route":"PROCEED_PHASE2","reason_codes":[],"llm_called":False}; p2=lambda sample,raw:{"status":"DONE","reason_codes":[],"llm_called":False,"topk_frozen":False}
            mod.infer_final_experiment(selection.manifest,root,live=True,rpc_client=rpc,phase1_processor=p1,phase2_processor=p2)
            frozen_queue=json.loads((root/"checkpoint.json").read_text(encoding="utf-8"))["phase2_queue"]
            restarted=FakeRpc(); mod.infer_final_experiment(selection.manifest,root,live=True,rpc_client=restarted,phase1_processor=p1,phase2_processor=p2)
            self.assertEqual(restarted.calls,[])
            self.assertEqual(json.loads((root/"checkpoint.json").read_text(encoding="utf-8"))["phase2_queue"],frozen_queue)

    def test_network_failure_keeps_checkpoint_without_frozen_output(self):
        class Broken(FakeRpc):
            def call(self,method,params):
                if method=="debug_traceTransaction": raise RuntimeError("network")
                return super().call(method,params)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); normal, checkpoint, dataset, prototypes=fixture_inputs(root); selection=mod.build_final_selection(normal,checkpoint,dataset,prototypes)
            result=mod.infer_final_experiment(selection.manifest,root,live=True,rpc_client=Broken(),phase1_processor=lambda s,r:{},phase2_processor=lambda s,r:{})
            self.assertEqual(result["run_status"],"STOPPED"); self.assertTrue((root/"checkpoint.json").exists()); self.assertFalse((root/"frozen_phase1_predictions.json").exists())

    def test_phase2_insufficient_evidence_does_not_call_provider(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); normal, checkpoint, dataset, prototypes=fixture_inputs(root); selection=mod.build_final_selection(normal,checkpoint,dataset,prototypes); calls=[]
            mod.infer_final_experiment(selection.manifest,root,live=True,rpc_client=FakeRpc(),phase1_processor=lambda s,r:{"final_route":"PROCEED_PHASE2","reason_codes":[],"llm_called":False},phase2_processor=lambda s,r:{"bridge_status":"INSUFFICIENT_SOURCE","llm_called":False,"topk_frozen":False,"reason_codes":["INSUFFICIENT_SOURCE"]})
            p2=json.loads((root/"frozen_phase2_predictions.json").read_text(encoding="utf-8")); self.assertTrue(all(not row["llm_called"] for row in p2["predictions"])); self.assertEqual(calls,[])

    def test_offline_mode_never_builds_live_adapters(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); normal, checkpoint, dataset, prototypes=fixture_inputs(root); selection=mod.build_final_selection(normal,checkpoint,dataset,prototypes)
            with patch.object(mod,"build_live_adapters",side_effect=AssertionError("must not load config")):
                self.assertEqual(mod.infer_final_experiment(selection.manifest,root,live=False)["run_status"],"LIVE_MODE_REQUIRED")

    def test_evaluator_top1_top3_miss_and_unavailable(self):
        from scripts.run_phase2_real_pilot import evaluate_top3_locations
        rows=[{"relative_file":"A.sol","function":"a()","start_line":1,"end_line":10},{"relative_file":"B.sol","function":"b()","start_line":20,"end_line":30}]
        self.assertEqual(evaluate_top3_locations(rows,{"file":"A.sol","line":5}),"TOP1_HIT")
        self.assertEqual(evaluate_top3_locations(rows,{"file":"B.sol","line":25}),"TOP3_HIT")
        self.assertEqual(evaluate_top3_locations(rows,{"file":"C.sol","line":1}),"MISS")
        self.assertEqual(evaluate_top3_locations(rows,None),"EVALUATION_UNAVAILABLE")

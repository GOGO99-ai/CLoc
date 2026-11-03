import copy, json, tempfile, unittest
from pathlib import Path
from src.cloc_v2.phase1_prompting import PROMPT_VERSION, SYSTEM_PROMPT_V3, USER_INSTRUCTIONS_V3, build_preview, preview_metrics, prompt_document, render_messages, write_preview

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
class Phase1PromptingTests(unittest.TestCase):
 def load(self): return json.loads((RUN/'phase1_scoring_packets_v2.json').read_text())
 def test_roles_content_and_v2_payload_compatibility(self):
  packets=self.load();before=copy.deepcopy(packets);messages=render_messages(packets,'phase1_batch_b001')
  self.assertEqual(['system','user'],[x['role'] for x in messages]);self.assertEqual(before,packets);self.assertIn('ORDER establishes execution order, not causality.',messages[0]['content']);self.assertIn('OBSERVED:',messages[0]['content']);self.assertIn('STRONG_ASSOCIATION:',messages[0]['content']);self.assertIn('HYPOTHESIS:',messages[0]['content']);self.assertIn('counter_evidence is existing supplied evidence',messages[1]['content']);self.assertIn('uncertainties are missing information',messages[1]['content']);self.assertIn('hidden chain-of-thought',messages[0]['content']);self.assertIn('node belongs to the chain',messages[1]['content']);self.assertIn('"scores"',messages[1]['content']);self.assertIn('BATCH_LLM_PAYLOAD_JSON:',messages[1]['content'])
 def test_schema_formula_and_deterministic_preview(self):
  packets=self.load();a=build_preview(packets,'phase1_batch_b001');b=build_preview(packets,'phase1_batch_b001');self.assertEqual(json.dumps(a,sort_keys=True),json.dumps(b,sort_keys=True));self.assertEqual(PROMPT_VERSION,a['prompt_version']);self.assertEqual({'batch_id','messages','model_parameters','input_sha256','prompt_version'},set(a));self.assertIn('result_relevance*20',a['messages'][1]['content']);self.assertIn('exactly two decimal places',a['messages'][1]['content']);self.assertGreater(preview_metrics(a)['estimated_tokens'],0)
 def test_prompt_document_and_atomic_preview_writer(self):
  self.assertIn('## System prompt',prompt_document());self.assertIn('## User prompt fixed instructions',prompt_document());self.assertIn('asset_anomaly',USER_INSTRUCTIONS_V3)
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'nested'/'preview.json';preview=build_preview(self.load(),'phase1_batch_b001');write_preview(preview,path);first=path.read_bytes();write_preview(preview,path);self.assertEqual(first,path.read_bytes())
 def test_invalid_packets_or_batch_are_rejected(self):
  with self.assertRaises(ValueError):render_messages({},'phase1_batch_b001')
  with self.assertRaises(ValueError):render_messages(self.load(),'missing')
if __name__=='__main__':unittest.main()

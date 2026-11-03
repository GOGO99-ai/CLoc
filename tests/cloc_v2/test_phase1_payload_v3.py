import copy,json,re,unittest
from pathlib import Path
from src.cloc_v2.phase1_payload_v3 import audit_llm_payload,build_phase1_scoring_packets_v3
from src.cloc_v2.phase1_prompting import USER_INSTRUCTIONS_V3_1,build_preview_v3_1,render_messages_v3_1
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
class Phase1PayloadV3Tests(unittest.TestCase):
 def load(self):return json.loads((RUN/'phase1_scoring_packets_v2.json').read_text())
 def test_aliasing_sanitization_and_determinism(self):
  raw=self.load();before=copy.deepcopy(raw);a=build_phase1_scoring_packets_v3(raw);b=build_phase1_scoring_packets_v3(raw);self.assertEqual(raw,before);self.assertEqual(json.dumps(a,sort_keys=True),json.dumps(b,sort_keys=True));self.assertEqual({'evm_address_count':0,'original_field_count':0,'arguments_raw_count':0,'resolver_metadata_count':0},audit_llm_payload(a))
  rendered=json.dumps([x['llm_payload'] for x in a['batches']]);self.assertIsNone(re.search(r'0x[0-9a-fA-F]{40}',rendered));self.assertIn('0xa70cfccf',rendered);self.assertIn('amount_raw',rendered);self.assertTrue(a['resolver_metadata']['address_dictionary'])
 def test_global_aliases_and_structural_completeness(self):
  raw=self.load();v3=build_phase1_scoring_packets_v3(raw);self.assertEqual([x['chain_ids'] for x in raw['batches']],[x['chain_ids'] for x in v3['batches']]);self.assertEqual(19,len({x for b in v3['batches'] for x in b['chain_ids']}));self.assertEqual(sum(len(b['chain_ids']) for b in v3['batches']),19)
  resolver=v3['resolver_metadata']['address_dictionary'];self.assertEqual(len(resolver),len(set(resolver)));self.assertTrue(all(re.fullmatch(r'a\d{3}',alias) for alias in resolver))
  for old,new in zip(raw['batches'],v3['batches']):
   self.assertEqual([x[0] for x in old['llm_payload']['edges']],[x[0] for x in new['llm_payload']['edges']]);self.assertEqual([x[0] for x in old['llm_payload']['chains']],[x[0] for x in new['llm_payload']['chains']])
 def test_v3_1_prompt_keeps_schema_and_empty_example(self):
  packet=build_phase1_scoring_packets_v3(self.load());messages=render_messages_v3_1(packet,'phase1_batch_b001');self.assertEqual(['system','user'],[x['role'] for x in messages]);self.assertIn('"source_requests": []',USER_INSTRUCTIONS_V3_1);self.assertIn('Priority means 1=high',USER_INSTRUCTIONS_V3_1);self.assertIn('at most three per chain',USER_INSTRUCTIONS_V3_1);preview=build_preview_v3_1(packet,'phase1_batch_b001');self.assertEqual({'batch_id','messages','model_parameters','input_sha256','prompt_version'},set(preview));self.assertNotIn('resolver_metadata',preview['messages'][1]['content'])
if __name__=='__main__':unittest.main()

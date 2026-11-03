import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
from src.cloc_v2.phase1_llm_client import ClientConfig,OpenAICompatibleTransport,Phase1Runner,TransportError,ValidationError,validate_batch_response
from src.cloc_v2.phase1_payload_v3 import build_phase1_scoring_packets_v3
from src.cloc_v2.phase1_prompting import PROMPT_VERSION_V3_1,prompt_document_v3_1

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'data'/'runs'/'eth'/'0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299'
class FakeTransport:
 def __init__(self,events):self.events=list(events);self.calls=0;self.requests=[]
 def send(self,request,api_key,timeout_seconds):
  self.calls+=1;self.requests.append(request);event=self.events.pop(0)
  if isinstance(event,Exception):raise event
  if event=='valid':return valid_provider_response(request)
  if event=='invalid':return {'id':'fake','model':'fake','choices':[{'finish_reason':'stop','message':{'content':'not json'}}]}
  return event
def valid_provider_response(request,source_count=0):
 payload=json.loads(request['messages'][1]['content'].split('BATCH_LLM_PAYLOAD_JSON:\n',1)[1]);responses=[]
 for chain in [item[0] for item in payload['chains']]:
  item={'chain_id':chain,'scores':{'result_relevance':1,'asset_anomaly':1,'state_dependency':1,'parameter_influence':1,'control_interaction':1,'causal_coherence':1},'total_score':'20.00','concise_rationale':'evidence only','key_node_ids':[],'key_edge_ids':[],'counter_evidence':[],'uncertainties':[],'source_requests':[]}
  responses.append(item)
 return {'id':'fake-request','model':'fake-model','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'batch_id':request['batch_id'],'responses':responses})}}],'usage':{'total_tokens':1}}
class ClientTests(unittest.TestCase):
 def packets(self):return build_phase1_scoring_packets_v3(json.loads((RUN/'phase1_scoring_packets_v2.json').read_text()))
 def runner(self,td,config=None,batch_id=None):
  tx=Path(td)/'data'/'runs'/'eth'/('0x'+'1'*64);out=tx/'phase1_llm';prompt=tx/'phase1_prompt_v3_1.md';packet_path=tx/'phase1_scoring_packets_v3.json';prompt.parent.mkdir(parents=True);prompt.write_text(prompt_document_v3_1());packet_path.write_text(json.dumps(self.packets(),ensure_ascii=False,indent=2)+'\n')
  return Phase1Runner(json.loads(packet_path.read_text()),prompt,out,config or ClientConfig(),packet_path,batch_id),out
 def req(self,runner):
  request=runner.requests()[0];request['_packets']=runner.packets;return request
 def test_dry_run_is_deterministic_and_never_uses_transport(self):
  with tempfile.TemporaryDirectory() as td:
   runner,out=self.runner(td);a=runner.dry_run();first=(out/'requests'/'phase1_batch_b001.request.json').read_bytes();b=runner.dry_run()
   self.assertFalse(a['complete']);self.assertEqual(first,(out/'requests'/'phase1_batch_b001.request.json').read_bytes());self.assertEqual(4,len(a['batches']));self.assertFalse((out/'complete_manifest.json').exists());self.assertFalse((out/'partial_manifest.json').exists());self.assertEqual(a,b)
 def test_execute_requires_explicit_model_parameters(self):
  with tempfile.TemporaryDirectory() as td:
   runner,_=self.runner(td)
   with self.assertRaises(ValueError):runner.config.validate_execute()
   with self.assertRaises(ValueError):Phase1Runner(self.packets(),Path(td)/'p',Path(td)/'broad',ClientConfig())
 def test_transport_forwards_optional_json_response_format_and_thinking(self):
  response=MagicMock();response.__enter__.return_value.read.return_value=b'{}'
  request={'model':'m','messages':[],'parameters':{'max_output_tokens':10,'omit_temperature':True,'response_format':{'type':'json_object'},'thinking':{'type':'disabled'}}}
  with patch('src.cloc_v2.phase1_llm_client.urllib.request.urlopen',return_value=response) as opened:
   self.assertEqual({},OpenAICompatibleTransport('https://example.test/v1').send(request,'secret',1))
  payload=json.loads(opened.call_args.args[0].data.decode('utf-8'));self.assertEqual({'type':'json_object'},payload['response_format']);self.assertEqual({'type':'disabled'},payload['thinking']);self.assertNotIn('temperature',payload)
 def test_transport_retries_only_temporary_failures_and_is_bounded(self):
  config=ClientConfig('openai_compatible','m',0,10,1,3,1)
  with tempfile.TemporaryDirectory() as td:
   runner,_=self.runner(td,config);request=self.req(runner);fake=FakeTransport([TransportError('http','rate',429),'valid'])
   with patch('src.cloc_v2.phase1_llm_client.time.sleep'):
    batch,ok,error=runner._execute_one(request,fake,'secret',False)
   self.assertTrue(ok);self.assertEqual(2,fake.calls);self.assertIsNone(error)
  with tempfile.TemporaryDirectory() as td:
   runner,_=self.runner(td,config);request=self.req(runner);fake=FakeTransport([TransportError('http','forbidden',403)])
   batch,ok,error=runner._execute_one(request,fake,'secret',False);self.assertFalse(ok);self.assertEqual(1,fake.calls)
  with tempfile.TemporaryDirectory() as td:
   runner,_=self.runner(td,config);request=self.req(runner);fake=FakeTransport([TransportError('timeout','x'),TransportError('timeout','x'),TransportError('timeout','x')])
   with patch('src.cloc_v2.phase1_llm_client.time.sleep'):
    batch,ok,error=runner._execute_one(request,fake,'secret',False)
   self.assertFalse(ok);self.assertEqual(3,fake.calls)
 def test_semantic_failure_saves_raw_but_not_validated(self):
  config=ClientConfig('openai_compatible','m',0,10,1)
  with tempfile.TemporaryDirectory() as td:
   runner,out=self.runner(td,config);request=self.req(runner);batch,ok,error=runner._execute_one(request,FakeTransport(['invalid']),'secret',False)
   self.assertFalse(ok);self.assertIn('semantic_failure',error);self.assertTrue((out/'raw_responses'/f'{batch}.raw.json').exists());self.assertFalse((out/'validated_responses'/f'{batch}.validated.json').exists())
 def test_validated_response_strictness_limits_and_key_secrecy(self):
  packet=self.packets();config=ClientConfig('openai_compatible','m',0,10,1)
  with tempfile.TemporaryDirectory() as td:
   runner,out=self.runner(td,config);request=self.req(runner);fake=FakeTransport(['valid']);batch,ok,_=runner._execute_one(request,fake,'super-secret-value',False)
   self.assertTrue(ok);self.assertTrue((out/'raw_responses'/f'{batch}.raw.json').exists());self.assertTrue((out/'validated_responses'/f'{batch}.validated.json').exists())
   self.assertNotIn('super-secret-value',''.join(x.read_text() for x in out.rglob('*.json')))
   response=json.loads(valid_provider_response(request)['choices'][0]['message']['content']);chain=response['responses'][0];chain['source_requests']=[{'node_id':next(iter(set(packet['batches'][0]['llm_payload']['chains'][0][4]))),'request_reason':'q','focus':'validation','priority':1} for _ in range(4)]
   with self.assertRaises(ValidationError):validate_batch_response(response,packet,'phase1_batch_b001',PROMPT_VERSION_V3_1)
   self.assertIsNotNone(validate_batch_response(response,packet,'phase1_batch_b001','phase1_prompt_v2'))
   response['extra']='x'
   with self.assertRaises(ValidationError):validate_batch_response(response,packet,'phase1_batch_b001',PROMPT_VERSION_V3_1)
 def test_resume_requires_matching_request_and_complete_requires_all_batches(self):
  config=ClientConfig('openai_compatible','m',0,10,1,1,1)
  with tempfile.TemporaryDirectory() as td:
   runner,out=self.runner(td,config);requests=runner.requests()
   for request in requests:request['_packets']=runner.packets
   fake=FakeTransport(['valid','valid','valid','valid']);manifest=runner.execute(fake,'secret');self.assertTrue(manifest['complete']);self.assertTrue((out/'complete_manifest.json').exists());self.assertEqual(4,fake.calls)
   resume=FakeTransport([TransportError('connection','must not call')]);manifest=runner.execute(resume,'secret',True);self.assertTrue(manifest['complete']);self.assertEqual(0,resume.calls)
  with tempfile.TemporaryDirectory() as td:
   runner,out=self.runner(td,config);fake=FakeTransport(['valid','invalid','valid','valid']);manifest=runner.execute(fake,'secret');self.assertFalse(manifest['complete']);self.assertFalse((out/'complete_manifest.json').exists())
 def test_single_batch_selection_writes_partial_manifest_and_rejects_unknown(self):
  with tempfile.TemporaryDirectory() as td:
   runner,out=self.runner(td,batch_id='phase1_batch_b004');manifest=runner.dry_run()
   self.assertEqual(['phase1_batch_b004'],[item['batch_id'] for item in manifest['batches']]);self.assertTrue((out/'partial_manifest.json').exists());self.assertFalse((out/'complete_manifest.json').exists());self.assertFalse((out/'dry_run_manifest.json').exists())
   self.assertEqual(['phase1_batch_b004.request.json'],[path.name for path in (out/'requests').glob('*.json')])
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(ValidationError):self.runner(td,batch_id='phase1_batch_unknown')
 def test_partial_execute_then_full_resume_completes_all_batches(self):
  config=ClientConfig('openai_compatible','m',0,10,1,1,1)
  with tempfile.TemporaryDirectory() as td:
   pilot,out=self.runner(td,config,'phase1_batch_b001');partial=pilot.execute(FakeTransport(['valid']),'secret')
   self.assertFalse(partial['complete']);self.assertTrue((out/'partial_manifest.json').exists());self.assertFalse((out/'complete_manifest.json').exists())
   full=Phase1Runner(pilot.packets,pilot.prompt_path,out,config,pilot.prompt_path.parent/'phase1_scoring_packets_v3.json',None);fake=FakeTransport(['valid','valid','valid']);manifest=full.execute(fake,'secret',True)
   self.assertTrue(manifest['complete']);self.assertEqual(3,fake.calls);self.assertTrue((out/'complete_manifest.json').exists())
 def test_packet_file_and_canonical_hashes_are_explicit_and_semantically_stable(self):
  with tempfile.TemporaryDirectory() as td:
   runner,out=self.runner(td);request=runner.requests()[0]
   self.assertIn('packets_file_sha256',request);self.assertIn('packets_canonical_sha256',request);self.assertIn('messages_canonical_sha256',request);self.assertIn('request_canonical_sha256',request)
   self.assertNotIn('packet_sha256',request);self.assertNotIn('input_sha256',request)
   packet_path=runner.prompt_path.parent/'phase1_scoring_packets_v3.json';self.assertEqual(runner.packets_file_sha256,__import__('hashlib').sha256(packet_path.read_bytes()).hexdigest());self.assertEqual(runner.packets_canonical_sha256,__import__('src.cloc_v2.phase1_scoring',fromlist=['canonical_sha256']).canonical_sha256(json.loads(packet_path.read_text())))
   self.assertEqual(runner.packets_canonical_sha256,__import__('src.cloc_v2.phase1_scoring',fromlist=['canonical_sha256']).canonical_sha256(runner.packets))
if __name__=='__main__':unittest.main()

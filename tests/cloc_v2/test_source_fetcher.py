import json,tempfile,unittest
from datetime import datetime,timezone
from pathlib import Path
from cloc.runtime.execution_policy import ExecutionPolicy
from src.cloc_v2.source_fetcher import SourceFetcher

A='0x'+'1'*40;B='0x'+'2'*40
class Response:
 def __init__(self,payload,status_code=200):self.payload=payload;self.status_code=status_code
 def json(self):return self.payload
class Session:
 def __init__(self,items):self.items=list(items);self.calls=0
 def get(self,*args,**kwargs):
  self.calls+=1;x=self.items.pop(0)
  if isinstance(x,Exception):raise x
  return x
def result(source='contract A{}',**more):
 x={'SourceCode':source,'ContractName':'A','ABI':'[]','Proxy':'0','Implementation':''};x.update(more)
 return {'status':'1','result':[x]}
class SourceFetcherTests(unittest.TestCase):
 def fetcher(self,session,td,allow=True):return SourceFetcher('eth',{'base_url':'https://example.invalid/v2/api','chainid':1,'key':'injected'},cache_root=td,policy=ExecutionPolicy(allow_explorer=allow),session=session,clock=lambda:datetime(2026,1,1,tzinfo=timezone.utc),max_retries=2)
 def test_single_multi_double_brace_abi_and_cache(self):
  with tempfile.TemporaryDirectory() as td:
   multi='{{"sources":{"a/A.sol":{"content":"contract A{}"},"lib/B.sol":{"content":"contract B{}"}}}}'
   s=Session([Response(result(multi))]);f=self.fetcher(s,td);r=f.acquire(A,origins=({'node_id':'n001'},))
   self.assertTrue(r.verified);self.assertEqual(r.source_format,'standard_json');self.assertEqual(2,len(r.files));self.assertEqual(1,s.calls)
   self.assertNotIn('injected',(Path(td)/'eth'/A/'manifest.json').read_text())
   cached=f.acquire(A);self.assertTrue(cached.cache_hit);self.assertEqual(1,s.calls)
   refreshed=Session([Response(result('contract Fresh{}'))]);f.session=refreshed;self.assertFalse(f.acquire(A,refresh=True).cache_hit);self.assertEqual(1,refreshed.calls)
   self.assertTrue((Path(td)/'eth'/A/'abi.json').exists())
 def test_offline_unverified_rate_timeout_and_path_safety(self):
  with tempfile.TemporaryDirectory() as td:
   self.assertEqual('offline_missing',self.fetcher(Session([]),td,False).acquire(A).status)
   self.assertEqual('unverified',self.fetcher(Session([Response(result(''))]),td).acquire(A).status)
  with tempfile.TemporaryDirectory() as td:
   self.assertEqual('rate_limited',self.fetcher(Session([Response({},429),Response({},429)]),td).acquire(A).status)
  with tempfile.TemporaryDirectory() as td:
   timed=self.fetcher(Session([TimeoutError(),TimeoutError()]),td).acquire(A);self.assertEqual('timeout',timed.status);self.assertEqual(2,timed.attempt_count)
  with tempfile.TemporaryDirectory() as td:
   bad='{"sources":{"../escape.sol":{"content":"bad"}}}'
   self.assertEqual('parse_error',self.fetcher(Session([Response(result(bad))]),td).acquire(A).status)
 def test_proxy_recursion_and_cycle(self):
  with tempfile.TemporaryDirectory() as td:
   proxy=result('contract P{}',Proxy='1',Implementation=B);impl=result('contract I{}')
   r=self.fetcher(Session([Response(proxy),Response(impl)]),td).acquire(A)
   self.assertEqual('verified',r.status);self.assertEqual('verified',r.proxy['resolution']);self.assertTrue((Path(td)/'eth'/B/'manifest.json').exists())
  with tempfile.TemporaryDirectory() as td:
   one=result('contract P{}',Proxy='1',Implementation=B);two=result('contract Q{}',Proxy='1',Implementation=A)
   r=self.fetcher(Session([Response(one),Response(two)]),td).acquire(A)
   self.assertEqual('verified',r.proxy['resolution'])
   child=json.loads((Path(td)/'eth'/B/'manifest.json').read_text())
   self.assertEqual('cycle',child['proxy']['resolution'])
  with tempfile.TemporaryDirectory() as td:
   f=self.fetcher(Session([Response(result('contract P{}',Proxy='1',Implementation=B))]),td);f.depth=0
   self.assertEqual('max_depth',f.acquire(A).proxy['resolution'])

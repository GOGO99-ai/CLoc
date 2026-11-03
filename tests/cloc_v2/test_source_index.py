import copy,json,tempfile,unittest
from pathlib import Path
from src.cloc_v2.source_cache import sha
from src.cloc_v2.source_index import build_source_index,ethereum_keccak256,parse_solidity_source,resolve_unique_internal_function_calls,selector_for_signature,serialize_source_index,write_source_index

A='0x'+'1'*40;B='0x'+'2'*40
SOURCE='''// function fake() {}
contract A {
 string constant S = "function no() { }";
 modifier onlyOwner() { _; }
 constructor(address owner) { if (owner != address(0)) { } }
 function foo(\n uint amount, address payable recipient\n ) external view onlyOwner returns (uint) { assembly { let x := 1 } return amount; }
 function foo(address who) external { }
 fallback() external payable { }
 receive() external payable { }
}
interface I { function ping(address who) external; }
library L { function help(bytes32 x) internal pure returns (bytes32) { return x; } }
'''
def acquisition():return {'schema_version':'source_acquisition_v1','chain':'eth','addresses':[{'requested_address':A}]}
def cache_contract(root,address,text=SOURCE,proxy=None):
 d=Path(root)/'eth'/address/'sources';d.mkdir(parents=True,exist_ok=True);p=d/'A.sol';p.write_text(text,encoding='utf8')
 manifest={'schema_version':'source_manifest_v1','status':'verified','verified':True,'files':[{'path':'sources/A.sol','sha256':sha(p)}],'proxy':proxy or {}}
 (d.parent/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
class SourceIndexTests(unittest.TestCase):
 def test_keccak_and_conservative_parser_specials_overloads_and_lines(self):
  self.assertEqual('a9059cbb',ethereum_keccak256(b'transfer(address,uint256)')[:4].hex());self.assertEqual('0xa9059cbb',selector_for_signature('transfer(address,uint256)'))
  parsed=parse_solidity_source(SOURCE);functions=parsed['functions'];self.assertEqual('PARSED',parsed['parse_status']);self.assertEqual(8,len(functions))
  self.assertEqual(['foo(uint256,address)','foo(address)'],[x['canonical_signature'] for x in functions if x['function_name']=='foo'])
  kinds={x['function_kind'] for x in functions};self.assertTrue({'CONSTRUCTOR','FALLBACK','RECEIVE','MODIFIER'}<=kinds)
  ping=next(x for x in functions if x['function_name']=='ping');self.assertIsNone(ping['body_start_line']);self.assertEqual(ping['signature_start_line'],13)
  foo=next(x for x in functions if x['canonical_signature']=='foo(uint256,address)');self.assertEqual(6,foo['signature_start_line']);self.assertEqual(8,foo['end_line'])
 def test_index_multifile_proxy_hash_and_determinism(self):
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A,proxy={'implementation_address':B});cache_contract(td,B,'contract Impl { function ping(address x) external {} }')
   raw=acquisition();before=copy.deepcopy(raw);one=build_source_index(raw,td);two=build_source_index(raw,td)
   self.assertEqual(raw,before);self.assertEqual(2,one['metadata']['address_count']);self.assertEqual(9,one['metadata']['function_count']);self.assertEqual(serialize_source_index(one),serialize_source_index(two))
   with tempfile.TemporaryDirectory() as out: p=Path(out)/'nested'/'index.json';write_source_index(one,p);self.assertEqual(serialize_source_index(one),p.read_text())
 def test_parse_failure_and_missing_cache_are_explicit(self):
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A,'contract Broken { function x() external {')
   indexed=build_source_index(acquisition(),td);self.assertEqual('PARSE_FAILED',indexed['addresses'][0]['files'][0]['parse_status'])
   missing=build_source_index(acquisition(),Path(td)/'other');self.assertEqual('SOURCE_MISSING',missing['addresses'][0]['status'])
 def test_unique_bare_internal_calls_resolve_but_overloads_and_members_do_not(self):
  source='''contract C {
 function entry() public { first(); obj.first(); overloaded(1); }
 function first() internal { second(); }
 function second() private {}
 function overloaded(uint256 x) internal {}
 function overloaded(address x) internal {}
}'''
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A,source);index=build_source_index(acquisition(),td)
   entry=next(item for item in index['addresses'][0]['functions'] if item['canonical_signature']=='entry()')
   result=resolve_unique_internal_function_calls(index,td,A,entry['function_id'])
   self.assertEqual(['first()'],[item['canonical_signature'] for item in result['calls']])
   self.assertIn('INTERNAL_FUNCTION_CALL_AMBIGUOUS_EXCLUDED',result['issue_codes'])
 def test_external_only_or_changed_source_is_not_invented_as_internal_call(self):
  source='''contract C { function entry() external { outside(); } function outside() external {} }'''
  with tempfile.TemporaryDirectory() as td:
   cache_contract(td,A,source);index=build_source_index(acquisition(),td)
   entry=next(item for item in index['addresses'][0]['functions'] if item['canonical_signature']=='entry()')
   result=resolve_unique_internal_function_calls(index,td,A,entry['function_id']);self.assertEqual([],result['calls'])
   (Path(td)/'eth'/A/'sources'/'A.sol').write_text('contract Changed {}',encoding='utf8')
   changed=resolve_unique_internal_function_calls(index,td,A,entry['function_id'])
   self.assertEqual(['INTERNAL_CALL_SOURCE_UNAVAILABLE'],changed['issue_codes'])

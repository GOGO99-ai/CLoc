from __future__ import annotations
import unittest
from src.cloc_v2.parameter_atoms import extract_parameter_atoms
from src.cloc_v2.transaction_graph import build_transaction_graph
def graph(params):return build_transaction_graph({'address':'0x'+'a'*40,'function':'f','params':params,'children':[]})
class ParameterAtomTests(unittest.TestCase):
 def test_uint_address_arrays_nested_and_display_hint(self):
  a='0x'+'B'*40;atoms=extract_parameter_atoms(graph(f'200000000 [2e8], {a}, [2, [{a}, 3]]'))
  got={(x.argument_path,x.atom_type,x.normalized_value) for x in atoms}
  self.assertIn(('arg[0]','UINT','200000000'),got);self.assertIn(('arg[1]','ADDRESS',a.lower()),got);self.assertIn(('arg[2][1][0]','ADDRESS',a.lower()),got);self.assertIn(('arg[2][1][1]','UINT','3'),got)
 def test_empty_selector_bytes_negative_boolean_are_ignored(self):
  self.assertEqual([],extract_parameter_atoms(graph('')));atoms=extract_parameter_atoms(graph('0x12345678, 0x'+'ab'*64+', -2, true'))
  self.assertEqual([],atoms)
 def test_zero_and_one_are_extracted_as_atoms(self):
  atoms=extract_parameter_atoms(graph('0, 1, 0x'+'0'*40));self.assertEqual(3,len(atoms))
if __name__=='__main__':unittest.main()

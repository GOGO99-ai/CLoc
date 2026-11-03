import json, tempfile, unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from src.cloc_v2.source_cache import atomic_json, cache_valid, norm_address, sha

ADDR='0x'+'1'*40
class SourceCacheTests(unittest.TestCase):
 def test_normalization_and_failed_ttl(self):
  self.assertEqual(norm_address(ADDR.upper()),ADDR)
  now=datetime(2026,1,8,tzinfo=timezone.utc)
  self.assertTrue(cache_valid({'status':'unverified','cached_at':(now-timedelta(days=6)).isoformat()},now))
  self.assertFalse(cache_valid({'status':'unverified','cached_at':(now-timedelta(days=8)).isoformat()},now))
  self.assertFalse(cache_valid({'status':'unverified','cached_at':now.isoformat()},now,refresh=True))
  with self.assertRaises(ValueError):norm_address('not-an-address')
 def test_verified_hash_and_atomic_json(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);f=root/'sources'/'A.sol';f.parent.mkdir();f.write_text('contract A{}')
   manifest={'status':'verified','cache_dir':str(root),'files':[{'path':'sources/A.sol','sha256':sha(f)}],'artifact_hashes':{}}
   self.assertTrue(cache_valid(manifest,datetime.now(timezone.utc)))
   f.write_text('changed');self.assertFalse(cache_valid(manifest,datetime.now(timezone.utc)))
   atomic_json(root/'x.json',{'b':1});self.assertEqual(json.loads((root/'x.json').read_text())['b'],1)

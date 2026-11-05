"""Fetch verified Solidity source and explicit proxy implementations."""
from __future__ import annotations
import json, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
from cloc.source.cache import address_dir, atomic_json, cache_valid, norm_address, sha
from cloc.source.models import SourceResult
from cloc.source.parsing import write_explorer_source_files
from cloc.runtime.redaction import redact_data

class _Response:

    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return json.loads(self._body.decode('utf-8'))

class _StdlibSession:

    def get(self, url, params, timeout):
        query = urlencode({k: v for k, v in params.items() if v is not None})
        sep = '&' if '?' in url else '?'
        try:
            with urlopen(url + sep + query, timeout=timeout) as response:
                return _Response(response.status, response.read())
        except HTTPError as exc:
            if exc.code == 429:
                return _Response(429, b'{}')
            raise

class SourceFetcher:
    """Acquire verified source with cache reuse and bounded proxy traversal."""

    def __init__(self, chain, config, cache_root='data/source_cache', policy=None, session=None, clock=None, max_proxy_depth=3, max_retries=3, timeout=20):
        self.chain = chain.lower()
        self.config = config
        self.root = Path(cache_root)
        self.policy = policy
        self.session = session or _StdlibSession()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.depth = max_proxy_depth
        self.retries = max_retries
        self.timeout = timeout

    def acquire(self, address, refresh=False, origins=(), visited=(), depth=0):
        """Resolve one address, following only explicit implementation addresses."""
        addr = norm_address(address)
        d = address_dir(self.root, self.chain, addr)
        mp = d / 'manifest.json'
        old = json.loads(mp.read_text()) if mp.exists() else None
        if cache_valid(old, self.clock(), refresh):
            cached = {k: v for k, v in old.items() if k in SourceResult.__dataclass_fields__}
            cached['cache_hit'] = True
            return SourceResult(**cached)
        if self.policy is not None and (not getattr(self.policy, 'allow_explorer', False)):
            return self._save(d, SourceResult(self.chain, addr, addr, 'offline_missing', failure={'code': 'offline_missing', 'message': 'Explorer disabled'}), origins)
        key = self.config.get('key')
        if not key:
            return self._save(d, SourceResult(self.chain, addr, addr, 'api_error', failure={'code': 'api_key_missing', 'message': 'API key missing'}), origins)
        data = None
        err = None
        attempts = 0
        for n in range(1, self.retries + 1):
            attempts = n
            try:
                params = {'module': 'contract', 'action': 'getsourcecode', 'address': addr, 'apikey': key, 'chainid': self.config.get('chainid')}
                r = self.session.get(self.config['base_url'], params=params, timeout=self.timeout)
                if getattr(r, 'status_code', 200) == 429:
                    err = ('rate_limited', 'HTTP 429')
                    continue
                data = r.json()
                break
            except TimeoutError:
                err = ('timeout', 'timeout')
            except OSError as e:
                err = ('network_error', str(e))
            except ValueError as e:
                err = ('parse_error', str(e))
            if n < self.retries:
                time.sleep(0)
        if data is None:
            code, message = err or ('network_error', 'no Explorer response')
            return self._save(d, SourceResult(self.chain, addr, addr, code, failure={'code': code, 'message': message}, attempt_count=attempts), origins)
        # Error responses may echo the request URL or API key.
        failed = data.get('status') != '1' or not data.get('result')
        atomic_json(d / 'explorer_response.json', self._safe_failure(data) if failed else data)
        if data.get('status') != '1' or not data.get('result'):
            text = (str(data.get('message', '')) + ' ' + str(data.get('result', ''))).lower()
            status = 'rate_limited' if 'rate limit' in text else 'unverified' if 'not verified' in text else 'api_error'
            return self._save(d, SourceResult(self.chain, addr, addr, status, failure={'code': status, 'message': str(data.get('result') or data.get('message'))}, attempt_count=attempts), origins)
        x = data['result'][0]
        if not isinstance(x, dict):
            return self._save(d, SourceResult(self.chain, addr, addr, 'parse_error', failure={'code': 'parse_error', 'message': 'Explorer result is not an object'}, attempt_count=attempts), origins)
        meta = {'contract_name': x.get('ContractName') or 'Unknown', 'compiler_version': x.get('CompilerVersion'), 'proxy': str(x.get('Proxy')) == '1', 'implementation': x.get('Implementation') or None}
        atomic_json(d / 'metadata.json', meta)
        abi = x.get('ABI')
        abi_path = None
        if abi and 'not verified' not in str(abi).lower():
            abi_path = 'abi.json'
            try:
                atomic_json(d / abi_path, json.loads(abi))
            except ValueError:
                (d / abi_path).write_text(str(abi), encoding='utf8')
        src = x.get('SourceCode') or ''
        if not src.strip():
            return self._save(d, SourceResult(self.chain, addr, addr, 'unverified', contract_name=meta['contract_name'], abi_path=abi_path, proxy=meta, attempt_count=attempts, artifact_hashes=self._artifact_hashes(d, abi_path)), origins)
        try:
            files, fmt = self._write_sources(d, src, meta['contract_name'])
        except Exception as e:
            return self._save(d, SourceResult(self.chain, addr, addr, 'parse_error', contract_name=meta['contract_name'], failure={'code': 'parse_error', 'message': str(e)}, attempt_count=attempts, artifact_hashes=self._artifact_hashes(d, abi_path)), origins)
        proxy = {'is_proxy': meta['proxy'], 'implementation_address': meta['implementation'], 'implementation_manifest_sha256': None}
        if meta['proxy'] and meta['implementation']:
            if depth >= self.depth:
                proxy['resolution'] = 'max_depth'
            elif norm_address(meta['implementation']) in visited + (addr,):
                proxy['resolution'] = 'cycle'
            else:
                child = self.acquire(meta['implementation'], refresh, (), visited + (addr,), depth + 1)
                cp = address_dir(self.root, self.chain, child.requested_address) / 'manifest.json'
                proxy['implementation_manifest_sha256'] = sha(cp) if cp.exists() else None
                proxy['resolution'] = child.status
        return self._save(d, SourceResult(self.chain, addr, addr, 'verified', True, source_format=fmt, contract_name=meta['contract_name'], files=files, abi_path=abi_path, proxy=proxy, attempt_count=attempts, artifact_hashes=self._artifact_hashes(d, abi_path)), origins)

    def _write_sources(self, d, src, name):
        root = d / 'sources'
        paths, fmt = write_explorer_source_files(src, root, name)
        files = []
        for rel in paths:
            p = root / rel
            files.append({'path': 'sources/' + rel.replace('\\', '/'), 'sha256': sha(p), 'bytes': p.stat().st_size})
        return (files, fmt)

    def _artifact_hashes(self, d, abi_path):
        names = ['explorer_response.json', 'metadata.json'] + ([abi_path] if abi_path else [])
        return {name: sha(d / name) for name in names if (d / name).exists()}

    def _save(self, d, r, origins):
        """Persist acquisition status and source provenance for later reuse."""
        d.mkdir(parents=True, exist_ok=True)
        r.origins = list(origins)
        if r.failure:
            r.failure = self._safe_failure(r.failure)
        payload = r.to_dict()
        payload.update({'schema_version': 'source_manifest_v1', 'cache_dir': str(d), 'cached_at': self.clock().isoformat()})
        atomic_json(d / 'manifest.json', payload)
        return r

    def _safe_failure(self, value):
        return redact_data(
            value, secrets=(self.config.get('key'),),
            endpoints=(self.config.get('base_url'), self.config.get('rpc_url')),
        )

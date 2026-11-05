"""Validate source caches and write source artifacts atomically."""
from __future__ import annotations
import hashlib, json, os, tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

def norm_address(a):
    """Validate and normalize a hexadecimal contract address."""
    a = str(a).lower()
    if not (a.startswith('0x') and len(a) == 42 and all((c in '0123456789abcdef' for c in a[2:]))):
        raise ValueError(f'invalid address: {a}')
    return a

def address_dir(root, chain, address):
    """Return the chain-specific cache directory for one address."""
    return Path(root) / chain.lower() / norm_address(address)

def sha(path):
    """Hash an artifact so cached source can be checked for changes."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def atomic_json(path, data):
    """Replace a JSON artifact only after its complete contents are written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile('w', encoding='utf8', dir=p.parent, delete=False)
    try:
        with h:
            h.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        os.replace(h.name, p)
    except Exception:
        Path(h.name).unlink(missing_ok=True)
        raise

def cache_valid(manifest, now, refresh=False, ttl_days=7):
    """Reuse verified source or unexpired negative-cache results."""
    if refresh or not manifest:
        return False
    if manifest.get('status') == 'offline_missing':
        return False
    if manifest.get('status') == 'verified':
        root = Path(manifest.get('cache_dir', ''))
        files = manifest.get('files', [])
        artifacts = manifest.get('artifact_hashes', {})
        own_ok = bool(files) and all(((root / f['path']).exists() and sha(root / f['path']) == f['sha256'] for f in files)) and all(((root / path).exists() and sha(root / path) == digest for path, digest in artifacts.items()))
        proxy = manifest.get('proxy', {})
        implementation = proxy.get('implementation_address')
        expected = proxy.get('implementation_manifest_sha256')
        implementation_ok = not implementation or not expected or ((root.parent / str(implementation).lower() / 'manifest.json').exists() and sha(root.parent / str(implementation).lower() / 'manifest.json') == expected)
        return own_ok and implementation_ok
    stamp = manifest.get('cached_at')
    try:
        return datetime.fromisoformat(stamp) <= now <= datetime.fromisoformat(stamp) + timedelta(days=ttl_days)
    except Exception:
        return False

"""Remove credentials and private endpoints from diagnostic output."""
import json
import re
from urllib.parse import parse_qsl, quote, quote_plus, urlsplit


_URL = re.compile(r"(?:https?|wss?)://[^\s\"'<>]+", re.IGNORECASE)


def redact_text(value, *, secrets=(), endpoints=()) -> str:
    """Hide supplied secrets, including encoded values and split URL details."""
    tokens = {item for item in secrets if isinstance(item, str) and item}
    for endpoint in endpoints:
        if not isinstance(endpoint, str) or not endpoint:
            continue
        tokens.add(endpoint)
        try:
            parts = urlsplit(endpoint)
            tokens.update(item for item in (parts.hostname, parts.username, parts.password) if item)
            tokens.update(item for _, item in parse_qsl(parts.query) if item)
            tokens.update(item for item in parts.path.split("/") if len(item) >= 12)
        except ValueError:
            pass
    variants = set()
    for token in tokens:
        variants.update((token, quote(token, safe=""), quote_plus(token), json.dumps(token)[1:-1]))
    text = str(value)
    for token in sorted(variants, key=len, reverse=True):
        text = text.replace(token, "[REDACTED]")
    return _URL.sub("[REDACTED_URL]", text)


def redact_data(value, *, secrets=(), endpoints=()):
    """Sanitize error payloads while preserving their JSON structure."""
    if isinstance(value, str):
        return redact_text(value, secrets=secrets, endpoints=endpoints)
    if isinstance(value, dict):
        return {
            redact_text(key, secrets=secrets, endpoints=endpoints):
            redact_data(item, secrets=secrets, endpoints=endpoints)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_data(item, secrets=secrets, endpoints=endpoints) for item in value]
    return value

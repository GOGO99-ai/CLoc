"""JSON client for OpenAI-compatible Chat Completions endpoints."""
from dataclasses import dataclass, field
import hashlib
import json
import os
import time
from urllib.parse import urlsplit
import requests


class LLMCallError(RuntimeError):
    """Expose fixed categories without provider URLs or credentials."""


@dataclass(frozen=True)
class JSONChatClient:
    model: str = "gpt-4o"
    base_url: str = field(default="https://api.openai.com/v1", repr=False)
    api_key: str = field(default="", repr=False)
    timeout: int = 120
    max_input_bytes: int = 120_000
    token_parameter: str = "max_completion_tokens"

    @classmethod
    def from_environment(cls, *, model=None, timeout=120, token_parameter=None):
        return cls(model=model or os.environ.get("CLOC_LLM_MODEL") or "gpt-4o",
            base_url=os.environ.get("CLOC_LLM_BASE_URL") or "https://api.openai.com/v1",
            api_key=os.environ.get("CLOC_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
            timeout=timeout,
            token_parameter=token_parameter or os.environ.get("CLOC_LLM_TOKEN_PARAMETER") or "max_completion_tokens")

    def complete(self, prompt, *, output_tokens):
        if not self.api_key:
            raise LLMCallError("LLM_API_KEY_MISSING")
        if (not self.model or self.token_parameter not in {"max_tokens", "max_completion_tokens"}
                or type(output_tokens) is not int or output_tokens < 1):
            raise LLMCallError("LLM_CONFIGURATION_INVALID")
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        if urlsplit(endpoint).scheme not in {"http", "https"} or self.timeout <= 0:
            raise LLMCallError("LLM_CONFIGURATION_INVALID")
        payload = {"model": self.model,
            "messages": [{"role": "system", "content": prompt.system_message},
                         {"role": "user", "content": prompt.user_message}],
            "response_format": {"type": "json_object"}, self.token_parameter: output_tokens}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        if len(encoded) > self.max_input_bytes:
            raise LLMCallError("LLM_INPUT_BUDGET_EXCEEDED")
        digest = hashlib.sha256(encoded).hexdigest()
        for attempt in (1, 2):
            try:
                response = requests.post(endpoint, json=payload,
                    headers={"Authorization": "Bearer " + self.api_key}, timeout=self.timeout,
                    allow_redirects=False)
            except (requests.RequestException, OSError):
                if attempt == 1:
                    time.sleep(1)
                    continue
                raise LLMCallError("LLM_TRANSPORT_FAILED") from None
            if (response.status_code == 429 or 500 <= response.status_code < 600) and attempt == 1:
                time.sleep(1)
                continue
            if response.status_code != 200:
                raise LLMCallError(f"LLM_HTTP_{response.status_code}")
            try:
                body = response.json()
                choices = body["choices"]
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError
                choice = choices[0]
                if choice["finish_reason"] != "stop":
                    raise LLMCallError("LLM_RESPONSE_INCOMPLETE")
                message = choice["message"]
                if message.get("refusal") or message.get("role") != "assistant":
                    raise LLMCallError("LLM_RESPONSE_UNAVAILABLE")
                content = json.loads(message["content"])
                if not isinstance(content, dict):
                    raise ValueError
            except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                raise LLMCallError("LLM_RESPONSE_INVALID") from None
            usage = body.get("usage") or {}
            usage = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                     if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0}
            return content, {"model": self.model, "request_sha256": digest,
                             "request_count": attempt, "input_bytes": len(encoded),
                             "output_token_budget": output_tokens, "usage": usage}
        raise LLMCallError("LLM_TRANSPORT_FAILED")

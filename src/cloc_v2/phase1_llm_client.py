"""Bounded, auditable phase-1 LLM client. Dry-run is the default execution mode."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from .phase1_payload_v3 import audit_llm_payload
from .phase1_prompting import PROMPT_VERSION_V3_1, build_preview_v3_1, preview_metrics_v3_1
from .phase1_scoring import FOCUS, canonical_sha256, total_score, write_text

TX_DIRECTORY = re.compile(r"0x[0-9a-fA-F]{64}$")


class ValidationError(ValueError): pass

class TransportError(RuntimeError):
    def __init__(self, kind: str, message: str, status_code: int | None = None):
        super().__init__(message); self.kind = kind; self.status_code = status_code


class Transport(Protocol):
    def send(self, request: dict, api_key: str, timeout_seconds: float) -> dict: ...


@dataclass(frozen=True)
class ClientConfig:
    provider: str = "UNSPECIFIED"
    model: str = "UNSPECIFIED"
    temperature: str | float = "UNSPECIFIED"
    max_output_tokens: str | int = "UNSPECIFIED"
    timeout_seconds: str | float = "UNSPECIFIED"
    max_attempts: int = 3
    concurrency: int = 1
    omit_temperature: bool = False

    def request_parameters(self) -> dict:
        return {"temperature": self.temperature, "max_output_tokens": self.max_output_tokens, "timeout_seconds": self.timeout_seconds, "omit_temperature": self.omit_temperature, "max_attempts": self.max_attempts, "concurrency": self.concurrency}

    def validate_execute(self) -> None:
        if self.provider != "openai_compatible": raise ValueError("--execute requires provider=openai_compatible")
        if not self.model or self.model == "UNSPECIFIED": raise ValueError("--execute requires an explicit model")
        if self.max_output_tokens == "UNSPECIFIED": raise ValueError("--execute requires --max-output-tokens")
        if self.timeout_seconds == "UNSPECIFIED": raise ValueError("--execute requires --timeout-seconds")
        if not self.omit_temperature and self.temperature == "UNSPECIFIED": raise ValueError("--execute requires --temperature or --omit-temperature")
        if self.max_attempts < 1 or self.concurrency < 1: raise ValueError("max_attempts and concurrency must be positive")


class OpenAICompatibleTransport:
    """Minimal Chat Completions transport. Authorization is never retained in artifacts."""
    def __init__(self, base_url: str): self.base_url = base_url.rstrip("/")
    def send(self, request: dict, api_key: str, timeout_seconds: float) -> dict:
        payload = {"model": request["model"], "messages": request["messages"], "max_tokens": request["parameters"]["max_output_tokens"]}
        if not request["parameters"]["omit_temperature"]: payload["temperature"] = request["parameters"]["temperature"]
        response_format = request["parameters"].get("response_format")
        if isinstance(response_format, dict): payload["response_format"] = response_format
        thinking = request["parameters"].get("thinking")
        if isinstance(thinking, dict): payload["thinking"] = thinking
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key}, method="POST")
        try:
            with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response: return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise TransportError("http", f"HTTP {exc.code}", exc.code) from exc
        except TimeoutError as exc: raise TransportError("timeout", "timeout") from exc
        except OSError as exc: raise TransportError("connection", str(exc)) from exc
        except json.JSONDecodeError as exc: raise TransportError("invalid_provider_json", "provider returned invalid JSON") from exc


def sha256_file(path: str | Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def validate_output_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not any(TX_DIRECTORY.fullmatch(parent.name) for parent in (resolved, *resolved.parents)):
        raise ValueError("output-dir must be inside an explicit transaction run directory")
    return resolved


def _batch_maps(packets: dict, batch_id: str) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    batch = next((item for item in packets["batches"] if item["batch_id"] == batch_id), None)
    if not batch: raise ValidationError("unknown batch_id")
    chain_ids: set[str] = set(); nodes: dict[str, set[str]] = {}; edges: dict[str, set[str]] = {}
    for chain in batch["llm_payload"]["chains"]:
        chain_id = chain[0]; chain_ids.add(chain_id); node_ids = set(chain[4]); edge_ids = set(chain[5])
        for support in chain[6]: edge_ids.update(item[0] for item in support[2])
        for branch in chain[7]: node_ids.update(branch[2]); edge_ids.update(branch[3])
        nodes[chain_id] = node_ids; edges[chain_id] = edge_ids
    return chain_ids, nodes, edges


def validate_batch_response(response: dict, packets: dict, batch_id: str, prompt_version: str = PROMPT_VERSION_V3_1) -> dict:
    if set(response) != {"batch_id", "responses"} or response.get("batch_id") != batch_id or not isinstance(response.get("responses"), list): raise ValidationError("response envelope is invalid")
    expected, allowed_nodes, allowed_edges = _batch_maps(packets, batch_id); seen: set[str] = set()
    source_limit = 3 if prompt_version == PROMPT_VERSION_V3_1 else 5
    response_keys = {"chain_id", "scores", "total_score", "concise_rationale", "key_node_ids", "key_edge_ids", "counter_evidence", "uncertainties", "source_requests"}
    score_keys = {"result_relevance", "asset_anomaly", "state_dependency", "parameter_influence", "control_interaction", "causal_coherence"}
    source_keys = {"node_id", "request_reason", "focus", "priority"}
    for item in response["responses"]:
        if not isinstance(item, dict) or set(item) != response_keys: raise ValidationError("unknown or missing response field")
        chain_id = item["chain_id"]
        if chain_id not in expected or chain_id in seen: raise ValidationError("unknown or duplicate chain_id")
        seen.add(chain_id)
        scores = item["scores"]
        if not isinstance(scores, dict) or set(scores) != score_keys or any(not isinstance(value, int) or not 0 <= value <= 5 for value in scores.values()): raise ValidationError("invalid six-dimensional scores")
        expected_total = f"{total_score(scores):.2f}"
        if not isinstance(item["total_score"], str) or item["total_score"] != expected_total: raise ValidationError("incorrect total_score")
        if not isinstance(item["concise_rationale"], str) or any(not isinstance(item[key], list) for key in ("key_node_ids", "key_edge_ids", "counter_evidence", "uncertainties", "source_requests")): raise ValidationError("invalid response value types")
        if not set(item["key_node_ids"]) <= allowed_nodes[chain_id] or not set(item["key_edge_ids"]) <= allowed_edges[chain_id]: raise ValidationError("evidence reference outside chain")
        if len(item["source_requests"]) > source_limit: raise ValidationError("too many source_requests for prompt version")
        for request in item["source_requests"]:
            if not isinstance(request, dict) or set(request) != source_keys or request["node_id"] not in allowed_nodes[chain_id] or not isinstance(request["request_reason"], str) or request["focus"] not in FOCUS or request["priority"] not in {1, 2, 3}: raise ValidationError("invalid source_request")
    if seen != expected: raise ValidationError("missing chain_id")
    return response


def build_request(packets: dict, batch_id: str, config: ClientConfig, packets_file_sha256: str | None, packets_canonical_sha256: str, prompt_file_sha256: str) -> dict:
    preview = build_preview_v3_1(packets, batch_id)
    messages = preview["messages"]
    payload_text = messages[1]["content"]
    leak = {**audit_llm_payload({"batches": [{"llm_payload": next(item for item in packets["batches"] if item["batch_id"] == batch_id)["llm_payload"]}]}), "message_evm_address_count": len(re.findall(r"0x[0-9a-fA-F]{40}", payload_text)), "message_arguments_raw_count": payload_text.count("arguments_raw"), "message_resolver_metadata_count": payload_text.count("resolver_metadata")}
    if any(leak.values()): raise ValidationError("LLM-visible payload failed leakage audit")
    base = {"schema_version": "phase1_llm_request_v1", "batch_id": batch_id, "provider": config.provider, "model": config.model, "parameters": config.request_parameters(), "messages": messages, "packets_file_sha256": packets_file_sha256, "packets_canonical_sha256": packets_canonical_sha256, "prompt_file_sha256": prompt_file_sha256, "messages_canonical_sha256": canonical_sha256(messages), "prompt_version": PROMPT_VERSION_V3_1, "leak_audit": leak}
    return {**base, "request_canonical_sha256": canonical_sha256(base)}


def _retryable(error: TransportError) -> bool:
    return error.kind in {"timeout", "connection"} or error.status_code == 429 or (error.status_code is not None and 500 <= error.status_code <= 599)


class Phase1Runner:
    def __init__(self, packets: dict, prompt_path: str | Path, output_dir: str | Path, config: ClientConfig, packets_path: str | Path | None = None, batch_id: str | None = None):
        if packets.get("schema_version") != "phase1_scoring_packets_v3": raise ValueError("expected phase1_scoring_packets_v3")
        self.packets, self.prompt_path, self.output_dir, self.config = packets, Path(prompt_path), validate_output_dir(output_dir), config
        self.packets_file_sha256 = sha256_file(packets_path) if packets_path is not None else None
        self.packets_canonical_sha256 = canonical_sha256(packets)
        self.prompt_file_sha256 = sha256_file(self.prompt_path)
        self.all_batch_ids = [batch["batch_id"] for batch in self.packets["batches"]]
        if batch_id is not None and batch_id not in self.all_batch_ids: raise ValidationError(f"unknown batch_id: {batch_id}")
        self.batch_id = batch_id

    def requests(self) -> list[dict]:
        selected = (batch for batch in self.packets["batches"] if self.batch_id is None or batch["batch_id"] == self.batch_id)
        return [build_request(self.packets, batch["batch_id"], self.config, self.packets_file_sha256, self.packets_canonical_sha256, self.prompt_file_sha256) for batch in selected]

    def _manifest_context(self) -> dict:
        return {"packets_file_sha256": self.packets_file_sha256, "packets_canonical_sha256": self.packets_canonical_sha256, "prompt_file_sha256": self.prompt_file_sha256, "config": {"provider": self.config.provider, "model": self.config.model, "parameters": self.config.request_parameters()}, "selected_batch_id": self.batch_id}

    def _write_partial_manifest(self, mode: str, entries: list[dict]) -> dict:
        manifest = {"schema_version": "phase1_llm_partial_manifest_v1", "mode": mode, **self._manifest_context(), "batches": entries, "complete": False}
        write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", self.output_dir / "partial_manifest.json")
        return manifest

    def dry_run(self) -> dict:
        requests = self.requests(); stage = self.output_dir / ".staging" / "requests"
        for request in requests: write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", stage / f"{request['batch_id']}.request.json")
        final = self.output_dir / "requests"; final.mkdir(parents=True, exist_ok=True)
        for staged in sorted(stage.glob("*.json")): os.replace(staged, final / staged.name)
        entries = [{"batch_id": request["batch_id"], "request_canonical_sha256": request["request_canonical_sha256"], **preview_metrics_v3_1({"messages": request["messages"]})} for request in requests]
        if self.batch_id is not None: return self._write_partial_manifest("dry_run", entries)
        manifest = {"schema_version": "phase1_llm_dry_run_v1", "mode": "dry_run", **self._manifest_context(), "batches": entries, "complete": False}
        write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", self.output_dir / "dry_run_manifest.json")
        return manifest

    def _execute_one(self, request: dict, transport: Transport, api_key: str, resume: bool) -> tuple[str, bool, str | None]:
        validated_path = self.output_dir / "validated_responses" / f"{request['batch_id']}.validated.json"
        if resume and validated_path.exists():
            cached = json.loads(validated_path.read_text(encoding="utf-8"))
            if cached.get("request_canonical_sha256") == request["request_canonical_sha256"]: return request["batch_id"], True, None
        attempts = []
        for number in range(1, self.config.max_attempts + 1):
            try:
                provider_response = transport.send(request, api_key, float(self.config.timeout_seconds)); attempts.append({"attempt": number, "status": "success", "wait_seconds": 0})
                raw = {"schema_version": "phase1_llm_raw_response_v1", "batch_id": request["batch_id"], "request_canonical_sha256": request["request_canonical_sha256"], "provider_request_id": provider_response.get("id"), "model": provider_response.get("model"), "finish_reason": ((provider_response.get("choices") or [{}])[0].get("finish_reason")), "usage": provider_response.get("usage"), "provider_response": provider_response, "attempts": attempts}
                raw_path = self.output_dir / "raw_responses" / f"{request['batch_id']}.raw.json"; write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", raw_path)
                content = ((provider_response.get("choices") or [{}])[0].get("message") or {}).get("content")
                try: validated = validate_batch_response(json.loads(content), self.packets, request["batch_id"])
                except (json.JSONDecodeError, ValidationError) as error: return request["batch_id"], False, f"semantic_failure:{error}"
                final = {"schema_version": "phase1_llm_validated_response_v1", "batch_id": request["batch_id"], "request_canonical_sha256": request["request_canonical_sha256"], "response": validated}
                write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", validated_path); return request["batch_id"], True, None
            except TransportError as error:
                wait = min(2 ** (number - 1), 8) if _retryable(error) and number < self.config.max_attempts else 0
                attempts.append({"attempt": number, "status": "transport_error", "error_type": error.kind, "status_code": error.status_code, "wait_seconds": wait})
                if not _retryable(error) or number == self.config.max_attempts:
                    raw = {"schema_version": "phase1_llm_raw_response_v1", "batch_id": request["batch_id"], "request_canonical_sha256": request["request_canonical_sha256"], "attempts": attempts, "transport_failure": {"error_type": error.kind, "status_code": error.status_code}}
                    write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", self.output_dir / "raw_responses" / f"{request['batch_id']}.raw.json"); return request["batch_id"], False, f"transport_failure:{error.kind}"
                time.sleep(wait)
        raise AssertionError("bounded retry loop escaped")

    def execute(self, transport: Transport, api_key: str, resume: bool = False) -> dict:
        self.config.validate_execute(); requests = self.requests()
        for request in requests: write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", self.output_dir / ".staging" / "requests" / f"{request['batch_id']}.request.json")
        final_requests = self.output_dir / "requests"; final_requests.mkdir(parents=True, exist_ok=True)
        for staged in sorted((self.output_dir / ".staging" / "requests").glob("*.json")): os.replace(staged, final_requests / staged.name)
        results = []
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            futures = [executor.submit(self._execute_one, request, transport, api_key, resume) for request in requests]
            for future in as_completed(futures): results.append(future.result())
        results.sort()
        complete = self.batch_id is None and len(results) == len(self.all_batch_ids) and all(success for _, success, _ in results)
        entries = [{"batch_id": batch, "validated": success, "error": error} for batch, success, error in results]
        manifest = {"schema_version": "phase1_llm_execution_manifest_v1", "mode": "execute", **self._manifest_context(), "results": entries, "complete": complete}
        if complete: write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", self.output_dir / "complete_manifest.json")
        elif self.batch_id is not None: self._write_partial_manifest("execute", entries)
        else: write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", self.output_dir / "execution_manifest.json")
        return manifest

"""Deterministic, offline role-separated prompts for phase-1 scoring."""
from __future__ import annotations

import json
from pathlib import Path

from .phase1_scoring import canonical_sha256, heuristic_tokens, write_text

PROMPT_VERSION = "phase1_prompt_v3"
PROMPT_VERSION_V3_1 = "phase1_prompt_v3_1"
SYSTEM_PROMPT_V3 = """You are an evidence-constrained blockchain execution analyst.

Your task is to evaluate candidate causal chains according to how plausibly they explain the observed transaction result.

Use only the supplied execution evidence. Do not assume source code, ground truth, vulnerability type, attack reports, protocol intent, or facts absent from the input.

Evidence hierarchy:

- OBSERVED: directly recorded in the execution evidence.
- STRONG_ASSOCIATION: strongly supported relationship, but not independent proof of causality.
- HYPOTHESIS: plausible interpretation requiring further evidence.

Mandatory restrictions:

- ORDER establishes execution order, not causality.
- PARAM, STATE, and ASSET associations do not independently prove causality.
- Function names and address aliases are navigation aids, not vulnerability evidence.
- Repeated calls, transfers, and balance reads are not automatically anomalous.
- Do not infer a named vulnerability merely from a familiar pattern or function name.
- Do not invent nodes, edges, values, functions, addresses, source code, or execution behavior.
- Do not reward chain length by itself.
- Perform detailed reasoning internally.
- Return only the required JSON.
- Do not output hidden chain-of-thought, Markdown, or fields outside the schema.
"""

USER_INSTRUCTIONS_V3 = """Evaluate every candidate causal chain in the supplied batch for: relevance to its specified result; anomalous asset flow; dependency between state changes and later behavior; influence of parameters or return values; material cross-contract control interaction; and overall causal coherence.

Perform this workflow internally only; do not output the detailed reasoning:
1. Identify the specific result the chain is intended to explain.
2. Abstract nodes and function labels into high-level mechanisms such as asset movement, state mutation and dependent use, parameter propagation, cross-contract control interaction, callback/nested execution, or accounting/value-calculation dependency.
3. Return to node_id and edge_id evidence and verify that the mechanism is actually supported.
4. Identify the strongest supporting evidence.
5. Identify existing counter-evidence and unresolved uncertainties.
6. Score all six dimensions independently.
7. Request source only when missing source information could materially change at least one score or the final ranking.

Use this integer scale for every dimension: 0=no relevant evidence or conflicting evidence; 1=very weak support, mainly generic behavior or HYPOTHESIS; 2=limited indirect support with a missing important link; 3=meaningful support with important uncertainty or an alternative explanation; 4=strong, mostly connected support with only limited missing evidence; 5=direct, specific, highly connected support with no important unresolved link.

Dimensions:
- result_relevance: directly explains the specified result; proximity alone is insufficient.
- asset_anomaly: relevant abnormal direction, amount relation, balance change, or transfer sequence; ordinary transfers are not automatically anomalous.
- state_dependency: later state reads or behavior truly depend on a prior state change; ORDER alone is insufficient.
- parameter_influence: a parameter, return value, amount, or address propagates along an explainable path and materially affects later behavior.
- control_interaction: cross-contract calls, callbacks, nested calls, or delegatecall materially participate; ordinary CALL is insufficient.
- causal_coherence: direction and connections are complete, branches support the explanation, HYPOTHESIS is not over-relied on, and alternatives are considered.

Evidence constraints:
- key_node_ids and key_edge_ids must belong to that chain. Prefer OBSERVED over STRONG_ASSOCIATION over HYPOTHESIS.
- counter_evidence is existing supplied evidence that weakens this explanation, not invented contrary evidence. Return [] when none exists.
- uncertainties are missing information that prevents determination. Keep them distinct from counter_evidence and do not repeat the same sentence across rationale, counter_evidence, and uncertainties.
- Never infer a vulnerability from a function name.

source_requests are optional and must use exactly the existing schema. Include at most three per chain, and only if the request node belongs to the chain, the missing source could materially change a score or ranking, the request asks one concrete code question, and runtime evidence cannot answer it. Valid focuses are access_control, validation, arithmetic, price_calculation, state_update, external_call, transfer_amount, callback_order. Do not request a whole chain, source merely because a function name looks suspicious, or a vulnerability classification.

Return JSON only, exactly in this shape. Return each supplied chain_id exactly once; do not rank chains and do not add fields:
{
  "batch_id": "<batch_id>",
  "responses": [
    {
      "chain_id": "<chain_id>",
      "scores": {
        "result_relevance": 0,
        "asset_anomaly": 0,
        "state_dependency": 0,
        "parameter_influence": 0,
        "control_interaction": 0,
        "causal_coherence": 0
      },
      "total_score": "0.00",
      "concise_rationale": "",
      "key_node_ids": [],
      "key_edge_ids": [],
      "counter_evidence": [],
      "uncertainties": [],
      "source_requests": [
        {"node_id": "<chain node id>", "request_reason": "<concrete unresolved code question>", "focus": "validation", "priority": 1}
      ]
    }
  ]
}

Scores must be integers from 0 to 5. Compute total_score exactly as ((result_relevance*20 + asset_anomaly*15 + state_dependency*15 + parameter_influence*15 + control_interaction*15 + causal_coherence*20) / 5), formatted with exactly two decimal places. concise_rationale must concisely distinguish supplied evidence from inference.
"""
USER_INSTRUCTIONS_V3_1 = USER_INSTRUCTIONS_V3.replace('''      "source_requests": [
        {"node_id": "<chain node id>", "request_reason": "<concrete unresolved code question>", "focus": "validation", "priority": 1}
      ]''', '''      "source_requests": []''').replace("\nScores must be integers", '''\nIf source is genuinely needed, every source_requests array item must be exactly:
{"node_id": "<chain node id>", "request_reason": "<one concrete unresolved code question>", "focus": "<one allowed focus>", "priority": 1}
Priority means 1=high: likely to materially change a score or ranking; 2=medium: likely to resolve an important uncertainty; 3=low: supplementary confirmation only. Return [] when no source request is necessary.

Scores must be integers''')


def render_messages(packets: dict, batch_id: str) -> list[dict[str, str]]:
    if packets.get("schema_version") != "phase1_scoring_packets_v2":
        raise ValueError("expected phase1_scoring_packets_v2")
    batch = next((item for item in packets.get("batches", []) if item.get("batch_id") == batch_id), None)
    if batch is None:
        raise ValueError(f"unknown batch_id: {batch_id}")
    payload = json.dumps(batch["llm_payload"], ensure_ascii=False, separators=(",", ":"))
    user = USER_INSTRUCTIONS_V3 + "\nBATCH_ID: " + batch_id + "\nBATCH_LLM_PAYLOAD_JSON:\n" + payload
    return [{"role": "system", "content": SYSTEM_PROMPT_V3}, {"role": "user", "content": user}]


def prompt_document() -> str:
    return "# Phase 1 Prompt v3\n\n## System prompt\n\n" + SYSTEM_PROMPT_V3 + "\n## User prompt fixed instructions\n\n" + USER_INSTRUCTIONS_V3


def build_preview(packets: dict, batch_id: str) -> dict:
    messages = render_messages(packets, batch_id)
    return {
        "batch_id": batch_id,
        "messages": messages,
        "model_parameters": {"model": "UNSPECIFIED", "temperature": "UNSPECIFIED", "max_output_tokens": "UNSPECIFIED"},
        "input_sha256": {"phase1_scoring_packets_v2": canonical_sha256(packets)},
        "prompt_version": PROMPT_VERSION,
    }


def preview_metrics(preview: dict) -> dict:
    system, user = preview["messages"]
    return {"system_prompt_characters": len(system["content"]), "fixed_user_instruction_characters": len(USER_INSTRUCTIONS_V3), "user_message_characters": len(user["content"]), "estimated_tokens": heuristic_tokens(system["content"] + user["content"]), "token_estimate_note": "heuristic ceil(ASCII/3)+non-ASCII; not exact"}


def write_preview(preview: dict, path: str | Path) -> None:
    write_text(json.dumps(preview, ensure_ascii=False, indent=2) + "\n", path)


def render_messages_v3_1(packets: dict, batch_id: str) -> list[dict[str, str]]:
    if packets.get("schema_version") != "phase1_scoring_packets_v3": raise ValueError("expected phase1_scoring_packets_v3")
    batch = next((item for item in packets.get("batches", []) if item.get("batch_id") == batch_id), None)
    if batch is None: raise ValueError(f"unknown batch_id: {batch_id}")
    payload = json.dumps(batch["llm_payload"], ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system", "content": SYSTEM_PROMPT_V3}, {"role": "user", "content": USER_INSTRUCTIONS_V3_1 + "\nBATCH_ID: " + batch_id + "\nBATCH_LLM_PAYLOAD_JSON:\n" + payload}]


def prompt_document_v3_1() -> str:
    return "# Phase 1 Prompt v3.1\n\n## System prompt\n\n" + SYSTEM_PROMPT_V3 + "\n## User prompt fixed instructions\n\n" + USER_INSTRUCTIONS_V3_1


def build_preview_v3_1(packets: dict, batch_id: str) -> dict:
    return {"batch_id": batch_id, "messages": render_messages_v3_1(packets, batch_id), "model_parameters": {"model": "UNSPECIFIED", "temperature": "UNSPECIFIED", "max_output_tokens": "UNSPECIFIED"}, "input_sha256": {"phase1_scoring_packets_v3": canonical_sha256(packets)}, "prompt_version": PROMPT_VERSION_V3_1}


def preview_metrics_v3_1(preview: dict) -> dict:
    system, user = preview["messages"]
    return {"system_prompt_characters": len(system["content"]), "fixed_user_instruction_characters": len(USER_INSTRUCTIONS_V3_1), "user_message_characters": len(user["content"]), "estimated_tokens": heuristic_tokens(system["content"] + user["content"]), "token_estimate_note": "heuristic ceil(ASCII/3)+non-ASCII; not exact"}

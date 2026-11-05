"""Keep observed amounts separate from structural retrieval features."""
import re

from cloc.reference.features import _main_anchor, _normalize_effective_address_map


def extract_amount_observations(graph, chains, ledger, features):
    """Use the same transfer scope and business anchor as feature extraction."""
    nodes = {node.node_id: node for node in graph.nodes}
    edges = {edge.edge_id: edge for edge in chains.edges}
    facts = {fact.fact_id: fact for fact in ledger.transfer_facts}
    records = {record.record_id: record for record in features.records}
    record_ids = {item.strict_chain_id: item.record_id for item in features.chain_statuses if item.record_created}
    observations = {}
    for chain in chains.chains:
        record_id = record_ids.get(chain.strict_chain_id)
        if record_id not in records:
            continue
        _, _, business = _main_anchor(chain.anchor, _normalize_effective_address_map(None), nodes)
        fact_ids = set(chain.result_fact_ids)
        for edge_id in chain.edge_ids:
            edge = edges.get(edge_id)
            if edge is not None and edge.edge_type == "ASSET":
                fact_ids.update((edge.src_fact_id, edge.dst_fact_id))
        rows = []
        supported = {(item.asset_identifier, item.direction) for item in records[record_id].asset_behaviors}
        for fact in sorted((facts[fid] for fid in fact_ids if fid in facts), key=lambda f: (f.trace_index, f.fact_id)):
            if fact.status != "COMPLETE" or fact.uncertainties or not isinstance(fact.amount_raw, str) or re.fullmatch(r"[0-9]+", fact.amount_raw) is None:
                continue
            asset = "NATIVE" if fact.asset_type == "NATIVE" else (fact.asset_address or {}).get("normalized")
            source = (fact.from_address or {}).get("normalized")
            target = (fact.to_address or {}).get("normalized")
            for direction, included in (("OUTFLOW", source in business), ("INFLOW", target in business)):
                if included and (asset, direction) in supported:
                    rows.append({"asset_identifier": asset, "direction": direction,
                                 "amount_raw": fact.amount_raw, "fact_id": fact.fact_id})
        observations[record_id] = rows
    return observations

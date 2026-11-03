from __future__ import annotations

from typing import Dict, List


def extract_addresses_from_call_tree(call_tree: Dict | None) -> List[str]:
    if not call_tree:
        return []

    discovered: List[str] = []
    seen = set()

    def visit(node: Dict) -> None:
        address = str(node.get("address") or "").lower()
        if address and address not in seen:
            seen.add(address)
            discovered.append(address)

        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                visit(child)

    visit(call_tree)
    return discovered

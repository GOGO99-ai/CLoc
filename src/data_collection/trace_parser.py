import re
from web3 import Web3  # legacy import shim for local dry-runs


class TraceNode:
    def __init__(
        self,
        type,
        gas=None,
        address=None,
        function=None,
        params=None,
        call_type=None,
        value=None,
        children=None,
    ):
        self.type = type
        self.gas = gas
        self.address = address
        self.function = function
        self.params = params
        self.call_type = call_type
        self.value = value
        self.children = children or []

    def to_dict(self):
        return {
            "type": self.type,
            "gas": self.gas,
            "address": self.address,
            "function": self.function,
            "params": self.params,
            "call_type": self.call_type,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
        }


class TraceParser:
    def __init__(self):
        self.call_pattern = re.compile(
            r"^\[(\d+)\]\s+(0x[a-fA-F0-9]+)::([^(]+)\((.*)\)(?:\s+\[([^\]]+)\])?$"
        )
        self.w3 = Web3()

    def parse(self, trace_text, contract_assets=None):
        # Keep the legacy parameter for compatibility, but ignore it so the
        # emitted call tree stays ABI-free.
        parts = trace_text.split("State Diff:")
        call_trace_part = parts[0]
        root_node = self._parse_call_tree(call_trace_part)
        return {"call_tree": root_node}

    def _parse_call_tree(self, text):
        lines = text.strip().split("\n")
        root = None
        stack = []
        for line in lines:
            prefix, content = self._split_prefix_and_content(line)
            if content is None:
                continue

            call_match = self.call_pattern.match(content)
            if not call_match:
                continue

            depth = self._estimate_depth(prefix)
            gas, addr, func, params, ctype = call_match.groups()

            node = TraceNode(
                "call",
                int(gas),
                addr,
                func,
                params=params,
                call_type=ctype,
            )

            if depth == 0:
                root = node
                stack = [root]
                continue

            while len(stack) > depth:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
                stack.append(node)
        return root

    def _split_prefix_and_content(self, line):
        stripped = line.strip()
        if not stripped or stripped in {"Traces:", "State Diff:"}:
            return "", None

        bracket_index = line.find("[")
        if bracket_index == -1:
            return "", None

        prefix = line[:bracket_index]
        content = line[bracket_index:].strip()
        return prefix, content

    def _estimate_depth(self, prefix):
        if not prefix or not prefix.strip():
            return 0

        normalized = (
            prefix.replace("鈹溾攢", "├")
            .replace("鈹斺攢", "└")
            .replace("閳规柡鏀?", "└")
            .replace("閳规壕鏀?", "├")
            .replace("│", "|")
            .replace("鈹?", "|")
            .replace("閳?", "|")
        )
        branch_count = normalized.count("├") + normalized.count("└")
        vertical_count = normalized.count("|")
        return max(1, branch_count + vertical_count)

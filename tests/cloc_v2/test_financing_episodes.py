from __future__ import annotations

import unittest

from src.cloc_v2.asset_continuity import build_asset_continuity
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.runtime_semantics import build_runtime_annotations
from src.cloc_v2.transaction_graph import build_transaction_graph


def addr(char: str) -> str:
    return "0x" + char * 40


def financing_graph(initiation_function: str, *, repayment_amount: str = "110", repayment_target: str | None = None, repayment_sender: str | None = None):
    provider, receiver, token = addr("a"), addr("b"), addr("c")
    return build_transaction_graph(
        {
            "type": "call", "address": addr("f"), "function": "root", "children": [
                {"type": "call", "address": provider, "function": initiation_function, "children": [
                    {"type": "call", "address": token, "function": "transfer", "params": f"{receiver}, 100", "children": []},
                    {"type": "call", "address": receiver, "function": "executeOperation", "children": [
                        {"type": "call", "address": token, "function": "transfer", "params": f"{repayment_target or provider}, {repayment_amount}", "children": []}
                    ]}
                ]}
            ]
        }
    )


class FinancingEpisodeTests(unittest.TestCase):
    def test_explicit_flash_loan_callback_and_return(self) -> None:
        graph = financing_graph("flashLoan")
        annotations = build_runtime_annotations(graph)
        ledger = build_observed_asset_ledger(graph, annotations)
        result = build_asset_continuity(graph, annotations, ledger)
        self.assertEqual(len(result.financing_episodes), 1)
        episode = result.financing_episodes[0]
        self.assertEqual((episode.financing_type, episode.status, episode.principal_raw, episode.repayment_raw), ("EXPLICIT_FLASH_LOAN", "COMPLETE", "100", "110"))
        self.assertEqual(episode.evidence_level, "STRONG_ASSOCIATION")

    def test_swap_requires_explicit_callback_and_repayment_constraints(self) -> None:
        graph = financing_graph("swap")
        annotations = build_runtime_annotations(graph)
        result = build_asset_continuity(graph, annotations, build_observed_asset_ledger(graph, annotations))
        self.assertEqual(result.financing_episodes[0].financing_type, "CALLBACK_SWAP_CANDIDATE")
        ordinary = build_transaction_graph({"type": "call", "address": addr("a"), "function": "root", "children": [{"type": "call", "address": addr("b"), "function": "swap", "children": []}]})
        ordinary_annotations = build_runtime_annotations(ordinary)
        self.assertEqual(build_asset_continuity(ordinary, ordinary_annotations, build_observed_asset_ledger(ordinary, ordinary_annotations)).financing_episodes, ())
        wrong_target = financing_graph("flashLoan", repayment_target=addr("d"))
        wrong_annotations = build_runtime_annotations(wrong_target)
        self.assertEqual(build_asset_continuity(wrong_target, wrong_annotations, build_observed_asset_ledger(wrong_target, wrong_annotations)).financing_episodes[0].status, "FUNDING_ONLY")

    def test_other_payer_is_unresolved_and_smaller_return_is_ambiguous(self) -> None:
        other_payer = addr("d")
        graph = financing_graph("flashLoan", repayment_amount="110")
        # Replace the callback's nested transfer parent by making the callback callee another address is not sufficient;
        # use a direct raw tree whose return call is made by a third-party child.
        provider, receiver, token = addr("a"), addr("b"), addr("c")
        raw = {"type": "call", "address": addr("f"), "function": "root", "children": [{"type": "call", "address": provider, "function": "flashLoan", "children": [{"type": "call", "address": token, "function": "transfer", "params": f"{receiver}, 100", "children": []}, {"type": "call", "address": receiver, "function": "executeOperation", "children": [{"type": "call", "address": other_payer, "function": "relay", "children": [{"type": "call", "address": token, "function": "transfer", "params": f"{provider}, 100", "children": []}]}]}]}]}
        third_party = build_transaction_graph(raw)
        annotations = build_runtime_annotations(third_party)
        self.assertEqual(build_asset_continuity(third_party, annotations, build_observed_asset_ledger(third_party, annotations)).financing_episodes[0].status, "REPAYMENT_UNRESOLVED")
        smaller = financing_graph("flashLoan", repayment_amount="99")
        smaller_annotations = build_runtime_annotations(smaller)
        self.assertEqual(build_asset_continuity(smaller, smaller_annotations, build_observed_asset_ledger(smaller, smaller_annotations)).financing_episodes[0].status, "AMBIGUOUS")

    def test_nested_callback_binds_to_nearest_financing_ancestor(self) -> None:
        outer_provider, outer_receiver = addr("a"), addr("b")
        inner_provider, inner_receiver, token = addr("d"), addr("e"), addr("c")
        raw = {
            "type": "call", "address": addr("f"), "function": "root", "children": [
                {"type": "call", "address": outer_provider, "function": "flashLoan", "children": [
                    {"type": "call", "address": token, "function": "transfer", "params": f"{outer_receiver}, 100", "children": []},
                    {"type": "call", "address": outer_receiver, "function": "executeOperation", "children": [
                        {"type": "call", "address": inner_provider, "function": "swap", "children": [
                            {"type": "call", "address": token, "function": "transfer", "params": f"{inner_receiver}, 50", "children": []},
                            {"type": "call", "address": inner_receiver, "function": "uniswapV2Call", "children": []},
                        ]},
                    ]},
                ]},
            ],
        }
        graph = build_transaction_graph(raw)
        annotations = build_runtime_annotations(graph)
        result = build_asset_continuity(graph, annotations, build_observed_asset_ledger(graph, annotations))
        self.assertEqual(2, len(result.financing_episodes))
        self.assertEqual(
            {episode.financing_type for episode in result.financing_episodes},
            {"EXPLICIT_FLASH_LOAN", "CALLBACK_SWAP_CANDIDATE"},
        )
        swap_episode = next(episode for episode in result.financing_episodes if episode.financing_type == "CALLBACK_SWAP_CANDIDATE")
        self.assertEqual(inner_provider.lower(), swap_episode.provider_address["normalized"])
        self.assertEqual(inner_receiver.lower(), swap_episode.receiver_address["normalized"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import unittest
import itertools
import json
import tempfile
from pathlib import Path
from dataclasses import replace
from decimal import Decimal

from src.cloc_v2.asset_continuity import build_asset_continuity
from src.cloc_v2.asset_continuity import observed_asset_ledger_sha256
from src.cloc_v2.observed_asset_ledger import build_observed_asset_ledger
from src.cloc_v2.result_detection import detect_result_candidates, serialize_result_detection
from src.cloc_v2.result_models import ResultDetectionConfig
from src.cloc_v2.runtime_semantics import build_runtime_annotations
from src.cloc_v2.transaction_graph import build_transaction_graph
from src.cloc_v2.outflow_scoring import grouped_outflow_score


def addr(char: str) -> str:
    return "0x" + char * 40


def transfer_branch(source: str, token: str, target: str, amount: str) -> dict:
    return {
        "type": "call",
        "address": source,
        "function": "relay",
        "children": [{"type": "call", "address": token, "function": "transfer", "params": f"{target}, {amount}", "children": []}],
    }


def context(branches: list[dict]):
    graph = build_transaction_graph({"type": "call", "address": addr("f"), "function": "root", "children": branches})
    annotations = build_runtime_annotations(graph)
    ledger = build_observed_asset_ledger(graph, annotations)
    continuity = build_asset_continuity(graph, annotations, ledger).to_dict()
    return graph, annotations, ledger, continuity


def signals(candidate) -> dict[str, Decimal]:
    return {item.signal_type: Decimal(item.value) for item in candidate.signals}


class ResultDetectionTests(unittest.TestCase):
    def test_grouped_formula_fixed_hand_calculations_and_legacy_difference(self):
        values = tuple(map(Decimal, ("0.8", "0.4", "0.6", "1", "0")))
        for lam, expected in (("0", "0.5333"), ("0.65", "0.5767"), ("1", "0.6000")):
            with self.subTest(lam=lam):
                score, audit = grouped_outflow_score(values, lam)
                self.assertEqual(score.quantize(Decimal('.0001')), Decimal(expected))
                self.assertEqual(Decimal(audit['asset_group_mean']), Decimal('.6'))
        old = sum(x*w for x,w in zip(values,map(Decimal,('.35','.30','.15','.10','.10'))))
        self.assertEqual(old, Decimal('.5900'))

    def test_group_symmetry_extremes_and_ranking_reversal(self):
        for lam in ('0','0.65','1'):
            self.assertEqual(grouped_outflow_score((Decimal(0),)*5,lam)[0],0)
            self.assertEqual(grouped_outflow_score((Decimal(1),)*5,lam)[0],1)
            scores = [grouped_outflow_score(tuple(map(Decimal, pair + ctx)),lam)[0]
                      for pair in (('.8','.4'),('.4','.8')) for ctx in itertools.permutations(('.6','1','0'))]
            self.assertEqual(len(set(scores)),1)
            self.assertTrue(all(0<=v<=1 for v in scores))
        a=tuple(map(Decimal,('.9','.9','.2','.2','.2')))
        b=tuple(map(Decimal,('.4','.4','.8','.8','.8')))
        self.assertLess(grouped_outflow_score(a,'0')[0],grouped_outflow_score(b,'0')[0])
        self.assertGreater(grouped_outflow_score(a,'1')[0],grouped_outflow_score(b,'1')[0])

    def test_invalid_lambda_and_numeric_zero(self):
        args=context([])
        for value in ('','abc','NaN','sNaN','Infinity','-Infinity','-0.01','1.01',None,False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                detect_result_candidates(*args,ResultDetectionConfig(outflow_lambda=value))
        self.assertEqual(detect_result_candidates(*args,ResultDetectionConfig(outflow_lambda=0)).to_dict()['detection_config']['outflow_lambda'],'0')

    def test_fixed_precision_and_profit_invariance(self):
        from decimal import localcontext
        args=context([transfer_branch(addr('a'),addr('c'),addr('b'),'101'),transfer_branch(addr('b'),addr('c'),addr('a'),'110'),transfer_branch(addr('b'),addr('c'),addr('f'),'20')])
        g,a,l,c=args; first,second,_=l.transfer_facts
        c['financing_episodes']=[{'status':'COMPLETE','asset_address':first.asset_address,'funding_fact_ids':[first.fact_id],'repayment_fact_ids':[second.fact_id]}]
        profits=[]; outflows=[]
        for lam in ('0','0.65','1'):
            result=detect_result_candidates(*args,ResultDetectionConfig(outflow_lambda=lam)).to_dict()
            profits.append([{k:v for k,v in r.items() if k!='result_id'} for r in result['result_candidates'] if r['result_type']=='R_PROFIT'])
        self.assertTrue(profits[0]); self.assertEqual(profits[0],profits[1]); self.assertEqual(profits[1],profits[2])
        for precision in (12,28,60):
            with localcontext() as ctx:
                ctx.prec=precision
                outflows.append([r for r in detect_result_candidates(*args).to_dict()['result_candidates'] if r['result_type']=='R_OUTFLOW'])
        self.assertEqual(outflows[0],outflows[1]);self.assertEqual(outflows[1],outflows[2])

    def test_cli_zero_reaches_saved_configuration(self):
        from scripts.detect_result_candidates import main
        args=context([transfer_branch(addr('a'),addr('c'),addr('b'),'17')])
        with tempfile.TemporaryDirectory() as temp:
            files=[]
            for name,obj in zip(('graph','annotations','ledger','continuity'),args):
                p=Path(temp)/f'{name}.json';p.write_text(json.dumps(obj.to_dict() if hasattr(obj,'to_dict') else obj),encoding='utf-8');files.append(str(p))
            out=Path(temp)/'out.json'
            self.assertEqual(main([arg for name,path in zip(('graph','annotations','ledger','continuity'),files) for arg in (f'--{name}',path)]+['--output',str(out),'--outflow-lambda','0']),0)
            self.assertEqual(json.loads(out.read_text())['detection_config']['outflow_lambda'],'0')

    def test_threshold_equality_and_tie_order_no_fallback(self):
        args=context([transfer_branch(addr('a'),addr('c'),addr('b'),'10') for _ in range(7)])
        cfg=ResultDetectionConfig(outflow_lambda='1',outflow_threshold='1.0000')
        result=detect_result_candidates(*args,cfg)
        self.assertEqual([r.source_fact_ids[0] for r in result.result_candidates],[f.fact_id for f in args[2].transfer_facts[:5]])
        self.assertTrue(all(r.selection_reason=='SCORE_THRESHOLD' for r in result.result_candidates))
        strict=replace(cfg,outflow_lambda='0',ensure_top1_per_asset=False)
        self.assertEqual(detect_result_candidates(*args,strict).result_candidates,())

    def test_terminal_outflow_signals_and_fixed_decimal_confidence(self) -> None:
        graph, annotations, ledger, continuity = context([transfer_branch(addr("a"), addr("c"), addr("f"), "100")])
        result = detect_result_candidates(graph, annotations, ledger, continuity, ResultDetectionConfig(outflow_threshold="0.0000"))
        candidate = result.result_candidates[0]
        self.assertEqual((candidate.result_type, candidate.evidence_level, candidate.amount_raw), ("R_OUTFLOW", "STRONG_ASSOCIATION", "100"))
        self.assertEqual(signals(candidate), {"MAGNITUDE_RANK": Decimal("1.0000"), "TERMINAL_RETENTION": Decimal("1.0000"), "LATE_EXECUTION": Decimal("1.0000"), "ROOT_RECEIVER": Decimal("1.0000"), "SOURCE_ACTIVITY": Decimal("1.0000")})
        self.assertEqual(candidate.confidence, "1.0000")

    def test_complete_candidate_does_not_inherit_method_boundary_notes(self) -> None:
        graph, annotations, ledger, continuity = context(
            [transfer_branch(addr("a"), addr("c"), addr("b"), "100")]
        )
        candidate = detect_result_candidates(graph, annotations, ledger, continuity).result_candidates[0]
        self.assertNotIn("observed asset ledger is partial", candidate.uncertainties)
        self.assertNotIn("token decimals and real balance changes are unknown", candidate.uncertainties)

    def test_fact_specific_missing_or_ambiguity_still_propagates_to_candidate(self) -> None:
        graph, annotations, ledger, continuity = context(
            [transfer_branch(addr("a"), addr("c"), addr("b"), "100")]
        )
        variants = (
            replace(ledger.transfer_facts[0], uncertainties=("FROM_ADDRESS_AMBIGUOUS",)),
            replace(
                ledger.transfer_facts[0],
                asset_address=None,
                uncertainties=("ASSET_ADDRESS_MISSING",),
            ),
        )
        for fact_variant in variants:
            with self.subTest(reason=fact_variant.uncertainties[0]):
                variant_ledger = replace(ledger, transfer_facts=(fact_variant,))
                variant_continuity = copy.deepcopy(continuity)
                variant_continuity["observed_asset_ledger_sha256"] = observed_asset_ledger_sha256(variant_ledger)
                candidate = detect_result_candidates(
                    graph,
                    annotations,
                    variant_ledger,
                    variant_continuity,
                ).result_candidates[0]
                self.assertIn(fact_variant.uncertainties[0], candidate.uncertainties)

    def test_retention_zero_and_partial_are_computed_from_remaining_inflows(self) -> None:
        token, source, middle, end = addr("c"), addr("a"), addr("b"), addr("d")
        graph, annotations, ledger, continuity = context([transfer_branch(source, token, middle, "100"), transfer_branch(middle, token, end, "60")])
        result = detect_result_candidates(graph, annotations, ledger, continuity)
        candidates = {candidate.source_fact_ids[0]: candidate for candidate in result.result_candidates if candidate.result_type == "R_OUTFLOW"}
        first, second = ledger.transfer_facts
        self.assertEqual(signals(candidates[first.fact_id])["TERMINAL_RETENTION"], Decimal("0.4000"))
        self.assertEqual(signals(candidates[second.fact_id])["TERMINAL_RETENTION"], Decimal("1.0000"))
        consumed_graph, consumed_annotations, consumed_ledger, consumed_continuity = context([transfer_branch(source, token, middle, "100"), transfer_branch(middle, token, end, "100")])
        consumed = detect_result_candidates(consumed_graph, consumed_annotations, consumed_ledger, consumed_continuity)
        consumed_first = consumed_ledger.transfer_facts[0]
        self.assertEqual(signals(next(item for item in consumed.result_candidates if item.source_fact_ids == (consumed_first.fact_id,)))["TERMINAL_RETENTION"], Decimal("0.0000"))

    def test_magnitude_ranks_only_within_same_asset_and_equal_amounts_match(self) -> None:
        source, target = addr("a"), addr("b")
        graph, annotations, ledger, continuity = context([
            transfer_branch(source, addr("c"), target, "10"),
            transfer_branch(source, addr("c"), target, "20"),
            transfer_branch(source, addr("d"), target, "999999"),
            transfer_branch(source, addr("e"), target, "7"),
            transfer_branch(source, addr("e"), target, "7"),
        ])
        result = detect_result_candidates(graph, annotations, ledger, continuity, ResultDetectionConfig(outflow_threshold="0.0000"))
        by_amount = {(candidate.asset_address["normalized"], candidate.amount_raw): signals(candidate)["MAGNITUDE_RANK"] for candidate in result.result_candidates if candidate.result_type == "R_OUTFLOW"}
        self.assertEqual(by_amount[(addr("c"), "10")], Decimal("0.0000"))
        self.assertEqual(by_amount[(addr("c"), "20")], Decimal("1.0000"))
        self.assertEqual(by_amount[(addr("d"), "999999")], Decimal("1.0000"))
        equal_scores = [signals(item)["MAGNITUDE_RANK"] for item in result.result_candidates if item.asset_address["normalized"] == addr("e")]
        self.assertEqual(equal_scores, [Decimal("1.0000"), Decimal("1.0000")])

    def test_threshold_top1_cap_and_duplicate_node_type_rules(self) -> None:
        token, source, target = addr("c"), addr("a"), addr("b")
        graph, annotations, ledger, continuity = context([transfer_branch(source, token, target, str(index + 1)) for index in range(6)])
        cap_result = detect_result_candidates(graph, annotations, ledger, continuity, ResultDetectionConfig(outflow_threshold="0.0000"))
        self.assertEqual(5, len([item for item in cap_result.result_candidates if item.result_type == "R_OUTFLOW"]))
        fallback_result = detect_result_candidates(graph, annotations, ledger, continuity, ResultDetectionConfig(outflow_threshold="1.0000"))
        self.assertEqual(1, len(fallback_result.result_candidates))
        self.assertEqual("TOP1_FALLBACK", fallback_result.result_candidates[0].selection_reason)
        self.assertEqual(len({(item.node_id, item.result_type) for item in cap_result.result_candidates}), len(cap_result.result_candidates))

    def test_invalid_funding_complete_repayment_incomplete_self_and_zero_are_excluded(self) -> None:
        token, provider, receiver, other = addr("c"), addr("a"), addr("b"), addr("d")
        graph, annotations, ledger, continuity = context([
            transfer_branch(provider, token, receiver, "100"),
            transfer_branch(receiver, token, provider, "110"),
            transfer_branch(receiver, token, other, "10"),
            transfer_branch(other, token, other, "5"),
            transfer_branch(other, token, receiver, "0"),
        ])
        funding, repayment, allowed, self_transfer, zero = ledger.transfer_facts
        continuity["financing_episodes"] = [{"status": "COMPLETE", "asset_address": funding.asset_address, "funding_fact_ids": [funding.fact_id], "repayment_fact_ids": [repayment.fact_id]}]
        continuity["self_transfers"] = [{"fact_id": self_transfer.fact_id}]
        incomplete = replace(ledger.transfer_facts[-1], status="INCOMPLETE", amount_raw=None)
        modified_ledger = replace(ledger, transfer_facts=ledger.transfer_facts[:-1] + (incomplete,))
        continuity["observed_asset_ledger_sha256"] = __import__("src.cloc_v2.asset_continuity", fromlist=["observed_asset_ledger_sha256"]).observed_asset_ledger_sha256(modified_ledger)
        result = detect_result_candidates(graph, annotations, modified_ledger, continuity)
        ids = {item.source_fact_ids[0] for item in result.result_candidates}
        self.assertEqual(ids, {allowed.fact_id})

    def test_profit_requires_complete_same_asset_after_repayment_and_terminal_retention(self) -> None:
        token, provider, receiver, root = addr("c"), addr("a"), addr("b"), addr("f")
        graph, annotations, ledger, continuity = context([
            transfer_branch(provider, token, receiver, "100"),
            transfer_branch(receiver, token, provider, "110"),
            transfer_branch(receiver, token, root, "20"),
        ])
        funding, repayment, post_repayment = ledger.transfer_facts
        complete_episode = {"status": "COMPLETE", "asset_address": token.lower() and funding.asset_address, "funding_fact_ids": [funding.fact_id], "repayment_fact_ids": [repayment.fact_id]}
        continuity["financing_episodes"] = [complete_episode]
        result = detect_result_candidates(graph, annotations, ledger, continuity)
        profits = [item for item in result.result_candidates if item.result_type == "R_PROFIT"]
        self.assertEqual(1, len(profits))
        self.assertEqual((post_repayment.fact_id,), profits[0].source_fact_ids)
        self.assertEqual(profits[0].confidence, "1.0000")
        for status in ("FUNDING_ONLY", "AMBIGUOUS"):
            variant = copy.deepcopy(continuity)
            variant["financing_episodes"][0]["status"] = status
            self.assertEqual([], [item for item in detect_result_candidates(graph, annotations, ledger, variant).result_candidates if item.result_type == "R_PROFIT"])
        wrong_asset = copy.deepcopy(continuity)
        wrong_asset["financing_episodes"][0]["asset_address"] = {"original": addr("d"), "normalized": addr("d")}
        self.assertEqual([], [item for item in detect_result_candidates(graph, annotations, ledger, wrong_asset).result_candidates if item.result_type == "R_PROFIT"])
        multiple = copy.deepcopy(continuity)
        multiple["financing_episodes"].append({"status": "FUNDING_ONLY", "asset_address": funding.asset_address, "funding_fact_ids": [funding.fact_id], "repayment_fact_ids": []})
        self.assertEqual([], [item for item in detect_result_candidates(graph, annotations, ledger, multiple).result_candidates if item.result_type == "R_PROFIT"])
        early_graph, early_annotations, early_ledger, early_continuity = context([
            transfer_branch(provider, token, receiver, "100"),
            transfer_branch(receiver, token, root, "20"),
            transfer_branch(receiver, token, provider, "110"),
        ])
        early_funding, early_candidate, early_repayment = early_ledger.transfer_facts
        early_continuity["financing_episodes"] = [{"status": "COMPLETE", "asset_address": early_funding.asset_address, "funding_fact_ids": [early_funding.fact_id], "repayment_fact_ids": [early_repayment.fact_id]}]
        early_result = detect_result_candidates(early_graph, early_annotations, early_ledger, early_continuity)
        self.assertNotIn(early_candidate.fact_id, {item.source_fact_ids[0] for item in early_result.result_candidates if item.result_type == "R_PROFIT"})

    def test_coverage_hash_failure_input_immutability_and_forbidden_outputs(self) -> None:
        graph, annotations, ledger, continuity = context([transfer_branch(addr("a"), addr("c"), addr("b"), "10")])
        original = copy.deepcopy(continuity)
        first = detect_result_candidates(graph, annotations, ledger, continuity)
        second = detect_result_candidates(graph, annotations, ledger, continuity)
        self.assertEqual(serialize_result_detection(first), serialize_result_detection(second))
        self.assertEqual(continuity, original)
        self.assertEqual({"R_OUTFLOW", "R_PROFIT", "R_STATE", "R_FAILURE"}, set(first.result_type_coverage))
        self.assertEqual([], [item for item in first.result_candidates if item.result_type in {"R_STATE", "R_FAILURE"}])
        rendered = serialize_result_detection(first).upper()
        self.assertFalse(any(term in rendered for term in ("ATTACKER", "VICTIM", "VULNERABLE", "ROOT_CAUSE", "PARAM", "STATE_EDGE")))
        invalid = copy.deepcopy(continuity)
        invalid["transaction_graph_sha256"] = "mismatch"
        with self.assertRaisesRegex(ValueError, "Asset continuity"):
            detect_result_candidates(graph, annotations, ledger, invalid)

    def test_function_names_do_not_create_results_without_transfer_facts(self) -> None:
        graph = build_transaction_graph({"type": "call", "address": addr("f"), "function": "root", "children": [{"type": "call", "address": addr("a"), "function": "withdraw", "children": []}, {"type": "call", "address": addr("b"), "function": "borrow", "children": []}, {"type": "call", "address": addr("c"), "function": "mint", "children": []}]})
        annotations = build_runtime_annotations(graph)
        ledger = build_observed_asset_ledger(graph, annotations)
        continuity = build_asset_continuity(graph, annotations, ledger).to_dict()
        self.assertEqual((), detect_result_candidates(graph, annotations, ledger, continuity).result_candidates)


if __name__ == "__main__":
    unittest.main()

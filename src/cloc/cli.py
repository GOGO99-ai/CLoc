"""CLoc stage commands, normal-library construction, and full analysis."""
import argparse
import json
from pathlib import Path

from cloc.pipeline import build_evidence, prepare_review, prepare_localization, analyze_transaction
from cloc.review.prompt import validate_review_response
from cloc.review.routing import decide_review_route
from cloc.localization.prompt import validate_ranking_response, materialize_ranking_result
from cloc.reference.storage import build_reference_library
from cloc.reporting.report import render_report
from cloc.runtime.llm import JSONChatClient
from cloc.trace.acquisition import acquire_trace


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parser():
    parser = argparse.ArgumentParser(description="CLoc: execution evidence, reference review, and source-informed ranking")
    commands = parser.add_subparsers(dest="command", required=True)
    library = commands.add_parser("build-reference", help="build a normal library from a curated trace manifest")
    library.add_argument("--manifest", type=Path, required=True)
    library.add_argument("--output", type=Path, required=True)
    for command in ("evidence", "review", "localize", "analyze"):
        stage = commands.add_parser(command)
        if command == "analyze":
            inputs = stage.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--trace", type=Path, help="cached JSON callTracer frame")
            inputs.add_argument("--tx-hash", help="transaction hash to acquire through RPC")
            stage.add_argument("--trace-cache", type=Path, default=Path("data/traces"))
            stage.add_argument("--model", help="override CLOC_LLM_MODEL")
            stage.add_argument("--token-parameter", choices=("max_tokens", "max_completion_tokens"))
            stage.add_argument("--review-output-tokens", type=int, default=4096)
            stage.add_argument("--ranking-output-tokens", type=int, default=8192)
            stage.add_argument("--cached-sources-only", action="store_true")
        else:
            stage.add_argument("--trace", type=Path, required=True)
        stage.add_argument("--chain", choices=("eth", "bsc"), required=True)
        stage.add_argument("--output", type=Path, required=True, help="output directory for analyze; JSON file for stage commands")
        if command in {"review", "analyze"}:
            stage.add_argument("--baseline", type=Path, required=True)
            stage.add_argument("--record-id", default="query")
        if command in {"review", "localize"}:
            stage.add_argument("--response", type=Path, help="saved model content JSON using the current schema")
        if command in {"localize", "analyze"}:
            stage.add_argument("--source-cache", type=Path, default=Path("data/source_cache"))
            stage.add_argument("--max-source-addresses", type=int, default=3)
            stage.add_argument("--top-k", type=int, choices=range(1, 6), default=5)
        if command == "localize":
            stage.add_argument("--fetch-sources", action="store_true")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-reference":
            summary = build_reference_library(args.manifest, args.output)
            print(json.dumps(summary, ensure_ascii=False))
            return 0
        raw = _read(args.trace) if args.trace else acquire_trace(args.tx_hash, args.chain, args.trace_cache)
        if args.command == "analyze":
            if args.review_output_tokens < 1 or args.ranking_output_tokens < 1 or args.max_source_addresses < 1:
                raise ValueError("TOKEN_AND_SOURCE_LIMITS_MUST_BE_POSITIVE")
            client = JSONChatClient.from_environment(model=args.model, token_parameter=args.token_parameter)
            report, artifacts = analyze_transaction(raw, args.chain, args.baseline, args.source_cache, client,
                record_id=args.record_id, fetch_sources=not args.cached_sources_only,
                max_source_addresses=args.max_source_addresses, top_k=args.top_k,
                review_tokens=args.review_output_tokens, ranking_tokens=args.ranking_output_tokens)
            if args.tx_hash:
                report["transaction_hash"] = args.tx_hash.lower()
            for name, value in artifacts.items():
                _write(args.output / (name + ".json"), value)
            _write(args.output / "report.json", report)
            (args.output / "report.md").write_text(render_report(report), encoding="utf-8")
            print(str((args.output / "report.json").resolve()))
            return 0 if report["status"] in {"NORMAL", "RANKED_CANDIDATES"} else 2
        bundle = build_evidence(raw, args.chain)
        code = 0
        if args.command == "evidence":
            result = bundle.to_dict()
        elif args.command == "review":
            preparation = prepare_review(bundle, args.baseline, args.record_id)
            result = preparation.to_dict()
            if preparation.prompt is None:
                result["route"] = "PROCEED_RANKING"
            else:
                response = _read(args.response) if args.response else None
                routing = decide_review_route(preparation.gate_evidence, preparation.prompt, response)
                result["routing"] = routing.to_dict()
                if response is not None and "LLM_RESPONSE_INVALID" not in routing.reason_codes:
                    result["validated_response"] = validate_review_response(response, preparation.prompt)
                elif response is not None:
                    code = 2
        else:
            preparation, acquisition = prepare_localization(bundle, args.source_cache,
                fetch_sources=args.fetch_sources, max_source_addresses=args.max_source_addresses, top_k=args.top_k)
            result = preparation.to_dict()
            result["acquisition"] = acquisition
            if preparation.prompt is None:
                code = 2
            elif args.response:
                validation = validate_ranking_response(_read(args.response), preparation.prompt)
                result["validation"] = validation.to_dict()
                if validation.valid:
                    result["ranking"] = materialize_ranking_result(validation, preparation.prompt)
                    result["source_locations"] = preparation.source_locations
                else:
                    code = 2
        _write(args.output, result)
        print(str(args.output.resolve()))
        return code
    except (OSError, ValueError, TypeError, KeyError) as error:
        # Paths and remote response bodies are not needed in a public error message.
        message = str(error) if isinstance(error, ValueError) and str(error).isupper() else type(error).__name__
        parser.exit(2, f"cloc: {message}\n")

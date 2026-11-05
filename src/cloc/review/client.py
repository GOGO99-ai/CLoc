"""Execute reference review and retain validated comparisons."""
from cloc.review.prompt import validate_review_response
from cloc.review.routing import decide_review_route
from cloc.runtime.llm import LLMCallError


def execute_review(preparation, client, *, output_tokens=4096):
    if preparation.prompt is None or preparation.gate_evidence is None:
        return {"status": "INELIGIBLE", "route": "PROCEED_RANKING",
                "reason_codes": list(preparation.eligibility.reason_codes)}
    audit = None
    try:
        response, audit = client.complete(preparation.prompt, output_tokens=output_tokens)
        validated = validate_review_response(response, preparation.prompt)
    except (LLMCallError, ValueError, TypeError) as error:
        return {"status": "REVIEW_UNAVAILABLE", "route": "PROCEED_RANKING",
                "reason_codes": [str(error) if isinstance(error, LLMCallError) else "REVIEW_RESPONSE_INVALID"],
                "audit": audit}
    routing = decide_review_route(preparation.gate_evidence, preparation.prompt, validated)
    return {"status": "VALIDATED", "route": routing.route, "response": validated,
            "routing": routing.to_dict(), "audit": audit}

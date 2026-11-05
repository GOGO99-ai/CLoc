"""Execute source-informed ranking with the published response contract."""
from cloc.localization.prompt import validate_ranking_response, materialize_ranking_result
from cloc.runtime.llm import LLMCallError


def execute_ranking(preparation, client, *, output_tokens=8192):
    if preparation.prompt is None:
        return {"status": preparation.preparation_status,
                "issues": [issue.to_dict() for issue in preparation.issues]}
    try:
        response, audit = client.complete(preparation.prompt, output_tokens=output_tokens)
    except LLMCallError as error:
        return {"status": "RANKING_UNAVAILABLE", "reason_codes": [str(error)]}
    validation = validate_ranking_response(response, preparation.prompt)
    if not validation.valid:
        return {"status": "INVALID_RESPONSE", "validation": validation.to_dict(), "audit": audit}
    return {"status": "RANKED_CANDIDATES", "ranking": materialize_ranking_result(validation, preparation.prompt),
            "source_locations": preparation.source_locations, "audit": audit}

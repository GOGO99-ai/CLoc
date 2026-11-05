"""Present validated rankings without treating function bounds as suspicious lines."""
def build_report(chain, review, ranking=None):
    if review.get("route") == "EARLY_NORMAL":
        return {"schema_version": "cloc_report_v2", "chain": chain, "status": "NORMAL",
                "review": review, "findings": [], "limitations": []}
    ranking = ranking or {"status": "RANKING_UNAVAILABLE"}
    result = ranking.get("ranking", {})
    locations = ranking.get("source_locations", {})
    findings = [{**item, "source_location": locations.get(item["source_slice_id"], {})}
                for item in result.get("findings", [])]
    return {"schema_version": "cloc_report_v2", "chain": chain,
            "status": ranking["status"], "review": review, "findings": findings,
            "limitations": result.get("limitations", []),
            "ranking_audit": {key: value for key, value in ranking.items() if key not in {"ranking", "source_locations"}}}


def render_report(report):
    lines = ["# CLoc analysis", "", "Status: " + report["status"], "Chain: " + report["chain"], ""]
    for unit in report.get("review", {}).get("response", {}).get("reviews", []):
        lines.extend([f"## Review {unit['unit_id']}: {unit['verdict']}", "", unit["explanation"], ""])
        for comparison in unit["reference_assessment"]:
            lines.extend([f"- {comparison['reference_id']}: {comparison['comparison']}",
                          "  Evidence: " + ", ".join(comparison["citations"])])
        lines.append("")
    for item in report["findings"]:
        lines.extend([f"## {item['rank']}. {item['function']} ({item['source_slice_id']})", "", item["explanation"], "",
                      "Evidence: " + ", ".join(item["evidence_ids"]),
                      "Source file: " + str(item["source_location"].get("relative_file", item["file_alias"]))])
        bounds = item["source_range"]
        lines.append(f"Supplied source range: {bounds['start']}–{bounds['end']}")
        selected = item["predicted_line_range"]
        lines.append(f"Selected suspicious lines: {selected['start']}–{selected['end']}" if selected else "Selected suspicious lines: unavailable")
        if item["source_uncertainties"]:
            lines.append("Source limitations: " + ", ".join(item["source_uncertainties"]))
        lines.append("")
    if report["limitations"]:
        lines.extend(["## Limitations", "", *["- " + text for text in report["limitations"]], ""])
    return "\n".join(lines) + "\n"

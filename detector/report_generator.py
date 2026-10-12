def generate_report(results):
    total = len(results)
    supported = 0
    contradicted = 0
    unknown = 0
    escalated = 0
    singleton_sets = 0

    for item in results:
        ver = item.get("verification", {})
        label = ver.get("label")

        if label == "supported":
            supported += 1
        elif label == "contradicted":
            contradicted += 1
        else:
            unknown += 1

        if ver.get("is_escalation_candidate", False):
            escalated += 1
        elif ver.get("conformal_set") and len(ver["conformal_set"]) == 1:
            singleton_sets += 1

    hallucination_rate = 0
    if total > 0:
        hallucination_rate = contradicted / total

    return {
        "summary": {
            "total_claims": total,
            "supported": supported,
            "contradicted": contradicted,
            "unknown": unknown,
            "escalated_claims": escalated,
            "singleton_conformal_claims": singleton_sets,
            "conformal_coverage_guarantee": "90%",
            "conformal_alpha": 0.10,
            "hallucination_rate": hallucination_rate
        },
        "claims": results
    }
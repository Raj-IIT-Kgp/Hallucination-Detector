def generate_report(results):

    total = len(results)

    supported = 0
    contradicted = 0
    unknown = 0


    for item in results:

        label = item["verification"]["label"]


        if label == "supported":
            supported += 1

        elif label == "contradicted":
            contradicted += 1

        else:
            unknown += 1



    hallucination_rate = 0


    if total > 0:
        hallucination_rate = contradicted / total



    return {

        "summary": {

            "total_claims": total,

            "supported": supported,

            "contradicted": contradicted,

            "unknown": unknown,

            "hallucination_rate": hallucination_rate

        },


        "claims": results

    }
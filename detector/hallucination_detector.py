from agents.claim_extractor import extract_claims
from verification.nli_verifier import verify_claim
from detector.report_generator import generate_report
from agents.correction_agent import correct_claim


class HallucinationDetector:


    def __init__(self, retriever):

        self.retriever = retriever



    def analyze(self, answer):

        claims = extract_claims(answer)

        results = []


        for claim in claims:

            evidence = self.retriever.search(
                claim,
                k=1
            )


            if len(evidence) == 0:
                continue


            verification = verify_claim(
                claim,
                evidence[0]
            )

            correction = None
            if verification["label"] == "contradicted":
                correction = correct_claim(
                    claim,
                    evidence[0]
                )

            results.append({
                "claim": claim,
                "evidence": evidence[0],
                "verification": verification,
                "correction": correction
            })

        return generate_report(results)


    def calculate_score(self, results):

        total = len(results)

        if total == 0:
            return 0


        hallucinated = 0


        for item in results:

            if item["verification"]["label"] == "contradiction":
                hallucinated += 1


        return hallucinated / total
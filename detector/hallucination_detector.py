from agents.claim_extractor import extract_claims
from agents.query_generator import generate_queries
from verification.nli_verifier import verify_claim
from detector.report_generator import generate_report
from agents.critic_agent import nli_override

class HallucinationDetector:

    def __init__(self, retriever):
        self.retriever = retriever

    def analyze(self, answer):
        claims = extract_claims(answer)
        results = []

        for claim in claims:
            query_info = generate_queries(claim, context=answer)
            search_queries = query_info["queries"]
            domain = query_info["domain"]
            print(f"Generated search queries: {search_queries} for claim: '{claim}' (Domain: {domain})")
            
            evidence = self.retriever.search(
                search_queries,
                claim=claim,
                domain=domain,
                k=3
            )

            if len(evidence) == 0:
                results.append({
                    "claim": claim,
                    "evidence": "No evidence found.",
                    "source_url": None,
                    "verification": {"label": "unknown", "confidence": 1.0},
                    "correction": None
                })
                continue

            evidence_text = "\n\n".join([f"Evidence {i+1}: {e['text']}" for i, e in enumerate(evidence)])
            source_url = ", ".join(list(set([e["url"] for e in evidence])))

            verification = verify_claim(
                claim,
                evidence_text,
                domain=domain
            )
            
            # CRITIC LLM OVERRIDE: If NLI returns UNKNOWN, use LLM to perform multi-hop logic
            if verification["label"] == "unknown":
                print(f"    NLI returned UNKNOWN. Triggering Critic LLM Override for multi-hop reasoning...")
                llm_label = nli_override(claim, evidence_text, domain=domain)
                print(f"    Critic LLM decided: {llm_label.upper()}")
                verification["label"] = llm_label

            results.append({
                "claim": claim,
                "evidence": evidence_text,
                "source_url": source_url,
                "verification": verification,
                "correction": None
            })

        report = generate_report(results)
        
        return {
            "report": report,
            "final_text": answer  # We no longer rebuild the text
        }

    def analyze_stream(self, answer):
        import json
        yield json.dumps({"status": "progress", "message": "Extracting claims..."}) + "\n"
        claims = extract_claims(answer)
        results = []

        for i, claim in enumerate(claims):
            yield json.dumps({"status": "progress", "message": f"Processing claim {i+1} of {len(claims)}..."}) + "\n"
            
            query_info = generate_queries(claim, context=answer)
            search_queries = query_info["queries"]
            domain = query_info["domain"]
            yield json.dumps({"status": "progress", "message": f"Searching ({domain} domain) for {search_queries}..."}) + "\n"
            
            evidence = self.retriever.search(
                search_queries,
                claim=claim,
                domain=domain,
                k=3
            )

            if len(evidence) == 0:
                results.append({
                    "claim": claim,
                    "evidence": "No evidence found.",
                    "source_url": None,
                    "verification": {"label": "unknown", "confidence": 1.0},
                    "correction": None
                })
                continue

            evidence_text = "\n\n".join([f"Evidence {i+1}: {e['text']}" for i, e in enumerate(evidence)])
            source_url = ", ".join(list(set([e["url"] for e in evidence])))

            yield json.dumps({"status": "progress", "message": f"Verifying claim against evidence..."}) + "\n"
            verification = verify_claim(
                claim,
                evidence_text,
                domain=domain
            )

            # CRITIC LLM OVERRIDE
            if verification["label"] == "unknown":
                yield json.dumps({"status": "progress", "message": "NLI returned UNKNOWN. Triggering Critic LLM for multi-hop reasoning..."}) + "\n"
                llm_label = nli_override(claim, evidence_text, domain=domain)
                verification["label"] = llm_label

            results.append({
                "claim": claim,
                "evidence": evidence_text,
                "source_url": source_url,
                "verification": verification,
                "correction": None
            })

        yield json.dumps({"status": "progress", "message": "Generating final report..."}) + "\n"
        report = generate_report(results)
        
        yield json.dumps({
            "status": "done",
            "report": report,
            "final_text": answer  # We no longer rebuild the text
        }) + "\n"


    def calculate_score(self, results):

        total = len(results)

        if total == 0:
            return 0


        hallucinated = 0


        for item in results:

            if item["verification"]["label"] == "contradiction":
                hallucinated += 1


        return hallucinated / total
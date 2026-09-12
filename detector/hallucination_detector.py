from agents.claim_extractor import extract_claims
from agents.query_generator import generate_queries
from verification.nli_verifier import verify_claim
from detector.report_generator import generate_report
from agents.correction_agent import correct_claim
from agents.rebuilder_agent import rebuild_response
from agents.critic_agent import review_correction, nli_override, evaluate_absence

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

            evidence_text = evidence[0]["text"]
            source_url = evidence[0]["url"]

            verification = verify_claim(
                claim,
                evidence_text,
                domain=domain
            )
            
            # CRITIC LLM OVERRIDE: If NLI returns UNKNOWN, use LLM to double check
            if verification["label"] == "unknown":
                print(f"    NLI returned UNKNOWN. Triggering Critic LLM Override...")
                llm_label = nli_override(claim, evidence_text, domain=domain)
                print(f"    Critic LLM decided: {llm_label.upper()}")
                verification["label"] = llm_label

            if verification["label"] == "unknown":
                print(f"    Critic LLM also returned UNKNOWN. Triggering Absence of Evidence Evaluation...")
                absence_label = evaluate_absence(claim, ", ".join(search_queries))
                print(f"    Absence Evaluator decided: {absence_label.upper()}")
                verification["label"] = absence_label
                if absence_label == "contradicted":
                    evidence_text = f"The comprehensive Wikipedia pages for {search_queries} contain no record of this event."

            correction = None
            if verification["label"] == "contradicted":
                # Debate Loop: Corrector vs. Critic
                print(f"--> Initiating Debate for hallucinated claim: '{claim}'")
                draft = correct_claim(claim, evidence_text)
                feedback = None
                
                for attempt in range(2):
                    print(f"    Critic analyzing draft (Attempt {attempt+1}): {draft}")
                    critique = review_correction(claim, draft, evidence_text)
                    
                    if critique["is_accurate"]:
                        print("    Critic approved: PASS")
                        break
                    else:
                        feedback = critique["feedback"]
                        print(f"    Critic rejected: FAIL - {feedback}")
                        draft = correct_claim(claim, evidence_text, feedback=feedback)
                        
                correction = draft

            results.append({
                "claim": claim,
                "evidence": evidence_text,
                "source_url": source_url,
                "verification": verification,
                "correction": correction
            })

        report = generate_report(results)
        
        # Finally, rebuild the original paragraph using the corrections from the report
        print("\nRebuilding the final corrected response...")
        final_text = rebuild_response(answer, report)
        
        return {
            "report": report,
            "final_text": final_text
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

            evidence_text = evidence[0]["text"]
            source_url = evidence[0]["url"]

            yield json.dumps({"status": "progress", "message": f"Verifying claim against evidence..."}) + "\n"
            verification = verify_claim(
                claim,
                evidence_text,
                domain=domain
            )
            
            if verification["label"] == "unknown":
                yield json.dumps({"status": "progress", "message": "NLI returned UNKNOWN. Triggering Critic LLM Override..."}) + "\n"
                llm_label = nli_override(claim, evidence_text, domain=domain)
                verification["label"] = llm_label

            if verification["label"] == "unknown":
                yield json.dumps({"status": "progress", "message": "Critic LLM returned UNKNOWN. Triggering Absence Evaluator..."}) + "\n"
                absence_label = evaluate_absence(claim, ", ".join(search_queries))
                verification["label"] = absence_label
                if absence_label == "contradicted":
                    evidence_text = f"The comprehensive Wikipedia pages for {search_queries} contain no record of this event."

            correction = None
            if verification["label"] == "contradicted":
                yield json.dumps({"status": "progress", "message": f"Initiating Debate Loop for hallucination..."}) + "\n"
                draft = correct_claim(claim, evidence_text)
                feedback = None
                
                for attempt in range(2):
                    yield json.dumps({"status": "progress", "message": f"Critic analyzing draft (Attempt {attempt+1})..."}) + "\n"
                    critique = review_correction(claim, draft, evidence_text)
                    
                    if critique["is_accurate"]:
                        break
                    else:
                        feedback = critique["feedback"]
                        draft = correct_claim(claim, evidence_text, feedback=feedback)
                        
                correction = draft

            results.append({
                "claim": claim,
                "evidence": evidence_text,
                "source_url": source_url,
                "verification": verification,
                "correction": correction
            })

        yield json.dumps({"status": "progress", "message": "Generating final report and rebuilding text..."}) + "\n"
        report = generate_report(results)
        final_text = rebuild_response(answer, report)
        
        yield json.dumps({
            "status": "done",
            "report": report,
            "final_text": final_text
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
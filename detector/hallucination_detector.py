from agents.claim_extractor import extract_claims
from agents.query_generator import generate_queries, rewrite_queries
from agents.entity_linker import extract_entities
from agents.knowledge_graph import KnowledgeGraph
from verification.nli_verifier import verify_claim
from detector.report_generator import generate_report
from agents.critic_agent import nli_override, evaluate_absence


class HallucinationDetector:

    def __init__(self, retriever):
        self.retriever = retriever

    def _verify_claim_full(self, claim, answer, domain, search_queries, requires_current=False):
        max_retries = 1
        retry_count = 0
        
        while retry_count <= max_retries:
            evidence = self.retriever.search(
                search_queries,
                claim=claim,
                domain=domain,
                k=3
            )

            if not evidence:
                if retry_count == 0:
                    print(f"    No evidence found. Triggering Self-Reflective Retrieval to rewrite queries...")
                    new_queries = rewrite_queries(claim, "No evidence returned.", search_queries)
                    if new_queries:
                        search_queries = new_queries
                        retry_count += 1
                        continue
                # If still no evidence, evaluate if absence means contradicted
                absence_label = evaluate_absence(claim, search_queries[0] if search_queries else "unknown")
                return ({"label": absence_label, "confidence": 1.0}, "No evidence found. LLM Absence Evaluation: " + absence_label, None, {"nodes": [], "edges": []})

            evidence_text = "\n\n".join([f"Evidence {i+1}: {e['text']}" for i, e in enumerate(evidence)])
            if requires_current:
                import datetime
                evidence_text = f"[SYSTEM TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d')}]\n" + evidence_text
            source_url = ", ".join(list(set([e["url"] for e in evidence])))

            if len(evidence) > 1:
                source_labels = set()
                for e in evidence:
                    res = verify_claim(claim, e['text'], domain=domain)
                    if res["label"] in ["supported", "contradicted"]:
                        source_labels.add(res["label"])
                
                if "supported" in source_labels and "contradicted" in source_labels:
                    # Defer to Multi-Agent Debate to resolve the contradiction
                    verification = {"label": "unknown", "confidence": 1.0}
                    evidence_text = "EPISTEMIC CONTRADICTION: Retrieved sources provide conflicting information regarding this claim.\n\n" + evidence_text
                else:
                    verification = verify_claim(claim, evidence_text, domain=domain)
            else:
                verification = verify_claim(claim, evidence_text, domain=domain)
            
            if verification["label"] == "unknown" and retry_count == 0:
                print(f"    NLI uncertain. Triggering Self-Reflective Retrieval to rewrite queries...")
                new_queries = rewrite_queries(claim, evidence_text, search_queries)
                if new_queries:
                    search_queries = new_queries
                    retry_count += 1
                    continue

            graph_json = {"nodes": [], "edges": []}
            if verification["label"] == "unknown":
                entities = extract_entities(claim, context=answer)
                if entities:
                    kg = KnowledgeGraph()
                    kg.build(entities)
                    graph_json = kg.get_graph_json()
                    kg_evidence = kg.get_multi_hop_evidence(claim, k=3)
                    if kg_evidence:
                        kg_text = "\n\n".join([f"KG Evidence [{e['node']}]: {e['text']}" for e in kg_evidence])
                        evidence_text = evidence_text + "\n\n" + kg_text
                        source_url = source_url + ", " + ", ".join(list(set([e["url"] for e in kg_evidence])))
                        verification = verify_claim(claim, evidence_text, domain=domain)

            if verification["label"] == "unknown":
                llm_label = nli_override(claim, evidence_text, domain=domain)
                verification["label"] = llm_label

            return verification, evidence_text, source_url, graph_json

    def analyze(self, answer):
        claims = extract_claims(answer)
        results = []

        for claim in claims:
            query_info = generate_queries(claim, context=answer)
            search_queries = query_info["queries"]
            domain = query_info["domain"]

            requires_current = query_info.get("requires_current_state", False)
            print(f"Generated search queries: {search_queries} for claim: '{claim}' (Domain: {domain}, Temporal: {requires_current})")

            verification, evidence_text, source_url, graph_json = self._verify_claim_full(
                claim, answer, domain, search_queries, requires_current
            )

            results.append({
                "claim": claim,
                "evidence": evidence_text,
                "source_url": source_url,
                "verification": verification,
                "graph": graph_json,
                "correction": None
            })

        report = generate_report(results)
        return {
            "report": report,
            "final_text": answer
        }

    def analyze_stream(self, answer, force_kg=False):
        import json
        yield json.dumps({"status": "progress", "message": "Extracting claims..."}) + "\n"
        claims = extract_claims(answer)
        results = []

        for i, claim in enumerate(claims):
            yield json.dumps({"status": "progress", "message": f"Processing claim {i+1} of {len(claims)}..."}) + "\n"

            query_info = generate_queries(claim, context=answer)
            search_queries = query_info["queries"]
            domain = query_info["domain"]
            requires_current = query_info.get("requires_current_state", False)
            
            max_retries = 1
            retry_count = 0
            
            while retry_count <= max_retries:
                msg = f"Searching ({domain} domain) for: {search_queries}..."
                if requires_current:
                    msg = f"Searching ({domain} domain) for: {search_queries} (Injecting Temporal Context)..."
                yield json.dumps({"status": "progress", "message": msg}) + "\n"

                # NLI pass
                evidence = self.retriever.search(search_queries, claim=claim, domain=domain, k=3)

                if not evidence:
                    if retry_count == 0:
                        yield json.dumps({"status": "progress", "message": "No evidence found. Triggering Self-Reflective Retrieval to rewrite queries..."}) + "\n"
                        new_queries = rewrite_queries(claim, "No evidence returned.", search_queries)
                        if new_queries:
                            search_queries = new_queries
                            retry_count += 1
                            continue

                    yield json.dumps({"status": "progress", "message": "Evaluating absence of evidence..."}) + "\n"
                    absence_label = evaluate_absence(claim, search_queries[0] if search_queries else "unknown")
                    
                    results.append({
                        "claim": claim,
                        "evidence": "No evidence found. LLM Absence Evaluation: " + absence_label,
                        "source_url": None,
                        "verification": {"label": absence_label, "confidence": 1.0},
                        "graph": {"nodes": [], "edges": []},
                        "correction": None
                    })
                    break

                evidence_text = "\n\n".join([f"Evidence {i+1}: {e['text']}" for i, e in enumerate(evidence)])
                if requires_current:
                    import datetime
                    evidence_text = f"[SYSTEM TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d')}]\n" + evidence_text
                source_url = ", ".join(list(set([e["url"] for e in evidence])))

                # --- Phase 3.5: Epistemic Contradiction Detection ---
                if len(evidence) > 1:
                    yield json.dumps({"status": "progress", "message": "Checking for epistemic contradictions between sources..."}) + "\n"
                    source_labels = set()
                    for e in evidence:
                        res = verify_claim(claim, e['text'], domain=domain)
                        if res["label"] in ["supported", "contradicted"]:
                            source_labels.add(res["label"])
                    
                    if "supported" in source_labels and "contradicted" in source_labels:
                        yield json.dumps({"status": "progress", "message": "Epistemic contradiction detected! Deferring to Multi-Agent Debate..."}) + "\n"
                        verification = {"label": "unknown", "confidence": 1.0}
                        evidence_text = "EPISTEMIC CONTRADICTION: Retrieved sources provide conflicting information regarding this claim.\n\n" + evidence_text
                    else:
                        yield json.dumps({"status": "progress", "message": "Running NLI verification..."}) + "\n"
                        verification = verify_claim(claim, evidence_text, domain=domain)
                else:
                    yield json.dumps({"status": "progress", "message": "Running NLI verification..."}) + "\n"
                    verification = verify_claim(claim, evidence_text, domain=domain)

                # --- Iterative RAG Trigger ---
                if verification["label"] == "unknown" and retry_count == 0:
                    yield json.dumps({"status": "progress", "message": "NLI uncertain due to poor context. Triggering Self-Reflective Retrieval to rewrite queries..."}) + "\n"
                    new_queries = rewrite_queries(claim, evidence_text, search_queries)
                    if new_queries:
                        search_queries = new_queries
                        retry_count += 1
                        continue

                # Knowledge Graph multi-hop
                graph_json = {"nodes": [], "edges": []}
                if verification["label"] == "unknown" or force_kg:
                    if verification["label"] != "unknown":
                        yield json.dumps({"status": "progress", "message": "Deep Scan active — forcing Knowledge Graph generation..."}) + "\n"
                    else:
                        yield json.dumps({"status": "progress", "message": "NLI uncertain — building Knowledge Graph for multi-hop reasoning..."}) + "\n"
                    entities = extract_entities(claim, context=answer)

                    if entities:
                        kg = KnowledgeGraph()
                        kg.build(entities)
                        
                        graph_json = kg.get_graph_json()

                        yield json.dumps({"status": "progress", "message": f"Knowledge Graph built with {len(graph_json['nodes'])} nodes. Traversing for evidence..."}) + "\n"

                        kg_evidence = kg.get_multi_hop_evidence(claim, k=3)
                        if kg_evidence:
                            kg_text = "\n\n".join([
                                f"KG Evidence [{e['node']}]: {e['text']}" for e in kg_evidence
                            ])
                            evidence_text = evidence_text + "\n\n" + kg_text
                            source_url = source_url + ", " + ", ".join(
                                list(set([e["url"] for e in kg_evidence]))
                            )
                            verification = verify_claim(claim, evidence_text, domain=domain)

                # Critic LLM final fallback
                if verification["label"] == "unknown":
                    yield json.dumps({"status": "progress", "message": "Invoking Multi-Agent Debate (Defense, Prosecution, Judge) for final logic override..."}) + "\n"
                    llm_label = nli_override(claim, evidence_text, domain=domain)
                    verification["label"] = llm_label

                results.append({
                    "claim": claim,
                    "evidence": evidence_text,
                    "source_url": source_url,
                    "verification": verification,
                    "graph": graph_json,
                    "correction": None
                })
                break

        yield json.dumps({"status": "progress", "message": "Generating final report..."}) + "\n"
        report = generate_report(results)

        yield json.dumps({
            "status": "done",
            "report": report,
            "final_text": answer
        }) + "\n"

    def calculate_score(self, results):
        total = len(results)
        if total == 0:
            return 0
        hallucinated = sum(1 for item in results if item["verification"]["label"] == "contradiction")
        return hallucinated / total
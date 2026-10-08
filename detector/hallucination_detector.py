from agents.claim_extractor import extract_claims
from agents.query_generator import generate_queries
from agents.entity_linker import extract_entities
from agents.knowledge_graph import KnowledgeGraph
from verification.nli_verifier import verify_claim
from detector.report_generator import generate_report
from agents.critic_agent import nli_override


class HallucinationDetector:

    def __init__(self, retriever):
        self.retriever = retriever

    def _verify_claim_full(self, claim, answer, domain, search_queries):
        """
        Core verification logic shared by analyze() and analyze_stream().
        Returns (verification dict, evidence_text, source_url, graph_json).
        """
        # --- Phase 3: Hybrid Web Retrieval ---
        evidence = self.retriever.search(
            search_queries,
            claim=claim,
            domain=domain,
            k=3
        )

        if not evidence:
            return (
                {"label": "unknown", "confidence": 1.0},
                "No evidence found.",
                None,
                {"nodes": [], "edges": []}
            )

        evidence_text = "\n\n".join([f"Evidence {i+1}: {e['text']}" for i, e in enumerate(evidence)])
        source_url = ", ".join(list(set([e["url"] for e in evidence])))

        # --- Phase 4: NLI Primary Verification ---
        verification = verify_claim(claim, evidence_text, domain=domain)

        # --- Phase 5a: Knowledge Graph Multi-Hop (when NLI is UNKNOWN) ---
        graph_json = {"nodes": [], "edges": []}
        if verification["label"] == "unknown":
            print(f"    NLI → UNKNOWN. Building Knowledge Graph for multi-hop reasoning...")
            entities = extract_entities(claim, context=answer)
            print(f"    Entities extracted: {[e.get('wikipedia_title') for e in entities]}")

            if entities:
                kg = KnowledgeGraph()
                kg.build(entities)
                graph_json = kg.get_graph_json()

                # Collect multi-hop evidence from the graph
                kg_evidence = kg.get_multi_hop_evidence(claim, k=3)
                if kg_evidence:
                    kg_text = "\n\n".join([
                        f"KG Evidence [{e['node']}]: {e['text']}" for e in kg_evidence
                    ])
                    combined_evidence = evidence_text + "\n\n" + kg_text
                    print(f"    Multi-hop KG evidence collected from {len(kg_evidence)} nodes.")

                    # Re-run NLI on the enriched, multi-hop evidence
                    verification = verify_claim(claim, combined_evidence, domain=domain)
                    evidence_text = combined_evidence
                    source_url = source_url + ", " + ", ".join(
                        list(set([e["url"] for e in kg_evidence]))
                    )

        # --- Phase 5b: Critic LLM Override (final fallback) ---
        if verification["label"] == "unknown":
            print(f"    KG + NLI still UNKNOWN. Triggering Critic LLM Override...")
            llm_label = nli_override(claim, evidence_text, domain=domain)
            print(f"    Critic LLM decided: {llm_label.upper()}")
            verification["label"] = llm_label

        return verification, evidence_text, source_url, graph_json

    def analyze(self, answer):
        claims = extract_claims(answer)
        results = []

        for claim in claims:
            query_info = generate_queries(claim, context=answer)
            search_queries = query_info["queries"]
            domain = query_info["domain"]
            print(f"Generated search queries: {search_queries} for claim: '{claim}' (Domain: {domain})")

            verification, evidence_text, source_url, graph_json = self._verify_claim_full(
                claim, answer, domain, search_queries
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
            yield json.dumps({"status": "progress", "message": f"Searching ({domain} domain) for: {search_queries}..."}) + "\n"

            # NLI pass
            evidence = self.retriever.search(search_queries, claim=claim, domain=domain, k=3)

            if not evidence:
                results.append({
                    "claim": claim,
                    "evidence": "No evidence found.",
                    "source_url": None,
                    "verification": {"label": "unknown", "confidence": 1.0},
                    "graph": {"nodes": [], "edges": []},
                    "correction": None
                })
                continue

            evidence_text = "\n\n".join([f"Evidence {i+1}: {e['text']}" for i, e in enumerate(evidence)])
            source_url = ", ".join(list(set([e["url"] for e in evidence])))

            yield json.dumps({"status": "progress", "message": "Running NLI verification..."}) + "\n"
            verification = verify_claim(claim, evidence_text, domain=domain)

            # Knowledge Graph multi-hop
            graph_json = {"nodes": [], "edges": []}
            if verification["label"] == "unknown":
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
                yield json.dumps({"status": "progress", "message": "Invoking Critic LLM for final logic override..."}) + "\n"
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
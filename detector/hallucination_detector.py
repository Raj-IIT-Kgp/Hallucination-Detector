from agents.claim_extractor import extract_claims
from agents.query_generator import generate_queries, rewrite_queries
from agents.entity_linker import extract_entities
from agents.knowledge_graph import KnowledgeGraph
from verification.nli_verifier import verify_claim, verify_claims_batch
from detector.report_generator import generate_report
from agents.critic_agent import nli_override, evaluate_absence
from statistical_layer.conformal_predictor import ConformalVerifier
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

_SHARED_ROLES = [
    "co-founder", "co founder", "cofounder", 
    "co-author", "co author", "co-creator", "co creator", 
    "member", "director", "investor", "actor"
]


class HallucinationDetector:

    def __init__(self, retriever, calib_file=None, debate_mode="dialectical"):
        self.retriever = retriever
        self.debate_mode = debate_mode

        # Load or locate pre-calibrated conformal parameters
        if calib_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            calib_file = os.path.join(base_dir, "experiments", "conformal_calibration.json")
        
        if os.path.exists(calib_file):
            self.conformal_verifier = ConformalVerifier(calib_file=calib_file)
        else:
            self.conformal_verifier = ConformalVerifier(alpha=0.10)
            self.conformal_verifier.q_hat = 0.8985
            self.conformal_verifier.calibrated_scores = np.linspace(0.0001, 0.999, 100)

    def _verify_snippets(self, claim, evidence_list, domain):
        """
        Runs batched NLI verification on all evidence snippets in a single forward pass on GPU/MPS.
        """
        if not evidence_list:
            return []
        texts = [e['text'] for e in evidence_list]
        verifications = verify_claims_batch(claim, texts, domain=domain)
        return list(zip(verifications, texts))


    def _verify_claim_full(self, claim, answer, domain, search_queries, requires_current=False):
        max_retries = 1
        retry_count = 0
        accumulated_evidence = []
        is_shared_role = any(role in claim.lower() for role in _SHARED_ROLES)
        
        while retry_count <= max_retries:
            evidence = self.retriever.search(
                search_queries,
                claim=claim,
                domain=domain,
                k=3
            )

            if evidence:
                seen_texts = {e['text'] for e in accumulated_evidence}
                for e in evidence:
                    if e['text'] not in seen_texts:
                        accumulated_evidence.append(e)
                        seen_texts.add(e['text'])
            elif accumulated_evidence:
                # Retain and utilize evidence from earlier pass rather than treating as absent
                evidence = list(accumulated_evidence)

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
                abs_probs = [1.0, 0.0, 0.0] if absence_label == "contradicted" else [0.0, 1.0, 0.0]
                conf_res = self.conformal_verifier.predict_set(abs_probs)
                verification = {
                    "label": absence_label,
                    "confidence": 1.0,
                    "probabilities": abs_probs,
                    "conformal_set": conf_res["conformal_set"],
                    "coverage_guarantee": conf_res["coverage_guarantee"],
                    "is_escalation_candidate": False,
                    "conformal_p_values": conf_res["p_values"],
                    "conformal_q_hat": conf_res["q_hat"],
                    "set_size": conf_res["set_size"],
                    "final_decision_policy": "absence_heuristic"
                }
                return (verification, "No evidence found. LLM Absence Evaluation: " + absence_label, None, {"nodes": [], "edges": []})

            evidence_text = "\n\n".join([f"Evidence {i+1}: {e['text']}" for i, e in enumerate(evidence)])
            if requires_current:
                import datetime
                evidence_text = f"[SYSTEM TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d')}]\n" + evidence_text
            source_url = ", ".join(list(set([e["url"] for e in evidence])))

            if len(evidence) > 1:
                # Hierarchical Evidence Scoring: Score snippets in PARALLEL
                snippet_results = self._verify_snippets(claim, evidence, domain)

                # 1. Calibrated Epistemic Contradiction Detection:
                # Requires genuine high-confidence disagreement (>= 0.75 on both sides)
                has_high_conf_supported = any(r["label"] == "supported" and r["confidence"] >= 0.75 for r, _ in snippet_results)
                has_high_conf_contradicted = any(r["label"] == "contradicted" and r["confidence"] >= 0.75 for r, _ in snippet_results)

                if has_high_conf_supported and has_high_conf_contradicted:
                    # Genuinely conflicting sources: escalate to debate
                    verification = {
                        "label": "unknown",
                        "confidence": 1.0,
                        "probabilities": [0.33, 0.34, 0.33]
                    }
                    evidence_text = "EPISTEMIC CONTRADICTION: Retrieved sources provide conflicting information regarding this claim.\n\n" + evidence_text
                else:
                    # 2. Per-Paragraph Max Entailment:
                    # If any single snippet provides decisive proof (confidence >= 0.80),
                    # use that decisive snippet directly to prevent context dilution!
                    # For shared/collaborative roles (e.g. co-founder, co-author), a snippet naming another
                    # person is not a decisive contradiction; require multi-snippet evaluation or debate.
                    decisive_snippets = [
                        r for r, _ in snippet_results 
                        if (r["label"] == "supported" and r["confidence"] >= 0.80) or
                           (r["label"] == "contradicted" and r["confidence"] >= 0.80 and not is_shared_role)
                    ]
                    if decisive_snippets:
                        decisive_snippets.sort(key=lambda x: x["confidence"], reverse=True)
                        verification = decisive_snippets[0]
                    else:
                        # 3. Cross-sentence reasoning: fall back to concatenated evidence
                        verification = verify_claim(claim, evidence_text, domain=domain)
            else:
                verification = verify_claim(claim, evidence_text, domain=domain)
            
            # --- Phase 3 Conformal Prediction Verification ---
            conf_res = self.conformal_verifier.predict_set(verification.get("probabilities", [0.0, 1.0, 0.0]))
            verification.update({
                "conformal_set": conf_res["conformal_set"],
                "coverage_guarantee": conf_res["coverage_guarantee"],
                "is_escalation_candidate": conf_res["is_escalation_candidate"],
                "conformal_p_values": conf_res["p_values"],
                "conformal_q_hat": conf_res["q_hat"],
                "set_size": conf_res["set_size"],
                "final_decision_policy": "conformal_singleton" if not conf_res["is_escalation_candidate"] else "escalation_required"
            })

            is_escalation = verification["label"] == "unknown" or conf_res["is_escalation_candidate"] or (is_shared_role and verification["label"] == "contradicted")

            if is_escalation and retry_count == 0:
                print(f"    NLI uncertain / Conformal dispersion {conf_res['conformal_set']}. Triggering Self-Reflective Retrieval to rewrite queries...")
                new_queries = rewrite_queries(claim, evidence_text, search_queries)
                if new_queries:
                    search_queries = new_queries
                    retry_count += 1
                    continue

            graph_json = {"nodes": [], "edges": []}
            if is_escalation:
                entities = search_queries if (search_queries and search_queries != [claim]) else extract_entities(claim, context=answer)
                if entities:
                    kg = KnowledgeGraph()
                    kg.build(entities, claim=claim)
                    graph_json = kg.get_graph_json()
                    kg_evidence = kg.get_multi_hop_evidence(claim, k=3)
                    if kg_evidence:
                        kg_text = "\n\n".join([f"KG Evidence [{e['node']}]: {e['text']}" for e in kg_evidence])
                        evidence_text = evidence_text + "\n\n" + kg_text
                        source_url = source_url + ", " + ", ".join(list(set([e["url"] for e in kg_evidence])))

                # Directly route to Multi-Agent Debate with augmented KG evidence,
                # bypassing redundant NLI cross-encoder evaluation on diluted concatenated text
                llm_label = nli_override(claim, evidence_text, domain=domain, mode=self.debate_mode)
                verification["label"] = llm_label
                verification["final_decision_policy"] = "multi_agent_debate"
                verification["debate_resolved"] = True

            return verification, evidence_text, source_url, graph_json

    def _process_claim(self, claim, answer):
        """
        Processes a single claim end-to-end. Designed to be called from a thread pool
        so that multiple claims can be processed concurrently.
        """
        query_info = generate_queries(claim, context=answer)
        search_queries = query_info["queries"]
        domain = query_info["domain"]
        requires_current = query_info.get("requires_current_state", False)

        print(f"Generated search queries: {search_queries} for claim: '{claim}' (Domain: {domain}, Temporal: {requires_current})")

        verification, evidence_text, source_url, graph_json = self._verify_claim_full(
            claim, answer, domain, search_queries, requires_current
        )

        return {
            "claim": claim,
            "evidence": evidence_text,
            "source_url": source_url,
            "verification": verification,
            "graph": graph_json,
            "correction": None
        }

    def analyze(self, answer):
        claims = extract_claims(answer)
        if not claims:
            report = generate_report([])
            return {
                "claims": [],
                "report": report,
                "final_text": answer
            }

        # Process all claims concurrently — each claim's retrieval + NLI runs in parallel
        results_map = {}
        with ThreadPoolExecutor(max_workers=max(1, min(len(claims), 5))) as executor:
            future_to_idx = {
                executor.submit(self._process_claim, claim, answer): i
                for i, claim in enumerate(claims)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_map[idx] = future.result()
                except Exception as e:
                    print(f"[HallucinationDetector] Claim {idx} failed: {e}")
                    results_map[idx] = {
                        "claim": claims[idx],
                        "evidence": f"Error during processing: {e}",
                        "source_url": None,
                        "verification": {"label": "unknown", "confidence": 0.0, "probabilities": [0.0, 1.0, 0.0]},
                        "graph": {"nodes": [], "edges": []},
                        "correction": None
                    }

        # Restore original order
        results = [results_map[i] for i in range(len(claims))]

        report = generate_report(results)
        return {
            "claims": results,
            "report": report,
            "final_text": answer
        }

    def analyze_stream(self, answer, force_kg=False):
        import json
        yield json.dumps({"status": "progress", "message": "Extracting claims..."}) + "\n"
        claims = extract_claims(answer)
        if not claims:
            yield json.dumps({"status": "done", "report": generate_report([])}) + "\n"
            return

        results = [None] * len(claims)

        # ── Pre-generate all queries in parallel before starting the main loop ──
        yield json.dumps({"status": "progress", "message": f"Pre-generating search queries for {len(claims)} claims in parallel..."}) + "\n"

        def _gen_queries(args):
            i, claim = args
            return i, generate_queries(claim, context=answer)

        query_map = {}
        with ThreadPoolExecutor(max_workers=max(1, min(len(claims), 5))) as executor:
            for i, query_info in executor.map(_gen_queries, enumerate(claims)):
                query_map[i] = query_info

        for i, claim in enumerate(claims):
            is_shared_role = any(role in claim.lower() for role in _SHARED_ROLES)
            yield json.dumps({"status": "progress", "message": f"Processing claim {i+1} of {len(claims)}..."}) + "\n"

            query_info = query_map[i]
            search_queries = query_info["queries"]
            domain = query_info["domain"]
            requires_current = query_info.get("requires_current_state", False)
            
            max_retries = 1
            retry_count = 0
            accumulated_evidence = []
            
            while retry_count <= max_retries:
                msg = f"Searching ({domain} domain) for: {search_queries}..."
                if requires_current:
                    msg = f"Searching ({domain} domain) for: {search_queries} (Injecting Temporal Context)..."
                yield json.dumps({"status": "progress", "message": msg}) + "\n"

                # NLI pass
                evidence = self.retriever.search(search_queries, claim=claim, domain=domain, k=3)

                if evidence:
                    seen_texts = {e['text'] for e in accumulated_evidence}
                    for e in evidence:
                        if e['text'] not in seen_texts:
                            accumulated_evidence.append(e)
                            seen_texts.add(e['text'])
                elif accumulated_evidence:
                    evidence = list(accumulated_evidence)

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
                    abs_probs = [1.0, 0.0, 0.0] if absence_label == "contradicted" else [0.0, 1.0, 0.0]
                    conf_res = self.conformal_verifier.predict_set(abs_probs)
                    verification = {
                        "label": absence_label,
                        "confidence": 1.0,
                        "probabilities": abs_probs,
                        "conformal_set": conf_res["conformal_set"],
                        "coverage_guarantee": conf_res["coverage_guarantee"],
                        "is_escalation_candidate": False,
                        "conformal_p_values": conf_res["p_values"],
                        "conformal_q_hat": conf_res["q_hat"],
                        "set_size": conf_res["set_size"],
                        "final_decision_policy": "absence_heuristic"
                    }
                    
                    results[i] = {
                        "claim": claim,
                        "evidence": "No evidence found. LLM Absence Evaluation: " + absence_label,
                        "source_url": None,
                        "verification": verification,
                        "graph": {"nodes": [], "edges": []},
                        "correction": None
                    }
                    break

                evidence_text = "\n\n".join([f"Evidence {i+1}: {e['text']}" for i, e in enumerate(evidence)])
                if requires_current:
                    import datetime
                    evidence_text = f"[SYSTEM TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d')}]\n" + evidence_text
                source_url = ", ".join(list(set([e["url"] for e in evidence])))

                # --- Phase 3.5: Hierarchical Evidence Scoring & Epistemic Contradiction ---
                if len(evidence) > 1:
                    yield json.dumps({"status": "progress", "message": "Evaluating evidence paragraphs with hierarchical scoring (parallel)..."}) + "\n"
                    # Run NLI on snippets in PARALLEL
                    snippet_results = self._verify_snippets(claim, evidence, domain)

                    has_high_conf_supported = any(r["label"] == "supported" and r["confidence"] >= 0.75 for r, _ in snippet_results)
                    has_high_conf_contradicted = any(r["label"] == "contradicted" and r["confidence"] >= 0.75 for r, _ in snippet_results)

                    if has_high_conf_supported and has_high_conf_contradicted:
                        yield json.dumps({"status": "progress", "message": "High-confidence epistemic contradiction detected between sources! Escalating to debate..."}) + "\n"
                        verification = {
                            "label": "unknown",
                            "confidence": 1.0,
                            "probabilities": [0.33, 0.34, 0.33]
                        }
                        evidence_text = "EPISTEMIC CONTRADICTION: Retrieved sources provide conflicting information regarding this claim.\n\n" + evidence_text
                    else:
                        decisive_snippets = [
                            r for r, _ in snippet_results 
                            if (r["label"] == "supported" and r["confidence"] >= 0.80) or
                               (r["label"] == "contradicted" and r["confidence"] >= 0.80 and not is_shared_role)
                        ]
                        if decisive_snippets:
                            decisive_snippets.sort(key=lambda x: x["confidence"], reverse=True)
                            verification = decisive_snippets[0]
                            yield json.dumps({"status": "progress", "message": f"Decisive snippet found ({verification['label']}, confidence {verification['confidence']:.2f}) — context dilution prevented."}) + "\n"
                        else:
                            yield json.dumps({"status": "progress", "message": "Running multi-sentence cross-paragraph NLI verification..."}) + "\n"
                            verification = verify_claim(claim, evidence_text, domain=domain)
                else:
                    yield json.dumps({"status": "progress", "message": "Running NLI verification..."}) + "\n"
                    verification = verify_claim(claim, evidence_text, domain=domain)

                # --- Phase 3 Conformal Prediction Verification ---
                conf_res = self.conformal_verifier.predict_set(verification.get("probabilities", [0.0, 1.0, 0.0]))
                verification.update({
                    "conformal_set": conf_res["conformal_set"],
                    "coverage_guarantee": conf_res["coverage_guarantee"],
                    "is_escalation_candidate": conf_res["is_escalation_candidate"],
                    "conformal_p_values": conf_res["p_values"],
                    "conformal_q_hat": conf_res["q_hat"],
                    "set_size": conf_res["set_size"],
                    "final_decision_policy": "conformal_singleton" if not conf_res["is_escalation_candidate"] else "escalation_required"
                })

                set_str = ", ".join(conf_res["conformal_set"]).upper()
                yield json.dumps({"status": "progress", "message": f"Conformal Set: [{set_str}] (Guaranteed Coverage: {conf_res['coverage_guarantee']})"}) + "\n"

                is_escalation = verification["label"] == "unknown" or conf_res["is_escalation_candidate"] or (is_shared_role and verification["label"] == "contradicted")

                # --- Iterative RAG Trigger ---
                if is_escalation and retry_count == 0:
                    yield json.dumps({"status": "progress", "message": f"Conformal set size {conf_res['set_size']} indicates posterior dispersion — triggering Self-Reflective Retrieval..."}) + "\n"
                    new_queries = rewrite_queries(claim, evidence_text, search_queries)
                    if new_queries:
                        search_queries = new_queries
                        retry_count += 1
                        continue

                # Knowledge Graph multi-hop
                graph_json = {"nodes": [], "edges": []}
                if is_escalation or force_kg:
                    if not is_escalation and force_kg:
                        yield json.dumps({"status": "progress", "message": "Deep Scan active — forcing Knowledge Graph generation..."}) + "\n"
                    else:
                        yield json.dumps({"status": "progress", "message": "Conformal escalation triggered — building Knowledge Graph for multi-hop reasoning..."}) + "\n"
                    entities = search_queries if (search_queries and search_queries != [claim]) else extract_entities(claim, context=answer)

                    if entities:
                        kg = KnowledgeGraph()
                        kg.build(entities, claim=claim)
                        
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

                # Critic LLM final fallback
                if is_escalation:
                    yield json.dumps({"status": "progress", "message": "Invoking Multi-Agent Debate for final logic override..."}) + "\n"
                    llm_label = nli_override(claim, evidence_text, domain=domain, mode=self.debate_mode)
                    verification["label"] = llm_label
                    verification["final_decision_policy"] = "multi_agent_debate"
                    verification["debate_resolved"] = True
                    yield json.dumps({"status": "progress", "message": f"Debate completed: Final verdict '{llm_label.upper()}'"}) + "\n"

                results[i] = {
                    "claim": claim,
                    "evidence": evidence_text,
                    "source_url": source_url,
                    "verification": verification,
                    "graph": graph_json,
                    "correction": None
                }
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
        hallucinated = sum(1 for item in results if item["verification"]["label"] in ["contradicted", "contradiction"])
        return hallucinated / total
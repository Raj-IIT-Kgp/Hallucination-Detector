from agents.claim_extractor import extract_claims
from agents.query_generator import generate_queries, rewrite_queries
from agents.entity_linker import extract_entities
from agents.knowledge_graph import KnowledgeGraph
from verification.nli_verifier import verify_claim, verify_claims_batch
from detector.report_generator import generate_report
from agents.critic_agent import nli_override, evaluate_absence
from statistical_layer.conformal_predictor import ConformalVerifier
import os
import re
import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

_SHARED_ROLES = [
    "co-founder", "co founder", "cofounder",
    "co-author", "co author", "co-creator", "co creator",
    "member", "director", "investor", "actor"
]

# Per-snippet NLI confidence required for a snippet to decide a claim on its own (Phase 4 Tier 1)
_DECISIVE_CONF = 0.80
# A lone decisive contradiction is unreliable: on 105 dev claims, claims with exactly one decisive
# contradicting snippet were actually true 12/21 times, vs 3/16 with three. CONTRADICTED is
# accepted directly only with this many agreeing snippets; otherwise the LLM arbiter reviews it.
_MIN_CONTRA_SNIPPETS = 2
# A web sentence containing >= this fraction of the claim's tokens is treated as a quote of the claim
_ECHO_OVERLAP = 0.8
# Snippets left with fewer tokens than this after echo stripping carry no independent evidence
_MIN_RESIDUAL_TOKENS = 8
# Bound on cached (claim, snippet) NLI results held by one detector instance
_NLI_CACHE_MAX = 5000

_WORD_RE = re.compile(r"\w+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_claim_echoes(claim, evidence):
    """
    Removes sentences from open-web snippets that merely restate the claim (blogs, fact-check
    pages and the original source quoting it verbatim), which NLI would otherwise read as
    entailment. Wikipedia/ArXiv paragraphs are left untouched: encyclopedic sentences that
    match a claim closely are genuine support.
    """
    claim_tokens = set(_WORD_RE.findall(claim.lower()))
    if not claim_tokens:
        return evidence

    cleaned = []
    for e in evidence:
        if e.get("source") != "web":
            cleaned.append(e)
            continue
        kept = [
            s for s in _SENT_SPLIT_RE.split(e["text"])
            if len(claim_tokens & set(_WORD_RE.findall(s.lower()))) / len(claim_tokens) < _ECHO_OVERLAP
        ]
        text = " ".join(kept).strip()
        if len(_WORD_RE.findall(text)) >= _MIN_RESIDUAL_TOKENS:
            cleaned.append({**e, "text": text})
    return cleaned


class HallucinationDetector:

    def __init__(self, retriever, calib_file=None, debate_mode="dialectical",
                 min_contra_snippets=_MIN_CONTRA_SNIPPETS, strip_claim_echoes=True, strict_premises=True):
        """
        One configuration serves every input; the keyword switches exist only for ablations.
          min_contra_snippets: decisive contradicting snippets needed to accept CONTRADICTED without
              LLM review (None restores the original max-confidence rule)
          strip_claim_echoes: drop open-web sentences that merely quote the claim
          strict_premises: arbiter treats a wrong name/title/date/attribute as CONTRADICTED
        """
        self.retriever = retriever
        self.debate_mode = debate_mode
        self.min_contra_snippets = min_contra_snippets
        self.strip_claim_echoes = strip_claim_echoes
        self.strict_premises = strict_premises
        # Per-(claim, snippet) NLI results: query-rewrite retries re-retrieve many of the same
        # paragraphs, which would otherwise be re-scored by the ~1s/batch cross-encoder.
        self._nli_cache = {}

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
        missing = [t for t in dict.fromkeys(texts) if (claim, t) not in self._nli_cache]
        if missing:
            if len(self._nli_cache) > _NLI_CACHE_MAX:
                self._nli_cache.clear()
            for t, v in zip(missing, verify_claims_batch(claim, missing, domain=domain)):
                self._nli_cache[(claim, t)] = v
        # Copies, because callers attach conformal fields to the chosen verification dict
        return [(dict(self._nli_cache[(claim, t)]), t) for t in texts]

    def _score_evidence(self, claim, evidence, evidence_text, domain, is_shared_role):
        """
        Phase 3.2 consensus check + Phase 4 hierarchical scoring.
        Returns (verification, evidence_text, needs_arbitration, progress_message).

        Decisive support settles a claim directly, as does a contradiction corroborated by at least
        min_contra_snippets decisive snippets with no decisive support. A lone or contested decisive
        contradiction keeps its NLI posterior but sets needs_arbitration, so the caller has the LLM
        arbiter review it (one call, no KG construction).
        """
        corroborate = self.min_contra_snippets is not None

        if len(evidence) <= 1:
            verification = verify_claim(claim, evidence_text, domain=domain)
            # A single-snippet evidence pool can never corroborate a contradiction
            needs_arbitration = (corroborate and verification["label"] == "contradicted"
                                 and verification["confidence"] >= _DECISIVE_CONF)
            return verification, evidence_text, needs_arbitration, "Running NLI verification..."

        # Hierarchical Evidence Scoring: Score snippets in PARALLEL
        snippet_results = self._verify_snippets(claim, evidence, domain)

        # 1. Calibrated Epistemic Contradiction Detection:
        # Requires genuine symmetric high-confidence disagreement (>= 0.85 on both sides within 0.12 margin)
        max_sup_conf = max([r["confidence"] for r, _ in snippet_results if r["label"] == "supported"], default=0.0)
        max_con_conf = max([r["confidence"] for r, _ in snippet_results if r["label"] == "contradicted"], default=0.0)
        if max_sup_conf >= 0.85 and max_con_conf >= 0.85 and abs(max_sup_conf - max_con_conf) < 0.12:
            # Genuinely conflicting sources: escalate to debate
            verification = {
                "label": "unknown",
                "confidence": 1.0,
                "probabilities": [0.33, 0.34, 0.33]
            }
            evidence_text = "EPISTEMIC CONTRADICTION: Retrieved sources provide conflicting information regarding this claim.\n\n" + evidence_text
            return verification, evidence_text, False, "High-confidence epistemic contradiction detected between sources! Escalating to debate..."

        strong_sup = sorted(
            [r for r, _ in snippet_results if r["label"] == "supported" and r["confidence"] >= _DECISIVE_CONF],
            key=lambda x: x["confidence"], reverse=True
        )
        strong_con = sorted(
            [r for r, _ in snippet_results if r["label"] == "contradicted" and r["confidence"] >= _DECISIVE_CONF],
            key=lambda x: x["confidence"], reverse=True
        )

        if corroborate and strong_con:
            # 2a. Corroborated contradictions
            if len(strong_con) >= self.min_contra_snippets and not strong_sup and not is_shared_role:
                verification = strong_con[0]
                return verification, evidence_text, False, f"Corroborated contradiction ({len(strong_con)} decisive snippets, confidence {verification['confidence']:.2f})."
            # Lone, contested or shared-role contradiction: keep the strongest NLI posterior, let the arbiter review
            verification = max(strong_sup + strong_con, key=lambda x: x["confidence"])
            return verification, evidence_text, True, "Uncorroborated or contested contradiction — routing to LLM review..."

        # 2. Per-Paragraph Max Entailment:
        # If any single snippet provides decisive proof (confidence >= 0.80),
        # use that decisive snippet directly to prevent context dilution!
        # For shared/collaborative roles, a single snippet asserting co-authorship/founding
        # cannot short-circuit rejection of an exclusive founder claim.
        decisive_snippets = strong_sup + [r for r in strong_con if not is_shared_role]
        if decisive_snippets:
            decisive_snippets.sort(key=lambda x: x["confidence"], reverse=True)
            verification = decisive_snippets[0]
            return verification, evidence_text, False, f"Decisive snippet found ({verification['label']}, confidence {verification['confidence']:.2f}) — context dilution prevented."

        # 3. Cross-sentence reasoning: fall back to concatenated evidence
        verification = verify_claim(claim, evidence_text, domain=domain)
        return verification, evidence_text, False, "Running multi-sentence cross-paragraph NLI verification..."

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
            if self.strip_claim_echoes:
                evidence = _strip_claim_echoes(claim, evidence)

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
                evidence_text = f"[SYSTEM TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d')}]\n" + evidence_text
            source_url = ", ".join(list(set([e["url"] for e in evidence])))

            verification, evidence_text, needs_arbitration, _ = self._score_evidence(
                claim, evidence, evidence_text, domain, is_shared_role
            )

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

            is_escalation = (
                verification["label"] == "unknown" or 
                conf_res["is_escalation_candidate"] or 
                (is_shared_role and verification["label"] == "contradicted")
            )

            # Only trigger query rewrite when evidence was inconclusive (unknown).
            # When label is already supported/contradicted with conformal dispersion,
            # evidence is already present—escalate directly to KG + Critic Debate.
            if verification["label"] == "unknown" and retry_count == 0:
                print(f"    NLI uncertain ({conf_res['conformal_set']}). Triggering Self-Reflective Retrieval to rewrite queries...")
                new_queries = rewrite_queries(claim, evidence_text, search_queries)
                if new_queries:
                    search_queries = new_queries
                    retry_count += 1
                    continue

            graph_json = {"nodes": [], "edges": []}
            if is_escalation:
                entities = search_queries if (search_queries and search_queries != [claim]) else extract_entities(claim, context=answer)
                if entities:
                    cached_pages = self.retriever.get_page_cache() if hasattr(self.retriever, "get_page_cache") else None
                    kg = KnowledgeGraph()
                    kg.build(entities, claim=claim, cached_pages=cached_pages, evidence=evidence_text)
                    graph_json = kg.get_graph_json()
                    kg_evidence = kg.get_multi_hop_evidence(claim, k=3)
                    if kg_evidence:
                        kg_text = "\n\n".join([f"KG Evidence [{e['node']}]: {e['text']}" for e in kg_evidence])
                        evidence_text = evidence_text + "\n\n" + kg_text
                        source_url = source_url + ", " + ", ".join(list(set([e["url"] for e in kg_evidence])))

                # Directly route to Multi-Agent Debate with augmented KG evidence,
                # bypassing redundant NLI cross-encoder evaluation on diluted concatenated text
                llm_label = nli_override(claim, evidence_text, domain=domain, mode=self.debate_mode, strict_premises=self.strict_premises)
                verification["label"] = llm_label
                verification["final_decision_policy"] = "multi_agent_debate"
                verification["debate_resolved"] = True

                # If the debate concluded SUPPORTED and the graph had nodes without edges,
                # ensure the supporting edge is linked between primary nodes using the verified evidence
                if llm_label == "supported" and graph_json.get("nodes") and len(graph_json["nodes"]) >= 2 and not graph_json.get("edges"):
                    n1 = graph_json["nodes"][0]["id"]
                    n2 = graph_json["nodes"][1]["id"]
                    from agents.knowledge_graph import _extract_direct_relationship_from_evidence
                    direct = _extract_direct_relationship_from_evidence(n1, n2, claim, evidence_text)
                    if direct:
                        kg.add_grounded_edge(direct["source"], direct["target"], direct["relationship"], quote=direct["quote"], verified=direct["verified"])
                        graph_json = kg.get_graph_json()
            elif needs_arbitration:
                # Lightweight review of an uncorroborated or contested contradiction: one LLM call over
                # the evidence already in hand, skipping KG construction (the expensive step)
                verification["label"] = nli_override(claim, evidence_text, domain=domain, mode=self.debate_mode, strict_premises=self.strict_premises)
                verification["final_decision_policy"] = "contradiction_arbitration"
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

    def analyze(self, answer, context=None, extract_atomic_facts=True):
        if extract_atomic_facts:
            claims = extract_claims(answer)
        else:
            claims = [answer.strip()] if answer and answer.strip() else []

        if not claims:
            report = generate_report([])
            return {
                "claims": [],
                "report": report,
                "final_text": answer
            }

        claim_context = context if context else answer
        # Process all claims concurrently — each claim's retrieval + NLI runs in parallel
        results_map = {}
        with ThreadPoolExecutor(max_workers=max(1, min(len(claims), 5))) as executor:
            future_to_idx = {
                executor.submit(self._process_claim, claim, claim_context): i
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
                if self.strip_claim_echoes:
                    evidence = _strip_claim_echoes(claim, evidence)

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
                    evidence_text = f"[SYSTEM TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d')}]\n" + evidence_text
                source_url = ", ".join(list(set([e["url"] for e in evidence])))

                # --- Phase 3.5: Hierarchical Evidence Scoring & Epistemic Contradiction ---
                if len(evidence) > 1:
                    yield json.dumps({"status": "progress", "message": "Evaluating evidence paragraphs with hierarchical scoring (parallel)..."}) + "\n"
                verification, evidence_text, needs_arbitration, score_msg = self._score_evidence(
                    claim, evidence, evidence_text, domain, is_shared_role
                )
                yield json.dumps({"status": "progress", "message": score_msg}) + "\n"

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
                # Only rewrite queries when evidence was inconclusive (unknown).
                # If label is already supported/contradicted, escalate directly to KG multi-hop + debate.
                if verification["label"] == "unknown" and retry_count == 0:
                    yield json.dumps({"status": "progress", "message": f"NLI uncertain ({conf_res['conformal_set']}) — triggering Self-Reflective Retrieval to rewrite queries..."}) + "\n"
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
                        cached_pages = self.retriever.get_page_cache() if hasattr(self.retriever, "get_page_cache") else None
                        kg = KnowledgeGraph()
                        kg.build(entities, claim=claim, cached_pages=cached_pages, evidence=evidence_text)
                        
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
                    llm_label = nli_override(claim, evidence_text, domain=domain, mode=self.debate_mode, strict_premises=self.strict_premises)
                    verification["label"] = llm_label
                    verification["final_decision_policy"] = "multi_agent_debate"
                    verification["debate_resolved"] = True
                    yield json.dumps({"status": "progress", "message": f"Debate completed: Final verdict '{llm_label.upper()}'"}) + "\n"

                    if llm_label == "supported" and graph_json.get("nodes") and len(graph_json["nodes"]) >= 2 and not graph_json.get("edges"):
                        n1 = graph_json["nodes"][0]["id"]
                        n2 = graph_json["nodes"][1]["id"]
                        from agents.knowledge_graph import _extract_direct_relationship_from_evidence
                        direct = _extract_direct_relationship_from_evidence(n1, n2, claim, evidence_text)
                        if direct:
                            kg.add_grounded_edge(direct["source"], direct["target"], direct["relationship"], quote=direct["quote"], verified=direct["verified"])
                            graph_json = kg.get_graph_json()
                elif needs_arbitration:
                    yield json.dumps({"status": "progress", "message": "Reviewing uncorroborated contradiction (no KG)..."}) + "\n"
                    llm_label = nli_override(claim, evidence_text, domain=domain, mode=self.debate_mode, strict_premises=self.strict_premises)
                    verification["label"] = llm_label
                    verification["final_decision_policy"] = "contradiction_arbitration"
                    verification["debate_resolved"] = True
                    yield json.dumps({"status": "progress", "message": f"Arbitration completed: Final verdict '{llm_label.upper()}'"}) + "\n"

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
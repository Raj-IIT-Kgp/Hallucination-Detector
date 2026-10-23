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
import inspect
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


def _decision_case(message):
    """Names the branch of the evidence-scoring rule from its progress message (for the UI trace)."""
    for prefix, case in (("Running NLI verification", "single"), ("High-confidence epistemic", "conflict"),
                         ("Corroborated contradiction", "corroborated"), ("Uncorroborated or contested", "review"),
                         ("Decisive snippet found", "decisive"), ("Running multi-sentence", "joint")):
        if message.startswith(prefix):
            return case
    return "unknown"


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
        # Retrievers that can report their re-ranking numbers (for the UI trace); others are called as before
        try:
            self._retriever_diagnostics = "diagnostics" in inspect.signature(retriever.search).parameters
        except (TypeError, ValueError, AttributeError):
            self._retriever_diagnostics = False
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
        Scores the evidence (see _decide_from_evidence) and attaches an explanation for the UI as
        verification["evidence_scores"]: each snippet's own NLI result and which snippet decided.
        The explanation reuses cached NLI scores, so it adds no model calls and cannot change the verdict.
        """
        verification, evidence_text, needs_arbitration, message = self._decide_from_evidence(
            claim, evidence, evidence_text, domain, is_shared_role
        )
        explanation = self._explain_evidence(claim, evidence, evidence_text, domain, verification)
        explanation.update({"case": _decision_case(message), "shared_role": is_shared_role,
                            "tau": _DECISIVE_CONF, "m": self.min_contra_snippets})
        verification["evidence_scores"] = explanation
        return verification, evidence_text, needs_arbitration, message

    def _explain_evidence(self, claim, evidence, evidence_text, domain, verification):
        """
        Per-snippet NLI results in evidence order, and decided_by: the index of the snippet whose
        posterior became the verdict, "joint" when the snippets were scored together (no snippet was
        decisive), or "conflict" when decisive snippets disagreed (§2.5 of pipeline.md).
        """
        if len(evidence) <= 1:
            results, decided_by = [verification], 0
        else:
            results = [r for r, _ in self._verify_snippets(claim, evidence, domain)]   # cached: no new inference
            if evidence_text.startswith("EPISTEMIC CONTRADICTION"):
                decided_by = "conflict"
            else:
                decided_by = next((i for i, r in enumerate(results)
                                   if r["label"] == verification["label"] and r["probabilities"] == verification["probabilities"]),
                                  "joint")
        snippets = []
        for e, r in zip(evidence, results):
            url = e.get("url") or ""
            snippets.append({
                "source": e.get("source") or ("arxiv" if "arxiv.org" in url else "wikipedia"),
                "url": url,
                "label": r["label"],
                "probabilities": list(r["probabilities"]),
                "logits": r.get("logits"),
                "decisive": r["confidence"] >= _DECISIVE_CONF,
            })
        # The quantities of the decision rule (§2.5): strongest support S and contradiction K, and the
        # number of decisive supporting / contradicting snippets
        def strongest(label):
            return max([r["confidence"] for r in results if r["label"] == label], default=0.0)
        return {
            "decided_by": decided_by, "snippets": snippets,
            "S": strongest("supported"), "K": strongest("contradicted"),
            "n_sup": sum(1 for s in snippets if s["decisive"] and s["label"] == "supported"),
            "n_con": sum(1 for s in snippets if s["decisive"] and s["label"] == "contradicted"),
        }

    def _decide_from_evidence(self, claim, evidence, evidence_text, domain, is_shared_role):
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

    def warm_up(self):
        """
        Runs each local model once so the first real request does not pay one-off start-up costs
        (MPS kernel compilation, lazy model and tokenizer loading). Nothing is cached and no
        network or LLM call is made, so later results are unaffected.
        """
        from agents.knowledge_graph import _get_encoder
        from nltk.tokenize import sent_tokenize
        verify_claims_batch("The sky is blue.", ["The sky appears blue because of Rayleigh scattering."])
        self.retriever.encoder.encode(["warm-up"], show_progress_bar=False)
        _get_encoder().encode(["warm-up"], show_progress_bar=False)
        sent_tokenize("Warm-up sentence one. Warm-up sentence two.")

    def _verify_claim_full(self, claim, answer, domain, search_queries, requires_current=False,
                           progress=None, force_kg=False, defer_display=False, trace=None):
        """
        Verifies one claim. Returns (verification, evidence_text, source_url, graph_json, display_task).

        progress(message, stage, data) receives stage events for the streaming UI; data, when given, holds
        the numbers computed at that step, which are also appended to trace (stored in the result as
        verification["trace"]) so the UI can show the computation. force_kg builds the
        Knowledge Graph even when the claim is not escalated (the UI's Deep Scan). With
        defer_display, the display-only graph edge (which can never change the verdict) is returned
        as display_task, a callable producing the enriched graph, instead of being computed before
        the result; otherwise display_task is None.
        """
        progress = progress or (lambda message, stage, data=None: None)
        trace = trace if trace is not None else []

        def emit(message, stage, data=None):
            if data is not None:
                trace.append(data)
            progress(message, stage, data)

        max_retries = 1
        retry_count = 0
        accumulated_evidence = []
        is_shared_role = any(role in claim.lower() for role in _SHARED_ROLES)

        while retry_count <= max_retries:
            msg = f"Searching ({domain} domain) for: {search_queries}"
            emit(msg + (" (Injecting Temporal Context)..." if requires_current else "..."), "retrieve")
            diagnostics = {}
            extra = {"diagnostics": diagnostics} if self._retriever_diagnostics else {}
            evidence = self.retriever.search(
                search_queries,
                claim=claim,
                domain=domain,
                k=3,
                **extra
            )
            retrieved = len(evidence)
            if self.strip_claim_echoes:
                evidence = _strip_claim_echoes(claim, evidence)
            emit(f"Retrieved {diagnostics.get('candidates', 0)} candidate paragraphs; kept {len(evidence)}", "retrieve",
                 {"kind": "retrieval", "pass": retry_count + 1, **diagnostics,
                  "echo_filtered": retrieved - len(evidence), "kept": len(evidence)})

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
                    emit("No evidence found. Triggering Self-Reflective Retrieval to rewrite queries...", "rewrite")
                    new_queries = rewrite_queries(claim, "No evidence returned.", search_queries)
                    emit("Queries rewritten", "rewrite", {"kind": "rewrite", "reason": "no evidence", "queries": new_queries or []})
                    if new_queries:
                        search_queries = new_queries
                        retry_count += 1
                        continue
                # If still no evidence, evaluate if absence means contradicted
                emit("Evaluating absence of evidence...", "arbiter")
                absence_label, llm_error = evaluate_absence(claim, search_queries[0] if search_queries else "unknown", return_error=True)
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
                if llm_error:
                    verification["llm_unavailable"] = llm_error[:300]
                emit(f"Absence evaluation: {absence_label}", "arbiter",
                     {"kind": "absence", "label": absence_label, "error": llm_error})
                verification["trace"] = trace
                return (verification, "No evidence found. LLM Absence Evaluation: " + absence_label, None,
                        {"nodes": [], "edges": []}, None)

            evidence_text = "\n\n".join([f"Evidence {i+1}: {e['text']}" for i, e in enumerate(evidence)])
            if requires_current:
                evidence_text = f"[SYSTEM TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d')}]\n" + evidence_text
            source_url = ", ".join(list(set([e["url"] for e in evidence])))

            if len(evidence) > 1:
                emit("Evaluating evidence paragraphs with hierarchical scoring (parallel)...", "nli")
            verification, evidence_text, needs_arbitration, score_msg = self._score_evidence(
                claim, evidence, evidence_text, domain, is_shared_role
            )
            emit(score_msg, "nli", {"kind": "nli", "pass": retry_count + 1, "label": verification["label"],
                                    "probabilities": verification.get("probabilities"), **verification["evidence_scores"]})

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
            emit(f"Conformal Set: [{set_str}] (Guaranteed Coverage: {conf_res['coverage_guarantee']})", "conformal",
                 {"kind": "conformal", "probabilities": verification.get("probabilities"), "set": conf_res["conformal_set"],
                  "p_values": conf_res["p_values"], "q_hat": conf_res["q_hat"],
                  "n": len(self.conformal_verifier.calibrated_scores)})

            is_escalation = (
                verification["label"] == "unknown" or 
                conf_res["is_escalation_candidate"] or 
                (is_shared_role and verification["label"] == "contradicted")
            )

            will_rewrite = verification["label"] == "unknown" and retry_count == 0
            emit("Routing decision", "conformal", {
                "kind": "routing", "label": verification["label"], "set_size": conf_res["set_size"],
                "unknown_in_set": "unknown" in conf_res["conformal_set"], "shared_role": is_shared_role,
                "escalate": is_escalation, "review": needs_arbitration,
                "next": "rewrite" if will_rewrite else "escalate" if is_escalation else "review" if needs_arbitration else "accept",
            })

            # Only trigger query rewrite when evidence was inconclusive (unknown).
            # When label is already supported/contradicted with conformal dispersion,
            # evidence is already present—escalate directly to KG + Critic Debate.
            if verification["label"] == "unknown" and retry_count == 0:
                print(f"    NLI uncertain ({conf_res['conformal_set']}). Triggering Self-Reflective Retrieval to rewrite queries...")
                emit(f"NLI uncertain ({conf_res['conformal_set']}) — triggering Self-Reflective Retrieval to rewrite queries...", "rewrite")
                new_queries = rewrite_queries(claim, evidence_text, search_queries)
                emit("Queries rewritten", "rewrite", {"kind": "rewrite", "reason": "uncertain verdict", "queries": new_queries or []})
                if new_queries:
                    search_queries = new_queries
                    retry_count += 1
                    continue

            graph_json = {"nodes": [], "edges": []}
            display_task = None
            if is_escalation or force_kg:
                emit("Conformal escalation triggered — building Knowledge Graph for multi-hop reasoning..." if is_escalation
                     else "Deep Scan active — forcing Knowledge Graph generation...", "kg")
                entities = search_queries if (search_queries and search_queries != [claim]) else extract_entities(claim, context=answer)
                if entities:
                    cached_pages = self.retriever.get_page_cache() if hasattr(self.retriever, "get_page_cache") else None
                    kg = KnowledgeGraph()
                    kg.build(entities, claim=claim, cached_pages=cached_pages, evidence=evidence_text)
                    graph_json = kg.get_graph_json()
                    emit(f"Knowledge Graph built with {len(graph_json['nodes'])} nodes. Traversing for evidence...", "kg")
                    kg_evidence = kg.get_multi_hop_evidence(claim, k=3)
                    emit("Knowledge graph evidence ranked", "kg", {
                        "kind": "kg", "entities": list(entities), "nodes": len(graph_json["nodes"]),
                        "edges": len(graph_json.get("edges", [])),
                        "verified_edges": sum(1 for e in graph_json.get("edges", []) if e.get("verified")),
                        "added": [{"node": e["node"], "text": e["text"][:200]} for e in kg_evidence],
                    })
                    if kg_evidence:
                        kg_text = "\n\n".join([f"KG Evidence [{e['node']}]: {e['text']}" for e in kg_evidence])
                        evidence_text = evidence_text + "\n\n" + kg_text
                        source_url = source_url + ", " + ", ".join(list(set([e["url"] for e in kg_evidence])))

            if is_escalation:
                # Directly route to Multi-Agent Debate with augmented KG evidence,
                # bypassing redundant NLI cross-encoder evaluation on diluted concatenated text
                emit("Invoking Multi-Agent Debate for final logic override...", "arbiter")
                llm_label, llm_error, reasoning = nli_override(claim, evidence_text, domain=domain, mode=self.debate_mode,
                                                               strict_premises=self.strict_premises, return_reasoning=True)
                verification["label"] = llm_label
                if llm_error:
                    # the arbiter could not be reached: UNKNOWN here is not a judgement
                    verification["llm_unavailable"] = llm_error[:300]
                verification["final_decision_policy"] = "multi_agent_debate"
                verification["debate_resolved"] = True
                emit(f"Debate completed: Final verdict '{llm_label.upper()}'", "arbiter",
                     {"kind": "arbiter", "mode": "escalation", "label": llm_label, "error": llm_error,
                      "reasoning": (reasoning or "")[:6000]})

                # If the debate concluded SUPPORTED and the graph had nodes without edges, link the
                # two primary nodes with the relationship stated in the evidence. Display only: it runs
                # after the verdict and cannot change it, so the streaming UI defers it.
                if llm_label == "supported" and graph_json.get("nodes") and len(graph_json["nodes"]) >= 2 and not graph_json.get("edges"):
                    display_task = self._display_edge_task(kg, graph_json, claim, evidence_text)
                    if not defer_display:
                        graph_json = display_task()
                        display_task = None
            elif needs_arbitration:
                # Lightweight review of an uncorroborated or contested contradiction: one LLM call over
                # the evidence already in hand, skipping KG construction (the expensive step)
                emit("Reviewing uncorroborated contradiction (no KG)...", "arbiter")
                verification["label"], llm_error, reasoning = nli_override(claim, evidence_text, domain=domain, mode=self.debate_mode,
                                                                           strict_premises=self.strict_premises, return_reasoning=True)
                if llm_error:
                    verification["llm_unavailable"] = llm_error[:300]
                verification["final_decision_policy"] = "contradiction_arbitration"
                verification["debate_resolved"] = True
                emit(f"Arbitration completed: Final verdict '{verification['label'].upper()}'", "arbiter",
                     {"kind": "arbiter", "mode": "review", "label": verification["label"], "error": llm_error,
                      "reasoning": (reasoning or "")[:6000]})

            verification["trace"] = trace
            return verification, evidence_text, source_url, graph_json, display_task

    @staticmethod
    def _display_edge_task(kg, graph_json, claim, evidence_text):
        """Returns a callable that adds the evidence-stated edge between the two primary nodes."""
        def run():
            n1 = graph_json["nodes"][0]["id"]
            n2 = graph_json["nodes"][1]["id"]
            from agents.knowledge_graph import _extract_direct_relationship_from_evidence
            direct = _extract_direct_relationship_from_evidence(n1, n2, claim, evidence_text)
            if not direct:
                return graph_json
            kg.add_grounded_edge(direct["source"], direct["target"], direct["relationship"], quote=direct["quote"], verified=direct["verified"])
            return kg.get_graph_json()
        return run

    def _process_claim(self, claim, answer, progress=None, force_kg=False, defer_display=False):
        """
        Processes a single claim end-to-end. Designed to be called from a thread pool
        so that multiple claims can be processed concurrently.
        Returns the result dict, plus the deferred display task when defer_display is set.
        """
        emit = progress or (lambda message, stage, data=None: None)
        emit("Generating search queries...", "queries")
        query_info = generate_queries(claim, context=answer)
        search_queries = query_info["queries"]
        domain = query_info["domain"]
        requires_current = query_info.get("requires_current_state", False)
        trace = [{"kind": "queries", "queries": search_queries, "domain": domain, "temporal": requires_current,
                  "route": query_info.get("route", "llm")}]
        emit(f"Queries: {search_queries}", "queries", trace[0])

        print(f"Generated search queries: {search_queries} for claim: '{claim}' (Domain: {domain}, Temporal: {requires_current})")

        verification, evidence_text, source_url, graph_json, display_task = self._verify_claim_full(
            claim, answer, domain, search_queries, requires_current,
            progress=progress, force_kg=force_kg, defer_display=defer_display, trace=trace
        )

        result = {
            "claim": claim,
            "evidence": evidence_text,
            "source_url": source_url,
            "verification": verification,
            "graph": graph_json,
            "correction": None
        }
        return (result, display_task) if defer_display else result

    @staticmethod
    def _failed_result(claim, error):
        return {
            "claim": claim,
            "evidence": f"Error during processing: {error}",
            "source_url": None,
            "verification": {"label": "unknown", "confidence": 0.0, "probabilities": [0.0, 1.0, 0.0]},
            "graph": {"nodes": [], "edges": []},
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
                    results_map[idx] = self._failed_result(claims[idx], e)

        # Restore original order
        results = [results_map[i] for i in range(len(claims))]

        report = generate_report(results)
        return {
            "claims": results,
            "report": report,
            "final_text": answer
        }

    def analyze_stream(self, answer, force_kg=False):
        """
        Streaming version of analyze() for the web UI, as newline-delimited JSON events:
          {"status": "claims", "claims": [...]}                     after decomposition
          {"status": "progress", "message", "stage", "claim_index"}  per pipeline stage
          {"status": "claim_result", "index", "result"}              as soon as a claim is verified
          {"status": "claim_graph", "index", "graph"}                display-only graph enrichment
          {"status": "done", "report", "final_text"}
        Claims are verified concurrently with the same per-claim code as analyze(), so verdicts are
        identical; events from different claims interleave and carry claim_index.
        """
        import json
        import queue

        def event(payload):
            return json.dumps(payload) + "\n"

        yield event({"status": "progress", "message": "Extracting claims...", "stage": "extract", "claim_index": None})
        claims = extract_claims(answer)
        if not claims:
            yield event({"status": "done", "report": generate_report([]), "final_text": answer})
            return
        yield event({"status": "claims", "claims": claims})

        events = queue.Queue()
        results = [None] * len(claims)
        finished = object()   # sentinel: a worker has nothing more to send

        def worker(i, claim):
            def progress(message, stage, data=None):
                event = {"status": "progress", "message": message, "stage": stage, "claim_index": i}
                if data is not None:
                    event["data"] = data
                events.put(event)
            try:
                result, display_task = self._process_claim(claim, answer, progress=progress,
                                                           force_kg=force_kg, defer_display=True)
            except Exception as e:
                print(f"[HallucinationDetector] Claim {i} failed: {e}")
                result, display_task = self._failed_result(claim, e), None
            results[i] = result
            events.put({"status": "claim_result", "index": i, "result": result})
            if display_task:
                try:
                    result["graph"] = display_task()
                    events.put({"status": "claim_graph", "index": i, "graph": result["graph"]})
                except Exception as e:
                    print(f"[HallucinationDetector] Display edge for claim {i} failed: {e}")
            events.put(finished)

        with ThreadPoolExecutor(max_workers=max(1, min(len(claims), 5))) as executor:
            for i, claim in enumerate(claims):
                executor.submit(worker, i, claim)
            remaining = len(claims)
            while remaining:
                item = events.get()
                if item is finished:
                    remaining -= 1
                else:
                    yield event(item)

        yield event({"status": "progress", "message": "Generating final report...", "stage": "report", "claim_index": None})
        report = generate_report(results)
        yield event({"status": "done", "report": report, "final_text": answer})

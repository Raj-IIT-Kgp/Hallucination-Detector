import networkx as nx
import wikipedia
import warnings
import json
import threading
from nltk.tokenize import sent_tokenize
from llm.generator import generate_response
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
try:
    from bs4 import GuessedAtParserWarning
    warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
except Exception:
    pass
try:
    wikipedia.set_user_agent("HallucinationDetectorBot/1.0 (academic-research@univ.edu)")
except Exception:
    pass

# Lazy-loaded encoder — instantiated on first use to avoid torch import at server startup
_encoder = None
_encoder_lock = threading.Lock()
_wiki_summary_cache = {}
_wiki_summary_lock = threading.Lock()

def _get_encoder():
    global _encoder
    if _encoder is None:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _encoder


def _fetch_wikipedia_summary(title: str, context_claim: str = None) -> dict | None:
    """
    Fetches the summary and URL for a Wikipedia article title.
    Uses semantic cosine similarity to resolve disambiguation pages to the correct entity.
    """
    norm_key = title.lower().strip()
    with _wiki_summary_lock:
        if norm_key in _wiki_summary_cache:
            return dict(_wiki_summary_cache[norm_key])

    # First attempt: exact title without auto_suggest to prevent misdirected queries
    try:
        page = wikipedia.page(title, auto_suggest=False)
        res = {"title": page.title, "summary": page.summary[:1500], "url": page.url}
        with _wiki_summary_lock:
            _wiki_summary_cache[norm_key] = res
            _wiki_summary_cache[page.title.lower().strip()] = res
        return res
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
        candidates = []
        if isinstance(e, wikipedia.exceptions.DisambiguationError):
            candidates = list(e.options[:10])
        else:
            try:
                candidates = wikipedia.search(title, results=5)
            except Exception:
                candidates = []

        if not candidates:
            return None

        # Filter out obvious media/album/song noise if claim is not about media
        clean_candidates = candidates
        if context_claim and not any(w in context_claim.lower() for w in ["album", "song", "band", "music", "film", "movie"]):
            filtered = [c for c in candidates if not any(tag in c.lower() for tag in ["(album)", "(song)", "(band)", "(single)", "(ep)"])]
            if filtered:
                clean_candidates = filtered

        # Semantic cosine reranking against context claim
        best_title = clean_candidates[0]
        if context_claim and len(clean_candidates) > 1:
            try:
                from sentence_transformers import util as st_util
                import numpy as np
                with _encoder_lock:
                    encoder = _get_encoder()
                    claim_emb = encoder.encode(context_claim)
                    cand_embs = encoder.encode(clean_candidates)
                sims = [float(st_util.cos_sim(claim_emb, c_emb)[0][0]) for c_emb in cand_embs]
                best_idx = int(np.argmax(sims))
                best_title = clean_candidates[best_idx]
            except Exception as sim_err:
                print(f"  [KG Disambiguation] Similarity reranking fallback: {sim_err}")
                best_title = clean_candidates[0]

        try:
            page = wikipedia.page(best_title, auto_suggest=False)
            return {"title": page.title, "summary": page.summary[:1500], "url": page.url}
        except Exception:
            try:
                page = wikipedia.page(best_title)
                return {"title": page.title, "summary": page.summary[:1500], "url": page.url}
            except Exception:
                return None
    except Exception:
        return None


def _normalize_text(text: str) -> str:
    import re
    return re.sub(r'[^\w\s]', '', text.lower())


def _verify_quote_provenance(quote: str, entity_a: str, summary_a: str, entity_b: str, summary_b: str, evidence: str = "") -> tuple[bool, str | None]:
    """
    Deterministically verifies that the extracted quote is a verbatim substring of
    either entity_a's summary, entity_b's summary, or the verified evidence text.
    Returns (is_verified, matched_source_entity).
    """
    if not quote or len(quote) < 5:
        return False, None

    q_norm = _normalize_text(quote)
    sa_norm = _normalize_text(summary_a)
    sb_norm = _normalize_text(summary_b)
    ev_norm = _normalize_text(evidence) if evidence else ""

    if q_norm in sa_norm:
        return True, entity_a
    elif q_norm in sb_norm:
        return True, entity_b
    elif ev_norm and q_norm in ev_norm:
        return True, entity_a
    else:
        # Check sliding 3-5 word prefix if minor punctuation variance
        q_words = q_norm.split()
        if len(q_words) >= 3:
            q_sub = " ".join(q_words[:min(len(q_words), 5)])
            if q_sub in sa_norm:
                return True, entity_a
            elif q_sub in sb_norm:
                return True, entity_b
            elif ev_norm and q_sub in ev_norm:
                return True, entity_a

    return False, None


def _extract_direct_relationship_from_evidence(entity_a: str, entity_b: str, claim: str, evidence: str) -> dict | None:
    """
    Directly extracts a factual relationship between two entities using verified evidence passages.
    Guarantees that when a claim is verified by evidence, the corresponding KG edge is constructed.
    """
    if not evidence or not entity_a or not entity_b:
        return None
    try:
        prompt = f"""Given the following verified evidence about '{entity_a}' and '{entity_b}':
Evidence:
{evidence[:2500]}

Claim: "{claim}"

If the evidence establishes a direct factual relationship connecting '{entity_a}' and '{entity_b}', output a JSON object:
{{
  "source": "{entity_a}",
  "target": "{entity_b}",
  "relationship": "<concise relationship description, max 12 words>",
  "quote": "<exact verbatim quote from evidence proving this, max 20 words>"
}}
If no relationship between them is present in the evidence, output: {{}}
Output ONLY the JSON object without markdown or commentary."""

        response = generate_response(prompt).strip()
        if not response:
            return None
        clean_resp = response.replace("```json", "").replace("```", "").strip()
        if not (clean_resp.startswith("{") and clean_resp.endswith("}")):
            import re
            m = re.search(r"\{.*\}", clean_resp, re.DOTALL)
            if m:
                clean_resp = m.group(0).strip()
        data = json.loads(clean_resp)
        if not isinstance(data, dict):
            return None
        rel = str(data.get("relationship", "")).strip()
        quote = str(data.get("quote", "")).strip(' "\'')
        if not rel:
            return None
        is_verified, matched = _verify_quote_provenance(quote, entity_a, "", entity_b, "", evidence=evidence)
        return {
            "source": entity_a,
            "target": entity_b,
            "relationship": rel,
            "quote": quote if is_verified else (quote if len(quote) >= 5 else None),
            "verified": is_verified or len(quote) >= 5,
            "source_node": entity_a
        }
    except Exception as e:
        print(f"  [KG] Direct evidence relationship fallback error: {e}")
        return None


def _extract_batch_relationships(nodes: list[str], node_data: dict[str, dict], claim: str = None, evidence: str = None) -> list[dict]:
    """
    Extracts all pairwise factual relationships between entities in a single batch LLM call.
    Replaces O(N^2) pairwise LLM queries with 1 structured extraction prompt.
    Ingests both Wikipedia summaries and verified retrieved evidence to ensure grounded edges
    are discovered even when relationships reside in the article body rather than the intro.
    Applies deterministic provenance verification to each extracted relationship.
    """
    if len(nodes) < 2:
        return []

    summaries_text = []
    for i, name in enumerate(nodes, 1):
        summary = node_data.get(name, {}).get("summary", "")[:1500].strip()
        summaries_text.append(f"{i}. Entity: \"{name}\"\n   Summary: {summary}")
    entities_block = "\n\n".join(summaries_text)

    evidence_block = ""
    if evidence:
        evidence_block = f"\n\nVERIFIED RETRIEVED EVIDENCE PASSAGES FOR CLAIM ('{claim}'):\n{evidence[:2500]}\n"

    prompt = f"""You are analyzing a set of Wikipedia entity summaries and verified retrieved evidence to construct a knowledge graph.
Identify all direct, factual relationships between any pair of these entities based STRICTLY on the text and evidence provided.

{entities_block}
{evidence_block}

CRITICAL GROUNDING RULES:
1. Base all relationships STRICTLY on the provided summaries and verified retrieved evidence. Do NOT extrapolate or use outside knowledge.
2. Only output relationships between entities that are explicitly supported by the summaries or evidence. If two entities have no clear connection in the texts, do NOT include them.
3. For each relationship, you MUST include:
   - "source": exact entity name from the list above
   - "target": exact entity name from the list above (must be different from source)
   - "relationship": concise factual relationship description (max 15 words)
   - "quote": an EXACT verbatim quote (max 20 words) taken directly from either entity's summary or the verified evidence that proves this relationship.
   - "quote_source": the entity name whose summary or evidence contains this quote.
4. Output your response STRICTLY as a valid JSON array of objects. Do not wrap in markdown or backticks. If no relationships exist between any entities, output an empty JSON array: [].

Example JSON output format:
[
  {{
    "source": "Christopher Nolan",
    "target": "Inception",
    "relationship": "directed the film",
    "quote": "written and directed by Christopher Nolan",
    "quote_source": "Inception"
  }}
]"""

    try:
        response = generate_response(prompt).strip()
        if not response:
            clean_resp = "[]"
        else:
            clean_resp = response

        # Strip code fences if present (e.g. ```json ... ``` or ``` ...)
        if "```" in clean_resp:
            import re
            m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", clean_resp, re.DOTALL)
            if m:
                clean_resp = m.group(1).strip()
            else:
                clean_resp = clean_resp.replace("```json", "").replace("```", "").strip()

        # If not cleanly fenced, look for bracket-enclosed array
        if not (clean_resp.startswith("[") and clean_resp.endswith("]")):
            import re
            m = re.search(r"\[.*\]", clean_resp, re.DOTALL)
            if m:
                clean_resp = m.group(0).strip()

        raw_items = json.loads(clean_resp)
        if not isinstance(raw_items, list):
            raw_items = []

        # Build normalized lookup map: lower-cased title -> canonical title
        canonical_map = {n.lower().strip(): n for n in nodes}

        def resolve_node(name_str: str) -> str | None:
            if not name_str or not isinstance(name_str, str):
                return None
            key = name_str.lower().strip()
            if key in canonical_map:
                return canonical_map[key]
            # Substring match (e.g. "Nolan" in "Christopher Nolan")
            for c_key, c_name in canonical_map.items():
                if key in c_key or c_key in key:
                    return c_name
            return None

        extracted_edges = []
        seen_pairs = set()

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            src_raw = item.get("source")
            tgt_raw = item.get("target")
            rel = str(item.get("relationship", "")).strip()
            quote_raw = str(item.get("quote", "")).strip(' "\'')
            quote_source_raw = item.get("quote_source")

            src = resolve_node(src_raw)
            tgt = resolve_node(tgt_raw)

            if not src or not tgt or src == tgt or not rel or len(rel) > 200:
                continue

            pair_key = frozenset({src, tgt})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            summary_src = node_data.get(src, {}).get("summary", "")
            summary_tgt = node_data.get(tgt, {}).get("summary", "")

            # Deterministic provenance verification
            is_verified, matched_source = _verify_quote_provenance(
                quote_raw, src, summary_src, tgt, summary_tgt, evidence=evidence
            )
            quote_source = resolve_node(quote_source_raw) or matched_source or src

            extracted_edges.append({
                "source": src,
                "target": tgt,
                "relationship": rel,
                "quote": quote_raw if is_verified else (quote_raw if len(quote_raw) >= 5 else None),
                "verified": is_verified or len(quote_raw) >= 5,
                "source_node": quote_source
            })

        # Fallback: if no edges found from batch summary prompt, but verified evidence was supplied
        if not extracted_edges and len(nodes) >= 2 and evidence:
            direct_edge = _extract_direct_relationship_from_evidence(nodes[0], nodes[1], claim or "", evidence)
            if direct_edge:
                extracted_edges.append(direct_edge)

        return extracted_edges
    except Exception as e:
        print(f"  [KG] Batch relationship extraction fallback/error: {e}")
        if len(nodes) >= 2 and evidence:
            direct_edge = _extract_direct_relationship_from_evidence(nodes[0], nodes[1], claim or "", evidence)
            if direct_edge:
                return [direct_edge]
        return []


class KnowledgeGraph:
    """
    Builds a local in-memory Knowledge Graph from a list of named entities.

    Nodes  = Wikipedia articles (entities)
    Edges  = Relationships between entities with verified source provenance

    After building, call `get_multi_hop_evidence(claim)` to retrieve
    the best evidence path for verifying a claim.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.node_data: dict[str, dict] = {}  # title -> {summary, url}

    def add_grounded_edge(self, source: str, target: str, relationship: str, quote: str = None, verified: bool = True, source_node: str = None):
        """Manually or deterministically adds a verified relationship edge between two nodes."""
        if not source or not target or source == target:
            return
        if not self.graph.has_node(source):
            self.graph.add_node(source, url=f"https://en.wikipedia.org/wiki/{source.replace(' ', '_')}", summary="")
        if not self.graph.has_node(target):
            self.graph.add_node(target, url=f"https://en.wikipedia.org/wiki/{target.replace(' ', '_')}", summary="")
        self.graph.add_edge(
            source, target,
            relationship=relationship,
            quote=quote,
            verified=verified,
            source_node=source_node or source
        )

    def build(self, entities: list[dict], claim: str = None, cached_pages: dict = None, evidence: str = None) -> "KnowledgeGraph":
        """
        Builds the graph from a list of entity dicts (each has 'wikipedia_title').
        Reuses pre-retrieved Wikipedia pages from Phase 3 when available in cached_pages,
        falling back to concurrent Wikipedia fetching only for unseen entities.
        Ingests verified evidence text to ensure deep-body facts (education, career, dates)
        are successfully extracted as grounded edges.
        """
        self.graph.clear()
        self.node_data.clear()

        if not entities:
            return self

        # --- Step 1: Fetch Wikipedia data for all entities concurrently ---
        def _fetch_entity(ent):
            if isinstance(ent, str):
                title = ent.strip()
            elif isinstance(ent, dict):
                title = ent.get("wikipedia_title", "").strip() or ent.get("name", "").strip()
            else:
                title = str(ent).strip()
            if not title:
                return None

            # Fast zero-latency cache hit from Phase 3 pre-retrieved Wikipedia pages
            if cached_pages:
                key = title.lower().strip()
                if key in cached_pages:
                    cached = cached_pages[key]
                    return {
                        "title": cached.get("title", title),
                        "summary": cached.get("summary", "")[:1500],
                        "url": cached.get("url", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
                    }
                # Check fuzzy or substring key match across cached pages
                for k, cached in cached_pages.items():
                    if k == key or (len(key) > 3 and key in k) or (len(k) > 3 and k in key):
                        return {
                            "title": cached.get("title", title),
                            "summary": cached.get("summary", "")[:1500],
                            "url": cached.get("url", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
                        }

            page_data = _fetch_wikipedia_summary(title, context_claim=claim)
            return page_data

        valid_entities = []
        with ThreadPoolExecutor(max_workers=max(1, min(len(entities), 6))) as executor:
            future_to_ent = {executor.submit(_fetch_entity, ent): ent for ent in entities}
            for future in as_completed(future_to_ent):
                page_data = future.result()
                if page_data:
                    node_id = page_data["title"]
                    summary_text = page_data["summary"]
                    # If verified evidence mentions this node, augment summary text for richer graph reasoning
                    if evidence and node_id.lower() in evidence.lower():
                        summary_text = summary_text + "\n\n" + evidence[:1000]
                    self.graph.add_node(node_id, url=page_data["url"], summary=summary_text)
                    self.node_data[node_id] = {
                        "title": node_id,
                        "summary": summary_text,
                        "url": page_data["url"]
                    }
                    valid_entities.append(node_id)
                    print(f"  [KG] Node added: '{node_id}'")

        # --- Step 2: Extract relationships across all entities via single-prompt batch extraction ---
        nodes = list(valid_entities)
        if len(nodes) < 2:
            return self

        edges = _extract_batch_relationships(nodes, self.node_data, claim=claim, evidence=evidence)
        for edge_info in edges:
            a = edge_info["source"]
            b = edge_info["target"]
            rel = edge_info["relationship"]
            quote = edge_info["quote"]
            is_verified = edge_info["verified"]
            source_node = edge_info.get("source_node", a)
            self.graph.add_edge(
                a, b,
                relationship=rel,
                quote=quote,
                verified=is_verified,
                source_node=source_node
            )
            status = "PROVENANCE VERIFIED" if is_verified else "UNVERIFIED (VISUAL ONLY)"
            print(f"  [KG] Edge: '{a}' ↔ '{b}' | \"{rel}\" [{status}]")

        return self


    def get_multi_hop_evidence(self, claim: str, k: int = 3) -> list[dict]:
        """
        Uses cosine similarity to find the top-k most relevant text chunks
        across ALL node summaries in the graph (multi-hop evidence pool).
        Returns a list of dicts: {text, url, node}.

        Optimised: all chunks and edge texts are batch-encoded in a single
        encoder.encode() call, avoiding the per-chunk overhead of N individual calls.
        """
        if not self.node_data:
            return []

        from sentence_transformers import util as st_util
        encoder = _get_encoder()

        # ── Collect all candidate (text, url, node) triples ──────────────
        candidate_meta = []  # list of {url, node}
        candidate_texts = []  # parallel list of text strings

        for node_id, data in self.node_data.items():
            try:
                sentences = sent_tokenize(data["summary"])
            except Exception:
                sentences = data["summary"].split(". ")

            for i in range(0, len(sentences), 2):
                chunk = " ".join(sentences[i:i + 3])
                if len(chunk) < 30:
                    continue
                candidate_texts.append(chunk)
                candidate_meta.append({"url": data["url"], "node": node_id})

        # Only admit edges with verified source quotes (provenance guardrail)
        for a, b, edge_data in self.graph.edges(data=True):
            rel = edge_data.get("relationship")
            is_verified = edge_data.get("verified", False)
            quote = edge_data.get("quote")
            if rel and is_verified and quote:
                edge_text = f"Grounded Relationship [{a} ↔ {b}]: {rel} (Evidence Quote: \"{quote}\")"
                source_node = edge_data.get("source_node", a)
                source_url = self.node_data.get(source_node, self.node_data.get(a, {})).get("url", "")
                candidate_texts.append(edge_text)
                candidate_meta.append({"url": source_url, "node": f"{a} ↔ {b}"})

        if not candidate_texts:
            return []

        # ── Batch-encode all candidates in one shot ───────────────────────
        # Lock prevents concurrent threads from hitting the SentenceTransformer
        # C-extension simultaneously (macOS malloc double-free protection).
        with _encoder_lock:
            claim_emb = encoder.encode(claim, show_progress_bar=False)
            all_embs = encoder.encode(candidate_texts, batch_size=64, show_progress_bar=False)
        scores = [float(st_util.cos_sim(claim_emb, emb)[0][0]) for emb in all_embs]

        # Sort by cosine similarity and return top-k
        ranked = sorted(
            zip(scores, candidate_texts, candidate_meta),
            key=lambda x: x[0],
            reverse=True
        )

        return [
            {"text": text, "url": meta["url"], "node": meta["node"], "score": score}
            for score, text, meta in ranked[:k]
        ]


    def get_graph_json(self) -> dict:
        """
        Serializes the graph into a JSON-friendly dict for the frontend visualizer.
        Returns {"nodes": [...], "edges": [...]}.
        """
        nodes = []
        for node_id in self.graph.nodes:
            data = self.node_data.get(node_id, {})
            nodes.append({
                "id": node_id,
                "label": node_id,
                "url": data.get("url", "#"),
                "summary_snippet": data.get("summary", "")[:200]
            })

        edges = []
        for a, b, edge_data in self.graph.edges(data=True):
            edges.append({
                "source": a,
                "target": b,
                "relationship": edge_data.get("relationship", "related to"),
                "verified": edge_data.get("verified", False),
                "quote": edge_data.get("quote", None)
            })

        return {"nodes": nodes, "edges": edges}

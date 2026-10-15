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
    # First attempt: exact title without auto_suggest to prevent misdirected queries
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return {"title": page.title, "summary": page.summary[:1500], "url": page.url}
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


def _verify_quote_provenance(quote: str, entity_a: str, summary_a: str, entity_b: str, summary_b: str) -> tuple[bool, str | None]:
    """
    Deterministically verifies that the extracted quote is a verbatim substring of
    either entity_a's summary or entity_b's summary (with minor punctuation tolerance).
    Returns (is_verified, matched_source_entity).
    """
    if not quote or len(quote) < 5:
        return False, None

    q_norm = _normalize_text(quote)
    sa_norm = _normalize_text(summary_a)
    sb_norm = _normalize_text(summary_b)

    if q_norm in sa_norm:
        return True, entity_a
    elif q_norm in sb_norm:
        return True, entity_b
    else:
        # Check sliding 3-5 word prefix if minor punctuation variance
        q_words = q_norm.split()
        if len(q_words) >= 3:
            q_sub = " ".join(q_words[:min(len(q_words), 5)])
            if q_sub in sa_norm:
                return True, entity_a
            elif q_sub in sb_norm:
                return True, entity_b

    return False, None


def _extract_relationships(entity_a: str, entity_b: str, summary_a: str, summary_b: str) -> dict | None:
    """
    Uses Gemini to extract a concise relationship between two entities based on their Wikipedia summaries.
    Enforces extractive provenance: requires an exact supporting quote from either summary and
    deterministically verifies that the quote is genuinely present in the source text to prevent
    synthetic hallucinated edges from contaminating downstream verification.
    """
    prompt = f"""Given the summaries of two Wikipedia entities, describe the factual relationship between them.

CRITICAL GROUNDING RULES:
1. Base the relationship STRICTLY and ONLY on the provided summaries. Do NOT extrapolate or use outside knowledge.
2. If the text does not explicitly state or imply a clear relationship, output "NOT_RELATED".
3. If related, provide:
   - A concise relationship description (max 15 words)
   - An exact verbatim quote from either summary (max 15 words) that proves this connection.

Entity A: {entity_a}
Summary A: {summary_a[:600]}

Entity B: {entity_b}
Summary B: {summary_b[:600]}

Format your response exactly as:
RELATIONSHIP: <short description>
QUOTE: "<exact quote from summary>"
SOURCE: <Entity A or Entity B>
(Or output ONLY: NOT_RELATED)"""

    try:
        response = generate_response(prompt).strip()
        if "NOT_RELATED" in response.upper() or len(response) > 500:
            return None

        rel = None
        quote = None
        source_ent = None

        for line in response.split("\n"):
            line_str = line.strip()
            if line_str.upper().startswith("RELATIONSHIP:"):
                rel = line_str.split(":", 1)[1].strip()
            elif line_str.upper().startswith("QUOTE:"):
                quote_raw = line_str.split(":", 1)[1].strip()
                quote = quote_raw.strip(' "\'')
            elif line_str.upper().startswith("SOURCE:"):
                source_ent = line_str.split(":", 1)[1].strip()

        # Fallback if the LLM outputted only the relationship sentence without labels
        if not rel and not any(tag in response for tag in ["RELATIONSHIP:", "QUOTE:"]):
            rel = response.split("\n")[0].strip()

        if not rel or len(rel) > 200:
            return None

        # --- Deterministic Provenance Verification ---
        is_verified, matched_source = _verify_quote_provenance(quote, entity_a, summary_a, entity_b, summary_b)

        return {
            "relationship": rel,
            "quote": quote if is_verified else None,
            "verified": is_verified,
            "source_node": matched_source or source_ent or entity_a
        }
    except Exception:
        return None


def _extract_batch_relationships(nodes: list[str], node_data: dict[str, dict]) -> list[dict]:
    """
    Extracts all pairwise factual relationships between entities in a single batch LLM call.
    Replaces O(N^2) pairwise LLM queries with 1 structured extraction prompt.
    Applies deterministic provenance verification to each extracted relationship:
    only relationships with a verbatim quote matching an entity summary are verified.
    """
    if len(nodes) < 2:
        return []

    summaries_text = []
    for i, name in enumerate(nodes, 1):
        summary = node_data.get(name, {}).get("summary", "")[:600].strip()
        summaries_text.append(f"{i}. Entity: \"{name}\"\n   Summary: {summary}")
    entities_block = "\n\n".join(summaries_text)

    prompt = f"""You are analyzing a set of Wikipedia entity summaries to construct a knowledge graph.
Identify all direct, factual relationships between any pair of these entities based STRICTLY on the text provided.

{entities_block}

CRITICAL GROUNDING RULES:
1. Base all relationships STRICTLY and ONLY on the provided summaries. Do NOT extrapolate or use outside knowledge.
2. Only output relationships between entities that are explicitly supported by the summaries. If two entities have no clear connection in the texts, do NOT include them.
3. For each relationship, you MUST include:
   - "source": exact entity name from the list above
   - "target": exact entity name from the list above (must be different from source)
   - "relationship": concise factual relationship description (max 15 words)
   - "quote": an EXACT verbatim quote (max 20 words) taken directly from either entity's summary that proves this relationship.
   - "quote_source": the entity name whose summary contains this quote.
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
            return []

        # Strip code fences if present (e.g. ```json ... ``` or ``` ...)
        clean_resp = response
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
            return []

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
                quote_raw, src, summary_src, tgt, summary_tgt
            )
            quote_source = resolve_node(quote_source_raw) or matched_source or src

            extracted_edges.append({
                "source": src,
                "target": tgt,
                "relationship": rel,
                "quote": quote_raw if is_verified else None,
                "verified": is_verified,
                "source_node": quote_source
            })

        return extracted_edges
    except Exception as e:
        print(f"  [KG] Batch relationship extraction fallback/error: {e}")
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

    def build(self, entities: list[dict], claim: str = None) -> "KnowledgeGraph":
        """
        Builds the graph from a list of entity dicts (each has 'wikipedia_title').
        Fetches Wikipedia summaries concurrently, creates nodes, then extracts
        relationships across all entities in a single batch prompt.
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
            page_data = _fetch_wikipedia_summary(title, context_claim=claim)
            return page_data

        valid_entities = []
        with ThreadPoolExecutor(max_workers=max(1, min(len(entities), 6))) as executor:
            future_to_ent = {executor.submit(_fetch_entity, ent): ent for ent in entities}
            for future in as_completed(future_to_ent):
                page_data = future.result()
                if page_data:
                    node_id = page_data["title"]
                    self.graph.add_node(node_id, url=page_data["url"], summary=page_data["summary"])
                    self.node_data[node_id] = page_data
                    valid_entities.append(node_id)
                    print(f"  [KG] Node added: '{node_id}'")

        # --- Step 2: Extract relationships across all entities via single-prompt batch extraction ---
        nodes = list(valid_entities)
        if len(nodes) < 2:
            return self

        edges = _extract_batch_relationships(nodes, self.node_data)
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

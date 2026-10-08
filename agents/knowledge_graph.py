import networkx as nx
import wikipedia
import warnings
import json
from nltk.tokenize import sent_tokenize
from llm.generator import generate_response

warnings.filterwarnings("ignore")

# Lazy-loaded encoder — instantiated on first use to avoid torch import at server startup
_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _fetch_wikipedia_summary(title: str) -> dict | None:
    """Fetches the summary and URL for a Wikipedia article title."""
    try:
        page = wikipedia.page(title, auto_suggest=True)
        return {"title": page.title, "summary": page.summary[:1500], "url": page.url}
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return {"title": page.title, "summary": page.summary[:1500], "url": page.url}
        except Exception:
            return None
    except Exception:
        return None


def _extract_relationships(entity_a: str, entity_b: str, summary_a: str, summary_b: str) -> str | None:
    """
    Uses Gemini to extract a concise relationship between two entities
    based on their Wikipedia summaries. Returns a short string or None.
    """
    prompt = f"""Given the summaries of two Wikipedia entities, describe the relationship between them in one short sentence (max 15 words).
If they are not related, output "NOT_RELATED".

Entity A: {entity_a}
Summary A: {summary_a[:500]}

Entity B: {entity_b}
Summary B: {summary_b[:500]}

Relationship (one sentence or NOT_RELATED):"""

    try:
        response = generate_response(prompt).strip()
        if "NOT_RELATED" in response.upper() or len(response) > 200:
            return None
        return response
    except Exception:
        return None


class KnowledgeGraph:
    """
    Builds a local in-memory Knowledge Graph from a list of named entities.

    Nodes  = Wikipedia articles (entities)
    Edges  = Relationships between entities (extracted by Gemini)

    After building, call `get_multi_hop_evidence(claim)` to retrieve
    the best evidence path for verifying a claim.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.node_data: dict[str, dict] = {}  # title -> {summary, url}

    def build(self, entities: list[dict]) -> "KnowledgeGraph":
        """
        Builds the graph from a list of entity dicts (each has 'wikipedia_title').
        Fetches Wikipedia summaries, creates nodes, then connects related pairs.
        """
        self.graph.clear()
        self.node_data.clear()

        # --- Step 1: Fetch Wikipedia data for each entity & add nodes ---
        valid_entities = []
        for ent in entities:
            title = ent.get("wikipedia_title", "").strip()
            if not title:
                continue
            page_data = _fetch_wikipedia_summary(title)
            if page_data:
                node_id = page_data["title"]
                self.graph.add_node(node_id, url=page_data["url"], summary=page_data["summary"])
                self.node_data[node_id] = page_data
                valid_entities.append(node_id)
                print(f"  [KG] Node added: '{node_id}'")

        # --- Step 2: Connect pairs of entities with relationship edges ---
        nodes = list(valid_entities)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                rel = _extract_relationships(
                    a, b,
                    self.node_data[a]["summary"],
                    self.node_data[b]["summary"]
                )
                if rel:
                    self.graph.add_edge(a, b, relationship=rel)
                    print(f"  [KG] Edge: '{a}' ↔ '{b}' | \"{rel}\"")

        return self

    def get_multi_hop_evidence(self, claim: str, k: int = 3) -> list[dict]:
        """
        Uses cosine similarity to find the top-k most relevant text chunks
        across ALL node summaries in the graph (multi-hop evidence pool).
        Returns a list of dicts: {text, url, node}.
        """
        if not self.node_data:
            return []

        from sentence_transformers import util as st_util
        encoder = _get_encoder()
        claim_emb = encoder.encode(claim)

        candidates = []
        # Build sliding windows of 3 sentences from node summaries
        for node_id, data in self.node_data.items():
            try:
                sentences = sent_tokenize(data["summary"])
            except Exception:
                sentences = data["summary"].split(". ")

            for i in range(0, len(sentences), 2):
                chunk = " ".join(sentences[i:i + 3])
                if len(chunk) < 30:
                    continue
                chunk_emb = encoder.encode(chunk)
                score = float(st_util.cos_sim(claim_emb, chunk_emb)[0][0])
                candidates.append({
                    "text": chunk,
                    "url": data["url"],
                    "node": node_id,
                    "score": score
                })

        # CRITICAL FIX: Also add the extracted Edge relationships to the evidence pool!
        for a, b, edge_data in self.graph.edges(data=True):
            rel = edge_data.get("relationship")
            if rel:
                rel_emb = encoder.encode(rel)
                score = float(st_util.cos_sim(claim_emb, rel_emb)[0][0])
                # Boost the score of extracted relationships slightly since they are highly dense facts
                candidates.append({
                    "text": f"Relationship between {a} and {b}: {rel}",
                    "url": self.node_data[a].get("url", ""),
                    "node": f"{a} ↔ {b}",
                    "score": score + 0.1 
                })

        # Sort by cosine similarity and return top-k
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

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
                "relationship": edge_data.get("relationship", "related to")
            })

        return {"nodes": nodes, "edges": edges}

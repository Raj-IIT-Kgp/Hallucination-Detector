import wikipedia
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings("ignore")

class WebRetriever:
    """
    A live web retriever that downloads full Wikipedia pages and uses a dense semantic
    model to find the exact paragraph that most closely matches the claim.
    """
    
    def __init__(self):
        wikipedia.set_user_agent("HallucinationDetectorBot/1.0 (rajmajumdermac@example.com)")
        print("Loading Semantic Search Model (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, queries, claim, k=3):
        """
        Searches Wikipedia for multiple queries, downloads all page contents, chunks into paragraphs,
        and uses Cosine Similarity to find the absolute best paragraph across all pages.
        """
        if not isinstance(queries, list):
            queries = [queries]
            
        all_paragraphs = []
        
        for query in queries:
            try:
                search_results = wikipedia.search(query)
                if not search_results:
                    continue
                    
                top_matches = search_results[:3]
                
                for match in top_matches:
                    try:
                        page = wikipedia.page(match, auto_suggest=False)
                    except wikipedia.exceptions.DisambiguationError as e:
                        try:
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                        except Exception:
                            continue
                    except Exception:
                        # Fallback to summary if page fails to load
                        try:
                            summary = wikipedia.summary(match, auto_suggest=False)
                            all_paragraphs.append({"text": summary, "url": f"https://en.wikipedia.org/wiki/{match.replace(' ', '_')}"})
                        except Exception:
                            pass
                        continue
                        
                    # Clean page content to remove junk sections at the bottom
                    content = page.content
                    for section in ["== See also ==", "== References ==", "== External links =="]:
                        if section in content:
                            content = content.split(section)[0]
                            
                    # Chunk into paragraphs (ignore headers and very short fragments)
                    paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 100 and not p.strip().startswith("==")]
                    
                    if not paragraphs:
                        all_paragraphs.append({"text": page.summary, "url": page.url})
                    else:
                        for p in paragraphs:
                            all_paragraphs.append({"text": p, "url": page.url})
                            
            except Exception as e:
                print(f"[WebRetriever] Could not fetch semantic evidence for '{query}': {e}")
                pass
                
            # --- DuckDuckGo Search ---
            try:
                ddg_results = DDGS().text(query, max_results=3)
                for res in ddg_results:
                    if "body" in res and "href" in res:
                        all_paragraphs.append({"text": res["body"], "url": res["href"]})
            except Exception as e:
                print(f"[WebRetriever] DDG search failed for '{query}': {e}")

        if not all_paragraphs:
            return []

        # Encode claim and all paragraphs
        claim_emb = self.encoder.encode(claim)
        
        # Extract just the text strings for embedding
        para_texts = [item["text"] for item in all_paragraphs]
        para_embs = self.encoder.encode(para_texts)
        
        # Compute cosine similarity
        scores = util.cos_sim(claim_emb, para_embs)[0]
        
        # Get the best paragraph
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        
        if best_score < 0.30:
            return []
            
        best_paragraph_data = all_paragraphs[best_idx]
        
        return [best_paragraph_data]

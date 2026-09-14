import wikipedia
from duckduckgo_search import DDGS
import arxiv
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings("ignore")
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize
import numpy as np

class WebRetriever:
    """
    A live web retriever that downloads full Wikipedia pages and uses a dense semantic
    model to find the exact paragraph that most closely matches the claim.
    """
    
    def __init__(self):
        wikipedia.set_user_agent("HallucinationDetectorBot/1.0 (rajmajumdermac@example.com)")
        print("Loading Semantic Search Model (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, queries, claim, domain="general", k=3):
        """
        Searches Wikipedia/DuckDuckGo or ArXiv, downloads all page contents, chunks into paragraphs,
        and uses Cosine Similarity to find the absolute best paragraph across all pages.
        """
        if not isinstance(queries, list):
            queries = [queries]
            
        all_paragraphs = []
        
        for query in queries:
            if domain == "scientific":
                try:
                    client = arxiv.Client()
                    search_obj = arxiv.Search(
                        query=query,
                        max_results=3,
                        sort_by=arxiv.SortCriterion.Relevance
                    )
                    for paper in client.results(search_obj):
                        if paper.summary:
                            all_paragraphs.append({"text": paper.summary, "url": paper.entry_id})
                except Exception as e:
                    print(f"[WebRetriever] ArXiv search failed for '{query}': {e}")
                    pass
                # Also fall back to DuckDuckGo just in case ArXiv misses it
                
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
                            
                    # Chunk using sliding window of sentences
                    try:
                        sentences = sent_tokenize(content)
                    except Exception:
                        import nltk
                        nltk.download('punkt', quiet=True)
                        nltk.download('punkt_tab', quiet=True)
                        sentences = sent_tokenize(content)

                    # Remove short/bad sentences
                    sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and not s.strip().startswith("==")]
                    
                    if not sentences:
                        all_paragraphs.append({"text": page.summary, "url": page.url})
                    else:
                        window_size = 5
                        overlap = 2
                        i = 0
                        while i < len(sentences):
                            chunk_sentences = sentences[i:i+window_size]
                            chunk_text = " ".join(chunk_sentences)
                            if len(chunk_text) > 100:
                                all_paragraphs.append({"text": chunk_text, "url": page.url})
                            i += (window_size - overlap)
                            
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

        # Deduplicate identical chunks
        unique_paragraphs = []
        seen = set()
        for p in all_paragraphs:
            if p["text"] not in seen:
                seen.add(p["text"])
                unique_paragraphs.append(p)
        all_paragraphs = unique_paragraphs

        # 1. DENSE SCORING (Cosine Similarity)
        claim_emb = self.encoder.encode(claim)
        para_texts = [item["text"] for item in all_paragraphs]
        para_embs = self.encoder.encode(para_texts)
        dense_scores = util.cos_sim(claim_emb, para_embs)[0].cpu().numpy()
        
        # Calculate dense ranks (0 is the best)
        dense_ranks = np.argsort(np.argsort(-dense_scores))
        
        # 2. SPARSE SCORING (BM25)
        tokenized_corpus = [doc.lower().split() for doc in para_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = claim.lower().split()
        sparse_scores = bm25.get_scores(tokenized_query)
        
        # Calculate sparse ranks (0 is the best)
        sparse_ranks = np.argsort(np.argsort(-sparse_scores))
        
        # 3. RECIPROCAL RANK FUSION (RRF)
        rrf_scores = []
        for i in range(len(all_paragraphs)):
            rrf_score = (1.0 / (60.0 + dense_ranks[i] + 1)) + (1.0 / (60.0 + sparse_ranks[i] + 1))
            rrf_scores.append(rrf_score)
            
        sorted_indices = np.argsort(-np.array(rrf_scores))
        
        top_k = []
        for i in sorted_indices[:k]:
            if dense_scores[i] >= 0.20 or sparse_scores[i] > 1.0:
                top_k.append(all_paragraphs[i])
                
        return top_k

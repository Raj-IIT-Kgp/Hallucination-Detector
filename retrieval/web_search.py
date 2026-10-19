import warnings
warnings.filterwarnings("ignore")
try:
    from bs4 import GuessedAtParserWarning
    warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
except Exception:
    pass

import functools
import os
import re
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import wikipedia
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
import arxiv
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Persistent requests.Session with connection pool for Wikipedia HTTP/1.1 Keep-Alive ──
_wiki_session = requests.Session()
_wiki_adapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
    max_retries=Retry(total=1, backoff_factor=0.2)
)
_wiki_session.mount("https://", _wiki_adapter)
_wiki_session.mount("http://", _wiki_adapter)

# ── Ensure Wikipedia uses an academic User-Agent and connection pool ──
try:
    wikipedia.set_user_agent("VeriFactBot/1.0 (academic research hallucination detection; contact@verifact.org)")
    def _patched_wiki_request(params):
        params['format'] = 'json'
        if 'action' not in params:
            params['action'] = 'query'
        headers = {'User-Agent': getattr(wikipedia.wikipedia, 'USER_AGENT', 'VeriFactBot/1.0 (academic research hallucination detection; contact@verifact.org)')}
        try:
            r = _wiki_session.get(wikipedia.wikipedia.API_URL, params=params, headers=headers, timeout=5.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {}
    wikipedia.wikipedia._wiki_request = _patched_wiki_request
except Exception:
    pass

@functools.lru_cache(maxsize=512)
def _cached_wiki_search(query: str) -> tuple[str, ...]:
    """Cached Wikipedia entity search with tuple return for LRU immutability."""
    try:
        results = wikipedia.search(query)
        return tuple(results) if results else ()
    except Exception:
        return ()


@functools.lru_cache(maxsize=256)
def _cached_wiki_fetch(match: str) -> tuple[tuple[dict, ...], dict | None]:
    """
    Fetches and chunks a Wikipedia page with LRU caching.
    Reuses connection pool session, parsing and chunking once into memory.
    Returns: (tuple of chunk dicts, structured page_entry dict)
    """
    try:
        try:
            page = wikipedia.page(match, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
            except Exception:
                return ((), None)
        except Exception:
            # Fallback to summary if page fails to load
            try:
                summary = wikipedia.summary(match, auto_suggest=False)
                page_entry = {
                    "title": match,
                    "summary": summary[:1500],
                    "url": f"https://en.wikipedia.org/wiki/{match.replace(' ', '_')}",
                    "content": summary
                }
                paragraphs = ({"text": summary, "url": page_entry["url"]},)
                return (paragraphs, page_entry)
            except Exception:
                return ((), None)

        # Clean page content to remove junk sections at the bottom
        content = page.content[:25000] if hasattr(page, 'content') else ""
        for section in ["== See also ==", "== References ==", "== External links =="]:
            if section in content:
                content = content.split(section)[0]

        summary_val = getattr(page, 'summary', '') or ''
        page_entry = {
            "title": getattr(page, 'title', match),
            "summary": summary_val[:1500] if summary_val else (content[:1500] if content else ""),
            "url": getattr(page, 'url', f"https://en.wikipedia.org/wiki/{match.replace(' ', '_')}"),
            "content": content
        }

        # Chunk using sliding window of sentences
        try:
            sentences = sent_tokenize(content)
        except Exception:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            sentences = sent_tokenize(content)

        sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and not s.strip().startswith("==")]
        paragraphs = []
        if not sentences:
            paragraphs.append({"text": page.summary, "url": page.url})
        else:
            window_size = 5
            overlap = 2
            i = 0
            while i < len(sentences):
                chunk_sentences = sentences[i:i + window_size]
                chunk_text = " ".join(chunk_sentences)
                if len(chunk_text) > 100:
                    paragraphs.append({"text": chunk_text, "url": page.url})
                i += (window_size - overlap)

        return (tuple(paragraphs), page_entry)
    except Exception as e:
        print(f"[WebRetriever] Could not fetch page '{match}': {e}")
        return ((), None)


class WebRetriever:
    """
    A live web retriever that downloads full Wikipedia pages and uses a dense semantic
    model to find the exact paragraph that most closely matches the claim.

    Optimised: Wikipedia page fetches, ArXiv searches, and DuckDuckGo searches
    for all queries are dispatched concurrently via a thread pool, cutting the
    dominant I/O wait time significantly.
    """
    
    def __init__(self):
        try:
            wikipedia.set_user_agent("VeriFactBot/1.0 (academic research hallucination detection; contact@verifact.org)")
        except Exception:
            pass
        print("Loading Semantic Search Model (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # SentenceTransformer uses PyTorch CPU routines, avoiding Apple Metal driver
        # conflicts with BART on MPS, while locking ensures thread safety.
        self._encode_lock = threading.Lock()
        self.page_cache = {}
        self._page_cache_lock = threading.Lock()

    def get_page_cache(self):
        """Returns a snapshot of pre-fetched Wikipedia pages/summaries for downstream reuse."""
        with self._page_cache_lock:
            return dict(self.page_cache)

    # ------------------------------------------------------------------ #
    #  Internal helpers — each fetches one source and returns paragraphs  #
    # ------------------------------------------------------------------ #

    def _fetch_arxiv(self, query):
        """Fetches ArXiv abstracts for a single query. Returns list of paragraph dicts."""
        paragraphs = []
        try:
            client = arxiv.Client(num_retries=0, page_size=3, delay_seconds=0.0)
            search_obj = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for paper in client.results(search_obj):
                if paper.summary:
                    paragraphs.append({"text": paper.summary, "url": paper.entry_id})
        except Exception:
            # Silence expected rate-limiting (HTTP 429) or network issues
            pass
        return paragraphs

    def _fetch_wikipedia_page(self, match):
        """Fetches and chunks a single Wikipedia page using LRU cache. Returns list of paragraph dicts."""
        paragraphs_tuple, page_entry = _cached_wiki_fetch(match)
        if page_entry:
            with self._page_cache_lock:
                self.page_cache[match.lower().strip()] = page_entry
                self.page_cache[page_entry["title"].lower().strip()] = page_entry
        return list(paragraphs_tuple)

    def _fetch_ddg(self, query):
        """Fetches DuckDuckGo results for a single query with backend='api' and strict 4.0s timeout."""
        paragraphs = []
        try:
            if isinstance(query, list):
                query = " ".join([str(q) for q in query])
            query_str = str(query).strip()
            if not query_str:
                return paragraphs

            def _search():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Try 'yahoo,duckduckgo' backends (fastest, avoids Cloudflare HTTP/2 resets)
                    try:
                        return list(DDGS(timeout=2.5).text(query_str, backend="yahoo,duckduckgo", max_results=3))
                    except Exception:
                        try:
                            return list(DDGS(timeout=2.0).text(query_str, backend="yahoo", max_results=3))
                        except Exception:
                            return []

            executor = ThreadPoolExecutor(max_workers=1)
            try:
                fut = executor.submit(_search)
                ddg_results = fut.result(timeout=3.0)
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                return paragraphs
            else:
                executor.shutdown(wait=False)

            if ddg_results:
                for res in ddg_results:
                    if "body" in res and "href" in res:
                        paragraphs.append({"text": res["body"], "url": res["href"]})
        except Exception:
            # DDG rate-limited or timed out; rely cleanly on Wikipedia/ArXiv
            pass
        return paragraphs


    def _fetch_all_for_query(self, query, domain):
        """
        Dispatches ArXiv (if scientific) and Wikipedia top-2 pages concurrently
        for a single entity query. Returns aggregated paragraph list.
        DuckDuckGo is intentionally NOT queried per-entity to eliminate SEO noise
        and prevent HTTP 429 rate limiting.
        """
        tasks = []

        if domain == "scientific":
            tasks.append(("arxiv", query))

        # Wikipedia: first get the search result titles with LRU caching, then fetch pages in parallel
        try:
            search_results = list(_cached_wiki_search(query.strip()))
        except Exception:
            search_results = []

        top_matches = search_results[:2] if search_results else []
        for match in top_matches:
            tasks.append(("wiki", match))

        if not tasks:
            return []

        paragraphs = []
        max_w = min(len(tasks), 4)
        executor = ThreadPoolExecutor(max_workers=max_w)
        future_map = {}
        for kind, arg in tasks:
            if kind == "arxiv":
                f = executor.submit(self._fetch_arxiv, arg)
            else:  # wiki
                f = executor.submit(self._fetch_wikipedia_page, arg)
            future_map[f] = kind

        try:
            for future in as_completed(future_map, timeout=5.0):
                try:
                    paragraphs.extend(future.result())
                except Exception as e:
                    print(f"[WebRetriever] Task failed: {e}")
        except TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=False)

        # Map original query string to the cached page of the top match
        with self._page_cache_lock:
            q_key = query.lower().strip()
            if q_key not in self.page_cache:
                for match in top_matches:
                    m_key = match.lower().strip()
                    if m_key in self.page_cache:
                        self.page_cache[q_key] = self.page_cache[m_key]
                        break

        return paragraphs

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def search(self, queries, claim, domain="general", k=3):
        """
        Searches Wikipedia (top-2 pages per entity query) and ArXiv concurrently,
        accompanied by a single propositional DuckDuckGo search on the claim string.
        Then uses Cosine Similarity + BM25 RRF to find the best paragraphs.
        """
        if not isinstance(queries, list):
            queries = [queries]

        flat_queries = []
        for q in queries:
            if isinstance(q, list):
                flat_queries.extend([str(item) for item in q if str(item).strip()])
            elif str(q).strip():
                flat_queries.append(str(q).strip())
        queries = flat_queries if flat_queries else [claim]

        # Fetch all entity queries concurrently (Wikipedia + ArXiv),
        # plus a SINGLE unified DuckDuckGo web search on the full claim string.
        all_paragraphs = []
        max_workers = min(len(queries) + 1, 6)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._fetch_all_for_query, q, domain) for q in queries]
            # Single-shot propositional web search: query DuckDuckGo ONCE for the claim proposition,
            # avoiding rate limits and SEO snippet pollution from multiple entity-noun queries.
            if claim and claim.strip():
                futures.append(executor.submit(self._fetch_ddg, claim.strip()))

            for future in as_completed(futures):
                try:
                    all_paragraphs.extend(future.result())
                except Exception as e:
                    print(f"[WebRetriever] Query fetch failed: {e}")

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

        para_texts = [item["text"] for item in all_paragraphs]

        # 1. DENSE SCORING (Cosine Similarity) — batch-encode for speed.
        # Lock ensures only one thread calls the SentenceTransformer C-extension
        # at a time (prevents macOS "double free" when claims run concurrently).
        with self._encode_lock:
            claim_emb = self.encoder.encode(claim, show_progress_bar=False)
            para_embs = self.encoder.encode(para_texts, batch_size=64, show_progress_bar=False)
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
        rrf_scores = [
            (1.0 / (60.0 + dense_ranks[i] + 1)) + (1.0 / (60.0 + sparse_ranks[i] + 1))
            for i in range(len(all_paragraphs))
        ]

        sorted_indices = np.argsort(-np.array(rrf_scores))

        top_k = []
        for i in sorted_indices:
            if dense_scores[i] >= 0.20 or sparse_scores[i] > 1.0:
                top_k.append(all_paragraphs[i])
                if len(top_k) >= k:
                    break

        if not top_k and all_paragraphs:
            best_idx = sorted_indices[0]
            # Soft fallback: only retain the best candidate if it exhibits at least marginal relevance.
            # Completely unrelated text (similarity < 0.15) is rejected to allow Phase 3.1 Absence Evaluation.
            if dense_scores[best_idx] >= 0.15 or sparse_scores[best_idx] > 0.5:
                top_k = [all_paragraphs[best_idx]]

        return top_k


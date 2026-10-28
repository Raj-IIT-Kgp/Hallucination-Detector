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
import time
from collections import OrderedDict
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import wikipedia
import requests
from dotenv import load_dotenv
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

# Retrieval caches: ~384-dim float32 vectors (~1.5 KB each) and short-lived DuckDuckGo results
_EMBEDDING_CACHE_MAX = 20000
_DDG_CACHE_TTL = 600.0
_DDG_CACHE_MAX = 512
# Web search engines (via the ddgs package), queried in parallel; the first to return results is
# used, since individual engines are frequently blocked or rate-limited.
_WEB_ENGINES = ("yahoo", "bing", "duckduckgo", "brave", "yandex", "mojeek")
_WEB_ENGINE_TIMEOUT = 5.0   # per engine
_WEB_DEADLINE = 5.0         # for the whole web search: the cap of a Wikipedia page fetch

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
# Wikimedia rate-limits API clients without a User-Agent that names a reachable contact much more
# tightly; same identity as agents/wikidata_graph.py
_WIKI_USER_AGENT = "VeriFact/1.0 (academic research prototype; https://github.com/Raj-IIT-Kgp/Hallucination-Detector)"
try:
    wikipedia.set_user_agent(_WIKI_USER_AGENT)
    def _patched_wiki_request(params):
        params['format'] = 'json'
        if 'action' not in params:
            params['action'] = 'query'
        headers = {'User-Agent': getattr(wikipedia.wikipedia, 'USER_AGENT', _WIKI_USER_AGENT)}
        try:
            # https directly: the library's http:// URL redirects, and the login cookie is https-only
            r = _wiki_session.get(wikipedia.wikipedia.API_URL.replace("http://", "https://"),
                                  params=params, headers=headers, timeout=5.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {}
    wikipedia.wikipedia._wiki_request = _patched_wiki_request
except Exception:
    pass

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_wiki_logged_in = False


def _wiki_login():
    """
    Logs the shared Wikipedia session in with the bot password in .env (WIKI_BOT_USER, WIKI_BOT_PASSWORD;
    created at Special:BotPasswords), so Wikimedia rate-limits it as a logged-in account instead of by IP
    address. Without those settings, or if the login fails, requests stay anonymous as before.
    """
    global _wiki_logged_in
    if _wiki_logged_in:
        return True
    load_dotenv()
    user, password = os.getenv("WIKI_BOT_USER"), os.getenv("WIKI_BOT_PASSWORD")
    if not user or not password:
        return False
    headers = {"User-Agent": _WIKI_USER_AGENT}
    try:
        token = _wiki_session.get(_WIKI_API, params={"action": "query", "meta": "tokens", "type": "login",
                                                     "format": "json"}, headers=headers, timeout=10.0
                                  ).json()["query"]["tokens"]["logintoken"]
        result = _wiki_session.post(_WIKI_API, data={"action": "login", "lgname": user, "lgpassword": password,
                                                     "lgtoken": token, "format": "json"}, headers=headers, timeout=10.0
                                    ).json().get("login", {})
    except Exception as e:
        print(f"[WebRetriever] Wikipedia login failed ({e}); continuing anonymously")
        return False
    _wiki_logged_in = result.get("result") == "Success"
    print(f"[WebRetriever] Wikipedia login: {'logged in as ' + result.get('lgusername', user) if _wiki_logged_in else result.get('reason', result)}")
    return _wiki_logged_in

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

        # The wikipedia library loads .content and .summary lazily with one HTTP request each;
        # issue both concurrently instead of back-to-back (same requests, one round trip saved).
        with ThreadPoolExecutor(max_workers=2) as prop_executor:
            content_future = prop_executor.submit(getattr, page, 'content', "")
            summary_future = prop_executor.submit(getattr, page, 'summary', '')
            content = (content_future.result() or "")[:25000]
            summary_val = summary_future.result() or ''

        # Clean page content to remove junk sections at the bottom
        for section in ["== See also ==", "== References ==", "== External links =="]:
            if section in content:
                content = content.split(section)[0]

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
            wikipedia.set_user_agent(_WIKI_USER_AGENT)
        except Exception:
            pass
        _wiki_login()
        print("Loading Semantic Search Model (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # SentenceTransformer uses PyTorch CPU routines, avoiding Apple Metal driver
        # conflicts with BART on MPS, while locking ensures thread safety.
        self._encode_lock = threading.Lock()
        self.page_cache = {}
        self._page_cache_lock = threading.Lock()
        # Paragraph-embedding cache (text -> vector). Wikipedia pages recur across searches
        # (query-rewrite retries, shared pages across claims) and re-encoding ~160 paragraphs
        # costs ~2.5s on CPU. Guarded by _encode_lock.
        self._embedding_cache = OrderedDict()
        # DuckDuckGo results per query string, reused within _DDG_CACHE_TTL so a query-rewrite
        # retry (which re-searches the same claim seconds later) does not pay another 2-3s.
        self._ddg_cache = {}
        self._ddg_cache_lock = threading.Lock()

    def _encode_cached(self, texts):
        """Encodes texts, reusing cached embeddings. Caller must hold _encode_lock."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._embedding_cache]
        if missing:
            embs = self.encoder.encode(missing, batch_size=64, show_progress_bar=False)
            for t, e in zip(missing, embs):
                self._embedding_cache[t] = e
        for t in texts:
            self._embedding_cache.move_to_end(t)
        while len(self._embedding_cache) > _EMBEDDING_CACHE_MAX:
            self._embedding_cache.popitem(last=False)
        return np.stack([self._embedding_cache[t] for t in texts])

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
        """Web search for a single query: the first of _WEB_ENGINES to return results within _WEB_DEADLINE."""
        paragraphs = []
        try:
            if isinstance(query, list):
                query = " ".join([str(q) for q in query])
            query_str = str(query).strip()
            if not query_str:
                return paragraphs

            with self._ddg_cache_lock:
                cached = self._ddg_cache.get(query_str)
            if cached and time.monotonic() - cached[0] < _DDG_CACHE_TTL:
                return [dict(p) for p in cached[1]]

            def _search(engine):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return list(DDGS(timeout=_WEB_ENGINE_TIMEOUT).text(query_str, backend=engine, max_results=3))

            ddg_results = []
            executor = ThreadPoolExecutor(max_workers=len(_WEB_ENGINES))
            futures = [executor.submit(_search, engine) for engine in _WEB_ENGINES]
            try:
                for fut in as_completed(futures, timeout=_WEB_DEADLINE):
                    try:
                        ddg_results = fut.result()
                    except Exception:
                        continue
                    if ddg_results:
                        break
            except TimeoutError:
                pass
            # engines still running finish in the background, bounded by their own timeout
            executor.shutdown(wait=False, cancel_futures=True)

            if ddg_results:
                for res in ddg_results:
                    if "body" in res and "href" in res:
                        # Tagged so the detector can strip open-web snippets that merely quote the claim
                        paragraphs.append({"text": res["body"], "url": res["href"], "source": "web"})
            # Only successful lookups are cached; timeouts/rate limits are retried as before
            if paragraphs:
                with self._ddg_cache_lock:
                    if len(self._ddg_cache) >= _DDG_CACHE_MAX:
                        self._ddg_cache.clear()
                    self._ddg_cache[query_str] = (time.monotonic(), [dict(p) for p in paragraphs])
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

    def search(self, queries, claim, domain="general", k=3, diagnostics=None):
        """
        Searches Wikipedia (top-2 pages per entity query) and ArXiv concurrently,
        accompanied by a single propositional DuckDuckGo search on the claim string.
        Then uses Cosine Similarity + BM25 RRF to find the best paragraphs.

        If diagnostics is a dict, it is filled with the re-ranking numbers (candidate counts per
        source and the top candidates' scores and ranks) for display; the result is unchanged.
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
            para_embs = self._encode_cached(para_texts)
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

        if diagnostics is not None:
            selected = {id(p) for p in top_k}
            def source_of(p):
                url = p.get("url") or ""
                return p.get("source") or ("arxiv" if "arxiv.org" in url else "wikipedia")
            counts = {}
            for p in all_paragraphs:
                counts[source_of(p)] = counts.get(source_of(p), 0) + 1
            diagnostics.update({
                "queries": list(queries),
                "candidates": len(all_paragraphs),
                "by_source": counts,
                "avgdl": round(float(np.mean([len(t) for t in tokenized_corpus])), 1) if tokenized_corpus else 0.0,
                "top": [{
                    "text": all_paragraphs[i]["text"][:160],
                    "source": source_of(all_paragraphs[i]),
                    "dense": round(float(dense_scores[i]), 3),
                    "bm25": round(float(sparse_scores[i]), 2),
                    "dense_rank": int(dense_ranks[i]) + 1,
                    "bm25_rank": int(sparse_ranks[i]) + 1,
                    "rrf": round(float(rrf_scores[i]), 5),
                    "selected": id(all_paragraphs[i]) in selected,
                } for i in sorted_indices[:8]],
                "selected": len(top_k),
            })

        return top_k


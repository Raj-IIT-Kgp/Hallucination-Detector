import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.web_search import (
    WebRetriever, 
    _wiki_session, 
    _cached_wiki_search, 
    _cached_wiki_fetch
)

def test_wiki_session_pooling():
    assert _wiki_session is not None, "Expected _wiki_session to exist"
    assert "https://" in _wiki_session.adapters, "Expected https adapter to be mounted"
    adapter = _wiki_session.adapters["https://"]
    assert getattr(adapter, "_pool_connections", 0) >= 20
    assert getattr(adapter, "_pool_maxsize", 0) >= 20
    print("PASS: _wiki_session connection pool properly configured (20 pooled connections).")


def test_lru_cache_search_and_fetch():
    # Test cached search
    _cached_wiki_search.cache_clear()
    info_before = _cached_wiki_search.cache_info()
    
    # First call (mock or live)
    res1 = _cached_wiki_search("Python (programming language)")
    # Second call should be a cache hit
    res2 = _cached_wiki_search("Python (programming language)")
    
    info_after = _cached_wiki_search.cache_info()
    assert info_after.hits >= 1, f"Expected cache hit on repeated search, got: {info_after}"
    assert res1 == res2
    print(f"PASS: _cached_wiki_search LRU hit verified (Hits: {info_after.hits})")

    # Test cached fetch
    _cached_wiki_fetch.cache_clear()
    t0 = time.time()
    paragraphs1, entry1 = _cached_wiki_fetch("Python (programming language)")
    t_first = time.time() - t0

    t1 = time.time()
    paragraphs2, entry2 = _cached_wiki_fetch("Python (programming language)")
    t_cached = time.time() - t1

    fetch_info = _cached_wiki_fetch.cache_info()
    assert fetch_info.hits >= 1, f"Expected cache hit on repeated fetch, got: {fetch_info}"
    assert paragraphs1 == paragraphs2
    assert entry1["title"] == entry2["title"]
    assert t_cached < 0.005, f"Cached call took too long: {t_cached:.4f}s"
    print(f"PASS: _cached_wiki_fetch LRU hit verified in {t_cached*1000:.3f}ms (vs {t_first*1000:.2f}ms on cold fetch)!")


if __name__ == "__main__":
    test_wiki_session_pooling()
    test_lru_cache_search_and_fetch()
    print("All Session Pooling and LRU Cache tests passed!")

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.web_search import WebRetriever

def test_fetch_ddg_bounded_timeout():
    retriever = WebRetriever()
    
    t0 = time.time()
    results = retriever._fetch_ddg("Python language official website release")
    elapsed = time.time() - t0

    # Ensure search either completes or gracefully times out within <= 4.0s
    assert elapsed < 4.0, f"DuckDuckGo search took too long: {elapsed:.2f}s (should be capped at 3s)"
    assert isinstance(results, list)
    print(f"PASS: _fetch_ddg completed in {elapsed:.2f}s (< 4s cap) with {len(results)} results.")

def test_fetch_ddg_empty_query():
    retriever = WebRetriever()
    results = retriever._fetch_ddg("")
    assert results == []
    print("PASS: _fetch_ddg handles empty input cleanly.")

if __name__ == "__main__":
    test_fetch_ddg_bounded_timeout()
    test_fetch_ddg_empty_query()
    print("All DDG timeout tests passed!")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.knowledge_graph import KnowledgeGraph
import agents.knowledge_graph as kg_module

def test_kg_build_reuses_cached_pages():
    # Setup mock cached pages simulating Phase 3 retrieval output
    cached_pages = {
        "ashneer grover": {
            "title": "Ashneer Grover",
            "summary": "Ashneer Grover is an Indian entrepreneur and former co-founder of BharatPe.",
            "url": "https://en.wikipedia.org/wiki/Ashneer_Grover"
        },
        "bharatpe": {
            "title": "BharatPe",
            "summary": "BharatPe is an Indian fintech company co-founded by Ashneer Grover.",
            "url": "https://en.wikipedia.org/wiki/BharatPe"
        }
    }

    # Track calls to network fetcher
    fetch_calls = []
    original_fetch = kg_module._fetch_wikipedia_summary

    def mocked_fetch(title, context_claim=None):
        fetch_calls.append(title)
        return original_fetch(title, context_claim)

    kg_module._fetch_wikipedia_summary = mocked_fetch

    try:
        kg = KnowledgeGraph()
        # Build using entities that exist in cached_pages
        entities = ["Ashneer Grover", "BharatPe"]
        kg.build(entities, claim="Ashneer Grover co-founded BharatPe", cached_pages=cached_pages)

        # Assert zero network calls were made because both entities were cached
        assert len(fetch_calls) == 0, f"Expected 0 network fetches, but got {len(fetch_calls)} calls: {fetch_calls}"

        # Assert nodes were properly added with summary and url from cache
        assert "Ashneer Grover" in kg.graph.nodes
        assert "BharatPe" in kg.graph.nodes
        assert kg.node_data["Ashneer Grover"]["summary"] == cached_pages["ashneer grover"]["summary"]
        assert kg.node_data["BharatPe"]["summary"] == cached_pages["bharatpe"]["summary"]
        print("PASS: KnowledgeGraph.build successfully reused cached_pages with 0 network calls!")

    finally:
        kg_module._fetch_wikipedia_summary = original_fetch


def test_kg_build_partial_cache():
    cached_pages = {
        "entity a": {
            "title": "Entity A",
            "summary": "Entity A summary text.",
            "url": "https://en.wikipedia.org/wiki/Entity_A"
        }
    }

    fetch_calls = []

    def mock_fetch(title, context_claim=None):
        fetch_calls.append(title)
        return {"title": title, "summary": f"{title} network summary.", "url": f"https://wiki/{title}"}

    original_fetch = kg_module._fetch_wikipedia_summary
    kg_module._fetch_wikipedia_summary = mock_fetch

    try:
        kg = KnowledgeGraph()
        entities = ["Entity A", "Entity B"]
        kg.build(entities, cached_pages=cached_pages)

        # Entity A should be from cache; Entity B should be fetched
        assert "Entity B" in fetch_calls, "Expected Entity B to be fetched"
        assert "Entity A" not in fetch_calls, "Expected Entity A to come from cache"
        assert "Entity A" in kg.graph.nodes
        assert "Entity B" in kg.graph.nodes
        print("PASS: Partial cache hit correctly fetched only missing entity!")

    finally:
        kg_module._fetch_wikipedia_summary = original_fetch


if __name__ == "__main__":
    test_kg_build_reuses_cached_pages()
    test_kg_build_partial_cache()
    print("All KnowledgeGraph page cache tests passed!")

import sys
from agents.knowledge_graph import KnowledgeGraph, _extract_relationships

def test_extract_relationships_provenance():
    # Mock summaries
    summary_a = "Christopher Nolan is a British filmmaker known for directing Inception and Oppenheimer."
    summary_b = "Inception is a 2010 science fiction film written and directed by Christopher Nolan."
    
    # Test relationship extraction
    res = _extract_relationships("Christopher Nolan", "Inception", summary_a, summary_b)
    print("Test direct relation:", res)
    assert res is not None, "Expected relation to be found"
    assert "relationship" in res
    assert "verified" in res
    assert res["verified"] is True, "Expected quote to be verified in text"
    assert res["quote"] is not None

def test_unrelated_provenance():
    summary_a = "Kangaroos are indigenous to Australia."
    summary_b = "Quantum mechanics is a fundamental theory in physics."
    
    res = _extract_relationships("Kangaroo", "Quantum Mechanics", summary_a, summary_b)
    print("Test unrelated:", res)
    assert res is None or res.get("relationship") is None or res.get("verified") is False

def test_evidence_pool_filtering():
    kg = KnowledgeGraph()
    kg.node_data = {
        "EntityA": {"summary": "EntityA is a pioneer in astronomy.", "url": "https://wiki/A"},
        "EntityB": {"summary": "EntityB discovered a new galaxy.", "url": "https://wiki/B"}
    }
    kg.graph.add_node("EntityA")
    kg.graph.add_node("EntityB")
    
    # 1. Add unverified edge (hallucinated or lack of quote)
    kg.graph.add_edge("EntityA", "EntityB", relationship="Invented fake relationship", quote=None, verified=False)
    ev_unverified = kg.get_multi_hop_evidence("Did EntityA invent anything with EntityB?")
    edge_hits = [e for e in ev_unverified if "fake relationship" in e["text"]]
    assert len(edge_hits) == 0, "Unverified edge should NOT be admitted into evidence pool!"
    
    # 2. Add verified edge with source quote
    kg.graph["EntityA"]["EntityB"]["relationship"] = "Collaborated on galaxy discovery"
    kg.graph["EntityA"]["EntityB"]["quote"] = "discovered a new galaxy"
    kg.graph["EntityA"]["EntityB"]["verified"] = True
    kg.graph["EntityA"]["EntityB"]["source_node"] = "EntityB"
    
    ev_verified = kg.get_multi_hop_evidence("Did EntityA collaborate on galaxy discovery with EntityB?")
    grounded_hits = [e for e in ev_verified if "Grounded Relationship" in e["text"]]
    assert len(grounded_hits) > 0, "Verified edge should be admitted into evidence pool!"
    print("Test evidence pool filtering passed successfully!")

def test_batch_relationship_extraction():
    from agents.knowledge_graph import _extract_batch_relationships
    node_data = {
        "Christopher Nolan": {
            "summary": "Christopher Nolan is a British filmmaker known for directing Inception and Oppenheimer.",
            "url": "https://en.wikipedia.org/wiki/Christopher_Nolan"
        },
        "Inception": {
            "summary": "Inception is a 2010 science fiction film written and directed by Christopher Nolan.",
            "url": "https://en.wikipedia.org/wiki/Inception"
        },
        "Kangaroo": {
            "summary": "Kangaroos are indigenous marsupials native to Australia.",
            "url": "https://en.wikipedia.org/wiki/Kangaroo"
        }
    }
    nodes = list(node_data.keys())
    edges = _extract_batch_relationships(nodes, node_data)
    print("Test batch extraction edges:", edges)
    assert isinstance(edges, list)
    # Nolan <-> Inception should be found and verified
    nolan_inception = [e for e in edges if ("Christopher Nolan" in (e["source"], e["target"])) and ("Inception" in (e["source"], e["target"]))]
    assert len(nolan_inception) == 1, f"Expected exactly 1 edge between Christopher Nolan and Inception, found: {nolan_inception}"
    assert nolan_inception[0]["verified"] is True
    assert nolan_inception[0]["quote"] is not None
    # Kangaroo should NOT have verified relations with Nolan or Inception
    kangaroo_edges = [e for e in edges if "Kangaroo" in (e["source"], e["target"]) and e["verified"]]
    assert len(kangaroo_edges) == 0, "Kangaroo should have 0 verified edges"
    print("Batch relationship extraction test passed!")

if __name__ == "__main__":
    test_extract_relationships_provenance()
    test_unrelated_provenance()
    test_batch_relationship_extraction()
    test_evidence_pool_filtering()
    print("All Knowledge Graph provenance tests passed!")


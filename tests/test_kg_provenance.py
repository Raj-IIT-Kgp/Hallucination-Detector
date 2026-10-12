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

if __name__ == "__main__":
    test_extract_relationships_provenance()
    test_unrelated_provenance()
    test_evidence_pool_filtering()
    print("All Knowledge Graph provenance tests passed!")

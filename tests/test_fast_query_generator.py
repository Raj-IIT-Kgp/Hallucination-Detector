import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_generator import generate_queries, fallback_queries
import agents.query_generator as qg_module

def test_fast_path_with_long_context():
    long_context = """
    Ashneer Grover is an Indian entrepreneur and former managing director of BharatPe.
    He co-founded the fintech company in 2018 alongside Shashvat Nakrani. He has also
    appeared as an investor and shark on the popular reality television show Shark Tank India.
    Later in 2022, he resigned following governance disputes.
    """ # > 40 words

    claim = "Ashneer Grover is the co-founder of BharatPe."

    # Monkeypatch generate_response to ensure LLM is NOT called
    llm_called = []
    def fail_if_called(prompt):
        llm_called.append(prompt)
        raise RuntimeError("LLM was unexpectedly called for a self-contained atomic claim!")

    original_gen = qg_module.generate_response
    qg_module.generate_response = fail_if_called

    try:
        t0 = time.time()
        res = generate_queries(claim, context=long_context)
        t_elapsed = time.time() - t0

        assert len(llm_called) == 0, "LLM was called!"
        assert isinstance(res, dict)
        assert "queries" in res
        assert "Ashneer Grover" in res["queries"]
        assert "BharatPe" in res["queries"]
        assert t_elapsed < 0.05, f"Execution took too long: {t_elapsed:.4f}s"
        print(f"PASS: Fast-path completed in {t_elapsed*1000:.2f}ms without LLM call! Queries: {res['queries']}")

    finally:
        qg_module.generate_response = original_gen


def test_pronoun_routes_to_llm():
    long_context = "Tim Cook is the chief executive officer of Apple Inc."
    claim = "He is the current CEO of Apple."

    llm_called = []
    def mock_llm(prompt, **kwargs):
        llm_called.append(prompt)
        return '{"queries": ["Tim Cook", "Apple Inc."], "domain": "general", "requires_current_state": true}'

    original_gen = qg_module.generate_response
    qg_module.generate_response = mock_llm

    try:
        res = generate_queries(claim, context=long_context)
        assert len(llm_called) == 1, "Expected LLM to be called to resolve pronoun 'He'!"
        assert "Tim Cook" in res["queries"]
        print("PASS: Pronoun 'He' correctly bypassed fast-path and routed to LLM!")
    finally:
        qg_module.generate_response = original_gen

def test_creative_work_routes_to_llm():
    claim = "Off the Wall is an album."
    llm_called = []
    def mock_llm(prompt, **kwargs):
        llm_called.append(prompt)
        return '{"queries": ["Off the Wall (album)"], "domain": "general", "requires_current_state": false}'

    original_gen = qg_module.generate_response
    qg_module.generate_response = mock_llm
    try:
        res = generate_queries(claim)
        assert len(llm_called) == 1, "Creative works must route to LLM to prevent fragmenting!"
        assert res["queries"] == ["Off the Wall (album)"]
        print("PASS: Creative work correctly bypassed fast-path and routed to LLM!")
    finally:
        qg_module.generate_response = original_gen

def test_alphanumeric_entity_extraction():
    claim = "Lockheed Martin F-35 Lightning II is an aircraft."
    # Should not call LLM, should extract full name without breaking on -35
    original_gen = qg_module.generate_response
    def fail_if_called(prompt):
        raise RuntimeError("Unexpected LLM call!")
    qg_module.generate_response = fail_if_called
    try:
        res = generate_queries(claim)
        assert res["queries"] == ["Lockheed Martin F-35 Lightning II"]
        print(f"PASS: Alphanumeric entity correctly extracted intact: {res['queries']}")
    finally:
        qg_module.generate_response = original_gen

def test_fallback_queries_search_the_relation():
    """Without the LLM rewrite, the second pass searches what the claim says about its entities."""
    assert fallback_queries("shahrukh khan is the richest actor in the world", ["Shahrukh Khan"]) == ["richest actor in the world"]
    assert fallback_queries("The Eiffel Tower was completed in 1889", ["Eiffel Tower"]) == ["completed in 1889"]
    # too little left after removing the names: the claim itself
    assert fallback_queries("Tim Cook is the CEO of Apple", ["Tim Cook", "Apple"]) == ["Tim Cook is the CEO of Apple"]
    print("PASS: fallback queries")


def test_failed_rewrite_still_runs_second_pass():
    """An empty LLM rewrite falls back to the claim's relation, and a claim with no evidence at all
    returns the four values the caller unpacks."""
    import detector.hallucination_detector as hd

    class Retriever:
        def __init__(self):
            self.calls = []
        def search(self, queries, claim=None, domain="general", k=3):
            self.calls.append(list(queries))
            return []

    original = (hd.rewrite_queries, hd.evaluate_absence)
    hd.rewrite_queries = lambda *a, **kw: []
    hd.evaluate_absence = lambda *a, **kw: ("unknown", None)
    try:
        retriever = Retriever()
        det = hd.HallucinationDetector(retriever)
        verification, evidence, sources, graph = det._verify_claim_full(
            "shahrukh khan is the richest actor in the world", "", "general", ["Shahrukh Khan"])
    finally:
        hd.rewrite_queries, hd.evaluate_absence = original
    assert retriever.calls == [["Shahrukh Khan"], ["richest actor in the world"]]
    step = next(s for s in verification["trace"] if s["kind"] == "rewrite")
    assert step["route"] == "fallback"
    assert verification["final_decision_policy"] == "absence_heuristic" and graph == {"nodes": [], "edges": []}
    print("PASS: failed rewrite falls back, no-evidence path returns four values")


if __name__ == "__main__":
    test_fast_path_with_long_context()
    test_pronoun_routes_to_llm()
    test_creative_work_routes_to_llm()
    test_alphanumeric_entity_extraction()
    test_fallback_queries_search_the_relation()
    test_failed_rewrite_still_runs_second_pass()
    print("All fast-path query generator tests passed!")

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import wikidata_graph as wg

# Offline Wikidata: a few entities in wbgetentities form, served by a stub of wg._api


def item(qid):
    return {"mainsnak": {"snaktype": "value", "datavalue": {"type": "wikibase-entityid", "value": {"id": qid}}},
            "rank": "normal", "references": [{}]}


def date(text, precision=11):
    return {"mainsnak": {"snaktype": "value", "datavalue": {"type": "time", "value": {"time": text, "precision": precision}}},
            "rank": "normal", "references": [{}, {}]}


ENTITIES = {
    "Q937": {"id": "Q937", "labels": {"en": {"value": "Albert Einstein"}},
             "descriptions": {"en": {"value": "German-born theoretical physicist"}},
             "sitelinks": {"enwiki": {"title": "Albert Einstein"}},
             "claims": {"P31": [item("Q5")], "P569": [date("+1879-03-14T00:00:00Z")], "P19": [item("Q3012")],
                        "P166": [item("Q38104"), item("Q1")]}},
    "Q312": {"id": "Q312", "labels": {"en": {"value": "Apple Inc."}}, "sitelinks": {"enwiki": {"title": "Apple Inc."}},
             "claims": {"P169": [item("Q265852")]}},
    "Q265852": {"id": "Q265852", "labels": {"en": {"value": "Tim Cook"}}, "sitelinks": {"enwiki": {"title": "Tim Cook"}},
                "claims": {"P31": [item("Q5")]}},
    "Q7186": {"id": "Q7186", "labels": {"en": {"value": "Marie Curie"}}, "sitelinks": {"enwiki": {"title": "Marie Curie"}},
              "claims": {"P31": [item("Q5")], "P569": [date("+1867-11-07T00:00:00Z")]}},
    "Q99": {"id": "Q99", "labels": {"en": {"value": "Mercury"}}, "claims": {"P31": [item("Q4167410")]}},
    "Q500": {"id": "Q500", "labels": {"en": {"value": "List of characters in The Lego Movie franchise"}}, "claims": {}},
}
TITLES = {"albert einstein": "Q937", "apple inc.": "Q312", "tim cook": "Q265852", "marie curie": "Q7186", "mercury": "Q99"}
LABELS = {"Q5": "human", "Q3012": "Ulm", "Q38104": "Nobel Prize in Physics", "Q1": "Copley Medal"}
SEARCH = {"Marie Curie favourite colour": [{"id": "Q500", "label": "List of characters in The Lego Movie franchise"}]}


def fake_api(params):
    if params["action"] == "wbsearchentities":
        return {"search": SEARCH.get(params["search"], [])}
    if "titles" in params:
        qid = TITLES.get(params["titles"].lower())
        return {"entities": {qid: ENTITIES[qid]} if qid else {"-1": {"missing": ""}}}
    out = {}
    for qid in params["ids"].split("|"):
        out[qid] = ENTITIES.get(qid) or {"id": qid, "labels": {"en": {"value": LABELS.get(qid, qid)}}}
    return {"entities": out}


def setup():
    wg._api = fake_api
    wg._entity_cache.clear()
    wg._label_cache.clear()


def test_resolution_is_exact_or_nothing():
    setup()
    assert wg.resolve_entity("Albert Einstein")["id"] == "Q937"
    # a search phrase with no page: the search hit shares no content word, so nothing is guessed
    assert wg.resolve_entity("Marie Curie favourite colour") is None
    # a disambiguation page is not an entity
    assert wg.resolve_entity("Mercury") is None
    print("Entity resolution tests passed!")


def test_claim_compared_with_record():
    setup()
    g = wg.build_graph("Albert Einstein was born in 1950.", ["Albert Einstein"])
    m = g["match"]
    assert (m["property"], m["status"], m["claimed"], m["recorded"]) == ("date of birth", "differs", "1950", ["14 March 1879"])
    kinds = {e["kind"] for e in g["edges"]}
    assert "claim" in kinds and "record" in kinds
    assert all(n["id"] != "Q500" for n in g["nodes"])

    m = wg.build_graph("Albert Einstein was born on 14 March 1879.", ["Albert Einstein"])["match"]
    assert m["status"] == "agrees" and m["agreeing"] == "14 March 1879"
    m = wg.build_graph("Albert Einstein was born on 15 March 1879.", ["Albert Einstein"])["match"]
    assert m["status"] == "differs"
    print("Claim comparison tests passed!")


def test_no_matching_property():
    setup()
    g = wg.build_graph("Marie Curie's favourite colour was green.", ["Marie Curie"])
    assert g["match"]["status"] == "unmatched"
    assert [n["label"] for n in g["nodes"] if n["kind"] == "entity"][0] == "Marie Curie"
    print("Unmatched property test passed!")


def test_statement_stored_on_the_other_entity():
    setup()
    g = wg.build_graph("Tim Cook is the CEO of Apple.", ["Tim Cook", "Apple Inc."])
    m = g["match"]
    assert m["property"] == "chief executive officer" and m["status"] == "agrees" and g["subject"] == "Q312"
    print("Reverse statement test passed!")


def test_stream_sends_graph_after_verdict():
    """The stream sends each claim's Wikidata graph after its verdict, and the verdict is unchanged."""
    import time
    import detector.hallucination_detector as hd

    det = hd.HallucinationDetector.__new__(hd.HallucinationDetector)
    det.wikidata_enabled = True
    result = {"claim": "c", "verification": {"label": "contradicted"}, "graph": {"nodes": [], "edges": []}}

    class SlowFuture:
        def result(self, timeout=None):
            time.sleep(0.05)
            return {"source": "wikidata", "nodes": [{"id": "Q937"}], "edges": []}

    def fake_process(claim, answer, progress=None, force_kg=False, on_wikidata=None):
        on_wikidata(SlowFuture())
        return dict(result, claim=claim)

    det._process_claim = fake_process
    original = hd.extract_claims_with_route
    hd.extract_claims_with_route = lambda text: (["Albert Einstein was born in 1950."], "fast_path")
    try:
        events = [json.loads(e) for e in det.analyze_stream("Albert Einstein was born in 1950.")]
    finally:
        hd.extract_claims_with_route = original
    statuses = [e["status"] for e in events]
    assert statuses.index("claim_result") < statuses.index("claim_wikidata") < statuses.index("done")
    verdict = next(e for e in events if e["status"] == "claim_result")["result"]["verification"]["label"]
    assert verdict == "contradicted"
    print("Stream ordering test passed!")


def test_arbiter_reads_wikidata_statements():
    """On escalation (forced here, as by Deep Scan) the claim's Wikidata statements join the evidence."""
    import detector.hallucination_detector as hd
    from concurrent.futures import Future

    class Retriever:
        def search(self, queries, claim=None, domain="general", k=5):
            return [{"text": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
                     "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Albert_Einstein"}]

    setup()
    graph = wg.build_graph("Albert Einstein was born in 1950.", ["Albert Einstein"])
    future = Future()
    future.set_result(graph)
    seen = {}
    original = (hd.nli_override, hd.rewrite_queries, hd.evaluate_absence)
    def fake_arbiter(claim, evidence, **kw):
        seen["evidence"] = evidence
        return "supported", None, ""
    hd.nli_override = fake_arbiter
    hd.rewrite_queries = lambda *a, **kw: None
    hd.evaluate_absence = lambda *a, **kw: ("unknown", None)
    try:
        det = hd.HallucinationDetector(Retriever())
        verification, evidence, sources, graph_json = det._verify_claim_full(
            "Albert Einstein was born in 1950.", "Albert Einstein was born in 1950.", "general", ["Albert Einstein"],
            force_kg=True, wikidata_future=future)
    finally:
        hd.nli_override, hd.rewrite_queries, hd.evaluate_absence = original
    assert "Wikidata Evidence [Albert Einstein]: Albert Einstein — date of birth: 14 March 1879" in evidence
    assert "KG Evidence [" not in evidence
    step = next(s for s in verification["trace"] if s["kind"] == "kg")
    assert step["source"] == "wikidata" and step["statements"] >= 1
    assert graph_json["source"] == "wikidata"
    print("Arbiter evidence test passed!")


if __name__ == "__main__":
    test_resolution_is_exact_or_nothing()
    test_claim_compared_with_record()
    test_no_matching_property()
    test_statement_stored_on_the_other_entity()
    test_stream_sends_graph_after_verdict()
    test_arbiter_reads_wikidata_statements()

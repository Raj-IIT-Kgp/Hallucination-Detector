import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.hallucination_detector import HallucinationDetector
from agents.knowledge_graph import _fetch_wikipedia_summary

class SeanPennMockRetriever:
    def search(self, queries, claim=None, domain="general", k=3):
        return [
            {
                "text": "Penn's early roles include the teen comedy Fast Times at Ridgemont High (1982), the crime drama Bad Boys (1983), and the war drama Casualties of War (1989).",
                "url": "https://en.wikipedia.org/wiki/Sean_Penn"
            },
            {
                "text": "Christopher Shannon Penn was an American actor. He was the younger brother of both actor Sean Penn and musician Michael Penn.",
                "url": "https://en.wikipedia.org/wiki/Chris_Penn"
            }
        ]

class MountRushmoreMockRetriever:
    def search(self, queries, claim=None, domain="general", k=3):
        return [
            {
                "text": "Mount Rushmore was sculpted by Gutzon Borglum, who oversaw the execution from 1927 to 1941 with the help of his son, Lincoln Borglum.",
                "url": "https://en.wikipedia.org/wiki/Mount_Rushmore"
            },
            {
                "text": "Ivan Houser was assistant sculptor to Gutzon Borglum in the early years. When Houser left, Gutzon's son, Lincoln, took over as assistant sculptor.",
                "url": "https://en.wikipedia.org/wiki/Lincoln_Borglum"
            }
        ]

def test_sean_penn_dilution_fix():
    print("\n--- Test 1: Sean Penn Context Dilution Fix ---")
    detector = HallucinationDetector(SeanPennMockRetriever())
    claim = "Sean Penn was in crime drama films."
    res = detector.analyze(claim)
    ver = res["report"]["claims"][0]["verification"]
    print(f"Claim: {claim}")
    print(f"Predicted Label: {ver['label']} (Confidence: {ver['confidence']:.4f})")
    assert ver["label"] == "supported", f"Expected supported, got {ver['label']}"
    print("[PASS] Sean Penn correctly identified as SUPPORTED (no dilution)!")

def test_mount_rushmore_contradiction_fix():
    print("\n--- Test 2: Mount Rushmore Calibrated Contradiction Fix ---")
    detector = HallucinationDetector(MountRushmoreMockRetriever())
    claim = "Mount Rushmore was built by Gutzon Borglum and his son."
    res = detector.analyze(claim)
    ver = res["report"]["claims"][0]["verification"]
    print(f"Claim: {claim}")
    print(f"Predicted Label: {ver['label']} (Confidence: {ver['confidence']:.4f})")
    assert ver["label"] == "supported", f"Expected supported, got {ver['label']}"
    print("[PASS] Mount Rushmore correctly identified as SUPPORTED!")

def test_global_warming_disambiguation_fix():
    print("\n--- Test 3: Global Warming Disambiguation Semantic Reranking ---")
    claim = "The abandonment of populated areas due to rising sea levels is caused by Global warming."
    page_data = _fetch_wikipedia_summary("Global Warning", context_claim=claim)
    assert page_data is not None, "Failed to fetch summary"
    print(f"Resolved Wikipedia Title: '{page_data['title']}'")
    assert "Jon Oliva" not in page_data["title"], "Disambiguation should avoid rock album!"
    print("[PASS] Global Warming disambiguated without falling into rock album trap!")

if __name__ == "__main__":
    test_sean_penn_dilution_fix()
    test_mount_rushmore_contradiction_fix()
    test_global_warming_disambiguation_fix()
    print("\n[ALL FIX VERIFICATIONS PASSED SUCCESSFULLY!]")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.hallucination_detector import HallucinationDetector

class MockRetriever:
    def search(self, queries, claim=None, domain="general", k=3):
        return [
            {"text": "The Eiffel Tower is located in Paris, France. It was designed by Gustave Eiffel.", "url": "https://example.com/paris"}
        ]

def test_pipeline():
    print("Initializing HallucinationDetector with Conformal Prediction Layer...")
    detector = HallucinationDetector(MockRetriever())
    
    assert detector.conformal_verifier is not None, "ConformalVerifier failed to initialize!"
    assert detector.conformal_verifier.q_hat is not None, "q_hat should be loaded!"
    print(f"Loaded conformal calibration: q_hat = {detector.conformal_verifier.q_hat:.4f}, n = {detector.conformal_verifier.n_samples}")

    test_input = "The Eiffel Tower is located in Paris."
    print(f"\nAnalyzing test statement: '{test_input}'")
    result = detector.analyze(test_input)

    report = result["report"]
    summary = report["summary"]
    claims = report["claims"]

    print("\n--- Summary ---")
    print(f"Total claims: {summary['total_claims']}")
    print(f"Supported: {summary['supported']}")
    print(f"Contradicted: {summary['contradicted']}")
    print(f"Unknown: {summary['unknown']}")
    print(f"Escalated claims: {summary['escalated_claims']}")
    print(f"Conformal Coverage Guarantee: {summary['conformal_coverage_guarantee']}")

    assert len(claims) > 0, "No claims extracted!"
    claim_0 = claims[0]
    ver = claim_0["verification"]

    print("\n--- Claim Verification Output ---")
    print(f"Claim: {claim_0['claim']}")
    print(f"Label: {ver['label']}")
    print(f"Confidence: {ver['confidence']:.4f}")
    print(f"Conformal Set: {ver['conformal_set']}")
    print(f"Coverage Guarantee: {ver['coverage_guarantee']}")
    print(f"Is Escalation Candidate: {ver['is_escalation_candidate']}")
    print(f"Conformal p-values: {ver['conformal_p_values']}")
    print(f"Final Decision Policy: {ver['final_decision_policy']}")

    assert "conformal_set" in ver, "conformal_set missing from verification dict!"
    assert "coverage_guarantee" in ver, "coverage_guarantee missing from verification dict!"
    assert "conformal_p_values" in ver, "conformal_p_values missing from verification dict!"
    assert "is_escalation_candidate" in ver, "is_escalation_candidate missing from verification dict!"

    print("\n[SUCCESS] Conformal Prediction Layer singleton test passed!")

class ConflictingMockRetriever:
    def search(self, queries, claim=None, domain="general", k=3):
        return [
            {"text": "The project was successfully launched in 2020.", "url": "https://sourceA.com"},
            {"text": "The project was cancelled and never launched in 2020.", "url": "https://sourceB.com"}
        ]

def test_pipeline_escalation():
    print("\n--- Testing Epistemic Contradiction & Conformal Escalation ---")
    detector = HallucinationDetector(ConflictingMockRetriever())
    test_input = "The project was launched in 2020."
    result = detector.analyze(test_input)

    ver = result["report"]["claims"][0]["verification"]
    print(f"Label: {ver['label']}")
    print(f"Conformal Set: {ver['conformal_set']}")
    print(f"Is Escalation Candidate: {ver['is_escalation_candidate']}")
    print(f"Final Decision Policy: {ver['final_decision_policy']}")

    assert ver["is_escalation_candidate"] is True, "Conflicting claim should be flagged as escalation candidate!"
    print("[SUCCESS] Escalation candidate correctly identified by conformal layer!")

if __name__ == "__main__":
    test_pipeline()
    test_pipeline_escalation()

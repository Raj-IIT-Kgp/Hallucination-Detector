import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.nli_verifier import verify_claim, verify_claims_batch

def test_verify_claim_inference_mode():
    claim = "Python was created by Guido van Rossum."
    evidence = "Guido van Rossum began working on Python in the late 1980s and released it in 1991."

    res = verify_claim(claim, evidence)
    assert isinstance(res, dict)
    assert res["label"] in ["supported", "contradicted", "unknown"]
    assert "confidence" in res
    assert len(res["probabilities"]) == 3
    assert abs(sum(res["probabilities"]) - 1.0) < 1e-4
    assert res["label"] == "supported", f"Expected supported, got {res['label']}"
    print(f"PASS: verify_claim executed under inference_mode (label: {res['label']}, confidence: {res['confidence']:.4f})")

def test_verify_claims_batch_inference_mode():
    claim = "Python was created by Guido van Rossum."
    evidence_list = [
        "Guido van Rossum designed Python as a successor to the ABC programming language.",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."
    ]

    res = verify_claims_batch(claim, evidence_list)
    assert isinstance(res, list)
    assert len(res) == 2
    for r in res:
        assert abs(sum(r["probabilities"]) - 1.0) < 1e-4
    assert res[0]["label"] == "supported"
    print(f"PASS: verify_claims_batch executed under inference_mode (batch size: {len(res)})")

if __name__ == "__main__":
    test_verify_claim_inference_mode()
    test_verify_claims_batch_inference_mode()
    print("All NLI inference_mode tests passed!")

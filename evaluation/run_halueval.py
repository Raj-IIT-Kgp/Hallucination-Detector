import sys
import os
import random
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks.halueval import load_halueval
from retrieval.web_search import WebRetriever
from detector.hallucination_detector import HallucinationDetector
from evaluation.metrics import precision, recall, f1_score
from tqdm import tqdm

def run_halueval_benchmark(num_samples=6):
    print("Loading HaluEval dataset...")
    data = load_halueval()
    
    # We structured load_halueval to return SUPPORTS and REFUTES
    supported_claims = [item for item in data if item["label"] == "SUPPORTS"]
    refuted_claims = [item for item in data if item["label"] == "REFUTES"]
    
    half = num_samples // 2
    sample_dataset = random.sample(supported_claims, half) + random.sample(refuted_claims, num_samples - half)
    random.shuffle(sample_dataset)
    
    retriever = WebRetriever()
    detector = HallucinationDetector(retriever=retriever)
    
    label_map = {
        "SUPPORTS": "supported",
        "REFUTES": "contradicted",
        "NOT ENOUGH INFO": "unknown"
    }
    
    y_true = []
    y_pred = []
    failures = []
    
    print(f"\nEvaluating FULL PIPELINE on {len(sample_dataset)} HaluEval QA samples...")
    print("WARNING: This will make live HTTP requests. Please be patient to avoid API rate limits.\n")
    
    for idx, item in enumerate(tqdm(sample_dataset)):
        claim = item["claim"]
        true_label = label_map.get(item["label"], "unknown")
        
        try:
            response = detector.analyze(claim)
            
            report_dict = response.get("report", {})
            claims_list = report_dict.get("claims", [])
            
            if not claims_list or len(claims_list) == 0:
                prediction = "unknown"
                evidence = "No report generated."
            else:
                claim_result = claims_list[0]
                prediction = claim_result["verification"]["label"]
                evidence = claim_result["evidence"]
                
            y_true.append(true_label)
            y_pred.append(prediction)
            
            if prediction != true_label:
                failures.append({
                    "claim": claim,
                    "true_label": true_label,
                    "prediction": prediction,
                    "evidence_used": evidence
                })
                
        except Exception as e:
            print(f"\nError processing claim '{claim}': {e}")
            y_true.append(true_label)
            y_pred.append("error")
            
        time.sleep(1.0)
        
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "contradicted" and p == "contradicted")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "contradicted" and p == "contradicted")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "contradicted" and p != "contradicted")
    
    p = precision(tp, fp)
    r = recall(tp, fn)
    f1 = f1_score(p, r)
    
    print("\n" + "="*50)
    print("          HALUEVAL END-TO-END RESULTS          ")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("\nHallucination Detection (Contradiction Class):")
    print(f"  Precision: {p:.2f}")
    print(f"  Recall:    {r:.2f}")
    print(f"  F1 Score:  {f1:.2f}")
    print("="*50)
    
    if failures:
        print("\n--- FAILURE LOG ---")
        for fail in failures[:10]:
            print(f"\nClaim: {fail['claim']}")
            print(f"True: {fail['true_label'].upper()} | Pred: {fail['prediction'].upper()}")
            print(f"Evidence Found: {fail['evidence_used']}")
            print("-" * 30)

if __name__ == "__main__":
    run_halueval_benchmark(num_samples=20)

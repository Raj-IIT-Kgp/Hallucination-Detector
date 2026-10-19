import sys
import os
import random
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
try:
    from bs4 import GuessedAtParserWarning
    warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
except Exception:
    pass

import torch
torch.set_num_threads(2)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks.climate_fever_dataset import load_climate_fever
from retrieval.web_search import WebRetriever
from detector.hallucination_detector import HallucinationDetector
from evaluation.metrics import precision, recall, f1_score
from tqdm import tqdm

def run_climate_fever_benchmark(num_samples=20, seed=42):
    print("=" * 60)
    print("  Climate-FEVER Benchmark Evaluation (NeurIPS, 2020)  ")
    print("  Real-world scientific, environmental & statistical claims")
    print("=" * 60)

    print("\nLoading Climate-FEVER dataset from HuggingFace...")
    ds = load_climate_fever(split='test')
    
    supported_claims = [item for item in ds if item["label"] == "SUPPORTS"]
    refuted_claims = [item for item in ds if item["label"] == "REFUTES"]
    
    print(f"Total pool available: {len(supported_claims)} SUPPORTS, {len(refuted_claims)} REFUTES")

    if seed is not None:
        random.seed(seed)
    
    half = num_samples // 2
    selected_supported = random.sample(supported_claims, min(half, len(supported_claims)))
    selected_refuted = random.sample(refuted_claims, min(num_samples - len(selected_supported), len(refuted_claims)))
    sample_dataset = selected_supported + selected_refuted
    random.shuffle(sample_dataset)
    
    retriever = WebRetriever()
    detector = HallucinationDetector(retriever=retriever)
    
    y_true = []
    y_pred = []
    failures = []
    
    print(f"\nEvaluating FULL PIPELINE on {len(sample_dataset)} Climate-FEVER samples...")
    print("WARNING: This will make live HTTP requests. Please be patient to avoid API rate limits.\n")
    for idx, item in enumerate(tqdm(sample_dataset)):
        claim = item["claim"]
        true_label = "supported" if item.get("label") == "SUPPORTS" or item.get("claim_label") == 0 else "contradicted"
        
        try:
            # Claims are atomic propositions; bypass redundant decomposition
            response = detector.analyze(claim, extract_atomic_facts=False)
            
            report_dict = response.get("report", {})
            claims_list = report_dict.get("claims", [])
            
            if not claims_list or len(claims_list) == 0:
                prediction = "unknown"
                evidence = "No report generated."
            else:
                predictions = [c["verification"]["label"] for c in claims_list]
                if "contradicted" in predictions:
                    prediction = "contradicted"
                elif "disputed" in predictions:
                    prediction = "disputed"
                elif "unknown" in predictions:
                    prediction = "unknown"
                else:
                    prediction = "supported"
                
                evidence = "\n".join([f"Fact {i+1}: {c.get('evidence', '')}" for i, c in enumerate(claims_list)])
                
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
    print("      CLIMATE-FEVER END-TO-END RESULTS      ")
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
    parser = argparse.ArgumentParser(description="Run Climate-FEVER benchmark evaluation.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to evaluate (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    
    run_climate_fever_benchmark(num_samples=args.samples, seed=args.seed)

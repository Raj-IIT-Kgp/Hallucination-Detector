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

from evaluation.benchmarks.freshqa_dataset import load_freshqa, DEFAULT_FACT_TYPES
from retrieval.web_search import WebRetriever
from detector.hallucination_detector import HallucinationDetector
from evaluation.metrics import precision, recall, f1_score
from tqdm import tqdm

def run_freshqa_benchmark(num_samples=20):
    print("=" * 60)
    print("  FreshQA Benchmark Evaluation (Google Research, 2023-2024)  ")
    print("  Fast-changing facts, dynamic web knowledge, & false premises")
    print(f"  Fact types: {', '.join(DEFAULT_FACT_TYPES)}")
    print("=" * 60)
    
    print("\nLoading FreshQA dataset...")
    data = load_freshqa(split="all")
    
    supported_claims = [item for item in data if item["label"] == "SUPPORTS"]
    refuted_claims = [item for item in data if item["label"] == "REFUTES"]
    
    print(f"Total pool available: {len(supported_claims)} SUPPORTS, {len(refuted_claims)} REFUTES")
    
    half = num_samples // 2
    selected_supported = random.sample(supported_claims, min(half, len(supported_claims)))
    selected_refuted = random.sample(refuted_claims, min(num_samples - len(selected_supported), len(refuted_claims)))
    sample_dataset = selected_supported + selected_refuted
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
    
    print(f"\nEvaluating FULL PIPELINE on {len(sample_dataset)} FreshQA claims...")
    print("WARNING: This will make live search requests. Rate-limiting throttled.\n")
    
    for idx, item in enumerate(tqdm(sample_dataset)):
        claim = item["claim"]
        true_label = label_map.get(item["label"], "unknown")
        
        try:
            response = detector.analyze(claim, extract_atomic_facts=False)
            report_dict = response.get("report", {})
            claims_list = report_dict.get("claims", [])
            
            if not claims_list:
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
                    "raw_question": item.get("raw_question", ""),
                    "fact_type": item.get("fact_type", ""),
                    "ground_truth": item.get("ground_truth", ""),
                    "true_label": true_label,
                    "prediction": prediction,
                    "evidence_used": evidence
                })
                
        except Exception as e:
            print(f"\n[!] Error processing FreshQA claim '{claim}': {e}")
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
    print("          FRESHQA BENCHMARK RESULTS            ")
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
            if fail.get('raw_question'):
                print(f"Original Question: {fail['raw_question']}")
            if fail.get('fact_type'):
                print(f"Fact Type: {fail['fact_type']}")
            if fail.get('ground_truth'):
                print(f"Ground Truth Ref: {fail['ground_truth']}")
            print(f"True: {fail['true_label'].upper()} | Pred: {fail['prediction'].upper()}")
            print(f"Evidence Found: {fail['evidence_used'][:300]}...")
            print("-" * 30)

    return {
        "accuracy": accuracy,
        "precision": p,
        "recall": r,
        "f1": f1,
        "total": len(y_true)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FreshQA benchmark")
    parser.add_argument("--samples", type=int, default=20, help="Number of random samples (default: 20)")
    args = parser.parse_args()
    
    run_freshqa_benchmark(num_samples=args.samples)

    # Results are printed/saved above. Exit immediately: interpreter shutdown joins every
    # thread-pool worker, and a DuckDuckGo/Wikipedia request stuck past its timeout would
    # otherwise keep the process alive indefinitely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

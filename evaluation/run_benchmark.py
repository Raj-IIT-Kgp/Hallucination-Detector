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

from retrieval.web_search import WebRetriever
from detector.hallucination_detector import HallucinationDetector
from evaluation.metrics import precision, recall, f1_score
from tqdm import tqdm

def run_fever_benchmark(num_samples=20):
    from evaluation.benchmarks.fever_dataset import load_fever
    
    print("=" * 60)
    print("  FEVER Benchmark Evaluation (Gold Evidence & Retrieval)  ")
    print("=" * 60)
    print("\nLoading FEVER dataset...")
    data = load_fever()
    
    supported_claims = [item for item in data if item["label"] == "SUPPORTS"]
    refuted_claims = [item for item in data if item["label"] == "REFUTES"]
    
    half = num_samples // 2
    selected_supported = random.sample(supported_claims, min(half, len(supported_claims)))
    selected_refuted = random.sample(refuted_claims, min(num_samples - len(selected_supported), len(refuted_claims)))
    sample_dataset = selected_supported + selected_refuted
    random.shuffle(sample_dataset)
    
    return _evaluate_dataset("FEVER", sample_dataset, num_samples)

def run_freshqa_benchmark_entry(num_samples=20, fact_types=None):
    from evaluation.benchmarks.freshqa_dataset import load_freshqa
    
    print("=" * 60)
    print("  FreshQA Benchmark Evaluation (Google Research, 2023-2024)  ")
    print("  Fast-changing facts, dynamic web knowledge, & false premises")
    if fact_types:
        print(f"  Filtering by fact types: {fact_types}")
    print("=" * 60)
    print("\nLoading FreshQA dataset...")
    data = load_freshqa(split="all", fact_types=fact_types)
    
    supported_claims = [item for item in data if item["label"] == "SUPPORTS"]
    refuted_claims = [item for item in data if item["label"] == "REFUTES"]
    
    half = num_samples // 2
    selected_supported = random.sample(supported_claims, min(half, len(supported_claims)))
    selected_refuted = random.sample(refuted_claims, min(num_samples - len(selected_supported), len(refuted_claims)))
    sample_dataset = selected_supported + selected_refuted
    random.shuffle(sample_dataset)
    
    return _evaluate_dataset("FreshQA", sample_dataset, num_samples)


def run_climate_fever_benchmark_entry(num_samples=20):
    from evaluation.benchmarks.climate_fever_dataset import load_climate_fever
    
    print("=" * 60)
    print("  Climate-FEVER Benchmark Evaluation (NeurIPS, 2020)  ")
    print("  Real-world scientific, environmental & statistical claims")
    print("=" * 60)
    print("\nLoading Climate-FEVER dataset...")
    data = load_climate_fever()
    
    supported_claims = [item for item in data if item["label"] == "SUPPORTS"]
    refuted_claims = [item for item in data if item["label"] == "REFUTES"]
    
    half = num_samples // 2
    selected_supported = random.sample(supported_claims, min(half, len(supported_claims)))
    selected_refuted = random.sample(refuted_claims, min(num_samples - len(selected_supported), len(refuted_claims)))
    sample_dataset = selected_supported + selected_refuted
    random.shuffle(sample_dataset)
    
    return _evaluate_dataset("Climate-FEVER", sample_dataset, num_samples)

def _evaluate_dataset(benchmark_name, sample_dataset, num_samples):
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
    
    print(f"\nEvaluating FULL PIPELINE on {len(sample_dataset)} {benchmark_name} samples...")
    print("WARNING: This will make live web/search requests. Throttling applied.\n")
    
    for idx, item in enumerate(tqdm(sample_dataset)):
        claim = item["claim"]
        true_label = label_map.get(item["label"], "unknown")
        
        # Build contextual grounding if prompt/question/speaker/date/source metadata exists
        context_parts = []
        if item.get("prompt"):
            context_parts.append(f"Prompt: {item['prompt']}")
        if item.get("raw_question"):
            context_parts.append(f"Question: {item['raw_question']}")
        if item.get("speaker"):
            context_parts.append(f"Speaker: {item['speaker']}")
        if item.get("claim_date"):
            context_parts.append(f"Date: {item['claim_date']}")
        if item.get("reporting_source"):
            context_parts.append(f"Source: {item['reporting_source']}")
        claim_context = " | ".join(context_parts) if context_parts else None

        try:
            response = detector.analyze(claim, context=claim_context, extract_atomic_facts=False)
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
                    "true_label": true_label,
                    "prediction": prediction,
                    "evidence_used": evidence
                })
                
        except Exception as e:
            print(f"\n[!] Error processing claim '{claim}': {e}")
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
    print(f"      {benchmark_name.upper()} BENCHMARK RESULTS")
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
            print(f"Evidence Found: {fail['evidence_used'][:300]}...")
            print("-" * 30)

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "precision": p,
        "recall": r,
        "f1": f1,
        "total": len(y_true)
    }

def main():
    parser = argparse.ArgumentParser(description="Run Hallucination Detector Benchmarks")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fever",
        choices=["fever", "freshqa", "climate_fever", "all"],
        help="Benchmark dataset to run (default: fever)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of random samples to evaluate (default: 20)"
    )
    parser.add_argument(
        "--stable",
        action="store_true",
        help="Filter for slow-changing and never-changing claims (FreshQA only, avoids 2023 temporal drift)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    
    freshqa_fact_types = ["slow-changing", "never-changing"] if args.stable else None

    if args.dataset == "fever":
        run_fever_benchmark(num_samples=args.samples)
    elif args.dataset == "freshqa":
        run_freshqa_benchmark_entry(num_samples=args.samples, fact_types=freshqa_fact_types)
    elif args.dataset == "climate_fever":
        run_climate_fever_benchmark_entry(num_samples=args.samples)
    elif args.dataset == "all":
        results = []
        for fn, name, kwargs in [
            (run_fever_benchmark, "FEVER", {}),
            (run_freshqa_benchmark_entry, "FreshQA", {"fact_types": freshqa_fact_types}),
            (run_climate_fever_benchmark_entry, "Climate-FEVER", {})
        ]:
            print(f"\n>>> Running {name} Benchmark ({args.samples} samples)...")
            res = fn(num_samples=args.samples, **kwargs)
            results.append(res)
            
        print("\n" + "="*65)
        print("                 BENCHMARK SUITE SUMMARY                 ")
        print("="*65)
        print(f"{'Benchmark':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
        print("-" * 65)
        for r in results:
            print(f"{r['benchmark']:<15} | {r['accuracy']:<9.1%} | {r['precision']:<10.2f} | {r['recall']:<10.2f} | {r['f1']:<10.2f}")
        print("="*65)

if __name__ == "__main__":
    main()

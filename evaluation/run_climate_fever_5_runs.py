import sys
import os
import random
import time
import json
import argparse
import warnings
import math

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


def run_5_sets_climate_fever(samples_per_set=20, num_sets=5, seed=42):
    print("=" * 65)
    print("      Climate-FEVER 5-Set Cross-Validation Benchmark Execution      ")
    print("      Evaluating peer-reviewed science, climate & statistical claims")
    print(f"      Configuration: {num_sets} sets × {samples_per_set} claims = {num_sets * samples_per_set} total claims")
    print("=" * 65)

    print("\nLoading Climate-FEVER dataset from HuggingFace...")
    data = load_climate_fever(split="test")

    supported = [item for item in data if item["label"] == "SUPPORTS"]
    refuted = [item for item in data if item["label"] == "REFUTES"]

    print(f"Total available: {len(supported)} SUPPORTS, {len(refuted)} REFUTES")

    half = samples_per_set // 2
    required_supp = half * num_sets
    required_ref = (samples_per_set - half) * num_sets

    if len(supported) < required_supp or len(refuted) < required_ref:
        raise ValueError("Not enough samples to construct non-overlapping sets.")

    # Shuffle with deterministic seed for non-overlapping stratified partition
    rng = random.Random(seed)
    shuffled_supp = list(supported)
    shuffled_ref = list(refuted)
    rng.shuffle(shuffled_supp)
    rng.shuffle(shuffled_ref)

    sets = []
    for s_idx in range(num_sets):
        s_supp = shuffled_supp[s_idx * half : (s_idx + 1) * half]
        s_ref = shuffled_ref[s_idx * (samples_per_set - half) : (s_idx + 1) * (samples_per_set - half)]
        batch = s_supp + s_ref
        rng.shuffle(batch)
        sets.append(batch)

    print("\nInitializing WebRetriever & HallucinationDetector (Single Instance Re-use)...")
    retriever = WebRetriever()
    detector = HallucinationDetector(retriever=retriever)

    set_results = []
    all_y_true = []
    all_y_pred = []
    all_failures = []

    for set_idx, batch in enumerate(sets):
        set_num = set_idx + 1
        print(f"\n" + "-" * 60)
        print(f"  EXECUTING SET {set_num}/{num_sets} ({len(batch)} claims)")
        print("-" * 60)

        y_true = []
        y_pred = []
        failures = []

        for item in tqdm(batch, desc=f"Set {set_num}"):
            claim = item["claim"]
            true_label = "supported" if item.get("label") == "SUPPORTS" or item.get("claim_label") == 0 else "contradicted"

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
                    fail_entry = {
                        "set": set_num,
                        "claim": claim,
                        "true_label": true_label,
                        "prediction": prediction,
                        "evidence_used": evidence[:350]
                    }
                    failures.append(fail_entry)
                    all_failures.append(fail_entry)

            except Exception as e:
                print(f"\n[!] Error processing claim '{claim}': {e}")
                y_true.append(true_label)
                y_pred.append("error")

            time.sleep(0.5)

        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        acc = correct / len(y_true) if y_true else 0.0

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == "contradicted" and p == "contradicted")
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != "contradicted" and p == "contradicted")
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == "contradicted" and p != "contradicted")

        p = precision(tp, fp)
        r = recall(tp, fn)
        f1 = f1_score(p, r)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        res = {
            "set": f"Set {set_num}",
            "evaluated_claims": len(y_true),
            "correct_claims": correct,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        set_results.append(res)

        print(f"\n>>> Set {set_num} Results: Accuracy={acc:.2%} ({correct}/{len(y_true)}), P={p:.2f}, R={r:.2f}, F1={f1:.2f}")

    # Pooled calculations
    total_eval = len(all_y_true)
    total_correct = sum(1 for t, p in zip(all_y_true, all_y_pred) if t == p)
    pooled_acc = total_correct / total_eval if total_eval else 0.0

    pooled_tp = sum(1 for t, p in zip(all_y_true, all_y_pred) if t == "contradicted" and p == "contradicted")
    pooled_fp = sum(1 for t, p in zip(all_y_true, all_y_pred) if t != "contradicted" and p == "contradicted")
    pooled_fn = sum(1 for t, p in zip(all_y_true, all_y_pred) if t == "contradicted" and p != "contradicted")

    pooled_p = precision(pooled_tp, pooled_fp)
    pooled_r = recall(pooled_tp, pooled_fn)
    pooled_f1 = f1_score(pooled_p, pooled_r)

    # Standard error & 95% Confidence Interval
    se = math.sqrt((pooled_acc * (1.0 - pooled_acc)) / total_eval) if total_eval > 0 else 0.0
    ci_margin = 1.96 * se
    ci_lower = max(0.0, pooled_acc - ci_margin)
    ci_upper = min(1.0, pooled_acc + ci_margin)

    print("\n" + "=" * 70)
    print("                 CLIMATE-FEVER 5-SET POOLED SUMMARY                  ")
    print("=" * 70)
    print(f"{'Batch Run':<12} | {'Evaluated':<10} | {'Correct':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 70)
    for r in set_results:
        print(f"{r['set']:<12} | {r['evaluated_claims']:<10} | {r['correct_claims']:<8} | {r['accuracy']:<9.2%} | {r['precision']:<10.2f} | {r['recall']:<8.2f} | {r['f1']:<8.2f}")
    print("-" * 70)
    print(f"{'Pooled Total':<12} | {total_eval:<10} | {total_correct:<8} | {pooled_acc:<9.2%} | {pooled_p:<10.2f} | {pooled_r:<8.2f} | {pooled_f1:<8.2f}")
    print("=" * 70)
    print(f"Standard Error (SE): {se:.4f} ({se:.2%})")
    print(f"Asymptotic 95% CI:   {pooled_acc:.2%} ± {ci_margin:.2%} ([{ci_lower:.2%}, {ci_upper:.2%}])")
    print(f"Total Failures:      {len(all_failures)} / {total_eval}")

    output_payload = {
        "set_results": set_results,
        "pooled": {
            "total_evaluated": total_eval,
            "total_correct": total_correct,
            "accuracy": pooled_acc,
            "precision": pooled_p,
            "recall": pooled_r,
            "f1": pooled_f1,
            "standard_error": se,
            "ci_margin_95": ci_margin,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper
        },
        "failures": all_failures
    }

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "climate_fever_5_runs_results.json")
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved raw payload and failure logs to: {out_file}")

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Climate-FEVER 5-set cross-validation benchmark.")
    parser.add_argument("--samples-per-set", type=int, default=20, help="Samples per set (default: 20)")
    parser.add_argument("--sets", type=int, default=5, help="Number of sets to run (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    run_5_sets_climate_fever(samples_per_set=args.samples_per_set, num_sets=args.sets, seed=args.seed)

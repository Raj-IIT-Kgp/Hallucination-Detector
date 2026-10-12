import sys
import os
import time
import random
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks.fever_dataset import load_fever
from verification.nli_verifier import verify_claim
from agents.critic_agent import nli_override


def run_debate_ablation(num_samples=15):
    """
    Ablation Study: Quantifies the marginal utility of Multi-Agent Debate.
    Compares:
      1. Baseline (No Debate): Strict NLI + Conservative Abstention (defaults to 'unknown')
      2. Treatment (With Debate): NLI + Multi-Agent Debate (Defense, Prosecution, Judge)
    """
    print("=" * 75)
    print("ABLATION STUDY: MULTI-AGENT DEBATE UTILITY & COST ANALYSIS")
    print("=" * 75)
    
    data = load_fever()
    supported = [item for item in data if item["label"] == "SUPPORTS" and len(item.get("evidence_text", "").strip()) > 10]
    refuted = [item for item in data if item["label"] == "REFUTES" and len(item.get("evidence_text", "").strip()) > 10]

    half = num_samples // 2
    samples = random.sample(supported, half) + random.sample(refuted, num_samples - half)
    random.seed(42)
    random.shuffle(samples)

    label_map = {"SUPPORTS": "supported", "REFUTES": "contradicted"}

    # Metrics trackers
    baseline_correct = 0
    debate_correct = 0
    total_escalated = 0
    flips_to_correct = 0
    flips_to_incorrect = 0
    
    baseline_latencies = []
    debate_latencies = []
    api_calls_baseline = 0
    api_calls_debate = 0

    print(f"Evaluating {len(samples)} claims across both conditions...\n")

    for item in tqdm(samples):
        claim = item["claim"]
        evidence = item.get("evidence_text", "")
        y_true = label_map.get(item["label"], "unknown")

        # --- CONDITION A: Baseline (No Debate) ---
        t0 = time.time()
        nli_res = verify_claim(claim, evidence)
        t_base = time.time() - t0
        baseline_latencies.append(t_base)

        baseline_pred = nli_res["label"]
        if baseline_pred == y_true:
            baseline_correct += 1

        # --- CONDITION B: With Multi-Agent Debate (Triggered on 'unknown' or uncertainty) ---
        t0 = time.time()
        if nli_res["label"] == "unknown":
            total_escalated += 1
            api_calls_debate += 3  # Defense + Prosecution + Judge
            debate_pred = nli_override(claim, evidence, domain="general")
            
            # Check if debate helped or hurt
            if debate_pred == y_true and baseline_pred != y_true:
                flips_to_correct += 1
            elif debate_pred != y_true and baseline_pred == y_true:
                flips_to_incorrect += 1
        else:
            debate_pred = baseline_pred
            
        t_deb = time.time() - t0 + t_base
        debate_latencies.append(t_deb)

        if debate_pred == y_true:
            debate_correct += 1

    n = len(samples)
    base_acc = baseline_correct / n
    deb_acc = debate_correct / n
    correction_rate = (flips_to_correct / total_escalated) if total_escalated > 0 else 0.0
    degradation_rate = (flips_to_incorrect / total_escalated) if total_escalated > 0 else 0.0

    results = {
        "num_evaluated_claims": n,
        "baseline_accuracy_no_debate": round(base_acc, 4),
        "treatment_accuracy_with_debate": round(deb_acc, 4),
        "accuracy_delta": round(deb_acc - base_acc, 4),
        "total_claims_escalated": total_escalated,
        "successful_corrections": flips_to_correct,
        "correction_rate": round(correction_rate, 4),
        "degradation_rate": round(degradation_rate, 4),
        "avg_latency_no_debate_sec": round(float(sum(baseline_latencies) / n), 3),
        "avg_latency_with_debate_sec": round(float(sum(debate_latencies) / n), 3),
        "llm_api_calls_baseline": api_calls_baseline,
        "llm_api_calls_debate": api_calls_debate
    }

    print("\n" + "=" * 75)
    print("ABLATION STUDY RESULTS:")
    print("=" * 75)
    print(f"Total Claims Evaluated:          {n}")
    print(f"Accuracy WITHOUT Debate:         {base_acc * 100:.2f}%")
    print(f"Accuracy WITH Debate:            {deb_acc * 100:.2f}%  (Delta: +{(deb_acc - base_acc) * 100:.2f}%)")
    print(f"Claims Escalated to Debate:      {total_escalated} ({total_escalated / n * 100:.1f}%)")
    print(f"Successful Corrections (Wrong->Right): {flips_to_correct} ({correction_rate * 100:.1f}%)")
    print(f"Sycophancy/Degradations (Right->Wrong):{flips_to_incorrect} ({degradation_rate * 100:.1f}%)")
    print(f"Avg Latency WITHOUT Debate:      {results['avg_latency_no_debate_sec']:.3f} s / claim")
    print(f"Avg Latency WITH Debate:         {results['avg_latency_with_debate_sec']:.3f} s / claim")
    print(f"Total LLM API Calls (Debate):    {api_calls_debate} calls")
    print("=" * 75)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_debate_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ablation results to: {save_path}")

    return results


if __name__ == "__main__":
    run_debate_ablation()

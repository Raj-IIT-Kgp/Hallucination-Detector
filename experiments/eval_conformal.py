import sys
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks.fever_dataset import load_fever
from verification.nli_verifier import tokenizer, model, labels as nli_labels
from statistical_layer.conformal_predictor import ConformalVerifier


def run_conformal_experiment(num_samples=50, calib_ratio=0.5, batch_size=8):
    print("=" * 70)
    print("STATISTICAL EXPERIMENT: SPLIT CONFORMAL PREDICTION EVALUATION")
    print("=" * 70)
    print("Loading FEVER gold evidence dataset...")
    data = load_fever()

    supported = [item for item in data if item["label"] == "SUPPORTS" and len(item.get("evidence_text", "").strip()) > 10]
    refuted = [item for item in data if item["label"] == "REFUTES" and len(item.get("evidence_text", "").strip()) > 10]

    half = num_samples // 2
    sampled_data = random.sample(supported, min(half, len(supported))) + random.sample(refuted, min(half, len(refuted)))
    random.seed(42)
    random.shuffle(sampled_data)

    label_map = {
        "SUPPORTS": "supported",
        "REFUTES": "contradicted",
        "NOT ENOUGH INFO": "unknown"
    }

    print(f"Extracting NLI probability distributions for {len(sampled_data)} verified samples (Batch size={batch_size})...")
    prob_vectors = []
    y_true_labels = []

    # Batched inference for fast computation
    claims = [item["claim"] for item in sampled_data]
    evidences = [item.get("evidence_text", "") for item in sampled_data]
    true_labels = [label_map.get(item["label"], "unknown") for item in sampled_data]

    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    for i in tqdm(range(0, len(claims), batch_size)):
        batch_claims = claims[i : i + batch_size]
        batch_evidences = evidences[i : i + batch_size]
        batch_true = true_labels[i : i + batch_size]

        inputs = tokenizer(
            batch_evidences,
            batch_claims,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        for p, t in zip(probs, batch_true):
            prob_vectors.append(p.tolist())
            y_true_labels.append(t)

    # Split into Calibration and Test sets
    n_total = len(y_true_labels)
    n_calib = int(n_total * calib_ratio)
    
    calib_probs = prob_vectors[:n_calib]
    calib_labels = y_true_labels[:n_calib]

    test_probs = prob_vectors[n_calib:]
    test_labels = y_true_labels[n_calib:]

    print(f"\nDataset Partition: {n_calib} Calibration Samples | {len(test_labels)} Test Samples")

    # Evaluate across a spectrum of alpha levels
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    nominal_coverages = [(1.0 - a) for a in alphas]
    empirical_coverages = []
    avg_set_sizes = []
    ambiguity_rates = []

    print("\n" + "=" * 70)
    print(f"{'Alpha':<8} | {'Nominal Cov':<12} | {'Empirical Cov':<14} | {'Avg Set Size':<12} | {'q_hat':<8}")
    print("-" * 70)

    for a in alphas:
        verifier = ConformalVerifier(alpha=a)
        q_hat = verifier.calibrate(calib_probs, calib_labels)

        # Test phase
        covered_count = 0
        set_sizes = []
        ambiguous_count = 0

        for p_vec, y_true in zip(test_probs, test_labels):
            pred_res = verifier.predict_set(p_vec)
            c_set = pred_res["conformal_set"]
            
            if y_true in c_set:
                covered_count += 1
            if pred_res["is_ambiguous"]:
                ambiguous_count += 1
            set_sizes.append(pred_res["set_size"])

        emp_cov = covered_count / len(test_labels)
        avg_size = np.mean(set_sizes)
        amb_rate = ambiguous_count / len(test_labels)

        empirical_coverages.append(emp_cov)
        avg_set_sizes.append(avg_size)
        ambiguity_rates.append(amb_rate)

        print(f"{a:<8.2f} | {f'{int((1-a)*100)}%':<12} | {f'{emp_cov*100:.2f}%':<14} | {avg_size:<12.2f} | {q_hat:<8.4f}")

    print("=" * 70)

    # Save calibration model for production use at alpha=0.10 (90% coverage)
    calib_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformal_calibration.json")
    prod_verifier = ConformalVerifier(alpha=0.10)
    prod_verifier.calibrate(calib_probs, calib_labels, save_path=calib_save_path)
    print(f"\n[Production Calibration] Saved alpha=0.10 calibration state to: {calib_save_path}")

    # Generate Statistical Plots
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Plot 1: Empirical vs Nominal Coverage (Calibration Reliability)
    plt.figure(figsize=(7, 6))
    plt.plot(nominal_coverages, empirical_coverages, marker='o', color='#1f77b4', linewidth=2.5, label='Empirical Conformal Coverage')
    plt.plot([0.5, 0.95], [0.5, 0.95], linestyle='--', color='#d62728', label='Theoretical Guarantee ($y = x$)')
    plt.title('Finite-Sample Coverage Validity (Split Conformal Prediction)', fontsize=13, fontweight='bold')
    plt.xlabel('Nominal Coverage Target ($1 - \\alpha$)', fontsize=11)
    plt.ylabel('Empirical Coverage on Test Set', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    cov_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformal_coverage_plot.png")
    plt.savefig(cov_plot_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] Coverage verification plot saved: {cov_plot_path}")

    # Plot 2: Efficiency (Set Size vs Nominal Coverage)
    plt.figure(figsize=(7, 6))
    plt.plot(nominal_coverages, avg_set_sizes, marker='s', color='#2ca02c', linewidth=2.5)
    plt.title('Prediction Set Efficiency vs. Target Coverage', fontsize=13, fontweight='bold')
    plt.xlabel('Nominal Coverage Target ($1 - \\alpha$)', fontsize=11)
    plt.ylabel('Average Conformal Set Size $|\\mathcal{C}(X)|$', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    size_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformal_set_size_plot.png")
    plt.savefig(size_plot_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] Efficiency plot saved: {size_plot_path}")

    return {
        "alphas": alphas,
        "nominal_coverages": nominal_coverages,
        "empirical_coverages": empirical_coverages,
        "avg_set_sizes": avg_set_sizes
    }


if __name__ == "__main__":
    run_conformal_experiment()

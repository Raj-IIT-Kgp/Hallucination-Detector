"""
Calibration of the NLI probabilities on the published fifteen-set evaluation (no LLM calls).

When the NLI model gives a verdict with probability 0.95, is it right 95% of the time? This script measures it
on the 600 benchmark claims and fits a temperature T (statistical_layer/calibration.py) that makes it so.

  population  claims whose final NLI pass gave a definite verdict (SUPPORTED or CONTRADICTED): the benchmarks'
              gold labels are only ever SUPPORTED or CONTRADICTED, so a calibrated "unknown" has no target.
              The 9 source-conflict claims carry no logits and are excluded.
  metrics     negative log-likelihood (NLL), multi-class Brier score, and top-label ECE with 10 equal-mass bins
  honesty     T is cross-fitted by set: each set is scored with the T fitted on the other nine, so no claim's
              calibrated probability depends on its own label. A paired bootstrap over claims gives 95%
              intervals for the improvements.
  output      experiments/calibration.json (T fitted on all claims, read by the pipeline) and
              experiments/calibration_reliability.png (reliability diagram)

The temperature only rescales the reported probabilities: the verdict (arg max), the decisiveness threshold
and the conformal layer still use the raw probabilities, so the measured benchmark results are unchanged.

usage: python experiments/calibration.py
"""
import sys
import os
import json

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statistical_layer.calibration import TemperatureScaler, expected_calibration_error, brier_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "evaluation")
OUT_DIR = os.path.join(ROOT, "experiments")
DATASETS = ("fever", "freshqa")
LABELS = ["contradicted", "unknown", "supported"]
BINS = 10
BOOTSTRAP = 10000
SEED = 0


def load_rows():
    rows = []
    for ds in DATASETS:
        with open(os.path.join(EVAL_DIR, f"{ds}_5_runs_claims.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                v = (r.get("result") or {}).get("verification") or {}
                nli = [s for s in v.get("trace") or [] if s.get("kind") == "nli"]
                if not nli or v.get("logits") is None or nli[-1]["label"] == "unknown":
                    continue
                rows.append({"dataset": ds, "set": r["set"], "logits": v["logits"],
                             "y": LABELS.index(r["gold"]), "route": r["decision_policy"]})
    return rows


def metrics(probs, y):
    probs = np.asarray(probs)
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    ece, table = expected_calibration_error(conf, correct, BINS)
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-300, None))))
    return {"nll": nll, "brier": brier_score(probs, y), "ece": ece, "mean_confidence": float(conf.mean()),
            "accuracy": float(correct.mean()), "reliability": [{"n": n, "confidence": c, "accuracy": a} for n, c, a in table]}


def per_claim(probs, y):
    """Per-claim NLL and Brier terms, and (confidence, correct), for the paired bootstrap."""
    probs = np.asarray(probs)
    onehot = np.eye(probs.shape[1])[y]
    return (-np.log(np.clip(probs[np.arange(len(y)), y], 1e-300, None)), ((probs - onehot) ** 2).sum(1),
            probs.max(1), probs.argmax(1) == y)


def bootstrap(raw, cal, y, rng):
    """95% percentile intervals of (calibrated - raw) for NLL, Brier and ECE, resampling claims with replacement."""
    a, b = per_claim(raw, y), per_claim(cal, y)
    n = len(y)
    out = {"nll": [], "brier": [], "ece": []}
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, n, n)
        out["nll"].append(b[0][i].mean() - a[0][i].mean())
        out["brier"].append(b[1][i].mean() - a[1][i].mean())
        out["ece"].append(expected_calibration_error(b[2][i], b[3][i], BINS)[0]
                          - expected_calibration_error(a[2][i], a[3][i], BINS)[0])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def auroc(score, correct):
    """Probability that a random correct verdict has a higher score than a random wrong one (ties count half)."""
    score, correct = np.asarray(score), np.asarray(correct, dtype=bool)
    pos, neg = score[correct], score[~correct]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def platt_fit(x, t):
    """Logistic regression of correctness t on one feature x: P(correct) = sigma(a x + b), by Newton's method."""
    X = np.column_stack([x, np.ones_like(x)])
    w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2), X.T @ (p - t))
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def top_label_comparison(z, y, set_ids, raw, cal):
    """
    Compare temperature scaling with Platt scaling of the verdict's confidence alone (two parameters:
    P(correct) = sigma(a logit(conf) + b)), both cross-fitted by set, on the top-label probability.
    """
    conf = raw.max(1)
    correct = (raw.argmax(1) == y).astype(float)
    x = np.log(np.clip(conf, 1e-12, 1 - 1e-12) / np.clip(1 - conf, 1e-12, None))
    platt = np.zeros(len(y))
    for s in sorted(set(set_ids)):
        held = set_ids == s
        a, b = platt_fit(x[~held], correct[~held])
        platt[held] = 1 / (1 + np.exp(-(a * x[held] + b)))

    def binary(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return {"ece": expected_calibration_error(p, correct, BINS)[0],
                "nll": float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p))),
                "brier": float(np.mean((p - correct) ** 2))}
    return {"raw": binary(conf), "temperature": binary(cal.max(1)), "platt": binary(platt),
            "auroc_confidence": auroc(conf, correct), "auroc_scaled_confidence": auroc(cal.max(1), correct)}


def plot(raw, cal, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    for m, label, color, marker in ((raw, "Raw NLI probabilities", "#d62728", "o"),
                                    (cal, "Temperature-scaled (cross-fitted)", "#1f77b4", "s")):
        r = m["reliability"]
        plt.plot([b["confidence"] for b in r], [b["accuracy"] for b in r], marker=marker, color=color, linewidth=2,
                 label=f"{label}, ECE = {100 * m['ece']:.1f}%")
    lo = min(min(b["confidence"], b["accuracy"]) for m in (raw, cal) for b in m["reliability"])
    lo = max(0.0, np.floor(lo * 10) / 10)
    plt.plot([lo, 1], [lo, 1], linestyle="--", color="#555555", label="Perfect calibration ($y = x$)")
    plt.title(f"Reliability of the NLI Verdict Confidence ({sum(b['n'] for b in raw['reliability'])} benchmark claims)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean confidence in bin (10 equal-mass bins)", fontsize=11)
    plt.ylabel("Accuracy in bin", fontsize=11)
    plt.xlim(lo, 1.0)
    plt.ylim(lo, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    rows = load_rows()
    z = np.array([r["logits"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows])
    set_ids = np.array([r["set"] for r in rows])

    raw = TemperatureScaler.softmax(z)

    # cross-fitted by set: set s of both benchmarks is scored with the T fitted on the other nine sets
    folds = {}
    cal = np.zeros_like(raw)
    for s in sorted(set(set_ids)):
        held = set_ids == s
        t = TemperatureScaler.fit(z[~held], y[~held])
        folds[int(s)] = t
        cal[held] = TemperatureScaler.softmax(z[held], t)

    t_all = TemperatureScaler.fit(z, y)
    rng = np.random.default_rng(SEED)
    m_raw, m_cal = metrics(raw, y), metrics(cal, y)
    m_in = metrics(TemperatureScaler.softmax(z, t_all), y)

    # the fast path, where the NLI verdict is final and its confidence is what the user sees
    fast = np.array([r["route"] == "conformal_singleton" for r in rows])
    by_route = {"fast_path": {"claims": int(fast.sum()), "raw": metrics(raw[fast], y[fast]),
                              "calibrated": metrics(cal[fast], y[fast])}}

    report = {
        "temperature": t_all,
        "claims": len(rows),
        "excluded": "claims whose final NLI verdict is UNKNOWN, and source-conflict claims (no logits)",
        "bins": BINS,
        "crossfit_by_set": {"temperature_per_fold": folds, "raw": m_raw, "calibrated": m_cal,
                            "bootstrap95_calibrated_minus_raw": bootstrap(raw, cal, y, rng)},
        "in_sample": m_in,
        "by_route": by_route,
        "top_label_crossfit": top_label_comparison(z, y, set_ids, raw, cal),
        "nll_curve": [{"T": t, "nll": TemperatureScaler.nll(z, y, t)} for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)],
    }
    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    plot(m_raw, m_cal, os.path.join(OUT_DIR, "calibration_reliability.png"))

    pc = lambda x: f"{100 * x:.1f}%"
    print(f"{len(rows)} claims with a definite NLI verdict; fitted T = {t_all:.3f} "
          f"(per fold {min(folds.values()):.3f}-{max(folds.values()):.3f})")
    for name, m in (("raw", m_raw), ("cross-fitted", m_cal), ("in-sample", m_in)):
        print(f"  {name:13} NLL {m['nll']:.4f}  Brier {m['brier']:.4f}  ECE {pc(m['ece'])}  "
              f"mean confidence {pc(m['mean_confidence'])}  accuracy {pc(m['accuracy'])}")
    ci = report["crossfit_by_set"]["bootstrap95_calibrated_minus_raw"]
    print("  bootstrap 95% of (cross-fitted - raw): " + ", ".join(f"{k} [{v[0]:+.4f}, {v[1]:+.4f}]" for k, v in ci.items()))
    print("  reliability (raw -> cross-fitted):")
    for a, b in zip(m_raw["reliability"], m_cal["reliability"]):
        print(f"    n={a['n']:3}  conf {pc(a['confidence'])} acc {pc(a['accuracy'])}   |   conf {pc(b['confidence'])} acc {pc(b['accuracy'])}")
    fp = by_route["fast_path"]
    print(f"  fast path ({fp['claims']} claims): ECE {pc(fp['raw']['ece'])} -> {pc(fp['calibrated']['ece'])}, "
          f"mean confidence {pc(fp['raw']['mean_confidence'])} -> {pc(fp['calibrated']['mean_confidence'])}, accuracy {pc(fp['raw']['accuracy'])}")
    tl = report["top_label_crossfit"]
    print(f"  top-label, cross-fitted (P(verdict correct)); AUROC of the confidence {tl['auroc_confidence']:.3f} raw, "
          f"{tl['auroc_scaled_confidence']:.3f} scaled:")
    for k in ("raw", "temperature", "platt"):
        print(f"    {k:11} ECE {pc(tl[k]['ece'])}  NLL {tl[k]['nll']:.4f}  Brier {tl[k]['brier']:.4f}")


if __name__ == "__main__":
    main()

"""
Ablations, selective prediction and interval estimates for precision, recall and F1, derived from the per-claim
records of the published fifteen-set evaluation (evaluation/<dataset>_5_runs_claims.jsonl). No LLM is called and
no model is run: every verdict below is read from, or reconstructed exactly from, each claim's computation trace.

  1. ablations: the verdict every claim would have received with parts of the cascade removed. The NLI label in
     the trace is the label the scoring rule gives before any LLM review (for a lone or contested contradiction,
     the highest-confidence decisive snippet, as in the original rule), so removing an LLM stage replaces the
     final verdict of the claims that stage decided by that label. Each variant is compared with the full
     pipeline by the exact McNemar test; the pooled comparisons form their own family, Holm-adjusted within
     it (separate from the eight tests of §4.5 of pipeline.md, whose adjusted p-values are unchanged).
       - always SUPPORTED: the trivial detector that never flags a hallucination
       - NLI only, first pass: the first NLI verdict, no query rewrite, no LLM verdict
       - NLI only: the final NLI verdict (after the query rewrite), no LLM verdict
       - no lightweight review: lone and contested contradictions accepted without review
       - no full escalation: escalated claims keep their NLI verdict (no Wikidata evidence, no arbitration)
  2. selective prediction (§4.3 of pipeline.md): the claims with a definite final verdict are ranked by the NLI
     probability of that verdict, and the detector abstains on the least confident first. The risk-coverage
     curve, its area (AURC, with a bootstrap 95% interval), the excess over the oracle ranking, and the
     selective risk at fixed coverages are reported per benchmark.
  3. intervals for the hallucination class: Wilson 95% intervals for precision and recall, and a bootstrap 95%
     percentile interval for F1, per benchmark and pooled.

usage: python experiments/ablation_selective.py
"""
import sys
import os
import json

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import wilson_interval
from statistical_tests import load_records, last, mcnemar_exact_p, holm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "evaluation")
OUT_DIR = os.path.join(ROOT, "experiments")
DATASETS = ("fever", "freshqa")
LABELS = ["contradicted", "unknown", "supported"]
COVERAGES = (0.8, 0.9)
BOOTSTRAP = 10000
SEED = 0

REVIEW, ESCALATION = "contradiction_arbitration", "multi_agent_debate"


# ---------------------------------------------------------------- per-claim rows

def nli_steps(rec):
    trace = ((rec.get("result") or {}).get("verification") or {}).get("trace") or []
    return [s for s in trace if s.get("kind") == "nli"]


def calls(rec, tier):
    return sum(1 for e in rec.get("llm_log") or [] if e.get("tier") == tier)


def load_rows():
    rows = []
    for ds in DATASETS:
        for rec in load_records(os.path.join(EVAL_DIR, f"{ds}_5_runs_claims.jsonl")):
            steps, queries = nli_steps(rec), last(rec, "queries")
            if not steps:
                sys.exit(f"record without an NLI step: {rec['claim']!r}")
            final_nli = steps[-1]
            route = rec["decision_policy"]
            heavy = calls(rec, "heavy")
            rows.append({
                "dataset": ds, "gold": rec["gold"], "route": route,
                "verdict": {
                    "always_supported": "supported",
                    "nli_first_pass": steps[0]["label"],
                    "nli_only": final_nli["label"],
                    "no_review": final_nli["label"] if route == REVIEW else rec["prediction"],
                    "no_escalation": final_nli["label"] if route == ESCALATION else rec["prediction"],
                    "full": rec["prediction"],
                },
                "llm_calls": {
                    "always_supported": 0,
                    "nli_first_pass": int(bool(queries) and queries.get("route") == "llm"),
                    "nli_only": calls(rec, "light"),
                    "no_review": rec["llm_calls"] - (heavy if route == REVIEW else 0),
                    "no_escalation": rec["llm_calls"] - (heavy if route == ESCALATION else 0),
                    "full": rec["llm_calls"],
                },
                # NLI probability of the final verdict (the vector that reached the conformal layer)
                "score": dict(zip(LABELS, final_nli["probabilities"])).get(rec["prediction"], 0.0),
            })
    return rows


# ---------------------------------------------------------------- metrics

def confusion(gold, pred):
    """Hallucination (gold contradicted) is the positive class; UNKNOWN on a false claim is a missed hallucination
    and on a true claim only an accuracy error (§4.1 of pipeline.md)."""
    gold, pred = np.asarray(gold), np.asarray(pred)
    pos = gold == "contradicted"
    tp = int(np.sum(pos & (pred == "contradicted")))
    fn = int(np.sum(pos & (pred != "contradicted")))
    fp = int(np.sum(~pos & (pred == "contradicted")))
    tn = int(np.sum(~pos & (pred == "supported")))
    return tp, fp, fn, tn


def f1_score(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def summarise(rows, variant):
    gold = [r["gold"] for r in rows]
    pred = [r["verdict"][variant] for r in rows]
    full = [r["verdict"]["full"] for r in rows]
    correct = sum(g == p for g, p in zip(gold, pred))
    tp, fp, fn, tn = confusion(gold, pred)
    b = sum(f == g and p != g for g, p, f in zip(gold, pred, full))      # full pipeline right only
    c = sum(p == g and f != g for g, p, f in zip(gold, pred, full))      # variant right only
    return {
        "claims": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "wilson95": wilson_interval(correct, len(rows)),
        "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn), "f1": f1_score(tp, fp, fn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unknown": sum(p == "unknown" for p in pred),
        "llm_calls_per_claim": sum(r["llm_calls"][variant] for r in rows) / len(rows),
        "vs_full": {"full_right_only": b, "variant_right_only": c, "mcnemar_exact_p": mcnemar_exact_p(b, c)},
    }


def class_intervals(rows, rng):
    """Wilson intervals for precision and recall; percentile bootstrap interval for F1 (claims resampled)."""
    gold = np.array([r["gold"] for r in rows])
    pred = np.array([r["verdict"]["full"] for r in rows])
    tp, fp, fn, tn = confusion(gold, pred)
    pos, flagged = gold == "contradicted", pred == "contradicted"
    tp_i, fp_i = (pos & flagged).astype(int), (~pos & flagged).astype(int)
    fn_i = (pos & ~flagged).astype(int)
    idx = rng.integers(0, len(rows), size=(BOOTSTRAP, len(rows)))
    s_tp, s_fp, s_fn = tp_i[idx].sum(1), fp_i[idx].sum(1), fn_i[idx].sum(1)
    f1s = np.where(s_tp > 0, 2 * s_tp / np.maximum(2 * s_tp + s_fp + s_fn, 1), 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp), "precision_wilson95": wilson_interval(tp, tp + fp),
        "recall": tp / (tp + fn), "recall_wilson95": wilson_interval(tp, tp + fn),
        "f1": f1_score(tp, fp, fn), "f1_bootstrap95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


# ---------------------------------------------------------------- selective prediction

def risk_coverage(score, error, n_total):
    """Answer the k most confident claims, k = 1..A (A = claims with a definite verdict). Coverage is k / n_total,
    so abstentions the pipeline already makes count as uncovered; selective risk is the error rate among the k."""
    order = np.argsort(-np.asarray(score), kind="stable")
    err = np.asarray(error, dtype=float)[order]
    k = np.arange(1, len(err) + 1)
    return k / n_total, np.cumsum(err) / k


def aurc(score, error, n_total):
    return float(np.mean(risk_coverage(score, error, n_total)[1]))


def selective(rows, rng):
    answered = [r for r in rows if r["verdict"]["full"] != "unknown"]
    score = np.array([r["score"] for r in answered])
    error = np.array([r["verdict"]["full"] != r["gold"] for r in answered], dtype=float)
    n = len(rows)
    cov, risk = risk_coverage(score, error, n)
    oracle = aurc(-error, error, n)                        # every error ranked last
    observed = aurc(score, error, n)
    boot = []
    for _ in range(BOOTSTRAP // 10):                       # 1,000 resamples: AURC needs a sort per resample
        i = rng.integers(0, len(answered), len(answered))
        boot.append(aurc(score[i], error[i], n))
    at = {}
    for c in COVERAGES:
        k = int(round(c * n))
        at[str(c)] = float(risk[k - 1]) if k <= len(risk) else None
    return {
        "claims": n, "answered": len(answered), "max_coverage": len(answered) / n,
        "selective_risk_at_max_coverage": float(risk[-1]),
        "aurc": observed, "aurc_bootstrap95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "oracle_aurc": oracle, "excess_aurc": observed - oracle, "random_aurc": float(error.mean()),
        "selective_risk_at_coverage": at,
        "curve": {"coverage": cov.tolist(), "risk": risk.tolist()},
        "oracle_curve": dict(zip(("coverage", "risk"), (a.tolist() for a in risk_coverage(-error, error, n)))),
    }


def plot(sel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"fever": "FEVER", "freshqa": "FreshQA"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        s = sel[ds]
        ax.plot(s["curve"]["coverage"], np.array(s["curve"]["risk"]) * 100, color="#1f77b4", linewidth=2,
                label=f"Ranked by NLI confidence (AURC {100 * s['aurc']:.1f}%)")
        ax.plot(s["oracle_curve"]["coverage"], np.array(s["oracle_curve"]["risk"]) * 100, color="#2ca02c",
                linestyle="--", label=f"Oracle ranking (AURC {100 * s['oracle_aurc']:.1f}%)")
        ax.axhline(100 * s["random_aurc"], color="#7f7f7f", linestyle=":", label=f"Random ranking ({100 * s['random_aurc']:.1f}%)")
        ax.plot(s["max_coverage"], 100 * s["selective_risk_at_max_coverage"], "o", color="#d62728", markersize=7,
                label=f"Pipeline as run ({s['answered']} of {s['claims']} answered)")
        ax.set_title(f"{names[ds]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Coverage (share of claims answered)", fontsize=11)
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("Selective risk (% wrong among answered)", fontsize=11)
    axes[0].set_ylim(0, 25)
    fig.suptitle("Risk-coverage curves of the final verdicts", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main

VARIANTS = ("always_supported", "nli_first_pass", "nli_only", "no_review", "no_escalation", "full")


def main():
    rows = load_rows()
    rng = np.random.default_rng(SEED)
    groups = {ds: [r for r in rows if r["dataset"] == ds] for ds in DATASETS}
    groups["pooled"] = rows
    report = {
        "ablation": {g: {v: summarise(rs, v) for v in VARIANTS} for g, rs in groups.items()},
        "class_intervals": {g: class_intervals(rs, rng) for g, rs in groups.items()},
        "selective": {ds: selective(groups[ds], rng) for ds in DATASETS},
    }
    pooled = report["ablation"]["pooled"]
    adjusted = holm({v: s["vs_full"]["mcnemar_exact_p"] for v, s in pooled.items() if v != "full"})
    for v, p in adjusted.items():
        pooled[v]["vs_full"]["holm_within_ablation"] = p
    with open(os.path.join(OUT_DIR, "ablation_selective.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot(report["selective"], os.path.join(OUT_DIR, "risk_coverage.png"))
    print_report(report)


def print_report(rep):
    names = {"always_supported": "always SUPPORTED", "nli_first_pass": "NLI only, first pass", "nli_only": "NLI only",
             "no_review": "no lightweight review", "no_escalation": "no full escalation", "full": "full pipeline"}
    for g, table in rep["ablation"].items():
        print(f"\n== Ablation: {g}")
        print(f"{'variant':24s} {'acc':>6s} {'wilson95':>15s} {'P':>5s} {'R':>5s} {'F1':>5s} {'unk':>4s} {'calls':>6s}  vs full (b/c, p)")
        for v, s in table.items():
            p = f"{s['precision']:.2f}" if s["precision"] is not None else "  -  "
            lo, hi = s["wilson95"]
            vs = s["vs_full"]
            print(f"{names[v]:24s} {100 * s['accuracy']:5.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}] {p:>5s} {s['recall']:5.2f} "
                  f"{s['f1']:5.2f} {s['unknown']:4d} {s['llm_calls_per_claim']:6.2f}  "
                  f"{vs['full_right_only']}/{vs['variant_right_only']}, p={vs['mcnemar_exact_p']:.2g}"
                  + (f", Holm {vs['holm_within_ablation']:.2g}" if "holm_within_ablation" in vs else ""))
    print("\n== Precision, recall and F1 of the hallucination class (full pipeline)")
    for g, s in rep["class_intervals"].items():
        (pl, ph), (rl, rh), (fl, fh) = s["precision_wilson95"], s["recall_wilson95"], s["f1_bootstrap95"]
        print(f"{g:8s} P {s['precision']:.3f} [{pl:.3f}, {ph:.3f}]  R {s['recall']:.3f} [{rl:.3f}, {rh:.3f}]  "
              f"F1 {s['f1']:.3f} [{fl:.3f}, {fh:.3f}]")
    print("\n== Selective prediction (final verdicts ranked by the NLI probability of the verdict)")
    for ds, s in rep["selective"].items():
        lo, hi = s["aurc_bootstrap95"]
        at = ", ".join(f"{float(c):.0%}: {100 * r:.1f}%" for c, r in s["selective_risk_at_coverage"].items() if r is not None)
        print(f"{ds:8s} answered {s['answered']}/{s['claims']}, risk at max coverage {100 * s['selective_risk_at_max_coverage']:.1f}%; "
              f"AURC {100 * s['aurc']:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}], oracle {100 * s['oracle_aurc']:.2f}%, "
              f"random {100 * s['random_aurc']:.2f}%; risk at coverage {at}")


if __name__ == "__main__":
    main()

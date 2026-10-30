"""
A learned escalation rule, compared with the conformal trigger. The deployed cascade sends a claim to the LLM when
its conformal set is not a single label or the corroboration rule asks for a review (§2.7 of pipeline.md); both
depend on the NLI probabilities alone. This script asks whether the other signals in each claim's computation
trace rank the NLI verdicts that will turn out wrong better, and whether escalating by that ranking buys more
accuracy for the same number of LLM calls.

The router is a logistic regression that predicts "the final NLI verdict is wrong" from features fixed in advance,
all available before any LLM verdict:
  - the NLI probabilities, the largest one, the margin over the second and the entropy;
  - the corroboration rule's inputs and outcome: S, K, the numbers of supporting and contradicting snippets, the
    share of snippets that agree with the verdict, the case of the rule (one-hot) and the NLI label (one-hot);
  - the evidence: the number of snippets, the share from the web, the mean dense similarity of the selected
    snippets, whether the queries were rewritten, the size of the conformal set;
  - the claim: its length in words, whether it contains a number or a negation, the temporal and scientific flags.
The benchmark a claim comes from is not a feature. Training needs only gold labels and NLI verdicts, so a router can
be fitted on labelled claims without any LLM call. The features are standardised and the L2 penalty is chosen by
inner five-fold cross-validation on AUROC.

Every score is cross-fitted: ten folds, stratified by benchmark and by whether the NLI verdict is wrong; a claim's
score comes from the model fitted on the other nine folds, so no claim is scored by a model that saw it. The
analysis is repeated over 50 fold seeds to show how much it depends on the split.

A routing policy sends the k highest-scoring claims to the LLM. A claim sent keeps the LLM verdict of
experiments/llm_every_claim.py (the run's verdict for the claims the pipeline reviewed or escalated, one
arbitration call on the saved evidence for the others); a claim not sent keeps its NLI verdict. Sending the
pipeline's own 187 claims reproduces the pipeline exactly, and sending all 600 reproduces the LLM-only baseline.
LLM calls per claim are the query-generation calls of the run plus one call per claim sent.

Reported:
  1. AUROC for detecting a wrong NLI verdict: the learned score against the NLI confidence (1 - the largest NLI
     probability), with DeLong's test for two correlated AUROCs;
  2. accuracy against LLM calls per claim for the learned score, the confidence, random routing (its expectation)
     and the oracle (wrong NLI verdicts that the LLM fixes first);
  3. at the pipeline's budget (187 claims sent): the accuracy of each policy and the exact McNemar test against
     the pipeline, pooled and per benchmark;
  4. the budget the learned router needs to match the pipeline's accuracy and the LLM-only baseline's.
The two primary tests (AUROC against confidence, accuracy against the pipeline at equal budget) form their own
family, Holm-adjusted. No LLM is called.

usage: python experiments/learned_routing.py [--tag wikidata-kg]
"""
import sys
import os
import re
import json
import math
import argparse
import warnings

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import wilson_interval
from statistical_tests import load_records, last, mcnemar_exact_p, holm
from llm_every_claim import plan, SETTLED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "evaluation")
OUT_DIR = os.path.join(ROOT, "experiments")
DATASETS = ("fever", "freshqa")
LABELS = ["contradicted", "unknown", "supported"]       # order of the NLI probability vector
CASES = ["decisive", "corroborated", "review", "joint", "conflict", "single"]
FOLDS, SEED, REPEATS = 10, 0, 50
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
NEGATION = re.compile(r"\b(not|no|never|none|nor|neither|without|absent|cannot)\b|n't\b", re.I)


# ---------------------------------------------------------------- features

def features(rec):
    trace = ((rec.get("result") or {}).get("verification") or {}).get("trace") or []
    steps = [s for s in trace if s.get("kind") == "nli"]
    nli, conf, queries = steps[-1], last(rec, "conformal"), last(rec, "queries") or {}
    retrieval = [s for s in trace if s.get("kind") == "retrieval"]
    p = nli["probabilities"]
    top = sorted(p, reverse=True)
    snippets = nli.get("snippets") or []
    dense = [t["dense"] for t in (retrieval[-1].get("top") or []) if t.get("selected")] if retrieval else []
    f = {
        "p_contradicted": p[0], "p_unknown": p[1], "p_supported": p[2],
        "max_prob": top[0], "margin": top[0] - top[1], "entropy": -sum(q * math.log(q) for q in p if q > 0),
        "S": nli.get("S", 0.0), "K": nli.get("K", 0.0), "n_sup": nli.get("n_sup", 0), "n_con": nli.get("n_con", 0),
        "agreement": sum(s["label"] == nli["label"] for s in snippets) / len(snippets) if snippets else 0.0,
        "n_snippets": len(snippets),
        "web_share": sum(s["source"] == "web" for s in snippets) / len(snippets) if snippets else 0.0,
        "mean_dense": sum(dense) / len(dense) if dense else 0.0,
        "rewritten": float(len(steps) > 1), "set_size": len(conf["set"]),
        "claim_words": len(rec["claim"].split()), "has_number": float(bool(re.search(r"\d", rec["claim"]))),
        "has_negation": float(bool(NEGATION.search(rec["claim"]))),
        "temporal": float(bool(queries.get("temporal"))), "scientific": float(queries.get("domain") == "scientific"),
    }
    for y in LABELS:
        f[f"label_{y}"] = float(nli["label"] == y)
    for c in CASES:
        f[f"case_{c}"] = float(nli.get("case") == c)
    return f


def load_claims(tag):
    records = [(ds, r) for ds in DATASETS for r in load_records(os.path.join(EVAL_DIR, f"{ds}_5_runs_claims.jsonl"))]
    run = os.path.join(EVAL_DIR, "runs", tag)
    cache = {(r["dataset"], r["claim"]): r for r in load_records(os.path.join(run, "llm_every_claim.jsonl"))}
    shadow = {(r["dataset"], r["claim"]): r["label"] for r in load_records(os.path.join(run, "shadow_reviews.jsonl"))}
    llm = plan(records, cache, shadow, uniform=False)
    if any(r["verdict"] is None for r in llm):
        sys.exit("LLM verdicts missing: run python experiments/llm_every_claim.py --run first")
    claims = []
    for (ds, rec), base in zip(records, llm):
        claims.append({"dataset": ds, "gold": rec["gold"], "nli": base["nli"], "llm": base["verdict"],
                       "full": rec["prediction"], "sent": rec["decision_policy"] != SETTLED,
                       "light_calls": base["light_calls"], "features": features(rec)})
    return claims


# ---------------------------------------------------------------- scores

def crossfit_scores(X, y, strata, seed):
    """Out-of-fold probability that the NLI verdict is wrong."""
    out = np.zeros(len(y))
    for train, test in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X, strata):
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", max_iter=5000))
        model.fit(X[train], y[train])
        out[test] = model.predict_proba(X[test])[:, 1]
    return out


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return (rankdata(np.concatenate([pos, neg]))[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def delong(y, s1, s2):
    """DeLong's test for two correlated AUROCs on the same claims (structural components with mid-ranks)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    v10, v01, aucs = [], [], []
    for s in (s1, s2):
        x, z = s[pos], s[neg]
        tx, tz, tall = rankdata(x), rankdata(z), rankdata(np.concatenate([x, z]))
        v10.append((tall[:m] - tx) / n)                 # for each positive: share of negatives ranked below it
        v01.append(1 - (tall[m:] - tz) / m)             # for each negative: share of positives ranked above it
        aucs.append(v10[-1].mean())
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    diff = aucs[0] - aucs[1]
    se = math.sqrt(var)
    return {"auroc": aucs, "difference": diff, "difference95": [diff - 1.96 * se, diff + 1.96 * se],
            "p": 2 * norm.sf(abs(diff) / se)}


# ---------------------------------------------------------------- routing policies

def predictions(claims, sent):
    return [c["llm"] if s else c["nli"] for c, s in zip(claims, sent)]


def top_k(score, k):
    sent = np.zeros(len(score), dtype=bool)
    sent[np.argsort(-score, kind="stable")[:k]] = True
    return sent


def calls_per_claim(claims, sent):
    return (sum(c["light_calls"] for c in claims) + int(np.sum(sent))) / len(claims)


def curve(claims, score):
    """Accuracy after sending the k highest-scoring claims, for every k (one pass over the ranking)."""
    gain = np.array([(c["llm"] == c["gold"]) - (c["nli"] == c["gold"]) for c in claims])
    base = sum(c["nli"] == c["gold"] for c in claims)
    order = np.argsort(-score, kind="stable")
    return (base + np.concatenate([[0], np.cumsum(gain[order])])) / len(claims)


def summary(claims, pred):
    correct = sum(p == c["gold"] for p, c in zip(pred, claims))
    return {"claims": len(claims), "correct": correct, "accuracy": correct / len(claims),
            "wilson95": wilson_interval(correct, len(claims))}


def paired(claims, pred, ref):
    b = sum(r == c["gold"] and p != c["gold"] for p, r, c in zip(pred, ref, claims))     # reference right only
    c_ = sum(p == c["gold"] and r != c["gold"] for p, r, c in zip(pred, ref, claims))    # policy right only
    return {"reference_right_only": b, "policy_right_only": c_, "mcnemar_exact_p": mcnemar_exact_p(b, c_)}


def min_budget(acc, target):
    k = next((k for k, a in enumerate(acc) if a >= target - 1e-12), None)
    return k


# ---------------------------------------------------------------- report

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="wikidata-kg", help="run whose caches to use")
    args = parser.parse_args()

    claims = load_claims(args.tag)
    names = list(claims[0]["features"])
    X = np.array([[c["features"][f] for f in names] for c in claims])
    wrong = np.array([c["nli"] != c["gold"] for c in claims], dtype=int)
    strata = np.array([f"{c['dataset']}-{w}" for c, w in zip(claims, wrong)])
    n, light = len(claims), sum(c["light_calls"] for c in claims) / len(claims)

    pipeline_sent = np.array([c["sent"] for c in claims])
    k_pipe = int(pipeline_sent.sum())
    full = [c["full"] for c in claims]
    assert predictions(claims, pipeline_sent) == full, "routing the pipeline's claims must reproduce the pipeline"

    learned = crossfit_scores(X, wrong, strata, SEED)
    confidence = 1 - X[:, names.index("max_prob")]
    gain = np.array([(c["llm"] == c["gold"]) - (c["nli"] == c["gold"]) for c in claims])
    oracle = gain + 1e-9 * confidence                   # fixes first, then neutral claims, harmful ones last
    scores = {"learned": learned, "confidence": confidence}

    # 1. ranking wrong NLI verdicts
    test = delong(wrong, learned, confidence)
    ranking = {"wrong_nli_verdicts": int(wrong.sum()), "auroc_learned": test["auroc"][0],
               "auroc_confidence": test["auroc"][1], "auroc_pipeline_rule": auroc(wrong, pipeline_sent.astype(float)),
               "delong": test}

    # 2. accuracy against cost
    curves = {name: curve(claims, s) for name, s in {**scores, "oracle": oracle}.items()}
    base_acc, all_acc = curves["learned"][0], curves["learned"][-1]
    curves["random"] = base_acc + (all_acc - base_acc) * np.arange(n + 1) / n
    assert abs(all_acc - summary(claims, [c["llm"] for c in claims])["accuracy"]) < 1e-12

    # 3. equal budget
    at_budget = {"claims_sent": k_pipe, "llm_calls_per_claim": light + k_pipe / n, "policies": {}}
    preds = {"pipeline": full, **{name: predictions(claims, top_k(s, k_pipe)) for name, s in scores.items()}}
    for name, pred in preds.items():
        entry = {}
        for g in DATASETS + ("pooled",):
            idx = [i for i, c in enumerate(claims) if g == "pooled" or c["dataset"] == g]
            sub, p, r = [claims[i] for i in idx], [pred[i] for i in idx], [full[i] for i in idx]
            entry[g] = summary(sub, p)
            if name != "pipeline":
                entry[g]["vs_pipeline"] = paired(sub, p, r)
        if name == "learned":
            entry["pooled"]["vs_confidence"] = paired(claims, pred, preds["confidence"])
        at_budget["policies"][name] = entry
    overlap = int(np.sum(top_k(learned, k_pipe) & pipeline_sent))

    # 4. budgets to reach the pipeline and the LLM-only baseline
    targets = {"pipeline": summary(claims, full)["accuracy"], "llm_only": all_acc}
    budgets = {}
    for t, acc in targets.items():
        budgets[t] = {"accuracy": acc}
        for name in ("learned", "confidence"):
            k = min_budget(curves[name], acc)
            budgets[t][name] = {"claims_sent": k, "llm_calls_per_claim": None if k is None else light + k / n}
    best_k = int(np.argmax(curves["learned"]))
    peak = {"claims_sent": best_k, "accuracy": float(curves["learned"][best_k]), "llm_calls_per_claim": light + best_k / n}

    primary = holm({"auroc_vs_confidence": test["p"],
                    "accuracy_vs_pipeline": at_budget["policies"]["learned"]["pooled"]["vs_pipeline"]["mcnemar_exact_p"]})

    # robustness over fold seeds
    rep_auc, rep_acc = [], []
    for seed in range(REPEATS):
        s = learned if seed == SEED else crossfit_scores(X, wrong, strata, seed)
        rep_auc.append(auroc(wrong, s))
        rep_acc.append(curve(claims, s)[k_pipe])
    robustness = {"seeds": REPEATS,
                  "auroc": {"mean": float(np.mean(rep_auc)), "min": float(np.min(rep_auc)), "max": float(np.max(rep_auc))},
                  "accuracy_at_budget": {"mean": float(np.mean(rep_acc)), "min": float(np.min(rep_acc)),
                                         "max": float(np.max(rep_acc))}}

    # coefficients of the model fitted on all claims, for interpretation (standardised features)
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", max_iter=5000))
    model.fit(X, wrong)
    coef = model[-1].coef_[0]
    coefficients = sorted(({"feature": f, "coefficient": float(b)} for f, b in zip(names, coef)),
                          key=lambda d: -abs(d["coefficient"]))

    report = {"features": names, "folds": FOLDS, "seed": SEED, "base_llm_calls_per_claim": light,
              "ranking": ranking, "equal_budget": at_budget, "overlap_with_pipeline": overlap,
              "budgets": budgets, "learned_peak": peak, "primary_holm": primary, "robustness": robustness,
              "coefficients": coefficients,
              "curves": {name: [float(a) for a in acc] for name, acc in curves.items()}}
    with open(os.path.join(OUT_DIR, "learned_routing.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot(report, n, light, os.path.join(OUT_DIR, "learned_routing.png"))
    print_report(report)


def plot(rep, n, light, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = light + np.arange(n + 1) / n
    fig, ax = plt.subplots(figsize=(7, 4.4))
    styles = {"oracle": ("oracle", "0.6", ":"), "learned": ("learned router", "C0", "-"),
              "confidence": ("NLI confidence", "C1", "-"), "random": ("random", "0.6", "--")}
    for name, (label, colour, ls) in styles.items():
        ax.plot(x, 100 * np.array(rep["curves"][name]), ls, color=colour, label=label, lw=1.6)
    b = rep["equal_budget"]
    pipe = b["policies"]["pipeline"]["pooled"]["accuracy"]
    ax.plot(b["llm_calls_per_claim"], 100 * pipe, "o", color="C3", ms=7, label="pipeline (conformal trigger)", zorder=5)
    ax.plot(x[-1], 100 * rep["curves"]["learned"][-1], "s", color="k", ms=6, label="LLM on every claim", zorder=5)
    ax.axvline(b["llm_calls_per_claim"], color="C3", lw=0.8, alpha=0.4)
    ax.set_xlabel("LLM calls per claim")
    ax.set_ylabel("accuracy (%), 600 claims")
    ax.set_ylim(70, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_report(rep):
    pct = lambda v: f"{100 * v:.1f}%"
    r, d = rep["ranking"], rep["ranking"]["delong"]
    print(f"== Ranking the {r['wrong_nli_verdicts']} wrong NLI verdicts (cross-fitted, {rep['folds']} folds)")
    print(f"AUROC learned {r['auroc_learned']:.3f}, confidence {r['auroc_confidence']:.3f}, pipeline rule "
          f"{r['auroc_pipeline_rule']:.3f}; difference {d['difference']:+.3f} "
          f"[{d['difference95'][0]:+.3f}, {d['difference95'][1]:+.3f}], DeLong p = {d['p']:.2g}")
    b = rep["equal_budget"]
    print(f"\n== Equal budget: {b['claims_sent']} claims sent, {b['llm_calls_per_claim']:.2f} LLM calls per claim "
          f"(learned router shares {rep['overlap_with_pipeline']} claims with the pipeline)")
    for name, e in b["policies"].items():
        parts = []
        for g in DATASETS + ("pooled",):
            s = e[g]
            t = f"{g} {pct(s['accuracy'])}"
            if "vs_pipeline" in s:
                v = s["vs_pipeline"]
                t += f" (b/c {v['reference_right_only']}/{v['policy_right_only']}, p = {v['mcnemar_exact_p']:.2g})"
            parts.append(t)
        print(f"  {name:11s} " + "; ".join(parts))
    v = b["policies"]["learned"]["pooled"]["vs_confidence"]
    print(f"  learned vs confidence: b/c {v['reference_right_only']}/{v['policy_right_only']}, "
          f"p = {v['mcnemar_exact_p']:.2g}")
    print("\n== Budget needed")
    for t, e in rep["budgets"].items():
        print(f"  to reach {t} ({pct(e['accuracy'])}): " + "; ".join(
            f"{m} {e[m]['claims_sent']} claims ({e[m]['llm_calls_per_claim']:.2f} calls/claim)"
            if e[m]["claims_sent"] is not None else f"{m} never" for m in ("learned", "confidence")))
    p = rep["learned_peak"]
    print(f"  learned peak: {pct(p['accuracy'])} with {p['claims_sent']} claims sent ({p['llm_calls_per_claim']:.2f} calls/claim)")
    print(f"\n== Primary tests, Holm-adjusted: " + ", ".join(f"{k} {v:.2g}" for k, v in rep["primary_holm"].items()))
    rb = rep["robustness"]
    print(f"== Over {rb['seeds']} fold seeds: AUROC {rb['auroc']['mean']:.3f} [{rb['auroc']['min']:.3f}, "
          f"{rb['auroc']['max']:.3f}], accuracy at budget {pct(rb['accuracy_at_budget']['mean'])} "
          f"[{pct(rb['accuracy_at_budget']['min'])}, {pct(rb['accuracy_at_budget']['max'])}]")
    print("== Largest coefficients: " + ", ".join(f"{c['feature']} {c['coefficient']:+.2f}" for c in rep["coefficients"][:8]))


if __name__ == "__main__":
    main()

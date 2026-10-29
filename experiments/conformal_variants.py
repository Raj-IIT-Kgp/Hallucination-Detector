"""
Class-conditional coverage and alternative conformity scores for the conformal layer (§2.6 of pipeline.md).

The production layer is split conformal prediction with the score s(x, y) = 1 - pi_y(x) ("LAC"), calibrated on 100
FEVER claims with gold evidence. Its marginal coverage is 90% on gold evidence but 83.2% on the evidence the
pipeline retrieves. This script asks whether the shortfall is shared equally by true and false claims, and whether
two standard variants do better:
  - Mondrian (class-conditional) conformal prediction: the LAC score with a separate quantile for each label, which
    guarantees 1 - alpha coverage within each class. The gold labels are never UNKNOWN, so that label has no
    calibration claims; it keeps the pooled quantile, as in the production layer.
  - APS (adaptive prediction sets): s(x, y) = the total probability of the labels at least as probable as y, with
    one pooled quantile. The non-randomised form is used, so that routing stays deterministic; it is valid and
    slightly conservative.
  - randomised APS: s(x, y) = the total probability of the labels more probable than y plus U pi_y(x), with one
    uniform U per claim (fixed seed), which removes the conservativeness of APS at the price of random routing; it
    is evaluated as a reference only.
Each variant is evaluated
  1. on gold evidence: calibrated on the first 100 claims of experiments/fever_200_cache.json (the production
     calibration claims) and tested on the other 100;
  2. on retrieved evidence with the gold-evidence quantiles (the deployed setting): the 600 benchmark claims;
  3. on retrieved evidence recalibrated by two-fold cross-fitting, with the split of experiments/full_evaluation.py.
For 2 and 3 it reports coverage overall and for false and true claims (Wilson 95%), the mean set size, the number of
claims escalated under the routing rule (§2.7), and the estimated accuracy: a claim keeps its verdict unless its
route changes; a claim that would no longer be escalated or reviewed takes its NLI verdict, and a claim newly
escalated takes the LLM verdict on its saved evidence from experiments/llm_every_claim.py (claims that move between
the two LLM routes keep their LLM verdict). Each configuration is compared with the pipeline as run by the exact
McNemar test. No LLM is called.

usage: python experiments/conformal_variants.py [--tag wikidata-kg]
"""
import sys
import os
import json
import math
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import wilson_interval
from statistical_tests import load_records, last, mcnemar_exact_p
from llm_every_claim import plan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "evaluation")
OUT_DIR = os.path.join(ROOT, "experiments")
DATASETS = ("fever", "freshqa")
LABELS = ["contradicted", "unknown", "supported"]       # order of the NLI probability vector
CLASSES = ("contradicted", "supported")                 # the gold labels of the benchmarks
ALPHA = 0.10
N_CALIBRATION = 100                                     # the production calibration claims (experiments/eval_conformal.py)
CROSSFIT_SEED = 0                                       # the split of experiments/full_evaluation.py


# ---------------------------------------------------------------- conformity scores and sets

def lac(probs, y):
    return 1 - probs[LABELS.index(y)]


def aps(probs, y):
    """Total probability of the labels at least as probable as y (non-randomised APS)."""
    p = probs[LABELS.index(y)]
    return sum(q for q in probs if q > p) + p


def aps_randomised(probs, y, u):
    p = probs[LABELS.index(y)]
    return sum(q for q in probs if q > p) + u * p


METHODS = {"lac": lac, "mondrian": lac, "aps": aps, "aps_rand": aps_randomised}
RANDOM_SEED = 0


def uniforms(n, salt):
    """One uniform per claim, fixed by the seed and the set of claims (so calibration and test draws differ)."""
    rng = random.Random(f"{RANDOM_SEED}-{salt}")
    return [rng.random() for _ in range(n)]


def score_of(method, probs, y, u):
    return METHODS[method](probs, y, u) if method == "aps_rand" else METHODS[method](probs, y)


def quantile(scores, alpha=ALPHA):
    """Exact split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score, or 1 if there are too few."""
    s = sorted(scores)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return 1.0 if k > len(s) else s[k - 1]


def calibrate(method, probs, labels, us):
    """Quantile per label: one pooled value, or for Mondrian one per class with calibration claims (the others keep
    the pooled value)."""
    pooled = quantile([score_of(method, p, y, u) for p, y, u in zip(probs, labels, us)])
    q = {y: pooled for y in LABELS}
    if method == "mondrian":
        for c in CLASSES:
            q[c] = quantile([score_of(method, p, y, u) for p, y, u in zip(probs, labels, us) if y == c])
    return q


def prediction_set(method, probs, q, u):
    chosen = [y for y in LABELS if score_of(method, probs, y, u) <= q[y]]
    return chosen or [LABELS[max(range(3), key=lambda i: probs[i])]]


def coverage(method, q, probs, labels, us):
    sets = [prediction_set(method, p, q, u) for p, u in zip(probs, us)]
    hit = [y in s for s, y in zip(sets, labels)]
    out = {"claims": len(labels), "coverage": sum(hit) / len(hit), "wilson95": wilson_interval(sum(hit), len(hit)),
           "mean_set_size": sum(map(len, sets)) / len(sets), "singletons": sum(len(s) == 1 for s in sets)}
    for c in CLASSES:
        h = [x for x, y in zip(hit, labels) if y == c]
        out[c] = {"claims": len(h), "coverage": sum(h) / len(h) if h else None, "wilson95": wilson_interval(sum(h), len(h))}
    return out, sets


# ---------------------------------------------------------------- retrieved evidence: routing and accuracy

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
        nli, conf, routing = last(rec, "nli"), last(rec, "conformal"), last(rec, "routing")
        route = "escalate" if routing["escalate"] else "review" if routing["review"] else "accept"
        claims.append({"dataset": ds, "gold": rec["gold"], "probs": conf["probabilities"], "nli": nli["label"],
                       "shared_role": bool(routing.get("shared_role")), "review": bool(routing["review"]),
                       "route": route, "prediction": rec["prediction"], "llm": base["verdict"]})
    return claims


def route_with(claim, cset):
    escalate = (claim["nli"] == "unknown" or len(cset) > 1 or "unknown" in cset
                or (claim["shared_role"] and claim["nli"] == "contradicted"))
    return "escalate" if escalate else "review" if claim["review"] else "accept"


def estimate(claims, sets):
    """Estimated verdicts if routing used the given prediction sets (see the module docstring)."""
    verdicts, moved = [], {"newly_escalated": 0, "newly_accepted": 0, "between_llm_routes": 0}
    for c, s in zip(claims, sets):
        new = route_with(c, s)
        v = c["prediction"]
        if new != c["route"]:
            if new == "accept":
                moved["newly_accepted"] += 1
                v = c["nli"]
            elif c["route"] == "accept":
                moved["newly_escalated"] += 1
                v = c["llm"]
            else:
                moved["between_llm_routes"] += 1
        verdicts.append((new, v))
    return verdicts, moved


def summarise_retrieved(method, q_of, claims, us):
    """q_of(i) is the quantile dictionary used for claim i."""
    sets = [prediction_set(method, c["probs"], q_of(i), us[i]) for i, c in enumerate(claims)]
    labels = [c["gold"] for c in claims]
    hit = [y in s for s, y in zip(sets, labels)]
    out = {"coverage": sum(hit) / len(hit), "wilson95": wilson_interval(sum(hit), len(hit)),
           "mean_set_size": sum(map(len, sets)) / len(sets)}
    for c in CLASSES:
        h = [x for x, y in zip(hit, labels) if y == c]
        out[c] = {"claims": len(h), "coverage": sum(h) / len(h), "wilson95": wilson_interval(sum(h), len(h))}
    verdicts, moved = estimate(claims, sets)
    out.update(moved)
    out["escalated"] = sum(r == "escalate" for r, _ in verdicts)
    out["llm_decided"] = sum(r != "accept" for r, _ in verdicts)
    for ds in DATASETS + ("pooled",):
        idx = [i for i, c in enumerate(claims) if ds in ("pooled", c["dataset"])]
        correct = sum(verdicts[i][1] == claims[i]["gold"] for i in idx)
        out.setdefault("accuracy", {})[ds] = correct / len(idx)
    b = sum(c["prediction"] == c["gold"] and v != c["gold"] for c, (_, v) in zip(claims, verdicts))
    c_ = sum(v == c["gold"] and c["prediction"] != c["gold"] for c, (_, v) in zip(claims, verdicts))
    out["vs_pipeline"] = {"pipeline_right_only": b, "variant_right_only": c_, "mcnemar_exact_p": mcnemar_exact_p(b, c_)}
    return out


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="wikidata-kg", help="run whose LLM-verdict caches to use")
    args = parser.parse_args()

    gold = json.load(open(os.path.join(OUT_DIR, "fever_200_cache.json")))
    g_probs, g_labels = gold["prob_vectors"], gold["y_true_labels"]
    cal_p, cal_y = g_probs[:N_CALIBRATION], g_labels[:N_CALIBRATION]
    test_p, test_y = g_probs[N_CALIBRATION:], g_labels[N_CALIBRATION:]
    claims = load_claims(args.tag)

    n = len(claims)
    order = list(range(n))
    random.Random(CROSSFIT_SEED).shuffle(order)
    folds = [order[: n // 2], order[n // 2:]]
    fold_of = {i: f for f, idx in enumerate(folds) for i in idx}

    report = {"alpha": ALPHA, "gold": {}, "retrieved_gold_calibrated": {}, "retrieved_crossfit": {}, "quantiles": {}}
    u_cal, u_test, u_ret = uniforms(len(cal_p), "gold-calibration"), uniforms(len(test_p), "gold-test"), uniforms(n, "retrieved")
    for m in METHODS:
        q_gold = calibrate(m, cal_p, cal_y, u_cal)
        q_fold = [calibrate(m, [claims[i]["probs"] for i in idx], [claims[i]["gold"] for i in idx], [u_ret[i] for i in idx])
                  for idx in folds]
        report["quantiles"][m] = {"gold": q_gold, "crossfit_fold_a": q_fold[0], "crossfit_fold_b": q_fold[1]}
        report["gold"][m], _ = coverage(m, q_gold, test_p, test_y, u_test)
        report["retrieved_gold_calibrated"][m] = summarise_retrieved(m, lambda i: q_gold, claims, u_ret)
        # each claim is tested with the quantiles calibrated on the OTHER fold
        report["retrieved_crossfit"][m] = summarise_retrieved(m, lambda i: q_fold[1 - fold_of[i]], claims, u_ret)

    # the production layer must reproduce the routing of the run
    as_run = report["retrieved_gold_calibrated"]["lac"]
    if as_run["newly_escalated"] or as_run["newly_accepted"] or as_run["between_llm_routes"]:
        sys.exit(f"the LAC layer with the gold quantile does not reproduce the routing of the run: {as_run}")

    with open(os.path.join(OUT_DIR, "conformal_variants.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot(report, os.path.join(OUT_DIR, "conformal_variants.png"))
    print_report(report)


NAMES = {"lac": "LAC (production)", "mondrian": "Mondrian LAC", "aps": "APS", "aps_rand": "randomised APS"}


def plot(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"lac": "#2a6fb0", "mondrian": "#3c9a5f", "aps": "#e08a2c", "aps_rand": "#9467bd"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for m in METHODS:
        for setting, marker, fill in (("retrieved_gold_calibrated", "o", True), ("retrieved_crossfit", "s", False)):
            s = rep[setting][m]
            lo, hi = s["wilson95"]
            ax.errorbar([s["escalated"]], [100 * s["coverage"]], yerr=[[100 * (s["coverage"] - lo)], [100 * (hi - s["coverage"])]],
                        fmt=marker, color=colors[m], mfc=colors[m] if fill else "white", ms=7, capsize=3, lw=1.3)
    ax.axhline(90, color="#7f7f7f", ls="--", lw=1)
    ax.text(ax.get_xlim()[0], 90.4, " nominal 90%", fontsize=8, color="#7f7f7f")
    for m in METHODS:
        ax.plot([], [], "o", color=colors[m], label=NAMES[m])
    ax.plot([], [], "o", color="#555555", label="calibrated on gold evidence")
    ax.plot([], [], "s", color="#555555", mfc="white", label="recalibrated (cross-fitted)")
    ax.set_xlabel("Claims escalated (of 600)")
    ax.set_ylabel("Coverage on retrieved evidence (%)")
    ax.set_title("Coverage against escalations", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1]
    width = 0.1
    groups = [("retrieved_gold_calibrated", "gold-calibrated"), ("retrieved_crossfit", "cross-fitted")]
    xs = []
    for gi, (setting, _) in enumerate(groups):
        for mi, m in enumerate(METHODS):
            for ci, (c, hatch) in enumerate((("contradicted", ""), ("supported", "//"))):
                x = gi * 1.0 + (mi * 2 + ci) * width - 3.5 * width
                s = rep[setting][m][c]
                lo, hi = s["wilson95"]
                ax.bar(x, 100 * s["coverage"], width, color=colors[m], hatch=hatch, edgecolor="white",
                       yerr=[[100 * (s["coverage"] - lo)], [100 * (hi - s["coverage"])]], capsize=2, ecolor="#333333")
        xs.append(gi * 1.0)
    ax.axhline(90, color="#7f7f7f", ls="--", lw=1)
    ax.set_xticks(xs, [g for _, g in groups])
    ax.set_ylim(50, 102)
    ax.set_ylabel("Coverage on retrieved evidence (%)")
    ax.set_title("Coverage of false (solid) and true (hatched) claims", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_report(rep):
    def pct(x):
        return f"{100 * x:5.1f}%"
    print("== Gold evidence (calibrated on 100 claims, tested on 100)")
    for m, s in rep["gold"].items():
        lo, hi = s["wilson95"]
        print(f"{NAMES[m]:18s} coverage {pct(s['coverage'])} [{pct(lo)}, {pct(hi)}]  false {pct(s['contradicted']['coverage'])} "
              f"(n={s['contradicted']['claims']})  true {pct(s['supported']['coverage'])} (n={s['supported']['claims']})  "
              f"set size {s['mean_set_size']:.2f}")
    for setting, title in (("retrieved_gold_calibrated", "Retrieved evidence, gold-calibrated quantiles (deployed)"),
                           ("retrieved_crossfit", "Retrieved evidence, cross-fitted recalibration")):
        print(f"\n== {title}")
        for m, s in rep[setting].items():
            lo, hi = s["wilson95"]
            f, t = s["contradicted"], s["supported"]
            vs = s["vs_pipeline"]
            print(f"{NAMES[m]:18s} coverage {pct(s['coverage'])} [{pct(lo)}, {pct(hi)}]  "
                  f"false {pct(f['coverage'])} [{pct(f['wilson95'][0])}, {pct(f['wilson95'][1])}]  "
                  f"true {pct(t['coverage'])} [{pct(t['wilson95'][0])}, {pct(t['wilson95'][1])}]  "
                  f"size {s['mean_set_size']:.2f}  escalated {s['escalated']:3d} (new {s['newly_escalated']:3d})  LLM-decided {s['llm_decided']:3d}  "
                  f"acc FEVER {pct(s['accuracy']['fever'])} FreshQA {pct(s['accuracy']['freshqa'])} pooled {pct(s['accuracy']['pooled'])}  "
                  f"vs run {vs['pipeline_right_only']}/{vs['variant_right_only']} p={vs['mcnemar_exact_p']:.2g}")
    print("\n== Quantiles")
    for m, q in rep["quantiles"].items():
        print(f"{NAMES[m]:18s} " + "; ".join(f"{k}: " + ", ".join(f"{y[:3]} {v:.4f}" for y, v in qq.items()) for k, qq in q.items()))


if __name__ == "__main__":
    main()

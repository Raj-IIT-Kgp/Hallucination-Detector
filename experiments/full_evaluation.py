"""
One experiment that re-measures everything pipeline.md reports for the current pipeline.

It runs the five-set evaluation of FEVER and FreshQA (the same 200 claims as the published run:
seed 42) with full per-claim records (verdict, computation trace, evidence, LLM-call log), then
derives every result from those records:

  1. accuracy, precision, recall, F1, confusion counts, per set and pooled
  2. decision routes and the accuracy of each route
  3. latency, LLM calls per claim, and per-model / per-tier LLM latency
  4. the arbiter's effect: NLI verdict before arbitration vs. final verdict
  5. the corroboration rule (§2.5): how often a refutation is right, by number of decisive
     contradicting snippets
  6. claim-by-claim changes against the published run
  7. the conformal layer on retrieved evidence: coverage of the current q-hat, and a
     two-fold cross-fitted recalibration (calibrate on one half, test on the other)
  8. the estimated accuracy of the recalibrated pipeline: only claims whose route would change
     need new work; claims that would newly be escalated get one LLM review over their saved
     evidence (cached in shadow_reviews.jsonl, so a re-run never repeats a call)

Everything is resumable: if the API quota runs out, run the same command again.

usage: python experiments/full_evaluation.py [--tag NAME] [--skip-run] [--no-shadow]
"""
import sys
import os
import json
import math
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.run_5_runs import run_5_sets, run_dir, metrics, EVAL_DIR

DATASETS = ("fever", "freshqa")
LABELS = ["contradicted", "unknown", "supported"]      # order of the NLI probability vector
ALPHA = 0.10
CURRENT_Q_HAT = 0.898471511900425                      # experiments/conformal_calibration.json


# ---------------------------------------------------------------- loading

def load_records(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def steps(rec, kind):
    trace = ((rec.get("result") or {}).get("verification") or {}).get("trace") or []
    return [s for s in trace if s.get("kind") == kind]


def last(rec, kind):
    found = steps(rec, kind)
    return found[-1] if found else None


# ---------------------------------------------------------------- statistics

def wilson(successes, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = successes / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n))


def q_hat(scores, alpha=ALPHA):
    """Exact split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score (§2.6)."""
    s = sorted(scores)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return 1.0 if k > len(s) else s[k - 1]


def prediction_set(probs, q):
    chosen = [LABELS[i] for i, p in enumerate(probs) if 1 - p <= q]
    return chosen or [LABELS[max(range(3), key=lambda i: probs[i])]]


# ---------------------------------------------------------------- analyses

def route_accuracy(records):
    out = {}
    for r in records:
        route = out.setdefault(r.get("decision_policy") or "none", {"claims": 0, "correct": 0})
        route["claims"] += 1
        route["correct"] += r["gold"] == r["prediction"]
    return out


def arbiter_effect(records):
    """NLI verdict just before arbitration vs. the final verdict, for claims the LLM decided."""
    out = {}
    for r in records:
        arb, nli = last(r, "arbiter"), last(r, "nli")
        if not arb or not nli:
            continue
        group = out.setdefault(arb.get("mode", "?"), {"claims": 0, "fixed": 0, "broken": 0, "both_right": 0,
                                                       "both_wrong": 0, "llm_unavailable": 0})
        group["claims"] += 1
        if arb.get("error"):
            group["llm_unavailable"] += 1
            continue
        before, after = nli["label"] == r["gold"], r["prediction"] == r["gold"]
        key = {(False, True): "fixed", (True, False): "broken", (True, True): "both_right", (False, False): "both_wrong"}
        group[key[(before, after)]] += 1
    return out


def corroboration(records):
    """For claims with at least one decisive contradicting snippet: was the claim actually false?"""
    rows = {}
    for r in records:
        nli = last(r, "nli")
        if not nli or not nli.get("n_con"):
            continue
        key = "decisive support also present" if nli.get("n_sup") else ("3 or more" if nli["n_con"] >= 3 else str(nli["n_con"]))
        row = rows.setdefault(key, {"claims": 0, "actually_false": 0})
        row["claims"] += 1
        row["actually_false"] += r["gold"] == "contradicted"
    for row in rows.values():
        row["share_wrong_if_accepted"] = 1 - row["actually_false"] / row["claims"]
    return rows


def llm_usage(records):
    by_route, per_model = {}, {}
    for r in records:
        route = by_route.setdefault(r.get("decision_policy") or "none", [])
        route.append(r.get("llm_calls", 0))
        for call in r.get("llm_log", []):
            if call.get("ok") and call.get("model"):
                per_model.setdefault(f"{call['model']} ({call['tier']})", []).append(call["sec"])
    return {
        "mean_calls_per_claim": sum(r.get("llm_calls", 0) for r in records) / len(records) if records else 0,
        "mean_calls_by_route": {k: sum(v) / len(v) for k, v in by_route.items()},
        "latency_by_model_and_tier": {k: {"calls": len(v), "mean_sec": sum(v) / len(v), "p50_sec": sorted(v)[len(v) // 2]}
                                      for k, v in per_model.items()},
    }


def changes_vs_published(dataset, records):
    old = {(r["set"], r["claim"]): r for r in load_records(os.path.join(EVAL_DIR, f"{dataset}_5_runs_claims.jsonl"))}
    changed = []
    for r in records:
        o = old.get((r["set"], r["claim"]))
        if o and o["prediction"] != r["prediction"]:
            changed.append({"claim": r["claim"], "gold": r["gold"], "published": o["prediction"], "now": r["prediction"],
                            "effect": "fixed" if r["prediction"] == r["gold"] else "broke" if o["prediction"] == o["gold"] else "still wrong",
                            "route_published": o.get("decision_policy"), "route_now": r.get("decision_policy")})
    return changed


def conformal_on_retrieved(all_records, seed=0):
    """Coverage of the current q-hat on retrieved evidence, and a two-fold cross-fitted recalibration."""
    data = []
    for ds, r in all_records:
        conf = last(r, "conformal")
        if conf and conf.get("probabilities"):
            data.append((ds, r, conf["probabilities"]))
    def covered(probs, gold, q):
        return gold in prediction_set(probs, q)
    n = len(data)
    current = sum(covered(p, r["gold"], CURRENT_Q_HAT) for _, r, p in data)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    folds = [order[: n // 2], order[n // 2:]]
    fold_of = {i: f for f, idx in enumerate(folds) for i in idx}
    q_fold = [q_hat([1 - data[i][2][LABELS.index(data[i][1]["gold"])] for i in idx]) for idx in folds]
    # each claim is tested with the q-hat calibrated on the OTHER fold
    cross_q = {i: q_fold[1 - fold_of[i]] for i in range(n)}
    cross_cov = sum(covered(data[i][2], data[i][1]["gold"], cross_q[i]) for i in range(n))
    sizes = lambda q_of: sum(len(prediction_set(data[i][2], q_of(i))) for i in range(n)) / n if n else 0
    per_dataset = {}
    for ds in DATASETS:
        idx = [i for i in range(n) if data[i][0] == ds]
        per_dataset[ds] = {
            "claims": len(idx),
            "coverage_current_q_hat": sum(covered(data[i][2], data[i][1]["gold"], CURRENT_Q_HAT) for i in idx) / len(idx) if idx else None,
            "coverage_cross_fitted": sum(covered(data[i][2], data[i][1]["gold"], cross_q[i]) for i in idx) / len(idx) if idx else None,
        }
    return {
        "claims": n,
        "current_q_hat": CURRENT_Q_HAT,
        "coverage_current_q_hat": current / n if n else None,
        "coverage_current_q_hat_wilson95": wilson(current, n),
        "mean_set_size_current": sizes(lambda i: CURRENT_Q_HAT),
        "q_hat_fold_a": q_fold[0], "q_hat_fold_b": q_fold[1],
        "coverage_cross_fitted": cross_cov / n if n else None,
        "coverage_cross_fitted_wilson95": wilson(cross_cov, n),
        "mean_set_size_cross_fitted": sizes(lambda i: cross_q[i]),
        "q_hat_all_retrieved": q_hat([1 - p[LABELS.index(r["gold"])] for _, r, p in data]),
        "per_dataset": per_dataset,
    }, data, cross_q


def recalibrated_estimate(data, cross_q, tag, shadow):
    """
    Estimated verdicts if the conformal layer used the cross-fitted q-hat. A claim keeps its actual
    verdict unless its route changes: newly accepted claims take the NLI verdict, and newly escalated
    claims take an LLM review of their saved evidence (one call each, cached).
    """
    cache_path = os.path.join(run_dir(tag), "shadow_reviews.jsonl")
    cache = {(r["dataset"], r["claim"]): r for r in load_records(cache_path)}
    out = {ds: {"claims": 0, "correct": 0, "route_changes": 0, "newly_escalated": 0, "newly_accepted": 0, "pending": 0}
           for ds in DATASETS}
    changes = []
    for i, (ds, r, probs) in enumerate(data):
        routing, nli = last(r, "routing"), last(r, "nli")
        if not routing or not nli:
            continue
        new_set = prediction_set(probs, cross_q[i])
        new_escalate = (nli["label"] == "unknown" or len(new_set) > 1 or "unknown" in new_set
                        or (routing.get("shared_role") and nli["label"] == "contradicted"))
        old_route = "escalate" if routing["escalate"] else "review" if routing["review"] else "accept"
        new_route = "escalate" if new_escalate else "review" if routing["review"] else "accept"
        verdict = r["prediction"]
        if new_route != old_route:
            out[ds]["route_changes"] += 1
            if new_route == "accept":
                out[ds]["newly_accepted"] += 1
                verdict = nli["label"]
            elif old_route == "accept":
                out[ds]["newly_escalated"] += 1
                key = (ds, r["claim"])
                if key not in cache and shadow:
                    from agents.critic_agent import nli_override
                    label, error, _ = nli_override(r["claim"], (r.get("result") or {}).get("evidence", ""),
                                                   return_reasoning=True, strict_premises=True)
                    if error:
                        raise RuntimeError(f"LLM unavailable during the shadow review ({error[:80]}); run the command again later")
                    cache[key] = {"dataset": ds, "claim": r["claim"], "label": label}
                    with open(cache_path, "a") as f:
                        f.write(json.dumps(cache[key]) + "\n")
                if key in cache:
                    verdict = cache[key]["label"]
                else:
                    out[ds]["pending"] += 1
            changes.append({"dataset": ds, "claim": r["claim"], "gold": r["gold"], "route": f"{old_route} -> {new_route}",
                            "verdict": f"{r['prediction']} -> {verdict}"})
        out[ds]["claims"] += 1
        out[ds]["correct"] += verdict == r["gold"]
    for ds in DATASETS:
        c = out[ds]
        c["accuracy"] = c["correct"] / c["claims"] if c["claims"] else None
    return out, changes


# ---------------------------------------------------------------- report

def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def write_report(tag, report):
    lines = [f"# Full evaluation: {tag}", ""]
    for ds in DATASETS:
        d = report["datasets"].get(ds)
        if not d:
            continue
        p, old = d["pooled"], d.get("published_pooled") or {}
        lines += [f"## {ds.upper()}", "",
                  "| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |",
                  "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"]
        for name, m in (("now", p), ("published", old)):
            if m:
                lines.append(f"| {name} | {pct(m['accuracy'])} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['tp']} | {m['fp']} | "
                             f"{m['fn']} | {m['tn']} | {m['unknown']} | {m['mean_latency_sec']:.1f} s | "
                             f"{'—' if m.get('mean_llm_calls') is None else f'{m['mean_llm_calls']:.2f}'} |")
        lines += ["", "Routes: " + ", ".join(f"{k} {v['correct']}/{v['claims']}" for k, v in d["route_accuracy"].items()), "",
                  "Arbiter effect: " + json.dumps(d["arbiter_effect"]), "",
                  f"Changed vs published ({len(d['changes_vs_published'])}):"]
        lines += [f"- [{c['effect']}] {c['claim']} (gold {c['gold']}: {c['published']} -> {c['now']}; {c['route_published']} -> {c['route_now']})"
                  for c in d["changes_vs_published"]] + [""]
    lines += ["## Corroboration rule (both benchmarks)", "", "| Decisive contradicting snippets | Claims | Actually false | Wrong if accepted |",
              "| :--- | :--- | :--- | :--- |"]
    lines += [f"| {k} | {v['claims']} | {v['actually_false']} | {pct(v['share_wrong_if_accepted'])} |" for k, v in sorted(report["corroboration"].items())]
    c = report["conformal"]
    lines += ["", "## Conformal layer on retrieved evidence", "",
              f"- claims: {c['claims']}",
              f"- current q-hat {c['current_q_hat']:.4f}: coverage {pct(c['coverage_current_q_hat'])} "
              f"(Wilson 95% {pct(c['coverage_current_q_hat_wilson95'][0])}–{pct(c['coverage_current_q_hat_wilson95'][1])}), mean set size {c['mean_set_size_current']:.2f}",
              f"- cross-fitted q-hat (folds {c['q_hat_fold_a']:.4f} / {c['q_hat_fold_b']:.4f}): coverage {pct(c['coverage_cross_fitted'])} "
              f"(Wilson 95% {pct(c['coverage_cross_fitted_wilson95'][0])}–{pct(c['coverage_cross_fitted_wilson95'][1])}), mean set size {c['mean_set_size_cross_fitted']:.2f}",
              f"- q-hat from all retrieved-evidence claims: {c['q_hat_all_retrieved']:.4f}",
              "", "Estimated accuracy with the recalibrated layer: " + json.dumps(report["recalibrated_estimate"]), "",
              "## LLM usage", "", json.dumps(report["llm_usage"], indent=2)]
    path = os.path.join(run_dir(tag), "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="wikidata-kg")
    parser.add_argument("--skip-run", action="store_true", help="Only analyse the records already saved")
    parser.add_argument("--no-shadow", action="store_true", help="Make no LLM calls for the recalibration estimate")
    args = parser.parse_args()

    if not args.skip_run:
        for ds in DATASETS:
            if run_5_sets(ds, 20, 5, 42, fresh=False, tag=args.tag) is None:
                print("Stopped (LLM quota). Run the same command again to resume.")
                sys.exit(1)

    out_dir = run_dir(args.tag)
    report = {"tag": args.tag, "datasets": {}}
    all_records = []
    for ds in DATASETS:
        records = [r for r in load_records(os.path.join(out_dir, f"{ds}_5_runs_claims.jsonl")) if r.get("result")]
        if not records:
            continue
        all_records += [(ds, r) for r in records]
        published = os.path.join(EVAL_DIR, f"{ds}_5_runs_results.json")
        report["datasets"][ds] = {
            "pooled": metrics(records),
            "published_pooled": json.load(open(published))["pooled"] if os.path.exists(published) else None,
            "route_accuracy": route_accuracy(records),
            "arbiter_effect": arbiter_effect(records),
            "changes_vs_published": changes_vs_published(ds, records),
            "llm_usage": llm_usage(records),
        }
    records_only = [r for _, r in all_records]
    report["corroboration"] = corroboration(records_only)
    report["llm_usage"] = llm_usage(records_only)
    report["conformal"], data, cross_q = conformal_on_retrieved(all_records)
    report["recalibrated_estimate"], report["recalibration_changes"] = recalibrated_estimate(data, cross_q, args.tag, not args.no_shadow)

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {write_report(args.tag, report)}")


if __name__ == "__main__":
    main()

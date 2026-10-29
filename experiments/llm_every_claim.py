"""
Baseline: the LLM arbiter decides every claim. The cascade settles most claims by the NLI verdict and spends an
LLM call only on the claims the conformal layer and the corroboration rule flag; this baseline gives every claim
of the published fifteen-set evaluation one arbitration call on the same retrieved evidence, so that the two can
be compared on accuracy and cost.

Each claim's baseline verdict is the arbiter's verdict on its saved evidence (the evidence the pipeline itself
ended with, after any query rewrite), with the same prompt and settings as the live pipeline (dialectical mode,
strict premises, the claim's domain):
  - claims the pipeline sent to a lightweight review already have that verdict (taken from the record);
  - claims the pipeline escalated already have an arbiter verdict, but on evidence that includes the Wikidata
    statements. By default it is reused, which can only favour the baseline; with --uniform those claims are
    arbitrated again on the evidence without the Wikidata statements, so every claim sees retrieved evidence only;
  - claims the NLI verdict settled get a new call, unless experiments/full_evaluation.py or
    experiments/statistical_tests.py already made the same review (shadow_reviews.jsonl, domain "general"; reused
    only for claims whose domain is general).
New verdicts are cached in evaluation/runs/<tag>/llm_every_claim.jsonl, so a re-run never repeats a call and an
interrupted run resumes. Without --run no LLM is called and the number of calls still needed is printed.

The comparison: accuracy (Wilson 95%), precision, recall and F1 of the hallucination class, and the exact McNemar
test against the full pipeline, per benchmark and pooled, with a 95% interval for the paired difference in
accuracy and the number of claims McNemar's test would need to detect a difference of the observed size; LLM calls per claim (query generation and rewriting
calls as in the run, plus one arbitration call); and an estimate of the latency, adding the mean measured
arbitration time to every claim the NLI verdict settled.

usage: python experiments/llm_every_claim.py [--tag wikidata-kg] [--run] [--uniform]
"""
import sys
import os
import json
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import wilson_interval
import math

from statistical_tests import load_records, last, mcnemar_exact_p, n_mcnemar
from ablation_selective import confusion, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "evaluation")
OUT_DIR = os.path.join(ROOT, "experiments")
DATASETS = ("fever", "freshqa")
SETTLED, REVIEW, ESCALATION = "conformal_singleton", "contradiction_arbitration", "multi_agent_debate"
MIN_INTERVAL_SEC = 5.0      # at most 12 calls a minute, under the primary model's per-minute limit


def domain_of(rec):
    q = last(rec, "queries")
    return (q or {}).get("domain") or "general"


def without_wikidata(evidence):
    from agents.wikidata_graph import WIKIDATA_NOTE
    i = evidence.find(WIKIDATA_NOTE)
    return evidence[:i].rstrip() if i >= 0 else evidence


def plan(records, cache, shadow, uniform):
    """For every claim: where its baseline verdict comes from, and the evidence to arbitrate when a call is needed."""
    rows = []
    for ds, rec in records:
        key, route, domain = (ds, rec["claim"]), rec["decision_policy"], domain_of(rec)
        evidence = (rec.get("result") or {}).get("evidence", "")
        row = {"dataset": ds, "claim": rec["claim"], "gold": rec["gold"], "route": route, "domain": domain,
               "full": rec["prediction"], "nli": last(rec, "nli")["label"], "latency": rec["latency_sec"],
               "light_calls": sum(1 for e in rec.get("llm_log") or [] if e.get("tier") == "light"),
               "verdict": None, "source": None, "evidence": None}
        if route == REVIEW:
            row["verdict"], row["source"] = rec["prediction"], "run (review)"
        elif route == ESCALATION and not uniform:
            row["verdict"], row["source"] = rec["prediction"], "run (escalation, with Wikidata)"
        else:
            row["evidence"] = without_wikidata(evidence) if route == ESCALATION else evidence
            if key in cache and cache[key]["uniform"] == (route == ESCALATION):
                row["verdict"], row["source"] = cache[key]["label"], "new call"
            elif route == SETTLED and domain == "general" and key in shadow:
                row["verdict"], row["source"] = shadow[key], "cached review"
        rows.append(row)
    return rows


def arbitrate(rows, cache, cache_path):
    from agents.critic_agent import nli_override
    from llm.generator import get_call_log
    if os.environ.get("MODEL_PROVIDER", "ollama").lower() != "gemini":
        sys.exit("MODEL_PROVIDER is not gemini: the published results used Gemini (set it in .env)")
    todo, done, last_start = {}, 0, 0.0
    for r in rows:
        if r["verdict"] is None:
            todo.setdefault((r["dataset"], r["claim"]), r)     # a claim listed twice is arbitrated once
    for key, r in todo.items():
        wait = MIN_INTERVAL_SEC - (time.monotonic() - last_start)
        if wait > 0:
            time.sleep(wait)
        last_start = time.monotonic()
        log_before = len(get_call_log())
        label, error, _ = nli_override(r["claim"], r["evidence"], domain=r["domain"], strict_premises=True,
                                       return_reasoning=True)
        if error:
            sys.exit(f"LLM unavailable after {done} of {len(todo)} calls ({error[:100]}); run the command again later")
        calls = get_call_log()[log_before:]
        entry = {"dataset": key[0], "claim": key[1], "label": label, "domain": r["domain"],
                 "uniform": r["route"] == ESCALATION, "calls": calls}
        cache[key] = entry
        with open(cache_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        done += 1
        print(f"  [{done}/{len(todo)}] {label:12s} gold {r['gold']:12s} {key[0]}: {key[1][:70]}", flush=True)
    for r in rows:
        if r["verdict"] is None:
            r["verdict"], r["source"] = cache[(r["dataset"], r["claim"])]["label"], "new call"


def compare(rows, arb_sec):
    gold = [r["gold"] for r in rows]
    base = [r["verdict"] for r in rows]
    full = [r["full"] for r in rows]
    out = {}
    for name, pred in (("baseline", base), ("full", full)):
        correct = sum(g == p for g, p in zip(gold, pred))
        tp, fp, fn, tn = confusion(gold, pred)
        out[name] = {"claims": len(rows), "correct": correct, "accuracy": correct / len(rows),
                     "wilson95": wilson_interval(correct, len(rows)),
                     "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn),
                     "f1": f1_score(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "unknown": sum(p == "unknown" for p in pred)}
    out["baseline"]["llm_calls_per_claim"] = sum(r["light_calls"] + 1 for r in rows) / len(rows)
    out["full"]["llm_calls_per_claim"] = sum(r["light_calls"] + (r["route"] != SETTLED) for r in rows) / len(rows)
    out["full"]["mean_latency_sec"] = sum(r["latency"] for r in rows) / len(rows)
    out["baseline"]["mean_latency_sec_estimated"] = sum(r["latency"] + (arb_sec if r["route"] == SETTLED else 0)
                                                        for r in rows) / len(rows)
    b = sum(f == g and p != g for g, p, f in zip(gold, base, full))       # full pipeline right only
    c = sum(p == g and f != g for g, p, f in zip(gold, base, full))       # baseline right only
    n = len(rows)
    diff = (c - b) / n                                                    # baseline minus full pipeline
    se = math.sqrt(b + c - (c - b) ** 2 / n) / n                          # paired (Wald) standard error
    out["paired"] = {"full_right_only": b, "baseline_right_only": c, "mcnemar_exact_p": mcnemar_exact_p(b, c),
                     "difference": diff, "difference95": [diff - 1.96 * se, diff + 1.96 * se],
                     "claims_for_80pct_power": n_mcnemar((b + c) / n, diff) if b != c else None}
    # where the two differ: the claims the NLI verdict settled
    settled = [r for r in rows if r["route"] == SETTLED]
    nli_right = sum(r["nli"] == r["gold"] for r in settled)
    llm_right = sum(r["verdict"] == r["gold"] for r in settled)
    bs = sum(r["nli"] == r["gold"] and r["verdict"] != r["gold"] for r in settled)
    cs = sum(r["verdict"] == r["gold"] and r["nli"] != r["gold"] for r in settled)
    out["settled_claims"] = {"claims": len(settled), "nli_correct": nli_right, "llm_correct": llm_right,
                             "nli_right_only": bs, "llm_right_only": cs, "mcnemar_exact_p": mcnemar_exact_p(bs, cs),
                             "llm_unknown": sum(r["verdict"] == "unknown" for r in settled)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="wikidata-kg", help="run whose caches to use")
    parser.add_argument("--run", action="store_true", help="make the missing arbitration calls (one per claim)")
    parser.add_argument("--uniform", action="store_true",
                        help="arbitrate escalated claims again without their Wikidata statements")
    args = parser.parse_args()

    records = [(ds, r) for ds in DATASETS for r in load_records(os.path.join(EVAL_DIR, f"{ds}_5_runs_claims.jsonl"))]
    run = os.path.join(EVAL_DIR, "runs", args.tag)
    cache_path = os.path.join(run, "llm_every_claim.jsonl")
    cache = {(r["dataset"], r["claim"]): r for r in load_records(cache_path)}
    shadow = {(r["dataset"], r["claim"]): r["label"] for r in load_records(os.path.join(run, "shadow_reviews.jsonl"))}

    rows = plan(records, cache, shadow, args.uniform)
    missing = {(r["dataset"], r["claim"]) for r in rows if r["verdict"] is None}
    print(f"{len(rows)} claims; verdicts from the run: {sum((r['source'] or '').startswith('run') for r in rows)}, "
          f"cached reviews: {sum(r['source'] == 'cached review' for r in rows)}, "
          f"earlier new calls: {sum(r['source'] == 'new call' for r in rows)}; calls still needed: {len(missing)}")
    if missing and not args.run:
        print("no LLM call made; add --run to make them")
        return
    if missing:
        arbitrate(rows, cache, cache_path)

    secs = [c["sec"] for e in cache.values() for c in e.get("calls", []) if c.get("ok")]
    arb_sec = sum(secs) / len(secs) if secs else 2.94      # mean heavy-call time of the run if none was timed here
    report = {"uniform": args.uniform, "mean_arbitration_sec": arb_sec,
              "sources": {s: sum(r["source"] == s for r in rows) for s in sorted({r["source"] for r in rows})},
              "models": {}, "by_dataset": {}}
    for e in cache.values():
        for c in e.get("calls", []):
            report["models"][c.get("model")] = report["models"].get(c.get("model"), 0) + 1
    for g in DATASETS + ("pooled",):
        report["by_dataset"][g] = compare([r for r in rows if g == "pooled" or r["dataset"] == g], arb_sec)
    suffix = "_uniform" if args.uniform else ""
    with open(os.path.join(OUT_DIR, f"llm_every_claim{suffix}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_report(report)


def print_report(rep):
    print(f"\nsources: {rep['sources']}; models of the new calls: {rep['models']}; "
          f"mean arbitration call {rep['mean_arbitration_sec']:.2f} s")
    for g, s in rep["by_dataset"].items():
        print(f"\n== {g}")
        for name in ("full", "baseline"):
            m = s[name]
            lo, hi = m["wilson95"]
            lat = m.get("mean_latency_sec", m.get("mean_latency_sec_estimated"))
            print(f"{name:9s} acc {100 * m['accuracy']:5.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}]  P {m['precision']:.2f}  "
                  f"R {m['recall']:.2f}  F1 {m['f1']:.2f}  unknown {m['unknown']:3d}  calls/claim {m['llm_calls_per_claim']:.2f}  "
                  f"latency {lat:.1f} s{' (est.)' if name == 'baseline' else ''}")
        p, t = s["paired"], s["settled_claims"]
        print(f"paired: full right only {p['full_right_only']}, baseline right only {p['baseline_right_only']}, "
              f"McNemar p = {p['mcnemar_exact_p']:.3g}; difference {100 * p['difference']:+.1f} points "
              f"[{100 * p['difference95'][0]:+.1f}, {100 * p['difference95'][1]:+.1f}]; claims for 80% power: {p['claims_for_80pct_power']}")
        print(f"settled claims ({t['claims']}): NLI right {t['nli_correct']}, LLM right {t['llm_correct']}, "
              f"NLI only {t['nli_right_only']}, LLM only {t['llm_right_only']}, p = {t['mcnemar_exact_p']:.3g}, "
              f"LLM unknown {t['llm_unknown']}")


if __name__ == "__main__":
    main()

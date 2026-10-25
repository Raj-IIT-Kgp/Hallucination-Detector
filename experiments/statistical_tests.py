"""
Statistical tests on the published ten-set evaluation, and a selection-bias check of the corroboration rule.

Everything is derived from the per-claim records (evaluation/<dataset>_5_runs_claims.jsonl, which hold each
claim's computation trace); no LLM is called unless --shadow is given.

  1. Wilson 95% intervals for each benchmark's pooled accuracy (§3.3, §4.4)
  2. FEVER vs FreshQA: two-proportion z-test, Fisher's exact test, and the Newcombe 95% interval for the
     difference in accuracy (§3.3, §4.5)
  3. the LLM arbiter: exact McNemar test of the NLI verdict before arbitration against the final verdict (§2.9)
  4. the corroboration rule (§2.5): the minimum number m of decisive contradicting snippets was chosen by
     looking at these same 200 claims, so the reported accuracy could be optimistic. The verdict every claim
     would have received with m = 1, 2 or 3 is reconstructed from its trace; m is then chosen by
     cross-fitting (on the other folds only) and each claim is scored with the m chosen without it.
     With m = 3, corroborated contradictions would be reviewed by the LLM; those reviews are read from the
     cache of experiments/full_evaluation.py (shadow_reviews.jsonl) and, with --shadow, missing ones are made
     and cached (one call each). Without them, results are reported as a range.

usage: python experiments/statistical_tests.py [--tag wikidata-kg] [--shadow]
"""
import sys
import os
import json
import math
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import wilson_interval

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
DATASETS = ("fever", "freshqa")
CANDIDATE_M = (1, 2, 3)
PUBLISHED_M = 2


def load_records(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def last(rec, kind):
    trace = ((rec.get("result") or {}).get("verification") or {}).get("trace") or []
    found = [s for s in trace if s.get("kind") == kind]
    return found[-1] if found else None


# ---------------------------------------------------------------- tests

def normal_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def two_proportion_test(c1, n1, c2, n2):
    """Pooled two-proportion z-test of H0: p1 = p2."""
    p = (c1 + c2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (c1 / n1 - c2 / n2) / se if se else 0.0
    return z, normal_two_sided_p(z)


def fisher_exact_p(c1, n1, c2, n2):
    """Two-sided Fisher exact test on the 2x2 table [[c1, n1-c1], [c2, n2-c2]]: the probability, under the
    hypergeometric null with fixed margins, of a table at most as likely as the observed one."""
    total, col = n1 + n2, c1 + c2
    prob = lambda a: math.comb(n1, a) * math.comb(n2, col - a) / math.comb(total, col)
    observed = prob(c1)
    lo, hi = max(0, col - n2), min(n1, col)
    return min(1.0, sum(prob(a) for a in range(lo, hi + 1) if prob(a) <= observed * (1 + 1e-9)))


def newcombe_interval(c1, n1, c2, n2):
    """Newcombe's hybrid score interval for p1 - p2, built from the two Wilson intervals."""
    p1, p2 = c1 / n1, c2 / n2
    l1, u1 = wilson_interval(c1, n1)
    l2, u2 = wilson_interval(c2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def mcnemar_exact_p(b, c):
    """Exact McNemar test: under H0 each of the b + c discordant pairs falls either way with probability 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


# ---------------------------------------------------------------- the corroboration rule

def verdict_with_m(dataset, rec, m, reviews):
    """
    The verdict the claim would have received with min_contra_snippets = m, or None if that needs an LLM review
    that has not been made. Only a claim with decisive contradicting snippets, no decisive support, no shared
    role and no escalation depends on m (§2.5, §2.7): its NLI label is CONTRADICTED and its conformal set the
    same for every m, so only the choice between accepting the contradiction and reviewing it changes.
    """
    nli, routing = last(rec, "nli"), last(rec, "routing")
    if not nli or not routing or routing.get("escalate") or nli.get("n_sup") or nli.get("shared_role"):
        return rec["prediction"]
    n_con = nli.get("n_con") or 0
    if n_con == 0:
        return rec["prediction"]
    reviewed_now = n_con < PUBLISHED_M
    reviewed_then = n_con < m
    if reviewed_now == reviewed_then:
        return rec["prediction"]
    if not reviewed_then:
        return "contradicted"                       # accepted directly instead of reviewed
    return reviews.get((dataset, rec["claim"]))     # reviewed instead of accepted


def choose_m(rows, case):
    """The m with the highest accuracy on rows; a tie goes to the smaller m, which needs fewer LLM reviews.
    Missing reviews count as correct (case 'best') or wrong (case 'worst')."""
    def score(m):
        return sum(v == gold if v is not None else case == "best" for gold, v in ((r["gold"], r["v"][m]) for r in rows))
    return max(CANDIDATE_M, key=lambda m: (score(m), -m))


def crossfit(rows, fold_of):
    """Each fold is scored with the m chosen on the other folds. Returns, per case, the folds' chosen m and the
    cross-fitted number of correct verdicts (a range when unknown reviews are involved)."""
    out = {}
    for case in ("best", "worst"):
        chosen, correct, unknown = {}, 0, 0
        for f in sorted(set(fold_of(r) for r in rows)):
            m = choose_m([r for r in rows if fold_of(r) != f], case)
            chosen[f] = m
            for r in rows:
                if fold_of(r) == f:
                    v = r["v"][m]
                    unknown += v is None
                    correct += v == r["gold"]
        out[case] = {"chosen_m": chosen, "correct_known": correct, "unknown": unknown}
    return out


def corroboration_analysis(rows):
    n = len(rows)
    apparent = {}
    for m in CANDIDATE_M:
        known = [r for r in rows if r["v"][m] is not None]
        correct = sum(r["v"][m] == r["gold"] for r in known)
        apparent[m] = {"correct_min": correct, "correct_max": correct + n - len(known), "unknown": n - len(known),
                       "changed_vs_published": sum(r["v"][m] is not None and r["v"][m] != r["v"][PUBLISHED_M] for r in rows)}
    # paired comparison with the published rule, on the claims whose verdict differs
    paired = {}
    for m in CANDIDATE_M:
        if m == PUBLISHED_M:
            continue
        diff = [r for r in rows if r["v"][m] is not None and r["v"][m] != r["v"][PUBLISHED_M]]
        b = sum(r["v"][PUBLISHED_M] == r["gold"] and r["v"][m] != r["gold"] for r in diff)   # published right only
        c = sum(r["v"][m] == r["gold"] and r["v"][PUBLISHED_M] != r["gold"] for r in diff)   # alternative right only
        paired[m] = {"published_right_only": b, "alternative_right_only": c, "mcnemar_exact_p": mcnemar_exact_p(b, c)}
    folds = {
        "by_set": crossfit(rows, lambda r: r["set"]),
        "leave_one_out": crossfit(rows, lambda r: r["id"]),
    }
    return {"claims": n, "apparent": apparent, "paired_vs_published": paired, "crossfit": folds}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="wikidata-kg", help="run whose shadow-review cache to use")
    parser.add_argument("--shadow", action="store_true", help="make the missing LLM reviews for m = 3 (one call each)")
    args = parser.parse_args()

    records = {ds: load_records(os.path.join(EVAL_DIR, f"{ds}_5_runs_claims.jsonl")) for ds in DATASETS}
    report = {"accuracy": {}, "fever_vs_freshqa": None, "arbiter": {}, "corroboration": None}

    # 1. accuracy with Wilson intervals
    for ds, recs in records.items():
        c, n = sum(r["gold"] == r["prediction"] for r in recs), len(recs)
        report["accuracy"][ds] = {"correct": c, "n": n, "accuracy": c / n, "wilson95": wilson_interval(c, n)}

    # 2. FEVER vs FreshQA
    f, q = report["accuracy"]["fever"], report["accuracy"]["freshqa"]
    z, p = two_proportion_test(f["correct"], f["n"], q["correct"], q["n"])
    report["fever_vs_freshqa"] = {"difference": f["accuracy"] - q["accuracy"], "z": z, "p_two_proportion": p,
                                  "p_fisher_exact": fisher_exact_p(f["correct"], f["n"], q["correct"], q["n"]),
                                  "newcombe95": newcombe_interval(f["correct"], f["n"], q["correct"], q["n"])}

    # 3. the arbiter: NLI verdict just before arbitration vs final verdict, for claims the LLM decided
    for ds, recs in list(records.items()) + [("pooled", records["fever"] + records["freshqa"])]:
        fixed = broken = decided = 0
        for r in recs:
            arb, nli = last(r, "arbiter"), last(r, "nli")
            if not arb or not nli or arb.get("error"):
                continue
            decided += 1
            before, after = nli["label"] == r["gold"], r["prediction"] == r["gold"]
            fixed += after and not before
            broken += before and not after
        report["arbiter"][ds] = {"decided": decided, "fixed": fixed, "broken": broken,
                                 "mcnemar_exact_p": mcnemar_exact_p(fixed, broken)}

    # 4. the corroboration rule
    cache_path = os.path.join(EVAL_DIR, "runs", args.tag, "shadow_reviews.jsonl")
    reviews = {(r["dataset"], r["claim"]): r["label"] for r in load_records(cache_path)}
    rows = []
    for ds, recs in records.items():
        for r in recs:
            nli = last(r, "nli")
            if nli and nli.get("m") not in (None, PUBLISHED_M):
                sys.exit(f"record was produced with m = {nli.get('m')}, expected {PUBLISHED_M}")
            if args.shadow:
                for m in CANDIDATE_M:
                    if verdict_with_m(ds, r, m, reviews) is None:
                        from agents.critic_agent import nli_override
                        label, error, _ = nli_override(r["claim"], (r.get("result") or {}).get("evidence", ""),
                                                       return_reasoning=True, strict_premises=True)
                        if error:
                            sys.exit(f"LLM unavailable ({error[:80]}); run the command again later")
                        reviews[(ds, r["claim"])] = label
                        with open(cache_path, "a") as fh:
                            fh.write(json.dumps({"dataset": ds, "claim": r["claim"], "label": label}) + "\n")
            rows.append({"id": f"{ds}:{r['set']}:{r['claim']}", "set": r["set"], "gold": r["gold"],
                         "v": {m: verdict_with_m(ds, r, m, reviews) for m in CANDIDATE_M}})
    report["corroboration"] = corroboration_analysis(rows)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statistical_tests.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print_report(report)
    print(f"\nSaved: {out}")


def print_report(rep):
    pc = lambda x: f"{100 * x:.1f}%"
    print("Accuracy with Wilson 95% intervals")
    for ds, a in rep["accuracy"].items():
        print(f"  {ds:8} {a['correct']}/{a['n']} = {pc(a['accuracy'])}  [{pc(a['wilson95'][0])}, {pc(a['wilson95'][1])}]")
    t = rep["fever_vs_freshqa"]
    print(f"\nFEVER - FreshQA: {pc(t['difference'])}, Newcombe 95% [{pc(t['newcombe95'][0])}, {pc(t['newcombe95'][1])}], "
          f"z = {t['z']:.2f}, p = {t['p_two_proportion']:.4f} (Fisher exact p = {t['p_fisher_exact']:.4f})")
    print("\nArbiter (exact McNemar, NLI verdict before arbitration vs final verdict)")
    for ds, a in rep["arbiter"].items():
        print(f"  {ds:8} decided {a['decided']}: fixed {a['fixed']}, broken {a['broken']}, p = {a['mcnemar_exact_p']:.4g}")
    c = rep["corroboration"]
    print(f"\nCorroboration rule, {c['claims']} claims (accuracy for each m; a range where LLM reviews are missing)")
    for m, a in c["apparent"].items():
        rng = f"{a['correct_min']}" if not a["unknown"] else f"{a['correct_min']}-{a['correct_max']} ({a['unknown']} reviews missing)"
        print(f"  m = {m}: {rng} correct, {a['changed_vs_published']} known verdicts differ from m = {PUBLISHED_M}")
    for m, a in c["paired_vs_published"].items():
        print(f"  m = {PUBLISHED_M} vs m = {m}: m = {PUBLISHED_M} right only {a['published_right_only']}, "
              f"m = {m} right only {a['alternative_right_only']}, exact McNemar p = {a['mcnemar_exact_p']:.4g}")
    for name, cf in c["crossfit"].items():
        for case, r in cf.items():
            ms = sorted(set(r["chosen_m"].values()))
            print(f"  cross-fit ({name}, missing reviews {'correct' if case == 'best' else 'wrong'}): m chosen {ms}, "
                  f"{r['correct_known']} correct + {r['unknown']} unknown of {c['claims']}")


if __name__ == "__main__":
    main()

"""
Statistical tests on the published fifteen-set evaluation, and a selection-bias check of the corroboration rule.

Everything is derived from the per-claim records (evaluation/<dataset>_5_runs_claims.jsonl, which hold each
claim's computation trace); no LLM is called unless --shadow is given.

  1. Wilson 95% intervals for each benchmark's pooled accuracy (§3.3, §4.4)
  2. FEVER vs FreshQA: two-proportion z-test, Fisher's exact test, and the Newcombe 95% interval for the
     difference in accuracy (§3.3, §4.5)
  3. the LLM arbiter: exact McNemar test of the NLI verdict before arbitration against the final verdict (§2.9)
  4. the corroboration rule (§2.5): the minimum number m of decisive contradicting snippets was chosen by
     looking at these same 600 claims, so the reported accuracy could be optimistic. The verdict every claim
     would have received with m = 1, 2 or 3 is reconstructed from its trace; m is then chosen by
     cross-fitting (on the other folds only) and each claim is scored with the m chosen without it.
     With m = 3, corroborated contradictions would be reviewed by the LLM; those reviews are read from the
     cache of experiments/full_evaluation.py (shadow_reviews.jsonl) and, with --shadow, missing ones are made
     and cached (one call each). Without them, results are reported as a range.
  5. the three live runs (sets 1-5, 6-10, 11-15), per benchmark: chi-square test of homogeneity of their
     accuracies (§3.3)
  6. multiple comparisons: Holm (family-wise error) and Benjamini-Hochberg (false discovery rate) adjusted
     p-values over every test above (§4.5)
  7. power and sample size: claims needed for a given interval width, for each comparison to reach 80% power,
     each analytic formula checked by Monte Carlo simulation (§4.5)

usage: python experiments/statistical_tests.py [--tag wikidata-kg] [--shadow]
"""
import sys
import os
import json
import math
import argparse
from statistics import NormalDist

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import wilson_interval

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
DATASETS = ("fever", "freshqa")
CANDIDATE_M = (1, 2, 3)
PUBLISHED_M = 2
ALPHA = 0.05
POWER = 0.80
SIM_REPS = 20000
SEED = 0
RUNS = ((1, 5), (6, 10), (11, 15))    # the sets of each live run (the first is ORIGINAL_SETS in evaluation/run_5_runs.py)


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


def chi2_sf(x, df):
    """Upper tail Pr(X >= x) of the chi-square distribution with an integer number of degrees of freedom:
    for even df, exp(-x/2) sum_{i < df/2} (x/2)^i / i!; for odd df, erfc(sqrt(x/2)) plus the analogous series."""
    if x <= 0:
        return 1.0
    if df % 2 == 0:
        term, total = 1.0, 1.0
        for i in range(1, df // 2):
            term *= (x / 2) / i
            total += term
        return math.exp(-x / 2) * total
    total, term = math.erfc(math.sqrt(x / 2)), math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    for k in range(1, (df + 1) // 2):
        total += term
        term *= x / (2 * k + 1)
    return min(1.0, total)


def homogeneity_test(counts):
    """Chi-square test that k accuracies [(correct, n), ...] are equal: the 2 x k table of right and wrong
    verdicts against its expected counts under a common accuracy, with k - 1 degrees of freedom."""
    total_c, total_n = sum(c for c, _ in counts), sum(n for _, n in counts)
    p = total_c / total_n
    x2 = sum((c - n * p) ** 2 / (n * p) + ((n - c) - n * (1 - p)) ** 2 / (n * (1 - p)) for c, n in counts)
    return x2, len(counts) - 1, chi2_sf(x2, len(counts) - 1), min(min(n * p, n * (1 - p)) for _, n in counts)


def mcnemar_exact_p(b, c):
    """Exact McNemar test: under H0 each of the b + c discordant pairs falls either way with probability 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


# ---------------------------------------------------------------- multiple comparisons

def holm(pvalues):
    """
    Holm's step-down adjustment (controls the family-wise error rate, the chance of even one false rejection,
    under any dependence between the tests). With the p-values sorted p_(1) <= ... <= p_(k), the adjusted value is
        p~_(i) = max_{j <= i} min(1, (k - j + 1) p_(j)).
    """
    k = len(pvalues)
    order = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (k - i) * pvalues[name]))
        out[name] = running
    return out


def benjamini_hochberg(pvalues):
    """
    Benjamini-Hochberg step-up adjustment (controls the false discovery rate, the expected share of false
    rejections among all rejections; valid for independent or positively dependent tests):
        p~_(i) = min_{j >= i} min(1, k p_(j) / j).
    """
    k = len(pvalues)
    order = sorted(pvalues, key=pvalues.get)
    out, running = {}, 1.0
    for i in range(k - 1, -1, -1):
        running = min(running, min(1.0, k * pvalues[order[i]] / (i + 1)))
        out[order[i]] = running
    return out


# ---------------------------------------------------------------- power and sample size

_N = NormalDist()


def wilson_half_width(p, n, z=1.96):
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def n_for_half_width(p, width):
    """Smallest n whose Wilson 95% interval at accuracy p has a half-width of at most `width`."""
    n = 1
    while wilson_half_width(p, n) > width:
        n += 1
    return n


def power_two_proportion(p1, p2, n):
    """Power of the two-sided pooled z-test with n claims per group when the true accuracies are p1 and p2:
        1 - beta = Phi( (|p1 - p2| - z_{1-a/2} sqrt(2 pbar qbar / n)) / sqrt((p1 q1 + p2 q2) / n) )."""
    pbar = (p1 + p2) / 2
    za = _N.inv_cdf(1 - ALPHA / 2)
    return _N.cdf((abs(p1 - p2) - za * math.sqrt(2 * pbar * (1 - pbar) / n))
                  / math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n))


def n_two_proportion(p1, p2):
    """Claims per group for the two-sided z-test to reach POWER (the formula above solved for n)."""
    pbar = (p1 + p2) / 2
    za, zb = _N.inv_cdf(1 - ALPHA / 2), _N.inv_cdf(POWER)
    return math.ceil((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
                     / (p1 - p2) ** 2)


def simulate_two_proportion(p1, p2, n, rng):
    """Share of SIM_REPS simulated studies (n claims per group) in which the z-test rejects at ALPHA."""
    c1, c2 = rng.binomial(n, p1, SIM_REPS), rng.binomial(n, p2, SIM_REPS)
    p = (c1 + c2) / (2 * n)
    se = np.sqrt(p * (1 - p) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (c1 - c2) / n / se, 0.0)
    return float(np.mean(np.abs(z) > _N.inv_cdf(1 - ALPHA / 2)))


def n_mcnemar(psi, delta):
    """
    Pairs needed for McNemar's test to reach POWER (Connor, 1987), where psi is the probability that a pair is
    discordant (the two verdicts differ) and delta the difference between the two discordant probabilities:
        n = ( z_{1-a/2} sqrt(psi) + z_{1-b} sqrt(psi - delta^2) )^2 / delta^2.
    """
    za, zb = _N.inv_cdf(1 - ALPHA / 2), _N.inv_cdf(POWER)
    return math.ceil((za * math.sqrt(psi) + zb * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2)


def simulate_mcnemar(p_b, p_c, n, rng):
    """Share of SIM_REPS simulated studies of n pairs in which the exact McNemar test rejects at ALPHA."""
    counts = rng.multinomial(n, [p_b, p_c, 1 - p_b - p_c], SIM_REPS)[:, :2]
    cache = {}
    reject = 0
    for b, c in ((int(b), int(c)) for b, c in counts):
        if (b, c) not in cache:
            cache[(b, c)] = mcnemar_exact_p(b, c) < ALPHA
        reject += cache[(b, c)]
    return reject / SIM_REPS


def power_analysis(report, rows):
    rng = np.random.default_rng(SEED)
    out = {"alpha": ALPHA, "target_power": POWER, "simulations": SIM_REPS}
    # 1. interval width
    out["interval_width"] = {ds: {f"half_width_{w}": n_for_half_width(a["accuracy"], w / 100) for w in (3, 2)}
                             | {"half_width_now": wilson_half_width(a["accuracy"], a["n"])}
                             for ds, a in report["accuracy"].items()}
    # 2. FEVER vs FreshQA: power of the current design for a range of true differences around the pooled
    #    accuracy, and the claims needed for the observed difference
    f, q = report["accuracy"]["fever"], report["accuracy"]["freshqa"]
    n_now = f["n"]
    centre = (f["accuracy"] + q["accuracy"]) / 2
    grid = []
    for d in (0.05, 0.075, 0.10):
        p1, p2 = centre + d / 2, centre - d / 2
        grid.append({"difference": d, "p1": p1, "p2": p2, "power_analytic": power_two_proportion(p1, p2, n_now),
                     "power_simulated": simulate_two_proportion(p1, p2, n_now, rng),
                     "n_per_group_for_80": n_two_proportion(p1, p2)})
    n_obs = n_two_proportion(f["accuracy"], q["accuracy"])
    out["fever_vs_freshqa"] = {"n_per_group_now": n_now, "grid": grid, "observed": {
        "p1": f["accuracy"], "p2": q["accuracy"], "n_per_group_for_80": n_obs,
        "power_simulated_at_that_n": simulate_two_proportion(f["accuracy"], q["accuracy"], n_obs, rng)}}
    # 3. paired comparisons (McNemar): pairs needed for the observed discordance to reach 80% power
    paired = {}
    c = report["corroboration"]
    for name, (b, cc, n) in {
        "arbiter_fever": (report["arbiter"]["fever"]["fixed"], report["arbiter"]["fever"]["broken"], report["arbiter"]["fever"]["decided"]),
        "m3_vs_m2": (c["paired_vs_published"][3]["alternative_right_only"], c["paired_vs_published"][3]["published_right_only"], c["claims"]),
    }.items():
        p_b, p_c = b / n, cc / n
        # with no observed difference (b = c) there is no effect to plan for: no sample size reaches 80% power
        need = n_mcnemar(p_b + p_c, p_b - p_c) if b != cc else None
        paired[name] = {"discordant_better": b, "discordant_worse": cc, "pairs": n, "psi": p_b + p_c, "delta": p_b - p_c,
                        "pairs_for_80": need,
                        "power_simulated_at_that_n": simulate_mcnemar(p_b, p_c, need, rng) if need else None,
                        "power_simulated_now": simulate_mcnemar(p_b, p_c, n, rng)}
    out["mcnemar"] = paired
    return out


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

    # 5. the three live runs: do their accuracies differ more than chance allows?
    report["runs"] = {}
    for ds, recs in records.items():
        counts = []
        for lo, hi in RUNS:
            run = [r for r in recs if lo <= r["set"] <= hi]
            counts.append([sum(r["gold"] == r["prediction"] for r in run), len(run)])
        x2, df, p, min_expected = homogeneity_test(counts)
        report["runs"][ds] = {"sets": [list(r) for r in RUNS], "correct_n": counts, "chi2": x2, "df": df,
                              "p": p, "min_expected_count": min_expected}

    # 6. multiple comparisons over every test reported (one p-value per hypothesis: the z-test for FEVER vs
    #    FreshQA and the chi-square test for the runs, as reported in §3.3, and the exact McNemar test for the
    #    paired comparisons)
    family = {
        "fever_vs_freshqa": report["fever_vs_freshqa"]["p_two_proportion"],
        "arbiter_pooled": report["arbiter"]["pooled"]["mcnemar_exact_p"],
        "arbiter_fever": report["arbiter"]["fever"]["mcnemar_exact_p"],
        "arbiter_freshqa": report["arbiter"]["freshqa"]["mcnemar_exact_p"],
        "m2_vs_m1": report["corroboration"]["paired_vs_published"][1]["mcnemar_exact_p"],
        "m2_vs_m3": report["corroboration"]["paired_vs_published"][3]["mcnemar_exact_p"],
        "runs_fever": report["runs"]["fever"]["p"],
        "runs_freshqa": report["runs"]["freshqa"]["p"],
    }
    h, bh = holm(family), benjamini_hochberg(family)
    report["multiple_testing"] = {"alpha": ALPHA, "tests": {
        k: {"p": family[k], "holm": h[k], "bh": bh[k], "significant_raw": family[k] < ALPHA,
            "significant_holm": h[k] < ALPHA, "significant_bh": bh[k] < ALPHA} for k in sorted(family, key=family.get)}}

    # 7. power and sample size
    report["power"] = power_analysis(report, rows)

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
    print("\nThe three live runs (chi-square test of equal accuracy)")
    for ds, a in rep["runs"].items():
        runs = ", ".join(f"sets {lo}-{hi}: {c}/{n}" for (lo, hi), (c, n) in zip(a["sets"], a["correct_n"]))
        print(f"  {ds:8} {runs}; chi2 = {a['chi2']:.2f} (df {a['df']}), p = {a['p']:.4f}, "
              f"smallest expected count {a['min_expected_count']:.1f}")
    print(f"\nMultiple comparisons ({len(rep['multiple_testing']['tests'])} tests, alpha = {ALPHA})")
    for k, t in rep["multiple_testing"]["tests"].items():
        mark = lambda b: "yes" if b else "no"
        print(f"  {k:17} p = {t['p']:.4g}  Holm {t['holm']:.4g} ({mark(t['significant_holm'])})  "
              f"BH {t['bh']:.4g} ({mark(t['significant_bh'])})")
    pw = rep["power"]
    print(f"\nPower and sample size (alpha = {ALPHA}, target power {POWER:.0%}, {SIM_REPS} simulations)")
    for ds, w in pw["interval_width"].items():
        print(f"  {ds:8} Wilson half-width now {pc(w['half_width_now'])}; n for ±3 points: {w['half_width_3']}, ±2 points: {w['half_width_2']}")
    fq = pw["fever_vs_freshqa"]
    for g in fq["grid"]:
        print(f"  FEVER vs FreshQA, true difference {pc(g['difference'])}: power at n = {fq['n_per_group_now']} "
              f"{pc(g['power_analytic'])} (simulated {pc(g['power_simulated'])}); n per group for 80%: {g['n_per_group_for_80']}")
    o = fq["observed"]
    print(f"  observed {pc(o['p1'])} vs {pc(o['p2'])}: n per group for 80% = {o['n_per_group_for_80']} "
          f"(simulated power there {pc(o['power_simulated_at_that_n'])})")
    for k, m in pw["mcnemar"].items():
        print(f"  {k}: {m['discordant_better']} vs {m['discordant_worse']} of {m['pairs']} (power now {pc(m['power_simulated_now'])}); "
              + (f"pairs for 80%: {m['pairs_for_80']} (simulated power there {pc(m['power_simulated_at_that_n'])})"
                 if m["pairs_for_80"] else "no difference observed, so no sample size to plan for"))


if __name__ == "__main__":
    main()

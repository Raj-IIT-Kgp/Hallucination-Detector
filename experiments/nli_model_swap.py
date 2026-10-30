"""
A stronger NLI model in the cascade. Most of the pipeline's errors are NLI verdicts that are never escalated, so
this script asks how the cascade would perform with a stronger NLI model in place of BART-large-MNLI, on exactly
the evidence the pipeline retrieved.

The replacement is DeBERTa-v3-large fine-tuned on SNLI and MNLI (cross-encoder/nli-deberta-v3-large). Like BART it
has never seen FEVER, so neither benchmark is in its training data; the stronger DeBERTa checkpoints trained on
FEVER-NLI are deliberately not used.

For every claim of the published fifteen-set evaluation the saved evidence (the snippets that reached the final NLI
pass, without the Wikidata statements added on escalation) is scored again, and the pipeline's decision is rebuilt
with the production rules of detector/hallucination_detector.py:
  - per-snippet NLI; the epistemic-conflict check; decisive snippets (confidence >= tau = 0.8); the corroboration
    rule (m = 2, shared-role claims never corroborate); the joint fallback on the concatenated evidence;
  - the conformal set with the model's own q-hat. Probabilities from a different model need their own calibration,
    so the new model is calibrated as the production layer was (experiments/eval_conformal.py): 100 FEVER claims
    with gold evidence, half true and half false, alpha = 0.10. The production calibration draw was not seeded and
    cannot be rebuilt, so a seeded draw (none of them a benchmark claim) is used, and BART's q-hat on the same draw
    is reported next to its deployed value;
  - routing: escalate when the label is UNKNOWN, the conformal set is not a single label other than UNKNOWN, or a
    shared-role claim is contradicted; otherwise review a lone or contested contradiction; otherwise accept.
The retrieval is held fixed. A claim whose new verdict is UNKNOWN would have had its queries rewritten in a live
run; here it is escalated on the evidence in hand and charged one extra query-generation call.

A claim sent to the LLM takes the LLM verdict of experiments/llm_every_claim.py (the run's verdict for the claims
the pipeline reviewed or escalated, one arbitration call on the saved evidence for the others); a claim accepted
takes its NLI verdict. Rebuilding the decisions with BART must reproduce the run: every claim's NLI label and
route are checked against the record, so the comparison isolates the change of model.

Reported per benchmark and pooled: the NLI verdict alone (no LLM) for each model, and the cascade with each model:
accuracy (Wilson 95%), precision, recall and F1 of the hallucination class, routes, LLM calls per claim and the
exact McNemar test of the new model against BART; the coverage of each model's conformal sets on the retrieved
evidence. The two pooled tests form their own family, Holm-adjusted. Per-snippet scores are cached in
experiments/nli_model_swap_cache.json, so a re-run needs no inference. No LLM is called.

usage: python experiments/nli_model_swap.py [--tag wikidata-kg]
"""
import sys
import os
import re
import json
import math
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import wilson_interval
from statistical_tests import load_records, last, mcnemar_exact_p, holm
from llm_every_claim import plan, without_wikidata, SETTLED, REVIEW, ESCALATION
from ablation_selective import confusion, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "evaluation")
OUT_DIR = os.path.join(ROOT, "experiments")
CACHE = os.path.join(OUT_DIR, "nli_model_swap_cache.json")
DATASETS = ("fever", "freshqa")
LABELS = ["contradicted", "unknown", "supported"]
MODELS = {"bart": "facebook/bart-large-mnli", "deberta": "cross-encoder/nli-deberta-v3-large"}
ALPHA, N_CALIBRATION, CALIBRATION_SEED = 0.10, 100, 0
TAU, M = 0.80, 2                                       # detector/hallucination_detector.py
CONFLICT_CONF, CONFLICT_GAP = 0.85, 0.12
SHARED_ROLES = ["co-founder", "co founder", "cofounder", "co-author", "co author", "co-creator", "co creator",
                "member", "director", "investor", "actor"]
EPISTEMIC = "EPISTEMIC CONTRADICTION: Retrieved sources provide conflicting information regarding this claim.\n\n"


# ---------------------------------------------------------------- evidence

def nli_input(rec):
    """(snippet texts, the concatenated evidence text the joint fallback scored) of the final NLI pass."""
    text = without_wikidata(rec["result"]["evidence"])
    if text.startswith(EPISTEMIC):
        text = text[len(EPISTEMIC):]
    body = text
    if body.startswith("[SYSTEM TIMESTAMP:"):
        body = body.split("\n", 1)[1]
    parts = re.split(r"(?:^|\n\n)Evidence \d+: ", body)[1:]
    return parts, text


# ---------------------------------------------------------------- models

class NLI:
    def __init__(self, name):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        id2label = {int(i): l.lower() for i, l in self.model.config.id2label.items()}
        self.index = [next(i for i, l in id2label.items() if l.startswith(p))
                      for p in ("contradiction", "neutral", "entailment")]

    def __call__(self, premise, hypothesis, max_length=None):
        kw = {"max_length": max_length} if max_length else {}
        inputs = self.tok(premise, hypothesis, truncation=True, return_tensors="pt", **kw).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, self.index]
        return self.torch.softmax(logits, dim=0).cpu().tolist()


def result(probs):
    i = max(range(3), key=lambda j: probs[j])
    return {"label": LABELS[i], "confidence": probs[i], "probabilities": probs}


# ---------------------------------------------------------------- the production decision rule

def decide(claim, snippets, joint):
    """_decide_from_evidence: (verification, needs_review, case) from per-snippet results and the joint result."""
    shared = any(r in claim.lower() for r in SHARED_ROLES)
    if len(snippets) <= 1:
        v = joint
        return v, v["label"] == "contradicted" and v["confidence"] >= TAU, "single"
    sup = max([r["confidence"] for r in snippets if r["label"] == "supported"], default=0.0)
    con = max([r["confidence"] for r in snippets if r["label"] == "contradicted"], default=0.0)
    if sup >= CONFLICT_CONF and con >= CONFLICT_CONF and abs(sup - con) < CONFLICT_GAP:
        return {"label": "unknown", "confidence": 1.0, "probabilities": [0.33, 0.34, 0.33]}, False, "conflict"
    strong_sup = sorted([r for r in snippets if r["label"] == "supported" and r["confidence"] >= TAU],
                        key=lambda r: r["confidence"], reverse=True)
    strong_con = sorted([r for r in snippets if r["label"] == "contradicted" and r["confidence"] >= TAU],
                        key=lambda r: r["confidence"], reverse=True)
    if strong_con:
        if len(strong_con) >= M and not strong_sup and not shared:
            return strong_con[0], False, "corroborated"
        return max(strong_sup + strong_con, key=lambda r: r["confidence"]), True, "review"
    if strong_sup:
        return strong_sup[0], False, "decisive"
    return joint, False, "joint"


def conformal_set(probs, q):
    s = [y for y, p in zip(LABELS, probs) if 1 - p <= q]
    return s or [LABELS[max(range(3), key=lambda i: probs[i])]]


def route(claim, v, review, q):
    s = conformal_set(v["probabilities"], q)
    shared = any(r in claim.lower() for r in SHARED_ROLES)
    if v["label"] == "unknown" or len(s) > 1 or "unknown" in s or (shared and v["label"] == "contradicted"):
        return "escalate", s
    return ("review" if review else "accept"), s


def quantile(scores, alpha=ALPHA):
    s = sorted(scores)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return 1.0 if k > len(s) else s[k - 1]


# ---------------------------------------------------------------- scoring with cache

def calibration_claims(exclude):
    from evaluation.benchmarks.fever_dataset import load_fever
    data = load_fever()
    pools = {lab: [d for d in data if d["label"] == lab and len(d.get("evidence_text", "").strip()) > 10
                   and d["claim"] not in exclude] for lab in ("SUPPORTS", "REFUTES")}
    rng = random.Random(CALIBRATION_SEED)
    draw = rng.sample(pools["SUPPORTS"], N_CALIBRATION // 2) + rng.sample(pools["REFUTES"], N_CALIBRATION // 2)
    rng.shuffle(draw)
    return [{"claim": d["claim"], "evidence": d["evidence_text"],
             "gold": "supported" if d["label"] == "SUPPORTS" else "contradicted"} for d in draw]


def score_all(records, cache):
    need = {}
    for key, model in MODELS.items():
        c = cache.setdefault(key, {})
        for ds, rec in records:
            snippets, text = nli_input(rec)
            for t in snippets + [text]:
                k = rec["claim"] + "\x00" + t
                if k not in c:
                    need.setdefault(key, []).append((k, t, rec["claim"], None))
        for cl in cache["calibration"]:
            k = "calibration\x00" + cl["claim"] + "\x00" + cl["evidence"]
            if k not in c:
                need.setdefault(key, []).append((k, cl["evidence"], cl["claim"], 256))
    for key, todo in need.items():
        todo = list({k: (k, t, c, m) for k, t, c, m in todo}.values())
        print(f"scoring {len(todo)} pairs with {MODELS[key]}...", flush=True)
        model = NLI(MODELS[key])
        for i, (k, t, c, m) in enumerate(todo):
            cache[key][k] = model(t, c, max_length=m)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
        del model
        with open(CACHE, "w") as f:
            json.dump(cache, f)


# ---------------------------------------------------------------- evaluation

def rebuild(records, cache, key, q):
    out = []
    for ds, rec in records:
        snippets, text = nli_input(rec)
        per = [result(cache[key][rec["claim"] + "\x00" + t]) for t in snippets]
        joint = result(cache[key][rec["claim"] + "\x00" + text])
        v, review, case = decide(rec["claim"], per, joint)
        r, s = route(rec["claim"], v, review, q)
        out.append({"label": v["label"], "probabilities": v["probabilities"], "route": r, "set": s, "case": case})
    return out


def metrics(gold, pred):
    correct = sum(g == p for g, p in zip(gold, pred))
    tp, fp, fn, tn = confusion(gold, pred)
    return {"claims": len(gold), "correct": correct, "accuracy": correct / len(gold),
            "wilson95": wilson_interval(correct, len(gold)),
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "f1": f1_score(tp, fp, fn), "unknown": sum(p == "unknown" for p in pred)}


def paired(gold, new, ref):
    b = sum(r == g and n != g for g, n, r in zip(gold, new, ref))     # BART right only
    c = sum(n == g and r != g for g, n, r in zip(gold, new, ref))     # new model right only
    return {"bart_right_only": b, "new_right_only": c, "mcnemar_exact_p": mcnemar_exact_p(b, c)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="wikidata-kg", help="run whose caches to use")
    args = parser.parse_args()

    records = [(ds, r) for ds in DATASETS for r in load_records(os.path.join(EVAL_DIR, f"{ds}_5_runs_claims.jsonl"))]
    for ds, rec in records:
        n = [s for s in rec["result"]["verification"]["trace"] if s["kind"] == "nli"][-1]
        if len(nli_input(rec)[0]) != len(n["snippets"]):
            sys.exit(f"evidence of {rec['claim']!r} does not split into its {len(n['snippets'])} snippets")

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if "calibration" not in cache:
        cache["calibration"] = calibration_claims({rec["claim"] for _, rec in records})
    score_all(records, cache)

    run = os.path.join(EVAL_DIR, "runs", args.tag)
    llm_cache = {(r["dataset"], r["claim"]): r for r in load_records(os.path.join(run, "llm_every_claim.jsonl"))}
    shadow = {(r["dataset"], r["claim"]): r["label"] for r in load_records(os.path.join(run, "shadow_reviews.jsonl"))}
    llm = plan(records, llm_cache, shadow, uniform=False)
    if any(r["verdict"] is None for r in llm):
        sys.exit("LLM verdicts missing: run python experiments/llm_every_claim.py --run first")

    deployed_q = last(records[0][1], "conformal")["q_hat"]
    cal = cache["calibration"]
    q = {k: quantile([1 - cache[k]["calibration\x00" + c["claim"] + "\x00" + c["evidence"]][LABELS.index(c["gold"])]
                      for c in cal]) for k in MODELS}
    used_q = {"bart": deployed_q, "deberta": q["deberta"]}
    rebuilt = {k: rebuild(records, cache, k, used_q[k]) for k in MODELS}

    # BART must reproduce the run
    route_of = {SETTLED: "accept", REVIEW: "review", ESCALATION: "escalate"}
    mism = [(rec["claim"], b["label"], last(rec, "nli")["label"], b["route"], route_of[rec["decision_policy"]])
            for (ds, rec), b in zip(records, rebuilt["bart"])
            if b["label"] != last(rec, "nli")["label"] or b["route"] != route_of[rec["decision_policy"]]]
    if mism:
        for m in mism[:10]:
            print("  mismatch:", m)
        sys.exit(f"rebuilding with BART does not reproduce {len(mism)} claims")

    gold = [rec["gold"] for _, rec in records]
    preds, rows = {}, {}
    for k in MODELS:
        nli = [r["label"] for r in rebuilt[k]]
        sent = [r["route"] != "accept" for r in rebuilt[k]]
        preds[k] = {"nli_only": nli, "cascade": [l["verdict"] if s else n for l, s, n in zip(llm, sent, nli)]}
        rewrites = [r["route"] == "escalate" and r["label"] == "unknown" and len(
            [s for s in rec["result"]["verification"]["trace"] if s["kind"] == "nli"]) == 1
            for r, (_, rec) in zip(rebuilt[k], records)] if k != "bart" else [False] * len(records)
        rows[k] = [{"sent": s, "calls": l["light_calls"] + s + w, "route": r["route"], "set": r["set"]}
                   for s, l, r, w in zip(sent, llm, rebuilt[k], rewrites)]
    assert preds["bart"]["cascade"] == [rec["prediction"] for _, rec in records]

    report = {"models": MODELS, "q_hat": {"bart_deployed": deployed_q, "bart_seeded_draw": q["bart"],
                                          "deberta_seeded_draw": q["deberta"]},
              "calibration_claims": len(cal), "by_dataset": {}}
    for g in DATASETS + ("pooled",):
        idx = [i for i, (ds, _) in enumerate(records) if g == "pooled" or ds == g]
        G = [gold[i] for i in idx]
        entry = {}
        for stage in ("nli_only", "cascade"):
            entry[stage] = {k: metrics(G, [preds[k][stage][i] for i in idx]) for k in MODELS}
            entry[stage]["deberta_vs_bart"] = paired(G, [preds["deberta"][stage][i] for i in idx],
                                                     [preds["bart"][stage][i] for i in idx])
        for k in MODELS:
            rs = [rows[k][i] for i in idx]
            entry["cascade"][k].update({
                "routes": {r: sum(x["route"] == r for x in rs) for r in ("accept", "review", "escalate")},
                "llm_calls_per_claim": sum(x["calls"] for x in rs) / len(rs),
                "accepted_errors": sum(x["route"] == "accept" and preds[k]["nli_only"][i] != gold[i]
                                       for x, i in zip(rs, idx)),
            })
            cov = sum(gold[i] in rows[k][i]["set"] for i in idx)
            entry["cascade"][k]["conformal_coverage"] = {"coverage": cov / len(idx),
                                                         "wilson95": wilson_interval(cov, len(idx))}
        report["by_dataset"][g] = entry
    pooled = report["by_dataset"]["pooled"]
    report["primary_holm"] = holm({s: pooled[s]["deberta_vs_bart"]["mcnemar_exact_p"] for s in ("nli_only", "cascade")})
    with open(os.path.join(OUT_DIR, "nli_model_swap.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_report(report)


def print_report(rep):
    pct = lambda v: f"{100 * v:.1f}%"
    q = rep["q_hat"]
    print(f"q-hat: BART deployed {q['bart_deployed']:.4f} (seeded draw {q['bart_seeded_draw']:.4f}), "
          f"DeBERTa {q['deberta_seeded_draw']:.4f} ({rep['calibration_claims']} gold-evidence claims)")
    for g, e in rep["by_dataset"].items():
        print(f"\n== {g}")
        for stage in ("nli_only", "cascade"):
            for k in rep["models"]:
                s = e[stage][k]
                extra = ""
                if stage == "cascade":
                    extra = (f"  routes {s['routes']}, {s['llm_calls_per_claim']:.2f} calls/claim, "
                             f"accepted errors {s['accepted_errors']}, coverage {pct(s['conformal_coverage']['coverage'])}")
                print(f"  {stage:9s} {k:8s} {pct(s['accuracy'])} [{pct(s['wilson95'][0])}, {pct(s['wilson95'][1])}] "
                      f"P {s['precision'] or 0:.2f} R {s['recall'] or 0:.2f} F1 {s['f1'] or 0:.2f}{extra}")
            p = e[stage]["deberta_vs_bart"]
            print(f"  {stage:9s} DeBERTa vs BART: b/c {p['bart_right_only']}/{p['new_right_only']}, "
                  f"p = {p['mcnemar_exact_p']:.2g}")
    print("\nHolm-adjusted (pooled): " + ", ".join(f"{k} {v:.2g}" for k, v in rep["primary_holm"].items()))


if __name__ == "__main__":
    main()

"""
5-set benchmark evaluation for FEVER and FreshQA.

Each benchmark is split into `num_sets` non-overlapping stratified sets of `samples_per_set`
claims (half SUPPORTS, half REFUTES), drawn with a fixed seed so every run uses the same claims.
Per-claim results are appended to evaluation/<dataset>_5_runs_claims.jsonl as they finish, so an
interrupted run resumes where it stopped. A claim whose verification hit an LLM failure (quota,
timeout) is not scored as an abstention: it is retried once after a pause, and the run stops if
it fails again. The summary is written to evaluation/<dataset>_5_runs_results.json.

With --tag NAME the run is written to evaluation/runs/NAME/ instead (so earlier results are kept),
and each per-claim record also holds the claim's full result (verdict, computation trace and
complete evidence) and its LLM-call log, for analysis by experiments/full_evaluation.py.

usage: python evaluation/run_5_runs.py --dataset fever|freshqa|all [--fresh] [--tag NAME]
"""
import sys
import os
import random
import time
import json
import argparse
import warnings
import math

warnings.filterwarnings("ignore")
try:
    from bs4 import GuessedAtParserWarning
    warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
except Exception:
    pass

import torch
torch.set_num_threads(2)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.web_search import WebRetriever
from detector.hallucination_detector import HallucinationDetector
from evaluation.metrics import precision, recall, f1_score
from llm.generator import get_failure_count, get_call_count, get_call_log

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_MAP = {"SUPPORTS": "supported", "REFUTES": "contradicted"}
LLM_RETRY_PAUSE_SEC = 60


def load_items(dataset):
    if dataset == "fever":
        from evaluation.benchmarks.fever_dataset import load_fever
        return [{"claim": x["claim"], "label": x["label"]} for x in load_fever()]
    from evaluation.benchmarks.freshqa_dataset import load_freshqa
    return load_freshqa(split="all")


def make_sets(items, samples_per_set, num_sets, seed):
    """Non-overlapping stratified sets: each holds samples_per_set // 2 SUPPORTS and the rest REFUTES."""
    supported = [x for x in items if x["label"] == "SUPPORTS"]
    refuted = [x for x in items if x["label"] == "REFUTES"]
    half = samples_per_set // 2
    if len(supported) < half * num_sets or len(refuted) < (samples_per_set - half) * num_sets:
        raise ValueError("Not enough samples to construct non-overlapping sets.")

    rng = random.Random(seed)
    rng.shuffle(supported)
    rng.shuffle(refuted)
    sets = []
    for s in range(num_sets):
        batch = supported[s * half:(s + 1) * half] + refuted[s * (samples_per_set - half):(s + 1) * (samples_per_set - half)]
        rng.shuffle(batch)
        sets.append(batch)
    return sets


def verify(detector, item, details=None):
    """
    Returns (prediction, decision_policy, evidence) for one benchmark item; if details is a dict,
    the claim's full result (verification with its trace, complete evidence) is stored in it.
    """
    context = f"Question: {item['raw_question']}" if item.get("raw_question") else None
    claims = detector.analyze(item["claim"], context=context, extract_atomic_facts=False)["report"]["claims"]
    if not claims:
        return "unknown", None, "No report generated."
    if details is not None:
        details["result"] = {k: v for k, v in claims[0].items() if k != "graph"}
    labels = [c["verification"]["label"] for c in claims]
    if "contradicted" in labels:
        prediction = "contradicted"
    elif "unknown" in labels:
        prediction = "unknown"
    else:
        prediction = "supported"
    return prediction, claims[0]["verification"].get("final_decision_policy"), claims[0].get("evidence", "")


def metrics(records):
    gold = [r["gold"] for r in records]
    pred = [r["prediction"] for r in records]
    n = len(records)
    correct = sum(g == p for g, p in zip(gold, pred))
    tp = sum(g == "contradicted" and p == "contradicted" for g, p in zip(gold, pred))
    fp = sum(g != "contradicted" and p == "contradicted" for g, p in zip(gold, pred))
    fn = sum(g == "contradicted" and p != "contradicted" for g, p in zip(gold, pred))
    tn = sum(g == "supported" and p == "supported" for g, p in zip(gold, pred))
    p, r = precision(tp, fp), recall(tp, fn)
    lat = sorted(x["latency_sec"] for x in records)
    return {
        "evaluated_claims": n, "correct_claims": correct, "accuracy": correct / n if n else 0.0,
        "precision": p, "recall": r, "f1": f1_score(p, r),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "unknown": sum(x == "unknown" for x in pred),
        "mean_latency_sec": sum(lat) / n if n else 0.0,
        "p50_latency_sec": lat[n // 2] if n else 0.0,
        "p90_latency_sec": lat[min(n - 1, int(0.9 * n))] if n else 0.0,
        # records saved before LLM-call logging existed have no llm_calls field
        "mean_llm_calls": (sum(x["llm_calls"] for x in records) / n
                           if n and all("llm_calls" in x for x in records) else None),
    }


def run_dir(tag=None):
    """Where a run's files go: evaluation/ for the published run, evaluation/runs/TAG/ for a tagged one."""
    if not tag:
        return EVAL_DIR
    path = os.path.join(EVAL_DIR, "runs", tag)
    os.makedirs(path, exist_ok=True)
    return path


def run_5_sets(dataset, samples_per_set=20, num_sets=5, seed=42, fresh=False, tag=None):
    out_dir = run_dir(tag)
    claims_file = os.path.join(out_dir, f"{dataset}_5_runs_claims.jsonl")
    if fresh and os.path.exists(claims_file):
        os.remove(claims_file)
    done = {}
    if os.path.exists(claims_file):
        with open(claims_file) as f:
            for line in f:
                rec = json.loads(line)
                done[(rec["set"], rec["claim"])] = rec

    sets = make_sets(load_items(dataset), samples_per_set, num_sets, seed)
    total = samples_per_set * num_sets
    print(f"{dataset.upper()}: {num_sets} sets x {samples_per_set} claims (seed {seed}); {len(done)}/{total} already done")

    detector = HallucinationDetector(retriever=WebRetriever())
    with open(claims_file, "a") as out:
        for set_num, batch in enumerate(sets, 1):
            for item in batch:
                if (set_num, item["claim"]) in done:
                    continue
                for attempt in (1, 2):
                    failures_before = get_failure_count()
                    calls_before = get_call_count()
                    log_before = len(get_call_log())
                    details = {} if tag else None
                    t0 = time.perf_counter()
                    try:
                        prediction, policy, evidence = verify(detector, item, details)
                    except Exception as e:
                        prediction, policy, evidence = "error", None, str(e)
                    latency = time.perf_counter() - t0
                    llm_calls = get_call_count() - calls_before
                    if get_failure_count() == failures_before and prediction != "error":
                        break
                    if attempt == 1:
                        print(f"  LLM failure on '{item['claim'][:60]}'; retrying in {LLM_RETRY_PAUSE_SEC}s")
                        time.sleep(LLM_RETRY_PAUSE_SEC)
                else:
                    print("\nLLM calls are failing (likely the daily quota). Stopping; re-run the same command to resume.")
                    return None
                rec = {
                    "set": set_num, "claim": item["claim"], "gold": LABEL_MAP[item["label"]],
                    "prediction": prediction, "decision_policy": policy, "latency_sec": latency, "llm_calls": llm_calls,
                    "fact_type": item.get("fact_type"), "evidence": evidence[:300],
                }
                if tag:
                    rec["llm_log"] = get_call_log()[log_before:]
                    rec["result"] = details.get("result")
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done[(set_num, item["claim"])] = rec
                print(f"[set {set_num}] {len(done)}/{total} gold={rec['gold']:12} pred={prediction:12} {latency:5.1f}s")
                time.sleep(0.5)

    records = [done[(s, item["claim"])] for s, batch in enumerate(sets, 1) for item in batch]
    set_results = [{"set": f"Set {s}", **metrics([r for r in records if r["set"] == s])} for s in range(1, num_sets + 1)]
    pooled = metrics(records)
    se = math.sqrt(pooled["accuracy"] * (1 - pooled["accuracy"]) / pooled["evaluated_claims"])
    pooled.update({"standard_error": se, "ci_lower_95": max(0.0, pooled["accuracy"] - 1.96 * se),
                   "ci_upper_95": min(1.0, pooled["accuracy"] + 1.96 * se)})
    routing = {}
    for r in records:
        routing[r["decision_policy"]] = routing.get(r["decision_policy"], 0) + 1

    print(f"\n{'Set':<8} | {'N':>3} | {'Correct':>7} | {'Accuracy':>8} | {'P':>4} | {'R':>4} | {'F1':>4} | {'Mean lat':>8}")
    for r in set_results + [{"set": "Pooled", **pooled}]:
        print(f"{r['set']:<8} | {r['evaluated_claims']:>3} | {r['correct_claims']:>7} | {r['accuracy']:>8.2%} | "
              f"{r['precision']:4.2f} | {r['recall']:4.2f} | {r['f1']:4.2f} | {r['mean_latency_sec']:7.1f}s")
    print(f"95% CI: [{pooled['ci_lower_95']:.2%}, {pooled['ci_upper_95']:.2%}] | routing: {routing}")

    payload = {
        "dataset": dataset, "seed": seed, "samples_per_set": samples_per_set, "num_sets": num_sets,
        "set_results": set_results, "pooled": pooled, "routing": routing,
        "failures": [{k: v for k, v in r.items() if k not in ("result", "llm_log")}
                     for r in records if r["gold"] != r["prediction"]],
    }
    out_file = os.path.join(out_dir, f"{dataset}_5_runs_results.json")
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved summary to {out_file}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-set stratified evaluation on FEVER and/or FreshQA.")
    parser.add_argument("--dataset", choices=["fever", "freshqa", "all"], default="all")
    parser.add_argument("--samples-per-set", type=int, default=20)
    parser.add_argument("--sets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fresh", action="store_true", help="Discard saved per-claim results and start over")
    parser.add_argument("--tag", default=None, help="Write the run to evaluation/runs/TAG/ with full per-claim details")
    args = parser.parse_args()

    for ds in (["fever", "freshqa"] if args.dataset == "all" else [args.dataset]):
        if run_5_sets(ds, args.samples_per_set, args.sets, args.seed, args.fresh, args.tag) is None:
            break

    # Exit immediately: interpreter shutdown joins every thread-pool worker, and a
    # DuckDuckGo/Wikipedia request stuck past its timeout would otherwise keep the process alive.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

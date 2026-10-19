import sys
import os
import csv
import json
import re
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.generator import generate_response

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "freshqa.csv")
CACHE_JSON_PATH = os.path.join(DATA_DIR, "freshqa_claims.json")

def synthesize_freshqa_cache():
    if not os.path.exists(CSV_PATH):
        from evaluation.benchmarks.freshqa_dataset import load_freshqa
        load_freshqa()

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("id,") or ",question," in line:
            start_idx = i
            break

    reader = list(csv.DictReader(lines[start_idx:]))
    valid_rows = []
    for r in reader:
        q = r.get("question", "").strip()
        ans = r.get("answer_0", "").strip()
        if not q or not ans:
            continue
        valid_rows.append(r)

    print(f"Total valid FreshQA Q&A pairs to synthesize: {len(valid_rows)}")

    # Check if existing cache exists
    cached_claims = {}
    if os.path.exists(CACHE_JSON_PATH):
        try:
            with open(CACHE_JSON_PATH, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                cached_claims = {item["raw_question"]: item for item in cached_data}
            print(f"Loaded {len(cached_claims)} previously synthesized claims from cache.")
        except Exception:
            cached_claims = {}

    batch_size = 25
    batches = []
    current_batch = []
    
    for r in valid_rows:
        q = r.get("question", "").strip()
        if q in cached_claims:
            continue
        current_batch.append(r)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    if batches:
        print(f"Synthesizing {len(batches)} batches ({sum(len(b) for b in batches)} items) using Gemini...")
        for b_idx, batch in enumerate(batches):
            print(f"Processing batch {b_idx + 1}/{len(batches)}...")
            prompt = """You are a linguistic expert creating an academic fact-checking benchmark.
Convert each Question and Answer into a single clean, natural, grammatically correct declarative sentence asserting a factual proposition.
- If False Premise is TRUE: The question presupposes a falsehood. Convert the presupposition into an affirmative declarative sentence asserting that falsehood (e.g. Q: 'When was League of Legends 2 released?' -> 'League of Legends 2 was released in the past.')
- If False Premise is FALSE: Form an affirmative declarative sentence stating the verified fact by combining the question and answer (e.g. Q: 'Who is CEO of Google?' A: 'Sundar Pichai' -> 'Sundar Pichai is the CEO of Google.')
Do not use conversational meta-language like 'Regarding', 'The answer is', or 'According to'. State the fact directly as a crisp claim.

Questions & Answers:
"""
            for i, r in enumerate(batch):
                q = r.get("question", "").strip()
                ans = r.get("answer_0", "").strip()
                fp = r.get("false_premise", "").strip().upper() == "TRUE"
                prompt += f"{i+1}. Q: {q} | A: {ans} | FalsePremise: {fp}\n"

            prompt += "\nOutput ONLY a valid JSON array of objects with keys 'index' (1-based int) and 'claim' (string):\n[{\"index\": 1, \"claim\": \"...\"}, ...]"

            try:
                raw_resp = generate_response(prompt)
                json_match = re.search(r'\[.*\]', raw_resp, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    for item in parsed:
                        idx = item.get("index", 1) - 1
                        if 0 <= idx < len(batch):
                            row = batch[idx]
                            q = row.get("question", "").strip()
                            ans = row.get("answer_0", "").strip()
                            fp = row.get("false_premise", "").strip().upper() == "TRUE"
                            ft = row.get("fact_type", "").strip().lower()
                            split_val = row.get("split", "").strip().upper()

                            cached_claims[q] = {
                                "claim": item.get("claim", "").strip(),
                                "label": "REFUTES" if fp else "SUPPORTS",
                                "raw_question": q,
                                "fact_type": ft or ("false-premise" if fp else "factual"),
                                "ground_truth": ans,
                                "split": split_val
                            }
                time.sleep(1.0)
            except Exception as e:
                print(f"Error synthesizing batch {b_idx + 1}: {e}")
                time.sleep(2.0)

    # Save full synthesized dataset
    final_list = list(cached_claims.values())
    with open(CACHE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2)
    print(f"Successfully cached {len(final_list)} high-precision declarative claims to {CACHE_JSON_PATH}")

if __name__ == "__main__":
    synthesize_freshqa_cache()

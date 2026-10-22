import os
import csv
import io
import re
import json
import urllib.request

FRESHQA_SHEET_URL = "https://docs.google.com/spreadsheets/d/1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM/export?format=csv"
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "freshqa.csv")

def _question_to_assertion(q: str) -> str:
    """Converts a false-premise question into a natural declarative assertion for verification."""
    clean_q = q.strip().rstrip('?').strip()
    
    # "Why are all quickly verifiable problems also quickly solvable" -> "All quickly verifiable problems are quickly solvable."
    m = re.match(r'^Why\s+(are|is)\s+(.+)$', clean_q, re.IGNORECASE)
    if m:
        return f"{m.group(2)} {m.group(1)}."

    # "When did X purchase Y" -> "X purchased Y."
    m = re.match(r'^(?:When|Why|Where|How|What\s+year)\s+did\s+(.+?)\s+([a-z]+)\s*(.*)$', clean_q, re.IGNORECASE)
    if m:
        subj, verb, rest = m.groups()
        irreg = {
            'move': 'moved', 'buy': 'bought', 'purchase': 'purchased', 'win': 'won',
            'go': 'went', 'make': 'made', 'land': 'landed', 'become': 'became',
            'release': 'released', 'reach': 'reached', 'die': 'died', 'start': 'started', 'end': 'ended'
        }
        v_past = irreg.get(verb.lower(), verb + ('d' if verb.endswith('e') else 'ed'))
        return f"{subj} {v_past} {rest}.".strip()

    # "In what year did X become Y"
    m = re.match(r'^In\s+what\s+year\s+did\s+(.+?)\s+([a-z]+)\s*(.*)$', clean_q, re.IGNORECASE)
    if m:
        subj, verb, rest = m.groups()
        irreg = {'become': 'became', 'start': 'started', 'win': 'won', 'occur': 'occurred'}
        v_past = irreg.get(verb.lower(), verb + ('d' if verb.endswith('e') else 'ed'))
        return f"{subj} {v_past} {rest}.".strip()

    # "When/Where was X founded/released"
    m = re.match(r'^(?:When|Where|In\s+what\s+year)\s+was\s+(.+?)\s+([a-z]+(?:ed|en|t))\s*(.*)$', clean_q, re.IGNORECASE)
    if m:
        return f"{m.group(1)} was {m.group(2)} {m.group(3)}.".strip()

    # "What is/was the name of X"
    m = re.match(r'^(?:What|Who)\s+(?:is|was)\s+the\s+name\s+of\s+(.+)$', clean_q, re.IGNORECASE)
    if m:
        return f"{m.group(1)} exists."

    # "Which X is/are most effective against Y"
    m = re.match(r'^Which\s+(.+?)\s+(?:is|are)\s+(.+)$', clean_q, re.IGNORECASE)
    if m:
        return f"{m.group(1)} are {m.group(2)}."

    return f"{clean_q}."

def _qa_to_claim(q: str, ans: str) -> str:
    """Converts a question and verified true answer into a natural affirmative declarative claim."""
    clean_q = q.strip().rstrip('?').strip()
    clean_ans = ans.strip()

    # If the answer already contains a full sentence:
    if len(clean_ans.split()) >= 4 and any(w in clean_ans.lower() for w in [" is ", " are ", " was ", " were ", " has ", " have ", " does ", " did ", " not "]):
        return clean_ans if clean_ans.endswith('.') else f"{clean_ans}."

    # "Who is/was <rest>" -> "<ans> is/was <rest>."
    m = re.match(r'^(?:Who|What)\s+(is|was|are|were)\s+(.+)$', clean_q, re.IGNORECASE)
    if m:
        copula, rest = m.groups()
        return f"{clean_ans} {copula} {rest}."

    # "Where is/was <rest>" -> "<rest> is/was in <ans>."
    m = re.match(r'^Where\s+(is|was)\s+(.+)$', clean_q, re.IGNORECASE)
    if m:
        copula, rest = m.groups()
        return f"{rest} {copula} in {clean_ans}."

    # "How many <items> <rest>" -> "There are <ans> <items>."
    m = re.match(r'^How\s+many\s+(.+)$', clean_q, re.IGNORECASE)
    if m:
        return f"There are {clean_ans} {m.group(1)}."

    return f"Regarding {clean_q}, the answer is {clean_ans}."

CLAIMS_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "freshqa_claims.json")

# Benchmarks run on never- and slow-changing questions only: fast-changing gold answers are dated
# snapshots that drift out of date, so errors on them measure label staleness, not the detector.
# Pass fact_types=None to load every fact type.
DEFAULT_FACT_TYPES = ("never-changing", "slow-changing")

def load_freshqa(split="all", fact_types=DEFAULT_FACT_TYPES):
    """
    Loads FreshQA (Google Research, 2023-2024), a benchmark for dynamic, fast-changing,
    slow-changing, and false-premise real-world world knowledge.

    Returns a list of dicts:
      {
        'claim': str,
        'label': 'SUPPORTS' | 'REFUTES',
        'raw_question': str,
        'fact_type': str,
        'ground_truth': str
      }
    """
    # 0. Check for pre-synthesized high-precision claims cache
    if os.path.exists(CLAIMS_CACHE_PATH):
        try:
            with open(CLAIMS_CACHE_PATH, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            filtered = []
            for item in cached_data:
                if split != "all" and item.get("split", "").upper() != split.upper():
                    continue
                if fact_types and item.get("fact_type", "").lower() not in fact_types:
                    continue
                filtered.append(item)
            if filtered:
                return filtered
        except Exception as e:
            print(f"Warning: Failed to load claims cache ({e}), falling back to raw sheet parsing.")

    # 1. Download & cache locally if missing
    if not os.path.exists(CACHE_PATH):
        print("Downloading FreshQA dataset from Google Research live repository...")
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        req = urllib.request.Request(FRESHQA_SHEET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read().decode("utf-8")
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            content = f.read()

    lines = content.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("id,") or ",question," in line:
            start_idx = i
            break

    reader = csv.DictReader(lines[start_idx:])
    samples = []

    for r in reader:
        q = r.get("question", "").strip()
        ans = r.get("answer_0", "").strip()
        ft = r.get("fact_type", "").strip().lower()
        is_false_premise = r.get("false_premise", "").strip().upper() == "TRUE"
        r_split = r.get("split", "").strip().upper()

        if not q:
            continue

        if split != "all" and r_split != split.upper():
            continue

        if fact_types and ft not in fact_types:
            continue

        if is_false_premise:
            # False premise implies a hallucinated assertion
            assertion = _question_to_assertion(q)
            samples.append({
                "claim": assertion,
                "label": "REFUTES",
                "raw_question": q,
                "fact_type": ft or "false-premise",
                "ground_truth": ans
            })
        elif ans and not ans.lower().startswith("no "):
            # Verified true answer converted to affirmative claim
            claim_text = _qa_to_claim(q, ans)
            samples.append({
                "claim": claim_text,
                "label": "SUPPORTS",
                "raw_question": q,
                "fact_type": ft or "factual",
                "ground_truth": ans
            })

    return samples

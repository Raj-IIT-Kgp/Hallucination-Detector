import json
import re
from datetime import datetime
from llm.generator import generate_response

_ROLES_AND_TITLES = {
    'ceo', 'cto', 'cfo', 'coo', 'md', 'president', 'vice president', 
    'director', 'managing director', 'founder', 'co-founder', 'co founder', 
    'chairman', 'member', 'author', 'inventor', 'creator', 'actor', 
    'minister', 'prime minister', 'governor', 'king', 'queen'
}

_STOPWORDS = {
    'the', 'a', 'an', 'in', 'on', 'at', 'this', 'that', 'he', 'she', 
    'it', 'they', 'his', 'her', 'their', 'its', 'who', 'which', 'where',
    'off', 'up', 'down', 'no', 'not', 'only', 'all', 'more', 'most', 
    'such', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should'
}

_AMBIGUOUS_PRONOUNS = {
    'he', 'she', 'it', 'they', 'him', 'her', 'them', 
    'his', 'their', 'its', 'this', 'these', 'those'
}

_KNOWN_NAMES = {
    "bharatpe": "BharatPe",
    "iphone": "iPhone",
    "ipad": "iPad",
    "ebay": "eBay",
    "youtube": "YouTube"
}

def _preserve_or_title(text, original_text):
    """Preserves internal capitalizations (e.g. BharatPe) or title-cases words."""
    orig_words = {w.lower(): w for w in original_text.split()}
    result_words = []
    for w in text.split():
        lw = w.lower()
        if lw in _KNOWN_NAMES:
            result_words.append(_KNOWN_NAMES[lw])
        elif lw in orig_words and any(c.isupper() for c in orig_words[lw][1:]):
            result_words.append(orig_words[lw])
        else:
            result_words.append(w.capitalize())
    return ' '.join(result_words)

def _extract_fast_entities(claim):
    """
    Entity-aware extractor for concise atomic statements.
    Extracts named entities (e.g. ['Ashneer Grover', 'BharatPe']) rather than
    querying entire sentences like 'ashneer grover is the co founder of bharatpe'.
    Returns list of up to 3 entities if clearly found, else None.
    """
    clean = claim.strip().rstrip(".").strip()
    lower = clean.lower()

    # If the statement uses figurative sobriquets / epithets (e.g. 'King of Pop', 'Iron Lady'),
    # route to LLM to resolve the true underlying entity (e.g. Michael Jackson, Margaret Thatcher).
    epithet_markers = ["king of", "queen of", "father of", "mother of", "iron lady", "first man", "first woman", "voice of", "bard of"]
    if any(m in lower for m in epithet_markers):
        return None

    # Creative works (albums, films, songs, books) frequently need parenthetical disambiguation
    # or have lowercase words in their titles (e.g. "Off the Wall"). Route to LLM.
    creative_markers = {"album", "film", "movie", "song", "single", "novel", "soundtrack", "play", "series", "opera", "track", "record"}
    words_set = set(re.findall(r'\b\w+\b', lower))
    if words_set & creative_markers:
        return None

    # 1. Relational phrase splitting (captures explicit Subject-Relation-Object pairs)
    connectors = [
        r'\s+(?:is|was|are|were)\s+(?:the\s+|a\s+|an\s+)?(?:co-founder|co founder|founder|ceo|president|director|creator|author|inventor)\s+of\s+',
        r'\s+(?:is|was|are|were)\s+from\s+',
        r'\s+(?:is|was|are|were)\s+(?:the\s+|a\s+|an\s+)?(?:capital|headquarters|center|member|part|city|state|country)\s+(?:of|in)\s+',
        r'\s+(?:is|was|are|were)\s+(?:born|located|situated|based|headquartered)\s+in\s+',
        r'\s+(?:was|were)\s+(?:founded|created|invented|directed|acquired|bought|written|discovered)\s+by\s+',
        r'\s+(?:co-founded|co founded|founded|created|invented|directed|acquired|discovered)\s+',
        r'\s+(?:is|was|are|were)\s+(?:a|an|the)\s+'
    ]

    for pattern in connectors:
        parts = re.split(pattern, clean, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            subj = parts[0].strip()
            obj = parts[1].strip()
            # Clean leading/trailing prepositions, articles
            obj = re.sub(r'^(?:the|a|an|of|in|at|by|to|for|from)\s+', '', obj, flags=re.IGNORECASE).strip()
            subj = re.sub(r'^(?:the|a|an)\s+', '', subj, flags=re.IGNORECASE).strip()
            
            entities = []
            if subj and subj.lower() not in _STOPWORDS and subj.lower() not in _ROLES_AND_TITLES:
                entities.append(_preserve_or_title(subj, clean))
            if obj:
                # Remove temporal clauses like 'in 2018'
                obj_clean = re.split(r'\s+(?:in|on|at|during|since)\s+\d{4}', obj, flags=re.IGNORECASE)[0].strip()
                orig_obj_match = re.search(r'\b' + re.escape(obj_clean) + r'\b', clean)
                is_capitalized = orig_obj_match and any(c.isupper() for c in orig_obj_match.group(0))
                if obj_clean and obj_clean.lower() not in _STOPWORDS and obj_clean.lower() not in _ROLES_AND_TITLES and is_capitalized:
                    entities.append(_preserve_or_title(obj_clean, clean))
            if len(entities) >= 1:
                return entities[:3]

    return None

def generate_queries(claim, context=None):
    """
    Transforms a complex claim into a JSON array of up to 3 optimized search queries for Wikipedia.
    Now includes temporal context tracking.
    """
    # ── FAST-PATH FOR CONCISE / ATOMIC CLAIMS ──
    words = claim.strip().split()
    lower_claim = claim.lower()

    # Check if claim contains ambiguous pronouns requiring context coreference resolution
    claim_tokens = {w.strip(".,!?:;\"'()[]{}") for w in lower_claim.split()}
    has_pronouns = bool(claim_tokens & _AMBIGUOUS_PRONOUNS)

    # Fast-path eligible when concise (<= 15 words) and:
    # 1. Claim has no ambiguous pronouns (self-contained entities), OR
    # 2. Context is absent / identical to claim (no coreference context exists to resolve against)
    is_context_independent = (not has_pronouns) or (context is None) or (context.strip() == claim.strip())

    if len(words) <= 15 and is_context_independent:
        clean_claim = claim.strip().rstrip(".").strip()
        if clean_claim:
            entities = _extract_fast_entities(clean_claim)
            if entities:
                requires_current = any(w in lower_claim for w in ["current", "currently", "now", "present", "today", "as of", "active"])
                is_scientific = any(w in lower_claim for w in ["celsius", "fahrenheit", "quantum", "temperature", "species", "molecule", "climate", "dna"])
                return {
                    "queries": entities,
                    "domain": "scientific" if is_scientific else "general",
                    "requires_current_state": requires_current,
                    "route": "fast_path",
                }

    current_date = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
Identify up to 3 distinct Wikipedia articles that would best verify this claim.
If the claim mentions an event and a person/country, extract BOTH as separate entities.
Use the Context paragraph to figure out what pronouns refer to.

CRITICAL RULE 1: You MUST forcefully disambiguate ambiguous words. If an entity is just a single word (e.g., 'no', 'reconstructions', 'Pixels', 'Apple'), you MUST add contextual keywords to the query, such as 'Temperature reconstructions (climate)', or 'Pixels (2015 film)'. Do not allow single-word vague queries.
CRITICAL RULE 2: You MUST return at least one query. NEVER return an empty array []. If the claim is a weird fragment, output that exact fragment as the query!
CRITICAL RULE 3: You MUST classify the domain of the claim. If the claim contains highly specific scientific metrics (like climate data, physics, or biology), output "scientific". Otherwise, output "general".
CRITICAL RULE 4: You MUST determine if the claim is temporal. If the claim relies on the present day (e.g. "is currently", "the current CEO", or implicit present tense like "Tim Cook is the CEO of Apple"), set "requires_current_state" to true.

Output ONLY a valid JSON object in this format: {{"queries": ["Entity 1", "Entity 2"], "domain": "scientific", "requires_current_state": false}}. Do NOT include markdown blocks, explanation, or extra text.

Current System Date: {current_date}

EXAMPLES:
Claim: "The current CEO of OpenAI is Emmett Shear."
Output: {{"queries": ["OpenAI", "Emmett Shear"], "domain": "general", "requires_current_state": true}}

Claim: "Albert Einstein was born in 1950."
Output: {{"queries": ["Albert Einstein"], "domain": "general", "requires_current_state": false}}

Claim: "Global surface temperatures have continued to rise steadily beneath short-term natural cooling effects."
Output: {{"queries": ["Global surface temperatures", "Natural cooling effects"], "domain": "scientific", "requires_current_state": false}}

Context: {context if context else claim}
Claim: {claim}
Output:
"""
    
    try:
        response = generate_response(prompt, tier="light")
    except Exception as e:
        print(f"[QueryGenerator] LLM connection unavailable ({e}). Falling back to claim heuristic.")
        return {"queries": [claim], "domain": "general", "requires_current_state": False, "route": "fallback"}
    
    try:
        # Clean up the LLM output in case it wrapped it in markdown
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        parsed = json.loads(cleaned.strip())
        
        # Handle backward compatibility if the LLM still returns a list
        if isinstance(parsed, list):
            queries = parsed
            domain = "general"
            requires_current_state = False
        else:
            queries = parsed.get("queries", [])
            domain = parsed.get("domain", "general")
            requires_current_state = parsed.get("requires_current_state", False)
            
        queries = [str(q).strip() for q in queries if str(q).strip()]
        if not queries:
            queries = [claim]
            
        return {"queries": queries, "domain": domain, "requires_current_state": requires_current_state, "route": "llm"}
    except Exception as e:
        print(f"[QueryGenerator] Failed to parse JSON. Falling back. Error: {e}")
        # Fallback if parsing fails
        query = response.strip()
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        return {"queries": [query], "domain": "general", "requires_current_state": False, "route": "fallback"}

def rewrite_queries(claim, bad_evidence, previous_queries):
    """
    Self-Reflective Retrieval agent. Evaluates why the previous search failed
    and synthesizes new, divergent search queries to try again.
    """
    prompt = f"""
You are a Self-Reflective Search Agent. 
We are trying to verify this claim: "{claim}"

We previously searched using these queries: {previous_queries}
However, the retrieved evidence was unhelpful or irrelevant:
"{bad_evidence[:1000]}..."

Analyze why the previous search failed. Then, generate up to 2 NEW, DIVERGENT Wikipedia search queries.
Think outside the box. If the claim implies a specific event, person, or alternative name, use that.

Output ONLY a valid JSON array of strings. Do not include markdown blocks or explanations.
Example: ["Entity Alternate Name", "Specific Event Name"]

Output:
"""
    try:
        response = generate_response(prompt, tier="light")
    except Exception as e:
        print(f"[QueryGenerator] LLM rewrite failed: {e}")
        return []
    try:
        # 1. Clean markdown code blocks if present
        cleaned = response.strip()
        if "```" in cleaned:
            match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass

        # 2. Extract any JSON array containing quoted strings
        candidates = re.findall(r'\[\s*(?:"[^"]*"(?:\s*,\s*"[^"]*")*|\'[^\']*\'(?:\s*,\s*\'[^\']*\')*)\s*\]', response, re.DOTALL)
        for cand in reversed(candidates):
            try:
                cand_clean = cand.strip()
                if cand_clean.startswith("['"):
                    cand_clean = cand_clean.replace("'", '"')
                parsed = json.loads(cand_clean)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and len(parsed) > 0:
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                continue

        # 3. Fallback: search any bracketed segment
        bracket_candidates = re.findall(r'\[([^\[\]]+)\]', response, re.DOTALL)
        for cand in reversed(bracket_candidates):
            try:
                parsed = json.loads(f"[{cand}]")
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and len(parsed) > 0:
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                continue
    except Exception as e:
        print(f"[QueryGenerator] Failed to parse rewrite JSON. Error: {e}")
        
    return []


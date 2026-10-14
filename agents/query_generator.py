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
    'it', 'they', 'his', 'her', 'their', 'its', 'who', 'which', 'where'
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
    
    # 1. Capitalized Entity Sequences (if text has proper capitalization)
    has_mixed_case = any(c.isupper() for c in clean[1:])
    if has_mixed_case:
        cap_entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', clean)
        cleaned_caps = []
        for e in cap_entities:
            words = e.split()
            while words and words[0].lower() in _STOPWORDS:
                words = words[1:]
            if words:
                cleaned_caps.append(" ".join(words))

        filtered = [
            e for e in cleaned_caps 
            if e.lower() not in _STOPWORDS and e.lower() not in _ROLES_AND_TITLES and len(e.strip()) > 1
        ]
        if len(filtered) >= 1:
            return filtered[:3]

    # 2. Relational phrase splitting (for lowercase or uncapitalized input)
    connectors = [
        r'\s+(?:is|was|are|were)\s+(?:the\s+|a\s+|an\s+)?(?:co-founder|co founder|founder|ceo|president|director|creator|author|inventor)\s+of\s+',
        r'\s+(?:is|was|are|were)\s+from\s+',
        r'\s+(?:is|was|are|were)\s+(?:the\s+|a\s+|an\s+)?(?:capital|headquarters|center|member|part)\s+of\s+',
        r'\s+(?:is|was|are|were)\s+(?:born|located|situated|based|headquartered)\s+in\s+',
        r'\s+(?:was|were)\s+(?:founded|created|invented|directed|acquired|bought|written|discovered)\s+by\s+',
        r'\s+(?:co-founded|co founded|founded|created|invented|directed|acquired|discovered)\s+'
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
                if obj_clean and obj_clean.lower() not in _STOPWORDS and obj_clean.lower() not in _ROLES_AND_TITLES:
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
    # When claim is already a single concise statement (<= 12 words) without complex context
    words = claim.strip().split()
    lower_claim = claim.lower()
    if len(words) <= 12 and (context is None or context.strip() == claim.strip() or len(context.strip().split()) <= 15):
        clean_claim = claim.strip().rstrip(".").strip()
        if clean_claim:
            entities = _extract_fast_entities(clean_claim)
            if entities:
                requires_current = any(w in lower_claim for w in ["current", "currently", "now", "present", "today", "as of", "active"])
                is_scientific = any(w in lower_claim for w in ["celsius", "fahrenheit", "quantum", "temperature", "species", "molecule", "climate", "dna"])
                return {
                    "queries": entities,
                    "domain": "scientific" if is_scientific else "general",
                    "requires_current_state": requires_current
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
        response = generate_response(prompt)
    except Exception as e:
        print(f"[QueryGenerator] LLM connection unavailable ({e}). Falling back to claim heuristic.")
        return {"queries": [claim], "domain": "general", "requires_current_state": False}
    
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
            
        return {"queries": queries, "domain": domain, "requires_current_state": requires_current_state}
    except Exception as e:
        print(f"[QueryGenerator] Failed to parse JSON. Falling back. Error: {e}")
        # Fallback if parsing fails
        query = response.strip()
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        return {"queries": [query], "domain": "general", "requires_current_state": False}

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
        response = generate_response(prompt)
    except Exception as e:
        print(f"[QueryGenerator] LLM rewrite failed: {e}")
        return []
    try:
        cleaned = response.strip()
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        elif cleaned.startswith("```json"):
            cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
        queries = json.loads(cleaned.strip())
        if isinstance(queries, list) and len(queries) > 0:
            return queries
    except Exception as e:
        print(f"[QueryGenerator] Failed to parse rewrite JSON. Error: {e}")
        
    return []


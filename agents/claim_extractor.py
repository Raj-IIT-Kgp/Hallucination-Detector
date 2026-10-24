import nltk
import json
from llm.generator import generate_response

def extract_claims(answer):
    return extract_claims_with_route(answer)[0]


def extract_claims_with_route(answer):
    """
    Extracts individual claims from a paragraph.
    Upgraded to use an LLM for Atomic Fact Decomposition, breaking complex sentences
    into simple, independent facts.
    Includes an atomic fast-path to eliminate redundant 15-20s LLM calls for single-statement inputs.
    Returns (claims, route), where route is how the text was decomposed (shown in the UI):
    "fast_path" (already atomic, no LLM call), "llm", or "fallback" (NLTK sentences after an LLM failure).
    """
    if not answer or not answer.strip():
        return [], None

    # Fast-path: If the input is already a single short atomic assertion without clauses, skip redundant LLM call
    try:
        sentences = nltk.sent_tokenize(answer.strip())
        if len(sentences) == 1:
            s = sentences[0].strip()
            words = s.split()
            complex_markers = [";", "—", "--", ",", " and ", " but ", " although ", " whereas ", " which ", " because ", " while ", " who ", " where ", " that ", " with "]
            if len(words) <= 15 and not any(marker in s.lower() for marker in complex_markers):
                return [s], "fast_path"
    except Exception:
        pass

    prompt = f"""

Decompose the following text into a list of independent, atomic facts.
CRITICAL RULES:
1. You MUST resolve all pronouns (he, she, it) and implicit references (e.g. 'the associate', 'the film', 'the city') to their exact proper nouns from the context. Do not leave any vague references in the atomic facts.
2. STRICT POLARITY PRESERVATION: You MUST strictly preserve the exact truth value, falsity, and polarity (e.g. negations, 'not', 'never', 'outside', 'false', 'before/after') of the original text. NEVER 'correct', invert, or normalize false or hallucinated assertions into true ones.

Text: "{answer}"

If the text is a Question/Answer pair, first synthesize it into a declarative statement (e.g., "The answer to the question X is Y"), and then decompose that statement into atomic facts.

Output ONLY a valid JSON array of strings. Do not include markdown formatting or explanations.
Example: ["Tim Cook replaced Steve Jobs in 2011.", "Tim Cook is the CEO of Apple."]
"""
    try:
        response = generate_response(prompt, tier="heavy")
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        claims = json.loads(cleaned.strip())
        if isinstance(claims, list) and len(claims) > 0:
            return [str(c).strip() for c in claims], "llm"
    except Exception as e:
        print(f"[ClaimExtractor] Failed to decompose facts via LLM. Falling back to NLTK. Error: {e}")
        
    # Fallback to NLTK
    claims = nltk.sent_tokenize(answer)
    claims = [claim.strip() for claim in claims if claim.strip()]
    
    return claims, "fallback"
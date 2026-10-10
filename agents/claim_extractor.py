import nltk
import json
from llm.generator import generate_response

def extract_claims(answer):
    """
    Extracts individual claims from a paragraph.
    Upgraded to use an LLM for Atomic Fact Decomposition, breaking complex sentences
    into simple, independent facts.
    """
    prompt = f"""
Decompose the following text into a list of independent, atomic facts.
An atomic fact is a short, simple sentence containing exactly one assertion.
CRITICAL: You MUST resolve all pronouns (he, she, it) and implicit references (e.g. 'the associate', 'the film', 'the city') to their exact proper nouns from the context. Do not leave any vague references in the atomic facts.

Text: "{answer}"

If the text is a Question/Answer pair, first synthesize it into a declarative statement (e.g., "The answer to the question X is Y"), and then decompose that statement into atomic facts.

Output ONLY a valid JSON array of strings. Do not include markdown formatting or explanations.
Example: ["Tim Cook replaced Steve Jobs in 2011.", "Tim Cook is the CEO of Apple."]
"""
    try:
        response = generate_response(prompt)
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        claims = json.loads(cleaned.strip())
        if isinstance(claims, list) and len(claims) > 0:
            return [str(c).strip() for c in claims]
    except Exception as e:
        print(f"[ClaimExtractor] Failed to decompose facts via LLM. Falling back to NLTK. Error: {e}")
        
    # Fallback to NLTK
    claims = nltk.sent_tokenize(answer)
    claims = [claim.strip() for claim in claims if claim.strip()]
    
    return claims
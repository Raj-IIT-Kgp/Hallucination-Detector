import json
from llm.generator import generate_response

def generate_queries(claim, context=None):
    """
    Transforms a complex claim into a JSON array of up to 3 optimized search queries for Wikipedia.
    """
    prompt = f"""
Identify up to 3 distinct Wikipedia articles that would best verify this claim.
If the claim mentions an event and a person/country, extract BOTH as separate entities.
Use the Context paragraph to figure out what pronouns refer to.

CRITICAL RULE 1: You MUST forcefully disambiguate ambiguous words. If an entity is just a single word (e.g., 'no', 'reconstructions', 'Pixels', 'Apple'), you MUST add contextual keywords to the query, such as 'Temperature reconstructions (climate)', or 'Pixels (2015 film)'. Do not allow single-word vague queries.
CRITICAL RULE 2: You MUST return at least one query. NEVER return an empty array []. If the claim is a weird fragment, output that exact fragment as the query!

Output ONLY a valid JSON array of strings (e.g. ["Entity 1", "Entity 2"]). Do NOT include markdown blocks, explanation, or extra text.

EXAMPLES:
Claim: "Albert Einstein was born in 1950."
Output: ["Albert Einstein"]

Claim: "Raj invented facebook in 2005."
Output: ["Facebook", "Raj"]

Claim: "France and Argentina played in the 2022 World Cup Final."
Output: ["2022 FIFA World Cup", "France national football team", "Argentina national football team"]

Claim: "DreamWorks Animation produced Pixels."
Output: ["Pixels (2015 film)", "DreamWorks Animation"]

Context: {context if context else claim}
Claim: {claim}
Output:
"""
    
    response = generate_response(prompt)
    
    try:
        # Clean up the LLM output in case it wrapped it in markdown
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        queries = json.loads(cleaned.strip())
        
        if not isinstance(queries, list):
            queries = [str(queries)]
            
        if not queries or len(queries) == 0:
            queries = [claim]
            
        return queries
    except Exception as e:
        print(f"[QueryGenerator] Failed to parse JSON. Falling back. Error: {e}")
        # Fallback to a single string query if parsing fails
        query = response.strip()
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        return [query]


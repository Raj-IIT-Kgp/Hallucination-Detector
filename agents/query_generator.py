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
CRITICAL RULE 3: You MUST classify the domain of the claim. If the claim contains highly specific scientific metrics (like climate data, physics, or biology), output "scientific". Otherwise, output "general".

Output ONLY a valid JSON object in this format: {{"queries": ["Entity 1", "Entity 2"], "domain": "scientific"}}. Do NOT include markdown blocks, explanation, or extra text.

EXAMPLES:
Claim: "Albert Einstein was born in 1950."
Output: {{"queries": ["Albert Einstein"], "domain": "general"}}

Claim: "Raj invented facebook in 2005."
Output: {{"queries": ["Facebook", "Raj"], "domain": "general"}}

Claim: "Global surface temperatures have continued to rise steadily beneath short-term natural cooling effects."
Output: {{"queries": ["Global surface temperatures", "Natural cooling effects"], "domain": "scientific"}}

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
        
        parsed = json.loads(cleaned.strip())
        
        # Handle backward compatibility if the LLM still returns a list
        if isinstance(parsed, list):
            queries = parsed
            domain = "general"
        else:
            queries = parsed.get("queries", [])
            domain = parsed.get("domain", "general")
            
        if not queries or len(queries) == 0:
            queries = [claim]
            
        return {"queries": queries, "domain": domain}
    except Exception as e:
        print(f"[QueryGenerator] Failed to parse JSON. Falling back. Error: {e}")
        # Fallback if parsing fails
        query = response.strip()
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        return {"queries": [query], "domain": "general"}


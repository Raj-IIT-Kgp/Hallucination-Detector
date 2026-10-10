import json
from datetime import datetime
from llm.generator import generate_response

def generate_queries(claim, context=None):
    """
    Transforms a complex claim into a JSON array of up to 3 optimized search queries for Wikipedia.
    Now includes temporal context tracking.
    """
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
            requires_current_state = False
        else:
            queries = parsed.get("queries", [])
            domain = parsed.get("domain", "general")
            requires_current_state = parsed.get("requires_current_state", False)
            
        if not queries or len(queries) == 0:
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
    response = generate_response(prompt)
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        queries = json.loads(cleaned.strip())
        if isinstance(queries, list) and len(queries) > 0:
            return queries
    except Exception as e:
        print(f"[QueryGenerator] Failed to parse rewrite JSON. Error: {e}")
        
    return []


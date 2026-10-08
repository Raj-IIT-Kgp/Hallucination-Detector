import json
from llm.generator import generate_response


def extract_entities(claim: str, context: str = None) -> list[dict]:
    """
    Uses the LLM to extract named entities from a claim and resolve them
    to canonical Wikipedia page titles. Returns a list of entity dicts
    with 'name' and 'wikipedia_title' keys.
    """
    ctx = context if context else claim

    prompt = f"""You are an advanced Entity Linking system.
Extract all named entities (people, organizations, places, events, works) from the Claim below.
CRITICAL: If the claim references an entity implicitly through a relationship (e.g., "The director of Inception", "The 44th US President"), you MUST use your world knowledge to resolve who that person/entity actually is and include them in the list.

RULES:
1. Output ONLY a valid JSON array. No markdown.
2. Each item must have "name" (the entity as mentioned, or the resolved name) and "wikipedia_title" (canonical Wikipedia title).
3. Maximum 5 entities. If there are none, output [].

EXAMPLES:
Claim: "The director of Inception was born in London."
Output: [{{"name": "Inception", "wikipedia_title": "Inception"}}, {{"name": "London", "wikipedia_title": "London"}}, {{"name": "Christopher Nolan", "wikipedia_title": "Christopher Nolan"}}]

Claim: "Einstein won the Nobel Prize in Physics in 1922."
Output: [{{"name": "Einstein", "wikipedia_title": "Albert Einstein"}}, {{"name": "Nobel Prize in Physics", "wikipedia_title": "Nobel Prize in Physics"}}]

Context: {ctx}
Claim: {claim}
Output:"""

    response = generate_response(prompt).strip()

    try:
        # Strip markdown fences if present
        cleaned = response
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        entities = json.loads(cleaned.strip())
        if not isinstance(entities, list):
            return []
        return entities
    except Exception as e:
        print(f"[EntityLinker] Failed to parse entities: {e}. Raw: {response}")
        return []

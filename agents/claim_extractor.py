import json
from llm.generator import generate_response


def extract_claims(answer):

    prompt = f"""
You are an expert factual claim extractor.
Your task is to extract all distinct factual claims from the provided text.
You must output your response ONLY as a JSON array of strings. Do not include any other text, explanation, or markdown.

Example Input text:
"Paris is the capital of France. The Eiffel Tower was built in 1889."
Example Output:
["Paris is the capital of France", "The Eiffel Tower was built in 1889"]

Input text to extract claims from:
{answer}

Output:
"""


    response = generate_response(prompt)
    
    # Strip markdown block if present (common with local LLMs)
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    try:
        claims = json.loads(cleaned_response)
    except Exception as e:
        print(f"Error parsing JSON: {e}\nRaw response: {response}")
        claims = []

    return claims
import os
from llm.gemini import generate_gemini_response
from llm.ollama import generate_ollama_response

def generate_response(prompt):
    provider = os.environ.get("MODEL_PROVIDER", "ollama").lower()
    
    if provider == "ollama":
        return generate_ollama_response(prompt)
    else:
        return generate_gemini_response(prompt)
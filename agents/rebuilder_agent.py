import json
from llm.generator import generate_response

def rebuild_response(original_text, report):
    """
    Takes the original hallucinated text and the JSON report of verified claims,
    and asks the LLM to rewrite the entire text incorporating the factual corrections.
    """
    
    # Do a literal string replacement for each hallucinated claim.
    # This prevents the LLM from hallucinating new facts during the rebuild phase.
    final_text = original_text
    
    for claim in report["claims"]:
        if claim["correction"] is not None:
            # Replace the original false claim with the factual correction
            final_text = final_text.replace(claim["claim"], claim["correction"])
            
    return final_text

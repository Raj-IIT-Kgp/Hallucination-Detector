from llm.generator import generate_response



def correct_claim(claim, evidence, feedback=None):

    prompt = f"""
You are a factual correction agent. You are given a false claim and a piece of factual evidence.
Your task is to rewrite the false claim so that it is factually correct, based ONLY on the evidence provided.

RULES:
1. Base your correction STRICTLY on the evidence. Do not invent names, dates, or facts.
2. If the evidence states there is no record of the event, explicitly state that in your correction (e.g., "There is no official record of...").
3. If the evidence does not contain the exact answer, simply rewrite the claim to reflect what the evidence DOES say.
4. Output ONLY the corrected sentence. Do not include explanations.

False Claim: {claim}
Factual Evidence: {evidence}
"""
    
    if feedback:
        prompt += f"\nCRITIC FEEDBACK ON YOUR PREVIOUS ATTEMPT: {feedback}\nPlease try again and fix the issue mentioned in the feedback.\n"
        
    prompt += "\nCorrected Claim:\n"


    response = generate_response(prompt)


    return response.strip()
from llm.generator import generate_response

def review_correction(claim, draft_correction, evidence):
    """
    Acts as a Critic Agent that reviews the Correction Agent's work.
    Ensures the correction is logically sound and strictly matches the evidence.
    """
    
    prompt = f"""
You are a strict Critic Agent. Your job is to review a draft correction for a factual error.
You must ensure the draft correction is 100% supported by the evidence and does not introduce new hallucinations or weird logic.

Original False Claim: {claim}
Factual Evidence: {evidence}
Draft Correction (written by another agent): {draft_correction}

CHAIN OF THOUGHT REASONING:
Step 1: Identify all entities, dates, and relationships in the Draft Correction.
Step 2: Check if EACH entity/date/relationship is explicitly stated in the Factual Evidence.
Step 3: Compare the Draft Correction with the Original False Claim. Does it properly fix the error without adding unrelated fluff?
Step 4: Based on these steps, decide if it PASSES or FAILS.

RULES:
1. You must show your step-by-step reasoning first.
2. The LAST line of your response must be exactly "VERDICT: PASS" if the correction is perfect, or "VERDICT: FAIL - [reason]" if it has weird logic, hallucinations, or irrelevant details.

Reasoning and Verdict:
"""
    
    response = generate_response(prompt).strip()
    print(f"    Critic reasoning:\n{response}")
    
    lines = response.split('\n')
    verdict_line = next((line for line in reversed(lines) if line.strip().upper().startswith("VERDICT:")), response)
    
    if "VERDICT: PASS" in verdict_line.upper():
        return {"is_accurate": True, "feedback": None}
    else:
        # Extract the reason after "FAIL" if present
        feedback = verdict_line
        if "FAIL" in verdict_line.upper():
            parts = verdict_line.split("FAIL", 1)
            if len(parts) > 1:
                feedback = parts[1].strip(" :-")
        return {"is_accurate": False, "feedback": feedback}

def nli_override(claim, evidence, domain="general"):
    """
    Uses the LLM's common sense reasoning to override the rigid NLI model
    when the NLI model outputs 'unknown'.
    """
    
    academic_rule = ""
    if domain == "scientific":
        academic_rule = "\nCRITICAL RULE: You are evaluating peer-reviewed scientific abstracts. Academic literature uses cautious, probabilistic language (e.g., 'suggests', 'likely', 'potential'). You must interpret this cautious language as affirmative support (SUPPORTED) if the underlying statistical finding aligns with the claim.\n"
        
    prompt = f"""
You are an expert fact-checker with deep reading comprehension skills.
You need to determine if a claim is SUPPORTED or CONTRADICTED by the provided evidence.

Claim: "{claim}"
Evidence: "{evidence}"

CHAIN OF THOUGHT REASONING:
Step 1: Identify all entities and the core relationship/action in the Claim.
Step 2: Read the Evidence. Does it contain the same entities?
Step 3: Check if the relationship/action in the Evidence matches or contradicts the Claim. Be extremely precise about subtle word changes.
{academic_rule}
RULES:
1. Show your step-by-step reasoning first.
2. The LAST line of your response must be exactly "VERDICT: SUPPORTED", "VERDICT: CONTRADICTED", or "VERDICT: UNKNOWN".

Reasoning and Verdict:
"""
    response = generate_response(prompt).strip()
    
    lines = response.split('\n')
    verdict_line = next((line for line in reversed(lines) if line.strip().upper().startswith("VERDICT:")), response).upper()
    
    if "SUPPORTED" in verdict_line:
        return "supported"
    elif "CONTRADICTED" in verdict_line:
        return "contradicted"
    else:
        return "unknown"

def evaluate_absence(claim, entity_query):
    """
    Evaluates whether the absence of a claim from a comprehensive Wikipedia page
    implies that the claim is false (hallucinated).
    """
    prompt = f"""
We searched Wikipedia for the main entity '{entity_query}' to verify the following claim:
"{claim}"

The comprehensive Wikipedia page does NOT mention this claim at all.

RULES:
1. If this claim describes a massive, world-altering, or highly notable event (e.g. moving a famous monument, a major war, a world leader dying), it would definitely be on the Wikipedia page if it were true. Since it's missing, it is a hallucination. Output "CONTRADICTED".
2. If this claim describes a very minor, insignificant detail (like what someone ate for breakfast, or a tiny trivia fact), it might just be missing. Output "UNKNOWN".

Based on the magnitude of the claim "{claim}", does its absence from Wikipedia prove it is false?
OUTPUT ONLY ONE WORD: CONTRADICTED or UNKNOWN.
"""
    response = generate_response(prompt).strip().upper()
    
    if "CONTRADICTED" in response:
        return "contradicted"
    else:
        return "unknown"

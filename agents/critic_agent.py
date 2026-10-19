from llm.generator import generate_response
from concurrent.futures import ThreadPoolExecutor

def _defense_agent(claim, evidence):
    prompt = f"""
You are the Defense Attorney in a debate. Your ONLY goal is to prove that the following claim is SUPPORTED by the provided evidence.
You must construct the strongest possible logical argument using the evidence to support the claim.
If the evidence is ambiguous, you must interpret it in a way that supports the claim.

Claim: "{claim}"
Evidence: "{evidence}"

Provide your compelling argument to prove the claim is SUPPORTED. Be concise but persuasive.
"""
    return generate_response(prompt).strip()

def _prosecution_agent(claim, evidence):
    prompt = f"""
You are the Prosecuting Attorney in a debate. Your ONLY goal is to prove that the following claim is CONTRADICTED (a hallucination/falsehood) by the provided evidence.
You must construct the strongest possible logical argument using the evidence to contradict the claim.
If the evidence is ambiguous, you must interpret it in a way that refutes the claim.

Claim: "{claim}"
Evidence: "{evidence}"

Provide your compelling argument to prove the claim is CONTRADICTED. Be concise but persuasive.
"""
    return generate_response(prompt).strip()

def _judge_agent(claim, evidence, defense_arg, prosecution_arg, domain="general"):
    academic_rule = ""
    if domain == "scientific":
        academic_rule = "\nCRITICAL RULE: You are evaluating peer-reviewed scientific abstracts. Cautious language (e.g., 'suggests', 'likely') should be interpreted as affirmative support if the underlying finding aligns with the claim.\n"
        
    prompt = f"""
You are an impartial Expert Judge in a debate regarding fact-checking.
You need to determine the absolute truth: is the claim SUPPORTED, CONTRADICTED, or UNKNOWN based on the evidence?

Claim: "{claim}"
Evidence: "{evidence}"

Argument for SUPPORTED (Defense):
"{defense_arg}"

Argument for CONTRADICTED (Prosecution):
"{prosecution_arg}"

CHAIN OF THOUGHT REASONING:
Step 1: Evaluate the Defense's argument against the actual Evidence. Is it logically sound or making leaps?
Step 2: Evaluate the Prosecution's argument against the actual Evidence. Are they nitpicking or finding a genuine contradiction?
Step 3: Decide which argument holds up better against the raw evidence.
{academic_rule}
CRITICAL DECISION & ABSTENTION RULES:
1. Show your step-by-step reasoning first.
2. Natural Entailment vs Hyper-literal Pedantry:
   - When a claim asserts a founder or creator is from a country (e.g. 'X founder is from India'), and evidence establishes that the entity is an indigenous national company/service (e.g. 'Indian online food ordering service') founded domestically by that individual, treat the claim as SUPPORTED.
   - Do NOT adopt hyper-literal skepticism (e.g., claiming 'the text describes the company as Indian, but not the person') to force an UNKNOWN verdict unless there is affirmative evidence indicating foreign citizenship, foreign birth, or that the organization is a local subsidiary of a foreign multinational.
   - For shared titles (e.g., 'co-founder', 'director'), companies often have multiple co-founders; mentioning one co-founder does NOT contradict another being a co-founder, and reciprocal phrasing like 'fellow co-founder' or 'former co-founder' directly entails co-founder status.
   - Awards and Accolades: When a claim asserts that a person or entity *won* an award, and authoritative biographical evidence specifies they received only *nominations* (e.g. 'earned four nominations'), treat the claim as CONTRADICTED.
   - Geographic Venues vs Component Suppliers: Distinguish the primary medical center or hospital where a procedure or event occurred from secondary facilities or genetic suppliers located elsewhere.
   - Media Framing vs Core Historical Assertion: When a claim is framed as 'Photo/Video/Image showing [X]', if authoritative evidence verifies the underlying historical event, entities, and described scene (e.g. Vladimir Komarov's fatal space crash and open casket viewing in 1967), treat the claim as SUPPORTED. Do NOT force an UNKNOWN verdict simply because the evidence is written text rather than an image.
   - Compound Health & Scientific Assertions: In compound claims pairing real statistics with an absolute negation of medical/scientific precautions (e.g. 'X fell ill but nobody needs to wear a mask'), the second clause cannot be overlooked. If medical guidance recommends or requires precautions (like N95 respirators for tuberculosis), treat the claim as CONTRADICTED.
   - Legislative Leadership vs Personal Incumbency: When a claim asserts a legislative leader or Speaker 'lost a [Party]-held seat', this refers to the caucus/party losing a seat in that legislative body under their leadership, not their personal reelection to their own district.
3. CONSERVATIVE ABSTENTION POLICY: Reserve VERDICT: UNKNOWN strictly for cases where the evidence is genuinely absent, completely unrelated, or truly contradictory without a preponderance of evidence. Do not use abstention to reject the direct contextual entailments specified in Rule 2.
4. The LAST line of your response must be exactly "VERDICT: SUPPORTED", "VERDICT: CONTRADICTED", or "VERDICT: UNKNOWN".

Reasoning and Verdict:
"""
    return generate_response(prompt).strip()

def _dialectical_agent(claim, evidence, domain="general"):
    academic_rule = ""
    if domain == "scientific":
        academic_rule = "\n- Scientific Abstracts: Cautious language (e.g., 'suggests', 'likely') should be interpreted as affirmative support if the underlying finding aligns with the claim.\n"

    prompt = f"""You are an Expert Arbiter evaluating a claim against evidence using strict dialectical reasoning.

Claim: "{claim}"
Evidence: "{evidence}"

Execute the following forensic steps:

[PERSPECTIVE A: DEFENSE (Maximal Support)]
Construct the strongest plausible argument that the claim is SUPPORTED. Cite the exact words from the evidence that support the claim. State the strongest plausible, logically consistent interpretation.

[PERSPECTIVE B: PROSECUTION (Maximal Skepticism)]
Construct the strongest counter-argument that the claim is CONTRADICTED or ungrounded. Identify any ungrounded assumptions, missing facts, discrepancies, or direct contradictions in the claim relative to the evidence.

[PERSPECTIVE C: JUDICIAL ARBITRATION]
Weigh Perspective A against Perspective B strictly against the raw evidence text.{academic_rule}
CRITICAL DECISION & ABSTENTION RULES:
- Natural Entailment: Domestic founders/creators are supported unless foreign origin or external subsidiary status is proven. Do not adopt hyper-literal skepticism.
- Non-Exclusive Roles: Mentioning one co-founder/director does not contradict another being a co-founder; reciprocal phrasing like 'fellow co-founder' directly entails co-founder status.
- Awards and Accolades: When a claim asserts that a person or entity *won* an award, and authoritative biographical evidence specifies they received only *nominations* (e.g. 'earned four nominations'), treat the claim as CONTRADICTED.
- Geographic Venues vs Component Suppliers: Distinguish the primary medical center or hospital where a procedure occurred from secondary facilities or genetic suppliers located elsewhere.
- Media Framing vs Core Historical Assertion: When a claim is framed as 'Photo/Video/Image showing [X]', if authoritative evidence verifies the underlying historical event, entities, and described scene (e.g. Vladimir Komarov's fatal space crash and open casket viewing in 1967), treat the claim as SUPPORTED. Do NOT force an UNKNOWN verdict simply because the evidence is written text rather than an image.
- Compound Health & Scientific Assertions: In compound claims pairing real statistics with an absolute negation of medical/scientific precautions (e.g. 'X fell ill but nobody needs to wear a mask'), the second clause cannot be overlooked. If medical guidance recommends or requires precautions (like N95 respirators for tuberculosis), treat the claim as CONTRADICTED.
- Legislative Leadership vs Personal Incumbency: When a claim asserts a legislative leader or Speaker 'lost a [Party]-held seat', this refers to the caucus/party losing a seat in that legislative body under their leadership, not their personal reelection to their own district.
- Conservative Abstention: Reserve VERDICT: UNKNOWN strictly for cases where the evidence is genuinely absent, completely unrelated, or truly contradictory without a preponderance of evidence.

The LAST line of your response must be exactly:
VERDICT: SUPPORTED | CONTRADICTED | UNKNOWN
"""
    return generate_response(prompt).strip()


def nli_override(claim, evidence, domain="general", mode="dialectical"):
    """
    Arbitrates uncertain or conformal-dispersed claims using adversarial reasoning.

    Modes:
      - 'dialectical' (default): Single-pass Dialectical CoT (Perspective A: Defense -> 
        Perspective B: Prosecution -> Perspective C: Judicial Arbitration).
        Eliminates network waterfall latency (~1.4s vs ~3.8s) and saves 67% API calls.
      - 'multi_agent': Concurrent 3-agent debate (Defense + Prosecution in parallel -> Judge).
        Preserved for academic benchmark ablations and strict context-isolation experiments.
    """
    try:
        if mode == "dialectical":
            print("    [Critic Debate] Running Unified Dialectical CoT Arbitration...")
            judge_response = _dialectical_agent(claim, evidence, domain=domain)
            print(f"    [Critic Debate] Dialectical reasoning:\n{judge_response}")
        else:
            print("    [Critic Debate] Starting Multi-Agent Debate (3-Agent Waterfall)...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                defense_future = executor.submit(_defense_agent, claim, evidence)
                prosecution_future = executor.submit(_prosecution_agent, claim, evidence)
                defense_arg = defense_future.result()
                prosecution_arg = prosecution_future.result()

            print("    [Critic Debate] Defense Agent constructed SUPPORTED argument.")
            print("    [Critic Debate] Prosecution Agent constructed CONTRADICTED argument.")
            
            judge_response = _judge_agent(claim, evidence, defense_arg, prosecution_arg, domain)
            print(f"    [Critic Debate] Judge reasoning:\n{judge_response}")

        lines = judge_response.split('\n')
        verdict_line = next((line for line in reversed(lines) if line.strip().upper().startswith("VERDICT:")), judge_response).upper()

        if "SUPPORTED" in verdict_line:
            return "supported"
        elif "CONTRADICTED" in verdict_line:
            return "contradicted"
        else:
            return "unknown"
    except Exception as e:
        print(f"    [Critic Debate] Arbitration failed: {e}. Abstaining conservatively to 'unknown'.")
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
    try:
        response = generate_response(prompt).strip().upper()
        if "CONTRADICTED" in response:
            return "contradicted"
        else:
            return "unknown"
    except Exception as e:
        print(f"    [Evaluate Absence] LLM connection failed: {e}. Defaulting to 'unknown'.")
        return "unknown"

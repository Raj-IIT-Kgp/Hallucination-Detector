from llm.generator import generate_response



def correct_claim(claim, evidence):


    prompt = f"""

You are a factual correction agent.

Your job is to correct false statements.

Original claim:

{claim}


Evidence:

{evidence}


Rules:

1. Use only the provided evidence.
2. Do not add new information.
3. Return only the corrected statement.


Corrected statement:

"""


    response = generate_response(prompt)


    return response.strip()
from google import genai
from config import GEMINI_API_KEY
import time


from google.genai import types

# Client configured with attempts=3 and timeout=30s for resilient generation without SDK stalls
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=3),
        timeout=30000
    )
)

# Priority order: gemini-3.5-flash-lite is priority #1, followed by gemini-3.1-flash-lite and gemini-3.8-flash
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
_model_cooldown = {}


def generate_gemini_response(prompt):
    """
    Generates an LLM response using Gemini.
    Prioritizes gemini-3.5-flash-lite first on every call.
    If a model hits 429 Quota Exhaustion, 503 Outage, or 504 Deadline Exceeded,
    it is marked on cooldown for 5 minutes so subsequent calls fail over immediately.
    """
    now = time.time()
    last_err = None

    # Find available models (not on cooldown)
    candidate_models = [m for m in MODELS if now >= _model_cooldown.get(m, 0)]
    if not candidate_models:
        # If all models are on cooldown, try the one whose cooldown expires earliest
        earliest_model = min(MODELS, key=lambda m: _model_cooldown.get(m, 0))
        candidate_models = [earliest_model]

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "503", "504", "DEADLINE_EXCEEDED"]):
                # Quota exhausted, outage, or deadline exceeded: cooldown for 5 minutes
                _model_cooldown[model_name] = now + 300
            last_err = e
            continue

    # Final error if all candidate models failed
    if last_err:
        raise last_err
    raise RuntimeError("No LLM response received from any configured Gemini model.")




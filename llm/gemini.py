import threading
import time
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

# Client configured with attempts=1 for instant zero-wait failover across model pools
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1),
        timeout=25000
    )
)

# Active primary pool: balanced 50/50 round-robin to combine 15 RPM + 15 RPM = 30 RPM
PRIMARY_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

# Safety fallback pool: used only if both primary endpoints encounter burst rate limits
BACKUP_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash"]

_rr_counter = 0
_rr_lock = threading.Lock()
_model_cooldown = {}


def generate_gemini_response(prompt):
    """
    Generates an LLM response using an active dual-pool load balancer.
    Interleaves requests 50/50 between gemini-3.5-flash-lite and gemini-3.1-flash-lite
    to pool their independent 15 RPM quotas into a combined 30 RPM throughput.
    
    If any model encounters HTTP 429 or transient error, the client fails over
    INSTANTLY (0ms delay) to the next model in the pool, falling back to backup models
    if both primary endpoints are saturated.
    """
    global _rr_counter
    now = time.time()

    # Determine starting index via thread-safe round-robin
    with _rr_lock:
        start_idx = _rr_counter
        _rr_counter = (_rr_counter + 1) % len(PRIMARY_MODELS)

    # Order primary models starting from current round-robin turn
    ordered_primaries = [PRIMARY_MODELS[(start_idx + i) % len(PRIMARY_MODELS)] for i in range(len(PRIMARY_MODELS))]

    # Build candidate sequence: available primaries -> available backups -> earliest cooldown
    candidate_models = [m for m in ordered_primaries if now >= _model_cooldown.get(m, 0)]
    
    if not candidate_models:
        # Check backups if primaries are cooling down
        candidate_models = [m for m in BACKUP_MODELS if now >= _model_cooldown.get(m, 0)]

    if not candidate_models:
        # If everything is on cooldown, pick whichever model expires earliest
        all_models = PRIMARY_MODELS + BACKUP_MODELS
        earliest_model = min(all_models, key=lambda m: _model_cooldown.get(m, 0))
        candidate_models = [earliest_model]

    last_err = None
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
                # Rate limit / server pressure: cooldown for 45s (rolling 1-minute window reset)
                _model_cooldown[model_name] = now + 45
            last_err = e
            continue

    if last_err:
        raise last_err
    raise RuntimeError("No LLM response received from any configured Gemini model.")

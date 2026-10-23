import os
import threading
import time
import datetime
from zoneinfo import ZoneInfo
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

# The two models in use; each is the other's fallback. (Larger Gemini models have too small a
# free-tier daily quota to help.)
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

# LLM_ROUTING=primary (default): every call goes to gemini-3.5-flash-lite, which measured both faster
# and stronger, and falls back to gemini-3.1-flash-lite instantly when it is busy, rate-limited or
# out of daily quota. LLM_ROUTING=tiered: heavy tasks on 3.5, light tasks on 3.1 (the setup of the
# five-set results in pipeline.md §3.3). LLM_ROUTING=alternate: 50/50 between the two.
ROUTING = os.environ.get("LLM_ROUTING", "primary").lower()
PREFERRED = {
    "primary": {"heavy": "gemini-3.5-flash-lite", "light": "gemini-3.5-flash-lite"},
    "tiered": {"heavy": "gemini-3.5-flash-lite", "light": "gemini-3.1-flash-lite"},
}

_rr_counter = 0
_rr_lock = threading.Lock()
# Which model answered the calling thread's last successful request (read by the call log in llm/generator.py)
last_model = threading.local()
_model_cooldown = {}

# Errors that mean "try again later" (rate limit, quota, overload, timeout) rather than a bad request
_TRANSIENT = ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED",
              "timed out", "Timeout", "ConnectError", "Connection reset"]
# When every model fails with a transient error at once (e.g. a demand spike), wait and try them
# all again: one retry after 2 s, one after 5 s. Adds time only when all models are failing.
_RETRY_DELAYS_SEC = [2, 5]

# Models whose DAILY quota is used up (429 naming a per-day limit) are skipped until the quota
# resets, instead of being retried every 45 s all day; free-tier daily limits reset at midnight
# Pacific time. Maps model -> time at which it may be tried again.
_daily_exhausted = {}


def _all_daily_exhausted():
    now = time.time()
    return all(now < _daily_exhausted.get(m, 0) for m in MODELS)


def _next_quota_reset():
    pacific = ZoneInfo("America/Los_Angeles")
    now = datetime.datetime.now(pacific)
    midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
    return midnight.timestamp()


def generate_gemini_response(prompt, tier="light"):
    """Returns the model's text, retrying all models after a short wait if every one fails transiently."""
    last_err = None
    for attempt in range(len(_RETRY_DELAYS_SEC) + 1):
        if attempt:
            time.sleep(_RETRY_DELAYS_SEC[attempt - 1])
        try:
            return _generate_once(prompt, tier, ignore_cooldown=attempt > 0)
        except Exception as e:
            last_err = e
            # Retry only errors that a few seconds can fix: not a bad request, and not a used-up daily quota
            if not any(k in str(e) for k in _TRANSIENT) or _all_daily_exhausted():
                raise
            print(f"    [Gemini] All models unavailable ({str(e)[:80]}); "
                  + (f"retrying in {_RETRY_DELAYS_SEC[attempt]} s" if attempt < len(_RETRY_DELAYS_SEC) else "giving up"))
    raise last_err


def _generate_once(prompt, tier="light", ignore_cooldown=False):
    """
    One attempt over the models: the preferred model first (by LLM_ROUTING and the task's tier, or the
    next model in a 50/50 rotation with LLM_ROUTING=alternate), then the other. If a model fails with a rate limit
    or transient error, the request fails over instantly (no wait) to the other model.
    """
    global _rr_counter
    now = time.time()

    if ROUTING == "alternate":
        with _rr_lock:
            start_idx = _rr_counter
            _rr_counter = (_rr_counter + 1) % len(MODELS)
        ordered = [MODELS[(start_idx + i) % len(MODELS)] for i in range(len(MODELS))]
    else:
        preferred = PREFERRED.get(ROUTING, PREFERRED["primary"]).get(tier, MODELS[0])
        ordered = [preferred] + [m for m in MODELS if m != preferred]

    # On a retry after every model failed, ignore the short cooldowns: the point is to try them again
    def available(m):
        if now < _daily_exhausted.get(m, 0):
            return False   # daily quota used up: no point trying before it resets
        return ignore_cooldown or now >= _model_cooldown.get(m, 0)
    candidate_models = [m for m in ordered if available(m)]

    if not candidate_models:
        # If everything is on cooldown, pick whichever model expires earliest, preferring models
        # that still have daily quota
        pool = [m for m in MODELS if now >= _daily_exhausted.get(m, 0)]
        if not pool:
            # every model's daily quota is used up: fail at once instead of calling the API
            raise RuntimeError("429 RESOURCE_EXHAUSTED: the daily (PerDay) quota of every Gemini model is used up; "
                               "it resets at midnight Pacific time")
        earliest_model = min(pool, key=lambda m: _model_cooldown.get(m, 0))
        candidate_models = [earliest_model]

    last_err = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                last_model.name = model_name
                return response.text
        except Exception as e:
            err_str = str(e)
            if "PerDay" in err_str:
                _daily_exhausted[model_name] = _next_quota_reset()
                print(f"    [Gemini] {model_name}: daily quota used up; skipping it until the quota resets")
            elif any(k in err_str for k in _TRANSIENT):
                # Rate limit / server pressure: cooldown for 45s (rolling 1-minute window reset)
                _model_cooldown[model_name] = now + 45
            last_err = e
            continue

    if last_err:
        raise last_err
    raise RuntimeError("No LLM response received from any configured Gemini model.")

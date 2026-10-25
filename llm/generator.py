import os
import time
import threading
from llm.gemini import generate_gemini_response, last_model as gemini_last_model
from llm.ollama import generate_ollama_response

# Count of failed LLM calls (quota, timeout, server errors). Callers such as the critic agent
# turn a failure into an UNKNOWN verdict, so benchmarks read this counter to tell an
# infrastructure failure apart from a genuine abstention.
_failure_lock = threading.Lock()
_failure_count = 0
# Count of LLM calls made by the pipeline (successful or not; a call may internally retry on
# another Gemini model, which still counts once). Benchmarks read it to report LLM calls per claim.
_call_count = 0


def get_failure_count():
    with _failure_lock:
        return _failure_count


def get_call_count():
    with _failure_lock:
        return _call_count


# One entry per LLM call: tier, model that answered, duration, success. Benchmarks read the
# entries made while a claim was verified to report LLM cost and per-model latency.
_call_log = []


def get_call_log():
    with _failure_lock:
        return list(_call_log)


def generate_response(prompt, tier="light"):
    """
    tier names how demanding the task is: "heavy" for steps that decide or preserve a verdict
    (arbitration, absence check, claim decomposition), "light" for supporting steps (query generation and
    query rewriting). With Gemini it selects the preferred model (llm/gemini.py).
    """
    global _failure_count, _call_count
    provider = os.environ.get("MODEL_PROVIDER", "ollama").lower()
    with _failure_lock:
        _call_count += 1
    started = time.perf_counter()
    entry = {"tier": tier, "model": None, "sec": None, "ok": False}

    try:
        if provider == "ollama":
            text = generate_ollama_response(prompt)
            entry["model"] = "ollama"
        else:
            gemini_last_model.name = None
            text = generate_gemini_response(prompt, tier=tier)
            entry["model"] = getattr(gemini_last_model, "name", None)
        entry["ok"] = True
        return text
    except Exception:
        with _failure_lock:
            _failure_count += 1
        raise
    finally:
        entry["sec"] = round(time.perf_counter() - started, 3)
        with _failure_lock:
            _call_log.append(entry)

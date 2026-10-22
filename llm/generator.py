import os
import threading
from llm.gemini import generate_gemini_response
from llm.ollama import generate_ollama_response

# Count of failed LLM calls (quota, timeout, server errors). Callers such as the critic agent
# turn a failure into an UNKNOWN verdict, so benchmarks read this counter to tell an
# infrastructure failure apart from a genuine abstention.
_failure_lock = threading.Lock()
_failure_count = 0


def get_failure_count():
    with _failure_lock:
        return _failure_count


def generate_response(prompt):
    global _failure_count
    provider = os.environ.get("MODEL_PROVIDER", "ollama").lower()

    try:
        if provider == "ollama":
            return generate_ollama_response(prompt)
        else:
            return generate_gemini_response(prompt)
    except Exception:
        with _failure_lock:
            _failure_count += 1
        raise

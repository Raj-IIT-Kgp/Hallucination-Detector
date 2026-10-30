"""
VeriFact web server.

    python server.py                 production mode: waitress on 127.0.0.1:8080
    VERIFACT_DEBUG=1 python server.py   development mode: Flask debug server with auto-reload

Settings (environment variables): VERIFACT_HOST, VERIFACT_PORT, VERIFACT_DEBUG,
VERIFACT_MAX_CHARS, VERIFACT_RATE_LIMIT (live runs per client per window),
VERIFACT_RATE_WINDOW_SEC, VERIFACT_MAX_CONCURRENT (live runs at once), VERIFACT_CACHE_TTL_SEC.
"""
from flask import Flask, request, jsonify, send_from_directory, Response, g
from detector.hallucination_detector import HallucinationDetector
from retrieval.web_search import WebRetriever
from llm.generator import get_failure_count
from run_cache import RunCache
from collections import defaultdict, deque
import os
import json
import time
import uuid
import logging
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOST = os.environ.get("VERIFACT_HOST", "127.0.0.1")
PORT = int(os.environ.get("VERIFACT_PORT", "8080"))
DEBUG = os.environ.get("VERIFACT_DEBUG", "0") == "1"
MAX_CHARS = int(os.environ.get("VERIFACT_MAX_CHARS", "4000"))
RATE_LIMIT = int(os.environ.get("VERIFACT_RATE_LIMIT", "12"))
RATE_WINDOW_SEC = int(os.environ.get("VERIFACT_RATE_WINDOW_SEC", "600"))
MAX_CONCURRENT = int(os.environ.get("VERIFACT_MAX_CONCURRENT", "3"))
CACHE_TTL_SEC = int(os.environ.get("VERIFACT_CACHE_TTL_SEC", str(24 * 3600)))
# Recorded runs are replayed with gaps shortened to at most this long, so a replay still shows
# the pipeline's progress but takes seconds instead of the original run time
REPLAY_MAX_GAP_SEC = 0.6
VERSION = "1.0"

# Example inputs offered in the UI; their recorded runs are kept without expiry (demo_runs/)
SAMPLES = [
    {"label": "Factual answer",
     "text": "The Eiffel Tower is located in Paris, France. It was designed and built by Gustave Eiffel's company for the 1889 World's Fair."},
    {"label": "Hallucinated answer",
     "text": "Albert Einstein was born in 1950 in San Francisco, and he won the Nobel Prize in Literature in 1980."},
    {"label": "Mixed answer",
     "text": "The James Webb Space Telescope was launched in December 2021. It orbits the Sun near the second Lagrange point. It was built by the European Space Agency alone."},
    {"label": "False premise",
     "text": "Hikaru Nakamura became the World Chess Champion."},
    {"label": "Unverifiable claim",
     "text": "Marie Curie's favourite colour was green."},
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verifact")

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024   # request bodies are small JSON objects
store = WebRetriever()
detector = HallucinationDetector(store)
runs = RunCache(os.path.join(BASE_DIR, "run_cache"), os.path.join(BASE_DIR, "demo_runs"),
                [s["text"] for s in SAMPLES], CACHE_TTL_SEC)
STARTED_AT = time.time()
_state = {"warm": False}

# Live runs cost API quota and CPU: limit them per client and overall (replays are free)
_live_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
_rate_lock = threading.Lock()
_rate_history = defaultdict(deque)


def _client_ip():
    return request.remote_addr or "unknown"


def _rate_limited(ip):
    """Records a live run for ip; returns seconds to wait if the limit is exceeded, else 0."""
    now = time.time()
    with _rate_lock:
        history = _rate_history[ip]
        while history and now - history[0] > RATE_WINDOW_SEC:
            history.popleft()
        if len(history) >= RATE_LIMIT:
            return int(RATE_WINDOW_SEC - (now - history[0])) + 1
        history.append(now)
        return 0


def _error(message, status, **headers):
    response = jsonify({"error": message})
    response.status_code = status
    response.headers.update(headers)
    return response


def _read_input():
    """Returns (text, force_kg, live) from the JSON body, or raises ValueError with a message."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object with a 'text' field.")
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No text provided.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Text is too long ({len(text)} characters; the limit is {MAX_CHARS}).")
    return text, bool(data.get("forceKG", False)), bool(data.get("live", False))


@app.before_request
def _start_timer():
    g.request_id = uuid.uuid4().hex[:8]
    g.started = time.perf_counter()


@app.after_request
def _finish(response):
    response.headers["X-Request-ID"] = g.get("request_id", "")
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.path.startswith("/api/") and not response.is_streamed:
        log.info("%s %s %s %d %.0fms", g.get("request_id"), request.method, request.path,
                 response.status_code, (time.perf_counter() - g.get("started", time.perf_counter())) * 1000)
    return response


@app.errorhandler(404)
def _not_found(e):
    return _error("Not found.", 404) if request.path.startswith("/api/") else e


@app.errorhandler(405)
def _not_allowed(e):
    return _error("Method not allowed.", 405) if request.path.startswith("/api/") else e


@app.errorhandler(413)
def _too_large(e):
    return _error("Request body too large.", 413)


@app.errorhandler(500)
def _server_error(e):
    log.exception("%s unhandled error", g.get("request_id"))
    return _error("Internal server error.", 500)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    from verification.nli_verifier import model_name, device
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "uptime_sec": round(time.time() - STARTED_AT),
        "models_warm": _state["warm"],
        "llm_provider": os.environ.get("MODEL_PROVIDER", "ollama"),
        "nli_model": model_name,
        "nli_device": str(device),
        **runs.counts(),
    })


@app.route("/api/samples")
def samples():
    return jsonify([{**s, "recorded": runs.get(s["text"], False) is not None} for s in SAMPLES])


@app.route("/api/benchmarks")
def benchmarks():
    """Fifteen-set benchmark summaries written by evaluation/run_5_runs.py, for the Benchmarks tab."""
    results = {}
    for name in ("fever", "freshqa"):
        path = os.path.join(BASE_DIR, "evaluation", f"{name}_5_runs_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
            results[name]["route_outcomes"] = _route_outcomes(name)
            results[name]["latency"] = _latency(name)
    # Holm and Benjamini-Hochberg adjusted p-values over every test in pipeline.md (experiments/statistical_tests.py)
    path = os.path.join(BASE_DIR, "experiments", "statistical_tests.json")
    if os.path.exists(path):
        with open(path) as f:
            results["multiple_testing"] = json.load(f).get("multiple_testing")
    results["analyses"] = _analyses()
    return jsonify(results)


def _analyses():
    """Derived analyses of the published run, for the Benchmarks and Method tabs, when their scripts have run:
    ablations and intervals for precision, recall and F1 (experiments/ablation_selective.py), the LLM-only
    baseline (experiments/llm_every_claim.py) and the conformal procedures (experiments/conformal_variants.py)."""
    def load(name):
        path = os.path.join(BASE_DIR, "experiments", name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
    out = {}
    ablation = load("ablation_selective.json")
    if ablation:
        out["ablation"] = ablation["ablation"]
        out["class_intervals"] = ablation["class_intervals"]
    baseline = load("llm_every_claim.json")
    if baseline:
        out["llm_baseline"] = baseline["by_dataset"]
    variants = load("conformal_variants.json")
    if variants:
        out["conformal_variants"] = {k: variants[k] for k in ("alpha", "gold", "retrieved_gold_calibrated", "retrieved_crossfit")}
    return out


def _latency(name):
    """Measured latency per claim of the published run."""
    path = os.path.join(BASE_DIR, "evaluation", f"{name}_5_runs_claims.jsonl")
    if not os.path.exists(path):
        return {"full_run": None}
    with open(path) as f:
        values = sorted(json.loads(line)["latency_sec"] for line in f if line.strip())
    n = len(values)
    return {"full_run": {"claims": n, "mean": sum(values) / n, "p50": values[n // 2],
                         "p90": values[min(n - 1, int(0.9 * n))]} if n else None}


def _route_outcomes(name):
    """Claims and correct verdicts per decision route, from the per-claim results of the fifteen-set run."""
    path = os.path.join(BASE_DIR, "evaluation", f"{name}_5_runs_claims.jsonl")
    outcomes = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                route = outcomes.setdefault(record.get("decision_policy") or "unknown", {"claims": 0, "correct": 0})
                route["claims"] += 1
                route["correct"] += record.get("gold") == record.get("prediction")
    return outcomes


@app.route("/api/calibration")
def calibration():
    """The conformal calibration scores behind q_hat, for the calibration plot of the Under-the-hood view, and the
    temperature of the NLI probabilities (experiments/calibration.py) for the calibrated confidence."""
    cv = detector.conformal_verifier
    return jsonify({"alpha": cv.alpha, "q_hat": cv.q_hat, "n": len(cv.calibrated_scores),
                    "scores": [float(s) for s in cv.calibrated_scores],
                    "temperature": detector.temperature_scaler.temperature})


_coverage_pools = None


@app.route("/api/coverage_demo")
def coverage_demo():
    """Non-conformity scores at the true label, with the true labels, for the coverage animation of the Method tab:
    the 200 FEVER claims scored on gold evidence (experiments/eval_conformal.py; the first 100 are the calibration
    set) and the 600 benchmark claims scored on the evidence the pipeline retrieved."""
    global _coverage_pools
    if _coverage_pools is None:
        cv = detector.conformal_verifier
        labels = list(cv.labels)
        gold, retrieved, gold_labels, retrieved_labels = [], [], [], []
        path = os.path.join(BASE_DIR, "experiments", "fever_200_cache.json")
        if os.path.exists(path):
            with open(path) as f:
                cache = json.load(f)
            gold = [1 - p[labels.index(y)] for p, y in zip(cache["prob_vectors"], cache["y_true_labels"])]
            gold_labels = list(cache["y_true_labels"])
        for name in ("fever", "freshqa"):
            path = os.path.join(BASE_DIR, "evaluation", f"{name}_5_runs_claims.jsonl")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    probs = ((record.get("result") or {}).get("verification") or {}).get("probabilities")
                    if probs and record.get("gold") in labels:
                        retrieved.append(1 - probs[labels.index(record["gold"])])
                        retrieved_labels.append(record["gold"])
        # the true labels let the Method tab split coverage into false and true claims
        _coverage_pools = {"alpha": cv.alpha, "q_hat": cv.q_hat, "n": len(cv.calibrated_scores),
                           "calibration": [float(s) for s in cv.calibrated_scores],
                           "gold": gold, "retrieved": retrieved,
                           "gold_labels": gold_labels, "retrieved_labels": retrieved_labels}
    return jsonify(_coverage_pools)


@app.route("/api/benchmark_claims/<name>")
def benchmark_claims(name):
    """Per-claim records of the fifteen-set run (verdict, evidence, computation trace), for the claim explorer."""
    if name not in ("fever", "freshqa"):
        return _error("Unknown benchmark.", 404)
    path = os.path.join(BASE_DIR, "evaluation", f"{name}_5_runs_claims.jsonl")
    if not os.path.exists(path):
        return jsonify([])
    with open(path) as f:
        return jsonify([json.loads(line) for line in f if line.strip()])


@app.route("/api/docs")
def docs():
    """pipeline.md, rendered client-side in the Docs tab."""
    return send_from_directory(BASE_DIR, "pipeline.md", mimetype="text/markdown")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        text, _, _ = _read_input()
    except ValueError as e:
        return _error(str(e), 400)
    wait = _rate_limited(_client_ip())
    if wait:
        return _error(f"Rate limit reached ({RATE_LIMIT} checks per {RATE_WINDOW_SEC // 60} minutes). Try again in {wait} s.",
                      429, **{"Retry-After": str(wait)})
    if not _live_slots.acquire(blocking=False):
        return _error("The server is busy with other checks. Try again in a moment.", 503, **{"Retry-After": "10"})
    try:
        return jsonify(detector.analyze(text))
    finally:
        _live_slots.release()


def _replay(record):
    """Streams a recorded run, shortening long gaps; events keep their original time offsets ('t')."""
    yield json.dumps({"status": "replay", "recorded_at": record["recorded_at"],
                      "duration_sec": record["duration_sec"]}) + "\n"
    previous = 0.0
    for line in record["lines"]:
        t = json.loads(line).get("t", previous)
        time.sleep(min(max(t - previous, 0.0), REPLAY_MAX_GAP_SEC))
        previous = t
        yield line


@app.route("/api/analyze_stream", methods=["POST"])
def analyze_stream_endpoint():
    try:
        text, force_kg, live = _read_input()
    except ValueError as e:
        return _error(str(e), 400)
    request_id, ip = g.request_id, _client_ip()

    if not live:
        record = runs.get(text, force_kg)
        if record:
            log.info("%s replay of a run recorded %s", request_id, time.strftime("%Y-%m-%d %H:%M", time.localtime(record["recorded_at"])))
            return Response(_replay(record), mimetype="application/x-ndjson")

    wait = _rate_limited(ip)
    if wait:
        return _error(f"Rate limit reached ({RATE_LIMIT} live checks per {RATE_WINDOW_SEC // 60} minutes). Try again in {wait} s.",
                      429, **{"Retry-After": str(wait)})
    if not _live_slots.acquire(blocking=False):
        return _error("The server is busy with other checks. Try again in a moment.", 503, **{"Retry-After": "10"})

    def generate():
        started = time.perf_counter()
        failures_before = get_failure_count()
        lines, completed, failed, done_at = [], False, False, None
        try:
            for update in detector.analyze_stream(text, force_kg=force_kg):
                event = json.loads(update)
                event["t"] = round(time.perf_counter() - started, 3)   # time since the request started
                line = json.dumps(event) + "\n"
                lines.append(line)
                if event.get("status") == "done":     # late Wikidata graphs may follow the report
                    completed, done_at = True, time.perf_counter()
                yield line
        except Exception as e:
            failed = True
            log.exception("%s analysis failed", request_id)
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
        finally:
            _live_slots.release()
            duration = (done_at or time.perf_counter()) - started
            # Only clean, complete runs are recorded: an LLM failure can turn a verdict into UNKNOWN
            llm_failed = get_failure_count() != failures_before
            if completed and not failed and not llm_failed:
                runs.put(text, force_kg, lines, duration)
            log.info("%s live run: %d events, %.1fs, %s", request_id, len(lines), duration,
                     "recorded" if completed and not failed and not llm_failed
                     else "not recorded (LLM failure)" if llm_failed else "incomplete")

    return Response(generate(), mimetype="application/x-ndjson")


def _warm_up():
    detector.warm_up()
    _state["warm"] = True
    log.info("Models warmed up.")


if __name__ == "__main__":
    if DEBUG:
        # Flask's reloader runs a parent plus the serving child (WERKZEUG_RUN_MAIN=true); only the child serves
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            _warm_up()
        log.warning("Development mode: Flask debug server on http://%s:%d (do not expose it on a network)", HOST, PORT)
        app.run(debug=True, host=HOST, port=PORT, threaded=True)
    else:
        from waitress import serve
        _warm_up()
        log.info("VeriFact %s serving on http://%s:%d", VERSION, HOST, PORT)
        serve(app, host=HOST, port=PORT, threads=8, channel_timeout=300)

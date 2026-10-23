"""
Recorded verification runs for the web server.

A completed live run is stored as its event stream with each event's time offset, so an identical
input can be replayed (same events, same timing shape) without calling the pipeline again. Runs of
the example inputs are kept in demo_runs/ without expiry, so a demo works offline and without API
quota; other inputs are kept in run_cache/ for a limited time, because facts change.
"""
import hashlib
import json
import os
import threading
import time


def _normalize(text):
    return " ".join(text.split())


class RunCache:
    def __init__(self, cache_dir, demo_dir, demo_texts, ttl_sec):
        self.cache_dir = cache_dir
        self.demo_dir = demo_dir
        self.demo_texts = {_normalize(t) for t in demo_texts}
        self.ttl_sec = ttl_sec
        self._lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(demo_dir, exist_ok=True)

    @staticmethod
    def key(text, force_kg):
        return hashlib.sha256(f"{int(bool(force_kg))}\n{_normalize(text)}".encode()).hexdigest()[:32]

    def _path(self, text, force_kg):
        directory = self.demo_dir if _normalize(text) in self.demo_texts else self.cache_dir
        return os.path.join(directory, f"{self.key(text, force_kg)}.json")

    def get(self, text, force_kg):
        """The recorded run for this input, or None if there is none (or it has expired)."""
        path = self._path(text, force_kg)
        try:
            with open(path) as f:
                record = json.load(f)
        except (OSError, ValueError):
            return None
        if path.startswith(self.cache_dir) and time.time() - record.get("recorded_at", 0) > self.ttl_sec:
            return None
        return record

    def put(self, text, force_kg, lines, duration_sec):
        """Stores a completed run; lines are the NDJSON event lines exactly as streamed."""
        record = {
            "text": text,
            "force_kg": bool(force_kg),
            "recorded_at": time.time(),
            "duration_sec": round(duration_sec, 2),
            "lines": lines,
        }
        path = self._path(text, force_kg)
        tmp = f"{path}.{threading.get_ident()}.tmp"
        with self._lock:
            with open(tmp, "w") as f:
                json.dump(record, f)
            os.replace(tmp, path)   # atomic: a reader never sees a half-written file

    def counts(self):
        def n(directory):
            return sum(1 for f in os.listdir(directory) if f.endswith(".json"))
        return {"cached_runs": n(self.cache_dir), "demo_runs": n(self.demo_dir)}

# VeriFact — Hallucination Detector

VeriFact checks a text, typically an answer written by a language model, for hallucinations. It splits the text into atomic claims, retrieves evidence for each claim from Wikipedia, the web and arXiv, scores the evidence with a natural language inference (NLI) model, and uses split conformal prediction to decide when that verdict can stand and when the claim must be escalated to a knowledge graph and an LLM arbiter.

| Benchmark (100 claims, five sets of 20) | Accuracy (Wilson 95% CI) | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- |
| FEVER | 95.0% [88.8%, 97.8%] | 0.96 | 0.96 | 0.96 |
| FreshQA (never- and slow-changing) | 82.0% [73.3%, 88.3%] | 0.91 | 0.82 | 0.86 |

Mean latency per claim: **8.6 s** on FEVER and **12.2 s** on FreshQA.

**The full method, mathematics, evaluation and limitations are documented in [`pipeline.md`](pipeline.md).**

## Quick start

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your-key
```

On the first run, allow the NLI and embedding models (about 1.6 GB) to download:

```bash
HF_HUB_OFFLINE=0 python server.py
```

Then open <http://localhost:8080>. Later runs can use `python server.py`.

The server runs in production mode by default (waitress, bound to `127.0.0.1`). `VERIFACT_DEBUG=1 python server.py` starts Flask's development server with auto-reload instead; do not expose that one on a network. Other settings: `VERIFACT_HOST`, `VERIFACT_PORT`, `VERIFACT_MAX_CHARS` (input limit, default 4000), `VERIFACT_RATE_LIMIT` and `VERIFACT_RATE_WINDOW_SEC` (live checks per client, default 12 per 10 minutes), `VERIFACT_MAX_CONCURRENT` (live checks at once, default 3), `VERIFACT_CACHE_TTL_SEC` (how long a live run is kept for replay, default 86400 = 24 h), `VERIFACT_WIKIDATA=0` (turn off the Wikidata graph). `GET /api/health` reports the server's status.

**Recorded runs.** A completed live check is recorded; the same input then replays instantly, with no API calls, and is clearly marked as a replay with a *Run live instead* button. The example inputs' recordings are kept in `demo_runs/` (committed, no expiry), so a demo works offline; other inputs are kept for 24 hours in `run_cache/`.

## Web interface

- **Verify**: paste text and watch each claim go through the pipeline live, with its verdict, the NLI reading of every evidence snippet (and which one decided), the conformal set, a stage timeline, and a knowledge graph from Wikidata that sets each claim next to what Wikidata records (escalated claims also give these statements to the LLM arbiter).
- **Benchmarks**: the FEVER and FreshQA five-set results.
- **Method**: the pipeline stages and known limitations.
- **Docs**: `pipeline.md`, rendered.

## Evaluation

```bash
python evaluation/run_5_runs.py --dataset all      # five-set evaluation of FEVER and FreshQA (resumable)
python evaluation/run_benchmark.py --dataset fever --samples 20
python experiments/full_evaluation.py              # the full evaluation: five-set run plus every analysis in pipeline.md §2.5, §2.6, §2.9 and §3
python experiments/eval_conformal.py               # conformal calibration and coverage
python experiments/statistical_tests.py            # confidence intervals and hypothesis tests (no LLM call)
```

The five-set evaluation makes several hundred LLM calls; if the API quota runs out, it stops and resumes where it stopped when started again.

## Project layout

| Path | Contents |
| :--- | :--- |
| `agents/` | claim extraction, query generation, Wikidata knowledge graph, LLM arbiter |
| `retrieval/` | Wikipedia, web and arXiv retrieval; hybrid re-ranking |
| `verification/` | NLI model |
| `statistical_layer/` | split conformal prediction |
| `detector/` | the pipeline that connects the stages, and the report |
| `llm/` | Gemini and Ollama clients |
| `evaluation/` | benchmark loaders, runners and result files |
| `experiments/` | conformal calibration and the full evaluation report |
| `static/`, `server.py`, `run_cache.py` | web interface, server and recorded runs |
| `demo_runs/` | recorded runs of the example inputs |
| `tests/` | test scripts, each run directly, e.g. `python tests/test_hierarchical_scoring.py` |

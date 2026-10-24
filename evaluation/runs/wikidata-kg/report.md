# Full evaluation: wikidata-kg

## FEVER

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 95.0% | 0.96 | 0.96 | 0.96 | 48 | 2 | 2 | 47 | 2 | 8.6 s | 0.98 |
| published | 95.0% | 0.94 | 0.96 | 0.95 | 48 | 3 | 2 | 47 | 1 | 8.6 s | 1.16 |

Routes: conformal_singleton 83/86, contradiction_arbitration 4/5, multi_agent_debate 8/9

Arbiter effect: {"review": {"claims": 5, "fixed": 1, "broken": 1, "both_right": 3, "both_wrong": 0, "llm_unavailable": 0}, "escalation": {"claims": 9, "fixed": 3, "broken": 0, "both_right": 5, "both_wrong": 1, "llm_unavailable": 0}}

Changed vs published (2):
- [broke] Brian Michael Bendis began teaching English at the University of Oregon in fall of 2013. (gold supported: supported -> unknown; multi_agent_debate -> multi_agent_debate)
- [fixed] Shane Black is an American producer and actor. (gold supported: contradicted -> supported; multi_agent_debate -> multi_agent_debate)

## FRESHQA

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 82.0% | 0.91 | 0.82 | 0.86 | 41 | 4 | 9 | 41 | 7 | 12.2 s | 1.45 |
| published | 83.0% | 0.91 | 0.86 | 0.89 | 43 | 4 | 7 | 40 | 8 | 12.8 s | 1.87 |

Routes: conformal_singleton 50/58, contradiction_arbitration 21/25, multi_agent_debate 11/17

Arbiter effect: {"review": {"claims": 25, "fixed": 9, "broken": 1, "both_right": 12, "both_wrong": 3, "llm_unavailable": 0}, "escalation": {"claims": 17, "fixed": 4, "broken": 0, "both_right": 7, "both_wrong": 6, "llm_unavailable": 0}}

Changed vs published (5):
- [fixed] Clint Eastwood has won a Best Actor Academy Award. (gold contradicted: unknown -> contradicted; multi_agent_debate -> multi_agent_debate)
- [broke] Novak Djokovic lost in a specific round at the 2022 Australian Open. (gold contradicted: contradicted -> supported; multi_agent_debate -> conformal_singleton)
- [broke] A FIFA World Cup will be hosted next year. (gold contradicted: contradicted -> supported; multi_agent_debate -> conformal_singleton)
- [broke] Taylor Swift has been a UNICEF Goodwill Ambassador. (gold contradicted: contradicted -> unknown; multi_agent_debate -> multi_agent_debate)
- [fixed] France is the most recent team to advance to consecutive FIFA World Cup finals. (gold supported: unknown -> supported; multi_agent_debate -> multi_agent_debate)

## Corroboration rule (both benchmarks)

| Decisive contradicting snippets | Claims | Actually false | Wrong if accepted |
| :--- | :--- | :--- | :--- |
| 1 | 32 | 20 | 37.5% |
| 2 | 25 | 23 | 8.0% |
| 3 or more | 44 | 42 | 4.5% |
| decisive support also present | 8 | 2 | 75.0% |

## Conformal layer on retrieved evidence

- claims: 200
- current q-hat 0.8985: coverage 83.5% (Wilson 95% 77.7%–88.0%), mean set size 1.13
- cross-fitted q-hat (folds 0.9944 / 0.9941): coverage 90.5% (Wilson 95% 85.6%–93.8%), mean set size 1.80
- q-hat from all retrieved-evidence claims: 0.9941

Estimated accuracy with the recalibrated layer: {"fever": {"claims": 100, "correct": 93, "route_changes": 38, "newly_escalated": 34, "newly_accepted": 0, "pending": 0, "accuracy": 0.93}, "freshqa": {"claims": 100, "correct": 85, "route_changes": 56, "newly_escalated": 37, "newly_accepted": 0, "pending": 0, "accuracy": 0.85}}

## LLM usage

{
  "mean_calls_per_claim": 1.215,
  "mean_calls_by_route": {
    "conformal_singleton": 0.8958333333333334,
    "contradiction_arbitration": 1.8666666666666667,
    "multi_agent_debate": 2.230769230769231
  },
  "latency_by_model_and_tier": {
    "gemini-3.5-flash-lite (light)": {
      "calls": 187,
      "mean_sec": 1.5449358288770054,
      "p50_sec": 1.187
    },
    "gemini-3.5-flash-lite (heavy)": {
      "calls": 56,
      "mean_sec": 3.604767857142857,
      "p50_sec": 3.086
    }
  }
}

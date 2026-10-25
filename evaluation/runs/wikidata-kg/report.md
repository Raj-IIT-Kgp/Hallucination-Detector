# Full evaluation: wikidata-kg

## FEVER

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 91.0% | 0.95 | 0.89 | 0.92 | 89 | 5 | 11 | 93 | 3 | 8.6 s | 1.04 |
| published | 91.0% | 0.95 | 0.89 | 0.92 | 89 | 5 | 11 | 93 | 3 | 8.6 s | 1.04 |

Routes: conformal_singleton 146/157, contradiction_arbitration 19/21, multi_agent_debate 17/22

Arbiter effect: {"review": {"claims": 21, "fixed": 3, "broken": 1, "both_right": 16, "both_wrong": 1, "llm_unavailable": 0}, "escalation": {"claims": 22, "fixed": 5, "broken": 3, "both_right": 12, "both_wrong": 2, "llm_unavailable": 0}}

Changed vs published (0):

## FRESHQA

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 83.5% | 0.90 | 0.84 | 0.87 | 84 | 9 | 16 | 83 | 11 | 11.4 s | 1.43 |
| published | 83.5% | 0.90 | 0.84 | 0.87 | 84 | 9 | 16 | 83 | 11 | 11.4 s | 1.43 |

Routes: conformal_singleton 107/123, contradiction_arbitration 32/38, multi_agent_debate 28/39

Arbiter effect: {"review": {"claims": 38, "fixed": 13, "broken": 2, "both_right": 19, "both_wrong": 4, "llm_unavailable": 0}, "escalation": {"claims": 39, "fixed": 15, "broken": 4, "both_right": 13, "both_wrong": 7, "llm_unavailable": 0}}

Changed vs published (0):

## Corroboration rule (both benchmarks)

| Decisive contradicting snippets | Claims | Actually false | Wrong if accepted |
| :--- | :--- | :--- | :--- |
| 1 | 67 | 47 | 29.9% |
| 2 | 42 | 37 | 11.9% |
| 3 or more | 81 | 77 | 4.9% |
| decisive support also present | 16 | 7 | 56.2% |

## Conformal layer on retrieved evidence

- claims: 400
- current q-hat 0.8985: coverage 84.2% (Wilson 95% 80.4%–87.5%), mean set size 1.17
- cross-fitted q-hat (folds 0.9953 / 0.9696): coverage 88.2% (Wilson 95% 84.7%–91.0%), mean set size 1.64
- q-hat from all retrieved-evidence claims: 0.9906

Estimated accuracy with the recalibrated layer: {"fever": {"claims": 200, "correct": 183, "route_changes": 55, "newly_escalated": 47, "newly_accepted": 0, "pending": 0, "accuracy": 0.915}, "freshqa": {"claims": 200, "correct": 169, "route_changes": 78, "newly_escalated": 57, "newly_accepted": 0, "pending": 0, "accuracy": 0.845}}

## LLM usage

{
  "mean_calls_per_claim": 1.2325,
  "mean_calls_by_route": {
    "conformal_singleton": 0.875,
    "contradiction_arbitration": 1.88135593220339,
    "multi_agent_debate": 2.2459016393442623
  },
  "latency_by_model_and_tier": {
    "gemini-3.5-flash-lite (light)": {
      "calls": 373,
      "mean_sec": 1.3096085790884717,
      "p50_sec": 1.088
    },
    "gemini-3.5-flash-lite (heavy)": {
      "calls": 118,
      "mean_sec": 3.0615677966101695,
      "p50_sec": 2.731
    },
    "gemini-3.1-flash-lite (heavy)": {
      "calls": 2,
      "mean_sec": 19.9975,
      "p50_sec": 29.897
    }
  }
}

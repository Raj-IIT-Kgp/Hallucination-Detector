# Full evaluation: wikidata-kg

## FEVER

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 90.7% | 0.94 | 0.89 | 0.91 | 134 | 9 | 16 | 138 | 4 | 8.5 s | 1.05 |
| published | 90.7% | 0.94 | 0.89 | 0.91 | 134 | 9 | 16 | 138 | 4 | 8.5 s | 1.05 |

Routes: conformal_singleton 212/229, contradiction_arbitration 30/35, multi_agent_debate 30/36

Arbiter effect: {"review": {"claims": 35, "fixed": 4, "broken": 3, "both_right": 26, "both_wrong": 2, "llm_unavailable": 0}, "escalation": {"claims": 36, "fixed": 14, "broken": 4, "both_right": 16, "both_wrong": 2, "llm_unavailable": 0}}

Changed vs published (0):

## FRESHQA

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 82.3% | 0.91 | 0.80 | 0.85 | 117 | 12 | 30 | 130 | 15 | 11.3 s | 1.45 |
| published | 82.3% | 0.91 | 0.80 | 0.85 | 117 | 12 | 30 | 130 | 15 | 11.3 s | 1.45 |

Routes: conformal_singleton 156/184, contradiction_arbitration 47/54, multi_agent_debate 44/62

Arbiter effect: {"review": {"claims": 54, "fixed": 18, "broken": 2, "both_right": 29, "both_wrong": 5, "llm_unavailable": 0}, "escalation": {"claims": 62, "fixed": 24, "broken": 6, "both_right": 20, "both_wrong": 12, "llm_unavailable": 0}}

Changed vs published (0):

## Corroboration rule (both benchmarks)

| Decisive contradicting snippets | Claims | Actually false | Wrong if accepted |
| :--- | :--- | :--- | :--- |
| 1 | 99 | 70 | 29.3% |
| 2 | 62 | 55 | 11.3% |
| 3 or more | 113 | 108 | 4.4% |
| decisive support also present | 29 | 9 | 69.0% |

## Conformal layer on retrieved evidence

- claims: 600
- current q-hat 0.8985: coverage 83.2% (Wilson 95% 80.0%–85.9%), mean set size 1.18
- cross-fitted q-hat (folds 0.9946 / 0.9903): coverage 90.3% (Wilson 95% 87.7%–92.4%), mean set size 1.80
- q-hat from all retrieved-evidence claims: 0.9935

Estimated accuracy with the recalibrated layer: {"fever": {"claims": 300, "correct": 277, "route_changes": 107, "newly_escalated": 86, "newly_accepted": 0, "pending": 0, "accuracy": 0.9233333333333333}, "freshqa": {"claims": 300, "correct": 253, "route_changes": 134, "newly_escalated": 95, "newly_accepted": 0, "pending": 0, "accuracy": 0.8433333333333334}}

## LLM usage

{
  "mean_calls_per_claim": 1.2483333333333333,
  "mean_calls_by_route": {
    "conformal_singleton": 0.8595641646489104,
    "contradiction_arbitration": 1.8764044943820224,
    "multi_agent_debate": 2.316326530612245
  },
  "latency_by_model_and_tier": {
    "gemini-3.5-flash-lite (light)": {
      "calls": 562,
      "mean_sec": 1.2524750889679714,
      "p50_sec": 1.06
    },
    "gemini-3.5-flash-lite (heavy)": {
      "calls": 185,
      "mean_sec": 2.941864864864865,
      "p50_sec": 2.725
    },
    "gemini-3.1-flash-lite (heavy)": {
      "calls": 2,
      "mean_sec": 19.9975,
      "p50_sec": 29.897
    }
  }
}

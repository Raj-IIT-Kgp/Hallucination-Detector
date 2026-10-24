# Full evaluation: current-results

## FEVER

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 95.0% | 0.94 | 0.96 | 0.95 | 48 | 3 | 2 | 47 | 1 | 8.6 s | 1.16 |
| published | 95.0% | 0.94 | 0.96 | 0.95 | 48 | 3 | 2 | 47 | 1 | 8.6 s | 1.16 |

Routes: multi_agent_debate 11/13, conformal_singleton 80/82, contradiction_arbitration 4/5

Arbiter effect: {"escalation": {"claims": 13, "fixed": 6, "broken": 1, "both_right": 5, "both_wrong": 1, "llm_unavailable": 0}, "review": {"claims": 5, "fixed": 1, "broken": 1, "both_right": 3, "both_wrong": 0, "llm_unavailable": 0}}

Changed vs published (0):

## FRESHQA

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 83.0% | 0.91 | 0.86 | 0.89 | 43 | 4 | 7 | 40 | 8 | 12.8 s | 1.87 |
| published | 87.0% | 0.88 | 0.92 | 0.90 | 46 | 6 | 4 | 41 | 3 | 21.4 s | 1.88 |

Routes: conformal_singleton 48/54, multi_agent_debate 16/24, contradiction_arbitration 19/22

Arbiter effect: {"escalation": {"claims": 24, "fixed": 7, "broken": 2, "both_right": 9, "both_wrong": 6, "llm_unavailable": 0}, "review": {"claims": 22, "fixed": 8, "broken": 1, "both_right": 11, "both_wrong": 2, "llm_unavailable": 0}}

Changed vs published (18):
- [broke] 8 episodes of The Lord of the Rings: The Rings of Power season 1 have been released. (gold supported: supported -> unknown; multi_agent_debate -> multi_agent_debate)
- [broke] Donald Trump tweeted in 2022 after his unbanning from Twitter by Elon Musk. (gold contradicted: contradicted -> supported; multi_agent_debate -> conformal_singleton)
- [broke] Walt Disney Animation Studios' 60th animated film is being released this year. (gold contradicted: contradicted -> supported; multi_agent_debate -> conformal_singleton)
- [broke] The Japanese school year starts in April. (gold supported: supported -> unknown; contradiction_arbitration -> multi_agent_debate)
- [broke] Clint Eastwood has won a Best Actor Academy Award. (gold contradicted: contradicted -> unknown; multi_agent_debate -> multi_agent_debate)
- [fixed] There exists a smallest cube number which can be expressed as the sum of two different positive cube numbers in two different ways. (gold contradicted: supported -> contradicted; conformal_singleton -> contradiction_arbitration)
- [broke] Pope Emeritus Benedict XVI and Queen Elizabeth II both died in 2022. (gold supported: supported -> unknown; multi_agent_debate -> contradiction_arbitration)
- [fixed] The euro is Croatia's current national currency. (gold supported: contradicted -> supported; conformal_singleton -> conformal_singleton)
- [fixed] No humans have landed on Mars yet. (gold supported: contradicted -> supported; conformal_singleton -> contradiction_arbitration)
- [fixed] A FIFA World Cup will be hosted next year. (gold contradicted: supported -> contradicted; multi_agent_debate -> multi_agent_debate)
- [fixed] Parking at the Big E costs $10. (gold supported: unknown -> supported; multi_agent_debate -> conformal_singleton)
- [broke] Pele is currently staying in a hospital to receive cancer treatment. (gold contradicted: contradicted -> supported; multi_agent_debate -> conformal_singleton)
- [broke] An animal has landed on the moon. (gold contradicted: contradicted -> supported; conformal_singleton -> conformal_singleton)
- [fixed] The province that borders Shanghai to the north is Jiangsu. (gold supported: contradicted -> supported; multi_agent_debate -> multi_agent_debate)
- [broke] France is the most recent team to advance to consecutive FIFA World Cup finals. (gold supported: supported -> unknown; multi_agent_debate -> multi_agent_debate)
- [broke] Humans have been killed by dinosaurs. (gold contradicted: contradicted -> unknown; conformal_singleton -> contradiction_arbitration)
- [fixed] OpenAI has released GPT-5.5. (gold contradicted: supported -> contradicted; contradiction_arbitration -> contradiction_arbitration)
- [broke] Princeton University is the top-ranked university in the US according to the US News Ranking. (gold supported: supported -> contradicted; conformal_singleton -> multi_agent_debate)

## Corroboration rule (both benchmarks)

| Decisive contradicting snippets | Claims | Actually false | Wrong if accepted |
| :--- | :--- | :--- | :--- |
| 1 | 30 | 19 | 36.7% |
| 2 | 24 | 22 | 8.3% |
| 3 or more | 44 | 42 | 4.5% |
| decisive support also present | 12 | 5 | 58.3% |

## Conformal layer on retrieved evidence

- claims: 200
- current q-hat 0.8985: coverage 85.0% (Wilson 95% 79.4%–89.3%), mean set size 1.21
- cross-fitted q-hat (folds 0.9781 / 0.9966): coverage 90.0% (Wilson 95% 85.1%–93.4%), mean set size 1.73
- q-hat from all retrieved-evidence claims: 0.9882

Estimated accuracy with the recalibrated layer: {"fever": {"claims": 100, "correct": 94, "route_changes": 26, "newly_escalated": 23, "newly_accepted": 0, "pending": 0, "accuracy": 0.94}, "freshqa": {"claims": 100, "correct": 83, "route_changes": 43, "newly_escalated": 27, "newly_accepted": 0, "pending": 0, "accuracy": 0.83}}

## LLM usage

{
  "mean_calls_per_claim": 1.515,
  "mean_calls_by_route": {
    "multi_agent_debate": 3.5675675675675675,
    "conformal_singleton": 0.8897058823529411,
    "contradiction_arbitration": 1.8518518518518519
  },
  "latency_by_model_and_tier": {
    "gemini-3.5-flash-lite (light)": {
      "calls": 239,
      "mean_sec": 1.4259748953974896,
      "p50_sec": 1.161
    },
    "gemini-3.5-flash-lite (heavy)": {
      "calls": 64,
      "mean_sec": 3.946234375,
      "p50_sec": 2.835
    }
  }
}

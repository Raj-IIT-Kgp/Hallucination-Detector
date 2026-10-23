# Full evaluation: current-pipeline

## FEVER

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 95.0% | 0.98 | 0.96 | 0.97 | 48 | 1 | 2 | 47 | 3 | 15.0 s | 1.32 |
| published | 96.0% | 0.96 | 0.96 | 0.96 | 48 | 2 | 2 | 48 | 2 | 10.5 s | — |

Routes: multi_agent_debate 14/14, conformal_singleton 76/78, contradiction_arbitration 5/8

Arbiter effect: {"escalation": {"claims": 14, "fixed": 9, "broken": 0, "both_right": 5, "both_wrong": 0, "llm_unavailable": 0}, "review": {"claims": 8, "fixed": 0, "broken": 1, "both_right": 5, "both_wrong": 2, "llm_unavailable": 0}}

Changed vs published (7):
- [fixed] Brian Michael Bendis began teaching English at the University of Oregon in fall of 2013. (gold supported: contradicted -> supported; multi_agent_debate -> multi_agent_debate)
- [broke] 2016 Tour de France was won by a rider. (gold supported: supported -> unknown; conformal_singleton -> contradiction_arbitration)
- [broke] Underdog features the voice talents of Amy Adams. (gold supported: supported -> contradicted; conformal_singleton -> conformal_singleton)
- [broke] Middle-earth is a real place. (gold contradicted: contradicted -> supported; contradiction_arbitration -> conformal_singleton)
- [broke] Caroline Kennedy is Ambassador to Japan. (gold supported: supported -> unknown; conformal_singleton -> contradiction_arbitration)
- [fixed] Marvel vs. Capcom is part of a trilogy. (gold contradicted: unknown -> contradicted; multi_agent_debate -> multi_agent_debate)
- [fixed] Tamerlan Tsarnaev carjacked an SUV. (gold supported: contradicted -> supported; conformal_singleton -> contradiction_arbitration)

## FRESHQA

| | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | Unknown | Mean latency | LLM calls/claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| now | 87.0% | 0.88 | 0.92 | 0.90 | 46 | 6 | 4 | 41 | 3 | 21.4 s | 1.88 |
| published | 85.0% | 0.93 | 0.82 | 0.87 | 41 | 3 | 9 | 44 | 7 | 16.8 s | — |

Routes: conformal_singleton 49/54, multi_agent_debate 24/31, contradiction_arbitration 14/15

Arbiter effect: {"escalation": {"claims": 31, "fixed": 13, "broken": 1, "both_right": 11, "both_wrong": 6, "llm_unavailable": 0}, "review": {"claims": 15, "fixed": 8, "broken": 1, "both_right": 6, "both_wrong": 0, "llm_unavailable": 0}}

Changed vs published (14):
- [fixed] Donald Trump tweeted in 2022 after his unbanning from Twitter by Elon Musk. (gold contradicted: supported -> contradicted; conformal_singleton -> multi_agent_debate)
- [fixed] Walt Disney Animation Studios' 60th animated film is being released this year. (gold contradicted: supported -> contradicted; conformal_singleton -> multi_agent_debate)
- [fixed] Pope Emeritus Benedict XVI and Queen Elizabeth II both died in 2022. (gold supported: unknown -> supported; multi_agent_debate -> multi_agent_debate)
- [fixed] There is a current Protestant majority in Northern Ireland. (gold contradicted: unknown -> contradicted; multi_agent_debate -> multi_agent_debate)
- [broke] The euro is Croatia's current national currency. (gold supported: supported -> contradicted; conformal_singleton -> conformal_singleton)
- [broke] No humans have landed on Mars yet. (gold supported: supported -> contradicted; contradiction_arbitration -> conformal_singleton)
- [broke] A FIFA World Cup will be hosted next year. (gold contradicted: contradicted -> supported; multi_agent_debate -> multi_agent_debate)
- [fixed] Taylor Swift has been a UNICEF Goodwill Ambassador. (gold contradicted: unknown -> contradicted; multi_agent_debate -> multi_agent_debate)
- [broke] Parking at the Big E costs $10. (gold supported: supported -> unknown; conformal_singleton -> multi_agent_debate)
- [fixed] An animal has landed on the moon. (gold contradicted: unknown -> contradicted; contradiction_arbitration -> conformal_singleton)
- [broke] The province that borders Shanghai to the north is Jiangsu. (gold supported: supported -> contradicted; conformal_singleton -> multi_agent_debate)
- [fixed] Humans have been killed by dinosaurs. (gold contradicted: unknown -> contradicted; contradiction_arbitration -> conformal_singleton)
- [broke] There are 107 LA metro rail stations. (gold supported: supported -> contradicted; conformal_singleton -> multi_agent_debate)
- [fixed] There is no total solar eclipse in the continental US this year. (gold supported: contradicted -> supported; conformal_singleton -> contradiction_arbitration)

## Corroboration rule (both benchmarks)

| Decisive contradicting snippets | Claims | Actually false | Wrong if accepted |
| :--- | :--- | :--- | :--- |
| 1 | 25 | 14 | 44.0% |
| 2 | 28 | 24 | 14.3% |
| 3 or more | 47 | 46 | 2.1% |
| decisive support also present | 12 | 3 | 75.0% |

## Conformal layer on retrieved evidence

- claims: 200
- current q-hat 0.8985: coverage 85.0% (Wilson 95% 79.4%–89.3%), mean set size 1.25
- cross-fitted q-hat (folds 0.9238 / 0.9935): coverage 88.0% (Wilson 95% 82.8%–91.8%), mean set size 1.58
- q-hat from all retrieved-evidence claims: 0.9785

Estimated accuracy with the recalibrated layer: {"fever": {"claims": 100, "correct": 93, "route_changes": 16, "newly_escalated": 15, "newly_accepted": 0, "pending": 0, "accuracy": 0.93}, "freshqa": {"claims": 100, "correct": 86, "route_changes": 25, "newly_escalated": 17, "newly_accepted": 0, "pending": 0, "accuracy": 0.86}}

## LLM usage

{
  "mean_calls_per_claim": 1.6,
  "mean_calls_by_route": {
    "multi_agent_debate": 3.4444444444444446,
    "conformal_singleton": 0.9166666666666666,
    "contradiction_arbitration": 1.9130434782608696
  },
  "latency_by_model_and_tier": {
    "gemini-3.1-flash-lite (light)": {
      "calls": 204,
      "mean_sec": 5.2364950980392155,
      "p50_sec": 3.87
    },
    "gemini-3.5-flash-lite (heavy)": {
      "calls": 67,
      "mean_sec": 2.892805970149254,
      "p50_sec": 2.749
    },
    "gemini-3.5-flash-lite (light)": {
      "calls": 48,
      "mean_sec": 8.423125,
      "p50_sec": 2.014
    },
    "gemini-3.1-flash-lite (heavy)": {
      "calls": 1,
      "mean_sec": 28.848,
      "p50_sec": 28.848
    }
  }
}

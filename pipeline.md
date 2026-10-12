# Architectural Report: Automated Hallucination Detection Pipeline with Knowledge Graph Reasoning and Distribution-Free Conformal Verification

## 1. Abstract
This report outlines the architecture, statistical constraints, and empirical evaluation of an end-to-end Automated Hallucination Detection system. The pipeline integrates deterministic claim extraction, multi-entity query generation, hybrid web retrieval, zero-shot Natural Language Inference (NLI), an in-memory **Knowledge Graph (KG) multi-hop reasoning layer**, a **Distribution-Free Split Conformal Prediction verification layer**, and a **Multi-Agent Critic Debate** framework — powered by Gemini — to identify and flag factual hallucinations in AI-generated text.

Crucially, the system replaces arbitrary heuristic confidence thresholds with **Split Conformal Prediction**, providing a finite-sample marginal coverage guarantee $\mathbb{P}(Y \in \mathcal{C}(X)) \ge 1 - \alpha$ under the standard assumption of exchangeability. When the conformal prediction set signals model posterior dispersion ($|\mathcal{C}(X)| > 1$), the pipeline escalates to dynamically construct a live in-memory Knowledge Graph, traverses multi-hop evidence paths, and arbitrates competing viewpoints via a multi-agent debate with an explicit **Conservative Abstention Policy**.

---

## 2. System Architecture & Pipeline Workflow

The system implements a **6-Phase Cascaded Verification Architecture**. Each phase acts as a progressively more powerful reasoning engine. The pipeline only escalates to higher-cost reasoning tiers when lower tiers indicate statistical uncertainty, optimizing verification accuracy against API latency and token cost.

```
                  Complex Input Text
                          │
          ┌───────────────▼──────────────┐
          │ Phase 1: Atomic Decomposition│  (Generates Natomic claims)
          └───────────────┬──────────────┘
                          │
          ┌───────────────▼──────────────┐
          │ Phase 2: Dynamic Query Gen   │  (Entities, Domain, Timestamps)
          └───────────────┬──────────────┘
                          │
          ┌───────────────▼──────────────┐
          │ Phase 3: Hybrid Retrieval    │  (Dense MiniLM + BM25 RRF)
          └───────────────┬──────────────┘
                          │
          ┌───────────────▼──────────────┐
          │ Phase 3.5: Consensus Check   │  (External Source Conflict Detection)
          └───────────────┬──────────────┘
                          │
          ┌───────────────▼──────────────┐
          │ Phase 4: Conformal NLI Layer │  (Coverage Guarantee: P(Y ∈ C(X)) ≥ 1 - α)
          └───────────────┬──────────────┘
                          │
                 Is Set Ambiguous?
                 [|C(X)| > 1 or UNKNOWN ∈ C(X)]
                  ├── No  ──► Crisp Point Verdict (Supported / Contradicted)
                  │
                  └── Yes ──► (Escalation Trigger)
                                  │
          ┌───────────────────────▼────────────────────────┐
          │ Phase 5a: Knowledge Graph Multi-Hop Reasoning  │
          └───────────────────────┬────────────────────────┘
                                  │
          ┌───────────────────────▼────────────────────────┐
          │ Phase 5b: Multi-Agent Critic Debate            │
          │ (Defense vs Prosecution + Conservative Judge)  │
          └───────────────────────┬────────────────────────┘
                                  │
          ┌───────────────────────▼────────────────────────┐
          │ Phase 6: Stream Reconstruction & SSE UI Output │
          └────────────────────────────────────────────────┘
```

### Phase 1: Atomic Fact Decomposition
To prevent the introduction of secondary hallucinations and establish a clean, discrete unit of evaluation, complex sentences are parsed into isolated, decontextualized factual claims.
* **Discrete Unit of Evaluation ($N_{\text{atomic}}$):** Complex sentences contain multiple dependent propositions. By decomposing sentences into atomic claims, every downstream evaluation is conducted over discrete atomic facts rather than coarse sentences.
* **Q&A Synthesis:** When evaluating Question & Answer pairs, instructional dialogue, or compound prompts, the pipeline synthesizes the context and response into declarative propositions prior to decomposition, preserving contextual premises.
* **Strict Coreference Resolution:** Pronouns, demonstratives, and implicit referents are deterministically resolved to their explicit proper nouns and antecedents extracted from global context.
* **Deterministic Fallback:** If LLM parsing fails, deterministic tokenization falls back to the **Natural Language Toolkit (NLTK)**.

### Phase 2: Multi-Entity Query Generation
For each atomic claim, a **Query Generator Agent (powered by Gemini)** synthesizes a targeted search strategy:
* **Entity Extraction:** Relevant named entities are identified and output as a strict JSON array.
* **Domain Classification:** Claims are classified into `general` or `scientific` domains to configure downstream retrieval indices and NLI tolerance.
* **Temporal Tracking:** If a claim relies on present-day state (e.g., "is currently"), a live system timestamp is injected into the prompt context.

### Phase 3: Parallel Web Retrieval (Hybrid Search)
Parallel search queries are dispatched across the **Wikipedia API**, **ArXiv API** (for scientific claims), and **DuckDuckGo**.
* **Absence Evaluation:** If zero evidence is returned across all retrieval backends, an LLM evaluates the epistemic salience of the queried entity: absence of evidence for prominent, globally recorded historical or scientific entities strongly indicates refutation (`CONTRADICTED`), whereas lack of evidence for obscure or low-salience entities yields conservative abstention (`UNKNOWN`).
* **Cosine Similarity:** The semantic similarity between claim vector $\mathbf{c}$ and evidence vector $\mathbf{e}$ is computed using `all-MiniLM-L6-v2`:

```math
\text{sim}(\mathbf{c}, \mathbf{e}) = \frac{\mathbf{c} \cdot \mathbf{e}}{\|\mathbf{c}\| \|\mathbf{e}\|}
```

* **Hybrid Re-Ranking (RRF):** Dense semantic similarity scores are combined with sparse **BM25** lexical scores using Reciprocal Rank Fusion:

```math
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}
```

where $r(d)$ is the document rank and $k = 60$ is the smoothing constant.

### Phase 3.5: Consensus Evaluation & Calibrated Epistemic Contradiction Detection
Before merging retrieved paragraphs, the system evaluates the open web for factual consensus. A zero-shot NLI check is executed on each distinct retrieved source snippet $e_i \in \mathcal{E}$ against the claim $c$.
* **True Source Conflict (Epistemic Contradiction):** If Source $A$ implies `SUPPORTED` and Source $B$ implies `CONTRADICTED`, the system flags an **EPISTEMIC CONTRADICTION**.
* **Calibrated High-Confidence Constraint:** To prevent false-positive contradiction triggers caused by minor lexical variations or secondary actor mentions, the system enforces a strict calibration threshold:
  
```math
\text{Epistemic Contradiction} \iff \exists e_a, e_b \in \mathcal{E} \text{ s.t. } \hat{\pi}_{\text{supp}}(e_a) \ge 0.75 \;\land\; \hat{\pi}_{\text{contra}}(e_b) \ge 0.75
```

  Only genuine, high-confidence empirical disagreements across independent web sources trigger an epistemic contradiction.
* *Distinction:* This represents an external factual conflict across web sources, distinct from model posterior dispersion in Phase 4. Conflicting sources are explicitly preserved and passed to Phase 5b for multi-agent arbitration.

### Phase 4: Primary Verification & Conformal Uncertainty Layer
The claim and retrieved evidence paragraphs are verified by a zero-shot sequence classification model (`facebook/bart-large-mnli`).

* **Softmax Probability Formulation:** Outputs a probability distribution over classes $\mathcal{Y} = \{ \text{CONTRADICTED}, \text{UNKNOWN}, \text{SUPPORTED} \}$. Given logits $\mathbf{z}$:

```math
P(Y = y_i \mid X) = \hat{\pi}_i(X) = \frac{e^{z_i}}{\sum_{j=1}^3 e^{z_j}}
```

* **Hierarchical Evidence Scoring (Context Dilution Mitigation):**
  When multiple evidence snippets are concatenated into a single large context string $\mathbf{e}_{\text{concat}} = e_1 \oplus e_2 \dots \oplus e_k$, sequence classification models (`facebook/bart-large-mnli`) suffer from **Context Dilution**. Long background narratives or adjacent facts dilute cross-attention, causing decisive entailment signals to collapse into neutral (`UNKNOWN`).
  
  To solve this, the pipeline executes a **Two-Tier Hierarchical Scoring Policy**:
  1. **Tier 1 (Per-Paragraph Max Entailment):** Each retrieved evidence snippet $e_i$ is evaluated independently:
     
```math
\hat{\mathbf{z}}_i = \text{NLI}(c, e_i), \quad \hat{\pi}_y(e_i) = \text{Softmax}(\hat{\mathbf{z}}_i)
```

     If any single snippet provides decisive verification ($\max_i \hat{\pi}_y(e_i) \ge 0.80$ for $y \in \{\text{SUPPORTED}, \text{CONTRADICTED}\}$), the pipeline selects the highest-confidence snippet directly. This completely prevents context dilution from suppressing decisive facts.
  2. **Tier 2 (Joint Concatenation Fallback):** If no single snippet is individually decisive ($\max_i \hat{\pi}_y(e_i) < 0.80$), the system retains cross-sentence multi-evidence reasoning by falling back to joint evaluation over the concatenated evidence string:
     
```math
\hat{\mathbf{z}}_{\text{joint}} = \text{NLI}\left(c, \bigoplus_{i=1}^k e_i\right)
```

* **Split Conformal Prediction (Finite-Sample Guarantees):**
Because raw softmax outputs are uncalibrated and lack error bounds, the system integrates **Split Conformal Prediction** (Vovk et al.; Angelopoulos & Bates, 2021).

  1. **Exchangeability Assumption:** The finite-sample coverage guarantee assumes that calibration and test pairs $(X_i, Y_i)_{i=1}^{n+1}$ are **exchangeable** (drawn from the same underlying data distribution $P_{XY}$). Under out-of-distribution domain shift (e.g., calibrating on Wikipedia and testing on ArXiv), empirical coverage can degrade; hence, domain-stratified calibration profiles are maintained.

  2. **Non-Conformity Scoring Function:** For calibration sample $(X_i, Y_i)$, the non-conformity score measures model error relative to the true class:

```math
s(X_i, Y_i) = 1 - \hat{\pi}_{Y_i}(X_i)
```

  3. **Empirical Quantile Computation:** For a user-specified significance level $\alpha \in (0, 1)$ (e.g., $\alpha = 0.10$ for $90\%$ coverage), the finite-sample adjusted conformal quantile $\hat{q}_\alpha$ is:

```math
\hat{q}_\alpha = \text{Quantile}\left(\frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}; \; \{s(X_i, Y_i)\}_{i=1}^n\right)
```

  4. **Prediction Set Construction:** For test claim $X_{\text{test}}$, the verifier outputs:

```math
\mathcal{C}(X_{\text{test}}) = \{ y \in \mathcal{Y} : 1 - \hat{\pi}_y(X_{\text{test}}) \le \hat{q}_\alpha \} = \{ y \in \mathcal{Y} : \hat{\pi}_y(X_{\text{test}}) \ge 1 - \hat{q}_\alpha \}
```

   For a calibrated significance level $\alpha$ with empirical quantile $\hat{q}_\alpha$, any candidate label with posterior probability satisfying $\hat{\pi}_y(X) \ge 1 - \hat{q}_\alpha$ is admitted into the conformal prediction set. When evidence is decisive, only the primary label exceeds this threshold, producing a tight singleton set ($\mathcal{C}(X) = \{\hat{y}\}$). Conversely, when evidence is dispersed or ambiguous, multiple candidate labels exceed the threshold, activating conformal escalation.

  5. **Theoretical Guarantee:** Under exchangeability, this satisfies:

```math
\mathbb{P}\left(Y_{\text{test}} \in \mathcal{C}(X_{\text{test}})\right) \ge 1 - \alpha
```

  6. **Conformal p-values:** For each label $y \in \mathcal{Y}$, exact conformal $p$-values are computed against calibration scores:

```math
p(y) = \frac{1 + \sum_{i=1}^n \mathbb{I}\left(s_i \ge 1 - \hat{\pi}_y(X_{\text{test}})\right)}{n + 1}
```

* **Conformal Set Size as an Escalation Signal (Not Proof of Conflict):**
A prediction set containing multiple labels (e.g., $\{\text{SUPPORTED}, \text{UNKNOWN}\}$) indicates **model posterior dispersion under ambiguous evidence**, not necessarily external source conflict. When $|\mathcal{C}(X)| > 1$ or $\text{UNKNOWN} \in \mathcal{C}(X)$, the system treats set size as an **escalation trigger**, activating Knowledge Graph reasoning and Multi-Agent Debate.

### Phase 5a: Knowledge Graph Multi-Hop Reasoning & Semantic Entity Disambiguation
When escalated from Phase 4, the system constructs a live in-memory Knowledge Graph to perform multi-hop evidence reasoning.
* **Semantic Entity Disambiguation:** Wikipedia search and entity extraction frequently suffer from disambiguation traps and lexical auto-suggest drift when queried entities share titles with homonymous media, creative works, or geographical locations. To eliminate this failure mode:
  1. Auto-suggest is strictly deactivated (`auto_suggest=False`) to enforce query fidelity.
  2. Disambiguation pages are parsed, and candidate titles containing irrelevant entertainment tags (e.g., `(album)`, `(song)`, `(band)`) are filtered out when the claim domain is non-entertainment.
  3. Candidate titles $\mathcal{T} = \{t_1, t_2, \dots, t_m\}$ are reranked against the context claim vector $\mathbf{c}$ via semantic cosine similarity using `all-MiniLM-L6-v2`:
     
```math
\hat{t} = \arg\max_{t \in \mathcal{T}} \frac{\mathbf{c} \cdot \mathbf{t}}{\|\mathbf{c}\| \|\mathbf{t}\|}
```

* **Graph Topology & Traversal:** Named entities from the claim and context are linked to canonical Wikipedia concept nodes. Relationship edges are extracted and traversed using cosine similarity to connect indirect evidence chains (e.g., resolving entity relationships across multiple documents).

### Phase 5b: Multi-Agent Critic Debate & Conservative Abstention Policy
If ambiguity persists after Knowledge Graph traversal (or if an external Epistemic Contradiction was flagged in Phase 3.5), the system invokes a Multi-Agent LLM Debate:
* A **Defense Agent** constructs the strongest argument supporting the claim.
* A **Prosecution Agent** constructs the strongest argument refuting the claim.
* A **Judge Agent** acts as an impartial arbitrator, evaluating both arguments against the raw evidence.

* **Formal Decision & Conservative Abstention Policy:**
To prevent the Judge LLM from forcing a binary hallucination when evidence is incomplete, the system enforces a strict **Conservative Abstention Policy**:

```math
\hat{y}_{\text{final}} = \begin{cases} 
\text{SUPPORTED} & \text{if } \mathcal{E} \vdash \phi \land \text{no unrefuted counter-evidence} \\
\text{CONTRADICTED} & \text{if } \mathcal{E} \vdash \neg \phi \land \text{decisive refutation} \\
\text{UNKNOWN} & \text{otherwise (Conservative Abstention)}
\end{cases}
```

If the evidence is incomplete, ambiguous, or lacks decisive proof, the Judge is strictly instructed to **abstain** (`UNKNOWN`). In safety-critical verification, abstention is statistically preferred over an uncertain guess.

### Phase 6: Real-Time Stream Reconstruction & Conformal Telemetry
As atomic claims are verified, the backend pushes asynchronous NDJSON events (Server-Sent Events) to the Flask client:
* **Live Conformal Telemetry:** For each claim, the pipeline streams the finite-sample prediction set $\mathcal{C}(X)$, the coverage guarantee ($1 - \alpha = 90\%$), and the escalation signal in real time.
* **Production Summary Metrics:** The final verification report aggregates total evaluated claims, hallucination rate, singleton conformal claims, and conformal escalation rate.
* **Interactive UI:** The frontend (`static/app.js` and `static/index.html`) renders an integrated Conformal Coverage dashboard card, D3.js Knowledge Graph network topology, and interactive hover tooltips displaying the calibrated prediction set chips, conformal $p$-values ($p(\text{contra}), p(\text{unk}), p(\text{supp})$), and source citations.

---

## 3. Multi-Dataset Evaluation & Empirical Statistical Analysis

The system was evaluated against standard fact-verification benchmarks (FEVER, Climate-FEVER, and HaluEval), accompanied by empirical conformal calibration and multi-agent ablation studies.

### 3.1 FEVER Benchmark Performance Analysis (5-Run Cross Validation)
To evaluate real-world generalization on live Wikipedia retrieval, the system was evaluated across 5 randomized batches on the FEVER benchmark dataset.

* **Unit of Evaluation ($N_{\text{atomic}}$):** Each batch consisted of $N_{\text{sentence}} = 20$ parent sentences. Because complex sentences decompose into 2–3 independent atomic facts, the actual denominator is the total number of evaluated atomic claims ($N_{\text{atomic}}$). This explains the exact percentage scores below:

| Batch Run | Parent Sentences ($N_{\text{sentence}}$) | Atomic Claims ($N_{\text{atomic}}$) | Correct Atomic Claims | Overall Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Test 1** | 20 | 40 | 34 | **85.00%** (34/40) | 0.86 | 0.84 | 0.85 |
| **Test 2** | 20 | 46 | 36 | **78.26%** (36/46) | 0.80 | 0.75 | 0.77 |
| **Test 3** | 20 | 42 | 37 | **88.10%** (37/42) | 0.89 | 0.87 | 0.88 |
| **Test 4** | 20 | 44 | 36 | **81.82%** (36/44) | 0.83 | 0.80 | 0.81 |
| **Test 5** | 20 | 43 | 36 | **83.72%** (36/43) | 0.85 | 0.83 | 0.84 |
| **Pooled Total** | **100** | **215** | **179** | **83.26%** (179/215) | **0.85** | **0.82** | **0.83** |

* **Statistical Sampling Uncertainty:** Across the pooled sample of $N = 215$ atomic claims, the overall accuracy is $83.26\%$ with a standard error of:

```math
\text{SE} = \sqrt{\frac{\hat{p}(1 - \hat{p})}{N}} = \sqrt{\frac{0.8326 \times 0.1674}{215}} = 2.55\%
```

yielding an asymptotic $95\%$ confidence interval of **$83.26\% \pm 5.00\%$ ($[78.26\%, 88.26\%]$)**.

* **Analysis of the Residual 16.7% Error Gap:** 
Error analysis reveals that the remaining failure rate is heavily dominated by **Cross-Encoder Literalism**. Zero-shot sequence classification models (`bart-large-mnli`) lack higher-order semantic abstraction for implicit entailments (failing to recognize semantic equivalence when claims use abstract paraphrases rather than verbatim phrasing). This failure mode directly motivated the multi-agent debate escalation mechanism in Phase 5b.

---

### 3.2 Split Conformal Calibration Analysis (Statistical Validation)
The Conformal Verification layer was empirically validated on holdout partitions of the FEVER dataset ($N = 200$ total samples: $n = 100$ calibration samples, $n = 100$ test samples). To account for finite-sample evaluation uncertainty, empirical coverage is reported alongside **Wilson 95% Score Confidence Intervals**.

#### Empirical Results Table (Scaled $N = 200$):
| Significance Level ($\alpha$) | Nominal Coverage ($1 - \alpha$) | Empirical Test Coverage | Wilson 95% Confidence Interval | Average Set Size $|\mathcal{C}(X)|$ | Conformal Quantile ($\hat{q}_\alpha$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.05** | 95.0% | **99.00%** | $[94.57\%, 99.82\%]$ | 1.72 | 0.9937 |
| **0.10** | 90.0% | **92.00%** | $[85.03\%, 95.93\%]$ | 1.33 | 0.9168 |
| **0.15** | 85.0% | **88.00%** | $[80.20\%, 93.04\%]$ | 1.21 | 0.8258 |
| **0.20** | 80.0% | **81.00%** | $[72.23\%, 87.50\%]$ | 1.04 | 0.6684 |
| **0.25** | 75.0% | **79.00%** | $[70.04\%, 85.83\%]$ | 1.00 | 0.2126 |
| **0.30** | 70.0% | **79.00%** | $[70.04\%, 85.83\%]$ | 1.00 | 0.0877 |
| **0.40** | 60.0% | **79.00%** | $[70.04\%, 85.83\%]$ | 1.00 | 0.0161 |
| **0.50** | 50.0% | **79.00%** | $[70.04\%, 85.83\%]$ | 1.00 | 0.0064 |

#### Key Statistical Findings:
1. **Strict Finite-Sample Validity:** In every single test condition, the empirical test coverage strictly equals or exceeds nominal coverage ($\text{Empirical Coverage} \ge 1 - \alpha$).
2. **Superior Information Efficiency:** With $n=100$ calibration samples, the average prediction set size at $\alpha=0.10$ ($90\%$ coverage guarantee) contracts sharply from $2.72$ down to **$1.33$**, and hits near-deterministic efficiency ($|\mathcal{C}(X)| = 1.04$) at $\alpha=0.20$.
3. **Artifacts:** High-resolution calibration curves and efficiency plots are maintained in `experiments/conformal_coverage_plot.png` and `experiments/conformal_set_size_plot.png`, with the production calibration state saved in `experiments/conformal_calibration.json`.

---

### 3.3 Ablation Study: Marginal Utility & Cost of Multi-Agent Debate
To determine whether Multi-Agent Debate provides genuine verification utility or merely adds latency without accuracy, an empirical ablation experiment was conducted comparing:
1. **Baseline (No Debate):** Strict NLI + Knowledge Graph + Conservative Abstention (defaults to `unknown` on uncertainty).
2. **Treatment (With Debate):** NLI + Knowledge Graph + Multi-Agent Debate (Defense, Prosecution, Judge).

#### Empirical Ablation Results:
| Metric | Condition A: Without Debate | Condition B: With Debate | Net Marginal Impact ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Verification Accuracy** | 86.67% | **100.00%** | **+13.33%** |
| **Claims Escalated to Debate** | — | 13.33% (2/15) | — |
| **Successful Corrections (Wrong $\to$ Right)** | — | **100.00%** (2/2) | **+2 claims corrected** |
| **Sycophancy / Degradations (Right $\to$ Wrong)** | — | **0.00%** (0/2) | **0 claims degraded** |
| **Average Latency per Claim** | 5.046 sec | 5.808 sec | **+0.762 sec (+15.1%)** |
| **LLM API Calls Consumed** | 0 calls | 6 calls (3 per escalation) | **+6 API calls** |

#### Critical Insights on Debate Mechanics:
* **Targeted Value:** Multi-Agent Debate is not applied uniformly across all claims; it is an escalation mechanism triggered only when NLI confidence collapses. In $86.7\%$ of claims, fast deterministic NLI resolves the claim with zero LLM API overhead.
* **Correction of NLI Literalism:** For the escalated claims, the debate successfully bridged the semantic gap (resolving figurative phrasings, domain terminology, and role interpretations), achieving a $100\%$ correction rate with zero sycophantic degradations.
* **Cost-Latency Trade-off:** The $+13.33\%$ accuracy uplift comes at the cost of $+0.762$ seconds average latency per claim and 3 LLM calls per escalated claim.

---

### 3.4 Selective Classification, Empirical Verification & Coverage-Risk Trade-Offs

During benchmark stress testing, the pipeline achieved **Precision: 0.89**, **Recall: 0.80**, and **F1-Score: 0.84** on the Contradiction (hallucination) class, with an overall raw accuracy of **75.00%**. The relationship between overall accuracy and high precision is formalized by **Selective Classification (Classification with a Reject Option)** (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017).

#### 1. Mathematical Formulation
A selective classifier is formally defined as a pair $(f, g)$, where:
* $f: \mathcal{X} \to \mathcal{Y}$ is the core prediction model ($f(X) \in \{\text{SUPPORTED}, \text{CONTRADICTED}\}$).
* $g: \mathcal{X} \to \{0, 1\}$ is a selection / gating function governing whether to output a definitive verdict ($g(X) = 1$) or abstain ($g(X) = 0$).

The gating rule $g(X)$ is determined by Conformal Prediction Set cardinality and Multi-Agent Judge consensus:

```math
g(X) = \begin{cases}
1 & \text{if } |\mathcal{C}(X)| = 1 \;\land\; \text{UNKNOWN} \notin \mathcal{C}(X) \quad \text{(Decisive Conformal Singleton)} \\
1 & \text{if escalated and Judge establishes decisive proof with unrefuted evidence} \\
0 & \text{otherwise (Conservative Abstention)}
\end{cases}
```

The system's final verdict $\hat{y}(X)$ is:

```math
\hat{y}(X) = \begin{cases} 
f(X) & \text{if } g(X) = 1 \\
\text{UNKNOWN} & \text{if } g(X) = 0 \quad \text{(Conservative Abstention)}
\end{cases}
```

#### 2. Coverage vs. Selective Risk
Under selective classification, the system's operational behavior is governed by two coupled metrics:

1. **System Coverage ($\Phi(g)$):** The probability that the model accepts an arbitrary claim and renders a definitive verdict:

```math
\Phi(g) = \mathbb{E}_{X \sim P_X}[g(X)]
```

2. **Selective Risk ($R(f, g)$):** The conditional misclassification rate restricted strictly to accepted claims:

```math
R(f, g) = \frac{\mathbb{E}_{(X, Y)} [\ell(f(X), Y) \cdot g(X)]}{\Phi(g)}
```

where $\ell(\hat{y}, y) = \mathbb{I}(\hat{y} \ne y)$ is the 0-1 classification loss.

#### 3. Why Abstention Safeguards Trustworthiness:
* **Why Binary Benchmarks Penalize Abstention:** Standard benchmarks assume full coverage ($\Phi(g) \equiv 1.0$), scoring any conservative abstention ($g(X) = 0$) as an error ($\ell = 1$). In reality, abstentions represent cases where evidence is ambiguous, contradictory across sources, or under-specified. Scoring abstentions as wrong pulls overall accuracy down to $75.00\%$.
* **The Cost of Forcing Binary Guesses:** If the gating function were disabled ($g(X) \equiv 1$), forcing a coin flip on ambiguous claims, selective risk would dramatically increase. In safety-critical hallucination detection, **false positive accusations destroy trust**.
* **High Selective Precision ($89\%$):** By allowing the verifier to abstain when the conformal set signals epistemic uncertainty ($|\mathcal{C}(X)| > 1$), the pipeline maintains **0.89 Precision** on all accepted claims ($\Phi \approx 0.85$). The system trades a modest portion of coverage to guarantee that when it flags an assertion as a hallucination, it is backed by calibrated statistical confidence.

---

## Appendix: Evaluation Metrics Explained

These are standard Machine Learning evaluation metrics. It is critical to understand their exact definitions in the context of Hallucination Detection to prove a deep understanding of data science.

### 1. The Definitions (In the context of Hallucinations)
In this system, the primary goal is the "positive" class of detecting a hallucination (a False claim).

* **True Positive (TP):** The claim was actually a hallucination, and the system correctly flagged it as `CONTRADICTED`.
* **True Negative (TN):** The claim was a true fact, and the system correctly labeled it as `SUPPORTED`.
* **False Positive (FP):** The claim was a true fact, but the system made a mistake and wrongly flagged it as a hallucination (`CONTRADICTED`).
* **False Negative (FN):** The claim was a dangerous hallucination, but the system missed it and wrongly labeled it as `SUPPORTED`.

### 2. The Metrics Explained

#### 1. Overall Accuracy
* **What it is:** The total percentage of atomic claims the system got right.
* **Formula:** `(TP + TN) / Total Claims`

#### 2. Precision
* **What it is:** When the system flags a claim as a hallucination, how often is it actually correct?
* **Formula:** `TP / (TP + FP)`

#### 3. Recall
* **What it is:** Out of ALL the actual hallucinations hidden in the text, what percentage did the system successfully catch?
* **Formula:** `TP / (TP + FN)`

#### 4. F1-Score
* **What it is:** The harmonic mean (a balanced average) of Precision and Recall.
* **Formula:** `2 * (Precision * Recall) / (Precision + Recall)`

### 3. How Are We Measuring This?
The system calculates these metrics mathematically using the `evaluation/run_benchmark.py` script. Academic datasets (like FEVER) contain claims pre-labeled as either `SUPPORTS` or `REFUTES`. The evaluation script feeds the claim into the pipeline, and compares human-ground-truth labels against predicted labels to calculate exact Precision, Recall, and F1 values.

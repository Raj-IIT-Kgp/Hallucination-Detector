# Hallucination Detection Pipeline

This document explains, step by step, how the detector decides whether a statement is a hallucination. Every stage is described with its purpose, its input and output, the exact algorithm the code runs, the mathematics behind it, a worked example taken from real runs of the system, and the reasons for the design choices. Section 3 covers the evaluation benchmarks and results, and Section 4 defines the evaluation metrics (TP, FP, TN, FN, precision, recall, F1) used to report them, with the confidence intervals, hypothesis tests, multiple-comparison corrections and power analysis used to compare them, and the calibration of the NLI probabilities. Section 5 collects the limitations, and Section 6 explains how to install and run the system.

**Summary.** The detector splits a text into atomic claims, retrieves evidence for each claim from Wikipedia, the web and ArXiv, and scores it with a natural language inference (NLI) model. Split conformal prediction then decides whether that verdict can stand or whether the claim must be escalated to a Knowledge Graph and an LLM arbiter, so the expensive stages run only on ambiguous claims. Results on 300 claims per benchmark (fifteen sets of 20, §3.3):

| Benchmark | Accuracy (Wilson 95% CI) | Precision | Recall | F1 | Settled by the NLI verdict | Mean latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FEVER | 90.7% [86.8%, 93.5%] | 0.94 | 0.89 | 0.91 | 229 of 300 | 8.5 s |
| FreshQA (never- and slow-changing) | 82.3% [77.6%, 86.2%] | 0.91 | 0.80 | 0.85 | 184 of 300 | 11.3 s |

Both benchmarks were measured with the current pipeline, end to end with live retrieval and LLM calls. Precision, recall and F1 refer to the hallucination class. Claims settled by the NLI verdict alone are correct 89% of the time, and the LLM arbiter, used on the rest, fixed 60 NLI verdicts and broke 15 (exact McNemar test $p < 0.001$, significant after correction for all eight tests reported, §2.9, §4.5). The raw NLI probabilities overstate how often the model is right (mean confidence 95.7% at 84.6% accuracy); temperature scaling, fitted by maximum likelihood, corrects the reported confidence without changing any verdict (§4.6). The main weaknesses (false premises, time-dependent and negated claims, and the conditions of the conformal guarantee) are discussed in Section 5.

**Contents**

1. Problem formulation and overview
2. Pipeline stages
   - 2.1 Claim decomposition
   - 2.2 Query generation
   - 2.3 Evidence retrieval
   - 2.4 Hybrid re-ranking (dense embeddings, BM25, reciprocal rank fusion)
   - 2.5 NLI evidence scoring
   - 2.6 Conformal uncertainty layer
   - 2.7 Routing: query rewrite, absence evaluation, escalation
   - 2.8 Knowledge Graph evidence from Wikidata
   - 2.9 LLM arbitration
   - 2.10 Output, report and user interface
   - 2.11 End-to-end algorithm and cost model
   - 2.12 Execution details
3. Evaluation: benchmarks, protocol, results, error analysis
4. Evaluation metrics, hypothesis tests, power and calibration
5. Limitations and future work
6. Reproducibility: installation, configuration and commands

---

## 1. Problem Formulation and Overview

### 1.1 Task

A **hallucination** is a statement produced by a language model that is not supported by, or is contradicted by, real-world evidence. Given an input text $T$ (typically an LLM answer), the detector must:

1. split $T$ into atomic claims $`c_1, \dots, c_n`$, each expressing one checkable fact;
2. for each claim $c$, find an evidence set $`\mathcal{E}(c) = \{e_1, \dots, e_k\}`$ from external sources (Wikipedia, ArXiv, the web);
3. assign a verdict

$$
f(c, \mathcal{E}(c)) \in \mathcal{Y} = \lbrace \text{CONTRADICTED}, \; \text{UNKNOWN}, \; \text{SUPPORTED} \rbrace
$$

where `CONTRADICTED` means the claim is a hallucination, `SUPPORTED` means the evidence confirms it, and `UNKNOWN` means the detector abstains because the evidence is insufficient or genuinely conflicting.

The detector is designed to keep **false accusations** (a true claim marked `CONTRADICTED`) rare, because a hallucination detector that wrongly flags correct answers loses the user's trust faster than one that occasionally abstains.

### 1.2 Design Principle: a Cascade

Verifying a claim with a large language model is accurate but slow and costly; verifying it with a small NLI classifier is fast but makes characteristic mistakes. The pipeline is therefore a **cascade**: every claim first goes through cheap deterministic stages (retrieval, NLI), and only claims on which those stages are demonstrably uncertain are escalated to expensive stages (Knowledge Graph construction, LLM arbitration). The uncertainty that triggers escalation is measured statistically by split conformal prediction (§2.6) rather than by hand-picked confidence thresholds.

One detector configuration serves every input: the live server (`server.py`) and every benchmark construct `HallucinationDetector(retriever)` with the same defaults. No setting depends on which dataset is being evaluated.

### 1.3 Pipeline Diagram

```
                         Input text T
                             │
                ┌────────────▼────────────┐
                │ 2.1 Claim decomposition │
                └────────────┬────────────┘
                             │  claims c1 … cn (verified concurrently)
                ┌────────────▼────────────┐
                │ 2.2 Query generation    │◄─────────────────────────┐
                └────────────┬────────────┘                          │
                ┌────────────▼────────────┐                          │
                │ 2.3 Evidence retrieval  │                          │
                │ 2.4 Hybrid re-ranking   │                          │
                └────────────┬────────────┘                          │
                     evidence found?  ── no ──► first pass? ── yes ──┤
                             │ yes                   │ no            │
                ┌────────────▼────────────┐   Absence evaluation     │ 2.7 Query
                │ 2.5 NLI evidence scoring│   (LLM) ──► verdict      │     rewrite
                └────────────┬────────────┘                          │     (once)
                ┌────────────▼────────────┐                          │
                │ 2.6 Conformal prediction│                          │
                └────────────┬────────────┘                          │
                     label UNKNOWN on first pass? ── yes ────────────┘
                             │ no
                     escalation needed? (2.7)
            ┌────────────────┼──────────────────────────┐
            │ no             │ no, but contradiction    │ yes
            │                │ is lone or contested     │
            ▼                ▼                          ▼
      NLI verdict     2.9 LLM review            2.8 Knowledge Graph
      stands          (1 call, no KG)               Wikidata evidence
                                                        │
                                                 2.9 LLM arbitration
                             │
                ┌────────────▼────────────┐
                │ 2.10 Report & UI        │
                └─────────────────────────┘
```

### 1.4 Notation

| Symbol | Meaning |
| :--- | :--- |
| $c$ | an atomic claim |
| $`e_i`$, $\mathcal{E}$ | the $i$-th evidence snippet; the set of snippets retrieved for a claim |
| $\mathcal{Y}$ | the label set $`\{\text{CONTRADICTED}, \text{UNKNOWN}, \text{SUPPORTED}\}`$ |
| $`\mathbf{z} \in \mathbb{R}^3`$ | NLI logits |
| $`\hat{\pi}_y(e)`$ | NLI probability of label $y$ when snippet $e$ is the premise and $c$ the hypothesis |
| $`\mathbf{c}, \mathbf{e}`$ | sentence embeddings of a claim and a paragraph |
| $`\mathcal{C}(X)`$ | conformal prediction set for input $X$ |
| $\alpha$, $`\hat{q}_\alpha`$ | conformal significance level; calibrated conformal quantile |
| $`\mathbb{I}(\cdot)`$ | indicator function (1 if the condition holds, else 0) |

---

## 2. Pipeline Stages

### 2.1 Claim Decomposition (`agents/claim_extractor.py`)

**Purpose.** A single sentence often bundles several facts, some true and some false: *"Tim Cook replaced Steve Jobs in 2011 and founded Apple"* is half right. Verifying the sentence as a whole would force one verdict onto two facts. Decomposition splits the input into **atomic claims**, each containing exactly one checkable proposition, so that every downstream decision concerns a single fact.

**Input → Output.** Text $T$ → list of claims $`[c_1, \dots, c_n]`$.

**Algorithm.**

1. **Atomic fast path.** If $T$ is a single sentence (NLTK sentence tokenizer) of at most 15 words and contains none of the clause markers `;`, `—`, `--`, `,`, ` and `, ` but `, ` although `, ` whereas `, ` which `, ` because `, ` while `, ` who `, ` where `, ` that `, ` with `, the sentence is already atomic and is returned unchanged. No LLM is called.
2. **LLM decomposition.** Otherwise the LLM is asked for a JSON array of atomic facts under two mandatory rules:
   - **Coreference resolution:** every pronoun and implicit reference (*he*, *the film*, *the city*) is replaced by the explicit proper noun from the context, so each claim can be verified on its own.
   - **Polarity preservation:** the truth value, negations and polarity of the original text must be kept verbatim; the model must never "correct" a false statement into a true one. Without this rule an LLM tends to rewrite *"Prescott is outside the US"* as *"Prescott is in the US"*, silently deleting the very hallucination the detector is supposed to find.

   A question–answer pair is first turned into a declarative statement and then decomposed.
3. **Fallback.** If the LLM call fails or its output is not valid JSON, $T$ is split into sentences with NLTK.

The claims are then verified **concurrently** (a thread pool of up to 5 workers), and results are returned in the original order.

**Worked example.**

| Input | Route | Output |
| :--- | :--- | :--- |
| *"Juneau is the capital of Alaska."* | fast path (6 words, no clause marker) | `["Juneau is the capital of Alaska."]` |
| *"Albert Einstein was born in 1950 in San Francisco, and he won the Nobel Prize in Literature in 1980."* | LLM (contains `,` and ` and `) | `["Albert Einstein was born in 1950.", "Albert Einstein was born in San Francisco.", "Albert Einstein won the Nobel Prize in Literature in 1980."]` (recorded run of the interface's example input) |

**Design rationale.** The fast path removes one LLM call (≈1–2 s and one unit of API quota) for the very common case of a single short statement; the marker list is conservative, so any sentence that might contain two propositions still goes to the LLM.

### 2.2 Query Generation (`agents/query_generator.py`)

**Purpose.** Searching for the whole claim string returns noisy results; searching for the **entities** the claim is about (people, organisations, works, places) retrieves the encyclopedia articles that actually contain the deciding fact. This stage also classifies the claim's domain and whether it depends on the present date.

**Input → Output.** Claim $c$ (plus optional context) → `{"queries": [q_1, …, q_m], "domain": "general" | "scientific", "requires_current_state": bool}` with $m \le 3$.

**Algorithm.**

1. **Fast path (no LLM)**, used when the claim has at most 15 words and either contains no ambiguous pronoun (*he, she, it, they, him, her, them, his, their, its, this, these, those*) or comes without surrounding context from which a pronoun could be resolved:
   - Claims about creative works (*album, film, movie, song, single, novel, series, …*) or using epithets (*"king of"*, *"father of"*, *"iron lady"*, …) are **excluded** from the fast path, because they need disambiguation (*"Pixels"* the film vs. the unit) or resolution of the epithet to a real person.
   - The claim is split on relational patterns, tried in order: *"is the co-founder/founder/CEO/… of"*, *"is from"*, *"is the capital/headquarters/member/part of"*, *"was born/located/based in"*, *"was founded/directed/written by"*, *"founded/created/directed"*, *"is a/an/the"*. The subject and the (capitalised) object become the entities; leading articles are removed, temporal tails such as *"in 2018"* are dropped, and internal capitalisation is preserved (`BharatPe`, `iPhone`).
   - The domain is `scientific` if the claim contains a science keyword (*celsius, fahrenheit, quantum, temperature, species, molecule, climate, dna*), and the temporal flag is set if it contains *current, currently, now, present, today, as of* or *active*.
2. **LLM path**, otherwise: the LLM returns the JSON object above, instructed to (i) disambiguate single-word entities (e.g. *"Pixels (2015 film)"*), (ii) always return at least one query, (iii) classify the domain, and (iv) set the temporal flag for claims about the present day (*"the current CEO"*, or implicit present tense such as *"Tim Cook is the CEO of Apple"*). The current date is included in the prompt. If the call fails, or its reply contains no query, the claim itself becomes the only query; if the reply is not valid JSON, the raw reply is used as the only query.
3. **Timestamp injection.** For temporal claims the evidence text assembled later is prefixed with `[SYSTEM TIMESTAMP: <today>]`, so that the NLI fallback and the LLM arbiter can reason about recency.

**Worked example** (outputs from the actual code; the LLM rows are from the current benchmark run, §3.3):

| Claim | Route | Queries |
| :--- | :--- | :--- |
| *"Ashneer Grover is the co-founder of BharatPe."* | fast path, pattern *"is the co-founder of"* | `["Ashneer Grover", "BharatPe"]` |
| *"Barack Obama was born in Hawaii."* | fast path, pattern *"was born in"* | `["Barack Obama", "Hawaii"]` |
| *"Bones is a television series."* | LLM (creative-work keyword *"series"*) | `["Bones (TV series)"]` |
| *"The 8th president of the United States is buried in Kinderhook, New York."* | LLM (no relational pattern matches) | `["Martin Van Buren", "Kinderhook (town) New York"]`: the implicit reference is resolved |

**Design rationale.** Most short factual claims follow a subject–relation–object pattern, so the regular expressions recover the right entities in under a millisecond with no API cost; anything that needs world knowledge (epithets, ambiguous titles, pronouns) is deliberately routed to the LLM.

### 2.3 Evidence Retrieval (`retrieval/web_search.py`)

**Purpose.** Collect candidate paragraphs that might confirm or refute the claim, from complementary sources: Wikipedia (curated, encyclopedic, high precision), ArXiv (scientific claims) and the open web (recent events not yet in Wikipedia).

**Input → Output.** Queries $`q_1, \dots, q_m`$, claim $c$, domain → a pool of candidate paragraphs, each with its text, URL and source.

**Algorithm.**

1. **Wikipedia (per query).** The top 2 search results are fetched with `auto_suggest=False` (auto-suggest silently replaces titles with similar-looking ones). A disambiguation page resolves to its first option. The page body is cut at 25,000 characters and everything from *See also*, *References* or *External links* onward is removed.
2. **ArXiv (per query, `scientific` claims only).** Up to 3 paper abstracts.
3. **Web search (once per claim).** A single web search on the **full claim text**, rather than on each entity: entity-only web searches return commercial SEO pages and trigger rate limits, while the whole proposition finds pages discussing the claim itself. Six search engines (`yahoo`, `bing`, `duckduckgo`, `brave`, `yandex`, `mojeek`, through the `ddgs` package) are queried in parallel and the first to return results is used, with at most 3 results. Individual engines are frequently blocked or rate-limited, so querying several in parallel makes web evidence available far more often than any single engine. The whole web search is capped at 5.0 s, the same cap as a Wikipedia page fetch, so a slow or blocked web search can never stall the pipeline.
4. **Chunking.** Wikipedia text is split into sentences (sentences of 20 characters or fewer and section headings are dropped) and grouped into **overlapping windows of 5 sentences with stride 3**. Window $j$ covers sentences $3j+1, \dots, 3j+5$, so consecutive windows share 2 sentences and a fact stated across two adjacent sentences is always contained whole in some window. Windows shorter than 100 characters are discarded. Each window is one candidate paragraph.
5. **Deduplication** of identical paragraphs, then re-ranking (§2.4), which returns the top 3.
6. **Claim-echo filtering** (web snippets only, applied by the detector to the 3 re-ranked snippets), described below. A snippet removed here is not replaced, so a claim can reach NLI scoring with fewer than 3 snippets.

All queries, and the sources within each query, are fetched concurrently; each query waits at most 5 s for its sources.

**Claim-echo filtering — mathematics.** A web search for the exact claim frequently returns pages that merely *quote* it: the original opinion piece, a repost, or a fact-check that quotes a false claim before debunking it. An NLI model reads a verbatim quote as perfect entailment, so a false claim could "support itself". Let $T(x)$ be the set of lower-cased word tokens of a text $x$. Every sentence $s$ of a web snippet is removed if it contains at least 80% of the claim's tokens:

$$
\text{echo}(s) \iff \frac{\lvert T(c) \cap T(s) \rvert}{\lvert T(c) \rvert} \ge 0.8
$$

The remaining sentences are kept, and a snippet left with fewer than 8 tokens is dropped. Wikipedia and ArXiv text is never filtered, because an encyclopedic sentence that closely matches a claim is genuine support, not an echo.

**Worked example (echo filtering)** (current benchmark run, §3.3). The filter removed at least one web snippet for 37 of the 600 claims. For *"The Japanese school year starts in April."* (true), the web search returned *"The school year in Japan starts at the beginning of April, usually around April 7 or 8 …"* and *"… Japan's academic year starts in April and ends in March …"*. Both contain at least 80% of the claim's tokens, so both were removed as echoes, although they are independent statements of the fact rather than quotes of the claim. The arbiter was left with a Wikipedia paragraph about school terms and abstained (§3.5). On short claims a genuine paraphrase easily reaches the 80% threshold, which is the cost of this protection (§5).

### 2.4 Hybrid Re-Ranking (`retrieval/web_search.py`)

**Purpose.** A query usually returns many candidate paragraphs (68 in the example below); only the few most relevant ones should be passed to the NLI model. Two complementary relevance signals are combined: **dense** semantic similarity, which recognises paraphrases, and **sparse** lexical matching (BM25), which rewards exact names, numbers and rare words.

**Input → Output.** Claim $c$ and $N$ candidate paragraphs → the top $k = 3$ evidence snippets.

**Dense similarity.** Claim and paragraphs are embedded with `all-MiniLM-L6-v2`, a 6-layer sentence transformer (Reimers & Gurevych, 2019) producing 384-dimensional vectors (inputs truncated to 256 word pieces). For a text with token embeddings $`\mathbf{h}_1, \dots, \mathbf{h}_L`$ from the last layer, the sentence embedding is the L2-normalised mean:

$$
\bar{\mathbf{h}} = \frac{1}{L} \sum_{t=1}^{L} \mathbf{h}_t, \qquad \mathbf{e} = \frac{\bar{\mathbf{h}}}{\lVert \bar{\mathbf{h}} \rVert_2}
$$

and relevance is the cosine similarity, which for unit vectors is simply a dot product:

$$
\text{sim}(\mathbf{c}, \mathbf{e}) = \frac{\mathbf{c} \cdot \mathbf{e}}{\lVert \mathbf{c} \rVert \, \lVert \mathbf{e} \rVert} = \mathbf{c} \cdot \mathbf{e} \in [-1, 1]
$$

**Sparse similarity (BM25 Okapi; Robertson & Zaragoza, 2009).** Claim and paragraphs are lower-cased and split on whitespace. For a claim with tokens $`t_1, \dots, t_Q`$, the score of paragraph $d$ is

$$
\text{BM25}(d, c) = \sum_{j=1}^{Q} \text{IDF}(t_j) \cdot \frac{f(t_j, d) \, (k_1 + 1)}{f(t_j, d) + k_1 \left( 1 - b + b \, \dfrac{\lvert d \rvert}{\text{avgdl}} \right)}
$$

where $`f(t, d)`$ is the number of occurrences of $t$ in $d$, $\lvert d \rvert$ the paragraph length in tokens, $\text{avgdl}$ the average length over the $N$ candidates, $`k_1 = 1.5`$ controls term-frequency saturation, and $b = 0.75$ controls length normalisation. With $`n(t)`$ the number of candidates containing $t$:

$$
\text{IDF}(t) = \ln \frac{N - n(t) + 0.5}{n(t) + 0.5}
$$

A word that appears in more than half of the candidates would get a negative IDF; such values are replaced by $`0.25 \cdot \overline{\text{IDF}}`$ (a quarter of the mean IDF), so common words contribute a little but never penalise a paragraph.

**Reciprocal Rank Fusion.** The two scores live on incomparable scales (cosine in $[-1, 1]$, BM25 unbounded), so they are combined through their **ranks** rather than their values. With $`r_{\text{dense}}(d)`$ and $`r_{\text{bm25}}(d)`$ the 1-based ranks of $d$ under each score:

$$
\text{RRF}(d) = \frac{1}{\kappa + r_{\text{dense}}(d)} + \frac{1}{\kappa + r_{\text{bm25}}(d)}, \qquad \kappa = 60
$$

The constant $\kappa$ flattens the contribution of the very top ranks, so a paragraph ranked well by *both* signals beats one ranked first by only one of them (Cormack et al., 2009).

**Selection.** Paragraphs are visited in decreasing RRF order, and the first $k = 3$ that pass a relevance floor are kept:

$$
\text{keep}(d) \iff \text{sim}(\mathbf{c}, \mathbf{e}_d) \ge 0.20 \;\lor\; \text{BM25}(d, c) > 1.0
$$

If none qualifies, the single best paragraph is kept only if it is at least marginally relevant ($`\text{sim} \ge 0.15`$ or $`\text{BM25} > 0.5`$); otherwise retrieval returns nothing, which triggers the query rewrite and, eventually, absence evaluation (§2.7) instead of feeding unrelated text to the NLI model.

**Worked example** (current benchmark run, FEVER claim *"Bones is a cuisine."*, fast-path query `["Bones"]`): Wikipedia and the web gave 68 unique candidates (65 Wikipedia paragraphs and 3 web results, $\text{avgdl} = 90.8$ tokens). Top of the fused ranking:

| RRF | Dense rank (sim) | BM25 rank (score) | Source | Paragraph |
| :--- | :--- | :--- | :--- | :--- |
| 0.03252 | 2 (0.506) | 1 (7.66) | web | *"Bone swimming cuisine can be said to be a live fish cuisine…"* |
| 0.03125 | 4 (0.479) | 4 (3.01) | Wikipedia | *"A bone is a rigid organ that constitutes part of the skeleton in most vertebrate animals…"* |
| 0.03083 | 8 (0.465) | 2 (3.08) | Wikipedia | *"The largest bone in the body is the femur or thigh-bone…"* |
| 0.02854 | 16 (0.409) | 5 (2.97) | Wikipedia | *"A protrusion from a bone can be called…"* |

For the first row: $`\text{RRF} = \frac{1}{60 + 2} + \frac{1}{60 + 1} = 0.016129 + 0.016393 = 0.032522`$. The third paragraph, only eighth by dense similarity, enters the top 3 because BM25 ranks it second: fusion rewards agreement between the two rankings. The top three become the evidence. The query `"Bones"` retrieved the anatomy article rather than the TV series, which is enough here: the evidence shows what bones are, and no paragraph describes a cuisine of that name.

### 2.5 NLI Evidence Scoring (`verification/nli_verifier.py`, `detector/hallucination_detector.py`)

**Purpose.** Decide, for each evidence snippet, whether it entails the claim, contradicts it, or neither — the core verification signal — and combine the per-snippet judgements into one verdict.

**Input → Output.** Claim $c$ and evidence $`e_1, \dots, e_k`$ → a verdict $\hat{y}$ with a probability vector $`\hat{\boldsymbol{\pi}} = (\hat{\pi}_{\text{con}}, \hat{\pi}_{\text{unk}}, \hat{\pi}_{\text{sup}})`$, and a flag saying whether an LLM must review the verdict.

**The NLI model.** Natural Language Inference asks whether a *premise* entails, contradicts, or is neutral towards a *hypothesis*. The detector uses `facebook/bart-large-mnli`: BART-large (Lewis et al., 2020; a 12-layer encoder and 12-layer decoder Transformer, hidden size 1,024, 16 attention heads, ≈407M parameters) fine-tuned on the Multi-Genre NLI corpus (MultiNLI, Williams et al., 2018; ≈433k sentence pairs). The evidence is the premise and the claim the hypothesis, encoded as

`<s> evidence </s></s> claim </s>`

truncated to the model's 1,024-token limit (tokens are removed from the end of the longer of the two texts, in practice the evidence). The decoder's final hidden state at the last `</s>` token, $`\mathbf{h}_{\text{eos}} \in \mathbb{R}^{1024}`$, summarises the pair, and the classification head maps it to three logits:

$$
\mathbf{z} = \mathbf{W}_2 \tanh\left( \mathbf{W}_1 \mathbf{h}_{\text{eos}} + \mathbf{b}_1 \right) + \mathbf{b}_2, \qquad \mathbf{W}_1 \in \mathbb{R}^{1024 \times 1024}, \; \mathbf{W}_2 \in \mathbb{R}^{3 \times 1024}
$$

The model's output order (contradiction, neutral, entailment) is mapped through its `id2label` table onto the pipeline's labels (CONTRADICTED, UNKNOWN, SUPPORTED), and the logits are normalised with softmax:

$$
\hat{\pi}_y = \frac{\exp(z_y)}{\sum_{y' \in \mathcal{Y}} \exp(z_{y'})}, \qquad \hat{y} = \underset{y \in \mathcal{Y}}{\arg\max} \; \hat{\pi}_y, \qquad \text{confidence} = \max_{y} \hat{\pi}_y
$$

(Another NLI model can be selected with the `NLI_MODEL` environment variable; the label mapping adapts automatically, but the conformal layer must then be recalibrated, §2.6.)

**Why snippets are scored separately.** If all snippets are concatenated into one long premise, the decisive sentence is surrounded by background text and the model's attention is spread across it; decisive entailments collapse towards *neutral* (**context dilution**). Each snippet $`e_i`$ is therefore scored on its own, giving $`\hat{\boldsymbol{\pi}}(e_i)`$ for $i = 1, \dots, k$. (Each snippet is run in a separate forward pass, which avoids padding short snippets to the length of the longest one.) The per-snippet judgements are then combined in three steps. A snippet is called **decisive** for label $y$ when

$$
\hat{\pi}_y(e_i) \ge \tau = 0.80
$$

**Step 1 — Source-conflict check.** If one source confidently supports the claim and another confidently contradicts it, with similar confidence, the sources genuinely disagree and neither should win automatically. With $`S = \max_i \hat{\pi}_{\text{sup}}(e_i)`$ and $`K = \max_i \hat{\pi}_{\text{con}}(e_i)`$ (maxima over snippets whose predicted label is supported / contradicted):

$$
\text{conflict} \iff S \ge 0.85 \;\land\; K \ge 0.85 \;\land\; \lvert S - K \rvert < 0.12
$$

On conflict the verdict is set to `UNKNOWN` with uniform probabilities $(0.33, 0.34, 0.33)$, and the evidence text is prefixed with an *EPISTEMIC CONTRADICTION* note so the LLM arbiter knows the sources disagree. (This is disagreement between *sources*; uncertainty of the *model* is handled by the conformal layer, §2.6.)

**Step 2 — Decisive snippets with corroborated contradictions.** Support and contradiction are not equally reliable. MNLI-trained models often assign a confident *contradiction* to a snippet that is merely adjacent to the claim, because they learned that a premise describing a different situation contradicts the hypothesis. Example from the current run: the true claim *"Tamerlan Tsarnaev carjacked an SUV"* was "refuted" with $`\hat{\pi}_{\text{con}} = 0.91`$ by a web forum post doubting the official account, while the Wikipedia sentence *"the Tsarnaev brothers killed an MIT policeman, carjacked an SUV …"* scored only 0.57 for support; as a lone contradiction it went to the LLM review, which judged it `SUPPORTED`. On the 600 claims of the current evaluation (§3.3), the number of decisive contradicting snippets is the strongest available signal of whether a refutation is real:

| Decisive contradicting snippets | Claims | Claim actually false | Share of refutations that were wrong |
| :--- | :--- | :--- | :--- |
| 1 | 99 | 70 | 29% |
| 2 | 62 | 55 | 11% |
| 3 or more | 113 | 108 | 4% |
| decisive support also present | 29 | 9 | 69% |

A lone contradicting snippet was wrong 29% of the time and a contradiction contested by decisive support 69% of the time, against 11% for two agreeing snippets. Lone and contested contradictions are therefore reviewed by the LLM, and two agreeing snippets are accepted directly, so $m = 2$. Let

$$
D_{\text{sup}} = \lbrace  i : \hat{y}(e_i) = \text{SUPPORTED},\; \hat{\pi}_{\text{sup}}(e_i) \ge \tau  \rbrace, \qquad D_{\text{con}} = \lbrace  i : \hat{y}(e_i) = \text{CONTRADICTED},\; \hat{\pi}_{\text{con}}(e_i) \ge \tau  \rbrace
$$

The verdict is then

$$
\hat{y} = \begin{cases}
\text{SUPPORTED} & \text{if } \lvert D_{\text{sup}} \rvert \ge 1 \;\land\; D_{\text{con}} = \varnothing \\\\
\text{CONTRADICTED} & \text{if } \lvert D_{\text{con}} \rvert \ge m \;\land\; D_{\text{sup}} = \varnothing \;\land\; \neg\,\text{shared role} \\\\
\text{LLM review} & \text{if } \lvert D_{\text{con}} \rvert \ge 1 \text{ and neither case above applies} \\\\
\text{Step 3} & \text{if } D_{\text{sup}} = D_{\text{con}} = \varnothing
\end{cases}
\qquad m = 2
$$

The verdict takes the probability vector of its highest-confidence decisive snippet. **LLM review** means the claim keeps that NLI posterior (needed by the conformal layer) but is passed to the LLM arbiter for a final decision (§2.7, §2.9): this happens for a lone contradiction, for a contradiction contested by a decisive supporting snippet, and for shared-role claims.

*Shared role* means the claim contains a non-exclusive title (*co-founder, co-author, co-creator, member, director, investor, actor*). MNLI models exhibit an **exclusivity bias**: *"A co-founded X"* is "contradicted" by a snippet stating that *B* co-founded X, although both can be true. Contradictions of such claims are never accepted directly.

**Step 3 — Joint fallback.** If no snippet is decisive, the snippets are concatenated (`Evidence 1: … Evidence 2: …`, with the timestamp line for temporal claims) and scored once. This lets the model combine facts spread across sentences — the one case where concatenation helps — and its output is the verdict.

**Single-snippet evidence.** If only one snippet was retrieved, it is scored directly; a decisive contradiction from a single snippet can never be corroborated and is therefore sent to LLM review.

**Worked example** (current benchmark run, claim *"Bones is a cuisine."* of §2.4, FEVER label `REFUTES`):

| Snippet | $`\hat{\pi}_{\text{con}}`$ | $`\hat{\pi}_{\text{unk}}`$ | $`\hat{\pi}_{\text{sup}}`$ | Label | Decisive? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *"Bone swimming cuisine can be said to be a live fish cuisine…"* (web) | 0.2973 | 0.0717 | 0.6310 | supported | no (0.631 < 0.80) |
| *"A bone is a rigid organ…"* (Wikipedia) | 0.9594 | 0.0388 | 0.0018 | contradicted | yes |
| *"The largest bone in the body is the femur…"* (Wikipedia) | 0.9934 | 0.0063 | 0.0003 | contradicted | yes |

Step 1: $S = 0.631 < 0.85$, so there is no source conflict. Step 2: $`\lvert D_{\text{con}} \rvert = 2 \ge m`$ and $`D_{\text{sup}} = \varnothing`$, so the verdict is **CONTRADICTED** with $`\hat{\boldsymbol{\pi}} = (0.9934, 0.0063, 0.0003)`$, with no LLM call; the conformal set is the singleton {CONTRADICTED} (§2.6), so the verdict stands. Under a "highest confidence wins" rule the same verdict would follow here; the corroboration rule only changes the outcome when a contradiction stands alone or is contested.

**Calibrated confidence.** The probabilities above are over-confident on retrieved evidence: over the 560 benchmark claims whose NLI verdict is definite, the mean confidence is 95.7% but the accuracy 84.6%. The pipeline therefore also reports a **temperature-scaled** probability vector, $`\tilde{\boldsymbol{\pi}} = \text{softmax}(\mathbf{z} / T)`$, with $T = 2.48$ fitted by maximum likelihood (§4.6). Dividing all logits by the same $T > 0$ preserves their order, so the verdict is unchanged; only its stated confidence moves. For the *"Bones is a cuisine"* example above, the deciding snippet's logits $`\mathbf{z} = (4.084, -0.974, -3.995)`$ become $`\mathbf{z}/T = (1.646, -0.393, -1.610)`$, and the confidence in CONTRADICTED falls from 0.993 to 0.856. Decisions (the threshold $\tau$, the corroboration rule and the conformal layer) keep the raw probabilities they were designed and measured with, so the results of Section 3 are unaffected; the calibrated vector is recorded as `calibrated_probabilities`, and the interface shows its value for the verdict as the calibrated confidence next to the raw probabilities.

**Is $m = 2$ fitted to these claims?** The value of $m$ was read off the table above, which comes from the same claims the accuracy is reported on, so the reported accuracy could be optimistic. `experiments/statistical_tests.py` checks this by cross-fitting (§4.5, where the method and the reason it removes the bias are derived). For every claim, its trace determines the verdict it would have received with $m = 1$ (a lone contradiction is accepted without review, so the claim takes its NLI verdict) or $m = 3$ (two agreeing contradictions are also reviewed: one LLM review of the saved evidence, as in §2.6). This concerns 61 of the 62 claims with two decisive contradicting snippets in the table; the 62nd was escalated, so $m$ does not affect it. All 61 reviews were made (one LLM call each, cached in `shadow_reviews.jsonl`), so no verdict is missing. Then $m \in \lbrace 1, 2, 3 \rbrace$ is chosen by accuracy on fourteen of the fifteen sets and the fifteenth set is scored with it, and separately one claim at a time is left out; a tie goes to the smaller $m$, which needs fewer LLM calls. On all 600 claims $m = 1$ gives 503 correct verdicts, $m = 2$ 519 and $m = 3$ 519, so almost every fold chooses $m = 2$ (fourteen of the fifteen sets, and 598 of the 600 leave-one-out folds). The cross-fitted accuracy is 517 of 600, two claims below the 519 of the fixed $m = 2$: the optimism from choosing $m$ on these claims is at most about 0.3 points, far inside the confidence intervals of §3.3. The advantage of $m = 2$ over $m = 1$ rests on the 28 claims whose verdict differs: 21 are right only with $m = 2$, 5 only with $m = 1$, and 2 with neither, and the exact McNemar test (§4.5) gives $p = 0.002$, significant also after correction for multiple tests. Against $m = 3$ only 5 verdicts differ (2 right only with $m = 2$, 2 only with $m = 3$, 1 wrong with both; $p = 1$), so the two are indistinguishable, and $m = 3$ would cost 61 more LLM reviews. Cross-fitting covers the choice of $m$ only; the other constants of the rule ($\tau = 0.80$ and the conflict thresholds) are not re-validated here.

**Design rationale and limitations.** The rule is asymmetric on purpose: lone and contested contradictions are often wrong (29% and 69% of the time, against 11% for two agreeing snippets), and a false accusation is the more harmful error (§1.1). Support is not immune: of the 219 claims accepted as `SUPPORTED` on decisive support without review, 29 (13%) were wrong, 14 of the 65 (22%) with a single supporting snippet. Support is nonetheless accepted directly, because a missed hallucination is the less harmful error (§1.1) and reviewing every lone supporting snippet would add an LLM call for about one claim in nine. Corroborated contradictions — the typical case for genuinely false encyclopedic claims, which several Wikipedia sentences contradict — stay on the fast path, so the extra LLM cost is limited to the ambiguous cases (35 of 300 FEVER and 54 of 300 FreshQA claims in the current evaluation, §3.4). The NLI model cannot do arithmetic or date reasoning and ignores words such as *"currently"*; such cases surface as residual errors (§3.5). `HallucinationDetector(min_contra_snippets=None)` restores the original rule (the highest-confidence decisive snippet wins regardless of label) for ablation studies; a randomised check over 20,000 synthetic score configurations confirms it reproduces the original decisions exactly.

### 2.6 Conformal Uncertainty Layer (`statistical_layer/conformal_predictor.py`)

**Purpose.** The softmax output of a neural network is not a calibrated probability: a model can be 95% "confident" and wrong far more than 5% of the time. The pipeline needs a principled way to decide *when the NLI verdict can be trusted* and when the claim must be escalated. **Split conformal prediction** (Vovk et al., 2005; Angelopoulos & Bates, 2021) converts the softmax output into a **set of plausible labels** with a statistical guarantee that does not depend on the model being well calibrated.

**Input → Output.** Probability vector $`\hat{\boldsymbol{\pi}}(X)`$ → prediction set $`\mathcal{C}(X) \subseteq \mathcal{Y}`$, conformal p-values, and an escalation flag.

**Step 1 — Non-conformity score.** A non-conformity score measures how "unusual" a labelled example is for the model. The pipeline uses one minus the probability assigned to the true label:

$$
s(X, Y) = 1 - \hat{\pi}_{Y}(X) \in [0, 1]
$$

A confident correct prediction gives a score near 0; a prediction that gave the true label little probability gives a score near 1.

**Step 2 — Calibration.** On a held-out calibration set of $n$ labelled claims, never used for anything else, the scores $`s_i = s(X_i, Y_i)`$ are computed and sorted, $`s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}`$. For significance level $\alpha$, the conformal quantile is the $k$-th smallest score:

$$
\hat{q}_\alpha = s_{(k)}, \qquad k = \left\lceil (n + 1)(1 - \alpha) \right\rceil
$$

If $k > n$ the calibration set is too small for the requested level and $`\hat{q}_\alpha = 1`$ (every label is always included). The index is computed exactly from the sorted scores rather than with an interpolating quantile function, because interpolation shifts the index and breaks the finite-sample guarantee below.

The production calibration (`experiments/conformal_calibration.json`) uses $n = 100$ FEVER claims, each scored by the NLI model against its **gold** evidence sentences from the dataset (one forward pass on the concatenated gold sentences), and $\alpha = 0.10$, so $`k = \lceil 101 \times 0.9 \rceil = \lceil 90.9 \rceil = 91`$ and $`\hat{q}_{0.10} = s_{(91)} = 0.8985`$.

**Step 3 — Prediction set.** For a new claim $X$, every label whose score would not be unusually high is included:

$$
\mathcal{C}(X) = \lbrace  y \in \mathcal{Y} : s(X, y) \le \hat{q}_\alpha  \rbrace = \lbrace  y \in \mathcal{Y} : \hat{\pi}_y(X) \ge 1 - \hat{q}_\alpha  \rbrace
$$

With $`\hat{q} = 0.8985`$, a label is admitted when its probability is at least $`1 - 0.8985 = 0.1015`$. If no label qualifies (possible only with probability at most $\alpha$), the set falls back to the single most probable label.

**Step 4 — Coverage guarantee.** If the calibration claims and the new claim are **exchangeable** (their joint distribution is unchanged by reordering them, e.g. i.i.d. draws from the same distribution), then

$$
\mathbb{P}\left( Y_{n+1} \in \mathcal{C}(X_{n+1}) \right) \ge 1 - \alpha
$$

*Proof sketch.* Let $`s_{n+1} = s(X_{n+1}, Y_{n+1})`$ be the score of the new claim at its true label. By exchangeability, $`s_{n+1}`$ is equally likely to occupy any of the $n + 1$ positions in the sorted list of all $n + 1$ scores (ties broken at random). The true label is in the set exactly when $`s_{n+1} \le s_{(k)}`$, i.e. when $`s_{n+1}`$ is among the $k$ smallest of the $n + 1$ scores. Hence

$$
\mathbb{P}\left( Y_{n+1} \in \mathcal{C}(X_{n+1}) \right) = \mathbb{P}\left( s_{n+1} \le s_{(k)} \right) \ge \frac{k}{n + 1} = \frac{\lceil (n+1)(1-\alpha) \rceil}{n + 1} \ge 1 - \alpha
$$

If the scores have no ties, coverage is also bounded above by $`1 - \alpha + \frac{1}{n+1}`$, so the sets are not needlessly large. With $n = 100$ and $\alpha = 0.10$, coverage lies between $`91/101 = 90.1\%`$ and $`90\% + 1/101 \approx 91.0\%`$.

**What the guarantee does and does not say.** It is a **marginal** guarantee about the *set*, averaged over claims. It does not say that a single point verdict is right 90% of the time (a set $`\{\text{CONTRADICTED}, \text{UNKNOWN}\}`$ can contain the truth while the point verdict *CONTRADICTED* is wrong); it does not bound the precision of `CONTRADICTED` verdicts; and coverage can be lower for the rarer class and higher for the common one. It also requires exchangeability, and the deployment setting differs from the calibration setting in two ways: the live pipeline scores *retrieved* snippets (§2.3–2.5), which are noisier than FEVER's gold evidence, and its claims may come from other domains. The guarantee is therefore exact only for the gold-evidence setting; on retrieved evidence it was measured directly (below), and in deployment the conformal set is best read as a calibrated escalation trigger. The layer must also be recalibrated whenever the NLI model changes.

**Step 5 — Conformal p-values.** For each label $y$, the p-value is the fraction of calibration scores at least as large as the score that label would have:

$$
p(y) = \frac{1 + \sum_{i=1}^{n} \mathbb{I}\left( s_i \ge 1 - \hat{\pi}_y(X) \right)}{n + 1}
$$

A small $p(y)$ means that, had $y$ been the true label, the model's behaviour would have been more unusual than almost all calibration examples, i.e. $y$ is implausible. The set of Step 3 equals, up to ties, the set of labels with $p(y) > \alpha$. The p-values are shown in the UI.

**Step 6 — Escalation signal.** A set with more than one label, or one containing `UNKNOWN`, means the NLI posterior is spread across labels and the verdict should not stand on its own:

$$
\text{escalate}_{\text{conf}}(X) \iff \lvert \mathcal{C}(X) \rvert > 1 \;\lor\; \text{UNKNOWN} \in \mathcal{C}(X)
$$

A multi-label set signals *model* uncertainty under ambiguous evidence, not necessarily disagreement between sources (which Step 1 of §2.5 detects).

**Worked examples** (real outputs of `ConformalVerifier.predict_set`, $`\hat{q} = 0.8985`$, admission threshold 0.1015):

| $`\hat{\boldsymbol{\pi}} = (\pi_{\text{con}}, \pi_{\text{unk}}, \pi_{\text{sup}})`$ | $`\mathcal{C}(X)`$ | p-values (con, unk, sup) | Escalate? |
| :--- | :--- | :--- | :--- |
| (0.9934, 0.0063, 0.0003) — the *"Bones is a cuisine"* example of §2.5 | {CONTRADICTED} | (0.485, 0.050, 0.010) | no — verdict stands |
| (0.62, 0.30, 0.08) | {CONTRADICTED, UNKNOWN} | (0.208, 0.168, 0.089) | yes — two labels |
| (0.05, 0.45, 0.50) | {UNKNOWN, SUPPORTED} | (0.059, 0.208, 0.208) | yes — two labels, includes UNKNOWN |

In the second row, $0.62$ and $0.30$ exceed $0.1015$ but $0.08$ does not, so the set has two labels. In the first row, $p(\text{con}) = 0.485$ means 48 of the 100 calibration scores are at least $`1 - 0.9934 = 0.0066`$.

**Empirical validation** (`experiments/eval_conformal.py`: FEVER with gold evidence, $n = 100$ calibration claims and 100 separate test claims; coverage reported with 95% Wilson score intervals, §4.4):

| $\alpha$ | Nominal coverage $1 - \alpha$ | Empirical coverage | Wilson 95% CI | Mean set size | $`\hat{q}_\alpha`$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.05 | 95% | 99.00% | [94.55%, 99.82%] | 1.63 | 0.9881 |
| **0.10** | **90%** | **90.00%** | **[82.56%, 94.48%]** | **1.26** | **0.8985** |
| 0.15 | 85% | 85.00% | [76.72%, 90.69%] | 1.13 | 0.7787 |
| 0.20 | 80% | 79.00% | [70.02%, 85.83%] | 1.01 | 0.6286 |

At the production level $\alpha = 0.10$ the empirical coverage equals the nominal 90% with only 1.26 labels per set on average, so most claims receive a decisive singleton and only a minority are escalated. For $\alpha \ge 0.20$ the sets are already almost always singletons, so coverage cannot fall further and stays at the raw accuracy of the NLI model on gold evidence (79%). Plots: `experiments/conformal_coverage_plot.png`, `experiments/conformal_set_size_plot.png`.

**Coverage on retrieved evidence** (`experiments/full_evaluation.py`, the 600 claims of the current evaluation, §3.3). Here the NLI probabilities are the ones the live pipeline hands to the conformal layer, computed on retrieved snippets. For the recalibration the claims were split into two halves; each half was tested with the $`\hat{q}`$ calibrated on the other (cross-fitting), so no claim was used to calibrate its own threshold:

| Threshold | Coverage on retrieved evidence | Wilson 95% CI | Mean set size | Estimated accuracy (FEVER / FreshQA) |
| :--- | :--- | :--- | :--- | :--- |
| current, $`\hat{q} = 0.8985`$ (gold-evidence calibration) | 83.2% | [80.0%, 85.9%] | 1.18 | 90.7% / 82.3% (measured) |
| recalibrated on retrieved evidence, $`\hat{q} \approx 0.990`$–$`0.995`$ | 90.3% | [87.7%, 92.4%] | 1.80 | 92.3% / 84.3% (estimated) |

With the current threshold the coverage on live evidence is measurably below the nominal 90% (the interval lies below it), as the exchangeability argument predicted. Recalibrating brings coverage to 90.3%, with an interval that contains 90%, but only with a threshold between 0.990 and 0.995 (the two folds), which admits any label above about 0.5–1% probability: 181 more claims would be escalated, and the estimated accuracy rises from 86.5% to 88.3% pooled (530 against 519 correct). The gain is not significant: the recalibrated verdicts fix 21 claims and break 10 (exact McNemar test, §4.5, $p = 0.07$), at the cost of an LLM review for almost a third of all claims. (For the estimate, claims whose route would change were re-decided: newly escalated claims received one LLM review of their saved evidence, without the Knowledge Graph evidence.) The production threshold is therefore kept at 0.8985, and it is described as what it is in deployment: a calibrated escalation trigger with 83.2% measured coverage, not a 90% guarantee.

### 2.7 Routing: Query Rewrite, Absence Evaluation and Escalation (`detector/hallucination_detector.py`)

**Purpose.** Decide what happens to each claim after scoring: accept the verdict, search again, judge the absence of evidence, ask the LLM to review, or escalate to full reasoning.

**Self-reflective query rewrite (at most once per claim).** Retrieval can fail because a query was ambiguous, too specific, or used a name Wikipedia does not. On the **first pass only**, if no evidence was found or the verdict is `UNKNOWN`, a rewriter LLM receives the claim, the queries already tried, and up to 1,000 characters of the unhelpful evidence; it is asked to diagnose why the search failed and to propose up to 2 new, *divergent* queries (alternative names, the underlying event, a parent entity). If the rewriter returns no queries (the call failed or its reply had none), the second pass searches what the claim says about its entities instead: the claim with the entity names and leading auxiliaries removed, or the claim itself if fewer than two content words remain; this finds pages on the property (rankings, lists, events) rather than the entity's page again, and needs no LLM call. Retrieval and scoring then run again, including the web search, which is retried if it failed on the first pass. Evidence found by the first round is kept (deduplicated by text) and scored again if the second search returns nothing, so an empty second search cannot discard it; when the second search returns snippets, those are scored instead. Verdicts that are already `SUPPORTED` or `CONTRADICTED` are not re-searched, even when the conformal set is ambiguous; they go straight to escalation.

**Absence evaluation.** If evidence is still missing after the rewrite, the absence of evidence is itself informative, but only for prominent facts: if the claim describes a major, widely recorded event (a world leader's death, a famous monument being moved) and nothing is found, it is almost certainly false; if it describes a minor detail, the evidence may simply not be online. The LLM is asked which case applies and answers `CONTRADICTED` or `UNKNOWN`. The probability vector passed to the conformal layer is $(1, 0, 0)$ or $(0, 1, 0)$ respectively.

**Escalation rule.** After scoring, a claim is fully escalated (Knowledge Graph §2.8, then LLM arbitration §2.9) when

$$
\text{escalate}(X) \iff \hat{y} = \text{UNKNOWN} \;\lor\; \text{escalate}_{\text{conf}}(X) \;\lor\; \left( \text{shared role} \;\land\; \hat{y} = \text{CONTRADICTED} \right)
$$

If it is not escalated but §2.5 flagged a lone or contested contradiction, it gets a **lightweight LLM review**: one arbitration call over the evidence already retrieved, without the Knowledge Graph evidence. Otherwise the NLI verdict stands.

Each result records the route that produced it in `final_decision_policy`:

| Value | Route |
| :--- | :--- |
| `conformal_singleton` | NLI verdict accepted (decisive, single-label conformal set) |
| `contradiction_arbitration` | lightweight LLM review of a lone or contested contradiction |
| `multi_agent_debate` | full escalation: Knowledge Graph (Wikidata) evidence + LLM arbitration |
| `absence_heuristic` | no evidence after the rewrite; absence evaluation decided |

**Worked example** (current benchmark run, FEVER claim *"Hedda Gabler's world premiere was on January 31st, 1891."*, true): on the first pass one Wikipedia snippet supported the claim ($`\hat{\pi}_{\text{sup}} = 0.994`$) and another contradicted it ($`\hat{\pi}_{\text{con}} = 0.989`$), a source conflict ($S, K \ge 0.85$, $\lvert S - K \rvert < 0.12$), so the verdict was `UNKNOWN` and the conformal set contained all three labels. A query rewrite was triggered (`["Hedda Gabler premiere date", "Henrik Ibsen Hedda Gabler 1891"]`), but the second pass found the same conflict, so the claim was escalated: its Wikidata statements were added (§2.8) and the LLM arbiter concluded *SUPPORTED* (§2.9), the correct verdict.

### 2.8 Knowledge Graph Evidence from Wikidata (`agents/wikidata_graph.py`)

**Purpose.** A claim is escalated because the retrieved passages left it ambiguous. Many claims turn on a fact that a knowledge graph records exactly and the NLI model reads poorly from prose: a date, a place, an award, or what kind of thing an entity is. The pipeline therefore takes the claim's facts from **Wikidata**, the structured knowledge graph behind Wikipedia, in which every fact is a typed statement (entity, property, value) with references, and gives them to the arbiter as extra evidence. It makes no LLM call.

**Input → Output.** The claim and the entities named by query generation (§2.2) → a graph $G = (V, E)$ whose nodes are Wikidata entities and literal values and whose edges are Wikidata statements; for an escalated claim, its statements as evidence lines.

**Entity resolution.** Each entity name is resolved to a Wikidata item by the English Wikipedia page of exactly that title (redirects followed). Otherwise one search is made, and a result is accepted only if its label shares a content word with the name and, for a qualified title such as *"Colombiana (2011 film)"*, its description shares a word with the qualifier. A name that does not resolve is left out rather than replaced by a near match, and disambiguation pages are rejected. The subject is the resolved entity named earliest in the claim.

**Relation matching.** The claim's relation is matched to one of 77 common properties (date of birth, place of birth, award received, occupation, country, manufacturer, chief executive officer, …). Let $\mathbf{r}$ be the embedding (§2.4) of the claim without its subject's name and $\mathbf{r}'$ that of the claim without any entity name, and let $T_p$ be the label and phrasings of property $p$. Its score is

$$
s(p) = \max_{\mathbf{x} \in \{\mathbf{r}, \mathbf{r}'\}} \; \max_{t \in T_p} \cos(\mathbf{x}, \mathbf{t}) \;+\; 0.12\,[\text{the claim states a date}]\,[p \text{ is a date}] \;-\; 0.08\,[\text{no date}]\,[p \text{ is a date}] \;+\; 0.10\,[p \text{ is recorded for an entity of the claim}]
$$

where $[\cdot]$ is 1 if the condition holds and 0 otherwise. The first reading keeps a telling object (*"won the Nobel Prize"*), and the second keeps an object name from swamping the relation (*"was built by the European Space Agency"*). The claim's property is $`\arg\max_p s(p)`$ if $s(p) \ge 0.45$; if the claim names a second entity, its value is that entity, so date and number properties are excluded. Some statements are stored on the other entity (Apple records its chief executive officer, not Tim Cook his title), so the property is looked up on whichever claim entity records it.

**Comparison with the claim.** The claimed value (a date or number stated in the claim, or another entity the claim itself names, of the right kind: a person for properties such as *architect* or *spouse*) is compared with the recorded values. The result is *agrees*, *differs* (only for dates, numbers and properties with one true value, such as a father), *not listed* (for places and other many-valued properties, where an unlisted value is not necessarily wrong; a place also agrees if any level of the location, such as the country, lists it), or *not identified*.

**Statements selected.** The recorded values of the claim's property (up to 3; of many, those closest to the claim), statements linking the claim's entities, what the subject is (*instance of*, up to 2 values), and up to four further statements most similar to the claim, at most one of them about another entity. Only entities connected to the subject contribute, so the graph stays one connected picture.

**Evidence for the arbiter.** For an escalated claim, each selected statement becomes one evidence line, such as `Wikidata Evidence [Albert Einstein]: Albert Einstein — date of birth: 14 March 1879 (24 references in Wikidata).`, after a note that Wikidata can be incomplete or out of date; the claim's own value is never included, since it is the claim, not evidence. The lookup starts as soon as query generation has named the entities, in a background thread alongside retrieval and NLI scoring, so it is usually ready by the time a claim escalates; the claim waits for it at most 20 s. If Wikidata cannot be reached, the arbiter decides on the retrieved evidence alone.

**Example** (current benchmark run). *"The Crips are a band."* was escalated, and the arbiter received *Crips — instance of: gang*, *inception: 1969*, *headquarters location: Los Angeles* and *country: United States*; it judged the claim `CONTRADICTED`. For *"Albert Einstein was born in 1950."*, they include the recorded date of birth, *14 March 1879*.

**Coverage and cost.** On the 600 benchmark claims, using the queries recorded in their runs, 553 resolved to a Wikidata graph, built in a median of 1.6 s (2.6 s at the 90th percentile). Wikidata recorded a value for the claim's property that could be compared with the claim for 108 claims (52 agree, 15 differ, 41 not listed); for the rest, no property matched what the claim states (252, typically opinions, events or descriptions), the property had no statement (85), or the claimed value could not be identified (108). Starting the lookup adds 0.24 ms to a claim; its only shared resource is the embedding model, which it holds for a median of 49 ms.

### 2.9 LLM Arbitration (`agents/critic_agent.py`)

**Purpose.** Resolve the claims the NLI model could not settle reliably: escalated claims and lone or contested contradictions. A large language model can handle paraphrase, implicit entailment, multi-sentence reasoning and dates, which the NLI classifier cannot.

**Input → Output.** Claim and full evidence text (retrieved snippets, the Wikidata statements of an escalated claim, the timestamp line and the source-conflict note) → final label.

**Dialectical mode (default; one LLM call).** A single prompt forces the model to write three sections before its verdict:

1. **Perspective A — Defense (maximal support):** the strongest plausible case that the evidence supports the claim, quoting the evidence.
2. **Perspective B — Prosecution (maximal skepticism):** ungrounded assumptions, missing facts, discrepancies and direct contradictions.
3. **Perspective C — Judicial arbitration:** weigh A against B strictly against the raw evidence text.

Writing both opposing cases before judging counters the tendency of an autoregressive model to anchor on whichever argument it generated first.

**Multi-agent mode (`debate_mode="multi_agent"`; three LLM calls).** A Defense agent and a Prosecution agent are run concurrently and never see each other's argument; a separate Judge agent then reads both arguments and the evidence (following multi-agent debate, Du et al., 2023; Liang et al., 2023). It is kept for ablation studies; the dialectical mode gives the same structure in one call.

**Decision rules in the prompts include:**

- **Natural entailment over hyper-literal pedantry:** e.g. the founder of a national company is from that country unless evidence says otherwise; naming one co-founder does not contradict another being a co-founder.
- **Awards:** a claimed award *win* is contradicted by evidence of only a *nomination*.
- **Scientific claims:** cautious wording (*"suggests"*, *"likely"*) counts as support if the underlying finding matches the claim.
- **False premises and misidentified entities:** a claim that refers to an entity by a wrong name, title, date or attribute is `CONTRADICTED`, and the claim's premise must never be silently corrected to fit the evidence. This rule balances the anti-pedantry rules, which on their own would let the arbiter "repair" a false premise by mapping a wrong name onto the entity the evidence describes. The rule is stated generically, with no benchmark claim in the prompt; `strict_premises=False` removes it for ablations.

**Conservative abstention.** The verdict is read from the last line of the response beginning with `VERDICT:`:

$$
\hat{y}_{\text{final}} = \begin{cases}
\text{SUPPORTED} & \text{if the evidence entails } c \text{ and no counter-evidence stands} \\\\
\text{CONTRADICTED} & \text{if the evidence decisively refutes } c \\\\
\text{UNKNOWN} & \text{otherwise (abstention)}
\end{cases}
$$

The prompt reserves `UNKNOWN` for evidence that is absent, unrelated, or genuinely balanced, so the arbiter does not abstain merely to avoid a decision. If the LLM call itself fails (quota exhausted, overload, timeout), the verdict is also `UNKNOWN`, but the result is flagged (`llm_unavailable`) so that the interface reports an outage rather than an abstention, and the run is not recorded for replay.

**Worked example** (current benchmark run, the Hedda Gabler claim of §2.7, abridged): *Perspective A:* "Evidence 1 explicitly provides direct, unambiguous textual support for this exact statement, stating: 'The world premiere was staged on 31 January 1891 at the Residenztheater in Munich.'" *Perspective B:* "Evidence 2 mentions subsequent performances in February and April of 1891, and the Wikidata evidence provides publication and categorical metadata without mentioning the premiere date; none of these sources contradict the explicit date." *Perspective C:* "the record shows direct and uncontroverted support for the claim." **VERDICT: SUPPORTED.** The Wikidata statement *publication date: 1890* (the printed play) did not mislead the arbiter about the premiere.

**Measured value** (current evaluation, §3.3). The computation trace records the NLI verdict just before arbitration (`UNKNOWN` for a source conflict or an inconclusive joint reading), so the arbiter's effect can be read off for every claim it decided:

| Benchmark | Claims decided by the LLM | Fixed (NLI wrong, final right) | Broken (NLI right, final wrong) | Both right | Both wrong |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FEVER | 71 (35 reviews, 36 escalations) | 18 | 7 | 42 | 4 |
| FreshQA | 116 (54 reviews, 62 escalations) | 42 | 8 | 49 | 17 |
| **Total** | **187** | **60** | **15** | **91** | **21** |

The arbiter corrected 60 verdicts and spoiled 15. If arbitration did not change accuracy, each of the 75 claims whose correctness changed would be equally likely to be a fix or a break, so the exact McNemar test (§4.5) gives

$$
p = 2 \sum_{i=0}^{15} \binom{75}{i} \, 2^{-75} = \frac{2 \times 2{,}999{,}267{,}652{,}451{,}776}{37{,}778{,}931{,}862{,}957{,}161{,}709{,}568} = 1.6 \times 10^{-7}
$$

and the improvement is significant. Per benchmark: FreshQA 42 against 8 ($p = 1.2 \times 10^{-6}$); FEVER 18 against 7 ($p = 0.043$), significant on its own but not after correction for the eight tests of §4.5.

### 2.10 Output, Report and User Interface

**Per-claim result.** The claim; the evidence text and source URLs; the verdict with its probability vector and confidence, and the temperature-scaled probability vector (§4.6); the conformal set, p-values, $`\hat{q}`$ and set size; the decision route (`final_decision_policy`); and, for an escalated claim, its Wikidata graph (§2.8), whose statements were added to its evidence.

**Report.** A summary of the whole input:

$$
\text{hallucination rate} = \frac{\text{number of claims labelled CONTRADICTED}}{\text{total number of claims}}
$$

together with the numbers of supported, contradicted and unknown claims, of claims whose conformal set triggered escalation, and of the other claims with a single-label conformal set (which include contradictions that the LLM then reviews, §2.7).

**Streaming UI.** The Flask server (`/api/analyze_stream`) streams newline-delimited JSON events: the list of extracted claims with the decomposition route (fast path, LLM, or NLTK fallback), one progress event per stage (tagged with the stage name and the claim it belongs to), each claim's result as soon as it is verified, and the final report, and after each claim's result its Wikidata graph (below). Claims are verified concurrently with the same per-claim code as the batch API, so the verdicts are identical. The web page (`static/`) has four tabs:

- **Verify** shows a live stepper of the pipeline stages, combined over the claims in progress (a stage is active while any claim is in it, with a count of how many claims have reached it; optional stages such as query rewrite, Knowledge Graph and arbiter are marked as skipped when no claim needs them), and then:
  - an **answer-level verdict**: one contradicted claim flags the whole answer as a hallucination; otherwise the answer is reported as supported, or as partly unverifiable when the detector abstained on some claims;
  - **From text to claims**: each sentence of the input next to the atomic claims it was decomposed into, coloured by verdict, with the words the decomposition added (resolved pronouns, repeated subjects) underlined. Claims are matched to sentences by content-word overlap in the browser, so no extra LLM call is made;
  - one card per claim, filled in as results arrive: the claim's **decision path** (NLI reading → conformal set → route → final verdict, noting when the LLM changed the NLI reading), the NLI posterior with its calibrated confidence, the conformal set and p-values, and every evidence snippet with its own NLI reading, the snippet that decided the verdict, and its source;
  - an **Under the hood** view of any claim's computation as it happens: the sentence the claim was split from and the decomposition route; the generated queries; the re-ranking table with cosine similarity, BM25, both ranks and the RRF formula; every snippet's NLI logits and the softmax; the decision-rule quantities S, K, |D_sup|, |D_con| and the branch taken; the conformal scores, q̂, p-values and set, with a **calibration plot** placing the claim's three non-conformity scores against the histogram of the 100 calibration scores and q̂ (`/api/calibration`); the routing condition term by term; and the LLM arbiter's written defense, prosecution and judgement. All of it comes from a per-claim trace that the detector records in `verification["trace"]` without changing any decision;
  - a timeline of the time each claim spent in each phase, and the **Wikidata graph** of all claims drawn with D3.js (below): each claim's dashed edge in its verdict colour next to the Wikidata statements for the same property, hovering shows an entity's description or a statement's references, and clicking opens it in Wikipedia or Wikidata; each claim card also states what Wikidata records for it.
- **Benchmarks** charts the fifteen-set results of §3.3 from `evaluation/*_5_runs_results.json` (`/api/benchmarks`), with Wilson 95% intervals and the two-proportion test of FEVER against FreshQA with its multiple-comparison correction (§4.5), including the accuracy of each decision route (§3.4), and has a **claim explorer** over all 600 benchmark claims (`/api/benchmark_claims/<name>`): they can be filtered by outcome (errors or correct), route and text, and each one opens with its evidence and the full computation trace recorded during the benchmark run, so every error in §3.5 can be inspected step by step.
- **Method** animates the cascade, with claims flowing into the three routes in the measured proportions, summarises the stages and lists the known limitations. It also replays the coverage argument of §2.6 on stored NLI scores (`/api/coverage_demo`): each trial draws a random split of the 200 gold-evidence FEVER claims of `experiments/eval_conformal.py`, calibrates $`\hat{q}`$ on 100 of them and drops a new claim's score among the sorted calibration scores. A histogram counts the rank $R$ at which it lands, and a chart tracks the running coverage with its Wilson interval. When the new claim comes from the same pool, the ranks are uniform and coverage converges to $`k/(n+1)`$; when it comes from the 600 benchmark claims scored on retrieved evidence, scores pile up beyond $`\hat{q}`$ and coverage settles near the 83.2% measured in §2.6.
- **Docs** renders this document (`/api/docs`).

**Knowledge graph view.** The interface draws each claim's Wikidata graph (§2.8), for every claim, not only escalated ones; the stream sends it after the claim's verdict. The graph shows the claim as a dashed edge in its verdict colour, the recorded values for that property, any statements linking the claim's entities, and up to four further statements most similar to the claim, at most one of them about an entity other than the subject. For *"Albert Einstein was born in 1950"* it shows the claimed *1950* next to the recorded date of birth *14 March 1879*; for *"Marie Curie's favourite colour was green"* no property matches, and the card says so.

For a claim that is not escalated, nothing that decides the verdict reads this graph. `VERIFACT_WIKIDATA=0` turns off the display. A "Deep Scan" option adds the Wikidata evidence to every claim, not only escalated ones; it appears in the Under-the-hood view.

**Server.** In production mode the server runs under waitress, bound to localhost; it validates input (at most 4,000 characters), limits live checks per client and in total to protect the LLM quota, reports its state at `/api/health`, adds standard security headers, and logs every request with an ID. Every event carries its time since the request started, and a completed live run that had no LLM failure is recorded; the same input is then replayed from the recording, with its original timestamps and clearly marked as a replay, unless a live run is requested. The example inputs' recordings (`demo_runs/`) do not expire, so the interface can be demonstrated without network access or API quota.

### 2.11 End-to-End Algorithm and Cost Model

The verification of one claim (`_verify_claim_full`), in pseudo-code:

```text
queries, domain, temporal ← GENERATE_QUERIES(c)                    # §2.2
pool ← ∅
for pass in {1, 2}:                                                 # at most one rewrite
    E ← RERANK(RETRIEVE(queries, c, domain), c, k = 3)              # §2.3–2.4
    E ← STRIP_ECHOES(E, c);  pool ← pool ∪ E;  if E = ∅: E ← pool
    if E = ∅:
        if pass = 1: queries ← REWRITE(c, "no evidence") or FALLBACK(c); continue  # §2.7
        y ← ABSENCE_EVAL(c);  return y                              # §2.7
    text ← JOIN(E);  if temporal: text ← "[SYSTEM TIMESTAMP]" + text
    (ŷ, π̂, review) ← SCORE(c, E, text)                              # §2.5
    C ← CONFORMAL_SET(π̂)                                            # §2.6
    if ŷ = UNKNOWN and pass = 1: queries ← REWRITE(c, text) or FALLBACK(c); continue
    if ŷ = UNKNOWN or |C| > 1 or UNKNOWN ∈ C or (shared role and ŷ = CONTRADICTED):
        text ← text + KG_EVIDENCE(c, queries, text)                 # §2.8
        ŷ ← ARBITRATE(c, text)                                      # §2.9
    else if review:
        ŷ ← ARBITRATE(c, text)                                      # §2.9, no KG
    return ŷ
```

**LLM calls per claim** (dialectical mode; the multi-agent mode uses 3 calls per arbitration instead of 1):

| Stage | LLM calls |
| :--- | :--- |
| Claim decomposition (per input, not per claim) | 0 (fast path) or 1 |
| Query generation | 0 (fast path) or 1 |
| Query rewrite | 0 or 1 |
| Absence evaluation | 0 or 1 |
| Lightweight review | 0 or 1 |
| Full escalation | arbitration 1 (the Wikidata evidence needs no LLM call) |

A claim settled on the fast path with fast-path queries needs **no LLM call at all**. Measured in the current evaluation (§3.4): 229 of 300 FEVER and 184 of 300 FreshQA claims were settled by the NLI verdict without any arbitration call, and a claim costs 1.05 LLM calls on average on FEVER and 1.45 on FreshQA: 0.86 when the NLI verdict stands (mostly query generation), 1.88 with a lightweight review, and 2.32 with full escalation.

### 2.12 Execution Details

- **LLM backend (`llm/`).** With `MODEL_PROVIDER=gemini`, two models are used, each serving as the other's fallback (larger Gemini models were dropped: their free-tier daily quota is too small to help). With the default `LLM_ROUTING=primary`, every call goes to `gemini-3.5-flash-lite` and falls back to `gemini-3.1-flash-lite` only when 3.5 is rate-limited, overloaded or out of daily quota: in the current evaluation, with 491 of 493 calls answered by 3.5, supporting calls took 1.3 s and arbitration calls 3.1 s on average (§3.4). Because 3.5 is faster, claims finish sooner and calls arrive closer together: during a latency measurement the Gemini console showed 3.5 at 18 requests in one minute against its limit of 15, so under sustained load the per-minute limit, not speed, is what sends calls to 3.1. Each call is also tagged by how demanding its task is, **heavy** for the steps that decide or preserve a verdict (arbitration, absence evaluation, claim decomposition) and **light** for supporting steps (query generation and rewriting); `LLM_ROUTING=tiered` sends heavy calls to 3.5 and light calls to 3.1, and `LLM_ROUTING=alternate` alternates between the two regardless of task. In every mode, a model that returns HTTP 429, 503 or 504, or times out, is skipped for 45 seconds and the request moves to the next model immediately; a model whose *daily* quota is used up is skipped until the quota resets (midnight Pacific time), so requests go straight to the models that can still answer. If both models fail at once with such a temporary error (a demand spike), the request waits and tries them again, after 2 s and then 5 s, before giving up; when both daily quotas are used up it fails at once, since waiting cannot help. Otherwise a local Ollama model (`qwen2.5:3b`) is used.
- **Concurrency.** Claims, queries and data sources are processed in thread pools, and all network calls run in parallel. The NLI model and the embedding models are each protected by a lock, because their native code is not safe to call from several threads at once; on Apple GPUs (MPS) the command buffer is synchronised before the NLI lock is released. Embedding models run on the CPU and NLI on the GPU when available (`NLI_DEVICE` overrides the device).
- **Warm-up.** When the server starts, the NLI model, both embedding models and the NLTK sentence tokenizer are each run once on a dummy input, so the first real request does not pay one-off initialisation (a cold NLI call took 1.18 s against 0.22 s warm). No cache is filled and no LLM is called.
- **Caching.** Wikipedia searches and parsed pages (LRU caches), paragraph embeddings (keyed by text), web search results (10 minutes, successful lookups only), NLI scores per claim–snippet pair, and the Wikidata entities and labels looked up for the graph are cached. Every cache returns exactly what recomputation would, so outputs are unchanged; the effect is speed, mainly on query-rewrite retries, which re-retrieve many of the same pages.

---

## 3. Evaluation

### 3.1 Benchmarks

**FEVER** (Fact Extraction and VERification; Thorne et al., 2018). Claims were written by annotators by altering sentences from Wikipedia, then labelled `SUPPORTS`, `REFUTES` or `NOT ENOUGH INFO` against Wikipedia. The detector is evaluated on the `SUPPORTS` and `REFUTES` claims of the validation split (`copenlu/fever_gold_evidence`). FEVER tests entity retrieval and NLI on encyclopedic facts without temporal change, and is the primary benchmark.

**FreshQA** (Vu et al., Google Research, 2023, continuously maintained). Questions whose answers change over time, plus false-premise questions, with answers maintained by the authors in a public sheet (cached in `evaluation/data/freshqa.csv`). Each question has a `fact_type`: *never-changing*, *slow-changing* or *fast-changing*. Because the detector verifies claims rather than answering questions, every question–answer pair is converted once into one declarative claim by an LLM (`evaluation/prepare_freshqa_cache.py`, cached in `evaluation/data/freshqa_claims.json`):

| FreshQA item | Claim | Label |
| :--- | :--- | :--- |
| Normal question + answer, e.g. *"Who is the CEO of Google?" – "Sundar Pichai"* | *"Sundar Pichai is the CEO of Google."* | `SUPPORTS` |
| False-premise question, e.g. *"When was League of Legends 2 released?"* | the false premise as a claim: *"League of Legends 2 was released."* | `REFUTES` |

FreshQA tests web retrieval, temporal reasoning and false-premise detection. Its gold answers are snapshots dated by the maintainers, so a fast-changing answer can be out of date when the benchmark is run. All FreshQA runs therefore sample only *never-changing* and *slow-changing* questions (`DEFAULT_FACT_TYPES` in `evaluation/benchmarks/freshqa_dataset.py`; 298 true and 147 false claims).

Benchmarks built from synthetic perturbations of a fixed reference text (such as HaluEval) are not used: their "false" claims are often true in the open world that this detector searches, which would penalise correct verdicts.

### 3.2 Protocol

1. **Stratified sampling.** $N$ claims are drawn at random, half `SUPPORTS` and half `REFUTES`, so that precision and recall of the hallucination class are both measurable. `--seed` makes the draw reproducible.
2. **Verification.** Each claim is passed to `detector.analyze(claim, extract_atomic_facts=False)` — benchmark claims are already atomic — with live retrieval and LLM calls. FreshQA claims also receive the original question as context.
3. **Aggregation.** If an input yields several claims, it is predicted `CONTRADICTED` if any claim is contradicted, otherwise `UNKNOWN` if any is unknown, otherwise `SUPPORTED` (one false fact makes an answer a hallucination). With `extract_atomic_facts=False` there is exactly one claim.
4. **Scoring.** The prediction is compared with the gold label (`SUPPORTS` → `SUPPORTED`, `REFUTES` → `CONTRADICTED`). `UNKNOWN` and runtime errors count as wrong. Accuracy, precision/recall/F1 of the `CONTRADICTED` class (§4) and latency per claim (mean, median, 90th percentile) are reported.

**Fifteen-set evaluation.** The reported results use `evaluation/run_5_runs.py`: each benchmark is split into 15 non-overlapping stratified sets of 20 claims (10 `SUPPORTS`, 10 `REFUTES`) drawn with seed 42, so every run evaluates the same 300 claims per benchmark. Results are reported per set and pooled. Sets 1–5 are the original 100-claim evaluation and are drawn exactly as before; sets 6–15 continue down the same shuffled lists and skip any claim text already drawn, because FEVER lists some claims several times (once per evidence row). Sets 1–5 predate that rule and contain three FEVER claims twice (in different sets), so the FEVER records cover 297 distinct claims. FreshQA has only 147 never- and slow-changing `REFUTES` claims, so its fifteenth set holds the last 7 of them and 13 `SUPPORTS` claims (once one label runs out, the set is completed with the other; `--no-fill` stops instead); its 300 claims are 153 true and 147 false. A claim whose verification hits an LLM infrastructure failure (exhausted quota, timeout) is retried once after a pause and the run stops if it fails again, so infrastructure failures are never scored as abstentions; per-claim results are saved as they finish (`evaluation/<dataset>_5_runs_claims.jsonl`), and an interrupted run resumes where it stopped.

```bash
python evaluation/run_5_runs.py --dataset all          # FEVER then FreshQA, 15 sets x 20 claims each
python evaluation/run_5_runs.py --dataset fever        # one benchmark
python evaluation/run_benchmark.py --dataset fever --samples 20                 # quick single run
python evaluation/run_benchmark.py --dataset freshqa --samples 20              # never- and slow-changing questions only
```

### 3.3 Results (Current Pipeline)

All figures below are measured with the current pipeline (the single configuration described in Section 2), with live retrieval and LLM calls, in three runs in October 2026: sets 1–5 (the first 100 claims of each benchmark), sets 6–10 and sets 11–15 (100 more each), each added later without re-running the earlier ones. Every LLM call went to `gemini-3.5-flash-lite` first; 747 of the 749 calls of the two benchmarks were answered by it, 2 arbitration calls fell back to `gemini-3.1-flash-lite`, and no call failed. Each benchmark was run with a full record of every claim (verdict, computation trace, evidence and LLM calls), and `experiments/full_evaluation.py` derives all the results of this section and of §2.5, §2.6 and §2.9 from those records. In the first run, the 37 claims that reached full escalation in an earlier run were re-run end to end after the Knowledge Graph evidence moved to Wikidata (§2.8); its other 163 claims never reach that step, so the current code decides them exactly as recorded. Retrieval settings of the first run: the web search used two engines with a 3.0 s cap (web results for 125 of its 200 claims), and the second pass had no fallback for an empty query rewrite. The second and third runs used the settings of §2.3 and §2.7.

**FEVER: fifteen-set evaluation.** 15 non-overlapping stratified sets of 20 claims (10 `SUPPORTS`, 10 `REFUTES`), seed 42, 300 claims in total (summary in `evaluation/fever_5_runs_results.json`, per-claim records in `evaluation/fever_5_runs_claims.jsonl`):

| Set | Claims | Correct | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | `UNKNOWN` | Mean latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Set 1 | 20 | 19 | 95.00% | 1.00 | 1.00 | 1.00 | 10 | 0 | 0 | 9 | 1 | 9.0 s |
| Set 2 | 20 | 19 | 95.00% | 0.91 | 1.00 | 0.95 | 10 | 1 | 0 | 9 | 0 | 8.6 s |
| Set 3 | 20 | 18 | 90.00% | 0.90 | 0.90 | 0.90 | 9 | 1 | 1 | 9 | 0 | 8.8 s |
| Set 4 | 20 | 19 | 95.00% | 1.00 | 0.90 | 0.95 | 9 | 0 | 1 | 10 | 1 | 9.9 s |
| Set 5 | 20 | 20 | 100.00% | 1.00 | 1.00 | 1.00 | 10 | 0 | 0 | 10 | 0 | 6.9 s |
| Set 6 | 20 | 17 | 85.00% | 0.89 | 0.80 | 0.84 | 8 | 1 | 2 | 9 | 0 | 8.5 s |
| Set 7 | 20 | 17 | 85.00% | 0.89 | 0.80 | 0.84 | 8 | 1 | 2 | 9 | 0 | 8.0 s |
| Set 8 | 20 | 18 | 90.00% | 1.00 | 0.80 | 0.89 | 8 | 0 | 2 | 10 | 0 | 9.2 s |
| Set 9 | 20 | 17 | 85.00% | 0.90 | 0.90 | 0.90 | 9 | 1 | 1 | 8 | 1 | 8.0 s |
| Set 10 | 20 | 18 | 90.00% | 1.00 | 0.80 | 0.89 | 8 | 0 | 2 | 10 | 0 | 8.8 s |
| Set 11 | 20 | 18 | 90.00% | 1.00 | 0.80 | 0.89 | 8 | 0 | 2 | 10 | 0 | 7.3 s |
| Set 12 | 20 | 18 | 90.00% | 0.83 | 1.00 | 0.91 | 10 | 2 | 0 | 8 | 0 | 8.1 s |
| Set 13 | 20 | 18 | 90.00% | 0.90 | 0.90 | 0.90 | 9 | 1 | 1 | 9 | 0 | 8.6 s |
| Set 14 | 20 | 18 | 90.00% | 1.00 | 0.90 | 0.95 | 9 | 0 | 1 | 9 | 1 | 9.1 s |
| Set 15 | 20 | 18 | 90.00% | 0.90 | 0.90 | 0.90 | 9 | 1 | 1 | 9 | 0 | 9.1 s |
| **Pooled** | **300** | **272** | **90.67%** | **0.94** | **0.89** | **0.91** | 134 | 9 | 16 | 138 | 4 | 8.5 s |

Pooled Wilson 95% CI (§4.4): [86.8%, 93.5%]. With 20 claims per set, one error moves a set's accuracy by 5 points, so the pooled figure is the one to quote. The three runs scored 95, 87 and 90 of 100 (sets 1–5, 6–10 and 11–15); a chi-square test of equal accuracy across the runs gives $\chi^2 = 3.86$ with 2 degrees of freedom, $p = 0.15$ (§4.5), so the runs are consistent with one underlying accuracy.

**FreshQA: fifteen-set evaluation.** Same protocol on never- and slow-changing questions (§3.1), 300 claims (summary in `evaluation/freshqa_5_runs_results.json`, per-claim records in `evaluation/freshqa_5_runs_claims.jsonl`):

| Set | Claims | Correct | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | `UNKNOWN` | Mean latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Set 1 | 20 | 13 | 65.00% | 0.88 | 0.70 | 0.78 | 7 | 1 | 3 | 6 | 3 | 14.1 s |
| Set 2 | 20 | 17 | 85.00% | 0.90 | 0.90 | 0.90 | 9 | 1 | 1 | 8 | 1 | 12.5 s |
| Set 3 | 20 | 17 | 85.00% | 1.00 | 0.70 | 0.82 | 7 | 0 | 3 | 10 | 1 | 12.9 s |
| Set 4 | 20 | 16 | 80.00% | 0.89 | 0.80 | 0.84 | 8 | 1 | 2 | 8 | 2 | 12.1 s |
| Set 5 | 20 | 19 | 95.00% | 0.91 | 1.00 | 0.95 | 10 | 1 | 0 | 9 | 0 | 9.7 s |
| Set 6 | 20 | 17 | 85.00% | 0.82 | 0.90 | 0.86 | 9 | 2 | 1 | 8 | 0 | 10.2 s |
| Set 7 | 20 | 16 | 80.00% | 0.89 | 0.80 | 0.84 | 8 | 1 | 2 | 8 | 2 | 10.5 s |
| Set 8 | 20 | 18 | 90.00% | 1.00 | 0.80 | 0.89 | 8 | 0 | 2 | 10 | 0 | 9.2 s |
| Set 9 | 20 | 17 | 85.00% | 0.90 | 0.90 | 0.90 | 9 | 1 | 1 | 8 | 1 | 12.1 s |
| Set 10 | 20 | 17 | 85.00% | 0.90 | 0.90 | 0.90 | 9 | 1 | 1 | 8 | 1 | 10.7 s |
| Set 11 | 20 | 17 | 85.00% | 1.00 | 0.70 | 0.82 | 7 | 0 | 3 | 10 | 0 | 10.3 s |
| Set 12 | 20 | 14 | 70.00% | 0.82 | 0.90 | 0.86 | 9 | 2 | 1 | 5 | 3 | 12.8 s |
| Set 13 | 20 | 15 | 75.00% | 1.00 | 0.50 | 0.67 | 5 | 0 | 5 | 10 | 0 | 10.2 s |
| Set 14 | 20 | 16 | 80.00% | 1.00 | 0.60 | 0.75 | 6 | 0 | 4 | 10 | 1 | 11.5 s |
| Set 15 (13 true, 7 false) | 20 | 18 | 90.00% | 0.86 | 0.86 | 0.86 | 6 | 1 | 1 | 12 | 0 | 11.2 s |
| **Pooled** | **300** | **247** | **82.33%** | **0.91** | **0.80** | **0.85** | 117 | 12 | 30 | 130 | 15 | 11.3 s |

Pooled Wilson 95% CI (§4.4): [77.6%, 86.2%]. The three runs scored 82, 85 and 80 of 100 ($\chi^2 = 0.87$, 2 degrees of freedom, $p = 0.65$). Precision, recall and F1 refer to the `CONTRADICTED` (hallucination) class (definitions in §4.1); `UNKNOWN` counts as an error, and on a false claim as a missed hallucination (FN).

**FEVER vs FreshQA.** FreshQA is significantly harder. The difference in accuracy is 8.3 points (Newcombe 95% interval [2.9, 13.8] points), and a two-proportion z-test of equal accuracy gives $z = 2.99$, $p = 0.003$ (Fisher's exact test: $p = 0.004$; §4.5). The difference survives correction for all eight tests of this document, both the family-wise correction (Holm-adjusted $p = 0.015$) and the false-discovery-rate correction (Benjamini–Hochberg adjusted $p = 0.006$). Precision is close (0.94 vs 0.91); the gap is in recall (0.89 vs 0.80) and in abstentions (4 vs 15): FreshQA's false premises and claims about the present are more often accepted by the NLI model or left `UNKNOWN` by the arbiter when the evidence does not settle them, and some of its gold answers are contradicted by current evidence (§3.5).

### 3.4 Latency, Routing and Cost (Current Pipeline)

| Benchmark | Mean latency | Median | 90th percentile | LLM calls per claim |
| :--- | :--- | :--- | :--- | :--- |
| FEVER (300 claims) | 8.5 s | 8.0 s | 12.1 s | 1.05 |
| FreshQA (300 claims) | 11.3 s | 9.9 s | 19.5 s | 1.45 |

| Benchmark | NLI verdict stands | Lightweight review | Full escalation |
| :--- | :--- | :--- | :--- |
| FEVER (300 claims) | 229 | 35 | 36 |
| FreshQA (300 claims) | 184 | 54 | 62 |

Accuracy by route, the evidence that the conformal layer sends the right claims to the LLM: the claims settled by the NLI verdict alone are correct 89% of the time across both benchmarks (368 of 413), and the claims it passed on, which are the harder ones, are correct 81% of the time after LLM review (151 of 187).

| Benchmark | NLI verdict stands | Lightweight review | Full escalation |
| :--- | :--- | :--- | :--- |
| FEVER | 212 / 229 (93%) | 30 / 35 (86%) | 30 / 36 (83%) |
| FreshQA | 156 / 184 (85%) | 47 / 54 (87%) | 44 / 62 (71%) |

LLM calls by route (both benchmarks): 0.86 per claim when the NLI verdict stands, 1.88 with a lightweight review, 2.32 with full escalation. Every call was logged; all but 2 were answered by `gemini-3.5-flash-lite`:

| Task tier | Calls | Mean | Median |
| :--- | :--- | :--- | :--- |
| light: query generation and rewriting | 562 | 1.3 s | 1.1 s |
| heavy: arbitration, absence evaluation | 185 | 2.9 s | 2.7 s |
| heavy, fallback to `gemini-3.1-flash-lite` | 2 | 20.0 s | — |

Latency is measured end to end with live Wikipedia, web and Gemini calls on an Apple M1 laptop. Claims settled by the NLI verdict are fastest; fully escalated claims, which add the Knowledge Graph evidence and arbitration, account for the long tail.

### 3.5 Error Analysis

**FEVER (current pipeline, fifteen sets, 300 claims).** The 28 errors are 9 false accusations, 15 missed hallucinations and 4 abstentions:

1. **NLI fast-path errors (17).** Settled without LLM review. Four true claims were settled `CONTRADICTED` by two or more agreeing snippets, which met the corroboration rule: *"Diana, Princess of Wales was renowned as Lady Diana Spencer"*, contradicted by snippets giving her full name, Diana Frances Spencer; *"Warner Bros. released Avenged Sevenfold"*, contradicted by snippets about the band's releases on other labels; *"The album Cher was created by Cher"*; and *"Taran Killam is an American writer"*, contradicted by snippets describing him as an actor and comedian. Six false claims were settled `SUPPORTED` by a decisive supporting snippet about a related fact: *"Middle-earth is a real place"* (snippets on the geography of Tolkien's invented world), *"Stan Beeman is a fictional setting"* (he is a fictional character), *"Saturn is the second-largest moon in the Solar System"* (a snippet stating this of Titan, Saturn's largest moon), *"Chile is a food"* (a snippet on Chilean dishes), *"Aristotle spent almost 40 years at the Academy"* and *"I Kissed a Girl is part of Katy Perry's first studio album"*, each of which differs from the evidence in one number, title or role. A seventh, *"The Chrysler Building was the world's tallest building for 8 months"*, was settled `SUPPORTED` by the joint fallback (§2.5), although neither of its snippets supported it. From sets 11–15: three true claims were settled `CONTRADICTED` by two or more agreeing snippets: *"English people are descended from the Angles as a result of migration"*, contradicted after a query rewrite by snippets that trace English ancestry to migration from the northern Netherlands, Germany and southern Scandinavia, and to the Anglo-Saxon migration, without naming the Angles; *"Tool is a band"*, whose query `["Tool"]` retrieved only the Wikipedia article on tools as implements; and *"Viola Davis appeared in the novel Law & Order: Special Victims Unit"*, contradicted by snippets on her guest role in the television series. Three false claims were settled `SUPPORTED`: *"Saturn is larger than Jupiter"*, by a snippet stating that Saturn is "the second largest planet in the Solar System, after Jupiter", the reverse of the claim; and *"Mary of Teck's son abdicated the throne in 1936"* and *"Hourglass was James Taylor's first album in over half a decade"*, whose snippets state the claims as written (her eldest son, Edward VIII, became king and abdicated in 1936; *Hourglass* was "his first studio album in six years"), so in these two the detector followed evidence that agrees with the claim rather than with the gold label.
2. **Arbiter errors (7).** *"Southpaw was the fighting style Mike Tyson used"* (false): the evidence said Tyson sometimes switched to a southpaw stance, and the arbiter accepted that as support although his style was orthodox. *"Peking University is in a unitary sovereign state in East Asia"* and *"The Good Wife is a movie"* (both false) were judged `SUPPORTED`, the first by reading the description of China as entailed, the second because the evidence also named a 1987 film of the same title. *"The Washington Wizards were the winners of four conference titles"* (true) was judged `CONTRADICTED`, although the evidence did not state the number of titles. From sets 11–15: *"Victoria Palace Theatre is on the same side of Victoria Station"* and *"Heavy Metal music was developed in Europe"* (both false) were judged `SUPPORTED`, the first inferred from the evidence placing the theatre in Victoria Street, a street adjoining the station, although no snippet says which side it is on, the second from the evidence that the genre developed "largely in the United Kingdom and United States", which the arbiter took as satisfying *Europe*. *"Ashley Cole plays Major League Soccer for the Los Angeles Galaxy"* (true) was judged `CONTRADICTED`: the evidence calls him the former LA Galaxy captain who has since signed for Derby County, and the arbiter read the present-tense *plays* as contradicted.
3. **Abstentions (4).** *"Nuuk is the largest military center of Greenland"* (false): the evidence described Nuuk's civilian institutions only, and the arbiter, finding no statement either way, abstained. *"Brian Michael Bendis began teaching English at the University of Oregon in fall of 2013"* (true): the evidence and Wikidata confirm that he began teaching at the University of Oregon in fall 2013, but describe courses on graphic novels, so the arbiter treated the subject *English* as unverified. *"Matthias Corvinus, King of Hungary, established a royal theater troupe between 1458 and 1490"* (true): the evidence confirmed his reign and patronage of the arts but did not mention a theater troupe. From sets 11–15: *"The Gifted is from the United States"* (true): the retrieved evidence was about intellectual giftedness and US gifted-education law, and the Wikidata statements described only the United States, so the arbiter found nothing about an entity called *The Gifted* and abstained.

**FreshQA (current pipeline, fifteen sets, 300 claims, never- and slow-changing).**

| Fact type | Claims | Correct | Wrong | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| never-changing | 155 | 130 | 25 | 83.9% |
| slow-changing | 145 | 117 | 28 | 80.7% |

The 53 errors fall into four groups:

1. **Abstentions (15: 11 true claims, 4 false).** For fourteen, the evidence covered the topic but not the specific fact, and the arbiter returned `UNKNOWN` rather than guess: the number of episodes of *The Rings of Power* season 1, Biden's most-visited country, the year of both Benedict XVI's and Elizabeth II's deaths and the UMass NLP faculty count (all true); and *"Humans have been killed by dinosaurs"* and *"Taylor Swift has been a UNICEF Goodwill Ambassador"* (both false); in sets 6–10, the number of Tesla vehicle models, the cause of death of the Perceptron's inventor, whether *The Boy and the Heron* is the most recent Studio Ghibli film (all true) and an Alpha Kappa Delta Phi chapter at American University (false); and, in sets 11–15, the number of American companies to have exceeded a $4 trillion market capitalization (the evidence follows Apple only to $3 trillion), the United States' place in the FIBA Men's World Ranking (the evidence described how the ranking is computed, and the WBSC baseball ranking, but gave no standings), the most-starred GitHub repository (all true) and the number of living former New Zealand prime ministers (false; the evidence described lists of them but gave no count). For the fifteenth, *"The Japanese school year starts in April"* (true), the web search did state the fact, but the claim-echo filter removed both snippets as near-quotes of this short claim (§2.3), leaving only a paragraph about school terms.
2. **NLI fast-path errors (24).** Settled without LLM review. From sets 1–5: a negation (*"The Great Pyramid of Giza is not one of the New 7 Wonders"*, true, settled `CONTRADICTED`), two claims about the present matched to past evidence (*"Pele is currently staying in a hospital to receive cancer treatment"*, from 2022 evidence, and *"Walt Disney Animation Studios' 60th animated film is being released this year"*, which is *Encanto*, 2021), and two near-matches (*"Donald Trump tweeted in 2022 after his unbanning"*, supported by evidence about his reinstatement, not a tweet; *"Novak Djokovic lost in a specific round at the 2022 Australian Open"*, a false premise, since he did not play it, supported by a snippet about his first-round loss at a different tournament). From sets 6–10: five false premises or claims about the present accepted from related evidence (*"Jeff Bezos sold Amazon"*, *"Canada currently requires arriving travelers to provide proof of COVID vaccination via an app"*, *"X Corp. has a CEO"*, *"Microsoft has scheduled a date in the future for the release of security updates for Windows 7"*, *"There are days left until the next season of Atlanta airs"*), and three true claims contradicted by two or more agreeing snippets (*"Sunderland, Massachusetts is located to the north of Amherst"*, *"The Democratic Republic of the Congo was the first country to detect a case of mpox in the most recent outbreak"* and *"Qianlong was the emperor of China when the American Revolutionary War ended"*), which turn on geography, on dates, or on which outbreak is the most recent. From sets 11–15, eleven false claims were settled `SUPPORTED`: two claims about the present matched to past evidence (*"Yugoslavia currently has members of parliament"*, from snippets on the assemblies of the Kingdom and the Socialist Federal Republic, and *"There are days left until the 2022 FIFA World Cup begins"*, which was not recognised as temporal and was settled by the joint fallback (§2.5) from evidence dating the tournament to November–December 2022); five near-matches or false premises (*"A city hosted the Summer Olympics in 1940"*, from snippets on the Games awarded to Tokyo and then Helsinki, which state that they were cancelled; *"Drake topped Spotify's list of most-streamed artists for three consecutive years"*, from a snippet saying he topped it three times, in 2015–2016 and 2018; *"Hikaru Nakamura became the World Chess Champion"*, from snippets on his 2022 Fischer Random world title; *"Andrew Johnson was elected as President of the United States"*, from a snippet saying that he assumed the presidency after Lincoln's assassination; and *"The M1 MacBook Air has a dedicated graphics processing unit"*, settled by the joint fallback from evidence stating that it has no discrete graphics card but an integrated GPU); *"Taylor Swift has won a Golden Globe award"*, settled by the joint fallback although none of its snippets, which list her other awards, supported it; and three claims supported by web pages that assert them (*"Elon Musk acquired Instagram"*, from two pages reporting the purchase, one dated 1 April 2025; *"A FIFA Women's World Cup will take place in 2026"*, from a page announcing host cities for a 2026 tournament and two snippets on the bidding timeline for the 2031 and 2035 tournaments, whose dates fall in 2026; and *"Daddy Yankee has a next studio album coming up"*, from a page announcing an album for 30 October 2026, although another snippet, on rumours of a new album, added that the story seems to be false).
3. **Gold labels contradicted by current evidence (11).** The 2026 evidence states that Lewandowski left Barcelona at the end of the 2025–26 season (labelled as still playing there), that the LA Metro has 110 rail stations (labelled: 107), that a few species have landed on the Moon (labelled false), and that the 2027 Women's World Cup is to be held next year (*"A FIFA World Cup will be hosted next year"*, labelled false); *"Joe Biden announced his second presidential campaign in 2023"* was generated from a false-premise question but is true as stated. In sets 6–10, the evidence records Alphabet's market value at a record high in September 2025 (labelled false), gives Colleen Hoover's book count as 25 (labelled: 27) and names five asteroids and one suspected artificial satellite as the only objects detected before impact (labelled: 11 asteroids). In sets 11–15, the evidence states that 27 February 2026 marked the thirtieth anniversary of the Pokémon franchise (labelled as not yet celebrated) and that Josh Kerr set the current mile record of 3:42.66 on 18 July 2026 (labelled: 3:43.13), and it lists three public health emergencies of international concern declared in the five years before the October 2026 timestamp, the 2022 and 2024 mpox outbreaks and the 2026 Ituri Province Ebola epidemic (labelled: two). In these cases the detector followed the evidence, and even slow-changing labels are drifting out of date.
4. **Arbiter errors (3).** *"Princeton University is the top-ranked university in the US according to the US News Ranking"* (true): the arbiter read the evidence's dated ranking information as contradicting the present-tense claim. In sets 11–15, *"A natural wonder in the United States is part of the New 7 Wonders of Nature"* (false) was judged `SUPPORTED`: the evidence lists the seven New7Wonders winners, none in the United States, but the arbiter counted the Grand Canyon from the separate Seven Natural Wonders list, which the evidence says was created in response to the New 7 Wonders campaign. *"The largest acquisition deal of a tech company in history was Microsoft's purchase of Activision Blizzard for US$68.7 billion"* (true) was judged `CONTRADICTED`, although the evidence calls it the largest video game acquisition in history and says nothing of other tech acquisitions.

---

## 4. Evaluation Metrics

### 4.1 Confusion Matrix

The **positive class is a hallucination**: a claim whose gold label is `REFUTES`, which the detector should mark `CONTRADICTED`.

| | Predicted `CONTRADICTED` | Predicted `SUPPORTED` | Predicted `UNKNOWN` |
| :--- | :--- | :--- | :--- |
| **Claim is false** (gold `REFUTES`) | **TP** — hallucination caught | **FN** — hallucination missed | **FN** — hallucination missed |
| **Claim is true** (gold `SUPPORTS`) | **FP** — true claim wrongly accused | **TN** — true claim passed | not TP/FP/TN; counts as an accuracy error |

- **True Positive (TP):** the claim is false and the detector flagged it `CONTRADICTED`.
- **False Positive (FP):** the claim is true but the detector flagged it `CONTRADICTED` — a false accusation.
- **False Negative (FN):** the claim is false but the detector said `SUPPORTED` or `UNKNOWN` — a missed hallucination.
- **True Negative (TN):** the claim is true and the detector said `SUPPORTED`.

### 4.2 Metrics

With $N$ evaluated claims:

$$
\text{Accuracy} = \frac{\text{number of claims whose prediction equals the gold label}}{N}
$$

For a binary output without abstentions this equals $`(TP + TN)/N`$; with abstentions, a true claim predicted `UNKNOWN` is counted in $N$ but in none of TP, FP, TN.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

*When the detector says "hallucination", how often is it right?* High precision means few false accusations.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

*Of all hallucinations, what fraction does the detector catch?*

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,TP}{2\,TP + FP + FN}
$$

the harmonic mean of precision and recall, which is high only when both are.

**Worked example** (current pipeline, FEVER fifteen-set evaluation, 300 claims: 150 false and 150 true): 134 false claims were flagged `CONTRADICTED`, 15 were judged `SUPPORTED` and 1 ended `UNKNOWN`; 138 true claims were judged `SUPPORTED`, 9 were flagged `CONTRADICTED` and 3 ended `UNKNOWN`. So $TP = 134$, $FN = 16$, $FP = 9$, $TN = 138$, and the true claims left `UNKNOWN` count only as accuracy errors:

$$
\text{Accuracy} = \frac{134 + 138}{300} = 90.7\\%, \quad \text{Precision} = \frac{134}{134 + 9} = 0.94, \quad \text{Recall} = \frac{134}{134 + 16} = 0.89, \quad F_1 = \frac{2 \times 134}{2 \times 134 + 9 + 16} = 0.91
$$

### 4.3 Abstentions and Selective Classification

The benchmarks are binary, so every `UNKNOWN` counts as an error in accuracy and, for a false claim, as a missed hallucination. A detector that may abstain is a **selective classifier** $(f, g)$: a predictor $f$ and a gate $`g(X) \in \{0, 1\}`$ that is 0 when the detector abstains (El-Yaniv & Wiener, 2010). Two quantities separate the effect of abstaining from the quality of the decisions actually made:

$$
\text{Coverage: } \Phi(g) = \mathbb{E}\left[ g(X) \right] \qquad \text{Selective risk: } R(f, g) = \frac{\mathbb{E}\left[ \mathbb{I}\left( f(X) \ne Y \right) \, g(X) \right]}{\Phi(g)}
$$

Coverage is the fraction of claims that receive a verdict; selective risk is the error rate among those claims. Abstaining on genuinely ambiguous evidence lowers coverage (and therefore benchmark accuracy) but keeps the selective risk — the error rate of the verdicts the user actually sees — low.

### 4.4 Uncertainty of a Measured Accuracy

An accuracy $\hat{p}$ measured on $N$ claims is an estimate of a binomial proportion. Its standard error and normal-approximation 95% confidence interval are

$$
\text{SE} = \sqrt{\frac{\hat{p}(1 - \hat{p})}{N}}, \qquad \hat{p} \pm 1.96 \cdot \text{SE}
$$

For example, $`\hat{p} = 0.80`$ on $N = 100$ gives $`\text{SE} = 4.0\%`$ and the interval $`[72.2\%, 87.8\%]`$; on $N = 40$ the half-width grows to about $`12.4\%`$. Differences of a few points between runs of 20–40 claims are therefore within noise, and comparisons between pipeline versions should use the same claims and larger samples.

Near 0% or 100% the normal approximation is poor: it can exceed 100%, and its actual coverage falls below the nominal 95%. Every interval in this document, for accuracies (§3.3) and for conformal coverage (§2.6), is therefore the **Wilson score interval** (Wilson, 1927), the interval recommended by Brown, Cai and DasGupta (2001).

*Derivation.* The normal approximation evaluates the standard error at the estimate $\hat{p}$. The score test of $`H_0: p = p_0`$ evaluates it at the hypothesised value instead: by the central limit theorem, if the true accuracy is $`p_0`$, then

$$
Z(p_0) = \frac{\hat{p} - p_0}{\sqrt{p_0 (1 - p_0) / N}} \approx \mathcal{N}(0, 1)
$$

The 95% interval is the set of values $`p_0`$ that this test does not reject at the 5% level, $`\lvert Z(p_0) \rvert \le z`$ with $z = 1.96$. Squaring both sides gives a quadratic inequality in $`p_0`$:

$$
(\hat{p} - p_0)^2 \le \frac{z^2}{N} \, p_0 (1 - p_0) \iff Q(p_0) = \left( 1 + \frac{z^2}{N} \right) p_0^2 - \left( 2 \hat{p} + \frac{z^2}{N} \right) p_0 + \hat{p}^2 \le 0
$$

Its leading coefficient is positive, so the solutions are the values between the two roots of $Q$, which the quadratic formula gives as

$$
\frac{\hat{p} + \dfrac{z^2}{2N} \pm z \sqrt{\dfrac{\hat{p}(1 - \hat{p})}{N} + \dfrac{z^2}{4N^2}}}{1 + \dfrac{z^2}{N}}
$$

*Properties.* (i) The interval always lies in $[0, 1]$: $`Q(0) = \hat{p}^2 \ge 0`$, $`Q(1) = (1 - \hat{p})^2 \ge 0`$, and the minimum of $Q$ lies at $`\big( 2\hat{p} + z^2/N \big) / \big( 2 (1 + z^2/N) \big) \in [0, 1]`$, so both roots lie in $[0, 1]$. (ii) Its centre is a weighted average of $\hat{p}$ and $\tfrac{1}{2}$,

$$
\frac{\hat{p} + z^2 / (2N)}{1 + z^2 / N} = \frac{N}{N + z^2} \, \hat{p} + \frac{z^2}{N + z^2} \cdot \frac{1}{2}
$$

so the interval is pulled towards $\tfrac{1}{2}$, more strongly for small $N$, and is asymmetric around $\hat{p}$ near 0 or 1. (iii) It never has zero width: at $\hat{p} = 1$ the roots are $`N/(N + z^2)`$ and 1. The normal approximation gives zero width there; for FEVER's Set 5 (20 of 20 correct, §3.3) it gives $[100\%, 100\%]$, while the Wilson interval is $[83.9\%, 100\%]$.

For FEVER's pooled 272 correct verdicts out of 300, the centre is $(0.9067 + 0.0064)/1.0128 = 0.9015$ and the half-width $`1.96 \sqrt{0.000282 + 0.000011} / 1.0128 = 0.0331`$, so the interval is [86.8%, 93.5%], where the normal approximation gives [87.4%, 94.0%]. Near 1 the difference is larger: for the 95 of 100 of FEVER's sets 1–5, the Wilson interval is [88.8%, 97.8%] and the normal approximation [90.7%, 99.3%].

### 4.5 Hypothesis Tests

The tests below are computed by `experiments/statistical_tests.py` from the per-claim records; it makes no LLM call.

**Two independent accuracies** (FEVER vs FreshQA, which share no claims). With $`c_1`$ of $`n_1`$ and $`c_2`$ of $`n_2`$ correct, and the pooled proportion $`\bar{p} = (c_1 + c_2)/(n_1 + n_2)`$, the two-proportion z-test of equal accuracy is

$$
z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\bar{p} (1 - \bar{p}) \left( \dfrac{1}{n_1} + \dfrac{1}{n_2} \right)}}, \qquad p = 2 \left( 1 - \Phi(\lvert z \rvert) \right)
$$

*Derivation.* The two accuracies are independent binomial proportions, so $`\mathrm{Var}(\hat{p}_1 - \hat{p}_2) = p_1 (1 - p_1)/n_1 + p_2 (1 - p_2)/n_2`$. Under $`H_0: p_1 = p_2 = p`$ this is $`p (1 - p)(1/n_1 + 1/n_2)`$, and the common $p$ is estimated by $\bar{p}$, its maximum-likelihood estimate under $`H_0`$. By the central limit theorem $z$ is then approximately standard normal under $`H_0`$.

**Fisher's exact test** (Fisher, 1922) avoids the normal approximation. Under $`H_0`$, with $`N = n_1 + n_2`$, the probability of $a$ and $b$ correct verdicts is $`\binom{n_1}{a} \binom{n_2}{b} \, p^{a+b} (1 - p)^{N - a - b}`$, and the total $T = C_1 + C_2$ is $`\text{Binomial}(N, p)`$. Conditioning on the observed total $t$, the unknown $p$ cancels:

$$
\Pr(C_1 = a \mid T = t) = \frac{\binom{n_1}{a} \binom{n_2}{t - a} \, p^t (1 - p)^{N - t}}{\binom{N}{t} \, p^t (1 - p)^{N - t}} = \frac{\binom{n_1}{a} \binom{n_2}{t - a}}{\binom{N}{t}}
$$

a hypergeometric distribution that is fully known under $`H_0`$. The two-sided p-value is the total probability of the tables with the same margins that are at most as probable as the observed one.

**The interval for the difference** $`\hat{\delta} = \hat{p}_1 - \hat{p}_2`$ is Newcombe's hybrid score interval (Newcombe, 1998), built from the two Wilson intervals $`[l_i, u_i]`$:

$$
\left[ \hat{\delta} - \sqrt{(\hat{p}_1 - l_1)^2 + (u_2 - \hat{p}_2)^2}, \;\; \hat{\delta} + \sqrt{(u_1 - \hat{p}_1)^2 + (\hat{p}_2 - l_2)^2} \right]
$$

**Several independent accuracies** (the three runs of a benchmark, §3.3). With $`c_j`$ of $`n_j`$ correct in run $j = 1, \dots, k$ and the pooled accuracy $`\bar{p} = \sum_j c_j / \sum_j n_j`$, Pearson's chi-square statistic compares the observed right and wrong counts with those expected if every run had accuracy $\bar{p}$:

$$
X^2 = \sum_{j=1}^{k} \left[ \frac{(c_j - n_j \bar{p})^2}{n_j \bar{p}} + \frac{\big( (n_j - c_j) - n_j (1 - \bar{p}) \big)^2}{n_j (1 - \bar{p})} \right] = \sum_{j=1}^{k} \frac{(c_j - n_j \bar{p})^2}{n_j \bar{p} (1 - \bar{p})}
$$

Each term is the squared z-score of run $j$ against the pooled accuracy. Under $`H_0`$ (equal accuracies) $X^2$ is approximately chi-square distributed with $k - 1$ degrees of freedom: one is lost because $\bar{p}$ is estimated from the same counts. For $k = 3$ runs there are 2 degrees of freedom, and since $`\chi^2_2`$ is the exponential distribution with mean 2, the p-value has the closed form $`p = \Pr(\chi^2_2 \ge X^2) = e^{-X^2/2}`$. The approximation needs expected counts of at least about 5 in every cell; the smallest here is 9.3 (FEVER's expected wrong verdicts per run). For FEVER, $\bar{p} = 272/300$ and runs of 95, 87 and 90 correct give $X^2 = 3.86$ and $p = e^{-1.93} = 0.15$.

**Two verdicts on the same claims** (the NLI verdict against the arbiter's, or $m = 2$ against $m = 1$). The verdicts are paired, so only the claims on which exactly one of them is right carry information. With $b$ claims right only under the first and $c$ right only under the second, equal accuracy implies $`b \sim \text{Binomial}(b + c, \tfrac{1}{2})`$, and the exact McNemar test gives

$$
p = \min \left( 1, \; 2 \sum_{i=0}^{\min(b, c)} \binom{b + c}{i} \, 2^{-(b + c)} \right)
$$

*Derivation.* Newcombe's interval recovers variances from the Wilson limits (Zou & Donner, 2008). For each proportion, the distances $`\hat{p}_i - l_i`$ and $`u_i - \hat{p}_i`$ play the role of $z$ standard errors below and above the estimate. The difference is smallest when $`p_1`$ is low and $`p_2`$ is high, so the lower limit of $\delta$ combines the lower distance of $`\hat{p}_1`$ with the upper distance of $`\hat{p}_2`$. Because the estimates are independent their variances add, so the distances add in quadrature; the upper limit is built symmetrically. The interval inherits the asymmetry of the Wilson intervals. It is an approximation whose coverage is close to 95% in simulation studies, not an exact interval.

*Derivation of the McNemar test* (McNemar, 1947). Each claim falls into one of four cells according to whether verdict A and verdict B are right, with probabilities $`\pi_{11}, \pi_{10}, \pi_{01}, \pi_{00}`$ (the first index for A). The accuracies are $`\pi_{11} + \pi_{10}`$ and $`\pi_{11} + \pi_{01}`$, so their difference is $`\pi_{10} - \pi_{01}`$: equal accuracy means $`\pi_{10} = \pi_{01}`$, and the claims on which both verdicts agree carry no information about it. Given $d = b + c$ discordant claims, each of them is "A right only" independently with probability $`\pi_{10} / (\pi_{10} + \pi_{01})`$, so $`b \mid d \sim \text{Binomial}\big(d, \pi_{10} / (\pi_{10} + \pi_{01})\big)`$, which under $`H_0`$ is $`\text{Binomial}(d, \tfrac{1}{2})`$. The two-sided p-value doubles the probability of the smaller tail. Treating the two sets of verdicts as independent samples would ignore that most claims are decided identically by both.

**Selection bias and cross-fitting** (the choice of $m$, §2.5). Let $`\text{acc}_S(m)`$ be the accuracy of the rule with parameter $m$ on a set $S$ of claims, and $A(m)$ its true accuracy, so that $`\mathbb{E}[\text{acc}_S(m)] = A(m)`$. If $m$ is chosen on $S$ itself, $`\hat{m} = \arg\max_m \text{acc}_S(m)`$, the reported $`\text{acc}_S(\hat{m}) = \max_m \text{acc}_S(m)`$ is biased upwards (Varma & Simon, 2006). The maximum is a convex function, so by Jensen's inequality

$$
\mathbb{E} \left[ \max_m \text{acc}_S(m) \right] \ge \max_m \mathbb{E} \left[ \text{acc}_S(m) \right] = \max_m A(m)
$$

On average the in-sample figure overstates even the best rule's true accuracy. Cross-fitting removes this bias. The claims are split into folds $`F_1, \dots, F_K`$ (here the fifteen sets, and separately $K = N$ folds of one claim). Each fold is scored with the parameter $`\hat{m}_{-k}`$ chosen on the other folds:

$$
\widehat{\text{acc}}_{\text{CF}} = \frac{1}{N} \sum_{k=1}^{K} \sum_{i \in F_k} \mathbb{I} \left( \hat{y}_i^{(\hat{m}_{-k})} = y_i \right)
$$

Claims are sampled independently, so $`\hat{m}_{-k}`$ is independent of the claims in $`F_k`$, and $`\mathbb{E} \big[ \mathbb{I}(\hat{y}_i^{(\hat{m}_{-k})} = y_i) \mid \hat{m}_{-k} \big] = A(\hat{m}_{-k})`$. Each term is therefore an unbiased estimate of the true accuracy of the parameter actually chosen. The cross-fitted accuracy estimates the accuracy of the whole procedure (choose $m$ from data, then apply it) without optimism.

*Missing reviews.* A verdict under $m = 3$ needs an LLM review of each claim with exactly two decisive contradicting snippets. If some of these reviews were not made, the analysis would bound them both ways: on any training set the score of $m = 3$ does not decrease with the number of missing reviews that are correct, while the scores of $m = 1$ and $m = 2$ do not depend on them, so counting all of them correct and then all of them wrong gives the extreme choices of $m$. All 61 reviews were made for the current evaluation (§2.5), so the cross-fitted accuracy is determined exactly.

**Multiple comparisons.** This document reports eight hypothesis tests. Even if every null hypothesis were true, the chance that at least one of eight independent tests falls below 0.05 would be $`1 - 0.95^8 = 34\%`$, so individual p-values overstate the evidence. Two corrections are applied to the whole family (choosing a smaller family after seeing the p-values would defeat the purpose), with one p-value per hypothesis: the two-proportion z-test for FEVER against FreshQA and the chi-square test for the three runs of each benchmark (both quoted in §3.3), and the exact McNemar test for the paired comparisons. With the p-values sorted, $`p_{(1)} \le \dots \le p_{(k)}`$, $k = 8$:

$$
\text{Holm:} \quad \tilde{p}_{(i)} = \max_{j \le i} \; \min \left( 1, \; (k - j + 1) \, p_{(j)} \right)
$$

$$
\text{Benjamini–Hochberg:} \quad \tilde{p}_{(i)} = \min_{j \ge i} \; \min \left( 1, \; \frac{k}{j} \, p_{(j)} \right)
$$

and a hypothesis is rejected when its adjusted $\tilde{p} < 0.05$.

*Holm's procedure* (Holm, 1979) controls the **family-wise error rate** (FWER), the probability of rejecting even one true null hypothesis, under any dependence between the tests. Let $`k_0`$ of the $k$ hypotheses be true. Suppose a true null is rejected, and let it be the first true null in the sorted order, at position $j$. Only false nulls come before it, so $`j - 1 \le k - k_0`$, and it was rejected because $`p \le \alpha / (k - j + 1) \le \alpha / k_0`$. A family-wise error therefore implies that the smallest p-value among the $`k_0`$ true nulls is at most $`\alpha / k_0`$, and by the union bound

$$
\text{FWER} \le \Pr \left( \bigcup_{i \in I_0} \left\lbrace p_i \le \frac{\alpha}{k_0} \right\rbrace \right) \le \sum_{i \in I_0} \Pr \left( p_i \le \frac{\alpha}{k_0} \right) \le k_0 \cdot \frac{\alpha}{k_0} = \alpha
$$

where $`I_0`$ is the set of true nulls and each true-null p-value satisfies $`\Pr(p_i \le u) \le u`$. Holm rejects everything Bonferroni (every $p$ multiplied by $k$) rejects, and possibly more.

*The Benjamini–Hochberg procedure* (Benjamini & Hochberg, 1995) controls the **false discovery rate** (FDR), the expected share of false rejections among all rejections, at $`\alpha k_0 / k \le \alpha`$ when the tests are independent or positively dependent (here the per-benchmark arbiter tests are positively dependent on the pooled one). It is less strict than Holm: it accepts that a small share of the significant findings may be false, rather than requiring that none is.

| Comparison | Result | Test | $p$ | Holm | BH |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LLM arbiter vs the NLI verdict it replaced, both benchmarks (187 claims, §2.9) | 60 fixed, 15 broken | exact McNemar | $`1.6 \times 10^{-7}`$ | **$`1.3 \times 10^{-6}`$** | **$`1.3 \times 10^{-6}`$** |
| LLM arbiter, FreshQA (116 claims) | 42 fixed, 8 broken | exact McNemar | $`1.2 \times 10^{-6}`$ | **$`8.1 \times 10^{-6}`$** | **$`4.7 \times 10^{-6}`$** |
| corroboration rule, $m = 2$ vs $m = 1$ (600 claims, §2.5) | 21 right only with $m = 2$, 5 only with $m = 1$ | exact McNemar | 0.002 | **0.015** | **0.006** |
| FEVER vs FreshQA accuracy (§3.3) | 90.7% vs 82.3%; difference 8.3 points [2.9, 13.8] | two-proportion z-test ($z = 2.99$; Fisher's exact test $p = 0.004$) | 0.003 | **0.015** | **0.006** |
| LLM arbiter, FEVER (71 claims) | 18 fixed, 7 broken | exact McNemar | 0.043 | 0.17 | 0.069 |
| FEVER, the three runs (§3.3) | 95, 87 and 90 of 100 | chi-square, 2 degrees of freedom ($X^2 = 3.86$) | 0.15 | 0.44 | 0.19 |
| FreshQA, the three runs (§3.3) | 82, 85 and 80 of 100 | chi-square, 2 degrees of freedom ($X^2 = 0.87$) | 0.65 | 1 | 0.74 |
| corroboration rule, $m = 2$ vs $m = 3$ (600 claims, §2.5) | 2 right only with $m = 2$, 2 only with $m = 3$ | exact McNemar | 1 | 1 | 1 |
| cross-fitted choice of $m$ vs fixed $m = 2$ (600 claims, §2.5) | 517 vs 519 correct | — | — | — | — |

Bold marks an adjusted $p < 0.05$. Four findings survive both corrections: the arbiter improves on the NLI verdict (pooled, and on FreshQA alone), $m = 2$ beats $m = 1$, and FreshQA is harder than FEVER. The arbiter's effect on FEVER alone is significant only before correction. The three runs of each benchmark are consistent with one another, and $m = 2$ and $m = 3$ cannot be told apart.

**Power and sample size.** A test that does not reject may simply have had too few claims. The **power** of a test is the probability that it rejects when a given effect is real, and it is used here to plan how many claims a conclusive evaluation needs. (Power computed at the observed effect only restates the p-value and says nothing new about a result already obtained; Hoenig & Heisey, 2001.) All figures use $\alpha = 0.05$ and a target power of 80%, and each formula is checked by simulating 20,000 evaluations from the assumed accuracies.

*Interval width.* The smallest $N$ whose Wilson half-width (§4.4) at the observed accuracy is at most $h$ is found by search: ±3.3 points now for FEVER (300 claims) needs 365 claims for ±3 and 817 for ±2; ±4.3 points for FreshQA needs 620 for ±3 and 1,396 for ±2.

*Two independent accuracies.* Under $`H_1`$ the difference $`\hat{p}_1 - \hat{p}_2`$ is approximately normal with mean $`\delta = p_1 - p_2`$ and variance $`\sigma_1^2 = (p_1 q_1 + p_2 q_2)/n`$ ($q = 1 - p$, $n$ claims per group). The z-test rejects when $`\lvert \hat{p}_1 - \hat{p}_2 \rvert > z_{1-\alpha/2} \, \sigma_0`$, with $`\sigma_0^2 = 2 \bar{p} \bar{q} / n`$ its null variance at $`\bar{p} = (p_1 + p_2)/2`$. Ignoring the negligible chance of rejecting in the wrong direction,

$$
1 - \beta = \Phi \left( \frac{\lvert \delta \rvert - z_{1-\alpha/2} \, \sigma_0}{\sigma_1} \right) \quad \Longrightarrow \quad n = \frac{\left( z_{1-\alpha/2} \sqrt{2 \bar{p} \bar{q}} + z_{1-\beta} \sqrt{p_1 q_1 + p_2 q_2} \right)^2}{\delta^2}
$$

| True difference (around 86.5%) | Power with 300 claims per benchmark | Simulated | Claims per benchmark for 80% power |
| :--- | :--- | :--- | :--- |
| 5 points | 43% | 44% | 733 |
| 7.5 points | 77% | 78% | 325 |
| 10 points | 95% | 95% | 183 |

At the observed accuracies (90.7% and 82.3%, a difference of 8.3 points) 263 claims per benchmark give 80% power (81% in simulation), so the present 300 suffice to detect a difference of this size reliably.

*Paired verdicts* (McNemar). With $\psi$ the probability that a claim is discordant (right under exactly one of the two verdicts) and $\delta$ the difference between the two discordant probabilities, the number of claims needed is (Connor, 1987)

$$
n = \frac{\left( z_{1-\alpha/2} \sqrt{\psi} + z_{1-\beta} \sqrt{\psi - \delta^2} \right)^2}{\delta^2}
$$

For the arbiter on FEVER (18 fixed, 7 broken of 71: $`\psi = 0.35`$, $`\delta = 0.15`$) this gives 113 arbitrated claims, about 480 FEVER claims at the observed arbitration rate, against 71 now; the simulated power of the exact test at that size is 77% (the exact test is slightly conservative). For $m = 3$ against $m = 2$ the discordant claims split evenly (2 against 2), so there is no observed effect to plan for. The results of the table that are not significant after correction are therefore expected at this sample size: they do not show that the arbiter does nothing on FEVER, only that 300 claims per benchmark cannot yet separate an effect of this size after correction.

### 4.6 Calibration of the NLI Probabilities

A probability is **calibrated** when it matches the frequency it predicts: among verdicts given with confidence $c$, a fraction $c$ should be right, $`\Pr(\hat{Y} = Y \mid \text{confidence} = c) = c`$. Calibration is computed by `experiments/calibration.py` from the per-claim records, without LLM calls, on the 560 benchmark claims whose final NLI verdict is SUPPORTED or CONTRADICTED (the gold labels are never UNKNOWN, so an UNKNOWN verdict has no calibration target; the 18 source-conflict claims carry no logits).

**Metrics.** For $N$ claims with gold labels $`y_i`$, probability vectors $`\boldsymbol{\pi}_i`$ and confidences $`c_i = \max_y \pi_{iy}`$:

$$
\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \ln \pi_{i y_i}, \qquad \text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{y \in \mathcal{Y}} \left( \pi_{iy} - \mathbb{I}(y = y_i) \right)^2, \qquad \text{ECE} = \sum_{b=1}^{B} \frac{\lvert B_b \rvert}{N} \left\lvert \bar{c}(B_b) - \text{acc}(B_b) \right\rvert
$$

The negative log-likelihood (NLL) and the Brier score (Brier, 1950) are proper scoring rules: their expectation is minimised by the true probabilities, so they reward calibration and sharpness together. NLL punishes confident mistakes hardest ($-\ln 0.001 = 6.9$). The **expected calibration error** (ECE; Naeini et al., 2015) sorts the claims by confidence into $B = 10$ bins of equal size (56 claims; equal-width bins would leave most of them empty, since most confidences exceed 0.95) and averages the gap between mean confidence and accuracy.

**Temperature scaling** (Guo et al., 2017) divides the three logits of §2.5 by one constant $T > 0$, $`\tilde{\pi}_y = \exp(z_y / T) / \sum_{y'} \exp(z_{y'} / T)`$. $T > 1$ softens the probabilities and $T < 1$ sharpens them, and since every logit is divided by the same positive number, their order and so the verdict never change. $T$ is the maximum-likelihood estimate. With $\beta = 1/T$, the mean NLL is

$$
L(\beta) = \frac{1}{N} \sum_{i=1}^{N} \left[ \ln \sum_{y} e^{\beta z_{iy}} - \beta z_{i y_i} \right]
$$

with derivatives

$$
L'(\beta) = \frac{1}{N} \sum_{i=1}^{N} \left( \mathbb{E}_{\tilde{\boldsymbol{\pi}}_i}[z_i] - z_{i y_i} \right), \qquad L''(\beta) = \frac{1}{N} \sum_{i=1}^{N} \mathrm{Var}_{\tilde{\boldsymbol{\pi}}_i}(z_i) \ge 0
$$

where $`\mathbb{E}_{\tilde{\boldsymbol{\pi}}_i}`$ and $`\mathrm{Var}_{\tilde{\boldsymbol{\pi}}_i}`$ treat the claim's logit as a random variable taking the value $`z_{iy}`$ with probability $`\tilde{\pi}_{iy}`$. (Differentiating the log-sum-exp gives the softmax-weighted mean, and differentiating that mean gives the variance.) The second derivative is a variance, so $L$ is convex and has a single minimum, found by Newton's method $`\beta \leftarrow \beta - L'(\beta) / L''(\beta)`$. Far from the minimum $L$ is not close to quadratic: from $\beta = 1$ the full Newton step overshoots to a negative $\beta$, so the step is halved until $L$ decreases (here twice, in the first iteration only). It then converges quadratically, reaching $T = 2.481$ to six digits in four iterations, the value a general-purpose optimiser also finds. The condition $L'(\beta) = 0$ has a direct reading: at the optimum, the expected logit under the scaled probabilities equals, on average, the logit of the true label.

**Protocol.** Fitting $T$ and scoring on the same claims would be optimistic (§4.5), so $T$ is cross-fitted: set $s$ of both benchmarks is scored with the $T$ fitted on the other nine sets. The fifteen fitted temperatures range from 2.43 to 2.56. A paired bootstrap (10,000 resamples of the claims; Efron & Tibshirani, 1993) gives 95% intervals for each change.

| Metric (560 claims) | Raw | Temperature-scaled (cross-fitted) | Change, 95% bootstrap interval |
| :--- | :--- | :--- | :--- |
| NLL | 0.898 | 0.585 | −0.313 [−0.423, −0.207] |
| Brier score | 0.296 | 0.286 | −0.010 [−0.028, +0.006] |
| ECE | 11.4% | 5.7% | −5.7 points [−8.9, −0.4] |
| Mean confidence | 95.7% | 79.6% | — |
| Accuracy | 84.6% | 84.6% | unchanged |

Reliability by bin (56 claims each, sorted by confidence; `experiments/calibration_reliability.png` plots it):

| Bin | Raw confidence | Accuracy | Scaled confidence | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 75.3% | 76.8% | 54.3% | 75.0% |
| 2 | 91.2% | 75.0% | 66.0% | 78.6% |
| 3 | 96.0% | 73.2% | 72.5% | 71.4% |
| 4 | 97.8% | 80.4% | 77.4% | 82.1% |
| 5 | 98.7% | 80.4% | 80.9% | 83.9% |
| 6 | 99.2% | 91.1% | 83.3% | 82.1% |
| 7 | 99.5% | 83.9% | 86.0% | 87.5% |
| 8 | 99.7% | 91.1% | 88.9% | 94.6% |
| 9 | 99.9% | 94.6% | 92.1% | 91.1% |
| 10 | 100.0% | 100.0% | 94.9% | 100.0% |

(The accuracy columns differ slightly because scaling can reorder confidences between claims whose other two logits differ.) The raw confidences are crowded above 0.95 while accuracy ranges from 73% to 100%. After scaling, the stated confidence tracks accuracy in most bins. The NLL improvement is clearly significant, because it comes from the confident mistakes the scaling softens, and so is the halving of the ECE (its bootstrap interval excludes zero). The Brier improvement is not significant: the Brier score also rewards ranking right verdicts above wrong ones, which a single temperature hardly changes (below), and each bin's accuracy still has a standard error of about $`\sqrt{0.85 \times 0.15 / 56} = 4.8`$ points. On the 413 claims settled by the NLI verdict alone (89.1% correct), the mean confidence falls from 98.2% to 83.1% and the ECE from 9.1% to 6.0%.

**Limits of one parameter.** A single temperature cannot fix every bin: the lowest bin, half of it (29 of 56) verdicts of the joint fallback (§2.5), becomes under-confident (54.3% stated, 75.0% right). A two-parameter Platt scaling of the verdict's confidence alone (Platt, 1999), $`\Pr(\text{right}) = \sigma\big(a \cdot \text{logit}(c) + b\big)`$, cross-fitted in the same way, reaches a top-label ECE of 5.5% against 5.7% (NLL 0.409 against 0.416, Brier 0.126 against 0.130), a difference within the noise above. Temperature scaling is kept because it has one parameter and yields a full probability vector rather than a single number.

**Calibration is not discrimination.** Scaling changes what a confidence means, but hardly which verdicts are ranked as more likely right. The raw confidence separates right from wrong verdicts with an AUROC of 0.672 (the probability that a random right verdict has a higher confidence than a random wrong one), and the cross-fitted scaled confidence with 0.673: only claims whose other two logits differ can swap places. Better routing therefore needs signals beyond the confidence, such as the number of agreeing snippets (§2.5). Decisions keep the raw scale (§2.5) because $\tau$, the conflict thresholds and the conformal quantile were set and measured on it; moving them to the calibrated scale would change which claims are routed to the LLM and would require a new measured run.

---

## 5. Limitations and Future Work

**Limitations of the method.**

1. **Missing specific facts and false premises.** When the evidence does not state the specific number, date or ranking a claim asserts, or (for a false premise such as *"An animal has landed on the Moon"*) does not state that the event never happened, the arbiter must decide from silence. It abstains rather than guess, so 19 of the 81 errors across both benchmarks are abstentions (§3.5).
2. **NLI blind spots.** The NLI model does not reason about time (*"this year"*, *"currently"*, planned vs. actual events), mishandles some negations, and accepts near-matching entities (*"World Fischer Random Chess Champion"* read as *"World Chess Champion"*). Because a decisive supporting snippet settles a claim without LLM review (§2.5), these errors reach the output directly: 17 of the 28 FEVER errors and 24 of the 53 FreshQA errors (§3.5).
3. **Scope of the conformal guarantee.** The conformal layer is calibrated on FEVER claims scored against gold evidence. On the pipeline's own retrieved evidence its measured coverage is 83.2% [80.0%, 85.9%], below the nominal 90%; recalibrating on retrieved evidence brings coverage to 90% but escalates 181 more claims for an accuracy gain that is not significant, so the production threshold is kept and the prediction set is used as a calibrated escalation trigger (§2.6).
4. **Dependence on an external LLM.** Claim decomposition, query generation, query rewriting and arbitration call a hosted LLM. Rate limits and quotas are absorbed by switching models (§2.12), but a different model may judge a borderline claim differently, and the arbiter abstains (`UNKNOWN`, flagged as an outage rather than a judgement, §2.9) if every model fails.
5. **Limits of the Knowledge Graph evidence.** Wikidata has no property for many claims (opinions, events, descriptions: 252 of the 600 benchmark claims matched none), can be incomplete or out of date, which matters most for FreshQA, and the property matching can pick a near property (§2.8).
6. **Claim-echo filter on short claims.** The filter removes web sentences that contain at least 80% of the claim's tokens (§2.3). For a short claim, an independent paraphrase reaches that threshold easily: in the current evaluation it removed the only snippets stating the fact for one true claim, which ended `UNKNOWN` (§3.5).
7. **Modest sample sizes.** The corroboration rule (§2.5) is measured on the 600 evaluation claims, and the arbiter's effect (§2.9) is measured on the 187 claims it decided; the smaller routes (e.g. 35 FEVER reviews) give indicative rather than precise accuracies. Cross-fitting shows that choosing $m = 2$ on these claims adds at most about 0.3 points of optimism to the reported accuracy, and its advantage over $m = 1$ is significant ($p = 0.002$), but $m = 2$ and $m = 3$ cannot be told apart (§2.5).

**Limitations of the evaluation.**

1. **Sample size.** With 300 claims per benchmark the Wilson 95% intervals are 7 points (FEVER) and 9 points (FreshQA) wide; differences of a few points between pipeline versions are within noise (§4.4). Detecting a true 5-point difference between two benchmarks with 80% power would need about 730 claims per benchmark, and the arbiter's effect on FEVER alone, after correction for multiple tests, more than the present 300 FEVER claims (about 480 at 80% power before correction, §4.5).
2. **Reproducibility of live runs.** Retrieval uses live Wikipedia and web search, and the LLM is not deterministic, so a re-run on the same claims can differ slightly: search results and Wikipedia pages change, and borderline claims may be routed differently.
3. **Label quality.** FreshQA gold answers are dated snapshots, and its questions are converted into claims by an LLM, which occasionally produces a claim whose label no longer matches. Fast-changing questions are excluded for this reason (§3.1), but even slow-changing labels drift: 11 of the 53 FreshQA errors in the current evaluation are claims on which current evidence contradicts the gold answer (§3.5).

**Future work.** Routing decisions from calibrated probabilities and additional features (the number of agreeing snippets, the source of the deciding snippet) with an explicit trade-off between accuracy and LLM calls, since the confidence alone ranks right and wrong verdicts only moderately (AUROC 0.67, §4.6); a non-conformity score better suited to retrieved evidence, so that a 90% guarantee can hold without escalating far more claims (§2.6); routing negated and time-dependent claims to the arbiter, or an NLI model trained on harder data, for the fast-path errors of §3.5; better retrieval for specific numeric facts, the main cause of the remaining abstentions; and a larger evaluation against a frozen evidence snapshot with re-checked labels, so that runs are exactly repeatable.

---

## 6. Reproducibility

**Requirements.** Python 3.12 and about 1.6 GB of disk space for the models (`facebook/bart-large-mnli`, 1.5 GB, and `all-MiniLM-L6-v2`, 87 MB), which are downloaded from Hugging Face on the first run. An Apple-silicon (MPS) or CUDA GPU is used for NLI when available; otherwise everything runs on the CPU. The reported latencies were measured on an Apple M1 laptop (§3.4).

**Installation.**

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt      # versions pinned to those used for the reported results
```

**Configuration.** Settings are read from environment variables, or from a `.env` file in the project root:

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `MODEL_PROVIDER` | `gemini` for the hosted Gemini models (used for all reported results), `ollama` for a local `qwen2.5:3b` | `ollama` |
| `GEMINI_API_KEY` | API key for Gemini | none |
| `LLM_ROUTING` | `primary`: every call to `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` as fallback (the setup of the reported results); `tiered`: heavy tasks on 3.5, light tasks on 3.1; `alternate`: 50/50 between the two | `primary` |
| `NLI_MODEL` | alternative NLI model (requires recalibrating the conformal layer, §2.6) | `facebook/bart-large-mnli` |
| `NLI_DEVICE` | force the NLI device (`cpu`, `mps`, `cuda`) | automatic |
| `VERIFACT_WIKIDATA` | `0` turns off the Wikidata graph in the web interface (§2.10) | `1` |
| `HF_HUB_OFFLINE` | the code sets `1` (no Hugging Face network calls); set it to `0` for the first run, so the models can be downloaded | `1` |

**Running.**

```bash
python server.py                                   # web interface at http://localhost:8080
python evaluation/run_5_runs.py --dataset all      # fifteen-set evaluation (§3.2), resumable
python experiments/full_evaluation.py              # the full evaluation of §3, including §2.5, §2.6 and §2.9 (resumable)
python experiments/eval_conformal.py               # conformal calibration and coverage (§2.6)
python experiments/statistical_tests.py            # intervals, hypothesis tests, multiple comparisons, power and the cross-fit of m (§2.5, §4.5); no LLM call (--shadow makes the missing m = 3 reviews)
python experiments/calibration.py                  # temperature scaling of the NLI probabilities and its evaluation (§4.6); no LLM call
python experiments/ablation_selective.py           # ablations, risk-coverage curves (§4.3), intervals for precision, recall and F1; no LLM call
python experiments/llm_every_claim.py              # baseline: the LLM arbiter decides every claim; --run makes the missing calls (cached, resumable)
python experiments/conformal_variants.py           # class-conditional coverage, Mondrian and APS conformal prediction (§2.6); no LLM call
python tests/test_hierarchical_scoring.py          # each file in tests/ runs as a standalone script
```

`python server.py` runs the production server (waitress, bound to `127.0.0.1`); `VERIFACT_DEBUG=1 python server.py` runs Flask's development server with auto-reload, which must not be exposed on a network. The server's limits are set with `VERIFACT_HOST`, `VERIFACT_PORT`, `VERIFACT_MAX_CHARS`, `VERIFACT_RATE_LIMIT`, `VERIFACT_RATE_WINDOW_SEC`, `VERIFACT_MAX_CONCURRENT` and `VERIFACT_CACHE_TTL_SEC` (defaults and details in `README.md`).

The fifteen-set evaluation of both benchmarks makes several hundred LLM calls. If a quota runs out, the run stops and resumes where it stopped when started again (§3.2).

---

## References

- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.* Journal of the Royal Statistical Society, Series B, 57(1), 289–300.
- Brier, G. W. (1950). *Verification of Forecasts Expressed in Terms of Probability.* Monthly Weather Review, 78(1), 1–3.
- Brown, L. D., Cai, T. T., & DasGupta, A. (2001). *Interval Estimation for a Binomial Proportion.* Statistical Science, 16(2), 101–133.
- Connor, R. J. (1987). *Sample Size for Testing Differences in Proportions for the Paired-Sample Design.* Biometrics, 43(1), 207–211.
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR.
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* arXiv:2305.14325.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
- El-Yaniv, R., & Wiener, Y. (2010). *On the Foundations of Noise-free Selective Classification.* JMLR.
- Fisher, R. A. (1922). *On the Interpretation of χ² from Contingency Tables, and the Calculation of P.* Journal of the Royal Statistical Society, 85(1), 87–94.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Hoenig, J. M., & Heisey, D. M. (2001). *The Abuse of Power: The Pervasive Fallacy of Power Calculations for Data Analysis.* The American Statistician, 55(1), 19–24.
- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of Statistics, 6(2), 65–70.
- Lewis, M., et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL.
- Liang, T., et al. (2023). *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.* arXiv:2305.19118.
- McNemar, Q. (1947). *Note on the Sampling Error of the Difference between Correlated Proportions or Percentages.* Psychometrika, 12(2), 153–157.
- Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI.
- Newcombe, R. G. (1998). *Interval Estimation for the Difference between Independent Proportions: Comparison of Eleven Methods.* Statistics in Medicine, 17(8), 873–890.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.* In Advances in Large Margin Classifiers, MIT Press.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* NAACL.
- Varma, S., & Simon, R. (2006). *Bias in Error Estimation when Using Cross-Validation for Model Selection.* BMC Bioinformatics, 7, 91.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Vu, T., et al. (2023). *FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.* arXiv:2310.03214.
- Williams, A., Nangia, N., & Bowman, S. (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MultiNLI).* NAACL.
- Wilson, E. B. (1927). *Probable Inference, the Law of Succession, and Statistical Inference.* Journal of the American Statistical Association, 22(158), 209–212.
- Zou, G. Y., & Donner, A. (2008). *Construction of Confidence Limits about Effect Measures: A General Approach.* Statistics in Medicine, 27(10), 1693–1702.

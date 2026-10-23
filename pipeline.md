# Hallucination Detection Pipeline

This document explains, step by step, how the detector decides whether a statement is a hallucination. Every stage is described with its purpose, its input and output, the exact algorithm the code runs, the mathematics behind it, a worked example taken from real runs of the system, and the reasons for the design choices. Section 3 covers the evaluation benchmarks and results, and Section 4 defines the evaluation metrics (TP, FP, TN, FN, precision, recall, F1) used to report them. Section 5 collects the limitations, and Section 6 explains how to install and run the system.

**Summary.** The detector splits a text into atomic claims, retrieves evidence for each claim from Wikipedia, the web and ArXiv, and scores it with a natural language inference (NLI) model. Split conformal prediction then decides whether that verdict can stand or whether the claim must be escalated to a Knowledge Graph and an LLM arbiter, so the expensive stages run only on ambiguous claims. Results on 100 claims per benchmark (five sets of 20, §3.3):

| Benchmark | Accuracy (95% CI) | Precision | Recall | F1 | Settled by the NLI verdict | Mean latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FEVER | 95.0% [90.7%, 99.3%] | 0.98 | 0.96 | 0.97 | 78 of 100 | 15.0 s |
| FreshQA (never- and slow-changing) | 87.0% [80.4%, 93.6%] | 0.88 | 0.92 | 0.90 | 54 of 100 | 21.4 s |

Latency in the table is from that run; with the current default LLM routing it was measured at 8.8 s (FEVER) and 11.0 s (FreshQA) on the first 20 claims of each benchmark, which took 20.5 s and 15.6 s in that run (§3.4). Precision, recall and F1 refer to the hallucination class. Claims settled by the NLI verdict alone are correct 95% of the time, and the LLM arbiter, used on the rest, fixed 30 NLI verdicts and broke 3 (§2.9). The main weaknesses (false premises, time-dependent and negated claims, and the conditions of the conformal guarantee) are discussed in Section 5.

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
   - 2.8 Knowledge Graph multi-hop reasoning
   - 2.9 LLM arbitration
   - 2.10 Output, report and user interface
   - 2.11 End-to-end algorithm and cost model
   - 2.12 Execution details
3. Evaluation: benchmarks, protocol, results, error analysis
4. Evaluation metrics
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
      stands          (1 call, no KG)               multi-hop evidence
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
| *"Juneau is the capital of Alaska."* | fast path (8 words, no clause marker) | `["Juneau is the capital of Alaska."]` |
| *"Tim Cook replaced Steve Jobs in 2011, and he is the CEO of Apple."* | LLM (contains `,` and ` and `) | `["Tim Cook replaced Steve Jobs in 2011.", "Tim Cook is the CEO of Apple."]` |

**Design rationale.** The fast path removes one LLM call (≈1–2 s and one unit of API quota) for the very common case of a single short statement; the marker list is conservative, so any sentence that might contain two propositions still goes to the LLM.

### 2.2 Query Generation (`agents/query_generator.py`)

**Purpose.** Searching for the whole claim string returns noisy results; searching for the **entities** the claim is about (people, organisations, works, places) retrieves the encyclopedia articles that actually contain the deciding fact. This stage also classifies the claim's domain and whether it depends on the present date.

**Input → Output.** Claim $c$ (plus optional context) → `{"queries": [q_1, …, q_m], "domain": "general" | "scientific", "requires_current_state": bool}` with $m \le 3$.

**Algorithm.**

1. **Fast path (no LLM)**, used when the claim has at most 15 words and either contains no ambiguous pronoun (*he, she, it, they, him, her, them, his, their, its, this, these, those*) or comes without surrounding context from which a pronoun could be resolved:
   - Claims about creative works (*album, film, movie, song, single, novel, series, …*) or using epithets (*"king of"*, *"father of"*, *"iron lady"*, …) are **excluded** from the fast path, because they need disambiguation (*"Pixels"* the film vs. the unit) or resolution of the epithet to a real person.
   - The claim is split on relational patterns, tried in order: *"is the co-founder/founder/CEO/… of"*, *"is from"*, *"is the capital/headquarters/member/part of"*, *"was born/located/based in"*, *"was founded/directed/written by"*, *"founded/created/directed"*, *"is a/an/the"*. The subject and the (capitalised) object become the entities; leading articles are removed, temporal tails such as *"in 2018"* are dropped, and internal capitalisation is preserved (`BharatPe`, `iPhone`).
   - The domain is `scientific` if the claim contains a science keyword (*celsius, fahrenheit, quantum, temperature, species, molecule, climate, dna*), and the temporal flag is set if it contains *current, currently, now, present, today, as of* or *active*.
2. **LLM path**, otherwise: the LLM returns the JSON object above, instructed to (i) disambiguate single-word entities (e.g. *"Pixels (2015 film)"*), (ii) always return at least one query, (iii) classify the domain, and (iv) set the temporal flag for claims about the present day (*"the current CEO"*, or implicit present tense such as *"Tim Cook is the CEO of Apple"*). The current date is included in the prompt. If the call or parsing fails, the claim itself becomes the only query.
3. **Timestamp injection.** For temporal claims the evidence text assembled later is prefixed with `[SYSTEM TIMESTAMP: <today>]`, so that the NLI fallback and the LLM arbiter can reason about recency.

**Worked example** (outputs from the actual code):

| Claim | Route | Queries |
| :--- | :--- | :--- |
| *"Ashneer Grover is the co-founder of BharatPe."* | fast path, pattern *"is the co-founder of"* | `["Ashneer Grover", "BharatPe"]` |
| *"Barack Obama was born in Hawaii."* | fast path, pattern *"was born in"* | `["Barack Obama", "Hawaii"]` |
| *"Michael Jackson is the King of Pop."* | LLM (epithet *"King of"*) | resolved by the LLM |
| *"The Good Wife is a movie."* | LLM (creative-work keyword *"movie"*) | `["The Good Wife (film)", "The Good Wife"]` |

**Design rationale.** Most short factual claims follow a subject–relation–object pattern, so the regular expressions recover the right entities in under a millisecond with no API cost; anything that needs world knowledge (epithets, ambiguous titles, pronouns) is deliberately routed to the LLM.

### 2.3 Evidence Retrieval (`retrieval/web_search.py`)

**Purpose.** Collect candidate paragraphs that might confirm or refute the claim, from complementary sources: Wikipedia (curated, encyclopedic, high precision), ArXiv (scientific claims) and the open web (recent events not yet in Wikipedia).

**Input → Output.** Queries $`q_1, \dots, q_m`$, claim $c$, domain → a pool of candidate paragraphs, each with its text, URL and source.

**Algorithm.**

1. **Wikipedia (per query).** The top 2 search results are fetched with `auto_suggest=False` (auto-suggest silently replaces titles with similar-looking ones). A disambiguation page resolves to its first option. The page body is cut at 25,000 characters and everything from *See also*, *References* or *External links* onward is removed.
2. **ArXiv (per query, `scientific` claims only).** Up to 3 paper abstracts.
3. **DuckDuckGo (once per claim).** A single web search on the **full claim text**, rather than on each entity: entity-only web searches return commercial SEO pages and trigger rate limits, while the whole proposition finds pages discussing the claim itself. Backends `yahoo,duckduckgo` with a 2.5 s timeout, fallback `yahoo` with 2.0 s, a hard cap of 3.0 s, and at most 3 results; a slow or rate-limited web search can therefore never stall the pipeline.
4. **Chunking.** Wikipedia text is split into sentences (sentences of 20 characters or fewer and section headings are dropped) and grouped into **overlapping windows of 5 sentences with stride 3**. Window $j$ covers sentences $3j+1, \dots, 3j+5$, so consecutive windows share 2 sentences and a fact stated across two adjacent sentences is always contained whole in some window. Windows shorter than 100 characters are discarded. Each window is one candidate paragraph.
5. **Deduplication** of identical paragraphs, then re-ranking (§2.4), which returns the top 3.
6. **Claim-echo filtering** (web snippets only, applied by the detector to the 3 re-ranked snippets), described below. A snippet removed here is not replaced, so a claim can reach NLI scoring with fewer than 3 snippets.

All queries, and the sources within each query, are fetched concurrently; each query waits at most 5 s for its sources.

**Claim-echo filtering — mathematics.** A web search for the exact claim frequently returns pages that merely *quote* it: the original opinion piece, a repost, or a fact-check that quotes a false claim before debunking it. An NLI model reads a verbatim quote as perfect entailment, so a false claim could "support itself". Let $T(x)$ be the set of lower-cased word tokens of a text $x$. Every sentence $s$ of a web snippet is removed if it contains at least 80% of the claim's tokens:

$$
\text{echo}(s) \iff \frac{\lvert T(c) \cap T(s) \rvert}{\lvert T(c) \rvert} \ge 0.8
$$

The remaining sentences are kept, and a snippet left with fewer than 8 tokens is dropped. Wikipedia and ArXiv text is never filtered, because an encyclopedic sentence that closely matches a claim is genuine support, not an echo.

**Worked example (echo filtering).** For the false claim *"'Global warming' is a myth — so say 80 graphs from 58 peer-reviewed scientific papers published in 2017."*, a web snippet read: *"These studies of past climate do not challenge the well-researched fact that human activities are responsible for climate change … 'Global warming' is a myth — so say 80 graphs from 58 peer-reviewed scientific papers published in 2017. Mr Delingpole writes"*. The quoted sentence has token overlap 1.0 and is removed; the refuting context (*"…do not challenge the well-researched fact…"*) is kept. A second snippet consisting only of the quote is dropped entirely.

### 2.4 Hybrid Re-Ranking (`retrieval/web_search.py`)

**Purpose.** A query usually returns many candidate paragraphs (86 in the example below); only the few most relevant ones should be passed to the NLI model. Two complementary relevance signals are combined: **dense** semantic similarity, which recognises paraphrases, and **sparse** lexical matching (BM25), which rewards exact names, numbers and rare words.

**Input → Output.** Claim $c$ and $N$ candidate paragraphs → the top $k = 3$ evidence snippets.

**Dense similarity.** Claim and paragraphs are embedded with `all-MiniLM-L6-v2`, a 6-layer sentence transformer producing 384-dimensional vectors (inputs truncated to 256 word pieces). For a text with token embeddings $`\mathbf{h}_1, \dots, \mathbf{h}_L`$ from the last layer, the sentence embedding is the L2-normalised mean:

$$
\bar{\mathbf{h}} = \frac{1}{L} \sum_{t=1}^{L} \mathbf{h}_t, \qquad \mathbf{e} = \frac{\bar{\mathbf{h}}}{\lVert \bar{\mathbf{h}} \rVert_2}
$$

and relevance is the cosine similarity, which for unit vectors is simply a dot product:

$$
\text{sim}(\mathbf{c}, \mathbf{e}) = \frac{\mathbf{c} \cdot \mathbf{e}}{\lVert \mathbf{c} \rVert \, \lVert \mathbf{e} \rVert} = \mathbf{c} \cdot \mathbf{e} \in [-1, 1]
$$

**Sparse similarity (BM25 Okapi).** Claim and paragraphs are lower-cased and split on whitespace. For a claim with tokens $`t_1, \dots, t_Q`$, the score of paragraph $d$ is

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

**Worked example** (real run, claim *"The Good Wife is a movie."*, queries `["The Good Wife (film)", "The Good Wife"]`): 147 Wikipedia paragraphs and 3 web results gave 86 unique candidates ($\text{avgdl} = 111.6$ tokens). Claim-token IDFs: *the* 0.848, *good* 1.053, *wife* 1.242, *is* 0.848, *a* 0.848, *movie.* 0 (the trailing full stop makes *"movie."* a token that occurs in no paragraph — a known limitation of whitespace tokenization). Top of the fused ranking:

| RRF | Dense rank (sim) | BM25 rank (score) | Source | Paragraph |
| :--- | :--- | :--- | :--- | :--- |
| 0.03252 | 1 (0.677) | 2 (7.40) | web | *"The Good Wife: Directed by Ken Cameron. With Rachel Ward, Bryan Brown…"* |
| 0.03202 | 4 (0.592) | 1 (7.83) | Wikipedia | *"The Good Wife is an American legal political drama television series that aired on CBS…"* |
| 0.03175 | 3 (0.620) | 3 (7.31) | web | *"The Good Wife is an American legal political drama television series…"* |
| 0.03008 | 7 (0.546) | 6 (6.50) | Wikipedia | cast list |

For the first row: $`\text{RRF} = \frac{1}{60 + 1} + \frac{1}{60 + 2} = 0.016393 + 0.016129 = 0.032522`$. The paragraph ranked first by BM25 ends up second because its dense rank is only 4. All three top paragraphs pass the relevance floor and become the evidence.

### 2.5 NLI Evidence Scoring (`verification/nli_verifier.py`, `detector/hallucination_detector.py`)

**Purpose.** Decide, for each evidence snippet, whether it entails the claim, contradicts it, or neither — the core verification signal — and combine the per-snippet judgements into one verdict.

**Input → Output.** Claim $c$ and evidence $`e_1, \dots, e_k`$ → a verdict $\hat{y}$ with a probability vector $`\hat{\boldsymbol{\pi}} = (\hat{\pi}_{\text{con}}, \hat{\pi}_{\text{unk}}, \hat{\pi}_{\text{sup}})`$, and a flag saying whether an LLM must review the verdict.

**The NLI model.** Natural Language Inference asks whether a *premise* entails, contradicts, or is neutral towards a *hypothesis*. The detector uses `facebook/bart-large-mnli`: BART-large (a 12-layer encoder and 12-layer decoder Transformer, hidden size 1,024, 16 attention heads, ≈407M parameters) fine-tuned on the Multi-Genre NLI corpus (MultiNLI, ≈433k sentence pairs). The evidence is the premise and the claim the hypothesis, encoded as

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

**Step 2 — Decisive snippets with corroborated contradictions.** Support and contradiction are not equally reliable. MNLI-trained models often assign a confident *contradiction* to a snippet that is merely adjacent to the claim, because they learned that a premise describing a different situation contradicts the hypothesis. Example: the true claim *"EMNLP is in Budapest, Hungary this year"* was "refuted" with $`\hat{\pi}_{\text{con}} > 0.9`$ by a paper abstract mentioning EMNLP but never its venue. On 105 development claims collected from earlier benchmark runs, the number of decisive contradicting snippets turned out to be the strongest available signal of whether a refutation is real:

| Decisive contradicting snippets | Claim actually false | Claim actually true (false refutation) | Share of refutations that were wrong |
| :--- | :--- | :--- | :--- |
| 1 | 9 | 12 | 57% |
| 2 | 9 | 6 | 40% |
| 3 | 13 | 3 | 19% |
| decisive support also present | 3 | 3 | 50% |

A single contradicting snippet was wrong more often than right; three agreeing snippets were right 13 times out of 16. The pattern was re-measured on the 200 claims of the current five-set evaluation (§3.3), which played no part in choosing the rule:

| Decisive contradicting snippets | Claims | Claim actually false | Share of refutations that were wrong |
| :--- | :--- | :--- | :--- |
| 1 | 25 | 14 | 44% |
| 2 | 28 | 24 | 14% |
| 3 or more | 47 | 46 | 2% |
| decisive support also present | 12 | 3 | 75% |

Lone and contested contradictions remain unreliable, and two agreeing snippets are already far safer, so $m = 2$ is kept. (The embedding similarity between snippet and claim was also tested as a relevance filter and did not separate the two groups: median similarity 0.62 for real refutations vs. 0.60 for false ones.) Let

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

**Worked example** (real run, claim *"The Good Wife is a movie."*, FEVER label `REFUTES`):

| Snippet | $`\hat{\pi}_{\text{con}}`$ | $`\hat{\pi}_{\text{unk}}`$ | $`\hat{\pi}_{\text{sup}}`$ | Label | Decisive? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *"The Good Wife: Directed by Ken Cameron…"* (a 1987 film of the same name) | 0.1943 | 0.0613 | 0.7444 | supported | no (0.744 < 0.80) |
| *"The Good Wife is an American legal political drama television series…"* (Wikipedia) | 0.9991 | 0.0008 | 0.0001 | contradicted | yes |
| same sentence, web copy | 0.9994 | 0.0005 | 0.0001 | contradicted | yes |

Step 1: $S = 0.744 < 0.85$, so there is no source conflict. Step 2: $`\lvert D_{\text{con}} \rvert = 2 \ge m`$ and $`D_{\text{sup}} = \varnothing`$, so the verdict is **CONTRADICTED** with $`\hat{\boldsymbol{\pi}} = (0.9994, 0.0005, 0.0001)`$, with no LLM call. Under a "highest confidence wins" rule the same verdict would follow here; the corroboration rule only changes the outcome when a contradiction stands alone or is contested.

**Design rationale and limitations.** The rule is asymmetric on purpose: a decisive *supporting* snippet is rarely wrong, while lone contradictions are wrong more often than right, and a false accusation is the more harmful error (§1.1). Corroborated contradictions — the typical case for genuinely false encyclopedic claims, which several Wikipedia sentences contradict — stay on the fast path, so the extra LLM cost is limited to the ambiguous cases (8 of 100 FEVER and 15 of 100 FreshQA claims in the five-set evaluation, §3.4). The NLI model cannot do arithmetic or date reasoning and ignores words such as *"currently"*; such cases surface as residual errors (§3.5). `HallucinationDetector(min_contra_snippets=None)` restores the original rule (the highest-confidence decisive snippet wins regardless of label) for ablation studies; a randomised check over 20,000 synthetic score configurations confirms it reproduces the original decisions exactly.

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
| (0.9994, 0.0005, 0.0001) — *The Good Wife* example | {CONTRADICTED} | (0.792, 0.010, 0.010) | no — verdict stands |
| (0.62, 0.30, 0.08) | {CONTRADICTED, UNKNOWN} | (0.208, 0.168, 0.089) | yes — two labels |
| (0.05, 0.45, 0.50) | {UNKNOWN, SUPPORTED} | (0.059, 0.208, 0.208) | yes — two labels, includes UNKNOWN |

In the second row, $0.62$ and $0.30$ exceed $0.1015$ but $0.08$ does not, so the set has two labels. In the first row, $p(\text{con}) = 0.792$ means 79 of the 100 calibration scores are at least $`1 - 0.9994 = 0.0006`$.

**Empirical validation** (`experiments/eval_conformal.py`: FEVER with gold evidence, $n = 100$ calibration claims and 100 separate test claims; coverage reported with 95% Wilson score intervals, §4.4):

| $\alpha$ | Nominal coverage $1 - \alpha$ | Empirical coverage | Wilson 95% CI | Mean set size | $`\hat{q}_\alpha`$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.05 | 95% | 99.00% | [94.57%, 99.82%] | 1.63 | 0.9881 |
| **0.10** | **90%** | **90.00%** | **[82.59%, 94.51%]** | **1.26** | **0.8985** |
| 0.15 | 85% | 85.00% | [76.70%, 90.73%] | 1.13 | 0.7787 |
| 0.20 | 80% | 79.00% | [70.04%, 85.83%] | 1.01 | 0.6286 |

At the production level $\alpha = 0.10$ the empirical coverage equals the nominal 90% with only 1.26 labels per set on average, so most claims receive a decisive singleton and only a minority are escalated. For $\alpha \ge 0.20$ the sets are already almost always singletons, so coverage cannot fall further and stays at the raw accuracy of the NLI model on gold evidence (79%). Plots: `experiments/conformal_coverage_plot.png`, `experiments/conformal_set_size_plot.png`.

**Coverage on retrieved evidence** (`experiments/full_evaluation.py`, the 200 claims of the five-set evaluation, §3.3). Here the NLI probabilities are the ones the live pipeline hands to the conformal layer, computed on retrieved snippets. For the recalibration the claims were split into two halves; each half was tested with the $`\hat{q}`$ calibrated on the other (cross-fitting), so no claim was used to calibrate its own threshold:

| Threshold | Coverage on retrieved evidence | Wilson 95% CI | Mean set size | Estimated accuracy (FEVER / FreshQA) |
| :--- | :--- | :--- | :--- | :--- |
| current, $`\hat{q} = 0.8985`$ (gold-evidence calibration) | 85.0% | [79.4%, 89.3%] | 1.25 | 95% / 87% (measured) |
| recalibrated on retrieved evidence, $`\hat{q} \approx 0.92`$–$0.99$ | 88.0% | [82.8%, 91.8%] | 1.58 | 93% / 86% (estimated) |

With the current threshold the coverage on live evidence is measurably below the nominal 90% (the interval lies below it), as the exchangeability argument predicted. Recalibrating restores coverage consistent with 90%, but only with a threshold near 0.98, which admits any label above about 2% probability: 32 more claims would be escalated, and the estimated accuracy is not higher. (For the estimate, claims whose route would change were re-decided: newly escalated claims received one LLM review of their saved evidence, without a Knowledge Graph.) The production threshold is therefore kept at 0.8985, and it is described as what it is in deployment: a calibrated escalation trigger with 85% measured coverage, not a 90% guarantee.

### 2.7 Routing: Query Rewrite, Absence Evaluation and Escalation (`detector/hallucination_detector.py`)

**Purpose.** Decide what happens to each claim after scoring: accept the verdict, search again, judge the absence of evidence, ask the LLM to review, or escalate to full reasoning.

**Self-reflective query rewrite (at most once per claim).** Retrieval can fail because a query was ambiguous, too specific, or used a name Wikipedia does not. On the **first pass only**, if no evidence was found or the verdict is `UNKNOWN`, a rewriter LLM receives the claim, the queries already tried, and up to 1,000 characters of the unhelpful evidence; it is asked to diagnose why the search failed and to propose up to 2 new, *divergent* queries (alternative names, the underlying event, a parent entity). Retrieval and scoring then run again. Evidence from both rounds is pooled (deduplicated by text), so a worse second search can never discard evidence found by the first. Verdicts that are already `SUPPORTED` or `CONTRADICTED` are not re-searched, even when the conformal set is ambiguous; they go straight to escalation.

**Absence evaluation.** If evidence is still missing after the rewrite, the absence of evidence is itself informative, but only for prominent facts: if the claim describes a major, widely recorded event (a world leader's death, a famous monument being moved) and nothing is found, it is almost certainly false; if it describes a minor detail, the evidence may simply not be online. The LLM is asked which case applies and answers `CONTRADICTED` or `UNKNOWN`. The probability vector passed to the conformal layer is $(1, 0, 0)$ or $(0, 1, 0)$ respectively.

**Escalation rule.** After scoring, a claim is fully escalated (Knowledge Graph §2.8, then LLM arbitration §2.9) when

$$
\text{escalate}(X) \iff \hat{y} = \text{UNKNOWN} \;\lor\; \text{escalate}_{\text{conf}}(X) \;\lor\; \left( \text{shared role} \;\land\; \hat{y} = \text{CONTRADICTED} \right)
$$

If it is not escalated but §2.5 flagged a lone or contested contradiction, it gets a **lightweight LLM review**: one arbitration call over the evidence already retrieved, without building a Knowledge Graph. Otherwise the NLI verdict stands.

Each result records the route that produced it in `final_decision_policy`:

| Value | Route |
| :--- | :--- |
| `conformal_singleton` | NLI verdict accepted (decisive, single-label conformal set) |
| `contradiction_arbitration` | lightweight LLM review of a lone or contested contradiction |
| `multi_agent_debate` | full escalation: Knowledge Graph + LLM arbitration |
| `absence_heuristic` | no evidence after the rewrite; absence evaluation decided |

**Worked example** (real run, FreshQA false-premise claim *"In 2021, an astronomer successfully lobbied the IAU to change Pluto's planetary classification from 'dwarf planet' back to 'planet'."*): the first pass ended with the verdict `UNKNOWN` (conformal set {CONTRADICTED, UNKNOWN}), so a query rewrite was triggered. The second pass was still ambiguous and the claim was escalated: the Knowledge Graph added the nodes *Pluto* and *Planetary protection* with one provenance-verified edge, and the LLM arbiter concluded *CONTRADICTED*, citing that the evidence still classifies Pluto as a dwarf planet and mentions no 2021 reversal — the correct verdict.

### 2.8 Knowledge Graph Multi-Hop Reasoning (`agents/knowledge_graph.py`)

**Purpose.** Some claims connect several entities (*"The director of Inception was born in London"*), and the deciding fact may sit in a different article from the one the claim names. For escalated claims, the pipeline builds a small graph of the entities involved, extracts **verified** relationships between them, and adds the most relevant facts as extra evidence for the arbiter ("multi-hop": reaching a fact through an intermediate entity).

**Input → Output.** Claim, its entities, and the evidence so far → a graph $G = (V, E)$ and up to 3 additional evidence passages.

**Nodes.** The entities are the claim's search queries from §2.2 (or, if the claim itself was used as the query, up to 5 entities extracted by an entity-linking LLM call that also resolves implicit references, e.g. *"the director of Inception"* → *Christopher Nolan*). Each node $v \in V$ stores the entity's Wikipedia summary $`\text{Summary}_v`$ (first 1,500 characters), reused from the pages fetched in §2.3 when available, so no new request is needed. If the node's title appears in the retrieved evidence, the first 1,000 characters of that evidence are appended to its summary, so facts from the article body are available too.

**Disambiguation.** Titles are requested with `auto_suggest=False`. When a title leads to a disambiguation page, candidate titles tagged as music (`(album)`, `(song)`, `(band)`, `(single)`, `(EP)`) are discarded unless the claim is about music or film, and the remaining candidates $\mathcal{T}$ are ranked by embedding similarity to the claim:

$$
\hat{t} = \underset{t \in \mathcal{T}}{\arg\max} \; \frac{\mathbf{c} \cdot \mathbf{t}}{\lVert \mathbf{c} \rVert \, \lVert \mathbf{t} \rVert}
$$

so that, for example, a claim about global temperatures resolves *"Global Warming"* to the climate article, not to a rock album of the same name.

**Edges.** With $\lvert V \rvert = N \ge 2$ nodes there are $\binom{N}{2}$ possible pairs; instead of one LLM call per pair, a **single** LLM call receives all summaries plus the first 2,500 characters of the evidence and returns every relationship it can ground in the text, as JSON objects with a source, a target, a relationship of at most 15 words, and a verbatim supporting quote $q$ of at most 20 words.

**Provenance verification.** Relationships extracted by an LLM can themselves be hallucinated, which would contaminate the evidence. Each edge is therefore checked deterministically. Let $\nu(x)$ lower-case a text and strip punctuation; then

$$
\text{verified}(u, v) \iff \nu(q) \sqsubseteq \nu(\text{Summary}_u) \;\lor\; \nu(q) \sqsubseteq \nu(\text{Summary}_v) \;\lor\; \nu(q) \sqsubseteq \nu(\text{Evidence})
$$

where $a \sqsubseteq b$ means $a$ occurs in $b$ as a contiguous substring. To tolerate small punctuation differences, an edge is also accepted when the first $`\min(\lvert q \rvert_{\text{words}}, 5)`$ words of the quote (at least 3) occur verbatim; quotes shorter than 5 characters are rejected. Unverified edges are drawn as dashed lines in the UI for exploration but are **never used as evidence**.

**Multi-hop evidence.** The candidate pool consists of

- every node summary split into chunks of 3 sentences, starting every 2 sentences (chunks under 30 characters dropped), and
- every verified edge rendered as a sentence: *"Grounded Relationship [u ↔ v]: rel (Evidence Quote: "q")"*.

All candidates are embedded in one batch and ranked by cosine similarity $\text{sim}(\mathbf{c}, \mathbf{e})$ to the claim (§2.4); the top 3 are appended to the evidence as `KG Evidence [node]: …`. Edge sentences compete on plain similarity with no manual bonus, so a relationship is used only if it is genuinely relevant to the claim.

**Illustrative example.** For *"The director of Inception was born in London."*, entity linking yields *Inception*, *London* and *Christopher Nolan*. The edge extractor returns (*Christopher Nolan*, *Inception*, "directed the film", quote *"written and directed by Christopher Nolan"*); the quote is found in the *Inception* summary, so the edge is verified. The Nolan summary chunk stating his birthplace and the verified edge rank highest by similarity and are passed to the arbiter, which can now link the director to his birthplace — two hops that no single retrieved paragraph covered.

### 2.9 LLM Arbitration (`agents/critic_agent.py`)

**Purpose.** Resolve the claims the NLI model could not settle reliably: escalated claims and lone or contested contradictions. A large language model can handle paraphrase, implicit entailment, multi-sentence reasoning and dates, which the NLI classifier cannot.

**Input → Output.** Claim and full evidence text (retrieved snippets, any Knowledge Graph evidence, the timestamp line and the source-conflict note) → final label.

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
- **False premises and misidentified entities:** a claim that refers to an entity by a wrong name, title, date or attribute is `CONTRADICTED`, and the claim's premise must never be silently corrected to fit the evidence. This rule balances the anti-pedantry rules, which on their own let the arbiter "repair" false premises: it once accepted *"Benjamin Ruth is a baseball player in the MLB who hit home runs"* while itself noting that Babe Ruth's given name is George Herman. The rule is stated generically, with no benchmark claim in the prompt; `strict_premises=False` removes it for ablations.

**Conservative abstention.** The verdict is read from the last line of the response beginning with `VERDICT:`:

$$
\hat{y}_{\text{final}} = \begin{cases}
\text{SUPPORTED} & \text{if the evidence entails } c \text{ and no counter-evidence stands} \\\\
\text{CONTRADICTED} & \text{if the evidence decisively refutes } c \\\\
\text{UNKNOWN} & \text{otherwise (abstention)}
\end{cases}
$$

The prompt reserves `UNKNOWN` for evidence that is absent, unrelated, or genuinely balanced, so the arbiter does not abstain merely to avoid a decision. If the LLM call itself fails (quota exhausted, overload, timeout), the verdict is also `UNKNOWN`, but the result is flagged (`llm_unavailable`) so that the interface reports an outage rather than an abstention, and the run is not recorded for replay.

**Worked example** (real arbitration, the Pluto claim of §2.7, abridged): *Perspective A:* "the evidence notes ongoing debate … but there is no direct evidence that an astronomer successfully lobbied the IAU in 2021." *Perspective B:* "Evidence 2 states that in 2006 the IAU redefined *planet* to exclude dwarf planets such as Pluto; the KG evidence classifies Pluto as a dwarf planet; there is no mention of any 2021 change." *Perspective C:* "the claim is contradicted by the current classification of Pluto as a dwarf planet." **VERDICT: CONTRADICTED.**

**Measured value** (five-set evaluation, §3.3). The computation trace records the NLI verdict just before arbitration (`UNKNOWN` for a source conflict or an inconclusive joint reading), so the arbiter's effect can be read off for every claim it decided:

| Benchmark | Claims decided by the LLM | Fixed (NLI wrong, final right) | Broken (NLI right, final wrong) | Both right | Both wrong |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FEVER | 22 (8 reviews, 14 escalations) | 9 | 1 | 10 | 2 |
| FreshQA | 46 (15 reviews, 31 escalations) | 21 | 2 | 17 | 6 |
| **Total** | **68** | **30** | **3** | **27** | **8** |

The arbiter corrected 30 verdicts for every 3 it spoiled.

### 2.10 Output, Report and User Interface

**Per-claim result.** The claim; the evidence text and source URLs; the verdict with its probability vector and confidence; the conformal set, p-values, $`\hat{q}`$ and set size; the decision route (`final_decision_policy`); and the Knowledge Graph (nodes with Wikipedia links and summaries, edges with relationship, quote and verification status).

**Report.** A summary of the whole input:

$$
\text{hallucination rate} = \frac{\text{number of claims labelled CONTRADICTED}}{\text{total number of claims}}
$$

together with the numbers of supported, contradicted and unknown claims, of escalated claims, and of claims settled by a single-label conformal set.

**Streaming UI.** The Flask server (`/api/analyze_stream`) streams newline-delimited JSON events: the list of extracted claims, one progress event per stage (tagged with the stage name and the claim it belongs to), each claim's result as soon as it is verified, and the final report. Claims are verified concurrently with the same per-claim code as the batch API, so the verdicts are identical; for a two-claim input this cut the response time from 51 s to 19 s. The web page (`static/`) has four tabs. **Verify** shows a live stepper of the pipeline stages, combined over the claims in progress (a stage is active while any claim is in it, with a count of how many claims have reached it; optional stages such as query rewrite, Knowledge Graph and arbiter are marked as skipped when no claim needs them), a small progress bar of each claim's own stages, fills in one card per claim as results arrive (verdict, decision route, NLI posterior, conformal set and p-values, and every evidence snippet with its own NLI reading, the snippet that decided the verdict, and its source), shows a timeline of the time each claim spent in each phase, offers an **Under the hood** view of any claim's computation as it happens (the generated queries; the re-ranking table with cosine similarity, BM25, both ranks and the RRF formula; every snippet's NLI logits and the softmax; the decision-rule quantities S, K, |D_sup|, |D_con| and the branch taken; the conformal scores, q̂, p-values and set; the routing condition term by term; and the LLM arbiter's written defense, prosecution and judgement), all from a per-claim trace that the detector records in `verification["trace"]` without changing any decision, highlights the input text by verdict, and draws the Knowledge Graph with D3.js: hovering a node shows its summary and Wikipedia link, hovering an edge shows its relationship, quote and whether it is verified. **Benchmarks** charts the five-set results of §3.3 from `evaluation/*_5_runs_results.json` (`/api/benchmarks`), including the accuracy of each decision route (§3.4). **Method** animates the cascade, with claims flowing into the three routes in the measured proportions, summarises the stages and lists the known limitations, and **Docs** renders this document (`/api/docs`). For display only, when arbitration concludes `SUPPORTED` on a graph whose nodes have no edges, one extra LLM call extracts the supporting relationship from the evidence so the graph shows it; this happens after the verdict and cannot change it, and the streaming UI sends the verdict first and the extra edge afterwards. A "Deep Scan" option builds the graph for every claim, not only escalated ones.

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
        if pass = 1: queries ← REWRITE(c, "no evidence"); continue  # §2.7
        y ← ABSENCE_EVAL(c);  return y                              # §2.7
    text ← JOIN(E);  if temporal: text ← "[SYSTEM TIMESTAMP]" + text
    (ŷ, π̂, review) ← SCORE(c, E, text)                              # §2.5
    C ← CONFORMAL_SET(π̂)                                            # §2.6
    if ŷ = UNKNOWN and pass = 1: queries ← REWRITE(c, text); continue
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
| Full escalation | entity linking 0 or 1, edge extraction 1 (if ≥ 2 nodes), arbitration 1, display edge 0 or 1 |

A claim settled on the fast path with fast-path queries needs **no LLM call at all**. Measured in the five-set evaluation (§3.4): 78 of 100 FEVER and 54 of 100 FreshQA claims were settled by the NLI verdict without any arbitration call, and a claim costs 1.32 LLM calls on average on FEVER and 1.88 on FreshQA: 0.92 when the NLI verdict stands (mostly query generation), 1.91 with a lightweight review, and 3.44 with full escalation.

### 2.12 Execution Details

- **LLM backend (`llm/`).** With `MODEL_PROVIDER=gemini`, two models are used, each serving as the other's fallback (larger Gemini models were dropped: their free-tier daily quota is too small to help). With the default `LLM_ROUTING=primary`, every call goes to `gemini-3.5-flash-lite` and falls back to `gemini-3.1-flash-lite` only when 3.5 is rate-limited, overloaded or out of daily quota: in the five-set evaluation 3.5 answered its calls in 2.9 s on average against 5.2 s for 3.1 (§3.4), so sending work to 3.1 by default made checks slower rather than faster. Because 3.5 is faster, claims finish sooner and calls arrive closer together: during the latency measurement of §3.4 the Gemini console showed 3.5 at 18 requests in one minute against its limit of 15, so under sustained load the per-minute limit, not speed, is what sends calls to 3.1. Each call is also tagged by how demanding its task is, **heavy** for the steps that decide or preserve a verdict (arbitration, absence evaluation, claim decomposition) and **light** for supporting steps (query generation and rewriting, entity linking, Knowledge Graph relations); `LLM_ROUTING=tiered` sends heavy calls to 3.5 and light calls to 3.1, the setup of the five-set results in §3.3, and `LLM_ROUTING=alternate` alternates between the two regardless of task. In every mode, a model that returns HTTP 429, 503 or 504, or times out, is skipped for 45 seconds and the request moves to the next model immediately; a model whose *daily* quota is used up is skipped until the quota resets (midnight Pacific time), so requests go straight to the models that can still answer. If both models fail at once with such a temporary error (a demand spike), the request waits and tries them again, after 2 s and then 5 s, before giving up; when both daily quotas are used up it fails at once, since waiting cannot help. Otherwise a local Ollama model (`qwen2.5:3b`) is used.
- **Concurrency.** Claims, queries and data sources are processed in thread pools, and all network calls run in parallel. The NLI model and the embedding models are each protected by a lock, because their native code is not safe to call from several threads at once; on Apple GPUs (MPS) the command buffer is synchronised before the NLI lock is released. Embedding models run on the CPU and NLI on the GPU when available (`NLI_DEVICE` overrides the device).
- **Warm-up.** When the server starts, the NLI model, both embedding models and the NLTK sentence tokenizer are each run once on a dummy input, so the first real request does not pay one-off initialisation (a cold NLI call took 1.18 s against 0.22 s warm). No cache is filled and no LLM is called.
- **Caching.** Wikipedia searches and parsed pages (LRU caches), paragraph embeddings (keyed by text), DuckDuckGo results (10 minutes, successful lookups only), NLI scores per claim–snippet pair, and Wikipedia summaries for the graph are cached. Every cache returns exactly what recomputation would, so outputs are unchanged; the effect is speed, mainly on query-rewrite retries, which re-retrieve many of the same pages (a repeated search took 3.05 s instead of 6.52 s in a controlled measurement).

---

## 3. Evaluation

### 3.1 Benchmarks

**FEVER** (Fact Extraction and VERification; Thorne et al., 2018). Claims were written by annotators by altering sentences from Wikipedia, then labelled `SUPPORTS`, `REFUTES` or `NOT ENOUGH INFO` against Wikipedia. The detector is evaluated on the `SUPPORTS` and `REFUTES` claims of the validation split (`copenlu/fever_gold_evidence`). FEVER tests entity retrieval and NLI on encyclopedic facts without temporal change, and is the primary benchmark.

**FreshQA** (Vu et al., Google Research, 2023, continuously maintained). Questions whose answers change over time, plus false-premise questions, with answers maintained by the authors in a public sheet (cached in `evaluation/data/freshqa.csv`). Each question has a `fact_type`: *never-changing*, *slow-changing* or *fast-changing*. Because the detector verifies claims rather than answering questions, every question–answer pair is converted once into one declarative claim by an LLM (`evaluation/prepare_freshqa_cache.py`, cached in `evaluation/data/freshqa_claims.json`):

| FreshQA item | Claim | Label |
| :--- | :--- | :--- |
| Normal question + answer, e.g. *"Who is the CEO of Google?" – "Sundar Pichai"* | *"Sundar Pichai is the CEO of Google."* | `SUPPORTS` |
| False-premise question, e.g. *"When was League of Legends 2 released?"* | the false premise as a claim: *"League of Legends 2 was released."* | `REFUTES` |

FreshQA tests web retrieval, temporal reasoning and false-premise detection. Its gold answers are snapshots dated by the maintainers, so a fast-changing answer can be out of date when the benchmark is run. In an earlier 40-claim run, fast-changing questions scored 38% against 78% for the rest, mostly through such out-of-date labels. All FreshQA runs therefore sample only *never-changing* and *slow-changing* questions (`DEFAULT_FACT_TYPES` in `evaluation/benchmarks/freshqa_dataset.py`; 298 true and 147 false claims).

Benchmarks built from synthetic perturbations of a fixed reference text (such as HaluEval) are not used: their "false" claims are often true in the open world that this detector searches, which would penalise correct verdicts.

### 3.2 Protocol

1. **Stratified sampling.** $N$ claims are drawn at random, half `SUPPORTS` and half `REFUTES`, so that precision and recall of the hallucination class are both measurable. `--seed` makes the draw reproducible.
2. **Verification.** Each claim is passed to `detector.analyze(claim, extract_atomic_facts=False)` — benchmark claims are already atomic — with live retrieval and LLM calls. FreshQA claims also receive the original question as context.
3. **Aggregation.** If an input yields several claims, it is predicted `CONTRADICTED` if any claim is contradicted, otherwise `UNKNOWN` if any is unknown, otherwise `SUPPORTED` (one false fact makes an answer a hallucination). With `extract_atomic_facts=False` there is exactly one claim.
4. **Scoring.** The prediction is compared with the gold label (`SUPPORTS` → `SUPPORTED`, `REFUTES` → `CONTRADICTED`). `UNKNOWN` and runtime errors count as wrong. Accuracy, precision/recall/F1 of the `CONTRADICTED` class (§4) and latency per claim (mean, median, 90th percentile) are reported.

**Five-set evaluation.** The reported results use `evaluation/run_5_runs.py`: each benchmark is split into 5 non-overlapping stratified sets of 20 claims (10 `SUPPORTS`, 10 `REFUTES`) drawn with seed 42, so every run evaluates the same 100 claims per benchmark. Results are reported per set and pooled. A claim whose verification hits an LLM infrastructure failure (exhausted quota, timeout) is retried once after a pause and the run stops if it fails again, so infrastructure failures are never scored as abstentions; per-claim results are saved as they finish (`evaluation/<dataset>_5_runs_claims.jsonl`), and an interrupted run resumes where it stopped.

```bash
python evaluation/run_5_runs.py --dataset all          # FEVER then FreshQA, 5 sets x 20 claims each
python evaluation/run_5_runs.py --dataset fever        # one benchmark
python evaluation/run_benchmark.py --dataset fever --samples 20                 # quick single run
python evaluation/run_benchmark.py --dataset freshqa --samples 20              # never- and slow-changing questions only
```

### 3.3 Results (Current Pipeline)

All figures below are measured with the current pipeline (the single configuration described in Section 2), with live retrieval and LLM calls (October 2026), by `experiments/full_evaluation.py`, which runs the five-set evaluation with a full record of every claim (verdict, computation trace, evidence and LLM calls) and derives all the results of this section and of §2.5, §2.6 and §2.9 from that one run. LLM calls were routed with `LLM_ROUTING=tiered`: the verdict-deciding calls on `gemini-3.5-flash-lite`, as in the current default, and the supporting calls (query generation, entity linking) on `gemini-3.1-flash-lite`, which the current default also sends to 3.5 (§2.12). No LLM call failed during the run.

**FEVER: five-set evaluation.** 5 non-overlapping stratified sets of 20 claims (10 `SUPPORTS`, 10 `REFUTES`), seed 42, 100 claims in total (summary in `evaluation/fever_5_runs_results.json`, per-claim records in `evaluation/fever_5_runs_claims.jsonl`):

| Set | Claims | Correct | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | `UNKNOWN` | Mean latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Set 1 | 20 | 19 | 95.00% | 1.00 | 1.00 | 1.00 | 10 | 0 | 0 | 9 | 1 | 20.5 s |
| Set 2 | 20 | 19 | 95.00% | 0.91 | 1.00 | 0.95 | 10 | 1 | 0 | 9 | 0 | 14.8 s |
| Set 3 | 20 | 18 | 90.00% | 1.00 | 0.90 | 0.95 | 9 | 0 | 1 | 9 | 1 | 10.9 s |
| Set 4 | 20 | 19 | 95.00% | 1.00 | 0.90 | 0.95 | 9 | 0 | 1 | 10 | 1 | 16.1 s |
| Set 5 | 20 | 20 | 100.00% | 1.00 | 1.00 | 1.00 | 10 | 0 | 0 | 10 | 0 | 12.8 s |
| **Pooled** | **100** | **95** | **95.00%** | **0.98** | **0.96** | **0.97** | 48 | 1 | 2 | 47 | 3 | 15.0 s |

Pooled 95% CI (normal approximation, §4.4): [90.7%, 99.3%]. With 20 claims per set, one error moves a set's accuracy by 5 points, so the pooled figure is the one to quote.

**FreshQA: five-set evaluation.** Same protocol on never- and slow-changing questions (§3.1), 100 claims (summary in `evaluation/freshqa_5_runs_results.json`, per-claim records in `evaluation/freshqa_5_runs_claims.jsonl`):

| Set | Claims | Correct | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN | `UNKNOWN` | Mean latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Set 1 | 20 | 17 | 85.00% | 0.90 | 0.90 | 0.90 | 9 | 1 | 1 | 8 | 1 | 15.6 s |
| Set 2 | 20 | 16 | 80.00% | 0.75 | 0.90 | 0.82 | 9 | 3 | 1 | 7 | 0 | 18.3 s |
| Set 3 | 20 | 18 | 90.00% | 1.00 | 0.90 | 0.95 | 9 | 0 | 1 | 9 | 1 | 32.1 s |
| Set 4 | 20 | 16 | 80.00% | 0.82 | 0.90 | 0.86 | 9 | 2 | 1 | 7 | 1 | 22.9 s |
| Set 5 | 20 | 20 | 100.00% | 1.00 | 1.00 | 1.00 | 10 | 0 | 0 | 10 | 0 | 18.3 s |
| **Pooled** | **100** | **87** | **87.00%** | **0.88** | **0.92** | **0.90** | 46 | 6 | 4 | 41 | 3 | 21.4 s |

Pooled 95% CI (normal approximation, §4.4): [80.4%, 93.6%]. Precision, recall and F1 refer to the `CONTRADICTED` (hallucination) class (definitions in §4.1); `UNKNOWN` counts as an error, and on a false claim as a missed hallucination (FN).

**Change from the previous evaluation.** The same 200 claims were evaluated before the LLM routing changed (calls then alternated between the two Flash-Lite models, with two larger models as backups; those results are kept in `evaluation/runs/published-2026-10-22/`). Comparing claim by claim: on FEVER 3 verdicts were fixed and 4 became wrong (96% then, 95% now); on FreshQA 8 were fixed and 6 became wrong (85% then, 87% now). Neither difference is statistically meaningful on 100 claims. The clearer change is in how FreshQA's errors are distributed: with the verdict-deciding calls always on the stronger model, the arbiter commits more often instead of abstaining, so abstentions fell from 7 to 3 and recall rose from 0.82 to 0.92, while precision fell from 0.93 to 0.88.

**FEVER vs FreshQA.** The confidence intervals now overlap ([90.7%, 99.3%] and [80.4%, 93.6%]), so the gap is suggestive rather than conclusive. Recall is similar (0.96 vs 0.92); the difference is in precision (0.98 vs 0.88): FreshQA has 6 false accusations: two are claims whose gold answer is contradicted by current evidence, and the other four are NLI or arbiter errors on negations, time and geography (§3.5).

### 3.4 Latency, Routing and Cost (Current Pipeline)

| Benchmark | Mean latency | Median | 90th percentile | LLM calls per claim |
| :--- | :--- | :--- | :--- | :--- |
| FEVER (100 claims, five sets) | 15.0 s | 11.3 s | 32.2 s | 1.32 |
| FreshQA (100 claims, five sets) | 21.4 s | 16.5 s | 39.7 s | 1.88 |

| Benchmark | NLI verdict stands | Lightweight review | Full escalation |
| :--- | :--- | :--- | :--- |
| FEVER (100 claims, five sets) | 78 | 8 | 14 |
| FreshQA (100 claims, five sets) | 54 | 15 | 31 |

Accuracy by route, the evidence that the conformal layer sends the right claims to the LLM: the claims settled by the NLI verdict alone are correct 95% of the time across both benchmarks (125 of 132), and the claims it passed on, which are the harder ones, are correct 84% of the time after LLM review (57 of 68).

| Benchmark | NLI verdict stands | Lightweight review | Full escalation |
| :--- | :--- | :--- | :--- |
| FEVER | 76 / 78 (97%) | 5 / 8 (63%) | 14 / 14 (100%) |
| FreshQA | 49 / 54 (91%) | 14 / 15 (93%) | 24 / 31 (77%) |

LLM calls by route (both benchmarks): 0.92 per claim when the NLI verdict stands, 1.91 with a lightweight review, 3.44 with full escalation. Every call was logged with the model that answered it:

| Model (task tier) | Calls | Mean | Median |
| :--- | :--- | :--- | :--- |
| `gemini-3.5-flash-lite` (heavy: arbitration) | 67 | 2.9 s | 2.7 s |
| `gemini-3.1-flash-lite` (light: queries, entities, relations) | 204 | 5.2 s | 3.9 s |
| `gemini-3.5-flash-lite` (light, as fallback) | 48 | 8.4 s | 2.0 s |
| `gemini-3.1-flash-lite` (heavy, as fallback) | 1 | 28.8 s | 28.8 s |

Latency is measured end to end with live Wikipedia, web and Gemini calls on an Apple M1 laptop. It is higher than in the previous evaluation (10.5 s and 16.8 s): `gemini-3.1-flash-lite` was markedly slower than 3.5 during the run, and its query-generation call sits on the critical path of most claims. This measurement is why the default routing now sends every call to 3.5 first (§2.12). Claims settled by the NLI verdict are fastest; fully escalated claims, which add Knowledge Graph construction and arbitration, account for the long tail.

**Latency with the current default** (`python evaluation/run_5_runs.py --dataset all --sets 1 --tag default-latency`: the first set of 20 claims of each benchmark, the same claims as Set 1 above, with every LLM call routed to `gemini-3.5-flash-lite` first). All 59 LLM calls were answered by 3.5, in 1.4 s on average, with no fallback and no failure:

| Benchmark (Set 1, 20 claims) | Mean | Median | 90th percentile | Accuracy | LLM calls per claim |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FEVER, current default | 8.8 s | 7.6 s | 16.0 s | 19 / 20 | 1.30 |
| FEVER, same claims in the run above | 20.5 s | 14.4 s | 44.1 s | 19 / 20 | — |
| FreshQA, current default | 11.0 s | 9.6 s | 20.0 s | 16 / 20 | 1.65 |
| FreshQA, same claims in the run above | 15.6 s | 12.6 s | 26.2 s | 17 / 20 | — |

Accuracy is unchanged within the noise of 20 claims (one FreshQA claim differs). Part of the speed-up may come from network and server conditions on the day, which differ between runs; the per-call logs attribute most of it to the faster model.

### 3.5 Error Analysis

**FEVER (current, five sets, 100 claims).** The 5 errors are 1 false accusation, 1 missed hallucination and 3 abstentions:

1. **NLI fast-path errors (2).** *"Underdog features the voice talents of Amy Adams"* (true) was settled `CONTRADICTED` because two snippets about her other films contradicted it with high confidence, which met the corroboration rule. *"Middle-earth is a real place"* (false) was settled `SUPPORTED` by a decisive snippet from Tolkien's own commentary on the "literary credibility" of his invented history.
2. **Abstentions (3).** *"2016 Tour de France was won by a rider"* (true; the evidence covered other years), *"Caroline Kennedy is Ambassador to Japan"* (true, labelled for her 2013–17 term; the evidence listed the office but not her) and *"Nuuk is the largest military center of Greenland"* (false; the evidence described Nuuk's civilian institutions only). In each case a lone or contested contradicting snippet sent the claim to review, and the arbiter, finding no direct statement either way, abstained.

**FreshQA (current, five sets, 100 claims, never- and slow-changing).**

| Fact type | Claims | Correct | Wrong | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| never-changing | 52 | 48 | 4 | 92.3% |
| slow-changing | 48 | 39 | 9 | 81.3% |

The 13 errors fall into four groups:

1. **Gold labels contradicted by current evidence (5).** The 2026 evidence states that Lewandowski left Barcelona at the end of the 2025–26 season (claim *"Lewandowski plays for Barcelona"*, labelled true), that the LA Metro has 110 rail stations (labelled: 107), and that OpenAI has released GPT-5.5 (labelled false); *"A FIFA World Cup will be hosted next year"* (labelled false) is true of the 2027 Women's World Cup, and *"Joe Biden announced his second presidential campaign in 2023"* was generated from a false-premise question but is true as stated. In these cases the detector followed the evidence, and even slow-changing labels are drifting out of date.
2. **NLI fast-path errors (4).** Two negations (*"The Great Pyramid of Giza is not one of the New 7 Wonders"* and *"No humans have landed on Mars yet"*, both true, both settled `CONTRADICTED`), a claim about the present matched to historical evidence (*"The euro is Croatia's current national currency"*, contradicted by snippets about the 1990s economy), and a near-match (a claim that a *cube* number is the sum of two cubes in two ways, supported by evidence about 1729, which is not a cube).
3. **Abstentions on true claims (3).** The specific fact was not retrieved: Biden's most-visited country, the price of parking at the Big E, and the UMass NLP faculty count.
4. **Arbiter error (1).** *"The province that borders Shanghai to the north is Jiangsu"* (true): the arbiter read the evidence's description of Shanghai's northern boundary as excluding Jiangsu.

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

**Worked example** (current pipeline, FEVER five-set evaluation, 100 claims: 50 false and 50 true): 48 false claims were flagged `CONTRADICTED`, 1 was judged `SUPPORTED` and 1 ended `UNKNOWN`; 47 true claims were judged `SUPPORTED`, 1 was flagged `CONTRADICTED` and 2 ended `UNKNOWN`. So $TP = 48$, $FN = 2$, $FP = 1$, $TN = 47$, and the 2 abstentions on true claims count only against accuracy:

$$
\text{Accuracy} = \frac{48 + 47}{100} = 95.0\\%, \quad \text{Precision} = \frac{48}{48 + 1} = 0.98, \quad \text{Recall} = \frac{48}{48 + 2} = 0.96, \quad F_1 = \frac{2 \times 48}{2 \times 48 + 1 + 2} = 0.97
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

Near 0% or 100% the normal approximation is poor (it can even exceed 100%), so the conformal coverage in §2.6 is reported with the **Wilson score interval**:

$$
\frac{\hat{p} + \dfrac{z^2}{2N} \pm z \sqrt{\dfrac{\hat{p}(1 - \hat{p})}{N} + \dfrac{z^2}{4N^2}}}{1 + \dfrac{z^2}{N}}, \qquad z = 1.96
$$

---

## 5. Limitations and Future Work

**Limitations of the method.**

1. **Missing specific facts and false premises.** When the evidence does not state the specific number, date or ranking a claim asserts, or (for a false premise such as *"An animal has landed on the Moon"*) does not state that the event never happened, the arbiter must decide from silence. It now commits more often than it used to (FreshQA abstentions fell from 7 to 3, §3.3), but 6 of the 18 errors across both benchmarks are still abstentions of this kind (§3.5).
2. **NLI blind spots.** The NLI model does not reason about time (*"this year"*, *"currently"*, planned vs. actual events), mishandles some negations, and accepts near-matching entities (*"World Fischer Random Chess Champion"* read as *"World Chess Champion"*). Because a decisive supporting snippet settles a claim without LLM review (§2.5), these errors reach the output directly: 2 of the 5 FEVER errors and 4 of the 13 FreshQA errors (§3.5).
3. **Scope of the conformal guarantee.** The conformal layer is calibrated on FEVER claims scored against gold evidence. On the pipeline's own retrieved evidence its measured coverage is 85% [79%, 89%], below the nominal 90%; recalibrating on retrieved evidence restores coverage but escalates many more claims without improving accuracy, so the production threshold is kept and the prediction set is used as a calibrated escalation trigger (§2.6).
4. **Dependence on an external LLM.** Query generation, query rewriting, entity linking, edge extraction and arbitration call a hosted LLM. Rate limits and quotas are absorbed by switching models (§2.12), but a different model may judge a borderline claim differently, and the arbiter abstains (`UNKNOWN`, flagged as an outage rather than a judgement, §2.9) if every model fails.
5. **Modest sample sizes.** The corroboration rule (§2.5) was derived from 105 development claims and confirmed on 200 evaluation claims, and the arbiter's effect (§2.9) is measured on the 68 claims it decided; the smaller routes (e.g. 8 FEVER reviews) give indicative rather than precise accuracies.

**Limitations of the evaluation.**

1. **Sample size.** With 100 claims per benchmark the 95% confidence intervals are about ±4 points (FEVER) and ±7 points (FreshQA); differences of a few points between pipeline versions are within noise (§4.4).
2. **Reproducibility of live runs.** Retrieval uses live Wikipedia and web search, and the LLM is not deterministic, so a re-run on the same claims can differ slightly: search results and Wikipedia pages change, and borderline claims may be routed differently.
3. **Label quality.** FreshQA gold answers are dated snapshots, and its questions are converted into claims by an LLM, which occasionally produces a claim whose label no longer matches. Fast-changing questions are excluded for this reason (§3.1), but even slow-changing labels drift: 5 of the 13 FreshQA errors in the current evaluation are claims on which current evidence contradicts the gold answer (§3.5).
4. **Routing of supporting LLM calls.** The reported results sent the supporting calls (query generation, entity linking) to `gemini-3.1-flash-lite`; the current default sends them to `gemini-3.5-flash-lite` as well (§2.12). The verdict-deciding calls are routed identically, so accuracy is expected to be similar, but the supporting calls shape retrieval, so the default configuration has not been measured separately.

**Future work.** A non-conformity score better suited to retrieved evidence, so that a 90% guarantee can hold without escalating far more claims (§2.6); routing negated and time-dependent claims to the arbiter, or an NLI model trained on harder data, for the fast-path errors of §3.5; better retrieval for specific numeric facts, the main cause of the remaining abstentions; and a larger evaluation against a frozen evidence snapshot with re-checked labels, so that runs are exactly repeatable.

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
| `LLM_ROUTING` | `primary`: every call to `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` as fallback; `tiered`: heavy tasks on 3.5, light tasks on 3.1 (the setup of the reported results); `alternate`: 50/50 between the two | `primary` |
| `NLI_MODEL` | alternative NLI model (requires recalibrating the conformal layer, §2.6) | `facebook/bart-large-mnli` |
| `NLI_DEVICE` | force the NLI device (`cpu`, `mps`, `cuda`) | automatic |
| `HF_HUB_OFFLINE` | the code sets `1` (no Hugging Face network calls); set it to `0` for the first run, so the models can be downloaded | `1` |

**Running.**

```bash
python server.py                                   # web interface at http://localhost:8080
python evaluation/run_5_runs.py --dataset all      # five-set evaluation (§3.2), resumable
python experiments/full_evaluation.py              # the full evaluation of §3, including §2.5, §2.6 and §2.9 (resumable)
python experiments/eval_conformal.py               # conformal calibration and coverage (§2.6)
python tests/test_hierarchical_scoring.py          # each file in tests/ runs as a standalone script
```

`python server.py` runs the production server (waitress, bound to `127.0.0.1`); `VERIFACT_DEBUG=1 python server.py` runs Flask's development server with auto-reload, which must not be exposed on a network. The server's limits are set with `VERIFACT_HOST`, `VERIFACT_PORT`, `VERIFACT_MAX_CHARS`, `VERIFACT_RATE_LIMIT`, `VERIFACT_RATE_WINDOW_SEC` and `VERIFACT_MAX_CONCURRENT` (defaults and details in `README.md`).

The five-set evaluation of both benchmarks makes several hundred LLM calls. If a quota runs out, the run stops and resumes where it stopped when started again (§3.2).

---

## References

- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR.
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* arXiv:2305.14325.
- El-Yaniv, R., & Wiener, Y. (2010). *On the Foundations of Noise-free Selective Classification.* JMLR.
- Lewis, M., et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL.
- Liang, T., et al. (2023). *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.* arXiv:2305.19118.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* NAACL.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Vu, T., et al. (2023). *FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.* arXiv:2310.03214.
- Williams, A., Nangia, N., & Bowman, S. (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MultiNLI).* NAACL.

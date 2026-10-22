# Hallucination Detection Pipeline

This document explains, step by step, how the detector decides whether a statement is a hallucination. Every stage is described with its purpose, its input and output, the exact algorithm the code runs, the mathematics behind it, a worked example taken from real runs of the system, and the reasons for the design choices. Section 3 covers the evaluation benchmarks and results, and Section 4 defines the evaluation metrics (TP, FP, TN, FN, precision, recall, F1) used to report them.

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

---

## 1. Problem Formulation and Overview

### 1.1 Task

A **hallucination** is a statement produced by a language model that is not supported by, or is contradicted by, real-world evidence. Given an input text $T$ (typically an LLM answer), the detector must:

1. split $T$ into atomic claims $`c_1, \dots, c_n`$, each expressing one checkable fact;
2. for each claim $c$, find an evidence set $`\mathcal{E}(c) = \{e_1, \dots, e_k\}`$ from external sources (Wikipedia, ArXiv, the web);
3. assign a verdict

$$
f(c, \mathcal{E}(c)) \in \mathcal{Y} = \{\text{CONTRADICTED}, \; \text{UNKNOWN}, \; \text{SUPPORTED}\}
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

1. **Fast path (no LLM)**, used when the claim has at most 15 words and contains no ambiguous pronoun (*he, she, it, they, him, her, them, his, their, its, this, these, those*):
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
4. **Chunking.** Wikipedia text is split into sentences (sentences shorter than 20 characters, such as headings, are dropped) and grouped into **overlapping windows of 5 sentences with stride 3**. Window $j$ covers sentences $3j+1, \dots, 3j+5$, so consecutive windows share 2 sentences and a fact stated across two adjacent sentences is always contained whole in some window. Windows shorter than 100 characters are discarded. Each window is one candidate paragraph.
5. **Claim-echo filtering** (web snippets only), described below.
6. **Deduplication** of identical paragraphs, then re-ranking (§2.4).

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

truncated from the right to the model's 1,024-token limit. The decoder's final hidden state at the last `</s>` token, $`\mathbf{h}_{\text{eos}} \in \mathbb{R}^{1024}`$, summarises the pair, and the classification head maps it to three logits:

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

A single contradicting snippet was wrong more often than right; three agreeing snippets were right 13 times out of 16. (The embedding similarity between snippet and claim was also tested as a relevance filter and did not separate the two groups: median similarity 0.62 for real refutations vs. 0.60 for false ones.) Let

$$
D_{\text{sup}} = \{ i : \hat{y}(e_i) = \text{SUPPORTED},\; \hat{\pi}_{\text{sup}}(e_i) \ge \tau \}, \qquad D_{\text{con}} = \{ i : \hat{y}(e_i) = \text{CONTRADICTED},\; \hat{\pi}_{\text{con}}(e_i) \ge \tau \}
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

**Design rationale and limitations.** The rule is asymmetric on purpose: a decisive *supporting* snippet is rarely wrong, while lone contradictions are wrong more often than right, and a false accusation is the more harmful error (§1.1). Corroborated contradictions — the typical case for genuinely false encyclopedic claims, which several Wikipedia sentences contradict — stay on the fast path, so the extra LLM cost is limited to the ambiguous cases (3 of 40 FEVER claims in the held-out evaluation). The NLI model cannot do arithmetic or date reasoning and ignores words such as *"currently"*; such cases surface as residual errors (§3.5). `HallucinationDetector(min_contra_snippets=None)` restores the original rule (the highest-confidence decisive snippet wins regardless of label) for ablation studies; a randomised check over 20,000 synthetic score configurations confirms it reproduces the original decisions exactly.

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

The production calibration (`experiments/conformal_calibration.json`) uses $n = 100$ FEVER claims and $\alpha = 0.10$, so $`k = \lceil 101 \times 0.9 \rceil = \lceil 90.9 \rceil = 91`$ and $`\hat{q}_{0.10} = s_{(91)} = 0.8985`$.

**Step 3 — Prediction set.** For a new claim $X$, every label whose score would not be unusually high is included:

$$
\mathcal{C}(X) = \{ y \in \mathcal{Y} : s(X, y) \le \hat{q}_\alpha \} = \{ y \in \mathcal{Y} : \hat{\pi}_y(X) \ge 1 - \hat{q}_\alpha \}
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

**What the guarantee does and does not say.** It is a **marginal** guarantee about the *set*, averaged over claims. It does not say that a single point verdict is right 90% of the time (a set $`\{\text{CONTRADICTED}, \text{UNKNOWN}\}`$ can contain the truth while the point verdict *CONTRADICTED* is wrong); it does not bound the precision of `CONTRADICTED` verdicts; and coverage can be lower for the rarer class and higher for the common one. It also requires exchangeability: when deployment claims differ strongly from the FEVER calibration claims (other domains, other NLI models), coverage can degrade, and the layer must be recalibrated (`experiments/eval_conformal.py`).

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

**Empirical validation** (`experiments/eval_conformal.py`: FEVER, $n = 100$ calibration claims and 100 separate test claims; coverage reported with 95% Wilson score intervals, §4.4):

| $\alpha$ | Nominal coverage $1 - \alpha$ | Empirical coverage | Wilson 95% CI | Mean set size | $`\hat{q}_\alpha`$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.05 | 95% | 99.00% | [94.57%, 99.82%] | 1.63 | 0.9881 |
| **0.10** | **90%** | **90.00%** | **[82.59%, 94.51%]** | **1.26** | **0.8985** |
| 0.15 | 85% | 85.00% | [76.70%, 90.73%] | 1.13 | 0.7787 |
| 0.20 | 80% | 79.00% | [70.04%, 85.83%] | 1.01 | 0.6286 |

At the production level $\alpha = 0.10$ the empirical coverage equals the nominal 90% with only 1.26 labels per set on average, so most claims receive a decisive singleton and only a minority are escalated. For $\alpha \ge 0.20$ the sets are already almost always singletons, so coverage cannot fall further and stays at the raw accuracy of the NLI model (79%). Plots: `experiments/conformal_coverage_plot.png`, `experiments/conformal_set_size_plot.png`.

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

The prompt reserves `UNKNOWN` for evidence that is absent, unrelated, or genuinely balanced, so the arbiter does not abstain merely to avoid a decision. If the LLM call itself fails (quota exhausted, timeout), the verdict is also `UNKNOWN`.

**Worked example** (real arbitration, the Pluto claim of §2.7, abridged): *Perspective A:* "the evidence notes ongoing debate … but there is no direct evidence that an astronomer successfully lobbied the IAU in 2021." *Perspective B:* "Evidence 2 states that in 2006 the IAU redefined *planet* to exclude dwarf planets such as Pluto; the KG evidence classifies Pluto as a dwarf planet; there is no mention of any 2021 change." *Perspective C:* "the claim is contradicted by the current classification of Pluto as a dwarf planet." **VERDICT: CONTRADICTED.**

**Measured value** (`experiments/ablation_debate.py`, 15 claims, same pipeline with and without arbitration): accuracy 86.67% → 100.00%; 2 claims were escalated, both were corrected and none was made worse; mean latency 5.05 s → 5.81 s per claim.

### 2.10 Output, Report and User Interface

**Per-claim result.** The claim; the evidence text and source URLs; the verdict with its probability vector and confidence; the conformal set, p-values, $`\hat{q}`$ and set size; the decision route (`final_decision_policy`); and the Knowledge Graph (nodes with Wikipedia links and summaries, edges with relationship, quote and verification status).

**Report.** A summary of the whole input:

$$
\text{hallucination rate} = \frac{\#\{\text{claims labelled CONTRADICTED}\}}{\#\{\text{claims}\}}
$$

together with the numbers of supported, contradicted and unknown claims, of escalated claims, and of claims settled by a single-label conformal set.

**Streaming UI.** The Flask server (`/api/analyze_stream`) streams one progress event per stage as newline-delimited JSON, so the user sees each claim's search, scoring, conformal set and arbitration as they happen. The web page (`static/`) shows each claim's conformal set and p-values, and draws the Knowledge Graph with D3.js: hovering a node shows its summary and Wikipedia link, hovering an edge shows its relationship, quote and whether it is verified. For display only, when arbitration concludes `SUPPORTED` on a graph whose nodes have no edges, one extra LLM call extracts the supporting relationship from the evidence so the graph shows it; this happens after the verdict and cannot change it. A "Deep Scan" option builds the graph for every claim, not only escalated ones.

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

A claim settled on the fast path with fast-path queries needs **no LLM call at all**. Measured averages on held-out claims: 1.27 calls per FEVER claim and 1.73 per FreshQA claim (§3.4).

### 2.12 Execution Details

- **LLM backend (`llm/`).** With `MODEL_PROVIDER=gemini`, requests alternate between `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`, combining their separate per-model rate limits, with `gemini-3.8-flash` and `gemini-3.5-flash` as backups; a model that returns HTTP 429, 503 or 504 is skipped for 45 seconds and the request moves to the next model immediately. Otherwise a local Ollama model (`qwen2.5:3b`) is used.
- **Concurrency.** Claims, queries and data sources are processed in thread pools, and all network calls run in parallel. The NLI model and the embedding models are each protected by a lock, because their native code is not safe to call from several threads at once; on Apple GPUs (MPS) the command buffer is synchronised before the NLI lock is released. Embedding models run on the CPU and NLI on the GPU when available (`NLI_DEVICE` overrides the device).
- **Caching.** Wikipedia searches and parsed pages (LRU caches), paragraph embeddings (keyed by text), DuckDuckGo results (10 minutes, successful lookups only), NLI scores per claim–snippet pair, and Wikipedia summaries for the graph are cached. Every cache returns exactly what recomputation would, so outputs are unchanged; the effect is speed, mainly on query-rewrite retries, which re-retrieve many of the same pages (a repeated search took 3.05 s instead of 6.52 s in a controlled measurement).

---

## 3. Evaluation

### 3.1 Benchmarks

**FEVER** (Fact Extraction and VERification; Thorne et al., 2018). Claims were written by annotators by altering sentences from Wikipedia, then labelled `SUPPORTS`, `REFUTES` or `NOT ENOUGH INFO` against Wikipedia. The detector is evaluated on the `SUPPORTS` and `REFUTES` claims of the validation split (`copenlu/fever_gold_evidence`). FEVER tests entity retrieval and NLI on encyclopedic facts without temporal change, and is the primary benchmark.

**FreshQA** (Vu et al., Google Research, 2023, continuously maintained). Questions whose answers change over time, plus false-premise questions, with answers maintained by the authors in a public sheet (cached in `evaluation/data/freshqa.csv`). Each question has a `fact_type`: *never-changing*, *slow-changing* or *fast-changing*. Because the detector verifies claims rather than answering questions, every question–answer pair is converted once into one declarative claim by an LLM (`evaluation/prepare_freshqa_cache.py`, cached in `evaluation/data/freshqa_claims.json`):

| FreshQA item | Claim | Label |
| :--- | :--- | :--- |
| Normal question + answer, e.g. *"Who is the CEO of Google?" – "Sundar Pichai"* | *"Sundar Pichai is the CEO of Google."* | `SUPPORTS` |
| False-premise question, e.g. *"When was League of Legends 2 released?"* | the false premise as a claim: *"League of Legends 2 was released in the past."* | `REFUTES` |

FreshQA tests web retrieval, temporal reasoning and false-premise detection. Its gold answers are snapshots dated by the maintainers, so a fast-changing answer can be out of date when the benchmark is run; `--stable` restricts sampling to never- and slow-changing questions.

Benchmarks built from synthetic perturbations of a fixed reference text (such as HaluEval) are not used: their "false" claims are often true in the open world that this detector searches, which would penalise correct verdicts.

### 3.2 Protocol

1. **Stratified sampling.** $N$ claims are drawn at random, half `SUPPORTS` and half `REFUTES`, so that precision and recall of the hallucination class are both measurable. `--seed` makes the draw reproducible.
2. **Verification.** Each claim is passed to `detector.analyze(claim, extract_atomic_facts=False)` — benchmark claims are already atomic — with live retrieval and LLM calls. FreshQA claims also receive the original question as context.
3. **Aggregation.** If an input yields several claims, it is predicted `CONTRADICTED` if any claim is contradicted, otherwise `UNKNOWN` if any is unknown, otherwise `SUPPORTED` (one false fact makes an answer a hallucination). With `extract_atomic_facts=False` there is exactly one claim.
4. **Scoring.** The prediction is compared with the gold label (`SUPPORTS` → `SUPPORTED`, `REFUTES` → `CONTRADICTED`). `UNKNOWN` and runtime errors count as wrong. Accuracy, precision/recall/F1 of the `CONTRADICTED` class (§4) and latency per claim (mean, median, 90th percentile) are reported.

```bash
python evaluation/run_benchmark.py --dataset fever --samples 20
python evaluation/run_benchmark.py --dataset freshqa --samples 20
python evaluation/run_benchmark.py --dataset freshqa --samples 20 --stable    # never- and slow-changing only
python evaluation/run_benchmark.py --dataset all --samples 20 --seed 42 --out results.json
python evaluation/run_freshqa_5_runs.py                                        # 5 non-overlapping sets of 20
```

### 3.3 Results

| Benchmark | Pipeline version | Claims | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FEVER | earlier version, 5 runs (with atomic decomposition) | 215 atomic claims | 83.26% (179/215) | 0.85 | 0.82 | 0.83 |
| FEVER | earlier version, single run | 20 | 95.00% (19/20) | 1.00 | 0.90 | 0.95 |
| **FEVER** | **current, held-out (Oct 2026)** | **40** | **92.50% (37/40)** | **1.00** | **0.85** | **0.92** |
| FreshQA | earlier version, 5 runs (`evaluation/freshqa_5_runs_results.json`) | 100 | 80.00% (80/100) | 0.84 | 0.82 | 0.83 |
| **FreshQA** | **current, held-out (Oct 2026)** | **40** | **65.00% (26/40)** | **0.70** | **0.80** | **0.74** |
| FreshQA, stable subset | current, held-out (Oct 2026) | 27 | 77.78% (21/27) | — | — | — |

Precision, recall and F1 refer to the `CONTRADICTED` (hallucination) class. The held-out claims (random seed 2026) were excluded from all development work, so no design decision was tuned on them. The rows come from different samples and pipeline versions, and the two versions have not been compared on the same claims, so differences between rows mix pipeline changes with sampling variation. With 40 claims the 95% confidence interval is roughly ±9 points at 90% accuracy and ±15 points at 65% (§4.4).

**Five-run FreshQA result (earlier version), per set:**

| Set | Claims | Correct | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 20 | 16 | 80.00% | 0.82 | 0.90 | 0.86 |
| 2 | 20 | 16 | 80.00% | 0.75 | 0.90 | 0.82 |
| 3 | 20 | 17 | 85.00% | 0.89 | 0.80 | 0.84 |
| 4 | 20 | 15 | 75.00% | 0.88 | 0.70 | 0.78 |
| 5 | 20 | 16 | 80.00% | 0.89 | 0.80 | 0.84 |
| **Pooled** | **100** | **80** | **80.00%** | **0.84** | **0.82** | **0.83** |

Pooled standard error: $`\sqrt{0.8 \times 0.2 / 100} = 4.0\%`$, 95% interval $`[72.2\%, 87.8\%]`$.

### 3.4 Latency and Routing (current pipeline, held-out runs)

| Benchmark | Mean latency | Median | 90th percentile | LLM calls per claim |
| :--- | :--- | :--- | :--- | :--- |
| FEVER | 15.8 s | 13.8 s | 31.9 s | 1.27 |
| FreshQA | 19.7 s | 15.0 s | 51.5 s | 1.73 |

| Benchmark | NLI verdict stands | Lightweight review | Full escalation |
| :--- | :--- | :--- | :--- |
| FEVER (40 claims) | 31 | 3 | 6 |
| FreshQA (40 claims) | 22 | 10 | 8 |

Latency is measured end to end with live Wikipedia, web and Gemini calls on an Apple M1 laptop. Claims settled by the NLI verdict are fastest; fully escalated claims, which add Knowledge Graph construction and arbitration, account for the long tail.

### 3.5 Error Analysis

**FEVER (current, held-out).** All 3 errors are false claims judged `SUPPORTED` on the direct NLI path; no true claim was flagged as a hallucination (precision 1.00). In earlier runs the dominant failure was **NLI literalism**: the classifier misses entailments stated as abstract paraphrases, which is what the arbitration stage addresses.

**FreshQA (current, held-out).** Accuracy depends strongly on the question's `fact_type`:

| Fact type | Correct | Wrong | Accuracy |
| :--- | :--- | :--- | :--- |
| never-changing | 11 | 4 | 73% |
| slow-changing | 10 | 2 | 83% |
| fast-changing | 5 | 8 | 38% |

The 14 errors fall into three groups:

1. **Out-of-date gold labels (8 errors, all fast-changing).** The gold answer is a dated snapshot that no longer holds, or a value that cannot be checked from encyclopedic sources. Examples: *"The latest version of Android is Android 16"* while the retrieved evidence documents Android 17; *"The next U.S. federal holiday is Memorial Day"*, recorded in spring and evaluated in October; *"The last time it rained in New York City was April 17, 2026."* In these cases the benchmark measures the freshness of its labels as much as the detector, so FreshQA is best reported on the stable subset, with fast-changing questions reported separately.
2. **Genuine NLI errors involving time (about 4 errors).** *"A city hosted the Summer Olympics in 1940"* was judged supported because the evidence says the 1940 games were "planned" and "awarded to Helsinki" (they were cancelled); a claim about what Canada *currently* requires was matched to evidence about 2020–21 policy. The NLI model does not distinguish planned from actual events or past from present.
3. **Question-to-claim conversion (2 errors).** The generated claim is arguably true: *"Joe Biden announced his second presidential campaign in 2023"* was produced from a false-premise question, yet Biden did announce his re-election campaign in 2023.

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
\text{Accuracy} = \frac{\#\{\text{claims whose prediction equals the gold label}\}}{N}
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

**Worked example** (current pipeline, FEVER held-out, 40 claims: 20 false and 20 true): 17 false claims were flagged `CONTRADICTED` and 3 were judged `SUPPORTED`; all 20 true claims were judged `SUPPORTED`. So $TP = 17$, $FN = 3$, $FP = 0$, $TN = 20$:

$$
\text{Accuracy} = \frac{17 + 20}{40} = 92.5\\%, \quad \text{Precision} = \frac{17}{17 + 0} = 1.00, \quad \text{Recall} = \frac{17}{17 + 3} = 0.85, \quad F_1 = \frac{2 \times 17}{2 \times 17 + 0 + 3} = 0.92
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

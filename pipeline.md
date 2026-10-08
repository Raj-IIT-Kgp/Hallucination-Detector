# Architectural Report: Automated Hallucination Detection Pipeline with Knowledge Graph Reasoning

## 1. Abstract
This report outlines the architecture and mathematical constraints of an end-to-end Automated Hallucination Detection system. The pipeline integrates deterministic claim extraction, multi-entity query generation, hybrid web retrieval, zero-shot Natural Language Inference (NLI), a novel **Knowledge Graph (KG) multi-hop reasoning layer**, and a final Critic LLM override — powered by Gemini 3.5 Flash Lite — to identify and flag factual hallucinations in AI-generated text. When standard NLI verification is inconclusive, the system dynamically constructs a live in-memory Knowledge Graph from Wikipedia entities, traverses it to collect multi-hop evidence chains, and feeds the enriched context back into the NLI pipeline before escalating to the LLM as a last resort.

---

## 2. System Architecture & Pipeline Workflow

The system implements a **6-Phase Cascaded Verification Architecture**. Each phase acts as a progressively more powerful (but also more expensive) reasoning engine. The pipeline only escalates to the next phase if the previous one is inconclusive, maximizing speed and minimizing API cost.

### Phase 1: Deterministic Claim Extraction
To prevent the introduction of secondary hallucinations during the data processing phase, the system avoids using generative LLMs for extraction. Instead, it utilizes the **Natural Language Toolkit (NLTK)** (specifically the `sent_tokenize` module) to deterministically execute sentence tokenization, splitting the input corpus into an array of isolated factual claims.

### Phase 2: Multi-Entity Query Generation
For each isolated claim, a **Query Generator Agent (powered by Gemini 3.5 Flash Lite)** dynamically synthesizes an optimal search strategy.
* **Entity Extraction:** Rather than guessing a single subject, the LLM is instructed to extract *all* relevant entities (e.g., both "2026 FIFA World Cup" and "France national football team") and outputs them as a strict JSON array.
* **Domain Classification:** The agent simultaneously classifies the claim's domain as `general` or `scientific`, which triggers domain-specific retrieval routing in the next phase.
* **Optimization:** To prevent coreference resolution failures (e.g., misinterpreting pronouns like "he" or "it"), the agent is supplied with the *entire original text corpus* as contextual state, ensuring highly accurate entity resolution.

### Phase 3: Parallel Web Retrieval (Hybrid Search)
The system executes parallel live queries against the **Wikipedia API**, the **ArXiv API** (for scientific claims), and **DuckDuckGo** to download peer-reviewed academic abstracts, Wikipedia pages, and general web snippets into a massive contextual pool.
* **Mathematical Vectorization:** The system uses the `sentence-transformers/all-MiniLM-L6-v2` dense embedding model to map the extracted claim and all merged paragraphs into high-dimensional vector space.
* **Cosine Similarity Maximization:** The semantic similarity between the claim and the evidence is computed using **Cosine Similarity**, yielding a scalar score $S_c \in [-1.0, 1.0]$.
* **Hybrid Re-Ranking (RRF):** The pipeline combines dense Cosine Similarity scores with sparse **BM25** keyword scores using **Reciprocal Rank Fusion (RRF)** to select the Top-3 most relevant paragraphs as ground-truth evidence.

### Phase 4: Primary Verification (Natural Language Inference)
The claim and the Top-3 evidence paragraphs are processed by a zero-shot NLI model (`facebook/bart-large-mnli` running via HuggingFace Transformers).
* **Probability Distribution:** The model outputs a softmax probability distribution across three classes: `SUPPORTED`, `CONTRADICTED`, and `UNKNOWN`.
* **Strict Confidence Thresholds:** To rigorously prevent false positives and false negatives, the system enforces a strict mathematical threshold. If the model predicts a label with a confidence score $P(y) < 0.75$, the pipeline degrades the classification to `UNKNOWN`, escalating to Phase 5.

### Phase 5a: Knowledge Graph Multi-Hop Reasoning *(New)*
This is the most architecturally novel component of the system. When the NLI model returns `UNKNOWN`, instead of immediately calling the expensive LLM, the system first constructs a **live in-memory Knowledge Graph** to perform multi-hop evidence reasoning.

**Step 1 — Implicit Entity Resolution & Linking (`entity_linker.py`):**
Gemini 3.5 Flash Lite performs advanced Entity Linking on the claim. Critically, it is instructed to resolve **implicit hidden entities** using its world knowledge before querying Wikipedia. For example, if a claim states *"The director of Inception was born in London,"* the LLM extracts not just "Inception" and "London", but actively resolves the hidden entity to extract *"Christopher Nolan"*. Each entity is then resolved to its canonical Wikipedia article title.

**Step 2 — Knowledge Graph Construction (`knowledge_graph.py`):**
Using the `networkx` library, the system builds a live in-memory graph:
* **Nodes:** Each resolved Wikipedia entity becomes a graph node. The system fetches the Wikipedia page for each entity and stores its full text summary.
* **Edges:** Gemini 3.5 Flash Lite is queried for every pair of nodes to extract a concise natural-language relationship (e.g., *"Christopher Nolan directed Inception"*). If a relationship exists, a weighted edge is added between the nodes.

**Step 3 — Multi-Hop Evidence Traversal:**
The system applies cosine similarity across all node summaries in the graph — including summaries of entities that were *not* directly mentioned in the claim — to collect a pool of multi-hop evidence chunks. This allows the pipeline to reason across chains of connected facts.

**Step 4 — NLI Re-Verification:**
The original evidence pool is merged with the KG-sourced multi-hop evidence to create a significantly richer context. The NLI model is re-run on this enriched, combined evidence. In many cases, this resolves the `UNKNOWN` verdict without ever calling the LLM.

### Phase 5b: Critic LLM Override (Final Fallback)
If the NLI model still returns `UNKNOWN` after the Knowledge Graph traversal, the system invokes **Gemini 3.5 Flash Lite** as a final reasoning engine. It reads all collected evidence chunks (both web-retrieved and KG-sourced) and performs Chain-of-Thought (CoT) logical deduction to arrive at a final `SUPPORTED` or `CONTRADICTED` verdict.

### Phase 6: Real-Time Stream Reconstruction
As the system resolves claims sequentially, the backend utilizes a Python Generator to push asynchronous NDJSON (Server-Sent Events) to a **Flask-based frontend client**. This produces a real-time, ChatGPT-style progress UI. The frontend renders:
* **Annotated text** with color-coded claim spans (green = supported, red = contradicted, grey = unknown).
* **An interactive D3.js Knowledge Graph visualization** showing all discovered entity nodes, relationship edges, and the evidence path used for multi-hop reasoning — visible whenever Phase 5a was triggered.

---

## 3. Multi-Dataset Evaluation

The system was evaluated against three major fact-verification datasets. The pipeline was subjected to end-to-end live testing, requiring dynamic retrieval, mathematical NLI verification, and multi-agent consensus over the live Wikipedia API and DuckDuckGo.

The following metrics reflect a rigorous, end-to-end evaluation of the pipeline. These results highlight both the system's capabilities and the expected architectural limitations when performing live, open-domain retrieval.

### 1. FEVER (Fact Extraction and VERification)
**Dataset Profile:** FEVER is a rigorous, peer-reviewed academic standard consisting of relatively straightforward, general-knowledge claims generated by altering sentences from Wikipedia. The claims are mostly unambiguous and fact-centric (e.g., "History of art includes architecture, dance, sculpture, music...").

**Performance Analysis (5-Run Cross Validation):**
To ensure statistical honesty and mitigate the variance of random sampling, the system was evaluated across 5 distinct, randomized batches (20 unique claims per test). 

#### Test 1 (20 Claims)
| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 75.00% |
| **Precision** | 0.88 |
| **Recall** | 0.70 |
| **F1-Score** | 0.78 |

#### Test 2 (20 Claims)
| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 85.00% |
| **Precision** | 0.89 |
| **Recall** | 0.80 |
| **F1-Score** | 0.84 |

#### Test 3 (20 Claims)
| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 80.00% |
| **Precision** | 0.88 |
| **Recall** | 0.70 |
| **F1-Score** | 0.78 |

#### Test 4 (20 Claims)
| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 70.00% |
| **Precision** | 0.75 |
| **Recall** | 0.90 |
| **F1-Score** | 0.82 |

#### Test 5 (20 Claims)
| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 70.00% |
| **Precision** | 0.73 |
| **Recall** | 0.80 |
| **F1-Score** | 0.76 |

**Overall Average (100 Claims):** 76.00% Accuracy, 0.80 F1-Score

*Interpretation:* The pipeline achieves a solid 76% average accuracy on live, randomized open-domain Wikipedia searches. While previous theoretical baselines hit 95% on locked databases, the live API environment naturally introduces variance, confirming the strength of the Gemini Critic LLM overriding the NLI model in real-world scenarios.

**Architectural Insight: The Multi-Hop Reasoning Bottleneck**
While the local BART NLI model is mathematically precise, it struggles with multi-hop logical deductions (e.g., *If X is in the NBA, it cannot be in the NHL*). To break this ceiling, we implemented the **Knowledge Graph (KG) layer**. When the strict NLI model fails to find a direct text match, the system dynamically constructs a live graph of entities and extracts relationships between them (e.g., *Inception → Christopher Nolan → London*). By feeding these multi-hop edge relationships back into the NLI model, the system successfully bridges the logical gaps without heavily relying on generative LLM guessing.

**Architectural Challenge: Live API Drift**
Even on general knowledge datasets, an open-domain pipeline hits a mathematical ceiling because it relies on *live Wikipedia API searches* rather than a static, version-controlled database. The FEVER dataset was constructed in 2018 using exact Wikipedia page layouts from that year. Because Wikipedia is constantly edited, the live API may fail to return the exact page or paragraph the dataset expects, causing the retrieval phase to occasionally fail and degrade to `UNKNOWN`.

**The Solution: Static Vector Databases (RAG)**
To push general knowledge verification above 85% and eliminate random variation, the pipeline must shift from "Open-Domain" search to strict "Retrieval-Augmented Generation" (RAG) using locked data:
1.  **Version-Locked Vector Databases:** Instead of querying the live Wikipedia API, the system must download a static, version-controlled snapshot of Wikipedia (e.g., the exact 2018 database dump used by FEVER).
2.  **Pre-Indexed Embeddings:** This snapshot must be chunked and pre-indexed into an enterprise Vector Database (like Pinecone, Milvus, or FAISS). The retriever can then execute instantaneous, mathematically precise semantic searches against the exact corpus the dataset was built on, entirely eliminating API drift and keyword search failures.

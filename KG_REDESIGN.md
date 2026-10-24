# Knowledge Graph Redesign

Status: **proposed, not implemented** (2026-10-24). Two approaches are described: a Wikidata-backed display graph (§3, recommended first step, no effect on accuracy or latency) and a claim-centred LLM graph that also changes the arbiter's evidence (§4, needs a re-evaluation). Approve one before starting.

## 1. Problem

The Knowledge Graph (KG, `agents/knowledge_graph.py`, pipeline.md §2.8) is built only for escalated claims and is meant to (a) add multi-hop evidence for the LLM arbiter and (b) show the user how the claim relates to the evidence. In practice it often does neither. Example: for *"Marie Curie's favourite colour was green."* the graph had two unconnected nodes, *Ève Curie* and *List of characters in The Lego Movie franchise*, and no Marie Curie node.

Measured on the recorded traces of the 200 benchmark claims (`evaluation/*_5_runs_claims.jsonl`, no API calls), 44 claims built a graph:

| Finding | Count |
| :--- | :--- |
| Graphs with no edges | 18 of 44 (41%) |
| Graphs built from rewritten search phrases instead of entities | 20 of 44 |
| KG passages given to the arbiter from a node unrelated to the claim's words | 23 of 131 |
| Edges marked "provenance-verified" | 28 of 28 (see cause 4: the flag is not a real check) |

Examples of unrelated arbiter evidence: *Julie Benz* and *Jamie Johnson (TV series)* for *"The Gifted is Swedish."*; *Angus Sutherland (actor)* for *"Shane Black is an American producer and actor."*; a self-loop *Ian Brennan (writer) ↔ Ian Brennan (writer)*.

## 2. Causes

1. **Entities are search queries.** `hallucination_detector.py` builds the graph from `search_queries` unless they equal the claim. After a query rewrite these are phrases ("Marie Curie favorite color", "Marie Curie wedding dress green"), not Wikipedia titles.
2. **Loose page resolution.** `KnowledgeGraph.build` matches cached pages by substring in either direction and takes the first hit; `_fetch_wikipedia_summary` falls back to `wikipedia.search` and picks the candidate most similar to the claim, with no minimum similarity, so a phrase with no page resolves to an arbitrary article.
3. **The claim is not in the graph.** Edges are extracted only *between* fetched pages. A subject–attribute claim (a colour, a date, a nationality) has a single entity, so `build` returns before extracting any relation.
4. **`verified` is not a verification.** `_extract_batch_relationships` and `_extract_direct_relationship_from_evidence` set `"verified": is_verified or len(quote) >= 5`, so any quote of five or more characters counts, and the UI legend "provenance-verified relation" overstates the check.
5. **No relevance threshold.** `get_multi_hop_evidence` always returns the top 3 chunks by cosine similarity, however low, from whatever pages became nodes.

## 3. Approach A (recommended first): Wikidata-backed display graph

Take the displayed graph from **Wikidata**, the structured knowledge graph behind Wikipedia, instead of asking the LLM to extract one. Every node is a real entity with a Wikidata ID, every edge a real Wikidata statement with references. It needs no LLM call and no API key.

### 3.1 Measurements (2026-10-24, from the development machine)

| Request | Time | Content |
| :--- | :--- | :--- |
| `wbgetentities` by English Wikipedia title (Marie Curie, Albert Einstein, Hikaru Nakamura, James Webb Space Telescope) | 1.0–2.0 s, 140–450 KB | 56–453 properties, 157–667 statements per entity; e.g. Einstein's date of birth 1879-03-14 with 23 references |
| SPARQL query for ~25 selected properties of two entities, with labels | 1.1–4.6 s, 62 KB | 131 labelled facts, e.g. *award received → Nobel Prize in Physics* |

`wbgetentities` is faster and more stable; value IDs (e.g. `Q3012`) are labelled with one extra `wbgetentities&props=labels` call for the few values shown.

### 3.2 Design

**Entity resolution.** The claim's entities come from query generation (the fast-path entities, or the LLM path's entity queries before any rewrite). Each is looked up with `wbgetentities?sites=enwiki&titles=<title>`, an exact sitelink match with no guessing; if that fails, one `wbsearchentities` call, accepted only if the label shares a content word with the entity name. Unresolved entities are left out rather than replaced by a near match.

**Claim edge.** The claim's relation is matched to a Wikidata property by embedding similarity (all-MiniLM-L6-v2, already loaded) between the claim and the labels of a curated list of about 40 common properties (date of birth/death, place of birth, spouse, award received, country of citizenship, occupation, educated at, employer, founded by, inception, headquarters location, director, author, publication date, member of, …), with a minimum similarity. The claim edge is drawn from the subject to the claimed value and coloured by the final verdict, next to what Wikidata records for that property:

| Claim | Graph shows |
| :--- | :--- |
| *Albert Einstein was born in 1950.* | claimed **1950** vs. recorded *date of birth* **14 March 1879** (mismatch) |
| *Albert Einstein won the Nobel Prize in Literature in 1980.* | recorded *award received* → **Nobel Prize in Physics** (near miss) |
| *Marie Curie's favourite colour was green.* | no matching property: "Wikidata has no record of this", with her birth date, spouse and awards as context |

**Context edges.** The 6–8 statements of the claim's entities most similar to the claim (embedding similarity of "property: value" to the claim), plus any statement that links two of the claim's entities directly. Each edge shows its property, value and reference count; value nodes (dates, numbers, strings) are drawn as rounded rectangles.

### 3.3 Latency

- The lookup starts in a background thread as soon as query generation has named the entities, so it runs in parallel with retrieval and NLI scoring, which take several seconds per claim.
- The verdict never waits for it. If it finishes after the verdict, the graph is sent afterwards through the existing deferred `claim_graph` event; per-claim time to verdict is unchanged.
- No LLM calls and no Gemini quota; the property matching uses the local embedding model (milliseconds).
- Because it is cheap, every claim gets a graph, not only escalated ones.

### 3.4 Accuracy

**No effect.** The graph is display-only: no stage that decides a verdict (retrieval, NLI, conformal set, LLM review or arbiter) reads it, and it adds no LLM call. The benchmark results in pipeline.md stay valid. To keep this true, the current LLM-built KG is still built for escalated claims and still given to the arbiter (its weaknesses in §1 remain inside the pipeline until Approach B), but it is no longer shown.

### 3.5 Limits

- Wikidata can be incomplete or out of date, which matters most for FreshQA's time-sensitive claims; the graph then shows what Wikidata records, labelled as such.
- It has no properties for many claims (favourite colours, opinions, "became World Chess Champion"); the graph then says there is no matching record.
- Property matching can miss or pick a near property; a minimum similarity keeps a miss as "no matching record" rather than a wrong match.
- It depends on the network; if Wikidata is unavailable, the claim has no graph and the verdict is unaffected. Recorded runs keep the graph.
- Requests must send a descriptive User-Agent (Wikidata policy); results are cached per entity for the server's lifetime.

### 3.6 Code changes

| File | Change |
| :--- | :--- |
| `agents/wikidata_graph.py` (new) | Entity resolution, statement fetch, value labelling, property matching, fact selection, per-entity cache |
| `detector/hallucination_detector.py` | Start the lookup after query generation; attach the graph to the result (or send it as `claim_graph` after the verdict); stop sending the LLM KG to the UI |
| `static/app.js`, `static/style.css` | One graph per claim in its card and in the Under-the-hood KG step; claim edge in the verdict colour with the recorded value; value nodes; legend "claimed / Wikidata statement" |
| `pipeline.md` | §2.8 and §2.10: the displayed graph comes from Wikidata and is display-only |
| `tests/` | Exact sitelink resolution; unrelated search hits rejected; property matching on the three examples above; missing-property case |

### 3.7 Verification

- The five demo claims show the graphs of §3.2 (Einstein birth year and Nobel Prize, Curie with no matching record, Nakamura's chess facts, JWST, Eiffel Tower), with no unrelated nodes.
- Offline over the 200 benchmark claims (their recorded queries): share of claims with a resolved entity, with a matched property, and with a graph; manual check of 20 graphs.
- Time to verdict on a few live runs is unchanged; recorded verdicts are identical with the graph on and off.
- Cost: no Gemini calls for the change itself; re-recording the five demos takes about 25 calls.

## 4. Approach B (later): claim-centred graph as arbiter evidence

Fixes the KG inside the pipeline, so that the arbiter stops reading unrelated passages. It changes verdicts on escalated claims and needs the re-evaluation in §4.3.

### 4.1 Design

**Nodes.** The claim's subject and object; objects that are values are value nodes. Further nodes only from verified evidence facts.

**Claim edge.** The claim as one or more triples (subject, relation, object), coloured by the final verdict; negations kept in the relation.

**Evidence facts.** Up to 6 facts from the retrieved evidence about the claim's entities, preferring the claim's own relation (for *"born in 1950"*: the actual birth date). Each fact carries a quote and is **verified** only if the normalised quote is a substring of the evidence or the node's summary. Unverified facts are never given to the arbiter.

**One LLM call** (light tier) replaces `extract_entities` and `_extract_batch_relationships` (and their fallbacks):

```json
{
  "claim_triples": [{"subject": "Marie Curie", "relation": "favourite colour", "object": "green", "object_type": "value"}],
  "entities": [{"name": "Marie Curie", "wikipedia_title": "Marie Curie"}],
  "evidence_facts": [{"subject": "Marie Curie", "relation": "born", "object": "7 November 1867", "quote": "born 7 November 1867"}]
}
```

**Page resolution.** Exact title first; a search candidate only if its title shares a content word with the entity and its similarity to the claim exceeds a threshold (start at 0.35); exact normalised title for cached pages; no self-loops.

**Arbiter evidence.** Verified facts as `KG Fact [subject → relation → object]: "quote"`, plus summary chunks of the claim's own entity pages above a similarity threshold. Wikidata statements from Approach A can be added here as structured evidence.

**Cost.** One light LLM call per graph instead of one to three today. If Wikidata statements replace the LLM extraction entirely, escalated claims get faster.

### 4.2 Code changes

| File | Change |
| :--- | :--- |
| `agents/knowledge_graph.py` | `extract_claim_graph(claim, evidence)`; strict page resolution; real provenance check (drop `or len(quote) >= 5`); thresholded `get_multi_hop_evidence` over verified facts and own-entity chunks |
| `detector/hallucination_detector.py` | Build from the claim, not `search_queries`; drop `_display_edge_task`; record triples and fact checks in the `kg` trace step |
| `agents/entity_linker.py` | No longer used by the pipeline |
| `pipeline.md` | §2.8 and §3 results after the re-evaluation |

### 4.3 Re-evaluation

Re-run only the claims whose route is `multi_agent_debate` (44 across FEVER and FreshQA) with `evaluation/run_5_runs.py`, keep the other 156 verdicts, and report before/after accuracy on those 44 and the pooled change; inspect every changed verdict in the claim explorer. Estimated cost: about 150–200 Gemini calls, plus about 25 to re-record the demos. Accuracy may rise on dates, places and awards (stated exactly in structured facts, weak spots of the NLI model) and fall on time-sensitive FreshQA claims where Wikidata is out of date; only the measurement decides.

## 5. Recommended order

1. **Approach A** now or when approved: fixes what users and reviewers see, with no accuracy or latency change and no Gemini cost.
2. Fix the `verified` flag (cause 4) together with A, since the legend is shown to users.
3. **Approach B** when the quota allows the §4.3 re-evaluation.

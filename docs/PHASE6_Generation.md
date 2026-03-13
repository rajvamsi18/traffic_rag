# Phase 6: Answer Generation

**Status:** ✅ Complete  
**Scripts:** `src/generator.py`, `src/main.py`  
**Input:** `RetrievalResult` objects from Phase 5 retriever  
**Output:** Natural language answers grounded in retrieved survey data

---

## What This Phase Does

Phase 6 is the final layer of the pipeline. It takes the chunks retrieved by Phase 5, assembles them into a structured prompt, sends that prompt to a language model, and returns a factual answer grounded strictly in the retrieved context.

The flow:

```
User question
      ↓
Phase 5 retriever → relevant chunks (text + metadata)
      ↓
generator.py → builds prompt with context + question
      ↓
LLM API (Groq / llama-3.3-70b-versatile)
      ↓
Grounded natural language answer
```

---

## The Most Important Design Decision — Honest Behaviour

The biggest failure mode in any RAG system is hallucination — the LLM confidently answering from its own training knowledge rather than the retrieved context. In a traffic data system, hallucination is particularly dangerous because the model might know general facts about Indian roads or Andhra Pradesh geography and silently mix them into answers without the user knowing.

The system prompt addresses this with seven explicit rules, each targeting a specific failure mode:

1. **Answer only from provided context** — prevents using outside knowledge
2. **Refuse clearly when context is insufficient** — prevents confident wrong answers
3. **Always cite location ID and road name** — makes every number traceable
4. **Only compare locations present in context** — prevents false cross-location comparisons
5. **Use retrieval notes when present** — helps the model explain why a location was selected
6. **Keep answers concise and factual** — prevents padding and speculation
7. **Structure multi-location answers clearly** — prevents garbled comparison responses

Temperature is set to `0.1` — the lowest useful setting. Higher temperature makes models more creative and varied, which is useful for writing tasks but harmful for factual data retrieval where we want the same correct answer every time.

---

## Why Groq Instead of Gemini

The original implementation used Gemini 2.0 Flash Lite. During Phase 6 testing, Gemini's free tier daily quota was exhausted after approximately 24 API calls across three debugging runs (8 queries × 3 attempts). Even after generating a new API key, the per-minute rate limit prevented reliable testing.

Groq was chosen as a replacement for three reasons. First, it offers `llama-3.3-70b-versatile` on a free tier with 30 requests per minute and no meaningful daily cap — significantly more generous than Gemini's free offering. Second, Groq uses the OpenAI-compatible chat completions format, which meant the code change was minimal — a different URL, key name, model name, and request structure. The system prompt, prompt builder, retry logic, and all other logic remained identical. Third, `llama-3.3-70b-versatile` is a strong model for factual question answering on structured data — the answer quality across all 8 test queries was accurate and well-structured.

The change required updating approximately 10 lines in `generator.py`. This is worth noting as an architectural point — the generator was designed with the LLM backend as a swappable component from the start, which is why the swap was painless.

The Gemini key and model configuration remain in `.env` and can be restored by changing three constants in `generator.py` if needed.

---

## Prompt Structure

Each prompt sent to the LLM has three clearly separated sections:

**System message** — passed as the `system` role in the OpenAI-compatible format. Contains the identity ("traffic data analyst for Guntur District"), the seven rules, and background facts about the dataset (84 locations, April 2017, SCF-adjusted AADT values).

**Context block** — each retrieved chunk is labelled with its location ID, chunk type, and road name before the text. For example:

```
[P526 — TRAFFIC — Hyderabad to Guntur Road]
Traffic Volume and Composition at Location P526...
```

This labelling matters because without it the LLM treats all context as one undifferentiated block and may attribute numbers from one location to another when multiple locations are present.

**Question + instruction** — the user's original question followed by an explicit reminder to answer from context only. Repeating the instruction at the end of the prompt (rather than only in the system message) has been shown to reduce the rate of instruction drift in long prompts.

---

## Test Results — All 8 Queries

After fixing the LLM backend and the routing bug described below, all 8 test queries were run and reviewed:

**Query 1 — "Which road has the highest truck traffic?"**  
Returned: P526, Hyderabad to Guntur Road, 1,179 trucks/day ✅  
Numbers verified against `P526.json` (truck_2axle + truck_3axle + mav = 1,179).

**Query 2 — "Which location has the lowest AADT?"**  
Returned: P605, Pakalapadu to Paladugu Via Abburu, 63 vehicles/day ✅  
Answer included the retrieval note explaining why P605 was selected.

**Query 3 — "Which road has the most two-wheelers?"**  
Returned: P528, Guntur to Bapatla to Chirala Road, 7,093 two-wheelers/day ✅

**Query 4 — "Tell me about location P606"**  
Returned a coherent paragraph synthesising all four chunk types — overview, traffic, directional, and peak — without being explicitly instructed to. Location, chainage, survey date, AADT, PCU, directional balance, and peak hour all present and accurate. ✅

**Query 5 — "What is the peak hour at Nadikudi?"**  
Returned: P526 (Nadikudi is the survey location name for P526), peak hour 18:00–19:00, 443 vehicles ✅

**Query 6 — "Compare P526 and P538"**  
Returned a structured comparison covering AADT, two-wheeler percentage, PCU, and breakdown of top goods vehicles for both locations. Concluded correctly that P538 has higher volumes. ✅

**Query 7 — "Which roads were surveyed near Tadikonda?"**  
Returned an honest refusal: *"The available survey data does not contain enough information to answer this question. The context only mentions a survey location near Sowpadu."* ✅  
The retriever returned P541 (near Sowpadu, adjacent to Tadikonda). The LLM correctly refused rather than claiming P541 answers the question. This is the known geographic retriever limitation documented in Phase 5 — a single-match name resolution rather than a geographic area search. The LLM behaved correctly given an imperfect retrieval.

**Query 8 — "What is the busiest time of day on the Hyderabad to Guntur Road?"**  
*Before fix:* Retrieved P528 (wrong location — highest AADT, not the Hyderabad to Guntur Road). LLM correctly refused. ❌  
*After fix:* Retrieved P526 correctly (see routing bug fix below). Answer: evening peak 18:00–19:00, 443 vehicles ✅

---

## Bug Found and Fixed — Routing Priority (Two Iterations)

### Iteration 1 — Routing order

**Symptom:** Query 8, "What is the busiest time of day on the Hyderabad to Guntur Road?", was routed to the superlative strategy and returned P528 — the highest-AADT location in the dataset. The LLM correctly refused to answer because the retrieved context was about the wrong road entirely.

**Root cause:** The `retrieve()` method called `classify_query()` first. The word "busiest" matched the superlative AADT pattern before name matching ever ran. "Hyderabad to Guntur Road" was never resolved to P526.

**Fix:** Moved name matching to run *before* `classify_query()` inside `retrieve()`. New priority order:

```
1. Named location check (road/village name) → location_specific
2. Superlative pattern                       → superlative
3. Location code (P526)                      → location_specific
4. Two location codes                        → comparison
5. None of the above                         → semantic search
```

---

### Iteration 2 — Name matcher false positives

**Symptom:** After moving name matching first, queries 1, 3 and 8 all returned P541 — a completely unrelated location. The strategy showed as `location_specific` for queries that should have been `superlative`.

**Root cause:** Two bugs in `_find_location_by_name()` that were harmless when it ran as a fallback but caused widespread false positives when it ran first:

**Bug A — Duplicate words counted multiple times.** P541's road name is "Road frm 5 200 of OM road to 8 20 OM road" — the word "road" appears three times. The matcher split the name into a list without deduplicating, so any query containing "road" scored 3 matches and incorrectly triggered P541. Fix: deduplicate with `set()` before counting.

**Bug B — Short common words treated as meaningful signals.** The word length threshold was `len(w) > 3`, meaning 4+ character words were considered significant. "road" is 4 characters and appears in virtually every road name and every query about roads. Fix: raised threshold to `len(w) >= 5`, which excludes "road", "from", "near", and other short common words while keeping genuinely meaningful location words like "Guntur" (6), "Hyderabad" (9), "Chirala" (7), "Bapatla" (7).

**Lesson:** The name matcher had subtle bugs masked by its original position in the routing chain. When it ran as a last-resort fallback, the bugs rarely triggered because most queries were caught by earlier checks first. Moving it to run first exposed every edge case at once. This is a common pattern in software — changing the order of operations can reveal latent bugs in components that appeared correct in their original context.

---

## Test Results — All 8 Queries (Final)

After both routing fixes, all 8 test queries returned correct answers:

**Query 1 — "Which road has the highest truck traffic?"**
Strategy: superlative | Retrieved: P526 ✅
The Hyderabad to Guntur Road, 1,179 trucks/day. Verified against `P526.json` (truck_2axle + truck_3axle + mav = 1,179).

**Query 2 — "Which location has the lowest AADT?"**
Strategy: superlative | Retrieved: P605 ✅
Pakalapadu to Paladugu Via Abburu, 63 vehicles/day. Answer included retrieval note.

**Query 3 — "Which road has the most two-wheelers?"**
Strategy: superlative | Retrieved: P528 ✅
Guntur to Bapatla to Chirala Road, 7,093 two-wheelers/day.

**Query 4 — "Tell me about survey location P606"**
Strategy: location_specific | Retrieved: P606, all 4 chunks ✅
Synthesised overview, traffic, directional and peak data into a coherent answer. Location, chainage, survey date, AADT, PCU, directional balance, peak hour all accurate.

**Query 5 — "What is the peak hour at Nadikudi?"**
Strategy: location_specific | Retrieved: P526 ✅
18:00–19:00, 443 vehicles. Correctly identified P526 as the Nadikudi survey point.

**Query 6 — "Compare P526 and P538"**
Strategy: comparison | Retrieved: P526, P538 ✅
Structured comparison covering AADT, two-wheeler percentage, traffic composition breakdown, and PCU for both locations. Concluded correctly that P538 has higher overall volumes.

**Query 7 — "Which roads were surveyed near Tadikonda?"**
Strategy: semantic | Retrieved: P577, P-562 ✅
Returned two genuinely relevant locations — P577 (Tadikoda to Meet NH 5 via Kantheru) and P-562 (Tadikonda to Rayapudi road). This is an improvement over earlier behaviour where a single mismatched location was returned. Moving name matching earlier freed the semantic search path to handle area-based geographic queries correctly.

**Query 8 — "What is the busiest time of day on the Hyderabad to Guntur Road?"**
Strategy: location_specific | Retrieved: P526 ✅
"Hyderabad to Guntur Road" correctly resolved to P526 via name matching. Answer: 18:00–19:00, 443 vehicles.

---

## Files in This Phase

`src/generator.py` — system prompt, prompt builder, Groq API call with retry logic, `generate_answer()` function that returns a structured response dict.

`src/main.py` — interactive pipeline entry point. Ties retriever and generator together. Supports `--test` flag (runs 8 built-in queries with a 5-second pause between each to stay within rate limits) and `--debug` flag (prints the full prompt alongside each answer for inspection).

---

## Changes Made to Earlier Files

### `src/retriever.py` — routing order and name matcher fixed (two changes)

This is the third time `retriever.py` has been updated after testing found a problem. Phase 5 fixed three bugs in pattern matching. Phase 6 found two more — both related to the name matcher — documented fully in the bug section above.

**Change 1:** The `retrieve()` method was updated to run name matching before `classify_query()`. The docstring explains the priority order and why it exists.

**Change 2:** `_find_location_by_name()` was updated with two fixes — word deduplication using `set()`, and raising the minimum word length from 4 to 5 characters to exclude short common words like "road" from triggering false matches.

Both changes are in `src/retriever.py` only. No other files were touched.

---

*Next: Phase 7 — Evaluation (RAGAS)*  
*The system is now functionally complete and producing correct, grounded answers across all four query types. Phase 7 builds a formal evaluation dataset of 15 question-answer pairs covering each query type and runs RAGAS metrics — faithfulness, answer relevancy, and context precision — to produce measurable, repeatable quality scores rather than relying on manual eyeballing.*
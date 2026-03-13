# Phase 5: Hybrid Retriever

**Status:** ✅ Complete  
**Script:** `src/retriever.py`  
**Input:** ChromaDB collection (`vectorstore/`) + JSON files (`data/processed/`)  
**Output:** `RetrievalResult` objects passed to the LLM in Phase 6

---

## What This Phase Does

Phase 5 sits between the user's query and ChromaDB. Rather than sending every query straight to vector search, it first figures out what *kind* of question is being asked, then routes it to whichever retrieval strategy will actually answer it correctly.

The need for this layer came directly from the Phase 4 verification failure. Pure semantic search returned wrong answers for two out of three test queries — not because ChromaDB was broken, but because semantic search is the wrong tool for certain question types. Phase 5 is the fix.

---

## Why a Retriever Layer Was Necessary

The Phase 4 finding was specific: semantic search failed on superlative queries ("which road has the most trucks?") because every traffic chunk is about trucks, roads, and traffic. The embedding model produces nearly identical vectors for all 84 traffic chunks because they're all structured the same way — only the numbers differ, and vector embeddings don't understand numerical magnitude.

I tried adding ranking sentences to the chunks ("P526 ranks 1st for truck traffic") hoping it would differentiate the vectors. It didn't — a single sentence at the end of a 300-word chunk contributes roughly 1/20th of the final averaged vector. The other 19/20ths, which are structurally identical across all chunks, swamp the signal entirely.

The correct fix isn't at the embedding layer — it's at the query handling layer. Before touching ChromaDB at all, classify what the user is asking and use the right tool for that question type.

---

## Four Query Types and How Each Is Handled

### Type 1 — Superlative Queries
*"Which road has the highest truck traffic?" / "Location with most two-wheelers?" / "Lowest AADT?"*

These questions ask for a ranking across all 84 locations. Vector search cannot answer them because it finds semantic similarity, not numerical order. The solution is to skip ChromaDB entirely and sort the JSON data directly.

The retriever maintains all 84 JSON records in memory. For a superlative query, it extracts the relevant metric from each record, sorts descending (or ascending for "lowest"), and returns the top location's traffic and overview chunks. The similarity score is set to `1.000` — not because it's a confidence score, but because this is a deterministic sort with no approximation involved.

### Type 2 — Location Specific Queries
*"Tell me about P606" / "What is the peak hour at Nadikudi?"*

When a specific location is mentioned — either by code (P606) or by name (Nadikudi) — semantic search isn't needed. The retriever fetches all four chunk types for that location directly from ChromaDB by ID. This is O(1): no embedding, no distance computation.

Location codes (P526, P-536) are detected with a regex pattern. Location names are handled by a two-tier name matcher described in the bugs section below.

### Type 3 — Comparison Queries
*"Compare P526 and P538" / "P526 vs P538"*

Both location codes are extracted from the query, and traffic + overview chunks are fetched directly for each. The LLM in Phase 6 receives both locations' data and handles the comparison in its response.

### Type 4 — General Semantic Queries
*"Which roads were surveyed near Guntur?" / "Roads with high goods vehicle percentage?"*

Broad questions without a specific location or superlative — these go to ChromaDB semantic search, returning the top 3 most relevant chunks. This is the only query type where vector similarity scores are meaningful.

---

## Implementation: Query Classification

The classifier runs before any retrieval happens. It checks patterns in this order:

**Step 1 — Superlative check:** Tests the query against a list of regex patterns. Each pattern maps a combination of trigger words ("highest", "most", "lowest", "busiest") and vehicle/metric terms to a specific metric key.

**Step 2 — Location code check:** Scans for P-number patterns (P526, P-536, p526). If two are found, it's a comparison. If one is found, it's location-specific.

**Step 3 — Location name check:** If no code was found, the name matcher scans all 84 records for a matching location or road name in the query. If found, treated as location-specific.

**Step 4 — General fallback:** If none of the above matched, the query goes to semantic search.

---

## Bugs Found and Fixed During Testing

### Bug 1 — Pattern ordering caused "highest truck traffic" to match AADT

**Symptom:** The query "which road has the highest truck traffic" was classified as `superlative:highest total AADT` instead of `superlative:highest truck traffic`. It retrieved P528 (highest AADT) instead of P526 (highest truck count).

**Root cause:** The AADT pattern was listed first in `SUPERLATIVE_PATTERNS`:
```
"highest ... traffic" → AADT
"highest ... truck"   → trucks
```
Python checks patterns in order and stops at the first match. The query "highest truck *traffic*" matched the AADT pattern on the word "traffic" before ever reaching the trucks pattern.

**Fix:** Reordered patterns so specific vehicle types (trucks, two-wheelers, goods, buses) always come before the generic AADT pattern. The AADT pattern was also narrowed — it now only matches `"total traffic"` or `"total vehicles"`, not bare "traffic", to prevent future collisions with vehicle-specific queries.

**Lesson:** When writing a pattern matcher with ordered rules, the most specific patterns must always come before the most general ones. The same rule applies to `if/elif` chains, routing tables, and URL matchers — specificity first.

---

### Bug 2 — "two wheelers" regex didn't match "two wheelers"

**Symptom:** The query "location with most two wheelers" was routed to semantic search instead of being classified as a superlative query. It retrieved P590 (287 two-wheelers/day) instead of P528 (7,093/day).

**Root cause:** The regex pattern was `two.?wheel\b` — the `\b` is a word boundary anchor, which matches at the transition between a word character and a non-word character. In the word "wheelers", the characters "ers" immediately follow "wheel" — there is no word boundary after "wheel". So the pattern never matched.

**Fix:** Changed to `two.?wheelers?` which explicitly allows the optional "ers" ending, matching both "wheeler" and "wheelers".

**Lesson:** Word boundary anchors are precise. `wheel\b` matches "wheel" and "wheel." but not "wheelers". When writing patterns for natural language input, always test plurals, verb forms, and compound words explicitly.

---

### Bug 3 — Single-word location names didn't trigger location-specific routing

**Symptom:** The query "what is the peak hour at Nadikudi" fell through to semantic search, which returned P556 — a completely different location with no connection to Nadikudi.

**Root cause:** The original name matcher required at least 2 significant words from a candidate to appear in the query:
```python
matches = sum(1 for w in words if w in q)
if matches >= 2:  # never true for single-word names
```
"Nadikudi" is one word. No matter how clearly it appeared in the query, it could never reach the threshold of 2.

The threshold existed for a good reason — road names like "Hyderabad to Guntur Road" contain common words like "road" and "to" that would create false positives if matched individually. But applying the same threshold to location names, which are short and unique, was wrong.

**Fix:** Two-tier matching strategy:
- **Location names** (e.g. "Nadikudi", "Nallapadu") — single-word exact substring match is sufficient, because village names are specific enough to be unambiguous. Returns immediately on first match.
- **Road and direction names** — still requires 2 or more significant words, because road names contain common words that would produce false positives on a single match.

---

## Verification — Why a Separate Script Wasn't Needed

After Phase 4, I wrote `verify_retrieval.py` to independently compute ground truth and compare it against what ChromaDB returned. That was necessary because semantic search produces approximations — a confidence score doesn't tell you whether the answer is actually correct.

Phase 5 is different. Three of the four strategies are deterministic lookups:

**Superlative** — sorts 84 numbers and returns the largest. The correctness is mathematical. I can verify P526 has the highest truck AADT by adding `truck_2axle + truck_3axle + mav` for every location and confirming P526's total (1,179) is the highest. No approximation involved.

**Location specific and comparison** — fetches chunks by exact ID. Either the chunk exists in ChromaDB or it doesn't. There is no "close enough" here.

Only semantic search (Type 4) still carries approximation risk. That risk is handled in Phase 6 — the LLM reads the retrieved text and determines whether it actually answers the question. If the retrieved chunks are irrelevant, the generator should say so rather than hallucinating an answer.

---

## One Gap Worth Being Honest About

The query "which roads were surveyed near Tadikonda" was classified as `location_specific` and returned P541 with score 1.000. But is P541 *actually* the only location near Tadikonda? Or are there several, and the retriever only returned one?

The name matcher finds the first (or best) location whose data contains the query term, then stops. For a geographic proximity query, the right behaviour is to return *all* matching locations — not just one. This is a known limitation of the current implementation.

The proper fix would be a geographic query handler that searches all 84 records for any whose location name, road name, or direction name contains the queried place name, and returns all of them. This would make the retriever genuinely useful for area-based questions. It's feasible with the current data structure and is a natural extension for a future iteration.

For Phase 6, the generator will include a note in its response when only one result was returned for a geographic query, signalling to the user that there may be other relevant locations.

---

## Final Test Results

After fixing the three bugs, all seven test queries passed:

```
"which road has the highest truck traffic"   → P526  ✅  superlative
"location with most two wheelers"            → P528  ✅  superlative
"which location has the lowest aadt"         → P605  ✅  superlative
"what is the peak hour at Nadikudi"          → P526  ✅  location_specific
"tell me about location P606"                → P606  ✅  location_specific
"compare P526 and P538"                      → P526, P538  ✅  comparison
"which roads were surveyed near Tadikonda"   → P541  ✅  location_specific
```

Every superlative and location query returns score `1.000` — not a similarity estimate but a deterministic result. Only general semantic queries return fractional scores, which reflect genuine approximation.

---

## Files in This Phase

`src/retriever.py` — the hybrid retriever. Contains the query classifier, all four retrieval strategies, the superlative pattern list, the metric extractor, and the two-tier name matcher.

---

## Changes Made to Earlier Files — and Why

Reading the phase documents in order, a question comes up naturally: if Phase 3 built `converter.py` and Phase 4 built the embedder, why are those files changing now in Phase 5? This section explains every change made to earlier files during this phase, so the timeline stays clear.

---

### `src/converter.py` — ranking infrastructure removed

**What was removed:** Three functions (`compute_rankings`, `build_ranking_sentence`, `ordinal`) and the logic that injected a ranking sentence into every traffic and peak chunk.

**Why it was added in the first place:** During Phase 4 verification, superlative queries like "which road has the most trucks?" returned wrong results. The first attempt at a fix was to make the correct chunk textually distinctive by appending a sentence like *"P526 ranks 1st for truck traffic among all 84 locations"* to the end of each traffic chunk. The idea was that a query about "highest truck traffic" would then semantically match P526's chunk specifically.

**Why it didn't work:** The `all-MiniLM-L6-v2` model produces one vector per chunk by averaging all sentence embeddings together. A single ranking sentence at the end of a 300-word chunk contributes roughly 1/20th of the final vector. The remaining 19/20ths — the vehicle counts and percentages that are structurally identical across all 84 traffic chunks — swamp the signal entirely. The similarity scores after re-embedding were bit-for-bit identical to before the change.

**Why it was removed now:** The correct fix was built in Phase 5 — superlative queries are now routed directly to JSON lookup, bypassing ChromaDB entirely. The ranking infrastructure in `converter.py` became dead code at that point. Keeping it would mean running a full district-wide sort on every converter run, bloating 168 chunk files with text that no longer serves a retrieval purpose, and slightly diluting the embedding quality of those chunks when they are retrieved via semantic search. Removing it makes the converter simpler, faster, and cleaner.

**The takeaway:** This is a normal part of iterative development. Code written as an experiment that was later superseded by a better solution at a different layer should be removed, not left in place. The experiment is documented in Phase 4 so the reasoning is preserved — the code itself doesn't need to stay.

---

### `src/diagnose.py` — updated twice, now a clean reset utility

`diagnose.py` was originally written during Phase 4 to verify that ranking sentences were correctly present in chunk files and in ChromaDB. At that point, a missing ranking sentence meant something had gone wrong — so the script checked for presence and blocked the reset if sentences were absent.

After the Phase 5 cleanup, the world changed. Ranking sentences were intentionally removed from all chunks. But `diagnose.py` still had the old logic — it checked for the presence of ranking sentences and treated their absence as a failure. This meant the script was reading the correct state of the system as a problem, blocking the ChromaDB reset, and printing misleading error messages.

The script went through two updates as a result:

**First update** — changed the ranking check to only apply to traffic and peak chunks (not overview or directional, which never had ranking sentences). This was an intermediate fix that still didn't fully resolve the issue because 168 traffic and peak chunks were now correctly missing ranking sentences and the script was still blocking on them.

**Second update** — removed the ranking check entirely. The script now simply verifies the chunk files are clean (no ranking sentences present), reports what ChromaDB currently has, and always proceeds to reset and re-embed regardless. The reset is unconditional because that is the entire purpose of the script — it exists to force a clean re-embed when the chunk content has changed.

The lesson here is the same as with `converter.py` — diagnostic scripts go stale just like production code. When the expected state of the system changes, every script that checks for that state needs updating too. A check that was correct in Phase 4 became a false negative in Phase 5, and a false negative in a diagnostic tool is particularly damaging because it erodes trust in the tooling itself.

---

*Next: Phase 6 — Generator (`src/generator.py`)*  
*The generator receives retrieved chunks from Phase 5 and sends them to the Gemini API along with the user's question. It constructs the prompt, handles the API response, and enforces honest behaviour — telling the user when retrieved context is insufficient rather than fabricating an answer.*
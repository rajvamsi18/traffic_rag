# Phase 4: Embeddings and Vector Database

**Status:** ✅ Complete  
**Script:** `src/embedder.py`  
**Verification:** `src/verify_retrieval.py`  
**Input:** `text_summaries/` — 336 text chunks  
**Output:** `vectorstore/` — ChromaDB persistent collection (`traffic_guntur`)

---

## What This Phase Does

Phase 4 is where the project stops being a data pipeline and starts becoming a search system.

The 336 text chunks from Phase 3 are readable by a human but meaningless to a computer. A computer can't answer "which road has the most trucks?" by reading text — it has no concept of meaning. What it can do is compare numbers. Phase 4 converts every chunk into a list of 384 numbers — a vector — that mathematically represents its meaning. Chunks that discuss similar topics end up with similar vectors. Chunks about completely different things end up with very different vectors.

Once all 336 chunks are stored as vectors in ChromaDB, a search query goes through the same conversion process and the database finds whichever vectors are closest to it in mathematical space. That closeness is what "semantic similarity" means in practice.

---

## Why These Tools

**sentence-transformers (`all-MiniLM-L6-v2`)** for generating the vectors. The model runs entirely on my local machine — no API calls, no cost, no internet dependency once it downloads the first time (~90MB cached to the HuggingFace cache folder). It produces 384-dimensional vectors, which is compact enough to be fast but expressive enough to capture the kind of factual, domain-specific meaning present in traffic survey text. It embedded all 336 chunks in 4.2 seconds on an M3 Mac.

The alternative would have been OpenAI's `text-embedding-ada-002` — better benchmark scores, but it costs money per API call and requires sending all the text to an external server. For a project of this size, the quality difference doesn't justify the cost or the external dependency.

**ChromaDB** for storing and searching the vectors. It runs locally as a Python library with no server to set up, persists to disk automatically at `vectorstore/`, and stores the vector, the original text, and metadata all together in one place. When a chunk is retrieved, you get back the text and the metadata (location_id, chunk_type, road_name) in the same call — no secondary lookups needed.

---

## What Gets Stored Per Chunk

For each of the 336 chunks, ChromaDB stores three things together:

The **embedding vector** — 384 floating point numbers representing the meaning of the chunk. This is what gets searched.

The **original text** — the full chunk as written by converter.py. This is what gets passed to the LLM in Phase 6 so it can read the actual content and formulate an answer.

The **metadata** — a small dictionary containing `location_id`, `chunk_type`, and `road_name`. This is what Phase 5 uses to filter results. For example, if a query specifically asks about peak hours, the retriever can request only `chunk_type == "peak"` chunks rather than searching all 336.

---

## Implementation Details

**Batching:** The model encodes chunks in batches of 64 rather than one at a time. This is more efficient because the model can process multiple texts in parallel on the hardware. With 336 chunks at 64 per batch, that's 6 batches total.

**Upsert instead of insert:** ChromaDB's `upsert` operation adds a vector if its ID doesn't exist, or updates it if it does. This means the script is safe to re-run at any point — it won't create duplicate vectors. If you change a chunk's text and re-run, the old vector gets replaced with the new one.

**Cosine similarity:** The collection is configured to use cosine similarity as its distance metric. For text embeddings this is the right choice — it measures the angle between two vectors rather than the raw distance, which means the similarity score is not affected by how long or short a chunk is.

---

## Verification — How I Checked the Retrieval Was Correct

The embedder's built-in sanity check only shows what was retrieved — it doesn't tell you whether that result is actually right. To verify properly, I wrote a separate `verify_retrieval.py` script that runs each query through ChromaDB and independently computes the ground truth directly from the JSON files, then compares the two.

For the truck traffic query, it scans all 84 JSON files and ranks locations by actual truck AADT (2-axle + 3-axle + MAV combined), then checks whether the retrieved location is genuinely the highest. For the Nadikudi query, it searches all JSON files for any location with "Nadikudi" in its name, road name, or direction names. For the two-wheelers query, it ranks all 84 locations by two-wheeler AADT and checks where the retrieved location falls.

This matters because a retrieval system can return a result with confidence and still be wrong. The similarity score only tells you how close the query and chunk are in vector space — it doesn't tell you whether the chunk contains the right answer. Ground truth verification is the only way to know for sure.

---

## Results

```
Total vectors stored : 336
Collection name      : traffic_guntur
Similarity metric    : cosine
Embedding dimensions : 384

Vectors by chunk type:
  overview        : 84
  traffic         : 84
  directional     : 84
  peak            : 84

Embedding time      : 4.2 seconds
```

---

## Verification Results and What They Revealed

Running `verify_retrieval.py` against the three test queries produced a result I didn't expect:

```
Query 1: "which road has the highest truck traffic"   ❌ INCORRECT
Query 2: "what is the peak hour at Nadikudi"          ✅ CORRECT
Query 3: "location with most two wheelers"            ❌ INCORRECT
```

For Query 1, the system retrieved P565 with zero truck traffic. The actual highest-truck location, P526 (1,179 trucks/day), wasn't even in the top 3. For Query 3, it retrieved P590 with only 287 two-wheelers per day, when the correct answer P528 has 7,093. The similarity scores being identical across all runs — 0.534 and 0.487 — confirmed the system was consistently wrong, not randomly wrong.

Query 2 worked because "Nadikudi" is a rare named entity. Only two chunks in the entire corpus contain that word, so the embedding correctly identified them as the closest match. The superlative queries failed for a completely different reason.

---

## What I Tried and Why It Didn't Work

My first instinct was that the chunks didn't contain enough distinguishing text for superlative queries — if all traffic chunks mention trucks, the model has no reason to rank P526's chunk higher than P565's. So I added a ranking sentence to each traffic and peak chunk, generated by computing district-wide rankings across all 84 JSON files first. P526's chunk would end with something like:

> *"Among all 84 surveyed locations in Guntur District, P526 ranks 1st for truck traffic (1,179 trucks/day)..."*

The logic was that a query about "highest truck traffic" would now semantically match P526's chunk specifically — because only one chunk says "1st for truck traffic."

It didn't work. The similarity scores after re-embedding were bit-for-bit identical to before. The reason, once I understood it, was obvious: `all-MiniLM-L6-v2` produces one vector per chunk by averaging the embeddings of all sentences together. A 300-word traffic chunk with one ranking sentence at the end means the ranking contributes roughly 1/20th of the final vector. The other 19/20ths — the vehicle counts and percentages that are structurally identical across all 84 traffic chunks — completely swamp it. The ranking sentence is too small to meaningfully shift where the vector lands in 384-dimensional space.

I also wrote `diagnose.py` during this process to verify the ranking sentences were actually making it into the chunk files and into ChromaDB, after a confusing run where the re-embed appeared to have happened but scores didn't change. The diagnose script caught that the reset hadn't actually run due to a logic error in the check — and fixed the issue by also handling the reset automatically.

---

## The Actual Fix — Query Routing in Phase 5

Trying to solve this through chunk content was the wrong layer. Superlative queries — "highest", "most", "lowest", "busiest" — are a fundamentally different type of question from lookup queries like "what is the peak hour at Nadikudi". They don't need semantic search at all. They need structured lookup: sort all 84 locations by the relevant metric and return the top result directly from the JSON data.

The right architecture separates these two query types at the retriever level:

```
User query
    ↓
Query classifier
    ↙                         ↘
Superlative query          Specific lookup query
("highest truck traffic")  ("peak hour at Nadikudi")
    ↓                              ↓
Direct JSON ranking           Semantic search
(sort 84 locations,           (ChromaDB finds
 return #1 by metric)          relevant chunks)
```

This is handled in Phase 5. The retriever will detect signal words like "highest", "most", "lowest", "busiest", "least" and route those queries to structured JSON lookup rather than vector search. Everything else goes through ChromaDB as normal.

This is one of the most important findings in the project — pure semantic search has a known blind spot with comparison and superlative queries. Recognising it during verification, understanding why the naive fix failed, and solving it at the correct architectural layer is a more honest and instructive outcome than a system that works on cherry-picked examples.

---

## One Thing Worth Knowing About Similarity Scores

A common misunderstanding when working with vector search for the first time is treating the similarity score as a confidence percentage. A score of 0.53 doesn't mean the system is 53% sure — it means the retrieved chunk is more similar to the query than any of the other 335 chunks. Whether 0.53 is "good enough" depends entirely on whether the correct chunk is at the top, not on the absolute number.

This also means that low scores don't always indicate a problem. If all 336 chunks are equally irrelevant to a query, ChromaDB will still return the closest one — it just won't be very close to any of them. The LLM in Phase 6 needs to handle this gracefully by acknowledging it doesn't have enough information, rather than hallucinating an answer from a weakly relevant chunk. Building that behaviour in is part of Phase 6.

---

## Files Created or Modified in This Phase

`src/embedder.py` — embeds all 336 chunks into ChromaDB using `all-MiniLM-L6-v2`

`src/verify_retrieval.py` — ground truth verification script; compares ChromaDB retrieval against actual JSON data for three test queries

`src/diagnose.py` — written during debugging to verify ranking sentences were correctly present in chunk files and in ChromaDB; also handles forced re-embed when needed

`src/converter.py` (modified from Phase 3) — ranking sentence logic added during Phase 4 verification attempt; `compute_rankings()` and `build_ranking_sentence()` functions added, and ranking sentence injected into traffic and peak chunks. The modification is documented here rather than in Phase 3 because the motivation came from Phase 4 findings.

---

*Next: Phase 5 — Retriever (`src/retriever.py`)*  
*The retriever adds query classification on top of ChromaDB — routing superlative queries to direct JSON lookup and specific queries to semantic search, with top-k retrieval for multi-hop questions.*
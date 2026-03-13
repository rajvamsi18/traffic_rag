# Phase 7 — RAGAS Evaluation

## What This Phase Does

Up to Phase 6, the pipeline was verified by running 8 test queries and checking whether the answers were correct. That is enough to build confidence during development, but it does not produce a repeatable, portable measure. If I change the prompt, swap the model, or update the text chunks, I have no automated way to detect whether quality improved or degraded.

Phase 7 adds formal evaluation using RAGAS — a library that produces numeric scores for RAG pipelines. The scores are computed by an LLM judge (the same Groq model used for generation), not by rule-based matching. This makes them appropriate for a system where answers are natural language paragraphs rather than exact-match strings.

The evaluation covers 15 hand-crafted test cases across all four query types the retriever supports: superlative, location-specific by code, location-specific by name, comparison, and semantic.

---

## RAGAS Metrics Explained

RAGAS measures three things:

**Faithfulness** asks: did the answer stay grounded in the retrieved context, or did the LLM add things it made up? To score this, RAGAS breaks the answer into individual factual statements and checks whether each one can be traced back to a retrieved chunk. A score of 1.0 means every claim in the answer was directly supported by the context. A score of 0.5 means half the claims were hallucinated.

**Answer Relevancy** asks: did the answer actually address the question that was asked? A technically accurate answer about the wrong thing scores low here. RAGAS measures this by generating reverse questions from the answer and checking how well they match the original question — if the generated reverse question aligns with what was asked, the answer was relevant.

**Context Precision** asks: were the retrieved chunks relevant to the question? If the retriever pulled chunks that were not needed to answer the question, this score drops even if the final answer was good. This metric requires a reference answer (ground truth) to judge whether each retrieved chunk contributed to producing the correct response.

---

## Results

| Metric | Score | Interpretation |
|---|---|---|
| Faithfulness | **0.941** | 94% of answer claims are grounded in retrieved context |
| Answer Relevancy | **0.893** | 89% of answers directly address the question asked |
| Context Precision | N/A | See note below |
| **Overall** | **0.917** | Average of the two scoreable metrics |

### Context Precision — Why It Returned N/A

Context Precision returned NaN for this evaluation run. This is a known limitation of the metric when ground truth references are brief.

The metric works by having an LLM judge read each retrieved chunk and the reference answer, and decide whether that chunk was necessary to produce the correct response. Our reference strings were deliberately concise — for example: "P605 has the lowest AADT with 63 vehicles per day." But our retrieved chunks are dense multi-paragraph summaries covering AADT, vehicle composition, peak hours, directional splits, and more. The LLM judge could not consistently determine which specific chunk "caused" the brief reference answer when the context contained far more information than the reference described.

This is a failure of the metric configuration, not the pipeline. The retriever is demonstrably returning correct chunks — every query in the per-query results shows the right location being retrieved. For a future run, Context Precision could be scored by expanding the reference strings to match the depth of the retrieved chunks, or by using a different metric like Context Recall which is less sensitive to this mismatch.

---

## Per-Query Results

All 15 queries returned correct answers with the right location IDs. Below is a summary by type.

**Superlative (5/5 correct)**
All five superlative queries correctly identified the right location by sorting the full 84-record JSON dataset directly, bypassing ChromaDB vector search. The answers are factually precise: P526 for highest truck traffic (1,179/day), P605 for lowest AADT (63/day), P528 for most two-wheelers (7,093/day) and highest total AADT (12,669/day), and P526 for highest goods vehicle traffic (1,688/day).

**Location Specific by Code (3/3 correct)**
Direct P-code lookups returned complete answers. The P606 query returned a full overview including chainage, AADT, vehicle composition breakdown, directional split, and peak hour — exactly what the four-chunk fetch strategy was designed to produce.

**Location Specific by Name (3/3 correct)**
Name-based queries resolved correctly: "Nadikudi" mapped to P526, "Hyderabad to Guntur Road" mapped to P526, and "Guntur to Bapatla to Chirala Road" mapped to P528. The name matcher's two-tier logic (single-word exact match for location names, multi-word for road names) worked as intended.

**Comparison (2/2 correct)**
Both comparison queries correctly retrieved and compared both requested locations. The P526 vs P538 answer included AADT, vehicle composition percentage, PCU totals, and a directional breakdown — the LLM used the full context from both locations' traffic chunks.

**Semantic (2/2 correct)**
The Tadikonda query returned both P577 and P-562, which is the same result as Phase 5/6 testing. The directional balance query returned P541 as the most balanced location with a 6 vehicles/day difference between directions, supported by comparison against P579 and P565.

---

## What the Scores Tell Us About the Design

The faithfulness score of 0.941 confirms the system prompt's instruction to "answer only from provided context" is working. The LLM is not inventing road names, traffic counts, or locations — it is reading from what the retriever returned.

The answer relevancy score of 0.893 is slightly lower because some answers go beyond the strict minimum. The comparison query for P526 vs P538, for example, produces a detailed breakdown of PCUs, vehicle composition percentages, and directional splits when the question only asked to compare traffic. This extra detail is useful and factually correct, but RAGAS interprets answers that contain more than what the question asked as slightly less relevant. This is a property of the metric, not a deficiency in the answers.

---

## Technical Notes

**RAGAS version:** 0.4.3

**LLM for evaluation:** llama-3.3-70b-versatile (Groq) — same model used for generation in Phase 6. Using the same model for both generation and evaluation introduces a mild self-referential bias, but since the evaluation LLM has no memory of what it generated, the scoring is still meaningful. Using a different model (e.g. GPT-4) as the judge would be more rigorous but requires a paid API key.

**Embeddings for Answer Relevancy:** all-MiniLM-L6-v2 (local, HuggingFace) — the same model used for ChromaDB retrieval. This means the relevancy measurement uses the same embedding space as the retrieval system, which is appropriate since it measures alignment between the answer and the original question.

**Rate limiting:** The Groq free tier allows 100,000 tokens/day. Running the full pipeline (15 queries × ~500 tokens each for generation) followed by RAGAS scoring (45 evaluation jobs × ~400 tokens each) consumes roughly 80,000–90,000 tokens in a single run. The evaluation script has retry logic with 65-second waits for per-minute limits, and a graceful exit if the daily limit is reached. RAGAS runs its 45 scoring jobs in parallel, which causes transient per-minute rate limit warnings — these are retried automatically and do not affect the final scores.

**Script:** `src/evaluate.py`
**Results:** `docs/evaluation_results.json`
**Run command:** `python src/evaluate.py --save`
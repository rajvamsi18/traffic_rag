# Traffic RAG — Guntur District Highway Traffic Query System

A retrieval-augmented generation (RAG) system for querying traffic count data from state highways in Guntur District, Andhra Pradesh. Built from scratch in pure Python - no LangChain, no orchestration framework, to demonstrate how each component of a RAG pipeline works under the hood.

---

## What It Does

The system lets you ask natural language questions about traffic survey data collected across 84 locations on Guntur District's state highway network. Questions like:

- *"Which road has the highest truck traffic?"*
- *"What is the peak hour at Nadikudi?"*
- *"Compare P526 and P538"*
- *"Which roads were surveyed near Tadikonda?"*

The system retrieves the relevant data, passes it to an LLM, and returns a grounded, factual answer, without the model hallucinating numbers it was never given.

---

## Dataset

**Source:** Traffic Counts on State Highways — Guntur District, Andhra Pradesh
**Published by:** AIKosh / IndiaAI (open government dataset)
**Format:** 86 XLS files (84 successfully processed; 2 missing from source)
**Coverage:** 84 survey locations, each recording 24-hour traffic counts by vehicle type and direction

Each XLS file covers one survey location and contains:
- Location metadata (road name, chainage, survey date, district)
- Hourly traffic counts in both directions
- Vehicle type breakdown (two-wheelers, cars, buses, LCVs, trucks, MAVs, tractors, etc.)
- AADT and PCU calculations

---

## Architecture

```
XLS files (raw data)
    ↓  Phase 2: extractor.py
JSON files (84 structured records)
    ↓  Phase 3: converter.py
Text chunks (336 .txt files — 4 per location)
    ↓  Phase 4: embedder.py
ChromaDB vector store (336 vectors, all-MiniLM-L6-v2)
    ↓  Phase 5: retriever.py  ←── query routing happens here
Relevant chunks (1–8 depending on query type)
    ↓  Phase 6: generator.py
Answer (Groq, llama-3.3-70b-versatile)
```

### Four Chunk Types Per Location

Each location produces four text chunks, each embedded separately:

| Chunk | Contents |
|---|---|
| Overview | Road name, location, chainage, survey date, AADT, PCU, dominant vehicle type |
| Traffic | Full vehicle composition breakdown by type, directional split, imbalance percentage |
| Peak | Peak hour, peak count, peak as % of daily total |
| Hourly | All 24 hours of traffic data in both directions |

Splitting by chunk type means a peak-hour query retrieves the peak chunk specifically, not a full dump of all data for every location.

### Four Retrieval Strategies

The retriever classifies every query before deciding how to retrieve:

**Superlative** — triggered by words like "highest", "most", "lowest". Skips ChromaDB entirely and sorts all 84 JSON records directly by the relevant metric. This is the only strategy that can reliably answer "which of all 84 is the best" — vector similarity search cannot do this because it doesn't see all records simultaneously.

**Location Specific** — triggered when a P-code (`P526`, `P-536`) or a recognisable location/road name is found in the query. Fetches all four chunks for that location by ID, bypassing vector search. Returns complete information rather than the top-k most similar chunks.

**Comparison** — triggered when two P-codes appear in the same query. Fetches the overview and traffic chunks for both locations, giving the LLM enough context to compare them directly.

**Semantic** — the default for everything else. Uses ChromaDB vector search (top-k=3) to find the most semantically similar chunks. This handles questions about geography, road characteristics, and anything the other strategies don't match.

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Data extraction | pandas + xlrd | XLS files required xlrd; pandas handled the irregular multi-row headers |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | Fast, local, no API cost, 384-dim vectors appropriate for short text chunks |
| Vector store | ChromaDB | Persistent, local, zero infrastructure |
| LLM | Groq llama-3.3-70b-versatile | Fast, free tier, OpenAI-compatible API |
| Evaluation | RAGAS 0.4.3 | LLM-based scoring for faithfulness and answer relevancy |
| Framework | Pure Python | No LangChain, no orchestration layer |

LangChain was deliberately excluded. Every component is wired together with explicit Python function calls, which makes the data flow traceable and the architecture explainable without needing to understand a framework's abstractions.

---

## Evaluation Results

Evaluated using RAGAS across 15 hand-crafted test cases covering all four query types.

| Metric | Score | What It Measures |
|---|---|---|
| Faithfulness | **0.941** | Are answer claims grounded in retrieved context? |
| Answer Relevancy | **0.893** | Does the answer address the question asked? |
| Context Precision | N/A | Were retrieved chunks relevant? (see note) |
| **Overall** | **0.917** | Average of the two scoreable metrics |

**Faithfulness 0.941** means 94% of the factual claims in each answer can be traced directly to a retrieved chunk. The system is not inventing road names, traffic counts, or location identifiers.

**Answer Relevancy 0.893** is slightly below 1.0 because some answers, particularly comparisons — provide more detail than the minimum. For example, a "compare P526 and P538" query returns AADT, vehicle composition percentages, PCU totals, and directional breakdowns. This extra depth is accurate and useful, but RAGAS interprets answers that go beyond the strict minimum as slightly less directly relevant to the question.

**Context Precision returned N/A.** The metric requires brief ground truth reference answers to judge which retrieved chunks were necessary. Our references were single sentences while retrieved chunks are dense multi-paragraph summaries, the LLM judge could not consistently assign credit. This is a metric configuration issue, not a retrieval failure. Per-query results confirm the correct location was retrieved in all 15 cases.

---

## Project Structure

```
traffic_rag/
├── data/
│   ├── raw/                  ← 86 XLS source files
│   └── processed/            ← 84 JSON records + _all_locations.json
├── text_summaries/           ← 336 text chunks + _chunks_manifest.json
├── vectorstore/              ← ChromaDB persistent collection
├── src/
│   ├── extractor.py          ← XLS → JSON (Phase 2)
│   ├── validator.py          ← data quality checks (Phase 2)
│   ├── converter.py          ← JSON → text chunks (Phase 3)
│   ├── embedder.py           ← chunks → ChromaDB (Phase 4)
│   ├── diagnose.py           ← retrieval diagnostics (Phase 4)
│   ├── retriever.py          ← hybrid query routing (Phase 5)
│   ├── generator.py          ← Groq LLM integration (Phase 6)
│   ├── main.py               ← interactive pipeline (Phase 6)
│   └── evaluate.py           ← RAGAS evaluation (Phase 7)
├── docs/
│   ├── PHASE2_DATA_EXTRACTION.md
│   ├── PHASE3_TEXT_CONVERSION.md
│   ├── PHASE4_EMBEDDINGS.md
│   ├── PHASE5_RETRIEVER.md
│   ├── PHASE6_GENERATOR.md
│   ├── PHASE7_EVALUATION.md
│   └── evaluation_results.json
├── .env                      ← GROQ_API_KEY
└── venv/
```

---

## Setup and Usage

**Requirements:** Python 3.13, a Groq API key (free tier at console.groq.com)

```bash
# Clone and set up
git clone <repo>
cd traffic_rag
python -m venv venv
source venv/bin/activate
pip install pandas xlrd sentence-transformers chromadb python-dotenv groq ragas langchain-groq langchain-community
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

**First-time pipeline setup** (run once, in order):
```bash
python src/extractor.py    # XLS → JSON
python src/validator.py    # check data quality
python src/converter.py    # JSON → 336 text chunks
python src/embedder.py     # embed into ChromaDB
```

**Using the system:**
```bash
python src/main.py              # interactive mode
python src/main.py --test       # run 8 built-in test queries
python src/main.py --debug      # show full prompt alongside answers
```

**Running evaluation:**
```bash
python src/evaluate.py --save   # runs all 15 test cases and saves scores
```

Note: evaluation consumes roughly 80,000–90,000 Groq tokens. The free tier allows 100,000 per day.

---

## Example Queries and Answers

**Superlative:**
> *Which road has the highest truck traffic?*
> The Hyderabad to Guntur Road, specifically at location P526, has the highest truck traffic, with 1,179 trucks per day.

**Location specific:**
> *Tell me about survey location P606.*
> Survey location P606 is on the Pulladigunta to Pericherla road, near Nallapadu at chainage 9.4 km. The AADT is 448 vehicles per day. The dominant vehicle type is two-wheelers at 60.5% of total. The peak traffic hour is 11:00–12:00, with 33 vehicles.

**Comparison:**
> *Which has more traffic, P605 or P606?*
> P606 has more traffic than P605. At P605 the AADT is 63 vehicles per day, whereas at P606 it is 448 vehicles per day.

**Semantic:**
> *Which roads were surveyed near Tadikonda?*
> Two roads near Tadikonda: Tadikoda to Meet NH 5 via Kantheru (P577) and Tadikonda to Rayapudi road (P-562).

---

## Key Design Decisions

**Why skip ChromaDB for superlative queries?** Vector similarity search finds the chunks most semantically similar to a query, it cannot answer "which of all 84 is the highest" because it only sees the top-k results, not all 84. The superlative strategy reads all 84 JSON records, sorts by the metric, and returns the winner. This is a case where embedding-based retrieval is the wrong tool for the job.

**Why four chunk types instead of one big chunk?** Embedding a full location summary in one chunk means a peak-hour query gets diluted by AADT data, vehicle composition data, and hourly breakdowns. Splitting by type lets the embedder separate concerns, "peak hour" questions match peak chunks, "vehicle composition" questions match traffic chunks, without needing the retriever to filter out irrelevant sections.

**Why pure Python over LangChain?** Every architectural choice in this project has a reason that can be stated in plain English. Using a framework would make the system work but would make those reasons invisible - the framework just does it. Being able to explain why each component exists and how it connects to the next is more valuable for learning and for interviews than having the framework handle it silently.

**Why Groq over OpenAI?** The project uses only free-tier APIs. Groq provides llama-3.3-70b-versatile at 30 requests/minute with no meaningful daily cap on a fresh account, which is sufficient for both generation and evaluation. Groq's API is OpenAI-compatible, so switching to a different provider requires changing only the base URL and model name.

---

## Limitations

- **84 of 86 locations processed.** Two XLS files were missing from the raw data directory and could not be extracted. The missing files are noted in the validator output.
- **Single survey date per location.** The dataset captures a single day of traffic at each location. AADT is estimated from this single day, it is labelled as AADT in the source files but should be interpreted as a survey-day estimate, not a multi-year average.
- **Groq free tier rate limits.** The 100,000 token/day limit means running evaluation and interactive queries on the same day may exhaust the daily allowance. The system has retry logic for per-minute limits but exits gracefully when the daily cap is hit.
- **Context Precision not scored.** See evaluation section above.

---

## Acknowledgements

This project was built with significant assistance from Claude (Anthropic). Claude helped with debugging across all seven phases, including five bugs in the retriever, two bugs in the generator pipeline, and four incompatible API changes in RAGAS 0.4.x, and contributed to the documentation structure and phase planning. The architectural decisions, dataset selection, and all code are my own work; Claude served as a debugging partner and sounding board throughout.

The dataset is published by the IndiaAI / AIKosh platform under an open government data licence. I am grateful to the Andhra Pradesh state government for making this data publicly available.
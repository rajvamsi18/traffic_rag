# Phase 3: Text Conversion

**Status:** ✅ Complete  
**Script:** `src/converter.py`  
**Input:** `data/processed/` — 84 JSON files  
**Output:** `text_summaries/` — 336 `.txt` chunk files + `_chunks_manifest.json`

---

## What This Phase Does

Phase 3 takes the structured JSON files from Phase 2 and converts them into natural language text — the format that the vector database can actually work with.

A vector database doesn't understand numbers and keys like `{"aadt": {"two_wheelers": 271}}`. It understands meaning embedded in text. So this phase is essentially a translation layer: every JSON file becomes a set of readable paragraphs that describe the same data in plain English, the way a traffic engineer would explain it.

The output of this phase is what gets embedded in Phase 4 and retrieved in Phase 5.

---

## The Core Design Decision — One Chunk or Many?

This was the most important design choice of the entire RAG pipeline, so it's worth explaining carefully.

The simplest approach would be one long paragraph per location — take all the data, describe it in one block of text, store it as one unit. Simple, and it works for basic queries.

The problem is retrieval quality. When a user asks *"which road has the most truck traffic?"*, the system embeds that question into a vector and finds the closest chunks. If each chunk contains everything about a location — two-wheelers, survey date, weather, peak hour, goods traffic — then the truck-related signal is diluted by all the other information. The system might miss the most relevant location because its truck data is buried inside a long paragraph about something else.

The solution is **focused chunks** — split each location into 4 shorter paragraphs, each answering a different type of question. This way, when someone asks about trucks, the retrieval finds a chunk that is *mostly about goods vehicles*, not a chunk that mentions trucks somewhere in the middle.

Each location produces exactly 4 chunks:

| Chunk | Focuses On | Best For Queries Like |
|---|---|---|
| **Overview** | Road name, location, chainage, survey date, weather | "Where is P562?" / "What roads were surveyed near Tadikonda?" |
| **Traffic** | AADT counts, all 16 vehicle types, group totals, PCU | "Which road has the most trucks?" / "What's the busiest location?" |
| **Directional** | Direction 1 vs Direction 2 comparison | "Is traffic balanced on the Hyderabad-Guntur road?" |
| **Peak Hour** | Peak/off-peak hours, daily motorised/non-motorised split | "Which location is most congested in the evening?" |

With 84 locations × 4 chunks = **336 total chunks** in the corpus.

---

## Why Not Overlap Chunking?

Overlap chunking is the standard approach in most RAG tutorials — you split a document every N words and repeat the last 50 or so words at the start of the next chunk, so meaning doesn't break at the boundary. It's the right technique when you're slicing through continuous prose that was never designed to be split.

It doesn't apply here because the source material is structured JSON, not a flowing document. These chunks aren't cut from something longer — they're generated from scratch by the converter, each one written as a self-contained unit from the beginning. There's no sentence that starts in one chunk and finishes in another, so there's no boundary problem to solve with overlap.

The split is also by topic rather than by length, which matters. Overlap between a traffic composition chunk and a peak hour chunk would just add noise — those two topics have nothing to carry over from one to the other.

One honest limitation this approach does have: a question like *"what is the peak hour traffic composition at P-562?"* spans two chunk types — the peak chunk has the hour, the traffic chunk has the composition. Retrieving only the single closest chunk might not give the full answer. This is a known problem in RAG called a multi-hop question, and it's handled in Phase 5 by retrieving the top 3 most relevant chunks instead of just 1, giving the LLM enough context to piece together a complete answer.

---

## What Each Chunk Contains

### Overview Chunk
Describes the survey point as a location — where it is, what road it's on, when the survey happened, and what the weather was. This chunk reads like the opening sentence of a field report.

Example (P-562):
> *"Location P-562 is a classified traffic count survey point on the Tadikonda to Rayapudi road, situated near Shakamaru at chainage 8.2 km in Guntur District, Andhra Pradesh. The survey was conducted on Saturday, 08 April 2017, under Normal Hot weather conditions..."*

### Traffic Chunk
The most information-dense chunk. Covers AADT total, PCU total, the three category breakdowns (Fast Passenger / Goods / Slow Modes), all individual vehicle type counts with percentages, and a note explaining what PCU values mean and why the PCU total differs from the raw vehicle count.

This chunk is designed to answer any vehicle-count-based question about a location.

### Directional Chunk
Compares Direction 1 vs Direction 2 — total vehicles, fast passenger, goods vehicles, and motorised totals for each direction. Automatically adds a plain-English observation if goods traffic is noticeably heavier in one direction (more than 20% difference), since this often indicates one-way freight patterns worth flagging.

### Peak Hour Chunk
Identifies the peak and off-peak hours, the vehicle volumes during those hours, and the motorised/non-motorised split. Classifies the peak time into a period (morning rush, midday, evening rush, night) to make the text more natural. Also includes a note explaining why the daily total from the Both_Directions sheet may differ slightly from the AADT figure — they measure slightly different things.

---

## Implementation Details

### Dynamic percentage calculations
Every vehicle count is automatically converted to a percentage of total traffic. This matters for retrieval — a query like *"which road has the highest proportion of goods vehicles?"* requires percentages to be present in the text, not just raw counts.

### Top goods vehicles summary
Rather than just listing all 7 goods vehicle types, the traffic chunk also generates a natural language sentence like *"Among goods vehicles, the highest volumes are: Tractors with Trailer (655/day), 3-Axle Trucks (582/day), LCV (55/day)"*. This makes the text richer for queries about specific freight types.

### Peak period classification
The peak hour number (e.g. `18`) is converted to a human-readable period: 07:00–10:00 = "morning rush hour", 15:00–19:00 = "evening rush hour", etc. This means a query like *"which locations have evening peak traffic?"* can match the text directly.

### Chunks manifest
Every chunk file is registered in `_chunks_manifest.json` with its `location_id`, `chunk_type`, file path, word count, and a one-line description. Phase 4 (embedding) reads this manifest to know which files to embed and what metadata to attach to each vector in ChromaDB.

---

## Bug Encountered — Key Naming Collision

This bug didn't appear in converter.py itself — it was exposed by running the converter, which forced me to look carefully at what the extractor was actually producing.

**Symptom:** The validator showed 84/84 files with missing `direction_1` and `direction_2` after updating the extractor.

**Root cause:** The extractor stores two completely different things under the name `direction_1`:
- From the Input sheet: the *name* of Direction 1 travel — a string like `"Pulladigunta to Pericherla road"`
- From the Analysis sheet: the *traffic counts* for Direction 1 — a dict like `{two_wheelers: 132, total_vehicles: 210, ...}`

Both were stored under the key `direction_1`. When the Input sheet and Analysis sheet results were merged with `{**metadata, **analysis}`, the Analysis dict silently overwrote the Input string. So `direction_1` contained the traffic counts, and the direction name was lost entirely.

**Fix:** Renamed the keys to make their purpose explicit:
- `direction_1` (string) → `dir1_name` — the direction name, e.g. `"Pulladigunta to Pericherla"`
- `direction_1` (dict) → `dir1_traffic` — the traffic count dictionary
- Same for `direction_2` → `dir2_name` and `dir2_traffic`

After renaming in the extractor, every other file that reads the JSON had to be updated too: the validator (7 places) and the converter (3 places). This is a good example of why naming things clearly from the start saves time — a key called `direction_1` is ambiguous because it could mean either the name or the data. `dir1_name` and `dir1_traffic` leave no room for confusion.

---

## Data Quality Observation

While reviewing the sample output, I noticed one location (P-562) had its survey day recorded as **"Saturaday"** — a typo in the original Excel file made by the surveyor. The code correctly extracted and reproduced it as-is.

This is intentional. Correcting typos in source data during extraction would be the wrong approach — it would mean the processed data no longer faithfully represents what was recorded. The right place to handle this is in the text template, which could normalise known day names before writing them, but for this project the impact is minimal (one field in one location).

---

## Final Results

```
Locations converted  : 84 / 84
Total chunks created : 336
Failed               : 0

Chunk statistics:
  Average words per chunk : 158
  Total words in corpus   : 53,055
```

**Chunk size is deliberately in the 100–200 word range.** This is a known sweet spot for RAG systems. Too short (under 50 words) and there isn't enough context for the embedding to capture meaning well. Too long (over 400 words) and the embedding averages over too much information, diluting the signal for specific queries.

---

## Sample Output

Below are all 4 chunks for location P-562 (Tadikonda to Rayapudi road, 2,888 AADT), which illustrates the range of information each chunk type captures:

**Overview chunk (109 words):** Location, road, chainage, date, weather  
**Traffic chunk (299 words):** Full AADT breakdown across all 16 vehicle types  
**Directional chunk (96 words):** 1,457 vs 1,378 vehicles/day, goods traffic heavier in Direction 1  
**Peak chunk (134 words):** 12:00–13:00 peak (238 vehicles), 47× peak-to-off-peak ratio

---

*Next: Phase 4 — Embeddings and Vector Database (`src/embedder.py`)*  
*Each of the 336 text chunks will be converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model and stored in ChromaDB with its metadata.*
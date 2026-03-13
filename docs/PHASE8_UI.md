# Phase 8 — Streamlit UI

## What This Phase Does

Phases 1–7 built and evaluated a complete RAG pipeline accessible only through a terminal. Phase 8 wraps that pipeline in a simple web interface so the system can be demonstrated without running command-line scripts mid-conversation.

The interface is built with Streamlit — a Python library that renders a browser UI from a plain Python script. No HTML, no JavaScript, no separate frontend. The entire UI is one file (`app.py`) that imports directly from the same `retriever.py` and `generator.py` used in production.

---

## What the UI Shows

**Sidebar** — dataset overview, explanation of the four retrieval strategies, and the two RAGAS evaluation scores (Faithfulness 0.941, Answer Relevancy 0.893) as persistent metrics visible throughout any demo.

**Example query buttons** — six clickable examples covering all four query types. Clicking any button populates the input box without typing. This is useful in an interview where you want to demonstrate different retrieval strategies quickly without searching for what to type.

**Answer card** — the LLM's response displayed in a dark card with a green left border. Formatted for readability at any screen size.

**Retrieval metadata badges** — directly above the answer, two sets of badges show: which retrieval strategy was used (e.g. `superlative:highest truck traffic`) and which location(s) were retrieved (e.g. `P526`). This makes the pipeline transparent — the viewer sees not just the answer but how the system decided to retrieve it.

**Retrieved chunks expander** — a collapsible section below the answer showing the raw text chunks that were passed to the LLM. This is the most important feature for a technical audience. It demonstrates that the answer came from grounded context, not from model memory.

---

## Why Streamlit

The alternatives were Flask (requires HTML templates and routing boilerplate), FastAPI with a frontend (requires JavaScript), or Gradio (simpler but less flexible layout). Streamlit was chosen because it stays entirely in Python, requires no separate build step, and produces a clean enough result for a local demo. The entire UI is readable and explainable in an interview — there is no framework magic to hide behind.

---

## Running the UI

```bash
cd ~/Desktop/traffic_rag
source venv/bin/activate
streamlit run app.py
```

Opens at `http://localhost:8501`. The retriever loads once on first run and is cached — subsequent queries do not reload the model or reconnect to ChromaDB.

The UI requires the same prerequisites as `main.py`: the vectorstore must exist (built by `embedder.py`), processed JSON files must exist (built by `extractor.py`), and `GROQ_API_KEY` must be set in `.env`.

---

## Phase 8 Bug Fix — Superlative Patterns for All Vehicle Types

During UI testing, queries like "which road has highest bus", "which road has highest car", and "which road has most tractors" returned incorrect answers. The retriever was falling through to semantic search instead of the superlative strategy.

Two bugs were found and fixed in `retriever.py`:

**Bug 1 — Incomplete bus pattern.** The regex `buses?` matches "buse" and "buses" but not "bus" — the `?` makes the `e` optional, not the `es`. Fixed to `bus(es)?` which correctly matches both "bus" and "buses".

**Bug 2 — Tractor singular not matched.** The regex `\btractor\b` with a word boundary does not match "tractors" (the `s` sits outside the boundary). Fixed to `tractors?` which matches both "tractor" and "tractors".

In the same fix, new superlative patterns were added for all vehicle types present in the dataset that were previously unrecognised:

| Vehicle type | JSON key | Query words recognised |
|---|---|---|
| Car / jeep / van | `car_jeep_van` | car, jeep, van, passenger car |
| Bus | `mini_bus` + `standard_bus` | bus, buses, mini bus, standard bus |
| LCV | `lcv` | lcv, light commercial, light goods |
| Tractor | `tractor_with_trailer` + `tractor_without_trailer` | tractor, tractors, farm vehicle |
| Auto rickshaw | `auto_rickshaw` | auto rickshaw, three wheel, autorickshaw |
| Tempo | `tempo` | tempo, minivan, mini van |
| MAV | `mav` | mav, multi-axle, articulated |
| Cycle | `cycle` | cycle, bicycle, cycling |

All patterns support both ascending ("lowest") and descending ("highest / most") queries.

---

## Script

`app.py` — located in the project root, not in `src/`. Run directly with `streamlit run app.py`.
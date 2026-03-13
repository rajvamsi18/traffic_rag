# Phase 2: Data Extraction

**Status:** ✅ Complete  
**Script:** `src/extractor.py`  
**Validation:** `src/validator.py`  
**Output:** `data/processed/` — 84 individual JSON files + `_all_locations.json`

---

## What This Phase Does

Phase 2 reads all 86 raw Excel traffic survey files and converts them into clean, structured JSON — one file per survey location. This JSON becomes the input for Phase 3 (text conversion) and ultimately feeds the RAG pipeline.

The main challenge here was that these files were filled by hand by different surveyors across Guntur District. Every file had the same *intended* structure, but human inconsistencies meant I couldn't just hardcode row numbers and assume fixed positions. The code had to be written defensively — search for labels, verify what you find, and have fallbacks when things aren't where you expect them.

---

## What Gets Extracted From Each File

Each XLS file has 9 sheets. I only use 3 of them — `Input`, `Analysis`, and `Both_Directions`. The others (raw 15-minute count sheets, SCF reference tables) are skipped.

### From the `Input` Sheet — Location Metadata

| Field | Description | Example |
|---|---|---|
| `location_id` | Unique survey point ID | `P526` |
| `location_name` | Village or locality name | `Nadikudi` |
| `road_name` | Human-readable road name | `Hyderabad to Guntur Road` |
| `road_ref_code` | Official road reference number | `R0000002-D04-A-1` |
| `direction_1` | Name of Direction 1 travel | `Hyderabad to Guntur` |
| `direction_2` | Name of Direction 2 travel | `Guntur to Hyderabad` |
| `chainage_km` | Distance marker on road (km) | `212.6` |
| `survey_date` | Date survey was conducted | `22 April 2017` |
| `survey_day` | Day of week | `Saturday` |
| `weather` | Weather during survey | `Normal Hot` |
| `pcu_weights` | PCU factor per vehicle type | `{"two_wheelers": 0.5, "mav": 4.5, ...}` |
| `seasonal_correction_factors` | SCF per vehicle category | `{"Car": 1.02, "TW": 1.03, ...}` |

### From the `Analysis` Sheet — Traffic Counts

For each of five row labels — **Direction 1, Direction 2, Both, ADT, AADT** — I extract all 16 vehicle type counts, plus group totals I calculate myself:

| Group Field | What It Sums |
|---|---|
| `total_fast_passenger` | Two Wheelers + Auto + Car + Mini Bus + Standard Bus |
| `total_goods` | Tempo + LCV + all trucks + MAV + tractors |
| `total_slow_modes` | Cycle + Cycle Rickshaw + Animal Drawn |
| `total_motorised` | All fast passenger + all goods |
| `total_non_motorised` | Slow modes + Other |

> Note: the Analysis sheet has category headers (Fast Passenger, Goods, Slow Modes) but no subtotal columns for those groups. Only `total_vehicles` (col 17) and `pcu_total` (col 18) are pre-calculated. Everything else I compute from individual vehicle counts.

**Key terms for context:**

- **ADT** — Average Daily Traffic. The raw counted vehicles on the survey day.
- **AADT** — Annual Average Daily Traffic. ADT adjusted by Seasonal Correction Factors to represent a typical day across the full year. This is the primary metric used in road planning.
- **PCU** — Passenger Car Units. Converts different vehicle types to a common scale — a motorcycle counts as 0.5 PCU, an articulated truck as 4.5 PCU. Used to measure actual road capacity stress rather than just vehicle count.
- **SCF** — Seasonal Correction Factor. Multiplier that adjusts for when in the year the survey was taken.

### From the `Both_Directions` Sheet — Hourly Patterns

| Field | Description |
|---|---|
| `peak_hour` | Time period with highest combined traffic |
| `peak_hour_volume` | Total vehicles during peak hour |
| `peak_hour_motorised` | Motorised vehicles during peak hour |
| `peak_hour_non_motorised` | Non-motorised vehicles during peak hour |
| `off_peak_hour` | Time period with lowest traffic |
| `off_peak_volume` | Vehicle count during off-peak hour |
| `daily_total` | Sum of all hourly counts |
| `hourly_distribution` | Full 24-hour breakdown with motorised/non-motorised split |

---

## Design Decisions

### Not hardcoding row or column numbers

The most fragile approach would be to say "row 14 is always AADT." Different surveyors added or removed rows in different files. Instead, I first scan for the numbered row (the row containing `1, 2, 3 ... 18`) to build a dynamic column position map, then search for exact label text to find each data row. This way the code still works even if someone inserted a blank row somewhere.

### Using exact match, not substring match, for row labels

The title in Row 0 of every Analysis sheet reads: `"...CLASSIFIED TRAFFIC VOLUME COUNT - 1 Day 24 Hours (ADT and AADT)"`. If I use `'AADT' in cell`, it matches this title row first and returns all zeros — because the title has no numeric data. Using `label == 'AADT'` (exact match) means only the actual AADT data row qualifies.

### Extracting PCU weights and SCF values

These aren't traffic counts, but I kept them because they're needed for reasoning questions later. PCU weights let the RAG system answer something like *"which road is under the most capacity stress?"* — raw vehicle counts alone can't do that. SCF values explain *why* AADT differs from ADT, which comes up when someone asks about seasonality.

---

## Bugs that wer Hit and How I Fixed Them

This section documents the actual problems I ran into during extraction, since they reveal things about the data that aren't obvious from just reading the Excel files.

---

### Bug 1 — All AADT Values Were Zero

**Symptom:** Every vehicle count in the AADT row extracted as `0.0`

**Root cause:** I was using `if 'AADT' in cell_label` to find the AADT row. The Analysis sheet title in Row 0 reads `"...1 Day 24 Hours (ADT and AADT)"` — so the substring `'AADT'` matched the title first. Row 0 has no vehicle data, so everything came out as zero and the code broke out of the loop before ever reaching the real data.

**Fix:** Switched to a `TARGET_LABELS` dictionary with exact string matches: `if label == 'AADT'`. The title row has a long string, never exactly `"AADT"`.

**Takeaway:** Substring search on structured data with known headers is a trap. Always use exact match when you know what you're looking for.

---

### Bug 2 — Float Vehicle Counts

**Symptom:** AADT values coming out as `3485.52`, `442.9` instead of clean integers

**Root cause:** Excel *displays* AADT rounded to the nearest whole number, but internally stores the full formula result: `ADT × SCF = 3384 × 1.03 = 3485.52`. Pandas reads the raw stored value, not what Excel shows on screen.

**Fix:** Added a `safe_int()` helper that rounds to the nearest integer for all vehicle count extractions. PCU weights stay as floats because they're factors, not counts.

**Takeaway:** What Excel displays and what it stores can be different. If your numbers look wrong but are close to right, check for this.

---

### Bug 3 — "Both" Row Missing in 21 Files

**Symptom:** The `both` section came out empty for 21 of 84 files

**Root cause:** In these files, the cell that should contain the text `"Both"` was left blank. Excel apparently autofilled it with the survey date (`2017-04-25 00:00:00`). My label search found no `"BOTH"` and skipped the row, even though the actual traffic data was right there.

**Fix:** Added a positional fallback. If the `"both"` row isn't found by label, I take the row immediately after the `direction_2` row. The structure in every file is always `Direction 1 → Direction 2 → Both` in sequence, regardless of what the labels say.

---

### Bug 4 — Direction Rows All Missing in 2 Files (P536, P546)

**Symptom:** `direction_1`, `direction_2`, and `both` all empty for P536 and P546

**Root cause:** Same issue as Bug 3 but more extreme. All three label cells were blank, all showing the survey date. Since none of the three labels were found, the Bug 3 fallback had no anchor point to work from.

**Fix:** Added a second fallback. When `direction_1` isn't found, I check whether `PCU values` was found. If yes, the three rows immediately after the PCU values row are always Direction 1, Direction 2, and Both — this held true across every file I checked.

---

### Bug 5 — Survey Day Missing in 7 Files

**Symptom:** `survey_day` empty for 7 files even though a date was present

**Root cause:** Those surveyors left the day-of-week cell blank in both the Input sheet and the Analysis sheet header.

**Fix:** If the day name is missing but the date was parsed successfully, I compute the day name from the date using Python: `datetime.strptime(survey_date, "%d %B %Y").strftime("%A")`. If the date is `"20 April 2017"`, Python gives back `"Thursday"`. No ambiguity.

---

### Bug 6 — Survey Date Missing in 5 Files (P533, P551, P574, P592, P600)

**Symptom:** `survey_date` empty even though I could see the date clearly in the Excel file

**Root cause — two separate issues working together:**

First, pandas parsed the date cell in these 5 files as a Python `datetime` object (`datetime.datetime(2017, 4, 24, 0, 0)`) instead of a string or serial number. My `parse_date()` function only handled strings, so it returned empty for this type.

Second — and this is the subtle one — extraction merges Input sheet results and Analysis sheet results like `{**metadata, **analysis}`. The Input sheet correctly extracted the date. But the Analysis sheet also tries to extract the date from its own header, failed on the same datetime-object issue, and returned `''`. Since analysis comes second in the merge, the empty string *overwrote* the correctly parsed value from Input.

**Fix:** Two changes. Added `if hasattr(raw_val, 'strftime')` at the top of `parse_date()` to handle datetime objects directly. And changed the Analysis sheet date assignment to `if parsed:` — only write it if non-empty, so a failed parse never overwrites a good value from Input.

**Takeaway:** When merging data from multiple sources, empty values from a secondary source should never silently overwrite valid values from the primary source. This is a pattern worth being careful about in general.

---

## Final Validation Results

```
✅ Ready (full data)        : 84
🟡 Usable (partial context) :  0
❌ Not ready (no AADT)      :  0
Total                       : 84

AADT STATISTICS
  Highest : P528 → 12,669 vehicles/day
  Lowest  : P605 →     63 vehicles/day
  Average :        1,770 vehicles/day
  Files with AADT > 0 : 84 / 84
```

All 84 files have complete AADT and directional traffic data. A handful of files have minor metadata gaps (survey date or day left blank by the original surveyor) — these will show as "Not recorded" in text summaries rather than causing failures.

---

## Sample Output

```json
{
  "source_file": "P526.xls",
  "location_id": "P526",
  "location_name": "Nadikudi",
  "road_name": "Hyderabad to Guntur Road",
  "direction_1": "Hyderabad to Guntur",
  "direction_2": "Guntur to Hyderabad",
  "chainage_km": 212.6,
  "survey_date": "22 April 2017",
  "survey_day": "Saturday",
  "weather": "Normal Hot",
  "aadt": {
    "two_wheelers": 3486,
    "truck_2axle": 208,
    "mav": 725,
    "total_vehicles": 6537,
    "pcu_total": 10082,
    "total_fast_passenger": 4827,
    "total_goods": 1689,
    "total_motorised": 6516,
    "total_non_motorised": 21
  },
  "peak_hour": "18:00 - 19:00",
  "peak_hour_volume": 412,
  "daily_total": 6391,
  "dominant_vehicle_type": "two wheelers"
}
```

---

## Things That Surprised Me About the Data

Two-wheelers dominate at almost every location. I expected mixed traffic, but motorcycles and scooters make up the majority of AADT at most survey points. It's a useful reminder that Indian road traffic looks very different from what most Western traffic planning literature assumes.

The range between locations is much wider than I expected. P528 sees nearly 200 times more traffic than P605. A blanket "average" answer to any query about Guntur traffic would be nearly meaningless — which is exactly why a retrieval-based approach makes more sense here than a single summary.

Every single survey was conducted in April 2017. The dataset is a one-month snapshot of the whole district. AADT corrects for seasonality mathematically using the SCF values, but it's worth knowing that all 84 files share the same time window. Any long-term trend analysis would need more data.

---

*Next: Phase 3 — Text Conversion (`src/converter.py`)*  
*Each JSON will be converted into a natural language paragraph ready for embedding into ChromaDB.*
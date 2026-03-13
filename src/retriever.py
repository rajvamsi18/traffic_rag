"""
retriever.py  —  Phase 5: Hybrid Retriever
-------------------------------------------
Sits between the user query and ChromaDB. Classifies each query into
one of four types and routes it to the appropriate retrieval strategy.

WHY A RETRIEVER LAYER?
    Pure semantic search (Phase 4) failed on superlative queries like
    "which road has the most trucks?" because all traffic chunks are
    semantically similar — the embedding model can't compare numbers.
    The retriever solves this by detecting query intent and routing to
    the right strategy before touching ChromaDB.

FOUR QUERY TYPES AND THEIR STRATEGIES:

    1. SUPERLATIVE  → "highest AADT", "most trucks", "busiest road"
       Strategy: Skip ChromaDB entirely. Sort all 84 JSON records by
       the relevant metric and return the top result directly.
       Why: Vector search cannot rank by numerical value. JSON can.

    2. LOCATION SPECIFIC → "traffic at P526", "peak hour at Nadikudi"
       Strategy: Semantic search filtered by location_id or name match.
       Returns all 4 chunk types for that location.
       Why: When a specific location is named, we want everything about it.

    3. COMPARISON → "compare P526 and P538", "P526 vs P538"
       Strategy: Direct fetch of both locations' chunks from ChromaDB
       by ID, no semantic search needed.
       Why: Both locations are named explicitly — no need to search.

    4. GENERAL → "what roads were surveyed near Guntur?"
       Strategy: Semantic search, top-k results (default k=3).
       Why: Broad questions benefit from multiple relevant chunks so
       the LLM can synthesise across locations.

OUTPUT FORMAT:
    Every retrieval returns a list of RetrievalResult objects:
        .chunk_id     → e.g. "P526_traffic"
        .location_id  → e.g. "P526"
        .chunk_type   → "overview" | "traffic" | "directional" | "peak"
        .road_name    → human readable road name
        .text         → full chunk text (passed to LLM)
        .score        → similarity score (0–1), or 1.0 for direct lookups
        .strategy     → which strategy was used (for transparency)
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from sentence_transformers import SentenceTransformer
import chromadb


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_NAME      = 'all-MiniLM-L6-v2'
COLLECTION_NAME = 'traffic_guntur'
DEFAULT_TOP_K   = 3


# ─────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id:    str
    location_id: str
    chunk_type:  str
    road_name:   str
    text:        str
    score:       float
    strategy:    str


# ─────────────────────────────────────────────
# SUPERLATIVE QUERY PATTERNS
# ─────────────────────────────────────────────
#
# Maps trigger phrases → (metric_key, sort_label, description)
# metric_key must match keys computed in _get_metric_value()

SUPERLATIVE_PATTERNS = [
    # ── Specific vehicle types FIRST (before generic traffic/AADT) ──
    # This ordering is critical — "highest truck traffic" must match trucks,
    # not the generic AADT pattern which also contains the word "traffic".

    # Trucks
    (r'\b(highest|most|maximum)\b.{0,30}\b(truck|lorry|hgv|goods vehicle)\b',
     'trucks', 'highest truck traffic'),
    (r'\b(lowest|least|minimum)\b.{0,30}\b(truck|lorry|hgv|goods vehicle)\b',
     'trucks_asc', 'lowest truck traffic'),

    # Two-wheelers — note: uses wheelers? to match both "wheeler" and "wheelers"
    (r'\b(highest|most|maximum)\b.{0,30}\b(two.?wheelers?|motorcycle|bike|scooter|2.?wheelers?)\b',
     'two_wheelers', 'highest two-wheeler count'),
    (r'\b(lowest|least|minimum)\b.{0,30}\b(two.?wheelers?|motorcycle|bike|scooter|2.?wheelers?)\b',
     'two_wheelers_asc', 'lowest two-wheeler count'),

    # Goods
    (r'\b(highest|most|maximum)\b.{0,30}\b(goods|freight|cargo|commercial)\b',
     'goods', 'highest goods vehicle traffic'),
    (r'\b(lowest|least|minimum)\b.{0,30}\b(goods|freight|cargo|commercial)\b',
     'goods_asc', 'lowest goods vehicle traffic'),

    # PCU
    (r'\b(highest|most|maximum)\b.{0,30}\b(pcu|passenger car unit|capacity)\b',
     'pcu', 'highest PCU load'),

    # Peak volume
    (r'\b(highest|most|maximum|busiest)\b.{0,30}\b(peak|congested|congestion)\b',
     'peak_volume', 'highest peak hour volume'),
    (r'\b(lowest|least|quietest)\b.{0,30}\b(peak|congestion)\b',
     'peak_volume_asc', 'lowest peak hour volume'),

    # Buses
    (r'\b(highest|most|maximum)\b.{0,20}\b(bus(es)?|bus traffic|standard.?bus|mini.?bus)\b',
     'buses', 'highest bus traffic'),
    (r'\b(lowest|least|minimum)\b.{0,20}\b(bus(es)?|bus traffic|standard.?bus|mini.?bus)\b',
     'buses_asc', 'lowest bus traffic'),

     # Cars / jeeps / vans
    (r'\b(highest|most|maximum)\b.{0,20}\b(car|jeep|van|passenger car)\b',
     'car_jeep_van', 'highest car/jeep/van count'),
    (r'\b(lowest|least|minimum)\b.{0,20}\b(car|jeep|van|passenger car)\b',
     'car_jeep_van_asc', 'lowest car/jeep/van count'),

     # LCV
    (r'\b(highest|most|maximum)\b.{0,20}\b(lcv|light commercial|light.?goods)\b',
     'lcv', 'highest LCV count'),
    (r'\b(lowest|least|minimum)\b.{0,20}\b(lcv|light commercial|light.?goods)\b',
     'lcv_asc', 'lowest LCV count'),

      # Tractors
    (r'\b(highest|most|maximum)\b.{0,20}\b(tractors?|farm vehicle)\b',
     'tractors', 'highest tractor count'),
    (r'\b(lowest|least|minimum)\b.{0,20}\b(tractors?|farm vehicle)\b',
     'tractors_asc', 'lowest tractor count'),

     # Auto rickshaws
    (r'\b(highest|most|maximum)\b.{0,20}\b(auto.?rickshaw|three.?wheel|autorickshaw)\b',
     'auto_rickshaw', 'highest auto rickshaw count'),
    (r'\b(lowest|least|minimum)\b.{0,20}\b(auto.?rickshaw|three.?wheel|autorickshaw)\b',
     'auto_rickshaw_asc', 'lowest auto rickshaw count'),

     # Tempo
    (r'\b(highest|most|maximum)\b.{0,20}\b(tempo|minivan|mini.?van)\b',
     'tempo', 'highest tempo count'),
    (r'\b(lowest|least|minimum)\b.{0,20}\b(tempo|minivan|mini.?van)\b',
     'tempo_asc', 'lowest tempo count'),

    # MAV (multi-axle vehicles)
    (r'\b(highest|most|maximum)\b.{0,20}\b(mav|multi.?axle|articulated)\b',
     'mav', 'highest MAV count'),
    (r'\b(lowest|least|minimum)\b.{0,20}\b(mav|multi.?axle|articulated)\b',
     'mav_asc', 'lowest MAV count'),

    # Cycles
    (r'\b(highest|most|maximum)\b.{0,20}\b(cycle|bicycle|cycling)\b',
     'cycles', 'highest cycle count'),
    (r'\b(lowest|least|minimum)\b.{0,20}\b(cycle|bicycle|cycling)\b',
     'cycles_asc', 'lowest cycle count'),


    # ── Generic AADT / total traffic LAST ──
    # Must come after specific types so "highest truck traffic" doesn't
    # match this pattern via the word "traffic".
    (r'\b(highest|most|busiest|maximum|largest)\b.{0,30}\b(aadt|total traffic|total vehicles)\b',
     'aadt_total', 'highest total AADT'),
    (r'\b(lowest|least|quietest|minimum|smallest)\b.{0,30}\b(aadt|total traffic|total vehicles)\b',
     'aadt_total_asc', 'lowest total AADT'),
    # Standalone "highest traffic" / "busiest road" without a specific vehicle type
    (r'\b(busiest|highest traffic|most traffic)\b',
     'aadt_total', 'highest total AADT'),
    (r'\b(quietest|lowest traffic|least traffic)\b',
     'aadt_total_asc', 'lowest total AADT'),
]


# ─────────────────────────────────────────────
# METRIC VALUE EXTRACTOR
# ─────────────────────────────────────────────

def _get_metric_value(data: dict, metric: str) -> float:
    """Extract the numeric value for a given metric from a location's JSON."""
    aadt = data.get('aadt', {})
    truck_keys = ['truck_2axle', 'truck_3axle', 'mav']

    mapping = {
        'aadt_total':     lambda: aadt.get('total_vehicles', 0) or 0,
        'aadt_total_asc': lambda: aadt.get('total_vehicles', 0) or 0,
        'trucks':         lambda: sum(aadt.get(k, 0) or 0 for k in truck_keys),
        'trucks_asc':     lambda: sum(aadt.get(k, 0) or 0 for k in truck_keys),
        'two_wheelers':   lambda: aadt.get('two_wheelers', 0) or 0,
        'two_wheelers_asc': lambda: aadt.get('two_wheelers', 0) or 0,
        'goods':          lambda: aadt.get('total_goods', 0) or 0,
        'goods_asc':      lambda: aadt.get('total_goods', 0) or 0,
        'pcu':            lambda: aadt.get('pcu_total', 0) or 0,
        'peak_volume':    lambda: data.get('peak_hour_volume', 0) or 0,
        'peak_volume_asc':lambda: data.get('peak_hour_volume', 0) or 0,
        'buses':          lambda: (aadt.get('mini_bus', 0) or 0) + (aadt.get('standard_bus', 0) or 0),
        'buses_asc':      lambda: (aadt.get('mini_bus', 0) or 0) + (aadt.get('standard_bus', 0) or 0),
        'car_jeep_van':   lambda: aadt.get('car_jeep_van', 0) or 0,
        'car_jeep_van_asc': lambda: aadt.get('car_jeep_van', 0) or 0,
        'lcv':              lambda: aadt.get('lcv', 0) or 0,
        'lcv_asc':          lambda: aadt.get('lcv', 0) or 0,
        'tractors':         lambda: (aadt.get('tractor_with_trailer', 0) or 0) + (aadt.get('tractor_without_trailer', 0) or 0),
        'tractors_asc':     lambda: (aadt.get('tractor_with_trailer', 0) or 0) + (aadt.get('tractor_without_trailer', 0) or 0),
        'auto_rickshaw':    lambda: aadt.get('auto_rickshaw', 0) or 0,
        'auto_rickshaw_asc':lambda: aadt.get('auto_rickshaw', 0) or 0,
        'tempo':            lambda: aadt.get('tempo', 0) or 0,
        'tempo_asc':        lambda: aadt.get('tempo', 0) or 0,
        'mav':              lambda: aadt.get('mav', 0) or 0,
        'mav_asc':          lambda: aadt.get('mav', 0) or 0,
        'cycles':           lambda: aadt.get('cycle', 0) or 0,
        'cycles_asc':       lambda: aadt.get('cycle', 0) or 0,
    }

    fn = mapping.get(metric)
    return float(fn()) if fn else 0.0


# ─────────────────────────────────────────────
# QUERY CLASSIFIER
# ─────────────────────────────────────────────

def classify_query(query: str):
    """
    Classifies a query into one of four types.
    Returns (query_type, extra_info)

    query_type: 'superlative' | 'location_specific' | 'comparison' | 'general'
    extra_info:
        superlative      → (metric_key, description)
        location_specific→ list of matched location identifiers
        comparison       → list of two location identifiers
        general          → None
    """
    q = query.lower().strip()

    # ── 1. Superlative check ──
    for pattern, metric, description in SUPERLATIVE_PATTERNS:
        if re.search(pattern, q):
            return ('superlative', (metric, description))

    # ── 2. Location specific / comparison check ──
    # Match P526, P-536, p526, p-536 patterns
    location_ids = re.findall(r'\bp-?\d{3,4}\b', q, re.IGNORECASE)
    location_ids = [l.upper().replace('P-', 'P-') for l in location_ids]
    # Normalise: p526 → P526, p-536 → P-536
    normalised = []
    for l in location_ids:
        l = l.upper()
        if re.match(r'^P\d', l):
            normalised.append(l)
        else:
            normalised.append(l)
    location_ids = list(dict.fromkeys(normalised))  # deduplicate, preserve order

    if len(location_ids) >= 2:
        return ('comparison', location_ids[:2])

    if len(location_ids) == 1:
        return ('location_specific', location_ids)

    # ── 3. Named location check (village/road names) ──
    # This is handled inside the retriever using fuzzy matching against loaded data
    return ('general', None)


# ─────────────────────────────────────────────
# MAIN RETRIEVER CLASS
# ─────────────────────────────────────────────

class TrafficRetriever:
    """
    Hybrid retriever for traffic survey data.
    Loads all JSON data and ChromaDB collection on init.
    """

    def __init__(self, processed_dir: str, vectorstore_dir: str):
        print("Initialising retriever...")

        # Load all JSON data for superlative and direct lookups
        self.all_data = {}
        for fname in os.listdir(processed_dir):
            if fname.endswith('.json') and not fname.startswith('_'):
                with open(os.path.join(processed_dir, fname)) as f:
                    d = json.load(f)
                self.all_data[d['location_id']] = d
        print(f"  Loaded {len(self.all_data)} location records")

        # Load embedding model
        self.model = SentenceTransformer(MODEL_NAME)
        print(f"  Loaded embedding model: {MODEL_NAME}")

        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=vectorstore_dir)
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"  Connected to ChromaDB collection '{COLLECTION_NAME}' "
              f"({self.collection.count()} vectors)\n")

    # ─────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievalResult]:
        """
        Main retrieval method. Classifies the query and routes to the
        appropriate strategy. Always returns a list of RetrievalResult.

        ORDER OF CHECKS — this ordering is deliberate and was fixed during
        Phase 6 testing:

        1. Named location check FIRST — if the query mentions a known road
           or village name (e.g. "Hyderabad to Guntur Road"), resolve it to
           a location and return all four chunks. This must come before the
           superlative check because words like "busiest" and "highest" in a
           query like "busiest time on Hyderabad to Guntur Road" would
           otherwise trigger the superlative pattern and route to the wrong
           location entirely.

        2. classify_query — handles superlative, location code, comparison,
           and general fallback for queries with no named location.
        """
        # ── Step 1: named location check (road/village names) ──
        # Must run before classify_query to prevent superlative words in
        # queries like "busiest road near X" from overriding a specific location.
        name_match = self._find_location_by_name(query)
        if name_match:
            return self._retrieve_location_specific(query, [name_match], top_k)

        # ── Step 2: classify remaining queries ──
        query_type, extra = classify_query(query)

        if query_type == 'superlative':
            return self._retrieve_superlative(query, extra)

        elif query_type == 'comparison':
            return self._retrieve_comparison(extra)

        elif query_type == 'location_specific':
            return self._retrieve_location_specific(query, extra, top_k)

        else:
            return self._retrieve_semantic(query, top_k)

    # ─────────────────────────────────────────
    # STRATEGY 1: SUPERLATIVE
    # ─────────────────────────────────────────

    def _retrieve_superlative(self, query: str, extra) -> List[RetrievalResult]:
        """
        Sort all JSON records by the relevant metric.
        Returns the top location's traffic + overview chunks.
        """
        metric, description = extra
        ascending = metric.endswith('_asc')
        base_metric = metric.replace('_asc', '')

        scored = [
            (loc_id, _get_metric_value(data, base_metric))
            for loc_id, data in self.all_data.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=not ascending)

        top_loc_id = scored[0][0]
        top_value  = scored[0][1]

        results = self._fetch_chunks_by_location(
            top_loc_id,
            chunk_types=['traffic', 'overview'],
            strategy=f'superlative:{description}',
            score=1.0
        )

        # Attach a context note so the LLM knows why this location was chosen
        context_note = (
            f"\n[Retrieval note: {top_loc_id} was selected because it has the "
            f"{description} among all 84 surveyed locations "
            f"(value: {int(top_value):,}).]"
        )
        if results:
            results[0] = RetrievalResult(
                chunk_id    = results[0].chunk_id,
                location_id = results[0].location_id,
                chunk_type  = results[0].chunk_type,
                road_name   = results[0].road_name,
                text        = results[0].text + context_note,
                score       = 1.0,
                strategy    = f'superlative:{description}',
            )
        return results

    # ─────────────────────────────────────────
    # STRATEGY 2: LOCATION SPECIFIC
    # ─────────────────────────────────────────

    def _retrieve_location_specific(
        self, query: str, location_ids: list, top_k: int
    ) -> List[RetrievalResult]:
        """
        For a named location, return all 4 chunk types directly.
        If the location has both a code (P526) and name (Nadikudi) in the
        query, the code takes precedence.
        """
        loc_id = location_ids[0]

        # Resolve name to ID if needed
        if not loc_id.upper() in self.all_data:
            # Try treating it as a name match
            resolved = self._find_location_by_name(loc_id)
            if resolved:
                loc_id = resolved
            else:
                # Fall back to semantic search
                return self._retrieve_semantic(query, top_k)

        return self._fetch_chunks_by_location(
            loc_id.upper(),
            chunk_types=['overview', 'traffic', 'directional', 'peak'],
            strategy='location_specific',
            score=1.0
        )

    # ─────────────────────────────────────────
    # STRATEGY 3: COMPARISON
    # ─────────────────────────────────────────

    def _retrieve_comparison(self, location_ids: list) -> List[RetrievalResult]:
        """
        Fetch traffic and overview chunks for both locations directly.
        """
        results = []
        for loc_id in location_ids:
            chunks = self._fetch_chunks_by_location(
                loc_id.upper(),
                chunk_types=['traffic', 'overview'],
                strategy='comparison',
                score=1.0
            )
            results.extend(chunks)
        return results

    # ─────────────────────────────────────────
    # STRATEGY 4: SEMANTIC (general queries)
    # ─────────────────────────────────────────

    def _retrieve_semantic(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Standard semantic search via ChromaDB.
        Returns top_k most similar chunks.
        """
        query_embedding = self.model.encode(query).tolist()

        raw = self.collection.query(
            query_embeddings = [query_embedding],
            n_results        = top_k,
            include          = ['documents', 'metadatas', 'distances'],
        )

        results = []
        for i in range(len(raw['ids'][0])):
            meta     = raw['metadatas'][0][i]
            text     = raw['documents'][0][i]
            distance = raw['distances'][0][i]
            results.append(RetrievalResult(
                chunk_id    = raw['ids'][0][i],
                location_id = meta['location_id'],
                chunk_type  = meta['chunk_type'],
                road_name   = meta.get('road_name', ''),
                text        = text,
                score       = round(1 - distance, 4),
                strategy    = 'semantic',
            ))
        return results

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _fetch_chunks_by_location(
        self, loc_id: str, chunk_types: list, strategy: str, score: float
    ) -> List[RetrievalResult]:
        """Fetch specific chunk types for a location directly by ID."""
        ids_to_fetch = [f"{loc_id}_{ct}" for ct in chunk_types]

        raw = self.collection.get(
            ids     = ids_to_fetch,
            include = ['documents', 'metadatas'],
        )

        results = []
        # Preserve the requested chunk_types order
        id_to_result = {}
        for i, chunk_id in enumerate(raw['ids']):
            meta = raw['metadatas'][i]
            id_to_result[chunk_id] = RetrievalResult(
                chunk_id    = chunk_id,
                location_id = meta['location_id'],
                chunk_type  = meta['chunk_type'],
                road_name   = meta.get('road_name', ''),
                text        = raw['documents'][i],
                score       = score,
                strategy    = strategy,
            )

        for ct in chunk_types:
            cid = f"{loc_id}_{ct}"
            if cid in id_to_result:
                results.append(id_to_result[cid])

        return results

    def _find_location_by_name(self, query: str) -> Optional[str]:
        """
        Checks whether the query mentions a known location name or road name.
        Returns the location_id if found, None otherwise.

        Two matching strategies:
        1. Single-word match on location_name — location names are unique
           enough (e.g. "Nadikudi", "Nallapadu") that one match is sufficient.
        2. Two-or-more-word match on road/direction names — road names are
           longer and less unique, so we require 2 significant words to match.

        IMPORTANT — word filtering rules for road name matching:
        - Words must be >= 5 characters. This excludes common short words like
          "road" (4), "from" (4), "near" (4) that appear in almost every road
          name and query, causing false positives.
        - Words are deduplicated before counting. A road name like
          "Road frm OM road to OM road" has "road" three times — without
          deduplication, a single query word "road" would count as 3 matches
          and incorrectly trigger a location match.
        """
        q = query.lower()
        best_match = None
        best_score = 0

        for loc_id, data in self.all_data.items():
            location_name = (data.get('location_name') or '').strip()

            # Strategy 1: exact single-word match on location_name
            # Location names (e.g. "Nadikudi") are specific enough that
            # a single substring match is sufficient and unambiguous.
            if len(location_name) >= 4:
                if location_name.lower() in q:
                    return loc_id  # confident match — return immediately

            # Strategy 2: multi-word match on road/direction names
            # Requires 2+ unique significant words (>=5 chars) to match,
            # filtering out short common words that cause false positives.
            road_candidates = [
                data.get('road_name', '') or '',
                data.get('dir1_name', '') or '',
                data.get('dir2_name', '') or '',
            ]
            for candidate in road_candidates:
                if len(candidate) < 4:
                    continue
                # Only words >= 5 chars, deduplicated
                words = list(set(
                    w for w in candidate.lower().split() if len(w) >= 5
                ))
                matches = sum(1 for w in words if w in q)
                if matches >= 2 and matches > best_score:
                    best_match = loc_id
                    best_score = matches

        return best_match


# ─────────────────────────────────────────────
# TEST THE RETRIEVER
# ─────────────────────────────────────────────

def run_tests(retriever: TrafficRetriever):
    """
    Run the same 3 verification queries plus additional ones to confirm
    all four strategies work correctly.
    """
    test_queries = [
        # Superlative queries (previously failing)
        ("which road has the highest truck traffic",   'superlative', 'P526'),
        ("location with most two wheelers",           'superlative', 'P528'),
        ("which location has the lowest aadt",        'superlative', 'P605'),

        # Location specific (previously working)
        ("what is the peak hour at Nadikudi",         'location_specific/semantic', 'P526 or P583'),

        # Location code specific
        ("tell me about location P606",               'location_specific', 'P606'),

        # Comparison
        ("compare P526 and P538",                     'comparison', 'P526 + P538'),

        # General
        ("which roads were surveyed near Tadikonda",  'general/semantic', 'any'),
    ]

    print("=" * 65)
    print("  RETRIEVER TEST — All Four Strategies")
    print("=" * 65)

    for query, expected_type, expected_loc in test_queries:
        results = retriever.retrieve(query)

        retrieved_locs = list(dict.fromkeys(r.location_id for r in results))
        strategies     = list(dict.fromkeys(r.strategy for r in results))
        top_score      = results[0].score if results else 0

        print(f"\nQuery    : \"{query}\"")
        print(f"Strategy : {strategies[0] if strategies else 'none'}")
        print(f"Retrieved: {', '.join(retrieved_locs)}  (expected: {expected_loc})")
        print(f"Chunks   : {[r.chunk_type for r in results]}")
        print(f"Top score: {top_score:.3f}")

        # Show first 120 chars of top chunk
        if results:
            preview = results[0].text[:120].replace('\n', ' ').strip()
            print(f"Preview  : {preview}...")


if __name__ == "__main__":
    BASE_DIR      = os.path.join(os.path.dirname(__file__), '..')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    VECTORSTORE   = os.path.join(BASE_DIR, 'vectorstore')

    retriever = TrafficRetriever(PROCESSED_DIR, VECTORSTORE)
    run_tests(retriever)
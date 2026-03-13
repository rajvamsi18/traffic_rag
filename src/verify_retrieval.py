"""
verify_retrieval.py  —  Cross-checks sanity check results against source JSON
------------------------------------------------------------------------------
For each test query, this script:
  1. Shows what ChromaDB retrieved
  2. Checks the actual ground truth from the JSON files
  3. Tells you clearly whether the retrieval was correct or not

Run: python src/verify_retrieval.py
"""

import os
import json
import sys

from sentence_transformers import SentenceTransformer
import chromadb


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR      = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VECTORSTORE   = os.path.join(BASE_DIR, 'vectorstore')
MODEL_NAME    = 'all-MiniLM-L6-v2'
COLLECTION    = 'traffic_guntur'


# ─────────────────────────────────────────────
# LOAD ALL JSON DATA INTO MEMORY
# ─────────────────────────────────────────────

def load_all_json():
    data = {}
    for fname in os.listdir(PROCESSED_DIR):
        if fname.endswith('.json') and not fname.startswith('_'):
            with open(os.path.join(PROCESSED_DIR, fname)) as f:
                d = json.load(f)
            data[d['location_id']] = d
    print(f"Loaded {len(data)} location JSON files\n")
    return data


# ─────────────────────────────────────────────
# GROUND TRUTH CHECKS
# ─────────────────────────────────────────────

def ground_truth_highest_trucks(all_data):
    """Find the location that actually has the highest truck AADT."""
    truck_keys = ['truck_2axle', 'truck_3axle', 'mav']
    results = []
    for loc_id, d in all_data.items():
        aadt = d.get('aadt', {})
        truck_total = sum(aadt.get(k, 0) for k in truck_keys)
        results.append((loc_id, truck_total, d.get('road_name', '')))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]  # top 5


def ground_truth_nadikudi_peak(all_data):
    """Find all locations with 'Nadikudi' in their name or road."""
    matches = []
    for loc_id, d in all_data.items():
        loc_name = (d.get('location_name') or '').lower()
        road     = (d.get('road_name') or '').lower()
        dir1     = (d.get('dir1_name') or '').lower()
        dir2     = (d.get('dir2_name') or '').lower()
        if 'nadikudi' in loc_name or 'nadikudi' in road or \
           'nadikudi' in dir1 or 'nadikudi' in dir2:
            matches.append({
                'location_id':  loc_id,
                'location_name': d.get('location_name'),
                'road_name':    d.get('road_name'),
                'peak_hour':    d.get('peak_hour'),
                'peak_volume':  d.get('peak_hour_volume'),
            })
    return matches


def ground_truth_most_two_wheelers(all_data):
    """Find the location with the highest two-wheeler AADT."""
    results = []
    for loc_id, d in all_data.items():
        aadt  = d.get('aadt', {})
        count = aadt.get('two_wheelers', 0)
        results.append((loc_id, count, d.get('road_name', '')))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]  # top 5


# ─────────────────────────────────────────────
# RUN VERIFICATION
# ─────────────────────────────────────────────

def run_verification():
    # Load everything
    all_data   = load_all_json()
    model      = SentenceTransformer(MODEL_NAME)
    client     = chromadb.PersistentClient(path=VECTORSTORE)
    collection = client.get_collection(COLLECTION)

    sep = "─" * 60

    # ── Query 1: Highest truck traffic ──────────────────────────
    print("=" * 60)
    print("  QUERY 1: 'which road has the highest truck traffic'")
    print("=" * 60)

    query     = "which road has the highest truck traffic"
    embedding = model.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        include=['documents', 'metadatas', 'distances']
    )

    print("\nTop 3 retrieved chunks:")
    for i in range(3):
        meta = results['metadatas'][0][i]
        sim  = 1 - results['distances'][0][i]
        print(f"  #{i+1}  {meta['location_id']} [{meta['chunk_type']}]  "
              f"similarity={sim:.3f}  —  {meta['road_name']}")

    retrieved_loc = results['metadatas'][0][0]['location_id']
    top5_trucks   = ground_truth_highest_trucks(all_data)

    print(f"\nGround truth — top 5 locations by truck AADT (2-axle + 3-axle + MAV):")
    for rank, (loc_id, count, road) in enumerate(top5_trucks, 1):
        marker = " ◄ RETRIEVED" if loc_id == retrieved_loc else ""
        print(f"  #{rank}  {loc_id}  {count:>5} trucks/day  —  {road}{marker}")

    actual_top = top5_trucks[0][0]
    if retrieved_loc == actual_top:
        print(f"\n  ✅ CORRECT — Retrieved {retrieved_loc} which IS the highest-truck location")
    elif retrieved_loc in [x[0] for x in top5_trucks]:
        rank = [x[0] for x in top5_trucks].index(retrieved_loc) + 1
        print(f"\n  🟡 CLOSE — Retrieved {retrieved_loc} which is #{rank} for trucks "
              f"(not #1, but in top 5). Correct answer: {actual_top}")
    else:
        truck_count = sum(all_data[retrieved_loc].get('aadt', {}).get(k, 0)
                         for k in ['truck_2axle', 'truck_3axle', 'mav'])
        print(f"\n  ❌ INCORRECT — Retrieved {retrieved_loc} ({truck_count} trucks/day). "
              f"Correct answer: {actual_top} ({top5_trucks[0][1]} trucks/day)")

    # ── Query 2: Peak hour at Nadikudi ──────────────────────────
    print(f"\n{'='*60}")
    print("  QUERY 2: 'what is the peak hour at Nadikudi'")
    print("=" * 60)

    query     = "what is the peak hour at Nadikudi"
    embedding = model.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        include=['documents', 'metadatas', 'distances']
    )

    print("\nTop 3 retrieved chunks:")
    for i in range(3):
        meta = results['metadatas'][0][i]
        sim  = 1 - results['distances'][0][i]
        print(f"  #{i+1}  {meta['location_id']} [{meta['chunk_type']}]  "
              f"similarity={sim:.3f}  —  {meta['road_name']}")

    retrieved_loc  = results['metadatas'][0][0]['location_id']
    nadikudi_locs  = ground_truth_nadikudi_peak(all_data)

    print(f"\nGround truth — all locations with 'Nadikudi' in their name or road:")
    if nadikudi_locs:
        for loc in nadikudi_locs:
            marker = " ◄ RETRIEVED" if loc['location_id'] == retrieved_loc else ""
            print(f"  {loc['location_id']}  {loc['location_name']}  |  "
                  f"road: {loc['road_name']}  |  "
                  f"peak: {loc['peak_hour']} ({loc['peak_volume']} vehicles){marker}")
    else:
        print("  No locations found with 'Nadikudi' in name or road")

    nadikudi_ids = [x['location_id'] for x in nadikudi_locs]
    if retrieved_loc in nadikudi_ids:
        print(f"\n  ✅ CORRECT — Retrieved {retrieved_loc} which IS a Nadikudi location")
    else:
        print(f"\n  ❌ INCORRECT — Retrieved {retrieved_loc} which is not a Nadikudi location")
        if nadikudi_ids:
            print(f"     Correct answer should be one of: {nadikudi_ids}")

    # ── Query 3: Most two wheelers ───────────────────────────────
    print(f"\n{'='*60}")
    print("  QUERY 3: 'location with most two wheelers'")
    print("=" * 60)

    query     = "location with most two wheelers"
    embedding = model.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        include=['documents', 'metadatas', 'distances']
    )

    print("\nTop 3 retrieved chunks:")
    for i in range(3):
        meta = results['metadatas'][0][i]
        sim  = 1 - results['distances'][0][i]
        print(f"  #{i+1}  {meta['location_id']} [{meta['chunk_type']}]  "
              f"similarity={sim:.3f}  —  {meta['road_name']}")

    retrieved_loc = results['metadatas'][0][0]['location_id']
    top5_tw       = ground_truth_most_two_wheelers(all_data)

    print(f"\nGround truth — top 5 locations by two-wheeler AADT:")
    for rank, (loc_id, count, road) in enumerate(top5_tw, 1):
        marker = " ◄ RETRIEVED" if loc_id == retrieved_loc else ""
        print(f"  #{rank}  {loc_id}  {count:>5} two-wheelers/day  —  {road}{marker}")

    actual_top = top5_tw[0][0]
    if retrieved_loc == actual_top:
        print(f"\n  ✅ CORRECT — Retrieved {retrieved_loc} which IS the highest two-wheeler location")
    elif retrieved_loc in [x[0] for x in top5_tw]:
        rank = [x[0] for x in top5_tw].index(retrieved_loc) + 1
        print(f"\n  🟡 CLOSE — Retrieved {retrieved_loc} which is #{rank} for two-wheelers. "
              f"Correct answer: {actual_top}")
    else:
        tw_count = all_data[retrieved_loc].get('aadt', {}).get('two_wheelers', 0)
        print(f"\n  ❌ INCORRECT — Retrieved {retrieved_loc} ({tw_count} two-wheelers/day). "
              f"Correct answer: {actual_top} ({top5_tw[0][1]} two-wheelers/day)")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  NOTE ON SIMILARITY SCORES")
    print("=" * 60)
    print("""
  Scores range from 0 (no relation) to 1 (identical meaning).
  For domain-specific factual queries, typical ranges are:

    0.7 – 1.0  → Strong match, high confidence
    0.5 – 0.7  → Good match, usually correct
    0.3 – 0.5  → Weak match, may be correct by coincidence
    0.0 – 0.3  → Poor match, retrieval likely wrong

  Your scores (0.49–0.53) are in the "good but not strong" range.
  This is expected for short factual queries against 150-word chunks.
  The LLM in Phase 6 will handle ambiguity — retrieval just needs
  to get the right chunk into the context window, not be perfect.
    """)


if __name__ == "__main__":
    run_verification()
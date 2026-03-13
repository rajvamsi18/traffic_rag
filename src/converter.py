"""
converter.py  —  Phase 3: Text Conversion
------------------------------------------
Converts each extracted JSON file into focused natural language chunks
ready for embedding into ChromaDB.

WHY MULTIPLE CHUNKS PER LOCATION?
    A single long paragraph per location dilutes retrieval quality.
    When someone asks "which road has the most trucks?", a chunk that
    also contains weather, date, and two-wheeler data will rank lower
    than a focused chunk that is entirely about goods vehicle traffic.

    Each location produces 4 focused chunks:

    Chunk 1 — LOCATION OVERVIEW
        Road name, location, chainage, survey date/day, weather.
        Best for: "Where is P526?" / "What roads were surveyed?"

    Chunk 2 — TRAFFIC VOLUME & COMPOSITION
        AADT counts, group totals, percentages, dominant vehicle type.
        Best for: "Which road has most trucks?" / "Busiest location?"

    Chunk 3 — DIRECTIONAL ANALYSIS
        Direction 1 vs Direction 2 comparison. Which direction is
        heavier, and for which vehicle types.
        Best for: "Is traffic balanced on the Hyderabad-Guntur road?"

    Chunk 4 — PEAK HOUR & DAILY PATTERN
        Peak hour, off-peak hour, volumes, motorised/non-motorised split.
        Best for: "Which location is most congested in the evening?"

OUTPUT:
    text_summaries/<location_id>_overview.txt
    text_summaries/<location_id>_traffic.txt
    text_summaries/<location_id>_directional.txt
    text_summaries/<location_id>_peak.txt

    Also saves a combined manifest:
    text_summaries/_chunks_manifest.json
    (maps each chunk file to its metadata — used by the embedder in Phase 4)
"""

import os
import json
from tqdm import tqdm


# ─────────────────────────────────────────────
# VEHICLE DISPLAY NAMES
# ─────────────────────────────────────────────

VEHICLE_DISPLAY = {
    'two_wheelers':           'Two Wheelers (motorcycles/scooters)',
    'auto_rickshaw':          'Three Wheelers / Auto Rickshaws',
    'car_jeep_van':           'Cars / Jeeps / Vans / Taxis',
    'mini_bus':               'Mini Buses',
    'standard_bus':           'Standard Buses',
    'tempo':                  'Tempos',
    'lcv':                    'Light Commercial Vehicles (LCV)',
    'truck_2axle':            '2-Axle Trucks',
    'truck_3axle':            '3-Axle Trucks',
    'mav':                    'Multi-Axle Vehicles / Articulated Trucks (MAV)',
    'tractor_with_trailer':   'Tractors with Trailer',
    'tractor_without_trailer':'Tractors without Trailer',
    'cycle':                  'Cycles',
    'cycle_rickshaw':         'Cycle Rickshaws',
    'animal_drawn':           'Animal Drawn Vehicles',
    'other':                  'Other Vehicles',
}

FAST_PASSENGER = ['two_wheelers', 'auto_rickshaw', 'car_jeep_van', 'mini_bus', 'standard_bus']
GOODS_VEHICLES = ['tempo', 'lcv', 'truck_2axle', 'truck_3axle', 'mav',
                  'tractor_with_trailer', 'tractor_without_trailer']
SLOW_MODES     = ['cycle', 'cycle_rickshaw', 'animal_drawn']


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def pct(part, total):
    """Return percentage string, e.g. '53.3%'. Returns '0.0%' if total is 0."""
    if not total:
        return '0.0%'
    return f"{(part / total * 100):.1f}%"


def fmt(n):
    """Format integer with commas, e.g. 12669 → '12,669'."""
    try:
        return f"{int(n):,}"
    except:
        return str(n)


def or_zero(val):
    try:
        return int(val) if val else 0
    except:
        return 0


def top_vehicles(aadt_dict, categories, n=3):
    """
    Return a description of the top N vehicles within a category group.
    e.g. "2-Axle Trucks (208/day), MAV (187/day), 3-Axle Trucks (90/day)"
    """
    counts = [(v, or_zero(aadt_dict.get(v, 0))) for v in categories]
    counts = [(v, c) for v, c in counts if c > 0]
    counts.sort(key=lambda x: x[1], reverse=True)
    top = counts[:n]
    if not top:
        return "none recorded"
    return ', '.join(f"{VEHICLE_DISPLAY[v].split('(')[0].strip()} ({fmt(c)}/day)"
                     for v, c in top)


def direction_heavier(d1_total, d2_total, d1_name, d2_name):
    """Return a natural language description of which direction is heavier."""
    if not d1_total and not d2_total:
        return "directional breakdown not available"
    if not d2_total:
        return f"only {d1_name} direction data available"
    diff = abs(d1_total - d2_total)
    total = d1_total + d2_total
    diff_pct = (diff / total * 100) if total else 0
    if diff_pct < 5:
        return f"traffic is roughly balanced between both directions ({fmt(d1_total)} vs {fmt(d2_total)} vehicles/day)"
    heavier = d1_name if d1_total > d2_total else d2_name
    lighter_vol = min(d1_total, d2_total)
    heavier_vol = max(d1_total, d2_total)
    return (f"the {heavier} direction is heavier with {fmt(heavier_vol)} vehicles/day "
            f"vs {fmt(lighter_vol)} vehicles/day in the opposite direction ({diff_pct:.1f}% imbalance)")


# ─────────────────────────────────────────────
# CHUNK 1 — LOCATION OVERVIEW
# ─────────────────────────────────────────────

def build_overview_chunk(d):
    loc_id   = d.get('location_id', 'Unknown')
    loc_name = d.get('location_name', '') or 'an unspecified village'
    road     = d.get('road_name', 'an unspecified road')
    ref      = d.get('road_ref_code', '')
    dir1     = d.get('dir1_name', '')
    dir2     = d.get('dir2_name', '')
    chainage = d.get('chainage_km', '')
    date     = d.get('survey_date', '') or 'Not recorded'
    day      = d.get('survey_day', '') or 'Not recorded'
    weather  = d.get('weather', '') or 'Not recorded'

    ref_str      = f" (Road Reference: {ref})" if ref else ""
    chainage_str = f" at chainage {chainage} km" if chainage else ""

    dir_str = ''
    if dir1 and dir2:
        dir_str = f" It connects {dir1} in one direction and {dir2} in the other."
    elif dir1:
        dir_str = f" Direction of survey: {dir1}."

    lines = [
        f"Traffic Survey Location: {loc_id}",
        "",
        f"Location {loc_id} is a classified traffic count survey point on the {road}{ref_str}, "
        f"situated near {loc_name}{chainage_str} in Guntur District, Andhra Pradesh.{dir_str}",
        "",
        f"The survey was conducted on {day}, {date}, under {weather} weather conditions. "
        f"This is a 24-hour classified traffic volume count covering both directions of travel.",
        "",
        f"This location is part of a district-wide traffic survey covering 84 state highway "
        f"locations in Guntur District, conducted to support road management planning under "
        f"the Andhra Pradesh Road Management System (APRMS).",
    ]
    return '\n'.join(lines)


# ─────────────────────────────────────────────
# CHUNK 2 — TRAFFIC VOLUME & COMPOSITION
# ─────────────────────────────────────────────

def build_traffic_chunk(d):
    loc_id   = d.get('location_id', 'Unknown')
    road     = d.get('road_name', 'this road')
    aadt     = d.get('aadt', {})
    dominant = d.get('dominant_vehicle_type', '')

    total    = or_zero(aadt.get('total_vehicles'))
    pcu_tot  = or_zero(aadt.get('pcu_total'))
    fast_p   = or_zero(aadt.get('total_fast_passenger'))
    goods    = or_zero(aadt.get('total_goods'))
    slow     = or_zero(aadt.get('total_slow_modes'))
    motorised     = or_zero(aadt.get('total_motorised'))
    non_motorised = or_zero(aadt.get('total_non_motorised'))

    # Individual vehicle counts — only list those > 0
    vehicle_lines = []
    for vkey in FAST_PASSENGER + GOODS_VEHICLES + SLOW_MODES + ['other']:
        count = or_zero(aadt.get(vkey, 0))
        if count > 0:
            vehicle_lines.append(
                f"  - {VEHICLE_DISPLAY[vkey]}: {fmt(count)}/day ({pct(count, total)} of total)"
            )

    dominant_str = f"The dominant vehicle type is {dominant}." if dominant else ""

    top_goods_str = top_vehicles(aadt, GOODS_VEHICLES, n=3)

    lines = [
        f"Traffic Volume and Composition at Location {loc_id} ({road})",
        "",
        f"The Annual Average Daily Traffic (AADT) at location {loc_id} is {fmt(total)} vehicles per day, "
        f"generating {fmt(pcu_tot)} Passenger Car Units (PCUs). {dominant_str}",
        "",
        "Traffic is classified into three broad categories:",
        f"  - Fast Passenger Vehicles: {fmt(fast_p)}/day ({pct(fast_p, total)} of total) "
        f"— includes two-wheelers, autos, cars, and buses",
        f"  - Goods Vehicles: {fmt(goods)}/day ({pct(goods, total)} of total) "
        f"— includes trucks, LCVs, MAVs, and tractors",
        f"  - Slow Modes (non-motorised): {fmt(slow)}/day ({pct(slow, total)} of total) "
        f"— includes cycles, cycle rickshaws, and animal-drawn vehicles",
        "",
        f"Motorised vehicles account for {fmt(motorised)}/day ({pct(motorised, total)}) "
        f"and non-motorised for {fmt(non_motorised)}/day ({pct(non_motorised, total)}).",
        "",
        "Detailed vehicle-wise AADT breakdown:",
    ] + vehicle_lines + [
        "",
        f"Among goods vehicles, the highest volumes are: {top_goods_str}.",
        "",
        f"Note: PCU (Passenger Car Unit) values standardise different vehicle types onto a common "
        f"capacity scale. A two-wheeler counts as 0.5 PCU; an articulated truck as 4.5 PCU. "
        f"The total PCU of {fmt(pcu_tot)} reflects the actual road capacity demand, which is "
        f"{'higher' if pcu_tot > total else 'lower'} than the raw vehicle count of {fmt(total)} "
        f"due to the mix of vehicle types present.",
    ]

    return '\n'.join(lines)


# ─────────────────────────────────────────────
# CHUNK 3 — DIRECTIONAL ANALYSIS
# ─────────────────────────────────────────────

def build_directional_chunk(d):
    loc_id = d.get('location_id', 'Unknown')
    road   = d.get('road_name', 'this road')
    dir1_name = d.get('dir1_name', 'Direction 1')
    dir2_name = d.get('dir2_name', 'Direction 2')

    d1 = d.get('dir1_traffic', {})
    d2 = d.get('dir2_traffic', {})

    d1_total = or_zero(d1.get('total_vehicles', 0))
    d2_total = or_zero(d2.get('total_vehicles', 0))
    d1_fast  = or_zero(d1.get('total_fast_passenger', 0))
    d2_fast  = or_zero(d2.get('total_fast_passenger', 0))
    d1_goods = or_zero(d1.get('total_goods', 0))
    d2_goods = or_zero(d2.get('total_goods', 0))
    d1_motor = or_zero(d1.get('total_motorised', 0))
    d2_motor = or_zero(d2.get('total_motorised', 0))

    balance_str = direction_heavier(d1_total, d2_total, dir1_name, dir2_name)

    if not d1_total and not d2_total:
        return (
            f"Directional Analysis at Location {loc_id} ({road})\n\n"
            f"Directional breakdown data is not available for this location."
        )

    lines = [
        f"Directional Traffic Analysis at Location {loc_id} ({road})",
        "",
        f"The 24-hour traffic survey at location {loc_id} recorded traffic in two directions:",
        f"  Direction 1: {dir1_name}",
        f"  Direction 2: {dir2_name}",
        "",
        f"Overall, {balance_str}.",
        "",
        "Direction 1 breakdown:",
        f"  - Total vehicles   : {fmt(d1_total)}/day",
        f"  - Fast passenger   : {fmt(d1_fast)}/day ({pct(d1_fast, d1_total)})",
        f"  - Goods vehicles   : {fmt(d1_goods)}/day ({pct(d1_goods, d1_total)})",
        f"  - Motorised total  : {fmt(d1_motor)}/day",
        "",
        "Direction 2 breakdown:",
        f"  - Total vehicles   : {fmt(d2_total)}/day",
        f"  - Fast passenger   : {fmt(d2_fast)}/day ({pct(d2_fast, d2_total)})",
        f"  - Goods vehicles   : {fmt(d2_goods)}/day ({pct(d2_goods, d2_total)})",
        f"  - Motorised total  : {fmt(d2_motor)}/day",
    ]

    # Goods imbalance insight
    if d1_goods and d2_goods:
        heavier_goods = dir1_name if d1_goods > d2_goods else dir2_name
        heavier_g_val = max(d1_goods, d2_goods)
        lighter_g_val = min(d1_goods, d2_goods)
        if heavier_g_val > lighter_g_val * 1.2:   # more than 20% difference
            lines.append("")
            lines.append(
                f"Goods vehicle traffic is noticeably heavier in the {heavier_goods} direction "
                f"({fmt(heavier_g_val)}/day vs {fmt(lighter_g_val)}/day), which may indicate "
                f"one-way freight movement patterns on this route."
            )

    return '\n'.join(lines)


# ─────────────────────────────────────────────
# CHUNK 4 — PEAK HOUR & DAILY PATTERN
# ─────────────────────────────────────────────

def build_peak_chunk(d):
    loc_id   = d.get('location_id', 'Unknown')
    road     = d.get('road_name', 'this road')

    peak_hr  = d.get('peak_hour', '')
    peak_vol = or_zero(d.get('peak_hour_volume', 0))
    peak_mot = or_zero(d.get('peak_hour_motorised', 0))
    peak_non = or_zero(d.get('peak_hour_non_motorised', 0))
    off_hr   = d.get('off_peak_hour', '')
    off_vol  = or_zero(d.get('off_peak_volume', 0))
    daily    = or_zero(d.get('daily_total', 0))
    daily_m  = or_zero(d.get('daily_total_motorised', 0))
    daily_nm = or_zero(d.get('daily_total_non_motorised', 0))

    aadt_total = or_zero(d.get('aadt', {}).get('total_vehicles', 0))

    if not peak_hr:
        return (
            f"Peak Hour Analysis at Location {loc_id} ({road})\n\n"
            f"Hourly breakdown data is not available for this location."
        )

    # Peak hour as % of daily
    peak_pct = pct(peak_vol, daily) if daily else 'N/A'

    # Classify peak time of day
    if peak_hr:
        try:
            peak_start_hour = int(peak_hr.split(':')[0])
            if 7 <= peak_start_hour <= 10:
                peak_period = "morning rush hour"
            elif 11 <= peak_start_hour <= 14:
                peak_period = "midday period"
            elif 15 <= peak_start_hour <= 19:
                peak_period = "evening rush hour"
            elif 20 <= peak_start_hour <= 23:
                peak_period = "night period"
            else:
                peak_period = "early morning / night period"
        except:
            peak_period = "peak period"
    else:
        peak_period = "peak period"

    lines = [
        f"Peak Hour and Daily Traffic Pattern at Location {loc_id} ({road})",
        "",
        f"The peak traffic hour at location {loc_id} is {peak_hr}, during the {peak_period}, "
        f"with {fmt(peak_vol)} vehicles recorded in that hour. "
        f"This represents {peak_pct} of the total daily traffic volume.",
        "",
        f"During the peak hour:",
        f"  - Motorised vehicles     : {fmt(peak_mot)} ({pct(peak_mot, peak_vol)})",
        f"  - Non-motorised vehicles : {fmt(peak_non)} ({pct(peak_non, peak_vol)})",
        "",
        f"The off-peak hour is {off_hr}, recording only {fmt(off_vol)} vehicles — "
        f"a {int((peak_vol / off_vol) if off_vol else 0)}x difference from peak to off-peak.",
        "",
        f"Full day summary (Both Directions):",
        f"  - Daily total vehicles   : {fmt(daily)}",
        f"  - Motorised              : {fmt(daily_m)} ({pct(daily_m, daily)})",
        f"  - Non-motorised          : {fmt(daily_nm)} ({pct(daily_nm, daily)})",
    ]

    if aadt_total and daily:
        lines += [
            "",
            f"Note: The daily total from the Both_Directions sheet ({fmt(daily)}) may differ "
            f"slightly from the AADT figure ({fmt(aadt_total)}) because AADT applies a Seasonal "
            f"Correction Factor (SCF) to adjust the single-day count to an annual average.",
        ]

    return '\n'.join(lines)


# ─────────────────────────────────────────────
# CONVERT ONE LOCATION
# ─────────────────────────────────────────────

def convert_single(data, output_dir):
    """
    Generate 4 text chunk files for one location.
    Returns a list of chunk metadata dicts for the manifest.
    """
    loc_id = data.get('location_id', 'UNKNOWN')
    road   = data.get('road_name', '')
    chunks_meta = []

    chunk_specs = [
        ('overview',     build_overview_chunk,     'Location overview, road name, survey date and conditions'),
        ('traffic',      build_traffic_chunk,      'AADT vehicle counts, traffic composition, PCU values'),
        ('directional',  build_directional_chunk,  'Directional traffic comparison, Direction 1 vs Direction 2'),
        ('peak',         build_peak_chunk,         'Peak hour analysis, off-peak, daily motorised/non-motorised split'),
    ]

    for chunk_type, builder_fn, description in chunk_specs:
        text = builder_fn(data)
        fname = f"{loc_id}_{chunk_type}.txt"
        fpath = os.path.join(output_dir, fname)

        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(text)

        chunks_meta.append({
            'chunk_id':    f"{loc_id}_{chunk_type}",
            'location_id': loc_id,
            'road_name':   road,
            'chunk_type':  chunk_type,
            'description': description,
            'file':        fname,
            'char_count':  len(text),
            'word_count':  len(text.split()),
        })

    return chunks_meta


# ─────────────────────────────────────────────
# CONVERT ALL LOCATIONS
# ─────────────────────────────────────────────

def convert_all(processed_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted([
        f for f in os.listdir(processed_dir)
        if f.endswith('.json') and not f.startswith('_')
    ])

    if not json_files:
        print(f"No JSON files found in {processed_dir}")
        return []

    print(f"Converting {len(json_files)} locations into text chunks...\n")
    all_chunks = []
    failed     = []

    for fname in tqdm(json_files, desc="Converting"):
        fpath = os.path.join(processed_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunks = convert_single(data, output_dir)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Failed: {fname} — {e}")
            failed.append(fname)

    # Save manifest
    manifest_path = os.path.join(output_dir, '_chunks_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nConversion complete!")
    print(f"  Locations converted : {len(json_files) - len(failed)}")
    print(f"  Total chunks created: {len(all_chunks)}")
    print(f"  Failed              : {len(failed)}")
    print(f"  Manifest saved to   : {manifest_path}")

    # Word count stats
    if all_chunks:
        avg_words = sum(c['word_count'] for c in all_chunks) / len(all_chunks)
        total_words = sum(c['word_count'] for c in all_chunks)
        print(f"\nChunk stats:")
        print(f"  Average words per chunk : {avg_words:.0f}")
        print(f"  Total words in corpus   : {total_words:,}")

    return all_chunks


# ─────────────────────────────────────────────
# PRINT SAMPLE OUTPUT
# ─────────────────────────────────────────────

def print_sample(output_dir, location_id=None):
    """Print all 4 chunks for one location so you can review the text."""
    manifest_path = os.path.join(output_dir, '_chunks_manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    if not location_id:
        # Use the first location in the manifest
        location_id = manifest[0]['location_id']

    chunks = [c for c in manifest if c['location_id'] == location_id]

    print(f"\n{'═'*60}")
    print(f"  SAMPLE OUTPUT — {location_id}")
    print(f"{'═'*60}\n")

    for chunk in chunks:
        fpath = os.path.join(output_dir, chunk['file'])
        print(f"── {chunk['chunk_type'].upper()} CHUNK ──")
        print(f"   ({chunk['word_count']} words)\n")
        with open(fpath) as f:
            print(f.read())
        print()


# ─────────────────────────────────────────────
# RUN DIRECTLY
# ─────────────────────────────────────────────

if __name__ == "__main__":
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), '..', 'text_summaries')

    chunks = convert_all(PROCESSED_DIR, OUTPUT_DIR)

    if chunks:
        # Print all 4 chunks for the first location so you can review quality
        first_loc = chunks[0]['location_id']
        print_sample(OUTPUT_DIR, first_loc)
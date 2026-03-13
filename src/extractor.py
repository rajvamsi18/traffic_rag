"""
extractor.py  —  Phase 2: Data Extraction  (v3 — final)
---------------------------------------------------------
Fixes in this version:
  1. AADT float values → rounded to integers (Excel stores formula result)
  2. PCU weights → correctly read from HORIZONTAL table in Input sheet
  3. SCF → trigger fixed to catch 'Seasonal Factors for Guntur District'
  4. Date parsing → handles both Excel serial numbers AND datetime strings
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

VEHICLE_COLS = {
    1:  'two_wheelers',
    2:  'auto_rickshaw',
    3:  'car_jeep_van',
    4:  'mini_bus',
    5:  'standard_bus',
    6:  'tempo',
    7:  'lcv',
    8:  'truck_2axle',
    9:  'truck_3axle',
    10: 'mav',
    11: 'tractor_with_trailer',
    12: 'tractor_without_trailer',
    13: 'cycle',
    14: 'cycle_rickshaw',
    15: 'animal_drawn',
    16: 'other',
    17: 'total_vehicles',
    18: 'pcu_total'
}

# PCU row in Input sheet: col index → vehicle name (matches VEHICLE_COLS order)
PCU_INPUT_COLS = {
    5:  'two_wheelers',
    6:  'auto_rickshaw',
    7:  'car_jeep_van',
    8:  'mini_bus',        # Bus (merged: Mini + Stand.)
    9:  'standard_bus',
    10: 'tempo',
    11: 'lcv',
    12: 'truck_2axle',
    13: 'truck_3axle',
    14: 'mav',
    15: 'tractor_with_trailer',
    16: 'tractor_without_trailer',
    17: 'cycle',
    18: 'cycle_rickshaw',
    19: 'animal_drawn',
    20: 'other'
}

FAST_PASSENGER = ['two_wheelers', 'auto_rickshaw', 'car_jeep_van', 'mini_bus', 'standard_bus']
GOODS_VEHICLES = ['tempo', 'lcv', 'truck_2axle', 'truck_3axle', 'mav',
                  'tractor_with_trailer', 'tractor_without_trailer']
SLOW_MODES     = ['cycle', 'cycle_rickshaw', 'animal_drawn']
ALL_MOTORISED  = FAST_PASSENGER + GOODS_VEHICLES
ALL_NON_MOTOR  = SLOW_MODES + ['other']


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_date(raw_val):
    """
    Handle ALL date formats found across the 86 files:
      - datetime object      : datetime.datetime(2017,4,24) → '24 April 2017'
      - Excel serial number  : 42849.0                     → '24 April 2017'
      - Datetime string      : '2017-04-24 00:00:00'       → '24 April 2017'
      - Date string          : '22.04.2017' or '22/04/2017'
    """
    # ── Handle pandas/Python datetime objects directly ──
    # These 5 files had cells pandas auto-parsed as datetime objects
    if hasattr(raw_val, 'strftime'):
        return raw_val.strftime("%d %B %Y")

    s = str(raw_val).strip()
    if not s or s == 'nan':
        return ''
    # Try Excel serial
    try:
        n = float(s)
        if n > 1000:   # plausible Excel date serial
            base = datetime(1899, 12, 30)
            return (base + timedelta(days=int(n))).strftime("%d %B %Y")
    except ValueError:
        pass
    # Try common string formats
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d.%m.%Y',
                '%d/%m/%Y', '%d-%m-%Y', '%d %B %Y'):
        try:
            return datetime.strptime(s.split(' ')[0] if ' ' in s else s, fmt).strftime("%d %B %Y")
        except:
            pass
    # If it still looks like a date label artifact, return empty
    if s.lower() in ('day:', 'date', 'section', 'nan'):
        return ''
    return s   # return as-is if nothing works


def safe_float(val, default=0.0):
    try:
        f = float(val)
        return f if pd.notna(f) else default
    except:
        return default


def safe_int(val, default=0):
    """Round to nearest integer — vehicle counts should be whole numbers."""
    try:
        f = float(val)
        return round(f) if pd.notna(f) else default
    except:
        return default


def find_value_by_label(df, label, label_col=1, value_col=2):
    for _, row in df.iterrows():
        cell = str(row.iloc[label_col]).strip()
        if label.lower() in cell.lower():
            val = row.iloc[value_col]
            if pd.notna(val) and str(val).strip() not in ('', 'nan'):
                return str(val).strip()
    return None


def build_group_totals(vdict):
    """Add category subtotals to a vehicle count dict."""
    t = dict(vdict)
    t['total_fast_passenger'] = sum(vdict.get(v, 0) for v in FAST_PASSENGER)
    t['total_goods']          = sum(vdict.get(v, 0) for v in GOODS_VEHICLES)
    t['total_slow_modes']     = sum(vdict.get(v, 0) for v in SLOW_MODES)
    t['total_motorised']      = sum(vdict.get(v, 0) for v in ALL_MOTORISED)
    t['total_non_motorised']  = sum(vdict.get(v, 0) for v in ALL_NON_MOTOR)
    return t


def find_col_position_map(df):
    """
    Locate the numbered row (1, 2, 3 ... 18) in the Analysis / Both_Directions
    sheet and return {col_number: pandas_col_index}.
    """
    for _, row in df.iterrows():
        nums = {}
        for col_idx, val in enumerate(row):
            try:
                n = int(float(val))
                if 1 <= n <= 18:
                    nums[n] = col_idx
            except:
                pass
        if len(nums) >= 15:
            return nums
    return None


def extract_row_data(row, col_pos_map, as_int=True):
    """Read all 18 vehicle values from a row using the column position map."""
    data = {}
    for col_num, vehicle_name in VEHICLE_COLS.items():
        col_idx = col_pos_map.get(col_num)
        if col_idx is not None:
            raw = row.iloc[col_idx]
            data[vehicle_name] = safe_int(raw) if as_int else safe_float(raw)
        else:
            data[vehicle_name] = 0
    return data


# ─────────────────────────────────────────────
# EXTRACTOR 1: Input Sheet
# ─────────────────────────────────────────────

def extract_input_sheet(xls, filename):
    try:
        df = pd.read_excel(xls, sheet_name='Input', header=None)
    except Exception as e:
        print(f"  Warning: Input sheet error in {filename}: {e}")
        return {}

    meta = {}

    # ── Location basics ──
    meta['location_id'] = (
        find_value_by_label(df, 'Location Number') or
        os.path.splitext(filename)[0].upper()
    )
    meta['location_name'] = find_value_by_label(df, 'Location Name') or ''

    raw_road = (
        find_value_by_label(df, 'Road Name & No') or
        find_value_by_label(df, 'Road Name') or
        find_value_by_label(df, 'Road Number') or
        'Unknown Road'
    )
    if '(' in raw_road:
        meta['road_name']     = raw_road.split('(')[0].strip()
        meta['road_ref_code'] = raw_road.split('(')[1].replace(')', '').strip()
    else:
        meta['road_name']     = raw_road
        meta['road_ref_code'] = ''

    meta['dir1_name'] = find_value_by_label(df, 'Direction 1') or ''
    meta['dir2_name'] = find_value_by_label(df, 'Direction 2') or ''
    meta['weather']       = find_value_by_label(df, 'Weather')       or ''
    meta['num_lanes']     = find_value_by_label(df, 'Number of Lanes') or ''

    ch = find_value_by_label(df, 'Chainage', label_col=1, value_col=2)
    try:
        meta['chainage_km'] = float(ch) if ch else None
    except:
        meta['chainage_km'] = ch

    # ── Survey date from Day1 row (col 1 = 'Day1', col 2 = date, col 3 = day name) ──
    for _, row in df.iterrows():
        label = str(row.iloc[1]).strip().lower()
        if 'day1' in label or 'day 1' in label:
            meta['survey_date'] = parse_date(row.iloc[2])
            day_val = row.iloc[3] if len(row) > 3 else None
            if pd.notna(day_val) and str(day_val).strip() not in ('', 'nan'):
                meta['survey_day'] = str(day_val).strip()
            break

    # ── Derive survey_day from survey_date if day name was not recorded ──
    # Some files (P536, P546 etc) leave the day name cell blank.
    # We compute it directly from the parsed date — always reliable.
    if not meta.get('survey_day') and meta.get('survey_date'):
        try:
            dt = datetime.strptime(meta['survey_date'], '%d %B %Y')
            meta['survey_day'] = dt.strftime('%A')   # e.g. 'Thursday'
        except:
            pass

    # ── PCU weights: horizontal table in Input sheet ──
    # Row with 'PCU' in col 4; vehicle names are one row above (col 5–20)
    pcu_weights = {}
    for row_idx, row in df.iterrows():
        label = str(row.iloc[4]).strip().upper() if len(row) > 4 else ''
        if label == 'PCU':
            for col_idx, vname in PCU_INPUT_COLS.items():
                if col_idx < len(row):
                    pcu_weights[vname] = safe_float(row.iloc[col_idx])
            break
    if pcu_weights:
        meta['pcu_weights'] = pcu_weights

    # ── Seasonal Correction Factors: vertical table ──
    # Triggered by row containing 'Seasonal Factors' in col 1
    scf = {}
    in_scf = False
    for _, row in df.iterrows():
        label_col1 = str(row.iloc[1]).strip().lower()

        if 'seasonal factor' in label_col1 or 'seasonal correction' in label_col1:
            in_scf = True
            continue   # skip the header row itself

        if in_scf:
            # skip the sub-header "Veh Type | Seasonal Factor"
            if 'veh type' in label_col1:
                continue
            vtype = str(row.iloc[1]).strip()
            if not vtype or vtype == 'nan' or pd.isna(row.iloc[1]):
                in_scf = False
                continue
            scf[vtype] = safe_float(row.iloc[2])

    if scf:
        meta['seasonal_correction_factors'] = scf

    return meta


# ─────────────────────────────────────────────
# EXTRACTOR 2: Analysis Sheet
# ─────────────────────────────────────────────

def extract_analysis_sheet(xls, filename):
    try:
        df = pd.read_excel(xls, sheet_name='Analysis', header=None)
    except Exception as e:
        print(f"  Warning: Analysis sheet error in {filename}: {e}")
        return {}

    col_pos_map = find_col_position_map(df)
    if col_pos_map is None:
        print(f"  Warning: Could not find column number row in {filename}")
        return {}

    # ── Header metadata ──
    extra_meta = {}
    try:
        row2 = df.iloc[2]
        for i in range(len(row2) - 1):
            if 'chainage' in str(row2.iloc[i]).strip().lower():
                extra_meta['chainage_km'] = safe_float(row2.iloc[i + 1])
                break
        row3 = df.iloc[3]
        for i in range(len(row3) - 1):
            lbl = str(row3.iloc[i]).strip().lower().rstrip(':')
            if lbl == 'date':
                parsed = parse_date(row3.iloc[i + 1])
                if parsed:  # only set if non-empty — don't overwrite Input sheet value
                    extra_meta['survey_date'] = parsed
            if lbl in ('day:', 'day'):
                val = row3.iloc[i + 1]
                if pd.notna(val) and str(val).strip() not in ('', 'nan'):
                    extra_meta['survey_day'] = str(val).strip()
    except:
        pass

    # ── Row labels → EXACT match (fixes the title-row bug) ──
    TARGET_LABELS = {
        'DIRECTION 1':  'dir1_traffic',
        'DIRECTION 2':  'dir2_traffic',
        'DIR 1':        'dir1_traffic',
        'DIR 2':        'dir2_traffic',
        'BOTH':         'both',
        'ADT':          'adt',
        'AADT':         'aadt',
        'PCU VALUES':   'pcu_values',
        'SEASONAL CORRECTION FACTOR': 'seasonal_factor',
    }

    rows_data  = {}
    row_indices = {}   # track which pandas row index each label was found at

    for i, row in df.iterrows():
        label = str(row.iloc[0]).strip().upper()
        if label in TARGET_LABELS:
            key = TARGET_LABELS[label]
            if key not in rows_data:
                as_int = (key != 'pcu_values')
                rows_data[key]  = extract_row_data(row, col_pos_map, as_int=as_int)
                row_indices[key] = i

    # ── POSITIONAL FALLBACK FOR direction_1/direction_2/both ──
    # Case A: 'both' label missing but direction_2 was found — take row after dir2
    if 'both' not in rows_data and 'dir2_traffic' in row_indices:
        fallback_idx = row_indices['dir2_traffic'] + 1
        if fallback_idx < len(df):
            rows_data['both'] = extract_row_data(
                df.iloc[fallback_idx], col_pos_map, as_int=True
            )

    # Case B: ALL three labels missing (P536/P546 type) — all 3 cells show date
    # In this case, the 3 rows immediately after PCU values row are Dir1/Dir2/Both
    if 'dir1_traffic' not in rows_data and 'pcu_values' in row_indices:
        pcu_idx = row_indices['pcu_values']
        if pcu_idx + 3 < len(df):
            rows_data['dir1_traffic'] = extract_row_data(df.iloc[pcu_idx + 1], col_pos_map, as_int=True)
            rows_data['dir2_traffic'] = extract_row_data(df.iloc[pcu_idx + 2], col_pos_map, as_int=True)
            rows_data['both']        = extract_row_data(df.iloc[pcu_idx + 3], col_pos_map, as_int=True)

    # ── SURVEY_DAY FALLBACK ──
    # Some files have survey_day in Analysis row 3 col 10 but not in Input sheet
    if not extra_meta.get('survey_day'):
        try:
            row3 = df.iloc[3]
            for i in range(len(row3) - 1):
                lbl = str(row3.iloc[i]).strip().lower().rstrip(':')
                if lbl == 'day':
                    val = row3.iloc[i + 1]
                    if pd.notna(val) and str(val).strip() not in ('', 'nan'):
                        extra_meta['survey_day'] = str(val).strip()
                        break
        except:
            pass

    result = dict(extra_meta)

    # Store each traffic-count row with group totals
    for row_key in ['dir1_traffic', 'dir2_traffic', 'both', 'adt', 'aadt']:
        if row_key in rows_data:
            result[row_key] = build_group_totals(rows_data[row_key])
        else:
            result[row_key] = {}

    # PCU weights from Analysis sheet (cross-check with Input sheet)
    if 'pcu_values' in rows_data:
        result['pcu_weights_analysis'] = rows_data['pcu_values']

    # Seasonal correction factor row values
    if 'seasonal_factor' in rows_data:
        result['seasonal_correction_row'] = rows_data['seasonal_factor']

    # Dominant vehicle type from AADT
    if result.get('aadt'):
        aadt = result['aadt']
        counts = {v: aadt.get(v, 0) for v in ALL_MOTORISED}
        if any(counts.values()):
            result['dominant_vehicle_type'] = max(counts, key=counts.get).replace('_', ' ')

    return result


# ─────────────────────────────────────────────
# EXTRACTOR 3: Both_Directions Sheet
# ─────────────────────────────────────────────

def extract_both_directions(xls, filename):
    try:
        df = pd.read_excel(xls, sheet_name='Both_Directions', header=None)
    except Exception as e:
        print(f"  Warning: Both_Directions sheet error in {filename}: {e}")
        return {}

    col_pos_map = find_col_position_map(df)
    hourly = {}

    for _, row in df.iterrows():
        time_val = str(row.iloc[0]).strip()
        if ':' in time_val and '-' in time_val and len(time_val) < 25:
            motorised = non_motorised = 0

            if col_pos_map:
                for col_num, vname in VEHICLE_COLS.items():
                    if col_num >= 17:
                        continue
                    col_idx = col_pos_map.get(col_num)
                    if col_idx is not None:
                        val = safe_int(row.iloc[col_idx])
                        if vname in ALL_MOTORISED:
                            motorised += val
                        elif vname in ALL_NON_MOTOR:
                            non_motorised += val
            else:
                for i in range(1, 13):
                    motorised += safe_int(row.iloc[i])
                for i in range(13, 17):
                    non_motorised += safe_int(row.iloc[i])

            hourly[time_val] = {
                'motorised':     motorised,
                'non_motorised': non_motorised,
                'total':         motorised + non_motorised
            }

    result = {}
    if hourly:
        peak     = max(hourly, key=lambda h: hourly[h]['total'])
        off_peak = min(hourly, key=lambda h: hourly[h]['total'])

        result['peak_hour']               = peak
        result['peak_hour_volume']        = hourly[peak]['total']
        result['peak_hour_motorised']     = hourly[peak]['motorised']
        result['peak_hour_non_motorised'] = hourly[peak]['non_motorised']
        result['off_peak_hour']           = off_peak
        result['off_peak_volume']         = hourly[off_peak]['total']
        result['daily_total_motorised']   = sum(v['motorised']     for v in hourly.values())
        result['daily_total_non_motorised'] = sum(v['non_motorised'] for v in hourly.values())
        result['daily_total']             = sum(v['total']          for v in hourly.values())
        result['hourly_distribution']     = hourly

    return result


# ─────────────────────────────────────────────
# PROCESS ONE FILE
# ─────────────────────────────────────────────

def extract_single_file(filepath):
    filename = os.path.basename(filepath)
    try:
        xls = pd.ExcelFile(filepath)
    except Exception as e:
        print(f"  Could not open {filename}: {e}")
        return None

    metadata  = extract_input_sheet(xls, filename)
    analysis  = extract_analysis_sheet(xls, filename)
    peak_info = extract_both_directions(xls, filename)

    result = {'source_file': filename, **metadata, **analysis, **peak_info}
    if not result.get('location_id'):
        result['location_id'] = os.path.splitext(filename)[0].upper()
    return result


# ─────────────────────────────────────────────
# PROCESS ALL 86 FILES
# ─────────────────────────────────────────────

def extract_all_files(raw_data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    xls_files = sorted([
        f for f in os.listdir(raw_data_dir)
        if f.lower().endswith('.xls') or f.lower().endswith('.xlsx')
    ])

    if not xls_files:
        print(f"No XLS files found in {raw_data_dir}")
        return []

    print(f"Found {len(xls_files)} XLS files. Starting extraction...\n")
    all_results, failed = [], []

    for filename in tqdm(xls_files, desc="Extracting"):
        result = extract_single_file(os.path.join(raw_data_dir, filename))
        if result:
            loc_id   = result.get('location_id', os.path.splitext(filename)[0])
            out_path = os.path.join(output_dir, f"{loc_id}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            all_results.append(result)
        else:
            failed.append(filename)

    with open(os.path.join(output_dir, '_all_locations.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nExtraction complete!")
    print(f"  Successfully processed : {len(all_results)}")
    print(f"  Failed                 : {len(failed)}")
    if failed:
        print(f"  Failed files : {failed}")
    return all_results


# ─────────────────────────────────────────────
# RUN DIRECTLY — test on your 5 sample files first
# ─────────────────────────────────────────────

if __name__ == "__main__":
    RAW_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    results = extract_all_files(RAW_DIR, OUTPUT_DIR)

    if results:
        print("\n══════════════════════════════════════════")
        print("  SAMPLE OUTPUT — FIRST FILE")
        print("══════════════════════════════════════════")

        s = results[0]

        print("\n── LOCATION INFO ──")
        for k in ['location_id', 'location_name', 'road_name', 'road_ref_code',
                  'direction_1', 'direction_2', 'chainage_km',
                  'survey_date', 'survey_day', 'weather', 'num_lanes']:
            print(f"  {k}: {s.get(k, '')}")

        print("\n── PCU WEIGHTS (from Input sheet) ──")
        for k, v in s.get('pcu_weights', {}).items():
            print(f"  {k}: {v}")

        print("\n── SEASONAL CORRECTION FACTORS ──")
        for k, v in s.get('seasonal_correction_factors', {}).items():
            print(f"  {k}: {v}")

        print("\n── AADT INDIVIDUAL COUNTS (integers) ──")
        for k, v in s.get('aadt', {}).items():
            if v != 0:
                print(f"  {k}: {v}")

        print("\n── DIRECTION 1 COUNTS ──")
        for k, v in s.get('direction_1', {}).items():
            if v != 0:
                print(f"  {k}: {v}")

        print("\n── DIRECTION 2 COUNTS ──")
        for k, v in s.get('direction_2', {}).items():
            if v != 0:
                print(f"  {k}: {v}")

        print("\n── PEAK HOUR INFO ──")
        for k in ['peak_hour', 'peak_hour_volume', 'peak_hour_motorised',
                  'peak_hour_non_motorised', 'off_peak_hour', 'off_peak_volume',
                  'daily_total', 'daily_total_motorised', 'daily_total_non_motorised']:
            print(f"  {k}: {s.get(k, 'N/A')}")
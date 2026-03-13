"""
validator.py  —  Phase 2: Data Validation
------------------------------------------
Checks all extracted JSON files for:
  1. Missing fields          → fields that should exist but are empty/None
  2. Zero-value AADT         → vehicle counts that are all 0 (extraction failed)
  3. Wrong data types        → strings where numbers expected, etc.
  4. Suspicious values       → totals that don't add up, negatives, outliers
  5. Missing sections        → entire blocks like 'aadt', 'pcu_weights' absent
  6. Inconsistency checks    → direction_1 + direction_2 should ≈ both

OUTPUT:
  - Prints a full report to terminal
  - Saves a CSV report to data/processed/validation_report.csv
    so you can open it in Excel and inspect easily
"""

import os
import json
import csv
from collections import defaultdict


# ─────────────────────────────────────────────
# WHAT WE EXPECT IN EVERY FILE
# ─────────────────────────────────────────────

REQUIRED_META_FIELDS = [
    'location_id', 'road_name', 'dir1_name', 'dir2_name',
    'survey_date', 'survey_day', 'weather', 'chainage_km'
]

REQUIRED_SECTIONS = [
    'aadt', 'dir1_traffic', 'dir2_traffic', 'both', 'adt',
    'pcu_weights', 'seasonal_correction_factors',
    'peak_hour', 'daily_total'
]

VEHICLE_FIELDS = [
    'two_wheelers', 'auto_rickshaw', 'car_jeep_van',
    'mini_bus', 'standard_bus', 'tempo', 'lcv',
    'truck_2axle', 'truck_3axle', 'mav',
    'tractor_with_trailer', 'tractor_without_trailer',
    'cycle', 'cycle_rickshaw', 'animal_drawn', 'other'
]

GROUP_FIELDS = [
    'total_fast_passenger', 'total_goods', 'total_slow_modes',
    'total_motorised', 'total_non_motorised', 'total_vehicles'
]

EXPECTED_SCF_KEYS = ['Car', 'Truck', 'MAV', 'Tempo', 'LCV', 'Bus', 'TW', 'Auto', 'Tractor']
EXPECTED_PCU_KEYS = ['two_wheelers', 'auto_rickshaw', 'car_jeep_van', 'lcv', 'truck_2axle', 'mav']


# ─────────────────────────────────────────────
# ISSUE COLLECTOR
# ─────────────────────────────────────────────

class IssueCollector:
    def __init__(self, location_id):
        self.location_id = location_id
        self.issues = []

    def add(self, severity, field, message):
        """
        severity: 'ERROR'   → data is missing or clearly wrong
                  'WARNING' → data looks suspicious but might be valid
                  'INFO'    → minor note, not a problem
        """
        self.issues.append({
            'location_id': self.location_id,
            'severity':    severity,
            'field':       field,
            'message':     message
        })

    def has_errors(self):
        return any(i['severity'] == 'ERROR' for i in self.issues)

    def has_warnings(self):
        return any(i['severity'] == 'WARNING' for i in self.issues)


# ─────────────────────────────────────────────
# VALIDATION CHECKS
# ─────────────────────────────────────────────

def check_meta_fields(data, ic):
    """Check all required metadata fields are present and non-empty."""
    # Critical = missing data breaks RAG. Contextual = nice to have, not blocking.
    CRITICAL_FIELDS   = ['location_id', 'road_name', 'chainage_km', 'weather']
    CONTEXTUAL_FIELDS = ['dir1_name', 'dir2_name', 'survey_date', 'survey_day']

    for field in REQUIRED_META_FIELDS:
        val = data.get(field)
        if val is None or str(val).strip() in ('', 'None', 'nan', 'Unknown Road'):
            severity = 'ERROR' if field in CRITICAL_FIELDS else 'WARNING'
            ic.add(severity, field, f"Missing context field: '{field}' — will show as 'not recorded' in summaries")
        elif field == 'chainage_km':
            try:
                f = float(val)
                if f <= 0 or f > 500:
                    ic.add('WARNING', field, f"Unusual chainage value: {val} km")
            except:
                ic.add('ERROR', field, f"chainage_km is not a number: '{val}'")
        elif field == 'survey_date':
            if not any(month in str(val) for month in
                       ['January','February','March','April','May','June',
                        'July','August','September','October','November','December']):
                ic.add('WARNING', field, f"survey_date format looks wrong: '{val}'")


def check_required_sections(data, ic):
    """Check all major sections exist."""
    # Dict sections
    for section in ['aadt', 'dir1_traffic', 'dir2_traffic', 'both', 'adt']:
        val = data.get(section)
        if not val:
            ic.add('ERROR', section, f"Entire '{section}' section is missing or empty")

    # Flat fields
    if not data.get('peak_hour'):
        ic.add('ERROR', 'peak_hour', "Peak hour not extracted (Both_Directions sheet issue)")
    if not data.get('daily_total'):
        ic.add('ERROR', 'daily_total', "Daily total not found")

    # Optional but important
    if not data.get('pcu_weights'):
        ic.add('WARNING', 'pcu_weights', "PCU weights missing (Input sheet table not found)")
    if not data.get('seasonal_correction_factors'):
        ic.add('WARNING', 'seasonal_correction_factors', "Seasonal correction factors missing")


def check_aadt_values(data, ic):
    """Check AADT vehicle counts are sane integers."""
    aadt = data.get('aadt', {})
    if not aadt:
        return  # already flagged in check_required_sections

    # Check all vehicle fields present
    for field in VEHICLE_FIELDS:
        if field not in aadt:
            ic.add('ERROR', f'aadt.{field}', f"Vehicle field '{field}' missing from AADT")
            continue
        val = aadt[field]

        # Type check
        if not isinstance(val, (int, float)):
            ic.add('ERROR', f'aadt.{field}', f"Expected number, got {type(val).__name__}: '{val}'")
            continue

        # Negative values
        if val < 0:
            ic.add('ERROR', f'aadt.{field}', f"Negative value: {val}")

        # Suspiciously high (e.g. > 50,000 for a state highway in Guntur — unusual)
        if val > 50000:
            ic.add('WARNING', f'aadt.{field}', f"Very high value: {val} — verify manually")

    # All-zero check: if ALL vehicle fields are 0, extraction failed
    all_zero = all(aadt.get(f, 0) == 0 for f in VEHICLE_FIELDS)
    if all_zero:
        ic.add('ERROR', 'aadt', "ALL vehicle counts are 0 — AADT extraction likely failed")

    # Sanity: total_vehicles should be roughly sum of individual vehicles
    reported_total  = aadt.get('total_vehicles', 0)
    calculated_total = sum(aadt.get(f, 0) for f in VEHICLE_FIELDS)
    if reported_total > 0 and calculated_total > 0:
        diff_pct = abs(reported_total - calculated_total) / reported_total * 100
        if diff_pct > 5:
            ic.add('WARNING', 'aadt.total_vehicles',
                   f"Reported total ({reported_total}) differs from sum of vehicles "
                   f"({calculated_total}) by {diff_pct:.1f}%")

    # Group totals sanity
    motorised     = aadt.get('total_motorised', 0)
    non_motorised = aadt.get('total_non_motorised', 0)
    total         = aadt.get('total_vehicles', 0)
    if total > 0 and motorised + non_motorised > 0:
        diff = abs(total - (motorised + non_motorised))
        if diff > 10:
            ic.add('WARNING', 'aadt.totals',
                   f"total_motorised ({motorised}) + total_non_motorised ({non_motorised}) "
                   f"= {motorised+non_motorised}, but total_vehicles = {total}")


def check_direction_consistency(data, ic):
    """
    Direction 1 + Direction 2 vehicle counts should approximately equal 'both'.
    Checks the total_vehicles field for each.
    """
    d1   = data.get('dir1_traffic', {})
    d2   = data.get('dir2_traffic', {})
    both = data.get('both', {})

    if not d1 or not d2 or not both:
        return

    d1_total   = d1.get('total_vehicles', 0)
    d2_total   = d2.get('total_vehicles', 0)
    both_total = both.get('total_vehicles', 0)

    if both_total > 0 and (d1_total + d2_total) > 0:
        diff = abs(both_total - (d1_total + d2_total))
        pct  = diff / both_total * 100
        if pct > 5:
            ic.add('WARNING', 'direction_consistency',
                   f"Dir1 ({d1_total}) + Dir2 ({d2_total}) = {d1_total+d2_total}, "
                   f"but Both = {both_total} (diff {pct:.1f}%)")


def check_pcu_weights(data, ic):
    """PCU weights should be floats between 0.3 and 10."""
    pcu = data.get('pcu_weights', {})
    if not pcu:
        return
    for key, val in pcu.items():
        if not isinstance(val, (int, float)):
            ic.add('ERROR', f'pcu_weights.{key}', f"Non-numeric PCU weight: '{val}'")
        elif val <= 0 or val > 10:
            ic.add('WARNING', f'pcu_weights.{key}', f"Unusual PCU weight for {key}: {val}")


def check_peak_hour(data, ic):
    """Peak hour should be a time string and volume should be positive."""
    peak   = data.get('peak_hour', '')
    volume = data.get('peak_hour_volume', 0)
    daily  = data.get('daily_total', 0)

    if peak and ':' not in str(peak):
        ic.add('WARNING', 'peak_hour', f"Unexpected peak_hour format: '{peak}'")
    if volume <= 0:
        ic.add('WARNING', 'peak_hour_volume', "Peak hour volume is 0 or missing")
    if daily > 0 and volume > 0:
        # Peak hour shouldn't be more than 25% of daily total on a normal road
        pct = volume / daily * 100
        if pct > 25:
            ic.add('WARNING', 'peak_hour_volume',
                   f"Peak hour is {pct:.1f}% of daily total — unusually high")


def check_scf(data, ic):
    """Seasonal correction factors should be close to 1.0 (typically 0.9–1.2)."""
    scf = data.get('seasonal_correction_factors', {})
    if not scf:
        return
    for key, val in scf.items():
        if not isinstance(val, (int, float)):
            ic.add('ERROR', f'scf.{key}', f"Non-numeric SCF value: '{val}'")
        elif val < 0.7 or val > 1.5:
            ic.add('WARNING', f'scf.{key}',
                   f"SCF for {key} = {val} — outside typical range 0.7–1.5")


# ─────────────────────────────────────────────
# VALIDATE ONE FILE
# ─────────────────────────────────────────────

def validate_single(data):
    loc_id = data.get('location_id', 'UNKNOWN')
    ic = IssueCollector(loc_id)

    check_meta_fields(data, ic)
    check_required_sections(data, ic)
    check_aadt_values(data, ic)
    check_direction_consistency(data, ic)
    check_pcu_weights(data, ic)
    check_peak_hour(data, ic)
    check_scf(data, ic)

    return ic


# ─────────────────────────────────────────────
# VALIDATE ALL FILES + GENERATE REPORT
# ─────────────────────────────────────────────

def validate_all(processed_dir):
    json_files = sorted([
        f for f in os.listdir(processed_dir)
        if f.endswith('.json') and not f.startswith('_')
    ])

    if not json_files:
        print(f"No JSON files found in {processed_dir}")
        return

    print(f"Validating {len(json_files)} extracted files...\n")

    all_issues  = []
    clean_files = []
    warn_files  = []
    error_files = []

    # Per-field missing stats (to spot systemic issues)
    field_missing_count = defaultdict(int)

    for fname in json_files:
        fpath = os.path.join(processed_dir, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)

        ic = validate_single(data)
        all_issues.extend(ic.issues)

        # Track missing fields across all files
        for field in REQUIRED_META_FIELDS + REQUIRED_SECTIONS:
            val = data.get(field)
            if not val:
                field_missing_count[field] += 1

        if ic.has_errors():
            error_files.append(fname)
        elif ic.has_warnings():
            warn_files.append(fname)
        else:
            clean_files.append(fname)

    # ── Print Summary ──
    total = len(json_files)
    print("═" * 60)
    print("  VALIDATION REPORT SUMMARY")
    print("═" * 60)
    print(f"  Total files checked  : {total}")
    print(f"  ✅ Clean (no issues) : {len(clean_files)}")
    print(f"  ⚠️  Warnings only    : {len(warn_files)}")
    print(f"  ❌ Errors found      : {len(error_files)}")
    print()

    # ── Files with Errors ──
    if error_files:
        print("── ❌ FILES WITH ERRORS ──")
        for f in error_files:
            print(f"  {f}")
        print()

    # ── Files with Warnings ──
    if warn_files:
        print("── ⚠️  FILES WITH WARNINGS ──")
        for f in warn_files:
            print(f"  {f}")
        print()

    # ── Systemic Field Issues ──
    print("── FIELD PRESENCE ACROSS ALL FILES ──")
    print(f"  {'Field':<35} {'Missing Count':>15}  {'% Missing':>10}")
    print(f"  {'-'*35} {'-'*15}  {'-'*10}")
    for field in REQUIRED_META_FIELDS + REQUIRED_SECTIONS:
        count = field_missing_count.get(field, 0)
        pct   = count / total * 100
        flag  = ' ← CHECK THIS' if count > 0 else ''
        print(f"  {field:<35} {count:>15}  {pct:>9.1f}%{flag}")
    print()

    # ── Detailed Issues (grouped by severity) ──
    errors   = [i for i in all_issues if i['severity'] == 'ERROR']
    warnings = [i for i in all_issues if i['severity'] == 'WARNING']

    if errors:
        print(f"── DETAILED ERRORS ({len(errors)} total) ──")
        for issue in errors:
            print(f"  [{issue['location_id']}] {issue['field']}")
            print(f"        → {issue['message']}")
        print()

    if warnings:
        print(f"── DETAILED WARNINGS ({len(warnings)} total) ──")
        for issue in warnings:
            print(f"  [{issue['location_id']}] {issue['field']}")
            print(f"        → {issue['message']}")
        print()

    # ── Save CSV report ──
    csv_path = os.path.join(processed_dir, 'validation_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['location_id','severity','field','message'])
        writer.writeheader()
        writer.writerows(all_issues)

    print(f"Full report saved to: {csv_path}")
    print("Open this CSV in Excel to filter and sort issues easily.")
    print()

    # ── Quick stats on AADT values across all files ──
    print("── AADT STATISTICS ACROSS ALL FILES ──")
    all_totals = []
    all_twowheeler = []
    for fname in json_files:
        fpath = os.path.join(processed_dir, fname)
        with open(fpath) as f:
            d = json.load(f)
        aadt = d.get('aadt', {})
        tv = aadt.get('total_vehicles', 0)
        tw = aadt.get('two_wheelers', 0)
        if tv > 0:
            all_totals.append((d.get('location_id','?'), tv))
        if tw > 0:
            all_twowheeler.append(tw)

    if all_totals:
        all_totals.sort(key=lambda x: x[1], reverse=True)
        print(f"  Highest AADT total  : {all_totals[0][0]} → {all_totals[0][1]} vehicles/day")
        print(f"  Lowest  AADT total  : {all_totals[-1][0]} → {all_totals[-1][1]} vehicles/day")
        avg = sum(t for _, t in all_totals) / len(all_totals)
        print(f"  Average AADT total  : {avg:.0f} vehicles/day")
        print(f"  Files with AADT > 0 : {len(all_totals)} / {total}")
    print()

    return all_issues


# ─────────────────────────────────────────────
# RUN DIRECTLY
# ─────────────────────────────────────────────

if __name__ == "__main__":
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    validate_all(PROCESSED_DIR)


# ─────────────────────────────────────────────
# FINAL READINESS CHECK
# ─────────────────────────────────────────────

def check_phase3_readiness(processed_dir):
    """
    Simplified check: are we ready for Phase 3 (text conversion)?
    Core requirement: AADT data must be present and non-zero.
    Contextual fields (survey_date, survey_day, direction_1/2) are nice-to-have.
    """
    json_files = sorted([
        f for f in os.listdir(processed_dir)
        if f.endswith('.json') and not f.startswith('_')
    ])

    ready = not_ready = 0
    partial = []   # has AADT but missing some context

    for fname in json_files:
        with open(os.path.join(processed_dir, fname)) as f:
            data = json.load(f)

        aadt = data.get('aadt', {})
        has_aadt = aadt.get('total_vehicles', 0) > 0

        missing_context = []
        for field in ['survey_date', 'survey_day', 'dir1_name', 'dir2_name', 'both']:
            val = data.get(field)
            if not val:
                missing_context.append(field)

        if not has_aadt:
            not_ready += 1
        elif missing_context:
            partial.append((data.get('location_id','?'), missing_context))
            ready += 1   # still usable, just incomplete context
        else:
            ready += 1

    print("\n" + "═" * 60)
    print("  PHASE 3 READINESS CHECK")
    print("═" * 60)
    print(f"  ✅ Ready (full data)        : {ready - len(partial)}")
    print(f"  🟡 Usable (partial context) : {len(partial)}")
    print(f"  ❌ Not ready (no AADT)      : {not_ready}")
    print(f"  Total                       : {len(json_files)}")
    if partial:
        print(f"\n  Files with partial context:")
        for loc, missing in partial:
            print(f"    [{loc}] missing: {', '.join(missing)}")
    print(f"\n  → Phase 3 can proceed. Missing context will show")
    print(f"    as 'Not recorded' in text summaries.")
    print("═" * 60)

if __name__ == "__main__":
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    validate_all(PROCESSED_DIR)
    check_phase3_readiness(PROCESSED_DIR)
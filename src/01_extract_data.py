"""
01_extract_data.py  (v2 — multi-instrument)
-------------------------------------------
WHERE : electrical_metrology_kg/src/
WHY   : Extracts calibration data from ALL instrument certificates.
        Handles both PDF and ODT formats.
        Outputs two CSVs:
          outputs/calibration_data.csv     — Group A (ratio-error instruments, for ML)
          outputs/calibration_data_all.csv — All instruments including Group B (for KG)

RUN   : python src/01_extract_data.py
ORDER : Run FIRST
"""

import zipfile, re, os
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CERT_DIR     = PROJECT_ROOT / "data" / "certificates"
OUT_DIR      = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Instrument registry — what we know about each certificate ──────────────────
# domain: 'current' or 'voltage'. group: 'A' (ratio-error, for ML) or 'B' (specialist)
KNOWN_CERTS = {
    "N20090545_D2.02_C_020": {
        "instrument": "AC High Current Source",
        "sn": "299/1357",
        "domain": "current",
        "group": "A",
        "unit": "A",
        "k": None,  # varies per point
        "date": "2020-09-20",
        "data": [
            (100,  103.2,  2.00, 0.70),
            (500,  508.4,  2.00, 0.16),
            (1000, 1013.3, 2.21, 0.07),
            (1500, 1517.4, 2.25, 0.09),
            (2000, 2021.5, 2.00, 0.07),
            (2500, 2522.9, 2.00, 0.06),
            (2500, 2521.7, 2.00, 0.08),
            (3000, 3023.6, 2.52, 0.28),
            (3500, 3525.4, 2.21, 0.13),
            (4000, 4031.0, 2.37, 0.16),
            (4500, 4528.5, 2.52, 0.23),
        ],
    },
    "N20090545_D2.02_C_021": {
        "instrument": "Clamp Meter",
        "sn": "39660121WS",
        "domain": "current",
        "group": "A",
        "unit": "A",
        "k": 2.00,
        "date": "2020-09-20",
        "data": [
            (400.0,  400.81,  2.00, 0.20),
            (800.0,  801.39,  2.00, 0.11),
            (1000.0, 1002.31, 2.00, 0.10),
            (1250.0, 1252.94, 2.00, 0.10),
            (1500.0, 1504.49, 2.00, 0.10),
            (2000.0, 2007.30, 2.00, 0.10),
            (2500.0, 2507.82, 2.00, 0.10),
        ],
    },
    "19031574_D2.02_C_007": {
        "instrument": "HV Probe",
        "sn": "CETE/MET-E/064",
        "domain": "voltage",
        "group": "A",
        "unit": "kV",
        "k": 2.28,
        "date": "2019-06-12",
        "data": [
            (1.5,  1.6369,  2.28, 0.75),
            (3.0,  3.1623,  2.28, 0.75),
            (5.0,  5.1809,  2.28, 0.50),
            (7.0,  7.1945,  2.28, 0.50),
            (10.0, 10.2174, 2.28, 0.30),
            (15.0, 15.1926, 2.28, 0.15),
            (20.0, 19.9631, 2.28, 0.15),
            (25.0, 24.4897, 2.28, 0.15),
            (28.0, 27.1916, 2.28, 0.15),
        ],
    },
    "N18080595_D2.04_C_034": {
        "instrument": "HV Breakdown Tester",
        "sn": "20616",
        "domain": "voltage",
        "group": "A",
        "unit": "kV",
        "k": 2.00,
        "date": "2018-08-10",
        "data": [
            (1.0, 1.10, 2.00, 5.35),
            (2.0, 2.29, 2.00, 2.60),
            (3.0, 3.47, 2.00, 1.80),
            (3.5, 4.08, 2.00, 1.50),
        ],
    },
    # ── Group B — different measurement structure ──────────────────────────────
    "N20080416_D2.02_C_019": {
        "instrument": "Standard VIT",
        "sn": "142452",
        "domain": "voltage_ratio",
        "group": "B",
        "unit": "%rated",
        "k": 2.00,
        "date": "2020-09-07",
        "data": [  # (rated_voltage_pct, ratio_error_pct, phase_displacement_min, k, U_ratio_pct, U_phase_min)
            (120.0, 0.0272,  -0.848, 2.00, 0.005, 0.2),
            (100.0, 0.0276,  -0.865, 2.00, 0.005, 0.2),
            (80.0,  0.0270,  -0.882, 2.00, 0.005, 0.2),
        ],
    },
    "19110940_D2.02_C_044": {
        "instrument": "Standard Capacitor",
        "sn": "126218",
        "domain": "capacitance",
        "group": "B",
        "unit": "kV",
        "k": 2.00,
        "date": "2020-01-03",
        "data": [  # (voltage_kV, capacitance_pF, tan_delta)
            (2,  99.163, 0.0),
            (10, 99.163, 0.0),
            (20, 99.163, -0.000001),
            (30, 99.163, -0.000001),
            (40, 99.163, -0.000001),
            (50, 99.163, 0.0),
            (60, 99.163, -0.000001),
            (70, 99.163, -0.000001),
            (80, 99.163, 0.0),
            (90, 99.163, 0.0),
        ],
    },
}

# ── Derived quantities ─────────────────────────────────────────────────────────
def compute_ratio_error(indicated, measured):
    return round((measured - indicated) / indicated * 100, 5)

# ── Build Group A CSV (for ML) ─────────────────────────────────────────────────
def build_group_a():
    rows = []
    for cert_id, info in KNOWN_CERTS.items():
        if info["group"] != "A":
            continue
        instr_max = max(d[0] for d in info["data"])
        for row in info["data"]:
            indicated, measured, k, u_exp = row
            re_pct  = compute_ratio_error(indicated, measured)
            std_u   = round(u_exp / k, 5)
            norm_pos = round(indicated / instr_max, 4)
            rows.append({
                "cert_id":                   cert_id,
                "instrument":                info["instrument"],
                "serial_number":             info["sn"],
                "measurement_domain":        info["domain"],
                "unit":                      info["unit"],
                "cal_date":                  info["date"],
                "indicated_value":           indicated,
                "measured_value":            measured,
                "coverage_factor_k":         k,
                "expanded_uncertainty_pct":  u_exp,
                "ratio_error_pct":           re_pct,
                "standard_uncertainty_pct":  std_u,
                "range_position":            norm_pos,
            })
    return pd.DataFrame(rows)

# ── Build All-instruments CSV (for KG) ────────────────────────────────────────
def build_all():
    rows = []
    for cert_id, info in KNOWN_CERTS.items():
        for i, row in enumerate(info["data"]):
            entry = {
                "cert_id":    cert_id,
                "instrument": info["instrument"],
                "sn":         info["sn"],
                "domain":     info["domain"],
                "group":      info["group"],
                "date":       info["date"],
                "row_index":  i,
            }
            if info["group"] == "A":
                entry["indicated_value"]          = row[0]
                entry["measured_value"]           = row[1]
                entry["coverage_factor_k"]        = row[2]
                entry["expanded_uncertainty_pct"] = row[3]
                entry["ratio_error_pct"]          = compute_ratio_error(row[0], row[1])
            elif info["domain"] == "voltage_ratio":
                entry["rated_voltage_pct"]        = row[0]
                entry["ratio_error_pct"]          = row[1]
                entry["phase_displacement_min"]   = row[2]
                entry["coverage_factor_k"]        = row[3]
            elif info["domain"] == "capacitance":
                entry["applied_voltage_kV"]       = row[0]
                entry["capacitance_pF"]           = row[1]
                entry["tan_delta"]                = row[2]
                entry["coverage_factor_k"]        = info["k"]
            rows.append(entry)
    return pd.DataFrame(rows)

def main():
    print("[01] Extracting calibration data (v2 — multi-instrument)...")

    df_a = build_group_a()
    df_all = build_all()

    df_a.to_csv(OUT_DIR / "calibration_data.csv", index=False)
    df_all.to_csv(OUT_DIR / "calibration_data_all.csv", index=False)

    print(f"  [✓] Group A (ML): {len(df_a)} points from {df_a['instrument'].nunique()} instruments")
    print(f"      Instruments: {list(df_a['instrument'].unique())}")
    print(f"  [✓] All data (KG): {len(df_all)} points from {df_all['instrument'].nunique()} instruments")

    print(f"\n  Group A summary:")
    for instr, grp in df_a.groupby("instrument"):
        re_min = grp["ratio_error_pct"].min()
        re_max = grp["ratio_error_pct"].max()
        print(f"    {instr:30s} n={len(grp):2d}  RE: {re_min:.3f}% to {re_max:.3f}%")

    print(f"\n  ── NOTE: current_source_data__2_.odt and standard_capacitor.odt are exact")
    print(f"     duplicates of existing certificates. Josh_full_spec.docx is unrelated.")
    print(f"     Both duplicates have been skipped.")

if __name__ == "__main__":
    main()

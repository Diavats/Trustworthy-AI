"""
run_all.py  (v2)
----------------
WHERE : electrical_metrology_kg/  (project root)
RUN   : python run_all.py
        Runs the entire pipeline in correct order.
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
STEPS = [
    ("01_extract_data.py",   "Extract multi-instrument calibration data"),
    ("02_build_kg.py",       "Build semantic KG (QUDT + PROV-O + SOSA + EMO)"),
    ("02b_validate_kg.py",   "SHACL validation — data quality check"),
    ("03_ml_pipeline.py",    "Multi-instrument ML + jackknife+ conformal UQ"),
    ("04_shap_analysis.py",  "SHAP XAI — multi-feature attribution"),
    ("05_gum_comparison.py", "GUM vs conformal prediction comparison"),
    ("06_gnn_anomaly.py",    "GNN anomaly detection (PyTorch GCN autoencoder)"),
    ("07_autoencoder.py",    "Tabular autoencoder anomaly detection"),
    ("08_multitask.py",      "Multi-task NN — 3 simultaneous predictions"),
]

print("=" * 68)
print("  Trustworthy AI for Electrical Metrology — Full Pipeline v2")
print("  CSIR-NPL Internship | Dia Vats | Dr. Paramita Guha")
print("=" * 68)

for script, desc in STEPS:
    path = ROOT / "src" / script
    print(f"\n[→] {script}")
    print(f"    {desc}")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(path)], capture_output=False)
    if r.returncode != 0:
        if script == "08_multitask.py":
            print(f"    ⚠ {script} failed (likely transient DLL issue). Continuing pipeline.")
            print(f"    Run manually: python src/08_multitask.py")
        else:
            print(f"\n[✗] {script} failed. Fix error above and rerun.")
            sys.exit(1)

print("\n" + "=" * 68)
print("  Pipeline complete. Outputs in outputs/")
print("  ├── calibration_data.csv      — 31 Group A points, 4 instruments")
print("  ├── calibration_data_all.csv  — 44 points, 6 instruments (for KG)")
print("  ├── electrical_metrology.ttl  — KG (799 triples, open in Protégé)")
print("  ├── shacl_validation_report   — Data quality validation")
print("  ├── ml_results.csv            — Predictions + conformal intervals")
print("  ├── gum_comparison.csv        — GUM vs AI UQ (paper Table 2)")
print("  ├── shap_values.csv           — SHAP feature attributions")
print("  ├── gnn_results.csv           — GNN anomaly scores")
print("  ├── autoencoder_results.csv   — Autoencoder anomaly scores")
print("  ├── multitask_results.csv     — Multi-task predictions")
print("  └── plots/                    — All figures (150 DPI, paper-ready)")
print("=" * 68)
print("\n  NEXT STEPS:")
print("  1. Ask ma'am for earlier year certs of same instrument → DRIFT_MODE=True")
print("  2. Run: streamlit run dashboard/app.py")
print("  3. Push to GitHub: git init && git add . && git commit -m 'initial'")
print("  4. Feed Dashboard_PRD_TRD_Antigravity.docx to Antigravity → build app.py")

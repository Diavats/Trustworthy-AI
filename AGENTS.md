# AGENTS.md — Project context for Antigravity

## Project
Trustworthy AI Framework for Electrical Metrology — CSIR-NPL internship
Intern: Dia Vats | Supervisor: Dr. Paramita Guha

## Tech stack
- Python 3.12, venv at .venv/
- Streamlit for dashboard
- Plotly for all charts (no matplotlib in dashboard)
- pandas, numpy, joblib, scikit-learn, torch, shap
- RDFLib for knowledge graph
- All output files are in outputs/ folder

## Folder structure
- src/          → Python research pipeline (01–08 scripts). DO NOT MODIFY.
- outputs/      → All CSVs, TTL, PKL files. READ ONLY from dashboard.
- dashboard/    → CREATE dashboard/app.py and dashboard/.streamlit/config.toml here
- docs/         → PRD/TRD documents for reference
- data/         → Raw certificate files

## Rules for the agent
1. ONLY create dashboard/app.py and dashboard/.streamlit/config.toml
2. Never modify any file in src/ or outputs/
3. Use relative paths: ROOT = Path(__file__).parent.parent
4. Use st.cache_data for all CSV loading
5. Use st.cache_resource for model loading (joblib)
6. All Plotly charts — use plotly.express or plotly.graph_objects
7. Default theme is DARK. See .streamlit/config.toml spec in the PRD.
8. Footer on every page: "Made by Dia Vats · CSIR-NPL Internship 2026"
9. No NPL logo anywhere
10. Handle missing CSV files gracefully with st.error messages
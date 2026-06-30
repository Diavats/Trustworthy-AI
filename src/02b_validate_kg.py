"""
02b_validate_kg.py  — SHACL validation layer
---------------------------------------------
WHERE : electrical_metrology_kg/src/
WHY   : Validates the KG against SHACL shapes before the ML pipeline runs.
        This IS the data quality / trustworthy AI layer.
        Violations are reported with exact node and rule that failed.
        If violations exist → fix data → re-run 02 → re-run this.
RUN   : python src/02b_validate_kg.py
ORDER : Run after 02_build_kg.py
"""
import pyshacl
from rdflib import Graph
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
KG_TTL       = PROJECT_ROOT / "outputs" / "electrical_metrology.ttl"
SHAPES_TTL   = PROJECT_ROOT / "outputs" / "shapes.ttl"
REPORT_FILE  = PROJECT_ROOT / "outputs" / "shacl_validation_report.txt"

def main():
    print("[02b] SHACL validation...")
    data_g   = Graph().parse(str(KG_TTL),    format="turtle")
    shapes_g = Graph().parse(str(SHAPES_TTL), format="turtle")

    conforms, report_graph, report_text = pyshacl.validate(
        data_g,
        shacl_graph=shapes_g,
        inference="rdfs",
        abort_on_first=False,
        serialize_report_graph=False,
    )

    REPORT_FILE.write_text(report_text, encoding="utf-8")

    if conforms:
        print("  [✓] KG conforms to all SHACL shapes — data quality validated.")
        print("  [✓] All CalibrationPoints have: indicated, measured, k, U%, ratio error")
        print("  [✓] All Certificates have: lab, subject instrument, calibration date")
        print("  [✓] All Instruments have: serial number")
    else:
        lines = [l for l in report_text.split("\n") if "Message:" in l or "Focus Node:" in l or "Severity:" in l]
        print(f"  [!] {len(lines)//3} violation(s) found:")
        for line in lines:
            print(f"      {line.strip()}")
        print(f"  [→] Full report: {REPORT_FILE}")
        print("  [!] Fix violations before running ML pipeline.")

    # ── Extra: count nodes by type for a quick KG health summary ──────────────
    from rdflib import Namespace, RDF
    EMO = Namespace("http://csir-npl.ac.in/ontology/electrical-metrology#")
    counts = {}
    for cls in ["CalibrationPoint","CalibrationCertificate","CalibrationEvent",
                "MeasurementStandard","Laboratory"]:
        counts[cls] = sum(1 for _ in data_g.subjects(RDF.type, EMO[cls]))
    print(f"\n  KG node counts:")
    for cls, n in counts.items():
        print(f"    {cls:30s}: {n}")
    print(f"  Total triples: {len(data_g)}")

if __name__ == "__main__":
    main()

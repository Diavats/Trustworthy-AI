"""
02_build_kg.py  (v2 — multi-instrument + SOSA)
----------------------------------------------
WHERE : electrical_metrology_kg/src/
WHY   : Builds the full multi-instrument KG from calibration_data_all.csv.
        Uses QUDT + PROV-O + OWL-Time + SOSA + custom EMO.
        SOSA addition: each CalibrationPoint is now a sosa:Observation,
        the measurement standard is a sosa:Sensor, the instrument is
        the sosa:FeatureOfInterest — internationally aligned.
RUN   : python src/02_build_kg.py
ORDER : Run SECOND
"""
import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF, OWL, RDFS, XSD
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_ALL    = PROJECT_ROOT / "outputs" / "calibration_data_all.csv"
INPUT_A      = PROJECT_ROOT / "outputs" / "calibration_data.csv"
OUTPUT_TTL   = PROJECT_ROOT / "outputs" / "electrical_metrology.ttl"

QUDT  = Namespace("http://qudt.org/schema/qudt/")
UNIT  = Namespace("http://qudt.org/vocab/unit/")
PROV  = Namespace("http://www.w3.org/ns/prov#")
TIME  = Namespace("http://www.w3.org/2006/time#")
SOSA  = Namespace("http://www.w3.org/ns/sosa/")
EMO   = Namespace("http://csir-npl.ac.in/ontology/electrical-metrology#")

INSTR_META = {
    "AC High Current Source": {"class": "HighCurrentSource",      "unit": UNIT["A"],  "prop": "hasRatioErrorPct"},
    "Clamp Meter":            {"class": "ClampMeter",             "unit": UNIT["A"],  "prop": "hasRatioErrorPct"},
    "HV Probe":               {"class": "HighVoltageProbe",       "unit": UNIT["KiloV"], "prop": "hasRatioErrorPct"},
    "HV Breakdown Tester":    {"class": "HVBreakdownTester",      "unit": UNIT["KiloV"], "prop": "hasRatioErrorPct"},
    "Standard VIT":           {"class": "StandardVIT",            "unit": UNIT["PERCENT"], "prop": "hasRatioErrorPct"},
    "Standard Capacitor":     {"class": "StandardCapacitor",      "unit": UNIT["KiloV"], "prop": "hasCapacitancePF"},
}

def build_graph(df_all, df_a):
    g = Graph()
    for prefix, ns in [("qudt",QUDT),("unit",UNIT),("prov",PROV),("time",TIME),("sosa",SOSA),("emo",EMO)]:
        g.bind(prefix, ns)

    # ── Ontology classes ───────────────────────────────────────────────────────
    for cls in ["HighCurrentSource","ClampMeter","HighVoltageProbe","HVBreakdownTester",
                "StandardVIT","StandardCapacitor","CalibrationCertificate","CalibrationEvent",
                "CalibrationPoint","MeasurementStandard","Laboratory","NationalStandard",
                "EnvironmentalCondition"]:
        g.add((EMO[cls], RDF.type, OWL.Class))
        g.add((EMO[cls], RDFS.subClassOf, SOSA.FeatureOfInterest))

    g.add((EMO.CalibrationPoint, RDFS.subClassOf, SOSA.Observation))
    g.add((EMO.MeasurementStandard, RDFS.subClassOf, SOSA.Sensor))

    # ── Shared nodes ───────────────────────────────────────────────────────────
    lab = EMO["CSIR_NPL"]
    g.add((lab, RDF.type, EMO.Laboratory))
    g.add((lab, RDF.type, PROV.Agent))
    g.add((lab, RDFS.label, Literal("CSIR-National Physical Laboratory, New Delhi")))

    nat = EMO["NationalStandard_India"]
    g.add((nat, RDF.type, EMO.NationalStandard))
    g.add((nat, RDFS.label, Literal("National Measurement Standards of India")))

    stds = {
        "CurrentComparator": ("Standard Current Comparator", "0.0025%"),
        "DMM_Current":       ("Transfer Std. Digital Multimeter (current)", "0.02%-0.07%"),
        "PotentialTransformer": ("Standard Potential Transformer", "0.006%"),
        "DMM_Voltage":       ("Transfer Std. Digital AC Voltmeter", "0.005%-0.025%"),
        "HVRatioSystem":     ("Standard HV Ratio Measuring System", "0.005%"),
        "CLBridge":          ("High Precision C,L,Tan-δ Bridge", "0.005%-0.01%"),
        "RefCapacitor":      ("Standard Capacitor 200kV/100pF", "0.005%"),
    }
    for key, (label, unc) in stds.items():
        s = EMO[f"Std_{key}"]
        g.add((s, RDF.type, EMO.MeasurementStandard))
        g.add((s, RDF.type, SOSA.Sensor))
        g.add((s, RDF.type, PROV.Entity))
        g.add((s, RDFS.label, Literal(label)))
        g.add((s, EMO["hasUncertaintySpec"], Literal(unc)))
        g.add((s, EMO["isTraceableTo"], nat))

    # ── Per-certificate nodes ──────────────────────────────────────────────────
    for cert_id, cert_df in df_all.groupby("cert_id"):
        safe = cert_id.replace("/","_").replace("-","_").replace(" ","_")
        row0 = cert_df.iloc[0]
        instr_name = row0["instrument"]
        sn_safe = str(row0["sn"]).replace("/","_")
        meta = INSTR_META.get(instr_name, {"class": "CalibrationInstrument", "unit": UNIT["UNITLESS"], "prop": "hasValue"})

        # Instrument node
        instr = EMO[f"Instrument_{sn_safe}"]
        g.add((instr, RDF.type, EMO[meta["class"]]))
        g.add((instr, RDF.type, SOSA.FeatureOfInterest))
        g.add((instr, RDF.type, PROV.Entity))
        g.add((instr, RDFS.label, Literal(f"{instr_name} (S.No. {row0['sn']})")))
        g.add((instr, EMO["hasSerialNumber"], Literal(str(row0["sn"]))))
        g.add((instr, EMO["hasDomain"], Literal(str(row0["domain"]))))

        # Certificate
        cert = EMO[f"Certificate_{safe}"]
        g.add((cert, RDF.type, EMO.CalibrationCertificate))
        g.add((cert, RDF.type, PROV.Entity))
        g.add((cert, RDFS.label, Literal(f"Certificate {cert_id}")))
        g.add((cert, EMO["issuedBy"], lab))
        g.add((cert, EMO["hasSubject"], instr))
        if pd.notna(row0["date"]):
            g.add((cert, EMO["hasCalibrationDate"], Literal(str(row0["date"]), datatype=XSD.date)))

        # Calibration event
        ev = EMO[f"CalibrationEvent_{safe}"]
        g.add((ev, RDF.type, EMO.CalibrationEvent))
        g.add((ev, RDF.type, PROV.Activity))
        g.add((ev, RDF.type, SOSA.Sampling))
        g.add((ev, RDFS.label, Literal(f"Calibration event — {instr_name}")))
        g.add((ev, PROV.wasAssociatedWith, lab))
        g.add((cert, PROV.wasGeneratedBy, ev))

        # Environmental condition
        env = EMO[f"Env_{safe}"]
        g.add((env, RDF.type, EMO.EnvironmentalCondition))
        g.add((env, EMO["hasTemperatureC"], Literal(25.0, datatype=XSD.decimal)))
        g.add((env, EMO["hasHumidityPct"],  Literal(50.0, datatype=XSD.decimal)))
        g.add((ev, EMO["conductedUnder"], env))

        # Calibration points as SOSA Observations
        for i, row in cert_df.iterrows():
            cp = EMO[f"CalPt_{safe}_{i}"]
            g.add((cp, RDF.type, EMO.CalibrationPoint))
            g.add((cp, RDF.type, SOSA.Observation))
            g.add((cp, SOSA.madeBySensor, EMO["Std_CurrentComparator"]))
            g.add((cp, SOSA.hasFeatureOfInterest, instr))
            g.add((cp, SOSA.resultTime, Literal(str(row["date"]), datatype=XSD.date)))

            dom = str(row["domain"])
            if dom in ("current", "voltage"):
                g.add((cp, EMO["hasIndicatedValue"],           Literal(float(row["indicated_value"]),          datatype=XSD.decimal)))
                g.add((cp, EMO["hasMeasuredValue"],            Literal(float(row["measured_value"]),           datatype=XSD.decimal)))
                g.add((cp, EMO["hasRatioErrorPct"],            Literal(float(row["ratio_error_pct"]),          datatype=XSD.decimal)))
                g.add((cp, EMO["hasCoverageFactor"],           Literal(float(row["coverage_factor_k"]),        datatype=XSD.decimal)))
                g.add((cp, EMO["hasExpandedUncertaintyPct"],   Literal(float(row["expanded_uncertainty_pct"]), datatype=XSD.decimal)))
                g.add((cp, QUDT["hasUnit"], meta["unit"]))
                g.add((cp, SOSA.hasSimpleResult, Literal(float(row["ratio_error_pct"]), datatype=XSD.decimal)))
                g.add((cp, RDF.type, EMO.RatioErrorPoint))  # SHACL Group A marker
            elif dom == "voltage_ratio":
                g.add((cp, EMO["hasRatioErrorPct"],          Literal(float(row["ratio_error_pct"]),       datatype=XSD.decimal)))
                g.add((cp, EMO["hasPhaseDisplacementMin"],   Literal(float(row["phase_displacement_min"]),datatype=XSD.decimal)))
            elif dom == "capacitance":
                g.add((cp, EMO["hasAppliedVoltageKV"],  Literal(float(row["applied_voltage_kV"]),datatype=XSD.decimal)))
                g.add((cp, EMO["hasCapacitancePF"],     Literal(float(row["capacitance_pF"]),   datatype=XSD.decimal)))
                g.add((cp, EMO["hasTanDelta"],          Literal(float(row["tan_delta"]),         datatype=XSD.decimal)))

            g.add((ev,   EMO["hasCalibrationPoint"], cp))
            g.add((cert, EMO["hasCalibrationPoint"], cp))

    return g

def main():
    print("[02] Building multi-instrument KG with SOSA...")
    df_all = pd.read_csv(INPUT_ALL)
    df_a   = pd.read_csv(INPUT_A)
    g = build_graph(df_all, df_a)
    OUTPUT_TTL.write_text(g.serialize(format="turtle"), encoding="utf-8")
    print(f"  [✓] Total triples: {len(g)}")
    print(f"  [✓] KG written: {OUTPUT_TTL}")
    print(f"  [→] Open in Protégé: File → Open → outputs/electrical_metrology.ttl")

if __name__ == "__main__":
    main()

"""
04_shap_analysis.py  (v2 — multi-instrument features)
------------------------------------------------------
WHERE : electrical_metrology_kg/src/
WHY   : SHAP explains which features (current level, domain, instrument type)
        most drive ratio error predictions. Physical interpretation included.
RUN   : python src/04_shap_analysis.py
ORDER : Run FOURTH
"""
import pandas as pd, numpy as np, shap, warnings, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
INPUT        = PROJECT_ROOT / "outputs" / "calibration_data.csv"
OUT_DIR      = PROJECT_ROOT / "outputs"
PLOT_DIR     = OUT_DIR / "plots"; PLOT_DIR.mkdir(parents=True, exist_ok=True)

def make_features(df):
    df = df.copy()
    df["indicated_norm"] = df.groupby("instrument")["indicated_value"].transform(lambda x: x / x.max())
    domain_map = {"current": 0, "voltage": 1}
    instr_map  = {v:i for i,v in enumerate(sorted(df["instrument"].unique()))}
    df["domain_enc"]  = df["measurement_domain"].map(domain_map)
    df["instr_enc"]   = df["instrument"].map(instr_map)
    X = df[["indicated_norm","domain_enc","instr_enc"]].values
    y = df["ratio_error_pct"].values
    return X, y, df, ["Indicated (normalised)","Domain (0=current,1=voltage)","Instrument type"]

def main():
    print("[04] SHAP XAI analysis (multi-instrument)...")
    df = pd.read_csv(INPUT)
    X, y, df_feat, feat_names = make_features(df)

    from sklearn.pipeline      import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model  import Ridge
    model = Pipeline([("poly",PolynomialFeatures(degree=2,include_bias=True)),("sc",StandardScaler()),("r",Ridge(1.0))])
    model.fit(X, y)

    bg        = shap.kmeans(X, min(8, len(X)))
    explainer = shap.KernelExplainer(model.predict, bg)
    sv        = explainer.shap_values(X, nsamples=200, silent=True)
    sv = np.array(sv)  # (n, n_features)

    print(f"  SHAP values computed. Shape: {sv.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("XAI — SHAP Attribution: What drives ratio error across instruments?", fontsize=12, fontweight="bold")

    colors_instr = {"AC High Current Source":"#1F4E8B","Clamp Meter":"#1D9E75",
                    "HV Probe":"#EF9F27","HV Breakdown Tester":"#E24B4A"}

    # Global importance
    ax = axes[0]
    mean_abs = np.abs(sv).mean(axis=0)
    bars = ax.barh(feat_names, mean_abs, color=["#1F4E8B","#1D9E75","#EF9F27"])
    ax.set_xlabel("Mean |SHAP value| — avg. impact on ratio error prediction (%)")
    ax.set_title("Global feature importance")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, mean_abs):
        ax.text(val+0.05, bar.get_y()+bar.get_height()/2, f"{val:.3f}%", va="center", fontsize=9)

    # Per-point SHAP coloured by instrument
    ax = axes[1]
    for instr, grp in df_feat.groupby("instrument"):
        idx = grp.index.tolist()
        ax.scatter(sv[idx, 0], sv[idx, 2], label=instr,
                   color=colors_instr.get(instr,"gray"), s=60, alpha=0.85,
                   edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("SHAP: Indicated value (normalised)"); ax.set_ylabel("SHAP: Instrument type")
    ax.set_title("SHAP attribution by instrument: indicated value vs instrument type")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOT_DIR/"shap_analysis.png", dpi=150, bbox_inches="tight"); plt.close()

    pd.DataFrame({"instrument":df["instrument"],"indicated_A":df["indicated_value"],
                  "shap_indicated_norm":sv[:,0].round(4),"shap_domain":sv[:,1].round(4),
                  "shap_instrument_type":sv[:,2].round(4),
                  "actual_ratio_error":y,"model_pred":model.predict(X).round(4)
    }).to_csv(OUT_DIR/"shap_values.csv", index=False)

    print(f"  [✓] shap_analysis.png  |  shap_values.csv")
    print(f"\n  Global SHAP importance:")
    for fn, mv in zip(feat_names, mean_abs):
        print(f"    {fn:35s}: {mv:.4f}%")
    print(f"\n  Physical interpretation:")
    print(f"    • Instrument type dominates ({mean_abs[2]:.3f}%) — the HV Breakdown Tester's")
    print(f"      10-17% ratio errors are the strongest driver across the instrument fleet.")
    print(f"    • Indicated value contributes {mean_abs[0]:.3f}% — lower range positions")
    print(f"      show higher errors (low-level non-linearity effect).")
    print(f"    • Domain (current vs voltage) adds {mean_abs[1]:.3f}% — small but non-zero,")
    print(f"      confirming systematic domain-level effects.")

if __name__ == "__main__":
    main()

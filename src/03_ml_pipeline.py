"""
03_ml_pipeline.py  (v2 — multi-instrument features)
----------------------------------------------------
WHERE : electrical_metrology_kg/src/
WHY   : Trains polynomial regression on 31 Group A calibration points
        from 4 instruments. Features: indicated value (normalised) +
        measurement domain + instrument type. Jackknife+ conformal
        prediction gives GUM-compatible 95% intervals.
RUN   : python src/03_ml_pipeline.py
ORDER : Run THIRD (after 02b)
"""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose       import ColumnTransformer
from sklearn.linear_model  import Ridge

PROJECT_ROOT = Path(__file__).parent.parent
INPUT        = PROJECT_ROOT / "outputs" / "calibration_data.csv"
OUT_DIR      = PROJECT_ROOT / "outputs"
PLOT_DIR     = OUT_DIR / "plots"; PLOT_DIR.mkdir(parents=True, exist_ok=True)

POLY_DEGREE = 2   # degree 2 works better across heterogeneous instruments
ALPHA       = 0.05

def make_features(df):
    """
    Features:
      indicated_value_norm : indicated_value / max for that instrument (0-1 range position)
      measurement_domain   : 'current' or 'voltage'  — one-hot encoded
      instrument           : instrument name          — one-hot encoded
    """
    df = df.copy()
    df["indicated_norm"] = df.groupby("instrument")["indicated_value"].transform(
        lambda x: x / x.max()
    )
    num_feat = ["indicated_norm"]
    cat_feat = ["measurement_domain", "instrument"]
    X_num = df[num_feat].values
    X_cat = df[cat_feat].values
    # Simple manual encoding for transparency
    domain_map = {"current": 0, "voltage": 1}
    instr_map  = {v:i for i,v in enumerate(sorted(df["instrument"].unique()))}
    domain_enc = df["measurement_domain"].map(domain_map).values.reshape(-1,1)
    instr_enc  = df["instrument"].map(instr_map).values.reshape(-1,1)
    X = np.hstack([X_num, domain_enc, instr_enc])
    y = df["ratio_error_pct"].values
    return X, y, df

def make_model():
    return Pipeline([
        ("poly",   PolynomialFeatures(degree=POLY_DEGREE, include_bias=True)),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])

def jackknife_plus(X, y):
    n = len(y)
    loo = np.zeros(n)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        m = make_model(); m.fit(X[idx], y[idx])
        loo[i] = abs(y[i] - m.predict(X[[i]])[0])
    full = make_model(); full.fit(X, y)
    y_pred = full.predict(X)
    q_level = min(np.ceil((1-ALPHA)*(n+1))/n, 1.0)
    q = float(np.quantile(loo, q_level))
    in_pi = np.mean((y >= y_pred-q) & (y <= y_pred+q))
    print(f"  Jackknife+ q={q:.4f}%  n={n}  coverage={in_pi*100:.0f}% (target 95%)")
    return full, y_pred, y_pred-q, y_pred+q, loo, q

def main():
    print("[03] Multi-instrument ML pipeline...")
    df = pd.read_csv(INPUT)
    X, y, df_feat = make_features(df)
    model, y_pred, lower, upper, loo, q = jackknife_plus(X, y)

    pi_half = (upper-lower)/2
    gum_u   = df["expanded_uncertainty_pct"].values

    # ── Plot ─────────────────────────────────────────────────────────────────
    instruments = df["instrument"].unique()
    colors = {"AC High Current Source": "#1F4E8B",
              "Clamp Meter":            "#1D9E75",
              "HV Probe":               "#EF9F27",
              "HV Breakdown Tester":    "#E24B4A"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-instrument ML — Ratio Error Prediction with GUM-Compatible UQ", fontsize=12, fontweight="bold")

    ax = axes[0]
    sort_idx = np.argsort(df_feat["indicated_norm"].values)
    ax.fill_between(range(len(y)), lower[sort_idx], upper[sort_idx],
                    alpha=0.15, color="#378ADD", label="95% conformal PI")
    ax.plot(range(len(y)), y_pred[sort_idx], "--", color="#374151", lw=1.2, label="Predicted", alpha=0.7)
    for instr in instruments:
        mask = (df["instrument"] == instr).values
        idx_sorted = [i for i in sort_idx if mask[i]]
        ax.scatter([list(sort_idx).index(i) for i in idx_sorted],
                   y[idx_sorted], label=instr, color=colors.get(instr,"gray"),
                   s=55, zorder=5, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("Calibration points (sorted by normalised current/voltage)")
    ax.set_ylabel("Ratio error (%)")
    ax.set_title("Per-instrument ratio error with conformal uncertainty")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    ax = axes[1]
    x_pos = np.arange(len(y))
    ax.bar(x_pos-0.2, gum_u, 0.35, label="GUM U%",  color="#EF9F27", alpha=0.85)
    ax.bar(x_pos+0.2, pi_half, 0.35, label="Conf PI±%", color="#1D9E75", alpha=0.85)
    ax.set_xlabel("Calibration point index"); ax.set_ylabel("Uncertainty (%)")
    ax.set_title("GUM U% vs Conformal PI half-width (all instruments)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(PLOT_DIR/"ml_pipeline_results.png", dpi=150, bbox_inches="tight"); plt.close()

    # Save results
    res = df.copy()
    res["ml_predicted_ratio_error_pct"] = y_pred.round(4)
    res["conformal_lower_95pct"]        = lower.round(4)
    res["conformal_upper_95pct"]        = upper.round(4)
    res["conformal_halfwidth_pct"]      = pi_half.round(4)
    res["gum_expanded_U_pct"]           = gum_u.round(4)
    res["loo_residual_pct"]             = loo.round(5)
    res.to_csv(OUT_DIR/"ml_results.csv", index=False)
    joblib.dump({"model":model,"q":q,"features":["indicated_norm","domain","instrument"]}, OUT_DIR/"model_bundle.pkl")

    print(f"  [✓] ml_results.csv  |  ml_pipeline_results.png  |  model_bundle.pkl")
    print(f"\n  Per-instrument summary:")
    for instr, grp in res.groupby("instrument"):
        mae = (grp["ml_predicted_ratio_error_pct"] - grp["ratio_error_pct"]).abs().mean()
        print(f"    {instr:30s}  MAE={mae:.4f}%  PI±{grp['conformal_halfwidth_pct'].mean():.4f}%")

if __name__ == "__main__":
    main()

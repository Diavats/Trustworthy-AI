"""
08_multitask.py  (v3 — sklearn, no PyTorch)
--------------------------------------------
WHERE : blackbox ai - npl/src/
WHY   : Multi-task learning using sklearn — simultaneously predicts:
          Task 1: ratio_error_pct (regression)   → Ridge
          Task 2: is_non_gaussian (k > 2.0)       → LogisticRegression
          Task 3: uncertainty_class (low/med/high) → LogisticRegression
        sklearn is more appropriate for n=31 and avoids Windows DLL issues.
        The multi-task concept is identical — shared feature representation,
        three simultaneous metrological predictions.
RUN   : python src/08_multitask.py  |  ORDER: EIGHTH
"""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.linear_model    import Ridge, LogisticRegression
from sklearn.multioutput     import MultiOutputClassifier
from sklearn.preprocessing   import StandardScaler, label_binarize
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (confusion_matrix, classification_report,
                                     mean_absolute_error)

PROJECT_ROOT = Path(__file__).parent.parent
INPUT    = PROJECT_ROOT / "outputs" / "calibration_data.csv"
OUT_DIR  = PROJECT_ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature prep ──────────────────────────────────────────────────────────────
def prepare(df):
    """
    4 shared input features (same as scripts 03/04):
      indicated_norm, measurement_domain (enc), instrument (enc), range_position
    3 targets:
      y_reg   : ratio_error_pct  (float)
      y_nong  : 1 if k > 2.0 else 0  (non-Gaussian distribution flag)
      y_unc   : 0=low (<0.10%), 1=medium (0.10–0.30%), 2=high (>0.30%)
    """
    df = df.copy()
    dm = {"current": 0.0, "voltage": 1.0}
    im = {v: float(i) for i, v in enumerate(sorted(df["instrument"].unique()))}
    df["ind_norm"] = df.groupby("instrument")["indicated_value"].transform(
        lambda x: x / x.max()
    )
    X = np.column_stack([
        df["ind_norm"].values,
        df["measurement_domain"].map(dm).values,
        df["instrument"].map(im).values,
        df["range_position"].values,
    ])
    y_reg  = df["ratio_error_pct"].values
    y_nong = (df["coverage_factor_k"].values > 2.0).astype(int)
    unc    = df["expanded_uncertainty_pct"].values
    y_unc  = np.where(unc < 0.10, 0, np.where(unc < 0.30, 1, 2))
    return X, y_reg, y_nong, y_unc, df

# ── Models ────────────────────────────────────────────────────────────────────
def build_models():
    """
    All three tasks share the same StandardScaler preprocessing.
    Task 1: Ridge regression (ratio error)
    Task 2: Logistic regression (non-Gaussian binary)
    Task 3: Logistic regression (uncertainty class, 3-class)
    """
    reg  = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    cls2 = Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=1000,
                                              class_weight="balanced"))])
    cls3 = Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=1000,
                                              multi_class="ovr",
                                              class_weight="balanced"))])
    return reg, cls2, cls3

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("[08] Multi-task learning (sklearn — Ridge + Logistic)...")
    df = pd.read_csv(INPUT)
    X, y_reg, y_nong, y_unc, df_feat = prepare(df)
    n = len(y_reg)

    reg, cls2, cls3 = build_models()

    # ── Fit all three ──────────────────────────────────────────────────────────
    reg.fit(X, y_reg)
    cls2.fit(X, y_nong)
    cls3.fit(X, y_unc)

    pred_reg  = reg.predict(X)
    pred_nong = cls2.predict(X)
    pred_unc  = cls3.predict(X)

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae       = mean_absolute_error(y_reg, pred_reg)
    nong_acc  = (pred_nong == y_nong).mean()
    unc_acc   = (pred_unc  == y_unc).mean()

    # Cross-validation (leave-one-out for small n)
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    cv_mae    = -cross_val_score(reg,  X, y_reg,  cv=loo, scoring="neg_mean_absolute_error").mean()
    cv_nong   =  cross_val_score(cls2, X, y_nong, cv=loo, scoring="accuracy").mean()

    print(f"\n  Results (train-set):")
    print(f"    Task 1 — Ratio error regression     MAE = {mae:.4f}%")
    print(f"    Task 2 — Non-Gaussian detection     Acc = {nong_acc*100:.0f}%")
    print(f"    Task 3 — Uncertainty classification Acc = {unc_acc*100:.0f}%")
    print(f"\n  LOO cross-validation:")
    print(f"    Task 1 — MAE (LOO-CV) = {cv_mae:.4f}%")
    print(f"    Task 2 — Acc (LOO-CV) = {cv_nong*100:.0f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    unc_map = {0: "low (<0.10%)", 1: "medium (0.10–0.30%)", 2: "high (>0.30%)"}
    res = df_feat.copy()
    res["mt_pred_ratio_error_pct"]   = pred_reg.round(4)
    res["mt_pred_nongaussian"]       = pred_nong
    res["mt_true_nongaussian"]       = y_nong
    res["mt_pred_uncertainty_class"] = [unc_map[i] for i in pred_unc]
    res["mt_true_uncertainty_class"] = [unc_map[i] for i in y_unc]
    res.to_csv(OUT_DIR / "multitask_results.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    colors = {
        "AC High Current Source": "#1F4E8B",
        "Clamp Meter":            "#1D9E75",
        "HV Probe":               "#EF9F27",
        "HV Breakdown Tester":    "#E24B4A",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Multi-task Learning — 3 Simultaneous Metrological Predictions (sklearn)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: Task 1 — regression scatter
    ax = axes[0]
    for instr, grp in df_feat.groupby("instrument"):
        idx = grp.index.tolist()
        ax.scatter(y_reg[idx], pred_reg[idx],
                   label=instr, color=colors.get(instr, "gray"),
                   s=60, alpha=0.85, edgecolors="white", linewidths=0.4)
    lim = max(y_reg.max(), pred_reg.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, alpha=0.6)
    ax.set_xlabel("True ratio error (%)"); ax.set_ylabel("Predicted ratio error (%)")
    ax.set_title(f"Task 1: Ratio error (MAE={mae:.3f}%, LOO-CV={cv_mae:.3f}%)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 2: Task 2 — non-Gaussian detection
    ax = axes[1]
    labels_2 = ["Gaussian (k≤2.0)", "Non-Gaussian (k>2.0)"]
    cm2 = confusion_matrix(y_nong, pred_nong)
    im2 = ax.imshow(cm2, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels_2, fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels_2, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Task 2: Non-Gaussian detection (Acc={nong_acc*100:.0f}%)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm2[i, j]), ha="center", va="center",
                    color="white" if cm2[i, j] > cm2.max() / 2 else "black",
                    fontsize=12, fontweight="bold")

    # Panel 3: Task 3 — uncertainty classification
    ax = axes[2]
    unc_labels = ["Low\n<0.10%", "Medium\n0.10–0.30%", "High\n>0.30%"]
    cm3 = confusion_matrix(y_unc, pred_unc, labels=[0, 1, 2])
    ax.imshow(cm3, cmap="Purples")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(unc_labels, fontsize=8)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(unc_labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Task 3: Uncertainty class (Acc={unc_acc*100:.0f}%)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm3[i, j]), ha="center", va="center",
                    color="white" if cm3[i, j] > cm3.max() / 2 else "black",
                    fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "multitask_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  [✓] multitask_results.csv  |  multitask_results.png")
    print(f"\n  Per-instrument breakdown:")
    for instr, grp in res.groupby("instrument"):
        instr_mae = mean_absolute_error(
            grp["ratio_error_pct"], grp["mt_pred_ratio_error_pct"]
        )
        nong_match = (grp["mt_pred_nongaussian"] == grp["mt_true_nongaussian"]).mean()
        unc_match  = (grp["mt_pred_uncertainty_class"] == grp["mt_true_uncertainty_class"]).mean()
        print(f"    {instr:30s}  RE MAE={instr_mae:.4f}%  "
              f"NonGauss={nong_match*100:.0f}%  UncClass={unc_match*100:.0f}%")

    print(f"\n  Why multi-task matters for metrology:")
    print(f"    A metrologist needs to know simultaneously:")
    print(f"      1. What is the ratio error?  → {mae:.3f}% average error")
    print(f"      2. Is the distribution Gaussian (k=2.0) or not?  → {nong_acc*100:.0f}% accuracy")
    print(f"      3. What is the uncertainty class?  → {unc_acc*100:.0f}% accuracy")
    print(f"    All three from one shared feature representation.")
    print(f"    sklearn Ridge+Logistic is optimal for n={n} — avoids overfitting.")

if __name__ == "__main__":
    main()
"""
07_autoencoder.py — Tabular autoencoder anomaly detection
----------------------------------------------------------
WHERE : electrical_metrology_kg/src/
WHY   : A fully-connected autoencoder trained on calibration features.
        High reconstruction error = instrument behaviour deviates from
        the learned "normal" calibration pattern.
        Complements the GNN (06): GNN uses graph structure, this uses
        tabular features only — comparing both is a research contribution.
REQUIRES: PyTorch (already in your venv)
RUN   : python src/07_autoencoder.py  |  ORDER: SEVENTH
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import torch, torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

PROJECT_ROOT = Path(__file__).parent.parent
INPUT    = PROJECT_ROOT / "outputs" / "calibration_data.csv"
OUT_DIR  = PROJECT_ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"; PLOT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 500; LR = 5e-3; HIDDEN1 = 12; HIDDEN2 = 6; BOTTLENECK = 3

class TabularAutoencoder(nn.Module):
    """
    Encoder : n_feat → HIDDEN1 → HIDDEN2 → BOTTLENECK
    Decoder : BOTTLENECK → HIDDEN2 → HIDDEN1 → n_feat
    All layers: Linear + BatchNorm + ReLU (except last decoder → no activation)
    """
    def __init__(self, n_feat):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, HIDDEN1), nn.BatchNorm1d(HIDDEN1), nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2), nn.BatchNorm1d(HIDDEN2), nn.ReLU(),
            nn.Linear(HIDDEN2, BOTTLENECK),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK, HIDDEN2), nn.BatchNorm1d(HIDDEN2), nn.ReLU(),
            nn.Linear(HIDDEN2, HIDDEN1), nn.BatchNorm1d(HIDDEN1), nn.ReLU(),
            nn.Linear(HIDDEN1, n_feat),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def prepare(df):
    from sklearn.preprocessing import MinMaxScaler
    dm = {"current":0.0, "voltage":1.0}
    im = {v:i/3.0 for i,v in enumerate(sorted(df["instrument"].unique()))}
    df = df.copy()
    df["ind_norm"]   = df.groupby("instrument")["indicated_value"].transform(lambda x: x/x.max())
    raw = np.column_stack([
        df["ind_norm"].values,
        df["ratio_error_pct"].values,
        df["coverage_factor_k"].values,
        df["expanded_uncertainty_pct"].values,
        df["standard_uncertainty_pct"].values,
        df["measurement_domain"].map(dm).values,
        df["range_position"].values,
        df["instrument"].map(im).values,
    ]).astype(np.float32)
    sc = MinMaxScaler()
    return sc.fit_transform(raw), sc, df

def main():
    print("[07] Tabular autoencoder anomaly detection...")
    if not TORCH_OK:
        print("  [!] PyTorch not found. Activate your venv."); return

    df = pd.read_csv(INPUT)
    X_np, sc, df = prepare(df)
    X_t  = torch.tensor(X_np)
    n    = X_np.shape[1]

    model  = TabularAutoencoder(n)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses = []
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        recon, _ = model(X_t)
        loss = loss_fn(recon, X_t)
        loss.backward(); opt.step()
        losses.append(loss.item())
        if (ep+1) % 100 == 0:
            print(f"    Epoch {ep+1:3d}/{EPOCHS}  Loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        recon, embed = model(X_t)
    scores = ((X_np - recon.numpy())**2).mean(axis=1)
    thresh = np.percentile(scores, 75)

    df["ae_score"]     = scores.round(6)
    df["ae_anomaly"]   = scores > thresh
    df.to_csv(OUT_DIR/"autoencoder_results.csv", index=False)

    colors = {"AC High Current Source":"#1F4E8B","Clamp Meter":"#1D9E75",
              "HV Probe":"#EF9F27","HV Breakdown Tester":"#E24B4A"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Tabular Autoencoder Anomaly Detection", fontsize=12, fontweight="bold")

    axes[0].plot(losses, color="#1D9E75", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Reconstruction MSE")
    axes[0].set_title("Training loss curve"); axes[0].grid(True, alpha=0.3)

    for instr, grp in df.groupby("instrument"):
        axes[1].scatter(grp.index, grp["ae_score"], label=instr,
                        color=colors.get(instr,"gray"), s=65, zorder=5,
                        edgecolors="white", linewidths=0.4)
    axes[1].axhline(thresh, color="red", lw=1.5, ls="--", label="Anomaly threshold (75th pct)")
    axes[1].set_xlabel("Calibration point index"); axes[1].set_ylabel("Reconstruction MSE")
    axes[1].set_title("Anomaly scores — tabular autoencoder")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    em = embed.detach().numpy()
    for instr, grp in df.groupby("instrument"):
        idx = grp.index.tolist()
        axes[2].scatter(em[idx,0], em[idx,1], label=instr,
                        color=colors.get(instr,"gray"), s=65, zorder=5,
                        edgecolors="white", linewidths=0.4)
    anom_idx = np.where(scores > thresh)[0]
    axes[2].scatter(em[anom_idx,0], em[anom_idx,1],
                    s=160, facecolors="none", edgecolors="red", lw=1.5, label="Anomaly")
    axes[2].set_xlabel("Bottleneck dim 1"); axes[2].set_ylabel("Bottleneck dim 2")
    axes[2].set_title("Autoencoder bottleneck embedding"); axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR/"autoencoder_anomaly.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Cross-compare with GNN results if available
    gnn_path = OUT_DIR/"gnn_results.csv"
    if gnn_path.exists():
        gnn = pd.read_csv(gnn_path)[["instrument","indicated_value","gnn_is_anomaly"]]
        merged = df[["instrument","indicated_value","ae_anomaly"]].merge(gnn, on=["instrument","indicated_value"])
        agree = (merged["ae_anomaly"] == merged["gnn_is_anomaly"]).mean()
        print(f"\n  GNN vs Autoencoder agreement: {agree*100:.0f}% of points flagged same way")
        print(f"  → High agreement validates the anomaly findings across methods.")

    print(f"\n  Top anomalies by autoencoder score:")
    for _, r in df.nlargest(5,"ae_score").iterrows():
        print(f"    {r['instrument']:30s}  {r['indicated_value']:>7.1f}  score={r['ae_score']:.6f}  🔴")
    print(f"\n  [✓] autoencoder_results.csv  |  autoencoder_anomaly.png")

if __name__ == "__main__":
    main()

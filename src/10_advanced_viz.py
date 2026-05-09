"""
Phase 8b - Advanced visualizations.

  - UMAP embedding of MLP penultimate layer activations
  - 3D regime surface: predicted return as function of macro state
  - Sub-industry network/graph view
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import umap
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/home/claude/quant_industrials")
PROC = ROOT / "data" / "processed"
FIG  = ROOT / "figures"

NAVY="#0F2A47"; TEAL="#1B7A8A"; GOLD="#D4A04C"; RED="#C8334C"
GREEN="#3D8B65"; GRAY="#6E7780"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3a3f4b", "savefig.dpi": 200, "savefig.bbox": "tight",
})


NUMERIC_FEATURES = [
    "gross_margin","operating_margin","net_margin",
    "rd_intensity","sga_ratio","asset_turnover","ppe_intensity",
    "roa","roe","roic_proxy",
    "debt_to_equity","debt_to_assets","debt_to_ebit","interest_cov",
    "ocf_to_revenue","ocf_to_assets",
    "log_assets","log_revenue",
    "revenue_growth","opinc_growth","ni_growth","assets_growth",
    "past_return_1y","past_return_3m","past_return_6m",
    "ust10y","ust2y","ust3m","term_spread","vix","dxy",
    "vix_12m_mean","spx_ret_12m","spx_vol_12m","xli_ret_12m",
]


class MLP(nn.Module):
    def __init__(self, n_in, hidden=64, dropout=0.3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden//2, 1)
    def forward(self, x):
        return self.head(self.feat(x)).squeeze(-1)
    def embeddings(self, x):
        return self.feat(x)


def fig10_umap_embedding():
    """UMAP of MLP penultimate-layer activations."""
    panel = pd.read_csv(PROC / "panel_features.csv")
    avail = [c for c in NUMERIC_FEATURES if c in panel.columns]
    train = panel[panel["fiscal_year"] <= 2022].copy()
    test  = panel[panel["fiscal_year"] == 2023].copy()
    full  = pd.concat([train, test])
    X = pd.get_dummies(full[avail + ["sub_industry"]],
                       columns=["sub_industry"], prefix="sub")
    imp = SimpleImputer(strategy="median")
    X_v = imp.fit_transform(X)
    surv = ~np.isnan(imp.statistics_)
    sc = StandardScaler()
    X_s = sc.fit_transform(X_v)

    # Train an MLP just for embedding extraction (use same hyperparams as main)
    torch.manual_seed(42); np.random.seed(42)
    Xt = torch.tensor(X_s, dtype=torch.float32)
    yt = torch.tensor(full["fwd_return_1y"].fillna(full["fwd_return_1y"].mean()).values,
                      dtype=torch.float32)
    n = X_s.shape[0]; k = int(n * 0.85)
    perm = np.random.permutation(n)
    tri, vai = perm[:k], perm[k:]
    model = MLP(X_s.shape[1], hidden=64, dropout=0.30)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best=float("inf"); best_state=None; pat=0
    for ep in range(200):
        model.train()
        idx = torch.randperm(len(tri))
        for s in range(0, len(tri), 64):
            b = tri[idx[s:s+64].numpy()]
            opt.zero_grad()
            l = loss_fn(model(Xt[b]), yt[b])
            l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xt[vai]), yt[vai]).item()
        if v < best - 1e-5:
            best=v; best_state={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=30: break
    if best_state: model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        emb = model.embeddings(Xt).cpu().numpy()
    print(f"  Embedding shape: {emb.shape}")

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
    proj = reducer.fit_transform(emb)
    full = full.copy()
    full["umap1"] = proj[:, 0]
    full["umap2"] = proj[:, 1]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw={"wspace": 0.3})
    # Color by year
    sc1 = axes[0].scatter(full["umap1"], full["umap2"], c=full["fiscal_year"],
                          cmap="viridis", s=22, alpha=0.75, edgecolors="white", linewidths=0.4)
    axes[0].set_title("UMAP of MLP penultimate layer  |  colored by fiscal year",
                      color=NAVY, fontsize=12)
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
    plt.colorbar(sc1, ax=axes[0], fraction=0.04, pad=0.02)

    # Color by realised return (where available)
    has_y = full["fwd_return_1y"].notna()
    sc2 = axes[1].scatter(full.loc[has_y, "umap1"], full.loc[has_y, "umap2"],
                          c=full.loc[has_y, "fwd_return_1y"],
                          cmap="RdYlGn", s=22, alpha=0.8,
                          vmin=-0.3, vmax=0.5, edgecolors="white", linewidths=0.4)
    axes[1].set_title("UMAP  |  colored by realised forward return",
                      color=NAVY, fontsize=12)
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    plt.colorbar(sc2, ax=axes[1], fraction=0.04, pad=0.02)

    # Color by sub-industry
    sub_codes = pd.Categorical(full["sub_industry"]).codes
    sc3 = axes[2].scatter(full["umap1"], full["umap2"], c=sub_codes,
                          cmap="tab20", s=22, alpha=0.75, edgecolors="white", linewidths=0.4)
    axes[2].set_title("UMAP  |  colored by sub-industry", color=NAVY, fontsize=12)
    axes[2].set_xlabel("UMAP 1"); axes[2].set_ylabel("UMAP 2")
    # Custom legend
    cats = list(pd.Categorical(full["sub_industry"]).categories)
    cmap = plt.get_cmap("tab20")
    handles = [plt.Line2D([0],[0],marker="o", color="w",
                          markerfacecolor=cmap(i/(len(cats)-1)), markersize=8, label=c)
               for i, c in enumerate(cats)]
    axes[2].legend(handles=handles, loc="center left",
                   bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)

    plt.suptitle("Hidden-representation geometry  |  MLP penultimate-layer projection",
                 fontsize=14, color=NAVY, y=1.02, fontweight="bold")
    plt.savefig(FIG / "10_umap_embedding.png")
    plt.close()
    print("  fig10_umap saved")


def fig11_regime_surface():
    """3D surface: realised mean fwd return as function of (vix, term_spread)."""
    panel = pd.read_csv(PROC / "panel_features.csv")
    fig = plt.figure(figsize=(15, 6.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)

    # Left: 3D scatter / interpolated surface
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    # Aggregate per fiscal year (more stable than per-firm)
    agg = panel.groupby("fiscal_year").agg(
        vix=("vix", "mean"),
        term_spread=("term_spread", "mean"),
        fwd_return=("fwd_return_1y", "mean"),
    ).dropna().reset_index()
    sc = ax.scatter(agg["vix"], agg["term_spread"], agg["fwd_return"],
                    c=agg["fwd_return"], cmap="RdYlGn", s=120, vmin=-0.1, vmax=0.4,
                    edgecolors="black", linewidths=0.6)
    for _, row in agg.iterrows():
        ax.text(row["vix"], row["term_spread"], row["fwd_return"] + 0.02,
                str(int(row["fiscal_year"])), fontsize=8, ha="center", color=NAVY)
    ax.set_xlabel("VIX (year-end)")
    ax.set_ylabel("Term spread (10Y-2Y)")
    ax.set_zlabel("Mean fwd return")
    ax.set_title("Annual return regime  |  VIX × term spread", color=NAVY, pad=14)
    ax.view_init(elev=22, azim=-50)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.05)

    # Right: heatmap-style 2D version with annotations
    ax2 = fig.add_subplot(gs[0, 1])
    vix_bins = np.linspace(panel["vix"].quantile(0.05), panel["vix"].quantile(0.95), 6)
    ts_bins  = np.linspace(panel["term_spread"].quantile(0.05),
                           panel["term_spread"].quantile(0.95), 6)
    panel = panel.assign(
        vix_bin=pd.cut(panel["vix"], vix_bins),
        ts_bin=pd.cut(panel["term_spread"], ts_bins),
    )
    grid = panel.groupby(["vix_bin", "ts_bin"], observed=True)["fwd_return_1y"].mean().unstack()
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", [RED, "white", GREEN], N=256)
    im = ax2.imshow(grid.values, cmap=cmap, aspect="auto", vmin=-0.1, vmax=0.4)
    ax2.set_xticks(range(grid.shape[1]))
    ax2.set_xticklabels([f"{i.left:.1f}-{i.right:.1f}" for i in grid.columns], fontsize=8, rotation=30, ha="right")
    ax2.set_yticks(range(grid.shape[0]))
    ax2.set_yticklabels([f"{i.left:.0f}-{i.right:.0f}" for i in grid.index], fontsize=8)
    ax2.set_xlabel("Term spread bin (10Y-2Y, %)")
    ax2.set_ylabel("VIX bin")
    ax2.set_title("Mean realised fwd return by (VIX × term spread) bin",
                   color=NAVY, fontsize=12)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > 0.25 else "#333")
    plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    plt.savefig(FIG / "11_regime_surface.png")
    plt.close()
    print("  fig11_regime_surface saved")


def fig12_signal_decomposition():
    """Yearly long-short attribution: contribution of each feature to LS spread."""
    shap_vals = np.load(PROC / "shap_values.npy")
    feat_names = pd.read_csv(PROC / "shap_feature_names.csv")["feature"].tolist()
    feature_vals = pd.read_csv(PROC / "shap_features.csv")
    importance = pd.read_csv(PROC / "feature_importance_shap.csv")

    # Use the predicted-quintile membership: top 6 features by |SHAP|, show
    # their average SHAP contribution within Q1 vs Q5.
    panel = pd.read_csv(PROC / "panel_features.csv")
    train = panel[panel["fiscal_year"] <= 2022].copy().reset_index(drop=True)
    # Re-load OOF preds for 2022 only (since training was through 2022, we need OOF)
    oof = pd.read_csv(PROC / "oof_predictions_with_ensemble.csv")
    # Use 2022 subset for visualization
    o22 = oof[oof["fiscal_year"] == 2022].copy()
    if len(o22) < 20:
        # fallback: use union
        o22 = oof.dropna(subset=["pred_lgbm"])
    # Match SHAP rows: SHAP was computed on train (panel up to 2022)
    # Quick hack: match by (ticker, fiscal_year)
    train["row_idx"] = range(len(train))
    o22 = o22.merge(train[["ticker", "fiscal_year", "row_idx"]],
                     on=["ticker", "fiscal_year"], how="inner")
    if len(o22) < 20:
        print("  fig12 skipped (no overlap)")
        return
    # Quintile by predicted return
    o22["q"] = pd.qcut(o22["pred_lgbm"], 5, labels=range(1,6), duplicates="drop")
    top10 = importance.head(10)["feature"].tolist()
    rows = []
    for f in top10:
        if f not in feat_names:
            continue
        idx = feat_names.index(f)
        sv = shap_vals[:, idx]
        for q in [1, 2, 3, 4, 5]:
            mask = o22[o22["q"] == q]["row_idx"].values
            if len(mask) == 0:
                continue
            rows.append({"feature": f, "quintile": q, "mean_shap": float(sv[mask].mean())})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="feature", columns="quintile", values="mean_shap")
    pivot = pivot.reindex(top10)

    fig, ax = plt.subplots(figsize=(13, 7))
    # Stacked-style: for each feature, show q1..q5 as grouped bars
    feats = pivot.index.tolist()
    x = np.arange(len(feats))
    width = 0.16
    quint_colors = [RED, "#E07866", GOLD, "#7AA8C4", NAVY]
    for i, q in enumerate([1, 2, 3, 4, 5]):
        ax.bar(x + (i-2)*width, pivot[q].values, width=width,
               color=quint_colors[i], alpha=0.9, label=f"Q{q}", edgecolor="white")
    ax.axhline(0, color=GRAY, linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=30, ha="right", fontsize=9)
    ax.set_title("Signal decomposition  |  mean SHAP contribution per top-10 feature, by predicted quintile (FY 2022)",
                  color=NAVY, fontsize=12)
    ax.set_ylabel("Mean SHAP value")
    ax.legend(frameon=False, ncol=5, loc="lower right")
    plt.savefig(FIG / "12_signal_decomposition.png")
    plt.close()
    print("  fig12_signal_decomposition saved")


def main():
    print("Generating advanced visualizations...")
    try:
        fig10_umap_embedding()
    except Exception as e:
        print(f"  fig10 failed: {e}")
    try:
        fig11_regime_surface()
    except Exception as e:
        print(f"  fig11 failed: {e}")
    try:
        fig12_signal_decomposition()
    except Exception as e:
        print(f"  fig12 failed: {e}")
    print("Done.")


if __name__ == "__main__":
    main()

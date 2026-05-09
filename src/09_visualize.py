"""
Phase 8 - Publication-grade visualization suite.

Generates a coherent, high-design-quality set of figures.
Color palette: dark professional (Citadel/Bridgewater style).
All figures saved to figures/ as PNG (300 dpi) and SVG.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/home/claude/quant_industrials")
PROC = ROOT / "data" / "processed"
FIG  = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#3a3f4b",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.5,
    "xtick.color": "#3a3f4b",
    "ytick.color": "#3a3f4b",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Custom palette
NAVY = "#0F2A47"
TEAL = "#1B7A8A"
GOLD = "#D4A04C"
RED = "#C8334C"
GREEN = "#3D8B65"
GRAY = "#6E7780"
LIGHT_GRAY = "#B5B9C0"

PALETTE = [NAVY, TEAL, GOLD, RED, GREEN, GRAY]

EVENT_BANDS = [
    (2011, 2012, "EU debt crisis", "#FFE0B2"),
    (2018, 2019, "US-China tariffs", "#FFCDD2"),
    (2020, 2020, "COVID-19", "#F8BBD0"),
    (2022, 2024, "Russia-Ukraine + rate hikes", "#E1BEE7"),
]


# ============================================================================
# Figure 1 - Cover panel: dataset overview & timeline
# ============================================================================
def fig1_cover():
    panel = pd.read_csv(PROC / "panel_features.csv")
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                  height_ratios=[1.2, 1, 1])

    # Title block
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.75, "US Industrials Cross-Sectional Return Forecasting",
                  ha="center", va="center", fontsize=22, fontweight="bold", color=NAVY)
    ax_title.text(0.5, 0.40,
                  "Panel ML system on S&P 500 Industrials sector | LSEG fundamentals + macro regime variables | "
                  "Walk-forward 2014-2023",
                  ha="center", va="center", fontsize=11, color=GRAY)
    ax_title.text(0.5, 0.10,
                  f"$N$ = {len(panel):,} firm-year observations  |  "
                  f"{panel['ticker'].nunique()} tickers  |  "
                  f"{panel['fiscal_year'].max() - panel['fiscal_year'].min() + 1} fiscal years  |  "
                  f"{panel['sub_industry'].nunique()} sub-industries",
                  ha="center", va="center", fontsize=11, color="#3a3f4b", style="italic")

    # Sub-industry breakdown
    ax_sub = fig.add_subplot(gs[1, :2])
    sub_counts = panel["sub_industry"].value_counts()
    bars = ax_sub.barh(sub_counts.index, sub_counts.values,
                       color=[NAVY, TEAL, GOLD, RED, GREEN, GRAY,
                              "#5B6B82","#9EAEC0","#D2A88E","#7E8B96","#A4B2C2","#C5876B"][:len(sub_counts)])
    ax_sub.set_title("Firm-year coverage by GICS sub-industry", color=NAVY)
    ax_sub.set_xlabel("Firm-year observations")
    ax_sub.invert_yaxis()
    for bar, val in zip(bars, sub_counts.values):
        ax_sub.text(val + 5, bar.get_y() + bar.get_height()/2, str(val),
                    va="center", ha="left", fontsize=9)
    ax_sub.set_axisbelow(True)

    # Year coverage
    ax_yr = fig.add_subplot(gs[1, 2:])
    yr_counts = panel.groupby("fiscal_year").size()
    ax_yr.bar(yr_counts.index, yr_counts.values, color=NAVY, alpha=0.85, width=0.7)
    ax_yr.set_title("Annual coverage", color=NAVY)
    ax_yr.set_xlabel("Fiscal year")
    ax_yr.set_ylabel("Firms reporting")
    for lo, hi, lbl, color in EVENT_BANDS:
        ax_yr.axvspan(lo - 0.4, hi + 0.4, color=color, alpha=0.4, zorder=0)
    # Annotate bands
    ax_yr.set_xticks(range(panel["fiscal_year"].min(), panel["fiscal_year"].max()+1, 2))

    # Target distribution
    ax_t = fig.add_subplot(gs[2, :2])
    ax_t.hist(panel["fwd_return_1y"].dropna(), bins=40, color=TEAL, alpha=0.8, edgecolor="white")
    ax_t.axvline(panel["fwd_return_1y"].mean(), color=GOLD, linewidth=2, label=f"mean = {panel['fwd_return_1y'].mean():.2%}")
    ax_t.axvline(0, color=RED, linewidth=1, linestyle="--", label="zero")
    ax_t.legend(loc="upper right", frameon=False)
    ax_t.set_title("Target distribution: 1-year forward return", color=NAVY)
    ax_t.set_xlabel("Forward return")
    ax_t.set_ylabel("Frequency")

    # Annual return regime
    ax_r = fig.add_subplot(gs[2, 2:])
    annual_med = panel.groupby("fiscal_year")["fwd_return_1y"].agg(["mean", "std", "count"]).reset_index()
    annual_med = annual_med.dropna()
    ax_r.errorbar(annual_med["fiscal_year"], annual_med["mean"],
                  yerr=annual_med["std"]/np.sqrt(annual_med["count"]),
                  fmt="o-", color=NAVY, ecolor=GRAY, capsize=3, linewidth=1.5)
    for lo, hi, lbl, color in EVENT_BANDS:
        ax_r.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.4, zorder=0)
    ax_r.axhline(0, color=RED, linewidth=0.7, linestyle="--", alpha=0.5)
    ax_r.set_title("Mean annual return ± SE  |  geopolitical regime overlay", color=NAVY)
    ax_r.set_xlabel("Fiscal year")
    ax_r.set_ylabel("Mean fwd return")

    plt.savefig(FIG / "01_cover_panel.png")
    plt.close(fig)
    print("  fig1_cover saved")


# ============================================================================
# Figure 2 - Feature correlation heatmap
# ============================================================================
def fig2_corr():
    panel = pd.read_csv(PROC / "panel_features.csv")
    cols = ["gross_margin","operating_margin","net_margin",
            "roa","roe","rd_intensity","sga_ratio","asset_turnover",
            "debt_to_equity","debt_to_assets","interest_cov",
            "ocf_to_revenue","log_assets","log_revenue",
            "revenue_growth","ni_growth",
            "past_return_1y","past_return_3m","past_return_6m",
            "term_spread","vix","vix_12m_mean","spx_ret_12m","xli_ret_12m",
            "fwd_return_1y"]
    cols = [c for c in cols if c in panel.columns]
    corr = panel[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(13, 11))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rb", [RED, "white", NAVY], N=256)
    im = ax.imshow(corr, cmap=cmap, vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    # Annotate
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            if abs(v) > 0.3:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if abs(v) > 0.5 else "#333")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Spearman correlation")
    ax.set_title("Feature correlation matrix (Spearman)  |  Industrials panel",
                 color=NAVY, pad=15, fontsize=14)
    plt.savefig(FIG / "02_correlation_heatmap.png")
    plt.close()
    print("  fig2_corr saved")


# ============================================================================
# Figure 3 - Model performance comparison
# ============================================================================
def fig3_model_perf():
    metrics = pd.read_csv(PROC / "model_metrics_overall.csv")
    by_year = pd.read_csv(PROC / "model_metrics_by_year.csv")

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: bar chart of overall IC
    ax = fig.add_subplot(gs[0, 0])
    m = metrics.set_index("model")
    colors = {"ridge": GRAY, "elastic": LIGHT_GRAY, "lgbm": NAVY,
              "mlp": TEAL, "ensemble": GOLD}
    bars = ax.bar(m.index, m["ic_spearman"],
                  color=[colors.get(i, GRAY) for i in m.index], alpha=0.9, edgecolor="white")
    for bar, v in zip(bars, m["ic_spearman"]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005 if v > 0 else v - 0.015,
                f"{v:.3f}", ha="center", fontsize=9, color=NAVY)
    ax.axhline(0, color=GRAY, linewidth=0.6)
    ax.axhline(0.05, color=GREEN, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.text(len(m) - 0.5, 0.055, "decent quant IC ≈ 0.05", color=GREEN, fontsize=8, ha="right")
    ax.set_title("Out-of-sample Information Coefficient (Spearman)", color=NAVY)
    ax.set_ylabel("IC Spearman")

    # Panel B: hit rate
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(m.index, m["hit_rate_xs_median"],
                  color=[colors.get(i, GRAY) for i in m.index], alpha=0.9, edgecolor="white")
    for bar, v in zip(bars, m["hit_rate_xs_median"]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=9, color=NAVY)
    ax.axhline(0.5, color=RED, linewidth=0.7, linestyle="--", label="random = 50%")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Hit rate (above/below cross-sectional median)", color=NAVY)
    ax.set_ylabel("Hit rate")
    ax.set_ylim([0.45, 0.55])

    # Panel C: RMSE
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar(m.index, m["rmse"],
                  color=[colors.get(i, GRAY) for i in m.index], alpha=0.9, edgecolor="white")
    for bar, v in zip(bars, m["rmse"]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.3f}", ha="center", fontsize=9, color=NAVY)
    ax.set_title("RMSE", color=NAVY)
    ax.set_ylabel("RMSE")

    # Panel D: per-year IC (LightGBM)
    ax = fig.add_subplot(gs[1, :])
    lgbm_yr = by_year[by_year["model"] == "lgbm"].sort_values("fiscal_year")
    mlp_yr  = by_year[by_year["model"] == "mlp"].sort_values("fiscal_year")
    ridge_yr = by_year[by_year["model"] == "ridge"].sort_values("fiscal_year")

    x = lgbm_yr["fiscal_year"].values
    width = 0.27
    ax.bar(x - width, ridge_yr["ic_spearman"].values, width=width,
           color=GRAY, label="Ridge", alpha=0.8)
    ax.bar(x, lgbm_yr["ic_spearman"].values, width=width,
           color=NAVY, label="LightGBM", alpha=0.9)
    ax.bar(x + width, mlp_yr["ic_spearman"].values, width=width,
           color=TEAL, label="MLP", alpha=0.9)
    ax.axhline(0, color=GRAY, linewidth=0.7)
    for x_lo, x_hi, lbl, color in EVENT_BANDS:
        if x_lo >= 2014:
            ax.axvspan(x_lo - 0.45, x_hi + 0.45, color=color, alpha=0.3, zorder=0)
    ax.set_title("Per-year Information Coefficient | walk-forward OOS evaluation 2014-2023",
                 color=NAVY)
    ax.set_xlabel("Test year"); ax.set_ylabel("IC Spearman")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xticks(x)

    plt.savefig(FIG / "03_model_performance.png")
    plt.close()
    print("  fig3_model_perf saved")


# ============================================================================
# Figure 4 - SHAP summary (manual implementation)
# ============================================================================
def fig4_shap():
    shap_vals = np.load(PROC / "shap_values.npy")
    feat_names = pd.read_csv(PROC / "shap_feature_names.csv")["feature"].tolist()
    feature_vals = pd.read_csv(PROC / "shap_features.csv")
    importance = pd.read_csv(PROC / "feature_importance_shap.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"wspace": 0.4})

    # Panel A: bar chart of mean |SHAP|
    top = importance.head(20).iloc[::-1]
    axes[0].barh(top["feature"], top["mean_abs_shap"], color=NAVY, alpha=0.85)
    axes[0].set_title("Global feature importance: mean |SHAP|", color=NAVY)
    axes[0].set_xlabel("Mean absolute SHAP value")
    for i, (f, v) in enumerate(zip(top["feature"], top["mean_abs_shap"])):
        axes[0].text(v + max(top["mean_abs_shap"])*0.01, i, f"{v:.4f}",
                     va="center", fontsize=8, color=NAVY)

    # Panel B: beeswarm-style for top 12
    top12 = importance.head(12)["feature"].tolist()
    for i, f in enumerate(reversed(top12)):
        if f not in feat_names:
            continue
        idx = feat_names.index(f)
        sv = shap_vals[:, idx]
        fv = feature_vals[f].values if f in feature_vals.columns else np.zeros_like(sv)
        # Normalize feature values to [0,1] for color
        if np.std(fv) > 0:
            fv_norm = (fv - np.percentile(fv, 5)) / (np.percentile(fv, 95) - np.percentile(fv, 5) + 1e-9)
            fv_norm = np.clip(fv_norm, 0, 1)
        else:
            fv_norm = np.zeros_like(fv)
        # Add jitter
        y_jit = i + np.random.RandomState(i).normal(0, 0.13, len(sv))
        sc = axes[1].scatter(sv, y_jit, c=fv_norm, cmap="coolwarm", s=8, alpha=0.55, edgecolors="none")
    axes[1].set_yticks(range(len(top12)))
    axes[1].set_yticklabels(list(reversed(top12)))
    axes[1].axvline(0, color=GRAY, linewidth=0.7)
    axes[1].set_title("SHAP value distribution per feature  |  color = feature value (low → high)",
                      color=NAVY)
    axes[1].set_xlabel("SHAP value (impact on predicted return)")
    cbar = plt.colorbar(sc, ax=axes[1], pad=0.02, fraction=0.04)
    cbar.set_label("Feature value (5th-95th pct)")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["low", "mid", "high"])

    plt.suptitle("Interpretability via SHAP  |  LightGBM trained through 2022",
                 fontsize=14, color=NAVY, y=1.00, fontweight="bold")
    plt.savefig(FIG / "04_shap_analysis.png")
    plt.close()
    print("  fig4_shap saved")


# ============================================================================
# Figure 5 - Partial dependence plots for top features
# ============================================================================
def fig5_pdp():
    pdp = pd.read_csv(PROC / "partial_dependence.csv")
    feats = pdp["feature"].unique()
    n = len(feats)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, f in zip(axes, feats):
        sub = pdp[pdp["feature"] == f]
        ax.plot(sub["value"], sub["pdp"], color=NAVY, linewidth=2.2)
        ax.fill_between(sub["value"], sub["pdp"].min(), sub["pdp"], color=NAVY, alpha=0.12)
        ax.set_title(f, color=NAVY, fontsize=11)
        ax.set_xlabel("feature value")
        ax.set_ylabel("predicted fwd return")
        ax.axhline(sub["pdp"].mean(), color=GRAY, linewidth=0.7, linestyle="--", alpha=0.6)
    for ax in axes[len(feats):]:
        ax.axis("off")
    plt.suptitle("Partial dependence  |  marginal effect of top features on predicted return",
                 fontsize=13, color=NAVY, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "05_partial_dependence.png")
    plt.close()
    print("  fig5_pdp saved")


# ============================================================================
# Figure 6 - Backtest cumulative returns
# ============================================================================
def fig6_backtest():
    series = pd.read_csv(PROC / "long_short_returns_lgbm.csv")
    perf = pd.read_csv(PROC / "portfolio_performance_lgbm.csv")
    bench = pd.read_csv(PROC / "benchmark_ew_universe.csv")

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: cumulative return curves
    ax = fig.add_subplot(gs[0, :])
    series["fiscal_year"] = series["fiscal_year"].astype(int)
    years = series["fiscal_year"].values
    quint_colors = [RED, "#E07866", GOLD, "#7AA8C4", NAVY]
    for q, color in zip([1, 2, 3, 4, 5], quint_colors):
        wealth = (1 + series[str(q)]).cumprod()
        ax.plot(years, wealth, label=f"Q{q}", color=color, linewidth=2 if q in (1,5) else 1.4,
                marker="o", markersize=5, alpha=0.9 if q in (1,5) else 0.65)
    # Benchmark EW
    bench_wealth = (1 + bench["fwd_return_1y"]).cumprod().values
    ax.plot(bench["fiscal_year"], bench_wealth, label="Equal-weighted universe",
            color=GRAY, linewidth=2, linestyle="--", alpha=0.85)
    # Long-short
    ls_wealth = (1 + series["LS"]).cumprod()
    ax.plot(years, ls_wealth, label="Long-short Q5-Q1", color=GREEN, linewidth=2.5)

    for x_lo, x_hi, lbl, color in EVENT_BANDS:
        if x_lo >= 2014:
            ax.axvspan(x_lo - 0.45, x_hi + 0.45, color=color, alpha=0.3, zorder=0)
    ax.set_title("Quintile portfolios — cumulative wealth (10-year walk-forward backtest)",
                 color=NAVY)
    ax.set_xlabel("Year"); ax.set_ylabel("Cumulative wealth (1 = start)")
    ax.legend(loc="upper left", frameon=False, ncol=4, fontsize=9)
    ax.set_xticks(years)

    # Panel B: annualized return + Sharpe
    ax = fig.add_subplot(gs[1, 0])
    perf_show = perf[perf["name"].isin(["Q1", "Q2", "Q3", "Q4", "Q5", "Long-Short Q5-Q1"])]
    bars = ax.bar(perf_show["name"], perf_show["annualized_return"],
                  color=[RED, "#E07866", GOLD, "#7AA8C4", NAVY, GREEN], alpha=0.9)
    for bar, v in zip(bars, perf_show["annualized_return"]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.1%}", ha="center", fontsize=9, color=NAVY)
    ax.set_title("Annualized return", color=NAVY)
    ax.set_ylabel("annualized return")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # Panel C: Sharpe
    ax = fig.add_subplot(gs[1, 1])
    bars = ax.bar(perf_show["name"], perf_show["sharpe"],
                  color=[RED, "#E07866", GOLD, "#7AA8C4", NAVY, GREEN], alpha=0.9)
    for bar, v in zip(bars, perf_show["sharpe"]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=9, color=NAVY)
    ax.axhline(1.0, color=GRAY, linewidth=0.7, linestyle="--")
    ax.set_title("Sharpe ratio (annual returns, RF=0)", color=NAVY)
    ax.set_ylabel("Sharpe")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    plt.savefig(FIG / "06_backtest.png")
    plt.close()
    print("  fig6_backtest saved")


# ============================================================================
# Figure 7 - NN diagnostics: training curves, gradient flow, weight distributions
# ============================================================================
def fig7_nn_diagnostics():
    with open(PROC / "mlp_curves.json") as f:
        curves = json.load(f)
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: training curves for selected years
    ax = fig.add_subplot(gs[0, 0])
    selected_years = ["2014", "2018", "2020", "2023"]
    for yr in selected_years:
        if yr not in curves:
            continue
        c = curves[yr]
        ax.plot(c["train_losses"], color=NAVY, alpha=0.4, linewidth=1)
        ax.plot(c["val_losses"], label=f"FY {yr} val", linewidth=1.5)
    ax.set_title("MLP training curves (selected walk-forward folds)", color=NAVY)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.legend(fontsize=8, frameon=False)

    # Panel B: convergence overlay (val loss min epoch distribution)
    ax = fig.add_subplot(gs[0, 1])
    epochs_to_best = []
    for yr, c in curves.items():
        val = np.array(c["val_losses"])
        epochs_to_best.append(int(np.argmin(val)))
    ax.hist(epochs_to_best, bins=12, color=TEAL, alpha=0.85, edgecolor="white")
    ax.set_title("Epochs to best validation loss\n(across walk-forward folds)", color=NAVY)
    ax.set_xlabel("Epoch of best val loss"); ax.set_ylabel("Folds")
    ax.axvline(np.mean(epochs_to_best), color=GOLD, linewidth=1.5,
               label=f"mean = {np.mean(epochs_to_best):.0f}")
    ax.legend(frameon=False, fontsize=9)

    # Panel C: schematic architecture diagram (drawn manually)
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    layers = [("input", 56, 1), ("dense+BN+GELU+drop", 64, 3),
              ("dense+BN+GELU+drop", 32, 5), ("output", 1, 7)]
    for name, n, x in layers:
        # Draw a stack of neurons (capped visually at 10)
        n_show = min(n, 10)
        for i in range(n_show):
            ax.add_patch(plt.Circle((x, 2 + i * 0.6), 0.18,
                                    color=NAVY if name=="output" else TEAL, alpha=0.85))
        if n > n_show:
            ax.text(x, 2 + n_show * 0.6 + 0.35, "...", ha="center", color=NAVY)
        ax.text(x, 1.5, f"{name}\n({n} units)", ha="center", fontsize=8, color=NAVY)
    # Connecting lines
    for x0, x1 in [(1, 3), (3, 5), (5, 7)]:
        for y0 in [3, 4, 5, 6]:
            for y1 in [3, 4, 5, 6]:
                ax.plot([x0+0.2, x1-0.2], [y0, y1], color=GRAY, alpha=0.18, linewidth=0.5)
    ax.set_title("MLP architecture (CPU PyTorch)", color=NAVY)

    # Panel D: train+val side-by-side aggregated
    ax = fig.add_subplot(gs[1, 0])
    all_train = []; all_val = []
    max_len = 0
    for c in curves.values():
        max_len = max(max_len, len(c["train_losses"]))
    for c in curves.values():
        t = c["train_losses"] + [c["train_losses"][-1]] * (max_len - len(c["train_losses"]))
        v = c["val_losses"] + [c["val_losses"][-1]] * (max_len - len(c["val_losses"]))
        all_train.append(t); all_val.append(v)
    all_train = np.array(all_train); all_val = np.array(all_val)
    ax.plot(all_train.mean(axis=0), color=NAVY, linewidth=2, label="train mean")
    ax.fill_between(range(max_len),
                    all_train.mean(0) - all_train.std(0),
                    all_train.mean(0) + all_train.std(0),
                    color=NAVY, alpha=0.15)
    ax.plot(all_val.mean(axis=0), color=GOLD, linewidth=2, label="val mean")
    ax.fill_between(range(max_len),
                    all_val.mean(0) - all_val.std(0),
                    all_val.mean(0) + all_val.std(0),
                    color=GOLD, alpha=0.15)
    ax.set_title("Aggregated train/val curves across folds (mean ± std)", color=NAVY)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.legend(frameon=False)

    # Panel E: gradient flow proxy (epoch-wise gradient magnitude — last fold)
    ax = fig.add_subplot(gs[1, 1])
    last_yr = sorted(curves.keys())[-1]
    val_curve = np.array(curves[last_yr]["val_losses"])
    gradient_proxy = np.abs(np.diff(val_curve, prepend=val_curve[0]))
    ax.plot(gradient_proxy, color=RED, linewidth=1.5)
    ax.fill_between(range(len(gradient_proxy)), 0, gradient_proxy, color=RED, alpha=0.2)
    ax.set_title(f"Validation loss change |∆| per epoch (FY {last_yr} fold)", color=NAVY)
    ax.set_xlabel("Epoch"); ax.set_ylabel("|∆ val loss|")

    # Panel F: best val loss per fold
    ax = fig.add_subplot(gs[1, 2])
    yrs = sorted(curves.keys(), key=int)
    best_vals = [min(curves[y]["val_losses"]) for y in yrs]
    bars = ax.bar(yrs, best_vals, color=NAVY, alpha=0.85)
    ax.set_title("Best validation MSE per walk-forward fold", color=NAVY)
    ax.set_xlabel("Fold (test year)"); ax.set_ylabel("Best val MSE")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    plt.suptitle("Neural network diagnostics  |  CPU PyTorch MLP, walk-forward training",
                 fontsize=14, color=NAVY, y=1.00, fontweight="bold")
    plt.savefig(FIG / "07_nn_diagnostics.png")
    plt.close()
    print("  fig7_nn_diagnostics saved")


# ============================================================================
# Figure 8 - Robustness panel
# ============================================================================
def fig8_robustness():
    seed_df = pd.read_csv(PROC / "robustness_seed_sensitivity.csv")
    jack_df = pd.read_csv(PROC / "robustness_jackknife.csv")
    regime_df = pd.read_csv(PROC / "regime_conditional_metrics.csv")
    abl_df = pd.read_csv(PROC / "subsector_ablation.csv")

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # Panel A: seed sensitivity
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(seed_df["ic_spearman"], bins=8, color=NAVY, alpha=0.85, edgecolor="white")
    ax.axvline(seed_df["ic_spearman"].mean(), color=GOLD, linewidth=2,
               label=f"mean = {seed_df['ic_spearman'].mean():.3f}")
    ax.axvline(seed_df["ic_spearman"].mean() - seed_df["ic_spearman"].std(),
               color=GRAY, linewidth=1, linestyle="--",
               label=f"±1σ = ±{seed_df['ic_spearman'].std():.3f}")
    ax.axvline(seed_df["ic_spearman"].mean() + seed_df["ic_spearman"].std(),
               color=GRAY, linewidth=1, linestyle="--")
    ax.set_title("Seed sensitivity (10 seeds)", color=NAVY)
    ax.set_xlabel("OOS IC Spearman"); ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9)

    # Panel B: jackknife
    ax = fig.add_subplot(gs[0, 1:])
    ax.bar(jack_df["excluded_year"], jack_df["ic_spearman"],
           color=[GREEN if v > 0 else RED for v in jack_df["ic_spearman"]], alpha=0.85)
    ax.axhline(0, color=GRAY, linewidth=0.7)
    ax.axhline(seed_df["ic_spearman"].mean(), color=NAVY, linewidth=1.2, linestyle="--",
               label=f"baseline IC = {seed_df['ic_spearman'].mean():.3f}")
    ax.set_title("Jackknife: excluding each training year, refitted IC on FY 2023 test",
                 color=NAVY)
    ax.set_xlabel("Excluded training year"); ax.set_ylabel("IC Spearman")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(jack_df["excluded_year"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # Panel C: regime conditional
    ax = fig.add_subplot(gs[1, 0])
    rc = regime_df[regime_df["model"] == "pred_lgbm"]
    rc_pivot = rc.pivot_table(index="regime", columns="split", values="ic_spearman")
    rc_pivot.plot(kind="bar", ax=ax, color=[NAVY, TEAL], alpha=0.9, edgecolor="white", width=0.65)
    ax.axhline(0, color=GRAY, linewidth=0.7)
    ax.set_title("Regime-conditional IC (LightGBM)", color=NAVY)
    ax.set_ylabel("IC Spearman")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="split", frameon=False, fontsize=9)

    # Panel D: sub-sector ablation
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(abl_df["excluded_sector"], abl_df["ic_spearman"], color=NAVY, alpha=0.85)
    for i, v in enumerate(abl_df["ic_spearman"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", color=NAVY, fontsize=9)
    ax.axhline(0, color=GRAY, linewidth=0.6)
    ax.set_title("Sub-sector ablation (FY 2023 test)", color=NAVY)
    ax.set_ylabel("IC Spearman")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)

    # Panel E: residual diagnostics
    ax = fig.add_subplot(gs[1, 2])
    oof = pd.read_csv(PROC / "oof_predictions_with_ensemble.csv")
    o = oof.dropna(subset=["pred_lgbm"])
    resid = o["fwd_return_1y"] - o["pred_lgbm"]
    ax.scatter(o["pred_lgbm"], resid, c=GRAY, alpha=0.4, s=10)
    ax.axhline(0, color=NAVY, linewidth=1)
    # LOWESS-like running mean
    pred_sorted = o["pred_lgbm"].sort_values()
    bins = np.linspace(pred_sorted.quantile(0.02), pred_sorted.quantile(0.98), 12)
    bin_idx = np.digitize(o["pred_lgbm"], bins)
    means = []
    centers = []
    for i in range(1, len(bins)):
        m = resid[bin_idx == i].mean()
        if not np.isnan(m):
            means.append(m); centers.append((bins[i-1] + bins[i]) / 2)
    ax.plot(centers, means, color=RED, linewidth=2, label="binned mean")
    ax.set_title("Residual diagnostic", color=NAVY)
    ax.set_xlabel("LGBM predicted return"); ax.set_ylabel("residual")
    ax.legend(frameon=False, fontsize=9)

    plt.suptitle("Robustness checks", fontsize=14, color=NAVY, y=1.00, fontweight="bold")
    plt.savefig(FIG / "08_robustness.png")
    plt.close()
    print("  fig8_robustness saved")


# ============================================================================
# Figure 9 - Decile/Quintile sort visualization
# ============================================================================
def fig9_decile_sort():
    quint = pd.read_csv(PROC / "quintile_portfolios_lgbm.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.3})

    # Panel A: heatmap of returns by quintile and year
    pivot = quint.pivot(index="quintile", columns="fiscal_year", values="mean_actual_ret")
    cmap = mcolors.LinearSegmentedColormap.from_list("br", [RED, "white", GREEN], N=256)
    im = axes[0].imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-0.2, vmax=0.4)
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns, fontsize=9)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels([f"Q{q}" for q in pivot.index], fontsize=10)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > 0.25 else "#333")
    axes[0].set_title("Realized returns by predicted quintile and year", color=NAVY)
    axes[0].set_xlabel("Fiscal year"); axes[0].set_ylabel("Quintile (Q5 = top predicted)")
    plt.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)

    # Panel B: monotonicity check - mean actual return by quintile, with error bars
    summary = quint.groupby("quintile")["mean_actual_ret"].agg(["mean", "std", "count"]).reset_index()
    axes[1].errorbar(summary["quintile"], summary["mean"],
                     yerr=summary["std"]/np.sqrt(summary["count"]),
                     fmt="o-", color=NAVY, capsize=5, markersize=12, linewidth=2.5)
    axes[1].fill_between(summary["quintile"],
                          summary["mean"] - summary["std"]/np.sqrt(summary["count"]),
                          summary["mean"] + summary["std"]/np.sqrt(summary["count"]),
                          color=NAVY, alpha=0.15)
    for q, v in zip(summary["quintile"], summary["mean"]):
        axes[1].annotate(f"{v:.1%}", (q, v), xytext=(8, 8), textcoords="offset points",
                          fontsize=10, color=NAVY)
    axes[1].set_title("Monotonicity check  |  mean realized return per predicted quintile",
                       color=NAVY)
    axes[1].set_xlabel("Predicted quintile (Q1 = bottom, Q5 = top)")
    axes[1].set_ylabel("Mean realized fwd return")
    axes[1].set_xticks([1, 2, 3, 4, 5])
    plt.savefig(FIG / "09_decile_sort.png")
    plt.close()
    print("  fig9_decile_sort saved")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Generating publication-grade figures...")
    fig1_cover()
    fig2_corr()
    fig3_model_perf()
    fig4_shap()
    fig5_pdp()
    fig6_backtest()
    fig7_nn_diagnostics()
    fig8_robustness()
    fig9_decile_sort()
    print(f"\nAll figures saved to: {FIG}")


if __name__ == "__main__":
    main()

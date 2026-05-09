"""
Phase 7 - Robustness checks.

  A) Seed sensitivity: refit MLP with 10 seeds, observe metric variance.
  B) Jackknife by year: drop each training year, observe IC stability.
  C) Regime-conditional analysis: split OOF preds by regime
     (high/low VIX, recession/expansion, event vs no-event years).
  D) Sub-sector ablation: test performance excluding Aerospace & Defense
     (idiosyncratic defense-spending dynamics).

Outputs:
  - robustness_seed_sensitivity.csv
  - robustness_jackknife.csv
  - regime_conditional_metrics.csv
  - subsector_ablation.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import torch
import torch.nn as nn
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/home/claude/quant_industrials")
PROC = ROOT / "data" / "processed"

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
    "event_us_china_tariffs","event_covid_shock","event_covid_aftermath",
    "event_russia_ukraine","event_israel_hamas","event_supply_chain_shock",
    "event_rate_hike_cycle","event_eu_debt_crisis",
]
TARGET = "fwd_return_1y"


def prepare(panel, train_filter, test_filter, features=NUMERIC_FEATURES):
    train = panel[train_filter]
    test = panel[test_filter]
    X_tr = pd.get_dummies(train[features + ["sub_industry"]], columns=["sub_industry"], prefix="sub")
    X_te = pd.get_dummies(test[features + ["sub_industry"]], columns=["sub_industry"], prefix="sub")
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    imp = SimpleImputer(strategy="median")
    X_tr_v = imp.fit_transform(X_tr)
    X_te_v = imp.transform(X_te)
    return X_tr_v, train[TARGET].values, X_te_v, test[TARGET].values


def train_lgbm(X_tr, y_tr, seed=42):
    n = X_tr.shape[0]; k = int(n * 0.85)
    m = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=4,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5,
        random_state=seed, verbose=-1)
    m.fit(X_tr[:k], y_tr[:k],
          eval_set=[(X_tr[k:], y_tr[k:])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def main():
    panel = pd.read_csv(PROC / "panel_features.csv")
    avail = [c for c in NUMERIC_FEATURES if c in panel.columns]

    # ============= A) Seed sensitivity for LGBM =============
    print("A) Seed sensitivity (LightGBM, 10 seeds)...")
    rows = []
    train_mask = panel["fiscal_year"] <= 2022
    test_mask  = panel["fiscal_year"] == 2023
    X_tr, y_tr, X_te, y_te = prepare(panel, train_mask, test_mask, avail)
    for seed in range(10):
        m = train_lgbm(X_tr, y_tr, seed=42 + seed)
        p = m.predict(X_te)
        ic = spearmanr(y_te, p)[0]
        rmse = np.sqrt(np.mean((y_te - p)**2))
        rows.append({"seed": 42 + seed, "ic_spearman": ic, "rmse": rmse})
    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(PROC / "robustness_seed_sensitivity.csv", index=False)
    print(f"  IC mean: {seed_df['ic_spearman'].mean():.3f}, std: {seed_df['ic_spearman'].std():.3f}")
    print(f"  RMSE mean: {seed_df['rmse'].mean():.3f}, std: {seed_df['rmse'].std():.3f}")

    # ============= B) Jackknife by year =============
    print("\nB) Jackknife by training year...")
    rows = []
    for excl_year in range(2009, 2023):
        train_mask = (panel["fiscal_year"] <= 2022) & (panel["fiscal_year"] != excl_year)
        X_tr, y_tr, X_te, y_te = prepare(panel, train_mask, test_mask, avail)
        m = train_lgbm(X_tr, y_tr, seed=42)
        p = m.predict(X_te)
        ic = spearmanr(y_te, p)[0]
        rmse = np.sqrt(np.mean((y_te - p)**2))
        rows.append({"excluded_year": excl_year, "ic_spearman": ic, "rmse": rmse})
    jack_df = pd.DataFrame(rows)
    jack_df.to_csv(PROC / "robustness_jackknife.csv", index=False)
    print(f"  IC across jackknives: mean={jack_df['ic_spearman'].mean():.3f}, "
          f"std={jack_df['ic_spearman'].std():.3f}, "
          f"range=[{jack_df['ic_spearman'].min():.3f}, {jack_df['ic_spearman'].max():.3f}]")

    # ============= C) Regime-conditional analysis =============
    print("\nC) Regime-conditional metrics on OOF predictions...")
    oof = pd.read_csv(PROC / "oof_predictions_with_ensemble.csv")
    macro = pd.read_csv(PROC / "macro_clean.csv", parse_dates=["date"])
    # VIX year-end average
    macro["year"] = macro["date"].dt.year
    annual_vix = macro.groupby("year")["vix"].mean().reset_index()
    annual_vix.columns = ["fiscal_year", "annual_vix"]
    oof = oof.merge(annual_vix, on="fiscal_year", how="left")
    median_vix = oof["annual_vix"].median()
    oof["regime_vix"] = np.where(oof["annual_vix"] >= median_vix, "high_vol", "low_vol")
    # Event vs non-event years
    event_years = {2011, 2012, 2018, 2019, 2020, 2022, 2023}
    oof["regime_event"] = np.where(oof["fiscal_year"].isin(event_years), "event", "no_event")

    rows = []
    for split_name in ["regime_vix", "regime_event"]:
        for regime, grp in oof.groupby(split_name):
            for model in ["pred_lgbm", "pred_mlp", "pred_ridge"]:
                if grp[model].notna().sum() < 5:
                    continue
                ic = spearmanr(grp[TARGET], grp[model])[0]
                rows.append({"split": split_name, "regime": regime, "model": model,
                             "n": len(grp), "ic_spearman": ic,
                             "actual_mean_ret": grp[TARGET].mean()})
    regime_df = pd.DataFrame(rows)
    regime_df.to_csv(PROC / "regime_conditional_metrics.csv", index=False)
    print(regime_df.round(3).to_string(index=False))

    # ============= D) Sub-sector ablation =============
    print("\nD) Sub-sector ablation (excluding Aerospace & Defense)...")
    rows = []
    for excl_sector in ["Aerospace & Defense", "Passenger Airlines", "Other"]:
        sub_panel = panel[panel["sub_industry"] != excl_sector]
        train_mask = sub_panel["fiscal_year"] <= 2022
        test_mask = sub_panel["fiscal_year"] == 2023
        if test_mask.sum() < 10:
            continue
        X_tr, y_tr, X_te, y_te = prepare(sub_panel, train_mask, test_mask, avail)
        m = train_lgbm(X_tr, y_tr, seed=42)
        p = m.predict(X_te)
        ic = spearmanr(y_te, p)[0]
        rmse = np.sqrt(np.mean((y_te - p)**2))
        rows.append({"excluded_sector": excl_sector, "n_test": int(test_mask.sum()),
                     "ic_spearman": ic, "rmse": rmse})
    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(PROC / "subsector_ablation.csv", index=False)
    print(abl_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

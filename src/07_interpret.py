"""
Phase 6 - Interpretability for the LightGBM model.

Trains a single LGBM on the full panel through 2022 (year 2023 used for OOS test),
then computes:
  - SHAP values (TreeExplainer)
  - Permutation importance
  - Partial Dependence (manually computed for top features)

Outputs:
  - shap_values.npy
  - shap_features.csv (feature names aligned to columns)
  - permutation_importance.csv
  - partial_dependence.csv
  - feature_importance_ranking.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.impute import SimpleImputer
import shap
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


def main():
    panel = pd.read_csv(PROC / "panel_features.csv")
    avail = [c for c in NUMERIC_FEATURES if c in panel.columns]
    print(f"Features used: {len(avail)}")

    train = panel[panel["fiscal_year"] <= 2022].copy()
    test  = panel[panel["fiscal_year"] == 2023].copy()

    X_tr = pd.get_dummies(train[avail + ["sub_industry"]], columns=["sub_industry"], prefix="sub")
    X_te = pd.get_dummies(test[avail + ["sub_industry"]], columns=["sub_industry"], prefix="sub")
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    imp = SimpleImputer(strategy="median")
    X_tr_v = imp.fit_transform(X_tr)
    X_te_v = imp.transform(X_te)
    # Imputer drops fully-NaN columns. Use the survived column indexes via statistics_.
    surv_mask = ~np.isnan(imp.statistics_)
    feat_names = [c for c, ok in zip(X_tr.columns.tolist(), surv_mask) if ok]
    print(f"Active features after imputation: {len(feat_names)} (dropped {(~surv_mask).sum()} all-NaN)")

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=4,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5,
        random_state=42, verbose=-1)
    n = X_tr_v.shape[0]; k = int(n * 0.85)
    model.fit(X_tr_v[:k], train["fwd_return_1y"].values[:k],
              eval_set=[(X_tr_v[k:], train["fwd_return_1y"].values[k:])],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_tr_v)
    np.save(PROC / "shap_values.npy", shap_vals)
    pd.DataFrame(X_tr_v, columns=feat_names).to_csv(PROC / "shap_features.csv", index=False)
    pd.DataFrame({"feature": feat_names}).to_csv(PROC / "shap_feature_names.csv", index=False)
    print(f"  SHAP shape: {shap_vals.shape}")

    # Mean absolute SHAP = global importance
    mean_abs = pd.DataFrame({"feature": feat_names,
                             "mean_abs_shap": np.abs(shap_vals).mean(axis=0)})
    mean_abs = mean_abs.sort_values("mean_abs_shap", ascending=False)
    mean_abs.to_csv(PROC / "feature_importance_shap.csv", index=False)
    print("\nTop 15 features by |SHAP|:")
    print(mean_abs.head(15).to_string(index=False))

    # Permutation importance (on test set)
    print("\nComputing permutation importance on test set...")
    base_pred = model.predict(X_te_v)
    base_mse = np.mean((test["fwd_return_1y"].values - base_pred)**2)
    perm_imp = []
    rng = np.random.RandomState(42)
    for i, fname in enumerate(feat_names):
        Xc = X_te_v.copy()
        rng.shuffle(Xc[:, i])
        p = model.predict(Xc)
        perm_imp.append({"feature": fname,
                         "delta_mse": float(np.mean((test["fwd_return_1y"].values - p)**2) - base_mse)})
    perm_df = pd.DataFrame(perm_imp).sort_values("delta_mse", ascending=False)
    perm_df.to_csv(PROC / "permutation_importance.csv", index=False)
    print("Top 10 permutation importance:")
    print(perm_df.head(10).to_string(index=False))

    # PDP for top 6 features
    print("\nComputing partial dependence for top 6 features...")
    top_feats = mean_abs.head(6)["feature"].tolist()
    pdp_data = []
    for f in top_feats:
        if f not in feat_names:
            continue
        idx = feat_names.index(f)
        col = X_tr_v[:, idx]
        grid = np.linspace(np.percentile(col, 5), np.percentile(col, 95), 30)
        for v in grid:
            X_pdp = X_tr_v.copy()
            X_pdp[:, idx] = v
            pdp_data.append({"feature": f, "value": v, "pdp": float(model.predict(X_pdp).mean())})
    pd.DataFrame(pdp_data).to_csv(PROC / "partial_dependence.csv", index=False)
    print(f"Saved PDP for: {top_feats}")

    # Save feature names list as importance ranking
    rank_df = mean_abs.merge(perm_df, on="feature", how="outer")
    rank_df.to_csv(PROC / "feature_importance_ranking.csv", index=False)


if __name__ == "__main__":
    main()

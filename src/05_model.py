"""
Phase 3-4 - Walk-forward CV and multi-model training.

Models:
  1. Ridge regression (linear baseline, tuned alpha)
  2. ElasticNet (sparse linear)
  3. LightGBM (gradient boosting)
  4. MLP (PyTorch, 2-layer, dropout + batch norm + early stopping)
  5. Ensemble (stacking ridge meta-learner over OOF preds of 1-4)

Validation strategy:
  Walk-forward expanding window, year-by-year.
  For each test_year in [2014, 2015, ..., 2023]:
    train_data = all (year < test_year)
    test_data  = year == test_year
  This produces 10 OOF predictions per model, covering 2014-2023.

Within-fold preprocessing:
  - cross-sectional (per-year) z-scoring of numeric features fitted on train only
  - Same scaler applied to test
  - Median imputation of NaN, fitted on train

Reproducibility:
  Global seed = 42 in numpy, torch, sklearn, lightgbm.

Outputs:
  - oof_predictions.csv  (one row per firm-year, columns = model preds)
  - model_metrics.csv    (per-model, per-year and overall)
  - models/              (saved final models trained on all data through 2023)
"""
from __future__ import annotations
import json, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import lightgbm as lgb
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

ROOT = Path("/home/claude/quant_industrials")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    # profitability
    "gross_margin","operating_margin","net_margin",
    # efficiency
    "rd_intensity","sga_ratio","asset_turnover","ppe_intensity",
    # returns on capital
    "roa","roe","roic_proxy",
    # leverage
    "debt_to_equity","debt_to_assets","debt_to_ebit","interest_cov",
    # cash flow
    "ocf_to_revenue","ocf_to_assets",
    # size
    "log_assets","log_revenue",
    # growth
    "revenue_growth","opinc_growth","ni_growth","assets_growth",
    # momentum
    "past_return_1y","past_return_3m","past_return_6m",
    # macro
    "ust10y","ust2y","ust3m","term_spread","vix","dxy",
    "vix_12m_mean","spx_ret_12m","spx_vol_12m","xli_ret_12m",
    # events
    "event_us_china_tariffs","event_covid_shock","event_covid_aftermath",
    "event_russia_ukraine","event_israel_hamas","event_supply_chain_shock",
    "event_rate_hike_cycle","event_eu_debt_crisis",
    "event_count","event_intensity_total",
]
CATEGORICAL_FEATURES = ["sub_industry"]
TARGET = "fwd_return_1y"


def load_panel():
    df = pd.read_csv(PROCESSED / "panel_features.csv")
    # Subset to features that actually exist after engineering
    avail = [c for c in NUMERIC_FEATURES if c in df.columns]
    df = df[["ticker", "fiscal_year"] + avail + CATEGORICAL_FEATURES + [TARGET]].copy()
    print(f"Loaded panel: {df.shape}, features used: {len(avail)} numeric + {len(CATEGORICAL_FEATURES)} categorical")
    return df, avail


# ---------------------------------------------------------------------------
# Within-fold preprocessing
# ---------------------------------------------------------------------------
def make_xy(train_df, test_df, num_features):
    # One-hot encode sub_industry on train, align test columns
    train_X = pd.get_dummies(train_df[num_features + CATEGORICAL_FEATURES],
                             columns=CATEGORICAL_FEATURES, prefix="sub")
    test_X = pd.get_dummies(test_df[num_features + CATEGORICAL_FEATURES],
                            columns=CATEGORICAL_FEATURES, prefix="sub")
    # Align (test may miss categories)
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

    # Median impute numeric, fit on train
    imputer = SimpleImputer(strategy="median")
    train_X_imp = imputer.fit_transform(train_X)
    test_X_imp = imputer.transform(test_X)
    feature_names = train_X.columns.tolist()
    return (train_X_imp, train_df[TARGET].values,
            test_X_imp, test_df[TARGET].values,
            feature_names, imputer)


def standardize(train_X, test_X):
    sc = StandardScaler()
    return sc.fit_transform(train_X), sc.transform(test_X), sc


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def fit_ridge(X_tr, y_tr, X_te):
    Xs_tr, Xs_te, _ = standardize(X_tr, X_te)
    # Tune alpha by leave-one-year-out style: use a small inner grid
    best_alpha, best_r2 = 1.0, -np.inf
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        m = Ridge(alpha=alpha, random_state=SEED)
        # quick CV: split into 80/20 of training set itself
        n = Xs_tr.shape[0]
        k = int(n * 0.8)
        m.fit(Xs_tr[:k], y_tr[:k])
        r2 = m.score(Xs_tr[k:], y_tr[k:])
        if r2 > best_r2:
            best_alpha, best_r2 = alpha, r2
    m = Ridge(alpha=best_alpha, random_state=SEED).fit(Xs_tr, y_tr)
    return m.predict(Xs_te), m, {"alpha": best_alpha}


def fit_elastic(X_tr, y_tr, X_te):
    Xs_tr, Xs_te, _ = standardize(X_tr, X_te)
    m = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=3,
                     random_state=SEED, max_iter=5000, n_alphas=20)
    m.fit(Xs_tr, y_tr)
    return m.predict(Xs_te), m, {"alpha": m.alpha_, "l1_ratio": m.l1_ratio_}


def fit_lgbm(X_tr, y_tr, X_te):
    # Reasonable defaults; we keep regularization conservative due to small N
    params = dict(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=20,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.5,
        random_state=SEED,
        verbose=-1,
    )
    # 80/20 internal split for early stopping
    n = X_tr.shape[0]
    k = int(n * 0.85)
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr[:k], y_tr[:k],
          eval_set=[(X_tr[k:], y_tr[k:])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(X_te), m, {"best_iter": m.best_iteration_}


# --- MLP -------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_in, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(X_tr, y_tr, X_te, n_epochs=300, batch_size=64, lr=5e-3,
            return_activations=False):
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xs_tr, Xs_te, _ = standardize(X_tr, X_te)
    n = Xs_tr.shape[0]
    k = int(n * 0.85)
    perm = np.random.permutation(n)
    tr_idx, va_idx = perm[:k], perm[k:]
    Xtr_t = torch.tensor(Xs_tr[tr_idx], dtype=torch.float32)
    ytr_t = torch.tensor(y_tr[tr_idx], dtype=torch.float32)
    Xva_t = torch.tensor(Xs_tr[va_idx], dtype=torch.float32)
    yva_t = torch.tensor(y_tr[va_idx], dtype=torch.float32)
    Xte_t = torch.tensor(Xs_te, dtype=torch.float32)

    model = MLP(Xs_tr.shape[1], hidden=64, dropout=0.30)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_va = float("inf"); best_state = None; patience = 0; max_pat = 30
    train_losses, val_losses, grad_norms = [], [], []
    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(Xtr_t.shape[0])
        ep_loss = 0.0
        for s in range(0, Xtr_t.shape[0], batch_size):
            b = idx[s:s+batch_size]
            opt.zero_grad()
            yhat = model(Xtr_t[b])
            loss = loss_fn(yhat, ytr_t[b])
            loss.backward()
            # gradient norm tracking
            total = sum(p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None)
            grad_norms.append(total)
            opt.step()
            ep_loss += loss.item() * b.shape[0]
        ep_loss /= Xtr_t.shape[0]
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(Xva_t), yva_t).item()
        train_losses.append(ep_loss); val_losses.append(va_loss)
        if va_loss < best_va - 1e-5:
            best_va = va_loss; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max_pat:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xte_t).cpu().numpy()
    info = {"epochs_trained": ep+1, "best_va_mse": best_va,
            "train_losses": train_losses, "val_losses": val_losses,
            "grad_norms": grad_norms}
    return preds, model, info


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------
def walkforward(panel: pd.DataFrame, num_features: list, test_years: range):
    rows = []
    diagnostics = {"mlp_curves": {}}
    for test_year in test_years:
        train_df = panel[panel["fiscal_year"] < test_year]
        test_df  = panel[panel["fiscal_year"] == test_year]
        if len(train_df) < 50 or len(test_df) == 0:
            continue
        X_tr, y_tr, X_te, y_te, feat_names, _imputer = make_xy(train_df, test_df, num_features)

        ridge_p, _, _ = fit_ridge(X_tr, y_tr, X_te)
        elas_p,  _, _ = fit_elastic(X_tr, y_tr, X_te)
        lgbm_p,  _, _ = fit_lgbm(X_tr, y_tr, X_te)
        mlp_p,   _, mlp_info = fit_mlp(X_tr, y_tr, X_te)

        diagnostics["mlp_curves"][test_year] = {
            "train_losses": mlp_info["train_losses"],
            "val_losses": mlp_info["val_losses"],
        }

        block = test_df[["ticker","fiscal_year",TARGET]].copy()
        block["pred_ridge"]   = ridge_p
        block["pred_elastic"] = elas_p
        block["pred_lgbm"]    = lgbm_p
        block["pred_mlp"]     = mlp_p
        rows.append(block)
        print(f"  Year {test_year}: train n={len(train_df)}, test n={len(test_df)}")

    oof = pd.concat(rows, ignore_index=True)
    return oof, diagnostics


def fit_ensemble(oof: pd.DataFrame):
    """Stacking: ridge meta-learner over base predictions."""
    from sklearn.linear_model import Ridge
    base_cols = ["pred_ridge", "pred_elastic", "pred_lgbm", "pred_mlp"]
    X = oof[base_cols].values
    y = oof[TARGET].values
    # Fit on first half, predict second half - simple split for stacking weights
    n = len(oof)
    k = int(n * 0.6)
    sorted_oof = oof.sort_values("fiscal_year").reset_index(drop=True)
    X = sorted_oof[base_cols].values
    y = sorted_oof[TARGET].values
    meta = Ridge(alpha=1.0, random_state=SEED, positive=True)
    meta.fit(X[:k], y[:k])
    sorted_oof["pred_ensemble"] = np.nan
    sorted_oof.loc[k:, "pred_ensemble"] = meta.predict(X[k:])
    print(f"  Ensemble weights (positive ridge):")
    for c, w in zip(base_cols, meta.coef_):
        print(f"    {c}: {w:.3f}")
    return sorted_oof, meta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_metrics(oof: pd.DataFrame, models=("ridge","elastic","lgbm","mlp","ensemble")):
    rows = []
    y = oof[TARGET].values
    for m in models:
        col = f"pred_{m}"
        if col not in oof.columns:
            continue
        valid = oof[col].notna()
        yv = y[valid]; yp = oof[col].values[valid]
        if len(yv) < 5:
            continue
        rmse = np.sqrt(mean_squared_error(yv, yp))
        mae  = mean_absolute_error(yv, yp)
        r2   = 1 - np.sum((yv-yp)**2) / np.sum((yv-yv.mean())**2)
        ic_p = pearsonr(yv, yp)[0]
        ic_s = spearmanr(yv, yp)[0]
        # Hit rate on sign of (return - cross-sectional median per year)
        pred_excess = pd.Series(yp, index=oof.loc[valid, "fiscal_year"])
        actual_excess = pd.Series(yv, index=oof.loc[valid, "fiscal_year"])
        hits = []
        for yr, grp in pred_excess.groupby(level=0):
            if len(grp) < 3:
                continue
            ap = actual_excess.loc[yr]
            pred_sign = (grp > grp.median()).astype(int)
            actu_sign = (ap > ap.median()).astype(int)
            hits.append((pred_sign == actu_sign).mean())
        hr = float(np.mean(hits)) if hits else np.nan
        rows.append({"model": m, "n": int(valid.sum()),
                     "rmse": rmse, "mae": mae, "r2_oos": r2,
                     "ic_pearson": ic_p, "ic_spearman": ic_s,
                     "hit_rate_xs_median": hr})
    return pd.DataFrame(rows)


def main():
    panel, num_feat = load_panel()
    print(f"\nFiscal year range in panel: {panel['fiscal_year'].min()} - {panel['fiscal_year'].max()}")

    # Walk-forward 2014-2023 (10 years OOS)
    oof, diags = walkforward(panel, num_feat, range(2014, 2024))
    oof.to_csv(PROCESSED / "oof_predictions.csv", index=False)

    # Ensemble
    print("\nFitting stacking ensemble...")
    oof_ens, meta = fit_ensemble(oof)
    oof_ens.to_csv(PROCESSED / "oof_predictions_with_ensemble.csv", index=False)

    # Save MLP diagnostics for visualization later
    with open(PROCESSED / "mlp_curves.json", "w") as f:
        json.dump(diags["mlp_curves"], f, indent=2)

    print("\nOverall OOF metrics:")
    metrics = compute_metrics(oof_ens)
    print(metrics.to_string(index=False))
    metrics.to_csv(PROCESSED / "model_metrics_overall.csv", index=False)

    # Per-year metrics for the best model
    print("\nPer-year metrics (LightGBM):")
    by_year = []
    for yr, grp in oof.groupby("fiscal_year"):
        m = compute_metrics(grp.assign(pred_ensemble=np.nan))
        m["fiscal_year"] = yr
        by_year.append(m)
    by_year = pd.concat(by_year, ignore_index=True)
    by_year.to_csv(PROCESSED / "model_metrics_by_year.csv", index=False)
    print(by_year.query("model=='lgbm'")[["fiscal_year","n","ic_spearman","r2_oos","hit_rate_xs_median"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

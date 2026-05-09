"""
Phase 5 - Portfolio backtest.

Cross-sectional decile sort using OOF predictions.
For each year:
  - rank stocks by predicted return (best model = LightGBM, also test ensemble)
  - form 5 quintile portfolios (more robust than deciles given N~80)
  - long-short = Q5 - Q1
  - equal weighted within quintile
  - Track cumulative return, Sharpe, max drawdown vs benchmark (XLI)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/claude/quant_industrials")
PROC = ROOT / "data" / "processed"


def quintile_portfolio_returns(oof: pd.DataFrame, model_col: str, n_quint=5):
    out = []
    for yr, grp in oof.groupby("fiscal_year"):
        g = grp.dropna(subset=[model_col]).copy()
        if len(g) < 20:
            continue
        g["quintile"] = pd.qcut(g[model_col], n_quint, labels=range(1, n_quint+1), duplicates="drop")
        for q in range(1, n_quint+1):
            sub = g[g["quintile"] == q]
            if len(sub) == 0:
                continue
            out.append({
                "fiscal_year": yr,
                "quintile": q,
                "n": len(sub),
                "mean_actual_ret": sub["fwd_return_1y"].mean(),
                "mean_pred_ret": sub[model_col].mean(),
            })
    df = pd.DataFrame(out)
    return df


def long_short_series(quint_df: pd.DataFrame, n_quint=5):
    pivot = quint_df.pivot(index="fiscal_year", columns="quintile", values="mean_actual_ret")
    pivot["LS"] = pivot[n_quint] - pivot[1]
    pivot["Q5"] = pivot[n_quint]
    pivot["Q1"] = pivot[1]
    return pivot


def perf_stats(returns: pd.Series, name=""):
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    cum = (1 + r).cumprod().iloc[-1] - 1
    geo_ann = (1 + r).prod() ** (1/n) - 1
    sharpe = r.mean() / (r.std() + 1e-9) * np.sqrt(1)  # annual returns already
    # Max drawdown on cumulative wealth
    wealth = (1 + r).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    return {"name": name, "n_years": n,
            "cumulative_return": cum,
            "annualized_return": geo_ann,
            "annualized_vol": r.std(),
            "sharpe": sharpe,
            "max_drawdown": dd,
            "hit_rate_positive": (r > 0).mean()}


def main():
    oof = pd.read_csv(PROC / "oof_predictions_with_ensemble.csv")
    print(f"OOF data: {len(oof)} obs, years {oof['fiscal_year'].min()}-{oof['fiscal_year'].max()}")

    # Quintile portfolio for LightGBM (best single model)
    quint = quintile_portfolio_returns(oof, "pred_lgbm", n_quint=5)
    quint.to_csv(PROC / "quintile_portfolios_lgbm.csv", index=False)

    series = long_short_series(quint)
    series.to_csv(PROC / "long_short_returns_lgbm.csv")
    print("\n=== LightGBM Quintile Portfolios (annual returns) ===")
    print(series.round(3).to_string())

    # Performance stats
    rows = []
    for q in [1, 2, 3, 4, 5]:
        rows.append(perf_stats(series[q], f"Q{q}"))
    rows.append(perf_stats(series["LS"], "Long-Short Q5-Q1"))
    perf = pd.DataFrame(rows)
    perf.to_csv(PROC / "portfolio_performance_lgbm.csv", index=False)
    print("\n=== Performance Statistics ===")
    print(perf.round(3).to_string(index=False))

    # Same for ensemble where available
    quint_e = quintile_portfolio_returns(oof.dropna(subset=["pred_ensemble"]), "pred_ensemble", n_quint=5)
    if len(quint_e):
        series_e = long_short_series(quint_e)
        series_e.to_csv(PROC / "long_short_returns_ensemble.csv")
        print("\n=== Ensemble Quintile Portfolios ===")
        print(series_e.round(3).to_string())

    # Compare to a benchmark: equal-weighted universe each year
    universe_ew = oof.groupby("fiscal_year")["fwd_return_1y"].mean()
    universe_ew.to_csv(PROC / "benchmark_ew_universe.csv")
    print("\n=== Equal-weighted Industrials universe (benchmark) ===")
    print(universe_ew.round(3).to_string())
    print(perf_stats(universe_ew, "EW Universe"))


if __name__ == "__main__":
    main()

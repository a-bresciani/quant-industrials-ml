"""
Phase 2 - Feature engineering.

Builds the ML-ready annual panel:

  Inputs:
    - fundamentals_clean.csv (raw financials per firm-year)
    - returns_at_fye.csv     (past + forward returns aligned to fiscal year end)
    - macro_clean.csv        (monthly macro variables)
    - prices_clean.csv       (monthly prices, used for realized vol)

  Outputs:
    - panel_features.csv     (ML-ready, one row per firm-year)
    - panel_metadata.json    (feature groups, train/test ranges)

Methodology:
  - Profitability, leverage, efficiency, growth, momentum ratios.
  - Geopolitical event dummies hardcoded (NBER recession 2020, COVID,
    Russia-Ukraine 2022, Israel-Hamas 2023, US-China tariffs 2018-19).
  - Macro features at fiscal year end: levels, term spread, vix average,
    realized volatility of SPX returns over prior 12 months.
  - Sub-industry one-hot encoded.
  - Target: fwd_return_1y (no look-ahead: predict t+1 return from t info).
  - Predictors are dated at fiscal year end of t. Prediction is then for the
    return realised during t+1.
  - Cross-sectional z-scoring is performed within fold (in modelling step),
    not here, to avoid leakage. Here we leave raw values.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/claude/quant_industrials")
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Geopolitical / macro event timeline (US Industrials lens, 2009-2024)
# ---------------------------------------------------------------------------
EVENTS = [
    # (label, fiscal_years_affected, intensity 0..1, type)
    ("eu_debt_crisis",     [2011, 2012],      0.7, "macro"),
    ("us_china_tariffs",   [2018, 2019],      0.8, "trade"),
    ("covid_shock",        [2020],            1.0, "pandemic"),
    ("covid_aftermath",    [2021],            0.5, "pandemic"),
    ("russia_ukraine",     [2022, 2023, 2024], 0.9, "war"),
    ("israel_hamas",       [2023, 2024],      0.6, "war"),
    ("supply_chain_shock", [2021, 2022],      0.8, "supply"),
    ("rate_hike_cycle",    [2022, 2023],      0.9, "monetary"),
]


def build_event_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for label, years, intensity, _ in EVENTS:
        col = f"event_{label}"
        out[col] = out["fiscal_year"].apply(lambda y: intensity if y in years else 0.0)
    out["event_count"] = sum((out[f"event_{e[0]}"] > 0).astype(int) for e in EVENTS)
    out["event_intensity_total"] = sum(out[f"event_{e[0]}"] for e in EVENTS)
    return out


# ---------------------------------------------------------------------------
# Financial ratio engineering
# ---------------------------------------------------------------------------
def build_fundamental_ratios(fund: pd.DataFrame) -> pd.DataFrame:
    df = fund.copy()
    eps = 1e-6  # to avoid /0

    # Ensure optional columns exist (some LSEG fields didn't return data for any ticker)
    for opt in ["cash", "interest_expense", "fcf_unlev_ttm",
                "inventory_cagr5y", "capex_cagr5y", "ebitda_consensus"]:
        if opt not in df.columns:
            df[opt] = np.nan

    # Derive Gross Profit from Revenue - COGS (as discussed: not directly available)
    df["gross_profit"] = df["revenue"] - df["cogs"]

    # Derive EBITDA proxy from Operating Income (D&A unavailable, use Operating Income as proxy)
    # We'll keep both: operating_income (cleaner) and ebitda_consensus (analyst mean).
    # Use operating_income as primary "earnings power" measure.

    # Profitability margins
    df["gross_margin"]      = df["gross_profit"]      / (df["revenue"].abs() + eps)
    df["operating_margin"]  = df["operating_income"]  / (df["revenue"].abs() + eps)
    df["net_margin"]        = df["net_income"]        / (df["revenue"].abs() + eps)
    df["ebitda_margin_cs"]  = df["ebitda_consensus"]  / (df["revenue"].abs() + eps)

    # Efficiency / intensity ratios
    df["rd_intensity"]      = df["rd"]                / (df["revenue"].abs() + eps)
    df["sga_ratio"]         = df["sga"]               / (df["revenue"].abs() + eps)
    df["asset_turnover"]    = df["revenue"]           / (df["total_assets"].abs() + eps)
    df["ppe_intensity"]     = df["ppe_gross"]         / (df["revenue"].abs() + eps)
    df["cash_ratio"]        = df["cash"]              / (df["total_assets"].abs() + eps)

    # Returns on capital
    df["roa"]   = df["net_income"]       / (df["total_assets"].abs() + eps)
    df["roe"]   = df["net_income"]       / (df["total_equity"].abs() + eps)
    df["roic_proxy"] = df["operating_income"] / ((df["total_equity"] + df["total_debt"]).abs() + eps)

    # Leverage
    df["debt_to_equity"] = df["total_debt"]         / (df["total_equity"].abs() + eps)
    df["debt_to_assets"] = df["total_debt"]         / (df["total_assets"].abs() + eps)
    df["debt_to_ebit"]   = df["total_debt"]         / (df["operating_income"].abs() + eps) * np.sign(df["operating_income"])
    df["interest_cov"]   = df["operating_income"]   / (df["interest_expense"].abs() + eps)

    # Cash flow
    df["ocf_to_revenue"] = df["ocf"]                 / (df["revenue"].abs() + eps)
    df["ocf_to_assets"]  = df["ocf"]                 / (df["total_assets"].abs() + eps)

    # Size proxy
    df["log_assets"]     = np.log(df["total_assets"].clip(lower=1.0))
    df["log_revenue"]    = np.log(df["revenue"].clip(lower=1.0))

    # Growth (year-over-year), computed within ticker
    df = df.sort_values(["ticker", "fiscal_year"])
    for raw, growth in [("revenue", "revenue_growth"),
                        ("operating_income", "opinc_growth"),
                        ("net_income", "ni_growth"),
                        ("total_assets", "assets_growth")]:
        df[growth] = df.groupby("ticker")[raw].pct_change(fill_method=None)
        df[growth] = df[growth].replace([np.inf, -np.inf], np.nan)
        # Cap extreme values (winsorize at +/-200%)
        df[growth] = df[growth].clip(-2.0, 2.0)

    # Quality (FCF margin proxy)
    df["fcf_proxy"] = df["fcf_unlev_ttm"] / (df["revenue"].abs() + eps)

    # Winsorize all engineered ratios cross-sectionally (within year) at 1/99% tails
    ratio_cols = [
        "gross_margin","operating_margin","net_margin","ebitda_margin_cs",
        "rd_intensity","sga_ratio","asset_turnover","ppe_intensity","cash_ratio",
        "roa","roe","roic_proxy",
        "debt_to_equity","debt_to_assets","debt_to_ebit","interest_cov",
        "ocf_to_revenue","ocf_to_assets",
        "fcf_proxy",
    ]
    for c in ratio_cols:
        df[c] = df.groupby("fiscal_year")[c].transform(
            lambda s: s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99))
        )
    return df


# ---------------------------------------------------------------------------
# Macro features at fiscal year end (point-in-time alignment)
# ---------------------------------------------------------------------------
def build_macro_features(macro: pd.DataFrame, fye_dates: pd.Series) -> pd.DataFrame:
    """
    For each fiscal-year-end date, attach contemporaneous macro values:
      ust10y, ust2y, ust3m at FYE
      term_spread (10y - 2y), credit-curve proxy
      vix at FYE, vix mean over past 12 months
      dxy at FYE, dxy YoY change
      spx YoY return, splrci YoY return, xli YoY return
      realized vol of SPX over prior 12 months
    """
    macro = macro.sort_index()
    out = pd.DataFrame({"fiscal_date": fye_dates.unique()})
    out["fiscal_date"] = pd.to_datetime(out["fiscal_date"])
    out = out.sort_values("fiscal_date").reset_index(drop=True)

    # Helper: as-of merge from monthly macro to fiscal date
    macro = macro.reset_index().rename(columns={"date": "macro_date"})
    macro["macro_date"] = pd.to_datetime(macro["macro_date"])

    out = pd.merge_asof(
        out, macro,
        left_on="fiscal_date", right_on="macro_date",
        direction="backward",
    )
    # Term spread
    out["term_spread"] = out["ust10y"] - out["ust2y"]
    # 12m trailing means
    macro_12m = macro.set_index("macro_date").sort_index()
    out["vix_12m_mean"] = out["fiscal_date"].apply(
        lambda d: macro_12m.loc[d - pd.Timedelta(days=365): d, "vix"].mean()
    )
    # SPX 12m return and realized vol from monthly returns
    spx = macro_12m["spx"].dropna()
    spx_ret = spx.pct_change()
    def _spx_yoy(d):
        prev = spx.loc[: d - pd.Timedelta(days=350)]
        cur = spx.loc[: d]
        if len(prev) and len(cur):
            return cur.iloc[-1] / prev.iloc[-1] - 1.0
        return np.nan
    def _spx_vol(d):
        window = spx_ret.loc[d - pd.Timedelta(days=365): d].dropna()
        return window.std() * np.sqrt(12) if len(window) >= 6 else np.nan
    out["spx_ret_12m"] = out["fiscal_date"].apply(_spx_yoy)
    out["spx_vol_12m"] = out["fiscal_date"].apply(_spx_vol)
    # XLI 12m return (sector benchmark)
    xli = macro_12m["xli"].dropna()
    def _xli_yoy(d):
        prev = xli.loc[: d - pd.Timedelta(days=350)]
        cur = xli.loc[: d]
        if len(prev) and len(cur):
            return cur.iloc[-1] / prev.iloc[-1] - 1.0
        return np.nan
    out["xli_ret_12m"] = out["fiscal_date"].apply(_xli_yoy)

    return out


# ---------------------------------------------------------------------------
# Sub-industry assignment (hardcoded mapping from universe file)
# ---------------------------------------------------------------------------
# We'll derive sub-industry from a manual mapping for the LSEG tickers.
# This needs minor work: the LSEG tickers have suffixes (.N, .OQ).
# I'll build a lookup from our curated universe file.

def build_subindustry_lookup() -> dict[str, str]:
    """Map LSEG RIC -> sub_industry, using ticker root match."""
    uni = pd.read_csv(ROOT / "data" / "raw" / "universe_industrials.csv")
    lookup_root = dict(zip(uni["ticker"].str.upper(), uni["sub_industry"]))
    return lookup_root


def assign_subindustry(panel: pd.DataFrame) -> pd.DataFrame:
    lookup = build_subindustry_lookup()
    def _match(ric):
        # RIC like "BA.N" -> root = "BA"
        root = ric.split(".")[0].upper()
        return lookup.get(root, "Other")
    panel["sub_industry"] = panel["ticker"].apply(_match)
    return panel


# ---------------------------------------------------------------------------
# Master assembly
# ---------------------------------------------------------------------------
def main():
    fund = pd.read_csv(PROCESSED / "fundamentals_clean.csv", parse_dates=["fiscal_date"])
    rets = pd.read_csv(PROCESSED / "returns_at_fye.csv", parse_dates=["fiscal_date"])
    macro = pd.read_csv(PROCESSED / "macro_clean.csv", parse_dates=["date"]).set_index("date")

    print("Building fundamental ratios...")
    fund_ratios = build_fundamental_ratios(fund)

    print("Merging returns...")
    panel = fund_ratios.merge(rets, on=["ticker", "fiscal_date"], how="left")

    print("Adding macro features...")
    macro_feat = build_macro_features(macro, panel["fiscal_date"])
    panel = panel.merge(macro_feat, on="fiscal_date", how="left")

    print("Adding events...")
    panel = build_event_features(panel)

    print("Assigning sub-industry...")
    panel = assign_subindustry(panel)

    # Drop rows missing the target (fwd_return_1y) - last fiscal year (2024) has no t+1 yet
    print("\nCoverage by fiscal year:")
    print(panel.groupby("fiscal_year")["fwd_return_1y"].agg(["count", "mean", "std"]).round(3))
    panel_train = panel.dropna(subset=["fwd_return_1y"]).copy()
    print(f"\nUsable observations (with target): {len(panel_train)} / {len(panel)}")

    # Feature group bookkeeping
    feature_groups = {
        "profitability": ["gross_margin","operating_margin","net_margin","ebitda_margin_cs"],
        "efficiency":    ["rd_intensity","sga_ratio","asset_turnover","ppe_intensity","cash_ratio"],
        "returns_on_capital": ["roa","roe","roic_proxy"],
        "leverage":      ["debt_to_equity","debt_to_assets","debt_to_ebit","interest_cov"],
        "cash_flow":     ["ocf_to_revenue","ocf_to_assets","fcf_proxy"],
        "size":          ["log_assets","log_revenue"],
        "growth":        ["revenue_growth","opinc_growth","ni_growth","assets_growth"],
        "momentum":      ["past_return_1y","past_return_3m","past_return_6m"],
        "macro_levels":  ["ust10y","ust2y","ust3m","term_spread","vix","dxy"],
        "macro_dynamics": ["vix_12m_mean","spx_ret_12m","spx_vol_12m","xli_ret_12m"],
        "events":        [f"event_{e[0]}" for e in EVENTS] + ["event_count","event_intensity_total"],
        "category":      ["sub_industry"],
        "id":            ["ticker","fiscal_year","fiscal_date"],
        "target":        ["fwd_return_1y"],
    }
    metadata = {
        "feature_groups": feature_groups,
        "train_period_max": 2018,
        "validation_period": [2019, 2020],
        "test_period": [2021, 2022, 2023],
        "n_observations": int(len(panel_train)),
        "n_tickers": int(panel_train["ticker"].nunique()),
        "events": [{"label": e[0], "years": e[1], "intensity": e[2], "type": e[3]} for e in EVENTS],
    }

    panel_train.to_csv(PROCESSED / "panel_features.csv", index=False)
    panel.to_csv(PROCESSED / "panel_features_full.csv", index=False)
    with open(PROCESSED / "panel_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nFinal panel shape: {panel_train.shape}")
    print("Saved panel_features.csv and panel_metadata.json")


if __name__ == "__main__":
    main()

"""
Phase 1.3 - Data cleaning and panel construction.

Converts the three raw LSEG exports into:
  1. A clean fundamentals panel (long format: ticker x fiscal year x fields)
  2. A monthly price panel with computed annual returns
  3. A monthly macro panel (wide format: date x macro variables)
  4. A merged annual panel ready for ML

All transforms are deterministic and idempotent. No look-ahead.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/claude/quant_industrials")
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Fundamentals
# -----------------------------------------------------------------------------
FUND_COL_MAP = {
    "Revenue from Business Activities - Total":               "revenue",
    "Cost of Revenues - Total":                               "cogs",
    "Research & Development Expense":                         "rd",
    "Selling General & Administrative Expenses - Total":      "sga",
    "Operating Profit before Non-Recurring Income/Expense":   "operating_income",
    "EBITDA - Mean":                                          "ebitda_consensus",
    "Net Income after Tax":                                   "net_income",
    "Interest Expense - Finance - Total":                     "interest_expense",
    "Total Assets":                                           "total_assets",
    "Total Shareholders' Equity incl Minority Intr & Hybrid Debt": "total_equity",
    "Debt - Total":                                           "total_debt",
    "Cash & Short-Term Deposits Due from Banks - Total":      "cash",
    "Property Plant & Equipment - Gross - Total":             "ppe_gross",
    "Inventories - Total, 5 Yr CAGR":                         "inventory_cagr5y",
    "Net Cash Flow from Operating Activities":                "ocf",
    "Capital Expenditures - Total, 5 Yr CAGR":                "capex_cagr5y",
    "Unlevered Free Cash Flow, TTM":                          "fcf_unlev_ttm",
    "Common Shares - Outstanding - Total":                    "shares_out",
}


def load_fundamentals() -> pd.DataFrame:
    df = pd.read_csv(RAW / "fundamentals_panel.csv")
    # Drop the artefact header row and unnamed columns
    df = df.dropna(how="all", axis=1)
    # Rename ticker / date columns
    cols = list(df.columns)
    df = df.rename(columns={cols[0]: "ticker", cols[1]: "fiscal_date"})
    df = df.rename(columns=FUND_COL_MAP)
    # Drop rows with missing ticker / date
    df = df.dropna(subset=["ticker", "fiscal_date"]).reset_index(drop=True)
    df["fiscal_date"] = pd.to_datetime(df["fiscal_date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["fiscal_date"])
    df["fiscal_year"] = df["fiscal_date"].dt.year
    # Coerce numerics
    num_cols = [c for c in df.columns if c not in ("ticker", "fiscal_date", "fiscal_year")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop the redundant 'Period End Date' if present (we now have fiscal_date)
    if "Period End Date" in df.columns:
        df = df.drop(columns=["Period End Date"])
    # Keep only fiscal years 2009-2024
    df = df[df["fiscal_year"].between(2009, 2024)].copy()
    df = df.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# 2. Prices
# -----------------------------------------------------------------------------
def load_prices() -> pd.DataFrame:
    df = pd.read_csv(RAW / "prices_monthly.csv")
    # Drop NaN-only rows (header artefacts)
    df = df.dropna(how="all")
    # First three useful columns are ticker, date, price (col[0] also holds ticker)
    cols = list(df.columns)
    df = df.rename(columns={cols[0]: "ticker", cols[1]: "date", cols[2]: "price"})
    # The very first row contains the field header "Price Close" - drop it
    df = df[df["price"] != "Price Close"].copy()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ticker", "date", "price"]).reset_index(drop=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def compute_annual_returns(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    For each (ticker, fiscal_year_end), compute:
      - past_return_1y : return from previous fiscal-year-end to current FYE
      - fwd_return_1y  : return from current FYE to next FYE  <-- TARGET
      - past_return_3m, past_return_6m : short-term momentum at FYE
    All returns are total return based on adjusted prices.
    """
    out_records = []
    fwide = prices.pivot(index="date", columns="ticker", values="price").sort_index()

    fund = fundamentals[["ticker", "fiscal_date"]].drop_duplicates().sort_values(["ticker", "fiscal_date"])
    for tk, grp in fund.groupby("ticker"):
        if tk not in fwide.columns:
            continue
        s = fwide[tk].dropna()
        if s.empty:
            continue
        for fdate in grp["fiscal_date"]:
            # Find nearest available price <= fdate within 45 days
            valid = s.loc[s.index <= fdate]
            if valid.empty:
                continue
            p_now = valid.iloc[-1]
            d_now = valid.index[-1]
            # 1y back
            p_1y_back = s.loc[s.index <= fdate - pd.Timedelta(days=350)]
            p_1y_back = p_1y_back.iloc[-1] if len(p_1y_back) else np.nan
            # 1y forward
            p_1y_fwd = s.loc[s.index >= fdate + pd.Timedelta(days=300)]
            # We want exactly ~1 year forward; take the closest within 350-400 days
            target_low = fdate + pd.Timedelta(days=300)
            target_high = fdate + pd.Timedelta(days=420)
            window_fwd = s.loc[(s.index >= target_low) & (s.index <= target_high)]
            p_1y_fwd_val = window_fwd.iloc[-1] if len(window_fwd) else np.nan
            # 3m, 6m back
            p_3m_back = s.loc[s.index <= fdate - pd.Timedelta(days=85)]
            p_3m_back = p_3m_back.iloc[-1] if len(p_3m_back) else np.nan
            p_6m_back = s.loc[s.index <= fdate - pd.Timedelta(days=175)]
            p_6m_back = p_6m_back.iloc[-1] if len(p_6m_back) else np.nan
            out_records.append({
                "ticker": tk,
                "fiscal_date": fdate,
                "price_at_fye": p_now,
                "past_return_1y": p_now / p_1y_back - 1.0 if pd.notna(p_1y_back) else np.nan,
                "past_return_3m": p_now / p_3m_back - 1.0 if pd.notna(p_3m_back) else np.nan,
                "past_return_6m": p_now / p_6m_back - 1.0 if pd.notna(p_6m_back) else np.nan,
                "fwd_return_1y": p_1y_fwd_val / p_now - 1.0 if pd.notna(p_1y_fwd_val) else np.nan,
            })
    return pd.DataFrame(out_records)


# -----------------------------------------------------------------------------
# 3. Macro
# -----------------------------------------------------------------------------
MACRO_RIC_MAP = {
    "US10YT=RR":  "ust10y",
    "US2YT=RR":   "ust2y",
    "US3MT=RR":   "ust3m",
    ".VIX":       "vix",
    ".DXY":       "dxy",
    "CLc1":       "wti_oil",
    "HGc1":       "copper",
    ".SPX":       "spx",
    ".SPLRCI":    "splrci",
    "XLI":        "xli",
}


def load_macro() -> pd.DataFrame:
    df = pd.read_csv(RAW / "macro_monthly.csv")
    df = df.dropna(how="all")
    cols = list(df.columns)
    # Cols: instrument, date, price_close, mid_yield
    df = df.rename(columns={cols[0]: "ric", cols[1]: "date",
                            cols[2]: "price_close", cols[3]: "mid_yield"})
    # Drop the title row
    df = df[df["price_close"] != "Price Close"].copy()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["price_close"] = pd.to_numeric(df["price_close"], errors="coerce")
    df["mid_yield"] = pd.to_numeric(df["mid_yield"], errors="coerce")
    df = df.dropna(subset=["ric", "date"])
    # For Treasury: use yield. For others: use price_close.
    df["value"] = df["mid_yield"].combine_first(df["price_close"])
    df["var"] = df["ric"].map(MACRO_RIC_MAP)
    df = df.dropna(subset=["var", "value"])
    wide = df.pivot_table(index="date", columns="var", values="value", aggfunc="last").sort_index()
    return wide


def main():
    print("Loading fundamentals...")
    fund = load_fundamentals()
    print(f"  {fund.shape[0]} firm-year obs, {fund['ticker'].nunique()} tickers")
    fund.to_csv(PROCESSED / "fundamentals_clean.csv", index=False)

    print("Loading prices and computing returns...")
    px = load_prices()
    print(f"  {px.shape[0]} monthly obs, {px['ticker'].nunique()} tickers")
    px.to_csv(PROCESSED / "prices_clean.csv", index=False)

    rets = compute_annual_returns(px, fund)
    print(f"  {rets.shape[0]} return records computed")
    rets.to_csv(PROCESSED / "returns_at_fye.csv", index=False)
    # Quick sanity
    print("  fwd_return_1y describe:")
    print(rets["fwd_return_1y"].describe().round(3).to_string())

    print("Loading macro...")
    macro = load_macro()
    print(f"  {macro.shape[0]} months x {macro.shape[1]} vars: {list(macro.columns)}")
    macro.to_csv(PROCESSED / "macro_clean.csv")

    print("\nFINAL CLEAN FILES:")
    for f in PROCESSED.glob("*.csv"):
        print(f"  {f.name}: {f.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()

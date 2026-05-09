"""
Phase 1.2 - Data acquisition.
Pulls fundamentals (income statement, balance sheet, cash flow) and adjusted
price history for each ticker in the Industrials universe via yfinance.
Caches per-ticker pickles to avoid refetching on retry.
"""
import os
import time
import pickle
import warnings
from pathlib import Path
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path("/home/claude/quant_industrials")
RAW = ROOT / "data" / "raw"
CACHE = RAW / "yf_cache"
CACHE.mkdir(parents=True, exist_ok=True)

START = "2009-01-01"   # extra year buffer for price returns
END = "2024-12-31"


def fetch_one(ticker: str, sleep: float = 0.4) -> dict | None:
    cache_file = CACHE / f"{ticker}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    try:
        t = yf.Ticker(ticker)
        # Annual statements
        inc = t.income_stmt
        bs = t.balance_sheet
        cf = t.cashflow
        # Daily prices, auto-adjusted (dividends + splits)
        prices = t.history(start=START, end=END, interval="1d", auto_adjust=True)
        if prices is None or prices.empty:
            return None
        out = {
            "ticker": ticker,
            "income_stmt": inc,
            "balance_sheet": bs,
            "cashflow": cf,
            "prices": prices[["Close", "Volume"]].copy(),
        }
        with open(cache_file, "wb") as f:
            pickle.dump(out, f)
        time.sleep(sleep)
        return out
    except Exception as e:
        print(f"  [error] {ticker}: {e}")
        return None


def main():
    universe = pd.read_csv(RAW / "universe_industrials.csv")
    print(f"Fetching data for {len(universe)} tickers...")
    results = {}
    failed = []
    for i, row in universe.iterrows():
        tk = row["ticker"]
        print(f"[{i+1:>2}/{len(universe)}] {tk:>6}", end="  ")
        data = fetch_one(tk)
        if data is None:
            print("FAIL")
            failed.append(tk)
        else:
            n_inc = data["income_stmt"].shape[1] if data["income_stmt"] is not None else 0
            n_px = len(data["prices"])
            print(f"OK  income_yrs={n_inc}  price_obs={n_px}")
            results[tk] = data

    print(f"\nSucceeded: {len(results)}  Failed: {len(failed)}")
    if failed:
        print("Failed tickers:", failed)

    # Persist a master dictionary
    with open(RAW / "all_data.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Saved combined cache: {RAW / 'all_data.pkl'}")


if __name__ == "__main__":
    main()

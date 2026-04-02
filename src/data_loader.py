"""
data_loader.py — Yahoo Finance data download, caching, and preprocessing.

Downloads OHLCV data for stocks + market data (SPY, VIX) for the regime filter.
Caches everything as CSV in data/ directory to avoid re-downloading.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from config import (
    DATA_DIR, UNIVERSE, SPY_TICKER, VIX_TICKER,
    START_DATE, END_DATE, ADV_PERIODS, vwap_proxy
)


def _ensure_dirs():
    """Create data directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "stocks"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "market"), exist_ok=True)


def _cache_path(ticker, subdir="stocks"):
    """Return the CSV cache path for a ticker."""
    safe_name = ticker.replace("^", "_").replace("-", "_")
    return os.path.join(DATA_DIR, subdir, f"{safe_name}.csv")


def download_stock(ticker, start=START_DATE, end=END_DATE, force=False):
    """
    Download OHLCV data for a single ticker from Yahoo Finance.
    Caches to CSV. Returns DataFrame or None on failure.
    """
    path = _cache_path(ticker)
    if os.path.exists(path) and not force:
        print(f"  ✓ {ticker} (cached)")
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    print(f"  Downloading {ticker}...")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            print(f"  ⚠️ No data for {ticker}")
            return None

        # Flatten multi-level columns if present (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only needed columns
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
        df.index.name = "Date"

        # Save cache
        df.to_csv(path)
        return df

    except Exception as e:
        print(f"  ❌ Failed to download {ticker}: {e}")
        return None


def download_all_stocks(force=False):
    """Download data for the entire universe. Returns dict of {ticker: DataFrame}."""
    _ensure_dirs()
    print(f"📦 Loading {len(UNIVERSE)} stocks (cached data used when available)...")
    data = {}
    downloaded = 0
    cached = 0
    for ticker in UNIVERSE:
        path = _cache_path(ticker)
        was_cached = os.path.exists(path) and not force
        df = download_stock(ticker, force=force)
        if df is not None:
            data[ticker] = df
            if was_cached:
                cached += 1
            else:
                downloaded += 1
    print(f"✅ Loaded {len(data)}/{len(UNIVERSE)} stocks ({cached} from cache, {downloaded} freshly downloaded)")
    return data


def download_market_data(force=False):
    """Download SPY and VIX data for the regime filter."""
    _ensure_dirs()
    print("📦 Loading market data (SPY + VIX)...")
    market = {}
    for ticker in [SPY_TICKER, VIX_TICKER]:
        path = _cache_path(ticker, subdir="market")
        if os.path.exists(path) and not force:
            print(f"  ✓ {ticker} (cached)")
            market[ticker] = pd.read_csv(path, index_col="Date", parse_dates=True)
        else:
            print(f"  Downloading {ticker}...")
            try:
                df = yf.download(ticker, start=START_DATE, end=END_DATE,
                                 auto_adjust=False, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
                df.index.name = "Date"
                df.to_csv(path)
                market[ticker] = df
            except Exception as e:
                print(f"  ❌ Failed to download {ticker}: {e}")
    return market


def preprocess_stock(df):
    """
    Add computed columns to a single stock's DataFrame:
      - VWAP (proxy): (H + L + 2*C) / 4
      - returns: daily close-to-close log returns
      - adv{d}: average daily dollar volume for past d days
    """
    df = df.copy()

    # Use adjusted close for returns calculation (handles splits/dividends)
    df["returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

    # VWAP proxy
    df["vwap"] = vwap_proxy(df["High"], df["Low"], df["Close"])

    # Average daily dollar volume (ADV) for various lookback periods
    dollar_volume = df["Close"] * df["Volume"]
    for d in ADV_PERIODS:
        df[f"adv{d}"] = dollar_volume.rolling(d).mean()

    return df


def build_panel(stock_data):
    """
    Build panel DataFrames (tickers × dates) for each field.

    Input: dict of {ticker: DataFrame}
    Output: dict of {field_name: DataFrame} where each DataFrame has
            dates as index and tickers as columns.
    """
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume",
              "vwap", "returns"] + [f"adv{d}" for d in ADV_PERIODS]

    # Preprocess all stocks
    processed = {}
    for ticker, df in stock_data.items():
        processed[ticker] = preprocess_stock(df)

    # Find common date range
    all_dates = None
    for ticker, df in processed.items():
        if all_dates is None:
            all_dates = set(df.index)
        else:
            all_dates = all_dates.intersection(set(df.index))

    all_dates = sorted(all_dates)
    print(f"📊 Panel: {len(processed)} stocks × {len(all_dates)} trading days "
          f"({all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')})")

    # Build panel for each field
    panels = {}
    for field in fields:
        panel_dict = {}
        for ticker, df in processed.items():
            if field in df.columns:
                panel_dict[ticker] = df.loc[df.index.isin(all_dates), field]
        panels[field] = pd.DataFrame(panel_dict, index=all_dates)

    return panels


def load_all_data(force=False):
    """
    Master function: download everything, preprocess, return panels + market data.

    Returns:
        panels: dict of {field: DataFrame(dates×tickers)}
        market: dict of {ticker: DataFrame}
    """
    stock_data = download_all_stocks(force=force)
    market_data = download_market_data(force=force)
    panels = build_panel(stock_data)
    return panels, market_data


if __name__ == "__main__":
    panels, market = load_all_data()
    print(f"\nPanel fields: {list(panels.keys())}")
    print(f"Close panel shape: {panels['Close'].shape}")
    print(f"Market data keys: {list(market.keys())}")

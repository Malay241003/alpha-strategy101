"""
crypto_data_loader.py — Download and cache crypto OHLCV data from Yahoo Finance.

Handles:
  - Downloading daily crypto data (BTC-USD, ETH-USD, etc.)
  - Caching to data/crypto/ directory
  - Handling coins with short history (skip if < MIN_HISTORY_DAYS)
  - Loading BTC as regime proxy
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_crypto(ticker, start_date, end_date, data_dir):
    """
    Download daily OHLCV for a single crypto from Yahoo Finance.
    Caches to CSV to avoid re-downloading.

    Args:
        ticker: yfinance format (e.g. "BTC-USD")
        start_date, end_date: date strings
        data_dir: cache directory

    Returns:
        DataFrame with OHLCV or None if download failed / too short
    """
    os.makedirs(data_dir, exist_ok=True)
    safe_name = ticker.replace("-", "_").replace("/", "_")
    cache_file = os.path.join(data_dir, f"{safe_name}.csv")

    # Load from cache if exists
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(df) > 0:
            print(f"  ✓ {ticker} (cached, {len(df)} days)")
            return df

    # Download fresh
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                         progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            print(f"  ✗ {ticker}: no data")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Save to cache
        df.to_csv(cache_file)
        print(f"  ✓ {ticker} (downloaded, {len(df)} days)")
        return df

    except Exception as e:
        print(f"  ✗ {ticker}: {str(e)[:60]}")
        return None


def load_all_crypto_data():
    """
    Load all crypto data for the configured universe.

    Handles coins with short history by:
      - Downloading all available data from START_DATE
      - Checking if coin has >= MIN_HISTORY_DAYS
      - Filling shorter-history coins with NaN (excluded from ML training
        for those dates, but included when data is available)

    Returns:
        panels: dict of {field: DataFrame(dates × tickers)}
        btc_data: DataFrame of BTC-USD for regime calculation
        valid_tickers: list of tickers that passed the minimum history check
    """
    from crypto_config import (
        UNIVERSE, UNIVERSE_TICKERS, REGIME_TICKER,
        START_DATE, END_DATE, DATA_DIR, MIN_HISTORY_DAYS,
        ADV_PERIODS, vwap_proxy
    )
    import numpy as np

    print(f"📦 Loading {len(UNIVERSE_TICKERS)} crypto assets (cached data used when available)...")

    all_data = {}
    valid_tickers = []
    skipped = 0

    for coin, ticker in zip(UNIVERSE, UNIVERSE_TICKERS):
        df = download_crypto(ticker, START_DATE, END_DATE, DATA_DIR)

        if df is None or len(df) < MIN_HISTORY_DAYS:
            if df is not None:
                print(f"  ⚠️ {coin}: only {len(df)} days (need {MIN_HISTORY_DAYS}), skipping")
            skipped += 1
            continue

        # Preprocess: Add derived fields
        df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["vwap"] = vwap_proxy(df["High"], df["Low"], df["Close"])
        
        dollar_volume = df["Close"] * df["Volume"]
        for d in ADV_PERIODS:
            df[f"adv{d}"] = dollar_volume.rolling(d).mean()

        all_data[coin] = df
        valid_tickers.append(coin)

    print(f"✅ Loaded {len(valid_tickers)}/{len(UNIVERSE)} crypto assets "
          f"({skipped} skipped for insufficient history)")

    # Build aligned panel DataFrames
    # Find all unique dates across all assets
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    fields = ["Open", "High", "Low", "Close", "Volume", "vwap", "returns"] + [f"adv{d}" for d in ADV_PERIODS]
    panels = {}

    for field in fields:
        panel = pd.DataFrame(index=all_dates, columns=valid_tickers)
        for ticker in valid_tickers:
            df = all_data[ticker]
            if field in df.columns:
                panel[ticker] = df[field].reindex(all_dates)
        panels[field] = panel.astype(float)

    # Drop dates where all values are NaN
    valid_mask = panels["Close"].notna().any(axis=1)
    for field in fields:
        panels[field] = panels[field].loc[valid_mask]

    # Load BTC for regime
    print(f"\n📦 Loading regime proxy ({REGIME_TICKER})...")
    btc_data = download_crypto(REGIME_TICKER, START_DATE, END_DATE, DATA_DIR)

    print(f"📊 Panel: {len(valid_tickers)} assets × {len(panels['Close'])} trading days "
          f"({panels['Close'].index[0].strftime('%Y-%m-%d')} to "
          f"{panels['Close'].index[-1].strftime('%Y-%m-%d')})")

    return panels, btc_data, valid_tickers

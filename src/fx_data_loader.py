"""
fx_data_loader.py — Fetches and caches historical daily FX data from yfinance.
"""

import sys, os
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_config import FX_UNIVERSE, DATA_DIR, DATA_START_DATE

def fetch_yfinance_forex_data(tickers=FX_UNIVERSE, start_date=DATA_START_DATE):
    """
    Download or load cached yfinance daily OHLCV for Forex pairs.
    """
    print(f"\n📦 Fetching 20+ Years of Macro Forex Data...")
    
    close_dict = {}
    high_dict = {}
    low_dict = {}
    open_dict = {}
    vol_dict = {}
    
    valid_tickers = []
    
    for ticker in tickers:
        safe_name = ticker.replace("=X", "")
        cache_file = os.path.join(DATA_DIR, f"{safe_name}_1d.csv")
        
        # Load from cache if recent, else download
        # (For simplicity in backtesting we pull full history if missing)
        df = None
        if os.path.exists(cache_file):
            print(f"  ✓ {safe_name} (checking cache)")
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Check if it has recent data (e.g. up to 2024 at least)
                if df.index[-1].year < 2024:
                    print(f"    - Cache stale. Redownloading...")
                    df = None
            except Exception:
                df = None
                
        if df is None:
            print(f"  ↓ Downloading {ticker} from {start_date}...")
            # Suppress yf logging noise
            df = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
            if df.empty:
                print(f"  ✗ Failed to download {ticker}")
                continue
            
            # Save to cache
            df.to_csv(cache_file)
            
        print(f"  ✓ {safe_name} Loaded ({len(df):,} days)")
        
        # Handle cases where some days might have zero volume (common in FX on Yahoo)
        # We fill Volume with 1 if it's strictly 0 to prevent div by zero in indicators
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].replace(0, 1)
        else:
            df["Volume"] = 1
            
        # Optional: standardize pairs that are inverted (e.g., USDJPY -> JPY=X means Quote is JPY)
        # For uniformity, we handle them as they are priced.
        
        close_dict[ticker] = df["Close"]
        high_dict[ticker] = df["High"]
        low_dict[ticker] = df["Low"]
        open_dict[ticker] = df["Open"]
        vol_dict[ticker] = df["Volume"]
        
        valid_tickers.append(ticker)
        
    if not valid_tickers:
        return {}, []
        
    panels = {
        "Close": pd.DataFrame(close_dict),
        "High": pd.DataFrame(high_dict),
        "Low": pd.DataFrame(low_dict),
        "Open": pd.DataFrame(open_dict),
        "Volume": pd.DataFrame(vol_dict),
    }
    
    # Forward fill weekends/holidays to align data perfectly
    for key in panels:
        panels[key] = panels[key].ffill()
        
    print(f"\n✅ All FX data loaded and aligned. Shape: {panels['Close'].shape[0]:,} Dates × {len(valid_tickers)} Pairs")
    print(f"  Timeline: {panels['Close'].index[0].date()} to {panels['Close'].index[-1].date()}")
    
    return panels, valid_tickers

if __name__ == "__main__":
    fetch_yfinance_forex_data()

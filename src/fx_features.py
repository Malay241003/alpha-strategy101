"""
fx_features.py — Computes Nonlinear Time Series Momentum features for Forex.

Focuses on:
  1. Multi-horizon momentum (1M, 3M, 6M, 12M)
  2. Rolling volatility regimes
  3. Cross-sectional strength (relative to other USD pairs)
"""

import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_atr(high, low, close, window=14):
    """Compute Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.DataFrame(
        np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
        index=high.index, columns=high.columns
    )
    return tr.rolling(window=window).mean()

def build_fx_features(panels):
    """
    Construct the feature matrix for the Forex Marco strategy.
    Expects Daily panels covering 20+ years.
    """
    print("\n📊 Building FX Macro Feature Matrix...")
    
    close = panels["Close"]
    high = panels["High"]
    low = panels["Low"]
    
    dates = close.index
    tickers = close.columns
    
    # ── 1. Time Series Momentum (Trend over various horizons) ──
    # 1 Month (~21 days), 3 Months (~63 days), 6 Months (~126 days), 12 Months (~252 days)
    ret_1m = close.pct_change(21)
    ret_3m = close.pct_change(63)
    ret_6m = close.pct_change(126)
    ret_12m = close.pct_change(252)
    
    # Simple moving average crossovers as state features
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    trend_50_200 = (sma_50 - sma_200) / close
    
    # ── 2. Volatility Regimes ──
    atr_14 = compute_atr(high, low, close, 14)
    vol_scale = atr_14 / close # Normalized volatility
    
    # Volatility expansion/contraction
    vol_1m = vol_scale.rolling(21).mean()
    vol_6m = vol_scale.rolling(126).mean()
    vol_regime = vol_1m / vol_6m.replace(0, 1e-10)
    
    # ── 3. Cross-Sectional Strength (Relative Momentum) ──
    # How strong is this pair's 3M return compared to the average of all USD pairs?
    # Ranks return from 0 (weakest) to 1 (strongest) each day across the row
    xs_rank_3m = ret_3m.rank(axis=1, pct=True)
    
    # ── Combine to Panel List ──
    records = []
    
    # Skip the first 252 days (1 year) to allow 12M momentum to initialize without NaNs
    valid_dates = dates[252:]
    
    for date in valid_dates:
        for ticker in tickers:
            if pd.isna(close.loc[date, ticker]):
                continue
                
            row = {"date": date, "ticker": ticker}
            
            # Momentum Features
            row["ret_1m"] = ret_1m.loc[date, ticker]
            row["ret_3m"] = ret_3m.loc[date, ticker]
            row["ret_6m"] = ret_6m.loc[date, ticker]
            row["ret_12m"] = ret_12m.loc[date, ticker]
            row["trend_50_200"] = trend_50_200.loc[date, ticker]
            
            # Volatility Features
            row["vol_regime"] = vol_regime.loc[date, ticker]
            
            # Cross-Sectional Feature
            row["xs_rank_3m"] = xs_rank_3m.loc[date, ticker]
            
            records.append(row)
            
    df = pd.DataFrame(records)
    
    print("  --- NaN counts per column before dropna ---")
    print(df.isna().sum())
    
    df.dropna(inplace=True)
    print(f"  ✅ Feature matrix built: {len(df):,} rows × {len(df.columns) - 2} features")
    
    return df

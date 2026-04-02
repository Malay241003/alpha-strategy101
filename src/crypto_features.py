"""
crypto_features.py — Compute SSRN-backed crypto-native ML features.

Unlike daily equity alphas, this module computes lower-timeframe 
microstructure features that perform well in crypto market prediction:
  1. Order Flow Imbalance (Volume * Delta Close)
  2. Aggressive Taker Volume Ratios
  3. Liquidity/Volume Shocks (vs moving averages)
  4. Realized Volatility & ATR Expansions
  5. Short-term momentum oscillators (RSI, MACD)
"""

import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════
# FEATURE FUNCTIONS
# ═══════════════════════════════════════════════

def compute_order_flow_imbalance(panels, window=10):
    """
    Approximates buying vs. selling pressure using the close position 
    relative to the candle's high/low range, scaled by volume.
    High value = strong buying pressure.
    Low value = strong selling pressure.
    """
    close = panels["Close"]
    high = panels["High"]
    low = panels["Low"]
    volume = panels["Volume"]
    
    range_hl = (high - low).replace(0, 1e-10)
    # Location indicator: +1 (close at high), -1 (close at low)
    location = (2 * close - high - low) / range_hl
    
    # Imbalance = location * volume
    imbalance = location * volume
    
    # Smooth to capture trend in order flow
    return imbalance.rolling(window=window).mean()


def compute_taker_buy_ratio(panels, window=10):
    """
    Binance specific: Taker buy base asset volume / Total volume.
    A ratio > 0.5 indicates aggressive market buys exceed market sells.
    """
    taker_buy = panels.get("taker_buy_base_asset_volume")
    total_vol = panels.get("Volume")
    
    if taker_buy is None or total_vol is None:
        return None
        
    ratio = taker_buy / total_vol.replace(0, 1e-10)
    return ratio.rolling(window=window).mean()


def compute_liquidity_shock(panels, short_window=3, long_window=24):
    """
    Detects sudden volume spikes. 
    Ratio of short-term volume MA to long-term volume MA.
    """
    volume = panels["Volume"]
    short_ma = volume.rolling(window=short_window).mean()
    long_ma = volume.rolling(window=long_window).mean()
    
    return short_ma / long_ma.replace(0, 1e-10)


def compute_atr_expansion(panels, window=14):
    """
    Measures volatility expansion. Current ATR vs 3x window ATR.
    """
    high = panels["High"]
    low = panels["Low"]
    close = panels["Close"]
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    # Calculate element-wise maximum across the 3 DataFrames
    tr = pd.DataFrame(
        np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
        index=high.index, columns=high.columns
    )
    
    atr_short = tr.rolling(window=window).mean()
    atr_long = tr.rolling(window=window * 3).mean()
    
    return atr_short / atr_long.replace(0, 1e-10)


def compute_rsi(close, window=14):
    """Standard Relative Strength Index"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Calculate RMA (Rowler Moving Average) style smoothing to match standard RSI
    for i in range(window, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (window - 1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (window - 1) + loss.iloc[i]) / window

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════
# MAIN BUILDER
# ═══════════════════════════════════════════════

def build_crypto_features(panels, btc_panels=None):
    """
    Construct the full feature matrix for all crypto assets.
    
    Returns:
        DataFrame: MultiIndex (date, ticker) with all computed features.
    """
    print("\n📊 STEP 4: Building Crypto-Native ML Feature Matrix...")
    
    dates = panels["Close"].index
    tickers = panels["Close"].columns
    
    close = panels["Close"]
    
    # Ensure dates and tickers are aligned
    print(f"  Calculating base features for {len(tickers)} assets...")
    
    # ── 1. Momentum & Trend ──
    returns_1h = close.pct_change(1)
    returns_4h = close.pct_change(4)
    returns_12h = close.pct_change(12)
    returns_24h = close.pct_change(24)
    
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    trend_9_21 = (ema_9 - ema_21) / close
    
    # Calculate RSI for all tickers
    rsi_14 = pd.DataFrame(index=dates, columns=tickers)
    for ticker in tickers:
        rsi_14[ticker] = compute_rsi(close[ticker], 14)
        
    # ── 2. Microstructure & Order Flow ──
    ofi_10 = compute_order_flow_imbalance(panels, 10)
    taker_ratio_4 = compute_taker_buy_ratio(panels, 4)
    taker_ratio_12 = compute_taker_buy_ratio(panels, 12)
    
    # ── 3. Volatility & Liquidity ──
    vol_24h = returns_1h.rolling(24).std() * np.sqrt(24 * 365)  # Annualized 1H vol
    liq_shock = compute_liquidity_shock(panels, 3, 24)
    atr_exp = compute_atr_expansion(panels, 14)
    
    # ── 4. BTC Regime / Market Context (If provided) ──
    if btc_panels is not None:
        btc_close = btc_panels["Close"].squeeze()
        btc_ret_24h = btc_close.pct_change(24)
        # Forward fill BTC features across all columns
        btc_trend_feature = pd.DataFrame(
            np.tile(btc_ret_24h.values, (len(tickers), 1)).T,
            index=dates, columns=tickers
        )
    else:
        btc_trend_feature = pd.DataFrame(0, index=dates, columns=tickers)

    # ── Combine to Long Format ──
    print("  Stacking features into panel format...")
    
    records = []
    
    # Skip warm-up period to avoid NaNs
    valid_dates = dates[30:] 
    
    for date in valid_dates:
        for ticker in tickers:
            if pd.isna(close.loc[date, ticker]):
                continue
                
            row = {"date": date, "ticker": ticker}
            
            # Momentum
            row["ret_1h"] = returns_1h.loc[date, ticker]
            row["ret_4h"] = returns_4h.loc[date, ticker]
            row["ret_12h"] = returns_12h.loc[date, ticker]
            row["ret_24h"] = returns_24h.loc[date, ticker]
            row["trend_9_21"] = trend_9_21.loc[date, ticker]
            row["rsi_14"] = rsi_14.loc[date, ticker]
            
            # Microstructure
            row["ofi_10"] = ofi_10.loc[date, ticker]
            
            if taker_ratio_4 is not None:
                row["taker_ratio_4"] = taker_ratio_4.loc[date, ticker]
                row["taker_ratio_12"] = taker_ratio_12.loc[date, ticker]
            
            # Volatility
            row["vol_24h"] = vol_24h.loc[date, ticker]
            row["liq_shock"] = liq_shock.loc[date, ticker]
            row["atr_exp"] = atr_exp.loc[date, ticker]
            
            # Market Concept
            row["btc_24h_ret"] = btc_trend_feature.loc[date, ticker]
            
            records.append(row)
            
    df = pd.DataFrame(records)
    
    # Drop columns that are completely NaN (e.g. features missing required data like taker_buy)
    df.dropna(axis=1, how='all', inplace=True)
    
    # Drop rows with NaN features (due to indicator warmup period)
    df = df.dropna()
    print(f"  ✅ Feature matrix built: {len(df):,} rows × {len(df.columns) - 2} features")
    
    return df

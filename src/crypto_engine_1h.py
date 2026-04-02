"""
crypto_engine_1h.py — 1-Hour resolution backtest engine for Crypto-Native strategy.

Handles:
  - Entering Long positions on ML buy signals
  - Tracking stop-loss and take-profit intra-day (intra-1H)
  - Simulating CoinDCX Futures fees (maker/taker, spread, slippage)
"""

import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════
# 1H ENGINE CONFIG
# ═══════════════════════════════════════════════
ENGINE_CONFIG_1H = {
    "TP_R": 2.0,                  # Take-profit in R (crypto 1H volatility)
    "SL_ATR_MULT": 2.0,           # Stop-loss buffer
    "ATR_PERIOD": 14,             # Standard ATR period for 1H
    "MAX_BARS_IN_TRADE": 48,      # Max holding period: 48 hours (2 days)
    "FEE_PCT": 0.00118,           # CoinDCX Futures Commission (0.118%)
    "SPREAD_PCT": 0.0010,         # Spread (0.10%)
    "SLIPPAGE_PCT": 0.0008,       # Slippage per side (0.08%)
    "MIN_SL_PCT": 0.005,          # Minimum SL distance (0.5%)
    "INITIAL_CAPITAL": 100000,
    "RISK_PCT": 1.0,              # Risk 1% per trade
}

def apply_crypto_costs(gross_r, entry_price, sl_price, cfg=ENGINE_CONFIG_1H):
    """Calculate net R after harsh crypto fees + slippage."""
    sl_distance_pct = abs(entry_price - sl_price) / entry_price
    if sl_distance_pct == 0:
        return {"netR": gross_r, "feeCostR": 0, "slippageCostR": 0}

    # Commission is charged on entry AND exit (2 ways)
    fee_cost_r = (cfg["FEE_PCT"] * 2) / sl_distance_pct
    
    # Spread + slippage happens on market entry and market stop/target
    slippage_cost_r = ((cfg["SPREAD_PCT"] + cfg["SLIPPAGE_PCT"]) * 2) / sl_distance_pct

    net_r = gross_r - fee_cost_r - slippage_cost_r
    return {
        "netR": round(net_r, 4),
        "feeCostR": round(fee_cost_r, 4),
        "slippageCostR": round(slippage_cost_r, 4),
    }

def compute_atr_1h(high, low, close, period=14):
    """Compute Average True Range for 1H candles."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

# ═══════════════════════════════════════════════
# BACKTEST SINGLE ASSET
# ═══════════════════════════════════════════════

def backtest_crypto_asset_1h(ticker, stock_df, ml_prob_series, regime_series=None):
    """
    Run 1H timeline backtest for a single crypto asset.
    """
    cfg = ENGINE_CONFIG_1H
    
    # Needs Open, High, Low, Close
    if "Close" not in stock_df.columns:
        return {"ticker": ticker, "trades": [], "metrics": None}
        
    atr = compute_atr_1h(stock_df["High"], stock_df["Low"], stock_df["Close"], cfg["ATR_PERIOD"])
    
    # Align dates
    common_dates = sorted(
        set(stock_df.index) & set(ml_prob_series.dropna().index) & set(atr.dropna().index)
    )
    
    if len(common_dates) < 50:
        return {"ticker": ticker, "trades": [], "metrics": None}
        
    trades = []
    in_trade = False
    
    # Trade state
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_idx = 0
    entry_date = None
    
    # Iterate through 1H bars
    for i, date in enumerate(common_dates):
        # We process the PREVIOUS bar's signal to enter AT THE OPEN of the CURRENT bar
        # to ensure no lookahead bias. ML prob is calculated at the end of the previous hour.
        
        if i == 0:
            continue
            
        prev_date = common_dates[i-1]
        
        # Prices for the CURRENT hour
        open_p = stock_df.loc[date, "Open"]
        high_p = stock_df.loc[date, "High"]
        low_p = stock_df.loc[date, "Low"]
        close_p = stock_df.loc[date, "Close"]
        
        # Signal from the PREVIOUS hour
        ml_prob = ml_prob_series.get(prev_date, np.nan)
        prev_atr = atr.get(prev_date, np.nan)
        regime = regime_series.get(prev_date, 1) if regime_series is not None else 1
        
        # 1. Manage Active Trade
        if in_trade:
            duration_hours = i - entry_idx
            risk_per_unit = abs(sl_price - entry_price)
            
            # SL hit
            if low_p <= sl_price:
                gross_r = (sl_price - entry_price) / risk_per_unit
                costs = apply_crypto_costs(gross_r, entry_price, sl_price)
                trades.append({
                    "ticker": ticker, "direction": "LONG", "entryDate": entry_date, "exitDate": date,
                    "entryPrice": entry_price, "exitPrice": sl_price, 
                    "R": costs["netR"], "exitType": "SL", "holding_hours": duration_hours
                })
                in_trade = False
                
            # TP hit
            elif high_p >= tp_price:
                gross_r = cfg["TP_R"]
                costs = apply_crypto_costs(gross_r, entry_price, sl_price)
                trades.append({
                    "ticker": ticker, "direction": "LONG", "entryDate": entry_date, "exitDate": date,
                    "entryPrice": entry_price, "exitPrice": tp_price,
                    "R": costs["netR"], "exitType": "TP", "holding_hours": duration_hours
                })
                in_trade = False
                
            # Time exit (48 hours)
            elif duration_hours >= cfg["MAX_BARS_IN_TRADE"]:
                gross_r = (close_p - entry_price) / risk_per_unit
                costs = apply_crypto_costs(gross_r, entry_price, sl_price)
                trades.append({
                    "ticker": ticker, "direction": "LONG", "entryDate": entry_date, "exitDate": date,
                    "entryPrice": entry_price, "exitPrice": close_p,
                    "R": costs["netR"], "exitType": "TIME", "holding_hours": duration_hours
                })
                in_trade = False
                
            continue # Can't open a new trade if we just closed one in the same hour
            
        # 2. Check for Entry
        if np.isnan(ml_prob) or np.isnan(prev_atr):
            continue
            
        # Stricter entry: Prob > 0.6 AND Regime is not Bear (0)
        if ml_prob > 0.60 and regime > 0:
            entry_price = open_p  # Enter at open of current candle
            sl_dist = prev_atr * cfg["SL_ATR_MULT"]
            
            # Distance must be at least minimum %
            if sl_dist / entry_price < cfg["MIN_SL_PCT"]:
                sl_dist = entry_price * cfg["MIN_SL_PCT"]
                
            sl_price = entry_price - sl_dist
            tp_price = entry_price + (sl_dist * cfg["TP_R"])
            
            entry_date = date
            entry_idx = i
            in_trade = True
            
    # Calculate simple metrics
    if not trades:
        return {"ticker": ticker, "trades": [], "metrics": None}
        
    r_values = [t["R"] for t in trades]
    won = len([r for r in r_values if r > 0])
    
    metrics = {
        "trades": len(trades),
        "winRate": f"{(won / len(trades) * 100):.1f}%",
        "expectancy": f"{np.mean(r_values):.2f}R",
        "totalR": f"{sum(r_values):.2f}R",
        "avgHrs": f"{np.mean([t['holding_hours'] for t in trades]):.1f}"
    }
    
    return {"ticker": ticker, "trades": trades, "metrics": metrics}

def run_crypto_1h_engine(panels, predictions_df, btc_panels=None):
    """
    Run the 1H backtest across the entire crypto universe based
    on the ML purged walk-forward predictions.
    """
    print(f"\n⚙️ STEP 7: Running 1H Engine Backtest...")
    
    # We use BTC 200 EMA to mask out bear markets from entries
    regime_series = None
    if btc_panels is not None:
        btc_close = btc_panels["Close"].squeeze()
        ema_200 = btc_close.ewm(span=200, adjust=False).mean()
        # 1 = Bull (BTC > EMA200), 0 = Bear (BTC < EMA200)
        regime_series = (btc_close > ema_200).astype(int)
        
    tickers = predictions_df["ticker"].unique()
    all_trades = []
    
    # Pivot predictions so we can easily `.get(date)` per ticker
    prob_piv = predictions_df.pivot_table(index="date", columns="ticker", values="prob")
    
    for ticker in tickers:
        stock_df = pd.DataFrame({
            "Open": panels["Open"].get(ticker),
            "High": panels["High"].get(ticker),
            "Low": panels["Low"].get(ticker),
            "Close": panels["Close"].get(ticker)
        })
        
        if ticker not in prob_piv.columns:
            continue
            
        prob_series = prob_piv[ticker]
        
        result = backtest_crypto_asset_1h(ticker, stock_df, prob_series, regime_series)
        
        if result["trades"]:
            all_trades.extend(result["trades"])
            print(f"  {ticker:>8s}: {result['metrics']['trades']:>4d} trades | "
                  f"WinRate: {result['metrics']['winRate']:>6s} | "
                  f"Exp: {result['metrics']['expectancy']:>6s}")
                  
    if not all_trades:
        print("  ❌ No trades taken in backtest.")
        return pd.DataFrame()
        
    trades_df = pd.DataFrame(all_trades)
    
    # Total Summary
    tot = len(trades_df)
    won = len(trades_df[trades_df["R"] > 0])
    avg_r = trades_df["R"].mean()
    sum_r = trades_df["R"].sum()
    
    print(f"\n  {'═'*50}")
    print(f"  📈 1H ENGINE RESULTS (ALL ASSETS)")
    print(f"  {'═'*50}")
    print(f"  Total Trades: {tot}")
    print(f"  Win Rate:     {(won/tot*100):.2f}%")
    print(f"  Expectancy:   {avg_r:.3f}R")
    print(f"  Net R Profit: {sum_r:.2f}R")
    print(f"  {'═'*50}")
    
    return trades_df

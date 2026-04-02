"""
fx_engine.py — Daily resolution backtest engine for Forex Macro Strategy.

Handles:
  - Entering Long/Short positions on ML probability signals
  - Tracking 1.5R stop-loss and 2.0R take-profit
  - Simulating Retail Spread (e.g., 1.5 pips)
"""

import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fx_config import ENGINE_CONFIG_FX

def compute_atr_daily(high, low, close, period=14):
    """Compute Average True Range for Daily candles."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

def apply_fx_spread(gross_r, entry_price, sl_price, ticker, cfg=ENGINE_CONFIG_FX):
    """
    Apply a simplified retail pip spread cost to the R-multiple return.
    Most majors trade at 4 decimal places (pip = 0.0001), JPY pairs at 2 (pip = 0.01).
    """
    pip_size = 0.01 if "JPY" in ticker else 0.0001
    spread_cost_price = cfg["SPREAD_PIPS"] * pip_size
    
    sl_distance = abs(entry_price - sl_price)
    
    if sl_distance == 0:
        return {"netR": gross_r, "spreadCostR": 0.0}

    # Slippage hits us crossing the bid/ask spread twice (entry & exit)
    total_spread_cost = spread_cost_price * 2
    spread_cost_r = total_spread_cost / sl_distance

    net_r = gross_r - spread_cost_r
    return {
        "netR": round(net_r, 4),
        "spreadCostR": round(spread_cost_r, 4),
    }

def backtest_fx_asset(ticker, stock_df, ml_prob_series):
    """
    Run Daily timeline backtest for a single FX pair.
    """
    cfg = ENGINE_CONFIG_FX
    
    if "Close" not in stock_df.columns:
        return {"ticker": ticker, "trades": [], "metrics": None}
        
    atr = compute_atr_daily(stock_df["High"], stock_df["Low"], stock_df["Close"], cfg["ATR_PERIOD"])
    
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
    
    for i, date in enumerate(common_dates):
        if i == 0: continue
        
        prev_date = common_dates[i-1]
        
        open_p = stock_df.loc[date, "Open"]
        high_p = stock_df.loc[date, "High"]
        low_p = stock_df.loc[date, "Low"]
        close_p = stock_df.loc[date, "Close"]
        
        ml_prob = ml_prob_series.get(prev_date, np.nan)
        prev_atr = atr.get(prev_date, np.nan)
        
        # 1. Manage Active Trade
        if in_trade:
            duration_days = i - entry_idx
            risk_per_unit = abs(sl_price - entry_price)
            
            # SL hit
            if low_p <= sl_price:
                gross_r = (sl_price - entry_price) / risk_per_unit
                costs = apply_fx_spread(gross_r, entry_price, sl_price, ticker)
                trades.append({
                    "ticker": ticker, "direction": "LONG", "entryDate": entry_date, "exitDate": date,
                    "entryPrice": entry_price, "exitPrice": sl_price, 
                    "R": costs["netR"], "exitType": "SL", "duration_days": duration_days
                })
                in_trade = False
                
            # TP hit
            elif high_p >= tp_price:
                gross_r = cfg["TP_R"]
                costs = apply_fx_spread(gross_r, entry_price, sl_price, ticker)
                trades.append({
                    "ticker": ticker, "direction": "LONG", "entryDate": entry_date, "exitDate": date,
                    "entryPrice": entry_price, "exitPrice": tp_price,
                    "R": costs["netR"], "exitType": "TP", "duration_days": duration_days
                })
                in_trade = False
                
            # Time exit 
            elif duration_days >= cfg["MAX_BARS_IN_TRADE"]:
                gross_r = (close_p - entry_price) / risk_per_unit
                costs = apply_fx_spread(gross_r, entry_price, sl_price, ticker)
                trades.append({
                    "ticker": ticker, "direction": "LONG", "entryDate": entry_date, "exitDate": date,
                    "entryPrice": entry_price, "exitPrice": close_p,
                    "R": costs["netR"], "exitType": "TIME", "duration_days": duration_days
                })
                in_trade = False
                
            continue 
            
        # 2. Check for Entry
        if np.isnan(ml_prob) or np.isnan(prev_atr):
            continue
            
        # Very Strict macro entry threshold (e.g. only highest conviction signals)
        if ml_prob > 0.60:
            entry_price = open_p  
            sl_dist = prev_atr * cfg["SL_ATR_MULT"]
            
            sl_price = entry_price - sl_dist
            tp_price = entry_price + (sl_dist * cfg["TP_R"])
            
            entry_date = date
            entry_idx = i
            in_trade = True
            
    if not trades:
        return {"ticker": ticker, "trades": [], "metrics": None}
        
    r_values = [t["R"] for t in trades]
    won = len([r for r in r_values if r > 0])
    
    metrics = {
        "trades": len(trades),
        "winRate": f"{(won / len(trades) * 100):.1f}%",
        "expectancy": f"{np.mean(r_values):.2f}R",
        "totalR": f"{sum(r_values):.2f}R",
    }
    
    return {"ticker": ticker, "trades": trades, "metrics": metrics}

def run_fx_engine(panels, predictions_df):
    """Run Daily backtest across Forex universe."""
    print(f"\n⚙️ Running FX Engine Backtest...")
    
    tickers = predictions_df["ticker"].unique()
    all_trades = []
    
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
        result = backtest_fx_asset(ticker, stock_df, prob_series)
        
        if result["trades"]:
            all_trades.extend(result["trades"])
            print(f"  {ticker:>8s}: {result['metrics']['trades']:>4d} trades | "
                  f"WinRate: {result['metrics']['winRate']:>6s} | "
                  f"Exp: {result['metrics']['expectancy']:>6s}")
                  
    if not all_trades:
        print("  ❌ No trades taken in backtest.")
        return pd.DataFrame()
        
    trades_df = pd.DataFrame(all_trades)
    
    tot = len(trades_df)
    won = len(trades_df[trades_df["R"] > 0])
    avg_r = trades_df["R"].mean()
    sum_r = trades_df["R"].sum()
    
    print(f"\n  {'═'*50}")
    print(f"  📈 FX MACRO ENGINE RESULTS (ALL PAIRS)")
    print(f"  {'═'*50}")
    print(f"  Total Trades: {tot}")
    print(f"  Win Rate:     {(won/tot*100):.2f}%")
    print(f"  Expectancy:   {avg_r:.3f}R")
    print(f"  Net R Profit: {sum_r:.2f}R")
    print(f"  {'═'*50}")
    
    return trades_df

"""
engine.py — Per-stock alpha backtest engine.

Modeled after the existing engine.js pattern. Handles:
  - Per-stock entry/exit management based on alpha signals
  - SL/TP calculation using ATR
  - Position R-tracking (MFE, MAE)
  - Cost calculation (commission, slippage, spread)
  - Time-based exit
  - US market hours filter (9:30 AM - 4:00 PM ET)
  - Diagnostics tracking

This engine backtests a SINGLE stock at a time. The main pipeline
(run_backtest.py) calls it for each stock in the universe.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from config import (
    INITIAL_CAPITAL, RISK_PER_TRADE_PCT, COMMISSION_PER_SHARE,
    SLIPPAGE_BPS, BULL, NEUTRAL, BEAR, CRISIS
)


# ─────────────────────────────────────────────
# ENGINE CONFIG (per-trade level)
# ─────────────────────────────────────────────
ENGINE_CONFIG = {
    "TP_R": 3.0,                 # Take-profit in R (3:1 reward-to-risk)
    "SL_ATR_MULT": 3.5,          # Stop-loss = ATR × this multiplier (wider = room to breathe)
    "ATR_PERIOD": 14,            # Period for ATR calculation
    "MIN_SL_PCT": 0.005,         # Minimum SL distance (0.5%) to avoid noise trades
    "MAX_BARS_IN_TRADE": 90,     # Max holding period in trading days (~4.5 months)
    "FEE_PCT": 0.001,            # Commission per side (0.1% for broker)
    "SLIPPAGE_PCT": 0.0005,      # Slippage per side (5 bps)
    "WARM_UP_BARS": 250,         # Indicator warm-up period (250 days ~ 1 year)
    "SIGNAL_THRESHOLD": 0.75,    # Alpha composite rank threshold to enter (top 25%)
    "EXIT_THRESHOLD": 0.35,      # Exit if rank drops below this
}


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def compute_atr(high, low, close, period=14):
    """
    Compute Average True Range (ATR) for a stock.

    Args:
        high, low, close: pd.Series (price data)
        period: lookback period

    Returns: pd.Series of ATR values
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def apply_costs(gross_r, entry_price, sl_price):
    """
    Apply realistic trading costs to a gross R-multiple.
    Mirrors the applyCosts function from engine.js.

    Returns: dict with netR and cost breakdown
    """
    sl_distance_pct = abs(entry_price - sl_price) / entry_price
    if sl_distance_pct == 0:
        return {"netR": gross_r, "feeCostR": 0, "slippageCostR": 0}

    fee_cost_r = (ENGINE_CONFIG["FEE_PCT"] * 2) / sl_distance_pct
    slippage_cost_r = (ENGINE_CONFIG["SLIPPAGE_PCT"] * 2) / sl_distance_pct

    net_r = gross_r - fee_cost_r - slippage_cost_r
    return {
        "netR": round(net_r, 4),
        "feeCostR": round(fee_cost_r, 4),
        "slippageCostR": round(slippage_cost_r, 4),
    }


def is_us_market_hours(timestamp):
    """
    Check if a timestamp falls within US equity market hours.
    NYSE/NASDAQ: 9:30 AM - 4:00 PM Eastern Time.

    For daily data, we only check if it's a weekday (Mon-Fri).
    If the data were intraday, this would check actual hours.

    Returns: True if it's a valid US trading day
    """
    if isinstance(timestamp, pd.Timestamp):
        # Weekday check: Monday=0 ... Friday=4
        return timestamp.weekday() < 5
    return True


def init_diagnostics():
    """Initialize diagnostic counters (mirrors initEntryDiagnostics)."""
    return {
        "totalBars": 0,
        "entries": 0,
        "regimeBlocked": 0,
        "signalBlocked": 0,
        "warmupBlocked": 0,
        "marketHoursBlocked": 0,
        "tradesCompleted": 0,
        "winsR": 0.0,
        "lossesR": 0.0,
    }


# ═══════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════

def backtest_stock(ticker, stock_df, composite_scores_series, regime,
                   opts=None):
    """
    Backtest a SINGLE stock using alpha signals, with entry/exit management.

    This is the Python equivalent of engine.js → backtestPair().

    Args:
        ticker: str, stock symbol (e.g. "AAPL")
        stock_df: DataFrame with OHLCV + indicators for this stock
        composite_scores_series: pd.Series of composite alpha scores for this
                                  stock (indexed by date)
        regime: pd.Series of regime labels (indexed by date)
        opts: optional dict to override ENGINE_CONFIG defaults

    Returns: dict with {ticker, trades, metrics, diagnostics}
    """
    cfg = {**ENGINE_CONFIG, **(opts or {})}
    diag = init_diagnostics()

    # Ensure we have required columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in stock_df.columns:
            return {"ticker": ticker, "trades": [], "metrics": None,
                    "diagnostics": diag}

    # Compute ATR
    atr = compute_atr(stock_df["High"], stock_df["Low"],
                      stock_df["Close"], cfg["ATR_PERIOD"])

    # Align all data to common dates
    common_dates = sorted(
        set(stock_df.index) &
        set(composite_scores_series.index) &
        set(regime.index) &
        set(atr.dropna().index)
    )

    if len(common_dates) < cfg["WARM_UP_BARS"] + 50:
        return {"ticker": ticker, "trades": [], "metrics": None,
                "diagnostics": diag}

    # ─── Trade State ───
    trades = []
    in_trade = False
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_idx = 0
    entry_date = None
    max_favorable_r = 0.0
    max_adverse_r = 0.0
    trade_regime = NEUTRAL

    one_r_dollars = INITIAL_CAPITAL * RISK_PER_TRADE_PCT / 100

    # ─── Bar Loop ───
    for i, date in enumerate(common_dates):
        diag["totalBars"] += 1

        # Warm-up period — skip
        if i < cfg["WARM_UP_BARS"]:
            diag["warmupBlocked"] += 1
            continue

        # US market hours filter
        if not is_us_market_hours(date):
            diag["marketHoursBlocked"] += 1
            continue

        close = stock_df.loc[date, "Close"]
        high = stock_df.loc[date, "High"]
        low = stock_df.loc[date, "Low"]
        current_atr = atr.loc[date] if date in atr.index else np.nan
        score = composite_scores_series.get(date, np.nan)
        day_regime = regime.get(date, NEUTRAL)

        if np.isnan(close) or np.isnan(current_atr):
            continue

        # ═══════════════════════════════
        # MANAGE OPEN TRADE
        # ═══════════════════════════════
        if in_trade:
            duration_days = i - entry_idx
            risk_per_unit = abs(sl_price - entry_price)

            if risk_per_unit == 0:
                in_trade = False
                continue

            # ─── REGIME EXIT: force-close if regime turned BEAR or CRISIS ───
            if day_regime in (BEAR, CRISIS):
                gross_r = (close - entry_price) / risk_per_unit
                costs = apply_costs(gross_r, entry_price, sl_price)

                trades.append({
                    "ticker": ticker,
                    "direction": "LONG",
                    "entryDate": entry_date,
                    "exitDate": date,
                    "entryPrice": round(entry_price, 2),
                    "exitPrice": round(close, 2),
                    "slPrice": round(sl_price, 2),
                    "tpPrice": round(tp_price, 2),
                    "R": costs["netR"],
                    "grossR": round(gross_r, 4),
                    **costs,
                    "holding_days": duration_days,
                    "MaxFavorableR": round(max_favorable_r, 2),
                    "MaxAdverseR": round(max_adverse_r, 2),
                    "regime": trade_regime,
                    "exitType": "REGIME_EXIT",
                })
                diag["tradesCompleted"] += 1
                diag["regimeExits"] = diag.get("regimeExits", 0) + 1
                if costs["netR"] > 0:
                    diag["winsR"] += costs["netR"]
                else:
                    diag["lossesR"] += costs["netR"]
                in_trade = False
                continue

            # R calculation (long direction only in our strategy)
            favorable_r = (high - entry_price) / risk_per_unit
            adverse_r = (entry_price - low) / risk_per_unit

            max_favorable_r = max(max_favorable_r, favorable_r)
            max_adverse_r = max(max_adverse_r, adverse_r)

            # ─── STOP LOSS CHECK ───
            stopped_out = low <= sl_price
            # ─── TAKE PROFIT CHECK ───
            take_profit = high >= tp_price

            if stopped_out:
                gross_r = (sl_price - entry_price) / risk_per_unit
                costs = apply_costs(gross_r, entry_price, sl_price)

                trades.append({
                    "ticker": ticker,
                    "direction": "LONG",
                    "entryDate": entry_date,
                    "exitDate": date,
                    "entryPrice": round(entry_price, 2),
                    "exitPrice": round(sl_price, 2),
                    "slPrice": round(sl_price, 2),
                    "tpPrice": round(tp_price, 2),
                    "R": costs["netR"],
                    "grossR": round(gross_r, 4),
                    **costs,
                    "holding_days": duration_days,
                    "MaxFavorableR": round(max_favorable_r, 2),
                    "MaxAdverseR": round(max_adverse_r, 2),
                    "regime": trade_regime,
                    "exitType": "SL",
                })
                diag["tradesCompleted"] += 1
                diag["lossesR"] += costs["netR"]
                in_trade = False

            elif take_profit:
                gross_r = cfg["TP_R"]
                costs = apply_costs(gross_r, entry_price, sl_price)

                trades.append({
                    "ticker": ticker,
                    "direction": "LONG",
                    "entryDate": entry_date,
                    "exitDate": date,
                    "entryPrice": round(entry_price, 2),
                    "exitPrice": round(tp_price, 2),
                    "slPrice": round(sl_price, 2),
                    "tpPrice": round(tp_price, 2),
                    "R": costs["netR"],
                    "grossR": round(gross_r, 4),
                    **costs,
                    "holding_days": duration_days,
                    "MaxFavorableR": round(max_favorable_r, 2),
                    "MaxAdverseR": round(max_adverse_r, 2),
                    "regime": trade_regime,
                    "exitType": "TP",
                })
                diag["tradesCompleted"] += 1
                diag["winsR"] += costs["netR"]
                in_trade = False

            # ─── TIME-BASED EXIT ───
            elif duration_days > cfg["MAX_BARS_IN_TRADE"]:
                gross_r = (close - entry_price) / risk_per_unit
                costs = apply_costs(gross_r, entry_price, sl_price)

                trades.append({
                    "ticker": ticker,
                    "direction": "LONG",
                    "entryDate": entry_date,
                    "exitDate": date,
                    "entryPrice": round(entry_price, 2),
                    "exitPrice": round(close, 2),
                    "slPrice": round(sl_price, 2),
                    "tpPrice": round(tp_price, 2),
                    "R": costs["netR"],
                    "grossR": round(gross_r, 4),
                    **costs,
                    "holding_days": duration_days,
                    "MaxFavorableR": round(max_favorable_r, 2),
                    "MaxAdverseR": round(max_adverse_r, 2),
                    "regime": trade_regime,
                    "exitType": "TIME",
                })
                diag["tradesCompleted"] += 1
                if costs["netR"] > 0:
                    diag["winsR"] += costs["netR"]
                else:
                    diag["lossesR"] += costs["netR"]
                in_trade = False

            # ─── SIGNAL EXIT: alpha rank dropped ───
            elif not np.isnan(score) and score < cfg["EXIT_THRESHOLD"]:
                gross_r = (close - entry_price) / risk_per_unit
                costs = apply_costs(gross_r, entry_price, sl_price)

                trades.append({
                    "ticker": ticker,
                    "direction": "LONG",
                    "entryDate": entry_date,
                    "exitDate": date,
                    "entryPrice": round(entry_price, 2),
                    "exitPrice": round(close, 2),
                    "slPrice": round(sl_price, 2),
                    "tpPrice": round(tp_price, 2),
                    "R": costs["netR"],
                    "grossR": round(gross_r, 4),
                    **costs,
                    "holding_days": duration_days,
                    "MaxFavorableR": round(max_favorable_r, 2),
                    "MaxAdverseR": round(max_adverse_r, 2),
                    "regime": trade_regime,
                    "exitType": "SIGNAL",
                })
                diag["tradesCompleted"] += 1
                if costs["netR"] > 0:
                    diag["winsR"] += costs["netR"]
                else:
                    diag["lossesR"] += costs["netR"]
                in_trade = False

            continue  # Don't look for entry while in a trade

        # ═══════════════════════════════
        # ENTRY LOGIC
        # ═══════════════════════════════

        # Regime filter: skip BEAR and CRISIS entries
        if day_regime in (BEAR, CRISIS):
            diag["regimeBlocked"] += 1
            continue

        # Signal filter: only enter if composite score rank > threshold
        if np.isnan(score) or score < cfg["SIGNAL_THRESHOLD"]:
            diag["signalBlocked"] += 1
            continue

        # ✅ ENTER: buy at close
        entry_price = close
        entry_date = date
        entry_idx = i
        trade_regime = day_regime

        # SL = below entry by ATR × multiplier
        sl_distance = current_atr * cfg["SL_ATR_MULT"]
        sl_price = entry_price - sl_distance

        # Sanity check: SL too tight
        if sl_distance / entry_price < cfg["MIN_SL_PCT"]:
            diag["signalBlocked"] += 1
            continue

        # TP = above entry by TP_R × SL distance
        tp_price = entry_price + sl_distance * cfg["TP_R"]

        max_favorable_r = 0.0
        max_adverse_r = 0.0
        in_trade = True
        diag["entries"] += 1

    # ─── Close any trade still open at end ───
    if in_trade:
        last_date = common_dates[-1]
        last_close = stock_df.loc[last_date, "Close"]
        risk_per_unit = abs(sl_price - entry_price)
        if risk_per_unit > 0:
            gross_r = (last_close - entry_price) / risk_per_unit
            costs = apply_costs(gross_r, entry_price, sl_price)
            trades.append({
                "ticker": ticker,
                "direction": "LONG",
                "entryDate": entry_date,
                "exitDate": last_date,
                "entryPrice": round(entry_price, 2),
                "exitPrice": round(last_close, 2),
                "slPrice": round(sl_price, 2),
                "tpPrice": round(tp_price, 2),
                "R": costs["netR"],
                "grossR": round(gross_r, 4),
                **costs,
                "holding_days": len(common_dates) - 1 - entry_idx,
                "MaxFavorableR": round(max_favorable_r, 2),
                "MaxAdverseR": round(max_adverse_r, 2),
                "regime": trade_regime,
                "exitType": "EOD",
            })
            diag["tradesCompleted"] += 1

    # ─── Compute per-stock metrics ───
    metrics = compute_stock_metrics(trades) if trades else None

    return {
        "ticker": ticker,
        "trades": trades,
        "metrics": metrics,
        "diagnostics": diag,
    }


# ═══════════════════════════════════════════════
# PER-STOCK METRICS
# ═══════════════════════════════════════════════

def compute_stock_metrics(trades):
    """
    Compute metrics for a single stock's trades.
    Mirrors computeMetrics from metrics.js.
    """
    if not trades:
        return None

    r_values = [t["R"] for t in trades]
    n = len(r_values)
    won = [r for r in r_values if r > 0]
    lost = [r for r in r_values if r <= 0]

    win_rate = len(won) / n * 100 if n > 0 else 0
    expectancy = np.mean(r_values) if n > 0 else 0
    total_profit = sum(won)
    total_loss = abs(sum(lost))
    net_profit = sum(r_values)

    # Max drawdown in R
    cumsum = np.cumsum(r_values)
    running_max = np.maximum.accumulate(cumsum)
    dd = cumsum - running_max
    max_dd_r = abs(dd.min()) if len(dd) > 0 else 0

    # Average holding time
    avg_time = np.mean([t["holding_days"] for t in trades]) if trades else 0

    return {
        "trades": n,
        "winRate": f"{win_rate:.2f}",
        "expectancy": f"{expectancy:.2f}",
        "maxDrawdownR": f"{max_dd_r:.2f}",
        "avgTimeInTradeBars": f"{avg_time:.1f}",
        "wonTrades": len(won),
        "lostTrades": len(lost),
        "totalProfit": f"{total_profit:.2f}",
        "totalLoss": f"{total_loss:.2f}",
        "netProfit": f"{net_profit:.2f}",
    }


# ═══════════════════════════════════════════════
# BATCH RUNNER (for all stocks)
# ═══════════════════════════════════════════════

def run_engine_all_stocks(panels, composite_scores, regime, opts=None):
    """
    Run the per-stock engine across the entire universe.

    Args:
        panels: dict of {field: DataFrame(dates×tickers)}
        composite_scores: DataFrame (dates × tickers) of composite alpha scores
        regime: pd.Series of regime labels

    Returns:
        all_trades: pd.DataFrame of every trade across all stocks
        stock_results: list of per-stock dicts {ticker, trades, metrics, diagnostics}
    """
    import json

    tickers = composite_scores.columns.tolist()
    stock_results = []
    all_trades = []

    print(f"\n🔧 Running engine on {len(tickers)} stocks...")

    for ticker in tickers:
        # Build per-stock DataFrame
        stock_df = pd.DataFrame({
            "Open": panels["Open"].get(ticker),
            "High": panels["High"].get(ticker),
            "Low": panels["Low"].get(ticker),
            "Close": panels["Close"].get(ticker),
            "Volume": panels["Volume"].get(ticker),
        })
        stock_df = stock_df.dropna()

        if len(stock_df) < 300:
            continue

        # Get composite score series for this stock
        scores = composite_scores[ticker] if ticker in composite_scores.columns else pd.Series()

        # Cross-sectional rank the scores (so threshold is meaningful)
        score_rank = composite_scores.rank(axis=1, pct=True)[ticker] \
            if ticker in composite_scores.columns else pd.Series()

        result = backtest_stock(ticker, stock_df, score_rank, regime, opts)
        stock_results.append(result)

        if result["trades"]:
            all_trades.extend(result["trades"])

        n_trades = len(result["trades"])
        if n_trades > 0:
            exp = result["metrics"]["expectancy"] if result["metrics"] else "N/A"
            print(f"  {ticker:>6s}: {n_trades:>4d} trades, Exp: {exp}R")

    # ─── Aggregate summary ───
    all_trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    if len(all_trades_df) > 0:
        total = len(all_trades_df)
        won = len(all_trades_df[all_trades_df["R"] > 0])
        net_r = all_trades_df["R"].sum()
        print(f"\n{'═' * 55}")
        print(f"  🔧 ENGINE AGGREGATE (all stocks)")
        print(f"{'═' * 55}")
        print(f"  Stocks traded:   {len([r for r in stock_results if r['trades']])}")
        print(f"  Total trades:    {total}")
        print(f"  Won / Lost:      {won} / {total - won}")
        print(f"  Win Rate:        {won/total*100:.2f}%")
        print(f"  Expectancy:      {all_trades_df['R'].mean():.2f}R")
        print(f"  Net Profit:      {net_r:.2f}R")
        print(f"{'═' * 55}")

        # Print JSON summary
        agg_metrics = compute_stock_metrics(all_trades)
        print(f"\n📋 Aggregate JSON:")
        print(json.dumps(agg_metrics, indent=2))

    return all_trades_df, stock_results

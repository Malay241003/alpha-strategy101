"""
backtester.py — Portfolio simulation engine.

Simulates daily PnL from the holdings matrix, accounting for
transaction costs and slippage. Outputs equity curve + trade log.
"""

import pandas as pd
import numpy as np
from config import (
    INITIAL_CAPITAL, COMMISSION_PER_SHARE, SLIPPAGE_BPS,
    REBALANCE_EVERY_N_DAYS, IN_SAMPLE_END, OUT_SAMPLE_START,
    RISK_PER_TRADE_PCT, BULL, NEUTRAL, BEAR, CRISIS
)


def run_backtest(holdings, panels, regime):
    """
    Simulate daily PnL from portfolio holdings.

    Args:
        holdings: DataFrame (dates × tickers) of portfolio weights (0 to 1)
        panels: dict of panel DataFrames (needs 'Close', 'returns')
        regime: pd.Series of regime labels

    Returns:
        dict with:
          - equity_curve: pd.Series of daily portfolio value
          - daily_returns: pd.Series of daily portfolio returns
          - trade_log: pd.DataFrame of all trades
          - turnover: pd.Series of daily turnover
          - regime: pd.Series (aligned to equity curve)
    """
    close = panels["Close"]
    returns = panels["returns"]

    # Align dates
    common_dates = sorted(
        set(holdings.index) & set(returns.index) & set(regime.index)
    )
    holdings = holdings.loc[common_dates]
    returns = returns.loc[common_dates]
    regime = regime.loc[common_dates]

    # ─── Apply rebalance frequency ───
    # Only rebalance every N days; hold positions constant in between
    rebal_holdings = holdings.copy()
    last_rebal = None
    rebal_count = 0
    regime_rank = {BULL: 3, NEUTRAL: 2, BEAR: 1, CRISIS: 0}
    for i, date in enumerate(common_dates):
        if i == 0:
            last_rebal = holdings.loc[date]
            continue

        # Detect regime downgrade (e.g. BULL→BEAR, NEUTRAL→CRISIS)
        prev_rank = regime_rank.get(regime.get(common_dates[i-1], NEUTRAL), 2)
        curr_rank = regime_rank.get(regime.get(date, NEUTRAL), 2)
        regime_downgraded = curr_rank < prev_rank

        if i % REBALANCE_EVERY_N_DAYS == 0 or regime_downgraded:
            last_rebal = holdings.loc[date]
            rebal_count += 1
        else:
            rebal_holdings.loc[date] = last_rebal
    holdings = rebal_holdings

    # ─── Compute daily portfolio return ───
    # weighted return = sum(weight_i * return_i) for each day
    # We use the PREVIOUS day's holdings to compute today's return (delay-1)
    prev_holdings = holdings.shift(1).fillna(0)
    port_returns = (prev_holdings * returns.fillna(0)).sum(axis=1)

    # ─── Compute turnover (for transaction cost) ───
    turnover = (holdings - prev_holdings).abs().sum(axis=1)

    # ─── Transaction costs ───
    # Slippage: proportional to turnover
    slippage_cost = turnover * (SLIPPAGE_BPS / 10000)
    # Commission: approximate as fraction of turnover × avg price
    commission_cost = turnover * COMMISSION_PER_SHARE * 0.01  # rough approximation

    # Net return
    net_returns = port_returns - slippage_cost - commission_cost

    # ─── Build equity curve ───
    equity = pd.Series(0.0, index=common_dates, name="equity")
    equity.iloc[0] = INITIAL_CAPITAL
    for i in range(1, len(common_dates)):
        equity.iloc[i] = equity.iloc[i - 1] * (1 + net_returns.iloc[i])

    # ─── Build R-multiple trade log ───
    # A "trade" = stock enters the portfolio → held for N days → exits
    # R = PnL / (1R), where 1R = RISK_PER_TRADE_PCT% of portfolio at entry
    one_r_dollars = INITIAL_CAPITAL * RISK_PER_TRADE_PCT / 100

    active_trades = {}  # ticker → {entry_date, entry_idx, entry_equity}
    completed_trades = []

    for i, date in enumerate(common_dates):
        for ticker in holdings.columns:
            was_held = prev_holdings.loc[date, ticker] > 0.001 if date in prev_holdings.index else False
            is_held = holdings.loc[date, ticker] > 0.001

            if not was_held and is_held:
                # === ENTRY ===
                active_trades[ticker] = {
                    "entry_date": date,
                    "entry_idx": i,
                    "entry_equity": equity.iloc[i],
                    "weight": holdings.loc[date, ticker],
                    "regime_at_entry": regime.get(date, "NEUTRAL"),
                    "max_favorable": 0.0,
                    "max_adverse": 0.0,
                }

            elif was_held and not is_held and ticker in active_trades:
                # === EXIT ===
                trade = active_trades.pop(ticker)
                entry_weight = trade["weight"]
                # PnL for this position = sum of (weight × daily return) over holding period
                holding_days = i - trade["entry_idx"]
                trade_pnl_pct = 0.0
                for j in range(trade["entry_idx"] + 1, i + 1):
                    d = common_dates[j]
                    ret = returns.loc[d, ticker] if d in returns.index and ticker in returns.columns else 0
                    if np.isfinite(ret):
                        daily_pnl = entry_weight * ret
                        trade_pnl_pct += daily_pnl
                        # Track MFE/MAE
                        running_sum = sum(
                            entry_weight * (returns.loc[common_dates[k], ticker]
                                            if common_dates[k] in returns.index
                                            and ticker in returns.columns
                                            and np.isfinite(returns.loc[common_dates[k], ticker])
                                            else 0)
                            for k in range(trade["entry_idx"] + 1, j + 1)
                        )
                        trade["max_favorable"] = max(trade["max_favorable"], running_sum)
                        trade["max_adverse"] = min(trade["max_adverse"], running_sum)

                # Convert PnL to R-multiple
                trade_pnl_dollars = trade_pnl_pct * INITIAL_CAPITAL
                r_multiple = trade_pnl_dollars / one_r_dollars if one_r_dollars > 0 else 0

                completed_trades.append({
                    "ticker": ticker,
                    "entry_date": trade["entry_date"],
                    "exit_date": date,
                    "holding_days": holding_days,
                    "pnl_pct": trade_pnl_pct,
                    "pnl_dollars": trade_pnl_dollars,
                    "R": round(r_multiple, 2),
                    "MaxFavorableR": round(trade["max_favorable"] * INITIAL_CAPITAL / one_r_dollars, 2),
                    "MaxAdverseR": round(trade["max_adverse"] * INITIAL_CAPITAL / one_r_dollars, 2),
                    "regime": trade["regime_at_entry"],
                })

            elif was_held and is_held and ticker in active_trades:
                # === STILL HOLDING — update MFE/MAE ===
                pass  # MFE/MAE tracked at exit for efficiency

    # Close any trades still open at the end
    for ticker, trade in active_trades.items():
        holding_days = len(common_dates) - 1 - trade["entry_idx"]
        trade_pnl_pct = 0.0
        for j in range(trade["entry_idx"] + 1, len(common_dates)):
            d = common_dates[j]
            ret = returns.loc[d, ticker] if d in returns.index and ticker in returns.columns else 0
            if np.isfinite(ret):
                trade_pnl_pct += trade["weight"] * ret

        trade_pnl_dollars = trade_pnl_pct * INITIAL_CAPITAL
        r_multiple = trade_pnl_dollars / one_r_dollars if one_r_dollars > 0 else 0
        completed_trades.append({
            "ticker": ticker,
            "entry_date": trade["entry_date"],
            "exit_date": common_dates[-1],
            "holding_days": holding_days,
            "pnl_pct": trade_pnl_pct,
            "pnl_dollars": trade_pnl_dollars,
            "R": round(r_multiple, 2),
            "MaxFavorableR": 0.0,
            "MaxAdverseR": 0.0,
            "regime": trade["regime_at_entry"],
        })

    trade_log = pd.DataFrame(completed_trades) if completed_trades else pd.DataFrame()

    # ─── Summary ───
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_years = len(common_dates) / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    avg_turnover = turnover.mean()

    print(f"\n📈 Backtest Results ({common_dates[0].strftime('%Y-%m-%d')} → "
          f"{common_dates[-1].strftime('%Y-%m-%d')}):")
    print(f"   Total Return: {total_return:.1%}")
    print(f"   CAGR: {cagr:.1%}")
    print(f"   Avg Daily Turnover: {avg_turnover:.2%}")
    print(f"   Completed Trades: {len(completed_trades)}")
    print(f"   Rebalances: {rebal_count}")

    return {
        "equity_curve": equity,
        "daily_returns": net_returns,
        "trade_log": trade_log,
        "turnover": turnover,
        "regime": regime,
        "holdings": holdings,
    }


def split_in_out_sample(results):
    """
    Split backtest results into in-sample (2008-2019) and out-of-sample (2020-2025).

    Returns: (in_sample_equity, out_sample_equity) tuple of Series
    """
    equity = results["equity_curve"]

    in_sample = equity[equity.index <= IN_SAMPLE_END]
    out_sample = equity[equity.index >= OUT_SAMPLE_START]

    if len(in_sample) > 0 and len(out_sample) > 0:
        print(f"\n📊 In-Sample ({in_sample.index[0].strftime('%Y')}–"
              f"{in_sample.index[-1].strftime('%Y')}): "
              f"{len(in_sample)} days")
        print(f"   Out-of-Sample ({out_sample.index[0].strftime('%Y')}–"
              f"{out_sample.index[-1].strftime('%Y')}): "
              f"{len(out_sample)} days")

    return in_sample, out_sample

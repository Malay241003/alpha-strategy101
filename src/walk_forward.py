"""
walk_forward.py — Walk-forward validation with sliding windows.

Ported from tradeBot/backtest/walkForward.js

For each sliding window:
  [0 ... trainDays-1]  →  indicator warm-up (no trades counted)
  [trainDays ... end]  →  test period (trades counted)

Windows advance by testDays (non-overlapping test periods).
All data is sliced to temporal boundaries to prevent look-ahead bias.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.engine import backtest_stock, compute_stock_metrics


def walk_forward_stock(ticker, stock_df, composite_scores_series, regime,
                       years_train=2, years_test=1, opts=None):
    """
    Walk-forward validation for a SINGLE stock.

    Args:
        ticker: str, stock symbol
        stock_df: DataFrame with OHLCV for this stock
        composite_scores_series: pd.Series of composite alpha rank scores
        regime: pd.Series of regime labels
        years_train: training period in years
        years_test: test period in years
        opts: optional engine config overrides

    Returns: dict with {
        ticker, windows, metrics, trades, windowResults
    }
    """
    # Get common dates
    common_dates = sorted(
        set(stock_df.index) &
        set(composite_scores_series.index) &
        set(regime.index)
    )

    if len(common_dates) < 500:
        return {"ticker": ticker, "windows": 0, "metrics": None,
                "trades": [], "windowResults": []}

    train_days = int(years_train * 252)
    test_days = int(years_test * 252)

    wf_trades = []
    window_results = []
    window_num = 0

    # Sliding window loop
    start = 0
    while start + train_days + test_days <= len(common_dates):
        window_num += 1

        # Temporal boundaries for this window
        window_dates = common_dates[start:start + train_days + test_days]
        window_start_date = window_dates[0]
        window_end_date = window_dates[-1]

        # Slice stock data to this window
        window_stock_df = stock_df.loc[
            (stock_df.index >= window_start_date) &
            (stock_df.index <= window_end_date)
        ]

        # Slice scores and regime to this window
        window_scores = composite_scores_series.loc[
            (composite_scores_series.index >= window_start_date) &
            (composite_scores_series.index <= window_end_date)
        ]

        window_regime = regime.loc[
            (regime.index >= window_start_date) &
            (regime.index <= window_end_date)
        ]

        # Override warm-up to be the training period
        engine_opts = dict(opts or {})
        engine_opts["WARM_UP_BARS"] = train_days

        # Run engine on this window
        result = backtest_stock(
            ticker, window_stock_df, window_scores, window_regime,
            opts=engine_opts
        )

        if result and result["trades"]:
            wf_trades.extend(result["trades"])

            window_metrics = compute_stock_metrics(result["trades"])
            window_results.append({
                "window": window_num,
                "startDate": str(window_start_date),
                "endDate": str(window_end_date),
                "metrics": window_metrics,
            })

        # Advance by test period (non-overlapping tests)
        start += test_days

    # Aggregate metrics
    aggregate_metrics = compute_stock_metrics(wf_trades) if wf_trades else None

    return {
        "ticker": ticker,
        "windows": window_num,
        "metrics": aggregate_metrics,
        "trades": wf_trades,
        "windowResults": window_results,
    }


def walk_forward_all_stocks(panels, composite_scores, regime,
                            screened_tickers, opts=None):
    """
    Run walk-forward validation on all screened stocks.

    Args:
        panels: dict of {field: DataFrame(dates×tickers)}
        composite_scores: DataFrame (dates × tickers)
        regime: pd.Series
        screened_tickers: list of ticker strings to validate
        opts: optional engine config overrides

    Returns:
        wf_results: dict of {ticker: wf_result_dict}
        all_wf_trades: list of all trades across all stocks
    """
    print(f"\n{'═' * 60}")
    print(f"  🔄 WALK-FORWARD VALIDATION")
    print(f"{'═' * 60}")
    print(f"  Stocks: {len(screened_tickers)}")
    print(f"  Windows: 2yr train / 1yr test (rolling)\n")

    # Cross-sectional rank for scores
    score_rank = composite_scores.rank(axis=1, pct=True)

    wf_results = {}
    all_wf_trades = []

    for i, ticker in enumerate(screened_tickers):
        # Build per-stock DataFrame
        stock_df = pd.DataFrame({
            "Open": panels["Open"].get(ticker),
            "High": panels["High"].get(ticker),
            "Low": panels["Low"].get(ticker),
            "Close": panels["Close"].get(ticker),
            "Volume": panels["Volume"].get(ticker),
        }).dropna()

        if len(stock_df) < 500:
            continue

        scores = score_rank[ticker] if ticker in score_rank.columns else pd.Series()

        result = walk_forward_stock(
            ticker, stock_df, scores, regime, opts=opts
        )
        wf_results[ticker] = result

        n_trades = len(result["trades"])
        n_windows = result["windows"]
        exp = result["metrics"]["expectancy"] if result["metrics"] else "N/A"
        print(f"  [{i+1:>3d}/{len(screened_tickers)}] {ticker:>6s}: "
              f"{n_windows} windows, {n_trades} trades, Exp: {exp}R")

        if result["trades"]:
            all_wf_trades.extend(result["trades"])

    # Summary
    total_trades = len(all_wf_trades)
    if total_trades > 0:
        r_values = [t["R"] for t in all_wf_trades]
        net_r = sum(r_values)
        won = sum(1 for r in r_values if r > 0)
        print(f"\n  {'═' * 55}")
        print(f"  📊 WF AGGREGATE")
        print(f"  {'═' * 55}")
        print(f"  Stocks validated: {len(wf_results)}")
        print(f"  Total WF trades: {total_trades}")
        print(f"  Won / Lost:      {won} / {total_trades - won}")
        print(f"  Win Rate:        {won/total_trades*100:.1f}%")
        print(f"  Expectancy:      {np.mean(r_values):.3f}R")
        print(f"  Net Profit:      {net_r:.2f}R")

    return wf_results, all_wf_trades

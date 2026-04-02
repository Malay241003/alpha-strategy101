"""
wf_evaluator.py — Walk-Forward Evaluator with institutional accept/reject criteria.

Ported from tradeBot/backtest/wfEvaluator.js

Accept criteria (adapted for daily data):
  1. ≥50% of windows profitable
  2. ≤25% zero-trade windows
  3. Max consecutive losing windows < 6
  4. Overall WF expectancy > 0
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def evaluate_wf(wf_result):
    """
    Evaluate a single stock's walk-forward results.

    Args:
        wf_result: dict from walk_forward_stock()

    Returns:
        verdict: dict with {windows, positivePct, zeroTradePct,
                             maxConsecLossWindows, medianWindowExpectancy,
                             overallExpectancy, ACCEPT}
    """
    windows = wf_result.get("windowResults", [])

    if not windows:
        return {
            "windows": 0,
            "positivePct": 0,
            "zeroTradePct": 100,
            "maxConsecLossWindows": 0,
            "medianWindowExpectancy": 0,
            "overallExpectancy": 0,
            "ACCEPT": False,
        }

    positive = 0
    zero_trade = 0
    consecutive_loss = 0
    max_consec_loss = 0
    expectations = []

    for w in windows:
        metrics = w.get("metrics", {})
        trades = int(metrics.get("trades", 0)) if metrics else 0
        exp = float(metrics.get("expectancy", 0)) if metrics else 0

        expectations.append(exp)

        if trades == 0:
            zero_trade += 1

        if exp > 0:
            positive += 1
            consecutive_loss = 0
        else:
            consecutive_loss += 1
            max_consec_loss = max(max_consec_loss, consecutive_loss)

    # Median expectancy
    sorted_exp = sorted(expectations)
    median_exp = sorted_exp[len(sorted_exp) // 2] if sorted_exp else 0

    # Overall WF expectancy
    overall_exp = float(wf_result["metrics"]["expectancy"]) if wf_result.get("metrics") else 0

    n_windows = len(windows)
    positive_pct = (positive / n_windows) * 100 if n_windows > 0 else 0
    zero_trade_pct = (zero_trade / n_windows) * 100 if n_windows > 0 else 0

    # Accept criteria
    accept = (
        positive_pct >= 50 and
        zero_trade_pct <= 25 and
        max_consec_loss < 6 and
        overall_exp > 0
    )

    return {
        "windows": n_windows,
        "positivePct": round(positive_pct, 1),
        "zeroTradePct": round(zero_trade_pct, 1),
        "maxConsecLossWindows": max_consec_loss,
        "medianWindowExpectancy": round(median_exp, 4),
        "overallExpectancy": round(overall_exp, 4),
        "ACCEPT": accept,
    }


def evaluate_all_stocks(wf_results):
    """
    Evaluate walk-forward results for all stocks.
    Removes stocks that fail, returns survivors.

    Args:
        wf_results: dict of {ticker: wf_result_dict}

    Returns:
        survivors: list of tickers that passed WF
        verdicts: dict of {ticker: verdict}
        removed: list of tickers that failed
    """
    print(f"\n{'═' * 60}")
    print(f"  🔍 WALK-FORWARD EVALUATOR")
    print(f"{'═' * 60}")
    print(f"  Criteria: ≥50% windows profitable, ≤25% empty, maxConsecLoss<6, exp>0\n")

    verdicts = {}
    survivors = []
    removed = []

    for ticker, wf_result in wf_results.items():
        verdict = evaluate_wf(wf_result)
        verdicts[ticker] = verdict

        status = "✅ PASS" if verdict["ACCEPT"] else "❌ FAIL"
        print(
            f"  {status}  {ticker:>6s}  "
            f"{verdict['windows']} windows | "
            f"{verdict['positivePct']:.0f}% profitable | "
            f"maxConsecLoss={verdict['maxConsecLossWindows']} | "
            f"exp={verdict['overallExpectancy']:.3f}"
        )

        if verdict["ACCEPT"]:
            survivors.append(ticker)
        else:
            removed.append(ticker)

    print(f"\n  Summary: {len(survivors)} PASSED, {len(removed)} FAILED "
          f"out of {len(wf_results)} stocks")

    if removed:
        print(f"\n  ❌ Removed:")
        for ticker in removed:
            v = verdicts[ticker]
            reasons = []
            if v["positivePct"] < 50:
                reasons.append(f"profitable={v['positivePct']:.0f}%<50%")
            if v["maxConsecLossWindows"] >= 6:
                reasons.append(f"consec_loss={v['maxConsecLossWindows']}≥6")
            if v["overallExpectancy"] <= 0:
                reasons.append(f"exp={v['overallExpectancy']:.3f}≤0")
            if v["zeroTradePct"] > 25:
                reasons.append(f"empty={v['zeroTradePct']:.0f}%>25%")
            print(f"     {ticker:>6s} — {', '.join(reasons)}")

    return survivors, verdicts, removed

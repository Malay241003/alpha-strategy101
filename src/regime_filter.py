"""
regime_filter.py — Market regime classification using SPY trend + VIX level.

Tiered detection (highest severity wins):
  1. CRISIS:  VIX > 35  (regardless of MA — fires within 1-2 days of crash)
  2. BEAR:    Close < 200MA AND VIX > 25  (fast bear — fires within ~5 days)
             OR  50MA < 200MA AND VIX > 28  (slow bear — catches grinding declines)
  3. BULL:    50MA > 200MA AND VIX < 20
  4. NEUTRAL: everything else
"""

import pandas as pd
import numpy as np
from config import (
    REGIME_SPY_FAST_MA, REGIME_SPY_SLOW_MA,
    REGIME_VIX_BULL_MAX, REGIME_VIX_BEAR_MIN,
    REGIME_VIX_CRISIS, REGIME_VIX_BEAR_FAST,
    BULL, NEUTRAL, BEAR, CRISIS,
    SPY_TICKER, VIX_TICKER
)


def compute_regime(market_data):
    """
    Classify each trading day into a regime using tiered detection.

    Priority: CRISIS > BEAR > NEUTRAL > BULL (highest severity wins).

    Args:
        market_data: dict with keys SPY_TICKER and VIX_TICKER,
                     each containing a DataFrame with 'Close' column.

    Returns:
        pd.Series indexed by date with values: 'BULL', 'NEUTRAL', 'BEAR', or 'CRISIS'
    """
    spy = market_data[SPY_TICKER]["Close"]
    vix = market_data[VIX_TICKER]["Close"]

    # Align VIX to SPY dates (VIX may have slightly different trading days)
    common_dates = spy.index.intersection(vix.index)
    spy = spy.loc[common_dates]
    vix = vix.loc[common_dates]

    # Compute moving averages
    spy_fast_ma = spy.rolling(REGIME_SPY_FAST_MA).mean()
    spy_slow_ma = spy.rolling(REGIME_SPY_SLOW_MA).mean()

    # ── Start with everything NEUTRAL ──
    regime = pd.Series(NEUTRAL, index=common_dates, name="regime")

    # ── Layer 1: BULL (lowest priority — will be overwritten by BEAR/CRISIS) ──
    bull_mask = (spy_fast_ma > spy_slow_ma) & (vix < REGIME_VIX_BULL_MAX)
    regime[bull_mask] = BULL

    # ── Layer 2: BEAR — slow detection (50MA < 200MA + high VIX) ──
    bear_slow_mask = (spy_fast_ma < spy_slow_ma) & (vix > REGIME_VIX_BEAR_MIN)
    regime[bear_slow_mask] = BEAR

    # ── Layer 3: BEAR — fast detection (Close < 200MA + elevated VIX) ──
    # This fires much earlier than the MA crossover in sharp crashes
    bear_fast_mask = (spy < spy_slow_ma) & (vix > REGIME_VIX_BEAR_FAST)
    regime[bear_fast_mask] = BEAR

    # ── Layer 4: CRISIS override (VIX spike — highest priority) ──
    # VIX > 35 = something is very wrong, go to cash immediately
    crisis_mask = vix > REGIME_VIX_CRISIS
    regime[crisis_mask] = CRISIS

    # Print summary
    counts = regime.value_counts()
    total = len(regime)
    print(f"📊 Regime classification ({total} days):")
    emojis = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴", "CRISIS": "🚨"}
    for r in [BULL, NEUTRAL, BEAR, CRISIS]:
        n = counts.get(r, 0)
        pct = n / total * 100
        emoji = emojis.get(r, "❓")
        print(f"   {emoji} {r}: {n} days ({pct:.1f}%)")

    return regime


def get_regime_summary(regime):
    """
    Get a DataFrame summarizing regime periods (start, end, duration).
    Useful for visualization overlay.
    """
    changes = regime != regime.shift(1)
    periods = []
    current_start = regime.index[0]
    current_regime = regime.iloc[0]

    for date, changed in changes.items():
        if changed:
            periods.append({
                "start": current_start,
                "end": date,
                "regime": current_regime,
                "days": (date - current_start).days
            })
            current_start = date
            current_regime = regime[date]

    # Add the last period
    periods.append({
        "start": current_start,
        "end": regime.index[-1],
        "regime": current_regime,
        "days": (regime.index[-1] - current_start).days
    })

    return pd.DataFrame(periods)


# ═══════════════════════════════════════════════════════════════════════
# MARKOV CHAIN ADAPTER
# ═══════════════════════════════════════════════════════════════════════

def compute_regime_markov_adapter(market_data, calibration_end=None):
    """
    Compute regime using the Markov Chain HMM.

    Wrapper that calls the Markov model and returns ONLY the regime Series
    (same signature as compute_regime) for pipeline compatibility.

    The exposure_scores are also returned as a second value for callers
    that want graduated position sizing.

    Returns:
        tuple: (regime_series, exposure_scores)
            - regime_series: pd.Series (same as compute_regime output)
            - exposure_scores: pd.Series of float [0.0, 1.0]
    """
    from src.markov_regime import compute_regime_markov
    return compute_regime_markov(market_data, calibration_end=calibration_end)


def compute_regime_auto(market_data, calibration_end=None):
    """
    Unified entry point: selects regime method based on config.MARKOV_REGIME_METHOD.

    Returns:
        tuple: (regime_series, exposure_scores_or_None)
            - If method="heuristic": exposure_scores is None (use REGIME_EXPOSURE dict)
            - If method="markov": exposure_scores is a pd.Series of floats
    """
    from config import MARKOV_REGIME_METHOD

    if MARKOV_REGIME_METHOD == "markov":
        print("🔮 Using Markov Chain HMM regime filter")
        return compute_regime_markov_adapter(market_data, calibration_end)
    else:
        print("📐 Using heuristic (MA + VIX) regime filter")
        regime = compute_regime(market_data)
        return regime, None


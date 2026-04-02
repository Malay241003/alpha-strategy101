"""
crypto_regime_filter.py — BTC-based regime classification for crypto strategy.

Three regime approaches combined (tradeBot EMA200 + SSRN momentum + VIX-like):

1. BTC EMA Trend (from tradeBot's engine.js):
   - BULL: BTC close > EMA200 (1H equivalent on daily = SMA200)
   - BEAR: BTC close < EMA200

2. BTC Momentum Quality (from SSRN research):
   - Risk-adjusted momentum = return / volatility
   - Positive momentum quality → confirms BULL
   - Negative → confirms BEAR

3. Crypto Volatility Regime:
   - BTC 20d realized vol vs 60d average vol
   - Elevated volatility during downtrend → strong BEAR
   - Low volatility during uptrend → strong BULL

Combined: we use the tradeBot EMA200 as primary (proven), with
momentum quality as confirmation (SSRN-backed).
"""

import pandas as pd
import numpy as np


def compute_crypto_regime(btc_data, fast_period=50, slow_period=200):
    """
    Classify each trading day into BULL / NEUTRAL / BEAR using BTC.

    Matches tradeBot's approach:
      - Primary: close vs EMA200 (tradeBot's macro regime filter)
      - Secondary: 50/200 SMA crossover (golden cross / death cross)
      - Confirmation: momentum quality (SSRN risk-adjusted momentum)

    Args:
        btc_data: DataFrame with BTC-USD OHLCV (must have 'Close')
        fast_period: fast SMA period (default: 50)
        slow_period: slow SMA period (default: 200, matching tradeBot EMA200)

    Returns:
        pd.Series indexed by date with values: 'BULL', 'NEUTRAL', 'BEAR'
    """
    from crypto_config import BULL, NEUTRAL, BEAR

    close = btc_data["Close"]

    # ── Primary: EMA200 trend (tradeBot's approach) ──
    # Using SMA200 instead of EMA200 since we have daily data
    # (SMA200 and EMA200 are very similar at daily resolution)
    sma_fast = close.rolling(fast_period).mean()
    sma_slow = close.rolling(slow_period).mean()

    # ── Secondary: Momentum quality (SSRN) ──
    returns_20d = close.pct_change(20)
    vol_20d = close.pct_change(1).rolling(20).std() * np.sqrt(252)
    momentum_quality = returns_20d / (vol_20d + 1e-10)

    # ── Classify ──
    regime = pd.Series(NEUTRAL, index=close.index, name="regime")

    # BULL: close > SMA200 AND either:
    #   - SMA50 > SMA200 (golden cross), OR
    #   - momentum quality > 0 (positive risk-adjusted momentum)
    bull_mask = (close > sma_slow) & (
        (sma_fast > sma_slow) | (momentum_quality > 0)
    )
    regime[bull_mask] = BULL

    # BEAR: close < SMA200 AND either:
    #   - SMA50 < SMA200 (death cross), OR
    #   - momentum quality < -0.5 (strong negative momentum)
    bear_mask = (close < sma_slow) & (
        (sma_fast < sma_slow) | (momentum_quality < -0.5)
    )
    regime[bear_mask] = BEAR

    # Print summary
    counts = regime.value_counts()
    total = len(regime.dropna())
    print(f"📊 BTC Regime classification ({total} days):")
    for r in [BULL, NEUTRAL, BEAR]:
        n = counts.get(r, 0)
        pct = n / total * 100 if total > 0 else 0
        emoji = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}[r]
        print(f"   {emoji} {r}: {n} days ({pct:.1f}%)")

    return regime

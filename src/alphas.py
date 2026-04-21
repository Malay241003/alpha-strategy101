"""
alphas.py — 10 selected alpha formulas from the '101 Formulaic Alphas' paper.

NOTE on output scales:
    Raw alpha scores have very different scales across formulas (e.g. alpha_101
    is bounded ~[-1,1], while alpha_36 sums 5 ranked terms ~[0,5]).  A
    neutralize + z-score step is needed before combining alphas into a
    composite signal.  See the combiner / portfolio construction layer.

Each alpha function takes a dict of panel DataFrames:
    panels = {
        "Open":    DataFrame (dates × tickers),
        "Close":   DataFrame,
        "High":    DataFrame,
        "Low":     DataFrame,
        "Volume":  DataFrame,
        "vwap":    DataFrame,
        "returns": DataFrame,
        "adv20":   DataFrame,
        ...
    }

Each returns a DataFrame (dates × tickers) of raw alpha scores.
"""

import numpy as np
import pandas as pd
from src.operators import (
    rank, scale, delay, delta, correlation, covariance,
    ts_min, ts_max, ts_argmax, ts_argmin, ts_rank,
    sum_, product, stddev, decay_linear,
    sign, log, abs_, signedpower
)


def _require_panel(panels, key):
    """Retrieve a panel by key with a clear error if missing."""
    if key not in panels:
        raise KeyError(
            f"Required panel '{key}' not found. "
            f"Available keys: {sorted(panels.keys())}"
        )
    return panels[key]


def alpha_101(panels):
    """
    Alpha#101: (close - open) / ((high - low) + 0.001)

    Intraday momentum — if close >> open, stock had bullish intraday action.
    Simplest alpha in the paper. Delay-1 (trade next day).
    """
    close = panels["Close"]
    open_ = panels["Open"]
    high = panels["High"]
    low = panels["Low"]
    return (close - open_) / ((high - low) + 0.001)


def alpha_9(panels):
    """
    Alpha#9: Adaptive mean-reversion / momentum.

    If min of delta(close,1) over past 5 days > 0: follow trend (delta)
    If max of delta(close,1) over past 5 days < 0: follow trend (delta)
    Otherwise: fade the move (-1 * delta)

    Self-adapting: becomes momentum in persistent trends, mean-reversion in chop.
    """
    close = panels["Close"]
    d1 = delta(close, 1)
    ts_min_d1 = ts_min(d1, 5)
    ts_max_d1 = ts_max(d1, 5)

    # cond1 and cond2 are mutually exclusive (min>0 implies max>0, max<0
    # implies min<0), but using np.where avoids any DataFrame boolean-
    # indexing misalignment risk and is cleaner than triple assignment.
    cond1 = ts_min_d1 > 0   # all 5 days up   → momentum
    cond2 = ts_max_d1 < 0   # all 5 days down → momentum

    result = pd.DataFrame(
        np.where(cond1, d1, np.where(cond2, d1, -1 * d1)),
        index=close.index, columns=close.columns,
    )
    # Propagate NaN where inputs were NaN (comparisons on NaN yield False,
    # which would silently map to the default branch instead of NaN).
    result[d1.isna()] = np.nan

    return result


def alpha_32(panels):
    """
    Alpha#32: scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230)))

    Multi-factor:
      - Term 1: Mean-reversion to 7-day MA (negative if above MA)
      - Term 2: 230-day correlation between VWAP and lagged close (long-term trend)
    Very low turnover due to long lookback periods.
    """
    close = panels["Close"]
    vwap = panels["vwap"]

    ma7 = sum_(close, 7) / 7
    term1 = scale(ma7 - close)
    term2 = 20 * scale(correlation(vwap, delay(close, 5), 230))

    return term1 + term2


def alpha_23(panels):
    """
    Alpha#23: ((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0

    Breakout fade: if today's high exceeds the 20-day average high
    (i.e., a breakout), bet against it by shorting the 2-day high change.
    Otherwise, stay flat (0).
    """
    high = panels["High"]

    ma20_high = sum_(high, 20) / 20
    breakout = ma20_high < high  # True when breaking out above avg
    d2_high = delta(high, 2)

    result = pd.DataFrame(
        np.where(breakout, -1 * d2_high, 0.0),
        index=high.index, columns=high.columns,
    )
    # Where inputs are NaN the comparison yields False → 0 (flat), which
    # hides missing data.  Propagate NaN explicitly.
    result[breakout.isna() | d2_high.isna()] = np.nan
    return result


def alpha_52(panels):
    """
    Alpha#52: (((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
               rank(((sum(returns, 240) - sum(returns, 20)) / 220))) *
              ts_rank(volume, 5)

    Long-term momentum + 5-day low breakout + volume:
      - Term 1: Change in 5-day low (bounce from recent low)
      - Term 2: Rank of 240-day vs 20-day return (long-term momentum filter)
      - Term 3: Volume rank (prefer volume confirmation)
    """
    low = panels["Low"]
    returns = panels["returns"]
    volume = panels["Volume"]

    ts_min_low_5 = ts_min(low, 5)
    term1 = (-1 * ts_min_low_5) + delay(ts_min_low_5, 5)
    term2 = rank((sum_(returns, 240) - sum_(returns, 20)) / 220)
    term3 = ts_rank(volume, 5)

    return term1 * term2 * term3


def alpha_7(panels):
    """
    Alpha#7: (adv20 < volume) ?
                ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7)))
              : (-1)

    Volume-conditional momentum:
      - If today's volume > average → signal based on 7-day price change
        (momentum direction, weighted by rank of magnitude)
      - If volume is low → default to -1 (slight short bias / no signal)
    """
    close = panels["Close"]
    volume = panels["Volume"]
    adv20 = _require_panel(panels, "adv20")

    d7 = delta(close, 7)
    momentum_signal = (-1 * ts_rank(abs_(d7), 60)) * sign(d7)

    vol_surge = adv20 < volume
    result = pd.DataFrame(
        np.where(vol_surge, momentum_signal, -1.0),
        index=close.index, columns=close.columns,
    )
    # NaN in adv20 or volume makes vol_surge False → defaults to -1,
    # which is a real short bias from missing data.  Mask those out.
    result[vol_surge.isna() | momentum_signal.isna()] = np.nan

    return result


def alpha_37(panels):
    """
    Alpha#37: rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close))

    Gap + trend:
      - Term 1: 200-day correlation between yesterday's gap and close
        (captures long-term gap-reversion pattern)
      - Term 2: Today's gap direction
    Very low turnover.
    """
    open_ = panels["Open"]
    close = panels["Close"]

    gap = open_ - close
    term1 = rank(correlation(delay(gap, 1), close, 200))
    term2 = rank(gap)

    return term1 + term2


def alpha_36(panels):
    """
    Alpha#36: (2.21 * rank(correlation((close - open), delay(volume, 1), 15))) +
              (0.7  * rank((open - close))) +
              (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5))) +
              rank(abs(correlation(vwap, adv20, 6))) +
              (0.6  * rank((((sum(close, 200) / 200) - open) * (close - open))))

    5-component diversified blend:
      1. Gap-volume correlation
      2. Intraday gap direction
      3. Lagged return rank (6-day ago returns, ranked over 5 days)
      4. VWAP-ADV correlation strength
      5. 200-day MA deviation × intraday return
    """
    close = panels["Close"]
    open_ = panels["Open"]
    volume = panels["Volume"]
    vwap = panels["vwap"]
    returns = panels["returns"]
    adv20 = _require_panel(panels, "adv20")

    term1 = 2.21 * rank(correlation(close - open_, delay(volume, 1), 15))
    term2 = 0.7 * rank(open_ - close)
    term3 = 0.73 * rank(ts_rank(delay(-1 * returns, 6), 5))
    term4 = rank(abs_(correlation(vwap, adv20, 6)))
    term5 = 0.6 * rank(((sum_(close, 200) / 200) - open_) * (close - open_))

    return term1 + term2 + term3 + term4 + term5


def alpha_19(panels):
    """
    Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) *
              (1 + rank((1 + sum(returns, 250)))))

    Secular trend / long-term momentum:
      - Sign part: reversal signal based on 7-day return
      - Multiplier: amplified by 250-day cumulative return rank
    Stocks with strong yearly momentum get bigger positions.

    NOTE on redundancy:  (close - delay(close,7)) + delta(close,7) equals
    2 * delta(close,7) because delta(close,7) ≡ close - delay(close,7).
    Wrapping in sign() makes the factor of 2 irrelevant (sign is preserved),
    so the formula is correct but the two-term expression is a no-op.
    Kept as-is to match the paper exactly.
    """
    close = panels["Close"]
    returns = panels["returns"]

    sign_part = -1 * sign((close - delay(close, 7)) + delta(close, 7))
    trend_rank = 1 + rank(1 + sum_(returns, 250))

    return sign_part * trend_rank


def alpha_12(panels):
    """
    Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))

    Volume reversal:
      - If volume increased → expect mean-reversion (fade the move)
      - If volume decreased → expect continuation
    High turnover but simple and uncorrelated with other alphas.
    """
    close = panels["Close"]
    volume = panels["Volume"]

    return sign(delta(volume, 1)) * (-1 * delta(close, 1))


# ─────────────────────────────────────────────
# Registry: maps alpha name to function
# ─────────────────────────────────────────────
ALPHA_REGISTRY = {
    "alpha_101": alpha_101,
    "alpha_9":   alpha_9,
    "alpha_32":  alpha_32,
    "alpha_23":  alpha_23,
    "alpha_52":  alpha_52,
    "alpha_7":   alpha_7,
    "alpha_37":  alpha_37,
    "alpha_36":  alpha_36,
    "alpha_19":  alpha_19,
    "alpha_12":  alpha_12,
}


def compute_all_alphas(panels):
    """
    Compute all 10 selected alphas.

    Returns: dict of {alpha_name: DataFrame(dates × tickers)}
    Raises: RuntimeError if any alpha fails (with all errors collected).
    """
    print("🧮 Computing alphas...")
    alpha_scores = {}
    errors = {}  # {name: exception} — surface failures instead of hiding them
    for name, func in ALPHA_REGISTRY.items():
        print(f"  Computing {name}...")
        try:
            score = func(panels)
            # Replace inf with NaN
            score = score.replace([np.inf, -np.inf], np.nan)
            alpha_scores[name] = score
        except Exception as e:
            print(f"  ❌ Failed to compute {name}: {e}")
            errors[name] = e

    if errors:
        err_summary = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(
            f"{len(errors)} alpha(s) failed to compute: {err_summary}"
        )

    print(f"✅ Computed {len(alpha_scores)} alphas")
    return alpha_scores

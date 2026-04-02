"""
scorer.py — Composite alpha scoring and portfolio selection.

Combines 10 alpha signals into a single composite score using
regime-dependent weights, then selects top-N stocks as the portfolio.
"""

import pandas as pd
import numpy as np
from src.operators import rank
from config import ALPHA_WEIGHTS, TOP_N_STOCKS, REGIME_EXPOSURE, BULL, NEUTRAL, BEAR, CRISIS


def compute_composite_score(alpha_scores, regime):
    """
    Compute the regime-weighted composite score for all stocks on each day.

    Args:
        alpha_scores: dict of {alpha_name: DataFrame(dates × tickers)}
        regime: pd.Series indexed by date with regime labels

    Returns:
        DataFrame (dates × tickers) of composite scores
    """
    # Get common dates across all alphas and the regime series
    common_dates = regime.index
    for name, scores in alpha_scores.items():
        common_dates = common_dates.intersection(scores.index)
    common_dates = sorted(common_dates)

    # Get columns (tickers) from any alpha
    first_alpha = next(iter(alpha_scores.values()))
    tickers = first_alpha.columns

    # First: rank each alpha cross-sectionally (so all are on [0,1] scale)
    ranked_alphas = {}
    for name, scores in alpha_scores.items():
        ranked_alphas[name] = rank(scores.loc[common_dates, tickers])

    # Compute composite score day by day
    composite = pd.DataFrame(0.0, index=common_dates, columns=tickers)

    for name in alpha_scores.keys():
        if name not in ALPHA_WEIGHTS:
            continue

        weights = ALPHA_WEIGHTS[name]
        alpha_ranked = ranked_alphas[name]

        # Build a weight series aligned with dates based on regime
        weight_series = regime.loc[common_dates].map(weights)

        # Multiply rank by weight and add to composite
        # weight_series is a Series (dates,), alpha_ranked is DataFrame (dates × tickers)
        composite += alpha_ranked.mul(weight_series, axis=0)

    print(f"✅ Composite score computed: {composite.shape[0]} days × {composite.shape[1]} stocks")
    return composite


def select_portfolio(composite_scores, regime):
    """
    Select the top-N stocks by composite score each day and assign weights.

    Args:
        composite_scores: DataFrame (dates × tickers) of composite scores
        regime: pd.Series indexed by date with regime labels

    Returns:
        DataFrame (dates × tickers) of portfolio weights (0 to max_weight)
    """
    holdings = pd.DataFrame(0.0, index=composite_scores.index,
                            columns=composite_scores.columns)

    for date in composite_scores.index:
        scores = composite_scores.loc[date].dropna()
        if len(scores) == 0:
            continue

        # Get regime for this day
        day_regime = regime.get(date, NEUTRAL)
        exposure = REGIME_EXPOSURE.get(day_regime, 0.5)

        # Select top N stocks
        n = min(TOP_N_STOCKS, len(scores))
        top_stocks = scores.nlargest(n).index

        # Equal-weight within the long basket, scaled by regime exposure
        weight = exposure / n
        holdings.loc[date, top_stocks] = weight

    # Summary
    avg_positions = (holdings > 0).sum(axis=1).mean()
    avg_exposure = holdings.sum(axis=1).mean()
    print(f"✅ Portfolio constructed: avg {avg_positions:.1f} positions, "
          f"avg {avg_exposure:.1%} exposure")

    return holdings

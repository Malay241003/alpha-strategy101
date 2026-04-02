"""
operators.py — All mathematical operators from Appendix A of '101 Formulaic Alphas'.

Two categories:
  1. TIME-SERIES operators: operate on a single stock's history (pandas Series).
  2. CROSS-SECTIONAL operators: operate across all stocks on a given day (DataFrame row).

Convention:
  - Time-series functions take a pd.Series and return a pd.Series.
  - Cross-sectional functions take a pd.DataFrame (dates × tickers) and return same.
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════
# CROSS-SECTIONAL OPERATORS
# ═══════════════════════════════════════════════

def rank(df):
    """
    Cross-sectional percentile rank (across all stocks for each day).
    Returns values in [0, 1].

    Input: DataFrame (dates × tickers)
    Output: DataFrame (dates × tickers) with ranks
    """
    if isinstance(df, pd.Series):
        # If a single series, just rank it as time-series percentile
        return df.rank(pct=True)
    return df.rank(axis=1, pct=True)


def scale(df, a=1.0):
    """
    Rescale x such that sum(abs(x)) = a, applied cross-sectionally.

    Input: DataFrame (dates × tickers)
    Output: DataFrame (dates × tickers) rescaled
    """
    if isinstance(df, pd.Series):
        abs_sum = df.abs().sum()
        return df * a / abs_sum if abs_sum != 0 else df * 0
    abs_sum = df.abs().sum(axis=1)
    abs_sum = abs_sum.replace(0, np.nan)
    return df.mul(a, axis=0).div(abs_sum, axis=0)


# ═══════════════════════════════════════════════
# TIME-SERIES OPERATORS (applied per-stock)
# ═══════════════════════════════════════════════

def delay(x, d):
    """Value of x, d days ago. Works on Series or DataFrame."""
    return x.shift(d)


def delta(x, d):
    """Today's value of x minus the value of x, d days ago."""
    return x - delay(x, d)


def correlation(x, y, d):
    """
    Rolling d-day Pearson correlation between x and y.
    Works on Series (single stock) or DataFrames (panel, column-wise).
    """
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        # Column-wise rolling correlation for panel
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            if col in y.columns:
                result[col] = x[col].rolling(d).corr(y[col])
        return result
    return x.rolling(d).corr(y)


def covariance(x, y, d):
    """Rolling d-day covariance between x and y."""
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            if col in y.columns:
                result[col] = x[col].rolling(d).cov(y[col])
        return result
    return x.rolling(d).cov(y)


def ts_min(x, d):
    """Time-series minimum over the past d days."""
    return x.rolling(d).min()


def ts_max(x, d):
    """Time-series maximum over the past d days."""
    return x.rolling(d).max()


def ts_argmax(x, d):
    """
    Which day (offset from today, counting back) ts_max occurred on.
    Returns 0 if max is today, d-1 if max was d days ago.
    """
    def _argmax_series(s):
        return s.rolling(d).apply(lambda w: d - 1 - np.argmax(w), raw=True)

    if isinstance(x, pd.DataFrame):
        return x.apply(_argmax_series)
    return _argmax_series(x)


def ts_argmin(x, d):
    """Which day (offset from today) ts_min occurred on."""
    def _argmin_series(s):
        return s.rolling(d).apply(lambda w: d - 1 - np.argmin(w), raw=True)

    if isinstance(x, pd.DataFrame):
        return x.apply(_argmin_series)
    return _argmin_series(x)


def ts_rank(x, d):
    """
    Time-series rank: percentile rank of current value within past d days.
    Returns values in [0, 1].
    """
    def _ts_rank_series(s):
        return s.rolling(d).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
        )

    if isinstance(x, pd.DataFrame):
        return x.apply(_ts_rank_series)
    return _ts_rank_series(x)


def sum_(x, d):
    """Rolling d-day sum."""
    return x.rolling(d).sum()


def product(x, d):
    """Rolling d-day product."""
    def _prod_series(s):
        return s.rolling(d).apply(np.prod, raw=True)

    if isinstance(x, pd.DataFrame):
        return x.apply(_prod_series)
    return _prod_series(x)


def stddev(x, d):
    """Rolling d-day standard deviation."""
    return x.rolling(d).std()


def decay_linear(x, d):
    """
    Weighted moving average over the past d days with linearly decaying
    weights: d, d-1, ..., 1 (rescaled to sum up to 1).
    """
    weights = np.arange(1, d + 1, dtype=float)  # [1, 2, ..., d]
    weights = weights / weights.sum()

    def _decay_series(s):
        return s.rolling(d).apply(lambda w: np.dot(w, weights), raw=True)

    if isinstance(x, pd.DataFrame):
        return x.apply(_decay_series)
    return _decay_series(x)


# ═══════════════════════════════════════════════
# BASIC MATH (convenience wrappers)
# ═══════════════════════════════════════════════

def sign(x):
    """Element-wise sign: -1, 0, or +1."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return np.sign(x)
    return np.sign(x)


def log(x):
    """Element-wise natural logarithm."""
    return np.log(x.replace(0, np.nan))


def abs_(x):
    """Element-wise absolute value."""
    return x.abs() if isinstance(x, (pd.DataFrame, pd.Series)) else np.abs(x)


def signedpower(x, a):
    """sign(x) * abs(x)^a"""
    return sign(x) * (abs_(x) ** a)

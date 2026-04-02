"""
sector_risk_model.py — Sector-level and stock-level risk decomposition.

Ported from tradeBot/scripts_us_stocks/sectorRiskModel.js

Computes:
  - Rolling beta vs equal-weight portfolio factor (OLS)
  - Idiosyncratic volatility (residual after market factor)
  - Downside semi-deviation (Sortino-style tail risk)
  - Full NxN correlation matrix with Ledoit-Wolf shrinkage
  - Sector-level intra/cross-sector correlations

References:
  - Fama & French (1993) "Common Risk Factors"
  - Ang, Chen & Xing (2006) "Downside Risk"
  - Ledoit & Wolf (2004) "A Well-Conditioned Estimator"
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.statistical_screen import SECTOR_MAP


def _mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

def _std(arr):
    return np.std(arr, ddof=0) if len(arr) > 1 else 0.0

def downside_semi_dev(arr, target=0.0):
    """Downside semi-deviation — only considers negative returns."""
    negatives = [r for r in arr if r < target]
    if not negatives:
        return 0.0
    return np.sqrt(sum((r - target) ** 2 for r in negatives) / len(arr))


def regression_beta(stock_returns, market_returns):
    """
    OLS regression: y = α + β·x + ε
    Returns beta, alpha, idiosyncratic volatility, R²
    """
    n = min(len(stock_returns), len(market_returns))
    if n < 10:
        return {"beta": 1.0, "alpha": 0.0, "idioVol": 0.0, "r2": 0.0}

    x = np.array(market_returns[:n])
    y = np.array(stock_returns[:n])

    x_mean, y_mean = x.mean(), y.mean()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
    ss_xx = ((x - x_mean) ** 2).sum()

    beta = ss_xy / ss_xx if ss_xx > 0 else 1.0
    alpha = y_mean - beta * x_mean

    residuals = y - alpha - beta * x
    idio_vol = float(np.std(residuals, ddof=0))

    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "beta": round(beta, 3),
        "alpha": round(alpha, 4),
        "idioVol": round(idio_vol, 4),
        "r2": round(r2, 3),
    }


def compute_correlation_matrix(returns_dict, tickers):
    """
    Compute NxN correlation matrix from daily return series.
    Returns raw correlation matrix.
    """
    n = len(tickers)
    min_len = min(len(returns_dict.get(t, [])) for t in tickers)
    if min_len < 20:
        return np.eye(n)

    arr = np.array([returns_dict[t][:min_len] for t in tickers])
    corr = np.corrcoef(arr)
    # Fix any NaN (constant series)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def ledoit_wolf_shrinkage(raw_corr, n_observations):
    """
    Ledoit-Wolf shrinkage toward identity matrix.
    Improves numerical stability for portfolio optimization.
    """
    n = raw_corr.shape[0]
    shrinkage = min(0.5, max(0.1, 1 / np.sqrt(max(n_observations, 1))))

    target = np.eye(n)
    shrunk = (1 - shrinkage) * raw_corr + shrinkage * target

    return shrunk, shrinkage


def compute_covariance_matrix(returns_dict, tickers):
    """
    Compute covariance matrix with Ledoit-Wolf shrinkage.
    Used by portfolio optimizer.
    """
    n = len(tickers)
    min_len = min(len(returns_dict.get(t, [])) for t in tickers)

    arr = np.array([returns_dict[t][:min_len] for t in tickers])
    T = min_len

    # Sample covariance
    means = arr.mean(axis=1, keepdims=True)
    centered = arr - means
    sample_cov = (centered @ centered.T) / (T - 1)

    # Shrinkage target: diagonal with average variance
    avg_var = np.trace(sample_cov) / n
    target = np.eye(n) * avg_var

    shrinkage = min(0.5, max(0.1, 1 / np.sqrt(max(T, 1))))
    cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * target

    return cov_matrix, shrinkage


def run_sector_risk_model(screened_stocks, engine_trades_df):
    """
    Full sector risk model analysis.

    Args:
        screened_stocks: list of stock stat dicts from statistical_screen
        engine_trades_df: DataFrame of all engine trades

    Returns:
        risk_model: dict with per-stock metrics, correlation data, sector analysis
    """
    tickers = [s["ticker"] for s in screened_stocks]
    n = len(tickers)

    if n < 3:
        print("  ⚠️ Too few stocks for risk model")
        return {"perStock": {}, "avgCorr": 0, "shrinkage": 0}

    print(f"\n{'═' * 60}")
    print(f"  📐 SECTOR RISK MODEL")
    print(f"{'═' * 60}")
    print(f"  Analyzing {n} screened stocks\n")

    # Build daily return series from engine trades
    # Approximate: distribute trade R across holding days
    returns_dict = {}
    for ticker in tickers:
        ticker_trades = engine_trades_df[engine_trades_df["ticker"] == ticker] if len(engine_trades_df) > 0 else pd.DataFrame()
        daily_r = []
        for _, t in ticker_trades.iterrows():
            days = max(1, int(t.get("holding_days", 1)))
            r_per_day = t["R"] / days
            daily_r.extend([r_per_day] * days)
        if not daily_r:
            daily_r = [0.0] * 100  # placeholder
        returns_dict[ticker] = daily_r

    # Align to common length
    min_len = min(len(returns_dict[t]) for t in tickers)
    for t in tickers:
        returns_dict[t] = returns_dict[t][:min_len]

    # Build equal-weight "market" factor
    market_returns = []
    for i in range(min_len):
        day_r = sum(returns_dict[t][i] for t in tickers) / n
        market_returns.append(day_r)

    # Per-stock risk metrics
    per_stock = {}
    print("  Stock    Sector            Beta   IdioVol    R²   DownSD   TotalVol")
    print("  ─────    ──────            ────   ───────    ──   ──────   ────────")

    for ticker in tickers:
        r = returns_dict[ticker]
        sector = SECTOR_MAP.get(ticker, "Unknown")
        reg = regression_beta(r, market_returns)
        total_vol = _std(r)
        dsd = downside_semi_dev(r)

        per_stock[ticker] = {
            "sector": sector,
            **reg,
            "totalVol": round(total_vol, 4),
            "downsideSemiDev": round(dsd, 4),
        }

        print(
            f"  {ticker:>6s}   {sector:<16s}  {reg['beta']:>5.2f}  {reg['idioVol']:>7.4f}  "
            f"{reg['r2']:>4.2f}  {dsd:>7.4f}  {total_vol:>7.4f}"
        )

    # Correlation matrix
    raw_corr = compute_correlation_matrix(returns_dict, tickers)
    shrunk_corr, shrinkage = ledoit_wolf_shrinkage(raw_corr, min_len)

    # Covariance matrix (for portfolio optimizer)
    cov_matrix, _ = compute_covariance_matrix(returns_dict, tickers)

    # Average portfolio correlation
    mask = ~np.eye(n, dtype=bool)
    avg_corr = shrunk_corr[mask].mean() if n > 1 else 0.0

    # Sector groupings
    sector_groups = {}
    for i, t in enumerate(tickers):
        sec = SECTOR_MAP.get(t, "Unknown")
        sector_groups.setdefault(sec, []).append(i)

    # Intra-sector correlations
    print(f"\n  Intra-sector correlations:")
    for sec, indices in sorted(sector_groups.items()):
        if len(indices) < 2:
            continue
        pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs.append(shrunk_corr[indices[i], indices[j]])
        avg = np.mean(pairs) if pairs else 0
        bar = "█" * max(0, round(avg * 20))
        print(f"    {sec:<16s} {bar} {avg:.3f} ({len(indices)} stocks)")

    avg_beta = np.mean([m["beta"] for m in per_stock.values()])
    print(f"\n  📊 Average portfolio correlation: {avg_corr:.3f}")
    print(f"  📊 Average beta: {avg_beta:.3f}")
    print(f"  📊 Ledoit-Wolf shrinkage: {shrinkage*100:.1f}%")

    return {
        "tickers": tickers,
        "perStock": per_stock,
        "covMatrix": cov_matrix,
        "corrMatrix": shrunk_corr,
        "shrinkage": shrinkage,
        "avgCorrelation": round(avg_corr, 3),
        "sectorGroups": sector_groups,
        "returnsDict": returns_dict,
    }

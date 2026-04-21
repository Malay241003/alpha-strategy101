"""
earnings_loader.py — Load Alpha Vantage earnings data and compute per-stock features.

Reads quarterly earnings reports from data/earnings_av/ and provides:
  - Last earnings surprise percentage
  - Days since last earnings report
  - Earnings beat streak (consecutive quarters of positive surprise)

These features are consumed by ml_scorer.py's build_features() function.
Data source: Alpha Vantage Earnings endpoint (cached as CSV).
"""

import os
import numpy as np
import pandas as pd
from bisect import bisect_right


def load_earnings_data(data_dir):
    """
    Load all earnings CSVs from data/earnings_av/ into a dictionary.

    Args:
        data_dir: Path to the project data/ directory

    Returns:
        dict of {ticker: DataFrame} where each DataFrame has columns:
            date, reportedEPS, estimatedEPS, surprise, surprisePercentage
        Sorted by date ascending.
    """
    earnings_dir = os.path.join(data_dir, "earnings_av")
    if not os.path.isdir(earnings_dir):
        print(f"  ⚠️ No earnings_av directory found at {earnings_dir}")
        return {}

    earnings = {}
    files = [f for f in os.listdir(earnings_dir) if f.endswith(".csv")]

    for f in files:
        # Derive ticker from filename: BRK_B.csv → BRK-B, AAPL.csv → AAPL
        ticker = f.replace(".csv", "").replace("_", "-")
        path = os.path.join(earnings_dir, f)

        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Only keep rows with valid data
            df = df.dropna(subset=["date", "surprisePercentage"])

            if len(df) > 0:
                earnings[ticker] = df
        except Exception as e:
            # Silently skip malformed files
            pass

    return earnings


def build_earnings_lookup(earnings_dict, tickers, dates):
    """
    Pre-compute earnings features for all (ticker, date) pairs.

    For each ticker on each date, looks up the most recent earnings report
    BEFORE that date (strictly causal — no look-ahead).

    Args:
        earnings_dict: dict from load_earnings_data()
        tickers: list of ticker symbols
        dates: sorted list of trading dates

    Returns:
        dict of {ticker: DataFrame} indexed by date with columns:
            earnings_surprise, days_since_earnings, earnings_beat_streak
    """
    lookup = {}

    for ticker in tickers:
        if ticker not in earnings_dict:
            # No earnings data for this ticker — features will be NaN
            continue

        edf = earnings_dict[ticker]
        earn_dates = edf["date"].values  # sorted numpy datetime64 array
        surprises = edf["surprisePercentage"].values

        # Pre-compute beat streak at each earnings report
        # beat_streak[i] = number of consecutive positive surprises ending at i
        beat_streaks = np.zeros(len(surprises), dtype=int)
        for i in range(len(surprises)):
            if surprises[i] > 0:
                beat_streaks[i] = (beat_streaks[i - 1] + 1) if i > 0 else 1
            else:
                beat_streaks[i] = 0

        # Convert earnings dates to pandas Timestamps for comparison
        earn_ts = pd.to_datetime(earn_dates)

        # For each trading date, find the most recent earnings before it
        surprise_vals = []
        days_since_vals = []
        streak_vals = []
        result_dates = []

        for date in dates:
            # Binary search: find the last earnings date strictly before `date`
            idx = bisect_right(earn_ts, date) - 1

            if idx < 0:
                # No earnings report before this date
                surprise_vals.append(np.nan)
                days_since_vals.append(np.nan)
                streak_vals.append(np.nan)
            else:
                surprise_vals.append(float(surprises[idx]))
                days_since = (date - earn_ts[idx]).days
                days_since_vals.append(min(days_since, 90))  # clip at 90 days
                streak_vals.append(float(min(beat_streaks[idx], 5)))  # clip at 5

            result_dates.append(date)

        lookup[ticker] = pd.DataFrame({
            "earnings_surprise": surprise_vals,
            "days_since_earnings": days_since_vals,
            "earnings_beat_streak": streak_vals,
        }, index=result_dates)

    return lookup


if __name__ == "__main__":
    """Quick test: load earnings and show stats."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8')
    from config import DATA_DIR, UNIVERSE

    print("=" * 60)
    print("  📊 EARNINGS LOADER — TEST")
    print("=" * 60)

    earnings = load_earnings_data(DATA_DIR)
    print(f"\n  Loaded earnings for {len(earnings)} tickers")

    # Show coverage
    covered = [t for t in UNIVERSE if t in earnings]
    missing = [t for t in UNIVERSE if t not in earnings]
    print(f"  Universe coverage: {len(covered)}/{len(UNIVERSE)}")
    if missing:
        print(f"  Missing: {missing}")

    # Show sample
    if "AAPL" in earnings:
        df = earnings["AAPL"]
        print(f"\n  AAPL: {len(df)} reports, {df['date'].min()} → {df['date'].max()}")
        print(f"  Last 3 reports:")
        print(df.tail(3)[["date", "reportedEPS", "estimatedEPS", "surprisePercentage"]].to_string(index=False))

    # Test lookup
    test_dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    lookup = build_earnings_lookup(earnings, ["AAPL", "MSFT"], test_dates)
    if "AAPL" in lookup:
        print(f"\n  AAPL lookup (2020 sample):")
        sample = lookup["AAPL"].dropna().head(5)
        print(sample.to_string())

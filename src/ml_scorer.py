"""
ml_scorer.py — LightGBM-based alpha scoring with purged walk-forward training.

Replaces the fixed-weight scorer.py with a trained ML model that learns
optimal alpha combinations from data.

Architecture:
  - Features: 10 alpha ranks + regime + volatility + volume rank + momentum + earnings
  - Target: Forward 5-day return > 0 (binary classification)
  - Training: Purged expanding-window walk-forward
    → Train on [start ... year_N], gap 63 days, predict [year_N+1]
    → Expands training window each iteration (more data over time)
  - Output: Probability score [0,1] for each stock on each day

References:
  - López de Prado (2018) "Advances in Financial Machine Learning"
    Chapter 7: Cross-Validation in Finance (purged k-fold)
  - Ke et al. (2017) "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lightgbm as lgb
from config import BULL, NEUTRAL, BEAR, CRISIS, DATA_DIR
from src.earnings_loader import load_earnings_data, build_earnings_lookup
import warnings
import json


# ═══════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════

def build_features(alpha_scores, panels, regime):
    """
    Build the feature matrix from alpha scores and market data.

    Features (per stock per day):
      1-10. Alpha rank scores (cross-sectional percentile rank for each alpha)
      11.   Regime (BULL=2, NEUTRAL=1, BEAR=0)
      12.   Realized volatility (20-day rolling std of returns)
      13.   Volume rank (cross-sectional percentile rank of volume)
      14.   Momentum (20-day return)
      15.   Momentum (60-day return)
      16.   RSI-like overbought/oversold (14-day)
      17.   Volume trend (volume / 20-day avg volume)
      18.   Distance from 20-day high (how far from recent high)
      19.   Earnings surprise (last reported surprise %)
      20.   Days since earnings (trading days since last report, clipped 0-90)
      21.   Earnings beat streak (consecutive quarters of positive surprise, clipped 0-5)

    Args:
        alpha_scores: dict of {alpha_name: DataFrame(dates × tickers)}
        panels: dict of {field: DataFrame(dates × tickers)}
        regime: pd.Series (dates → regime label)

    Returns:
        features_df: DataFrame with MultiIndex (date, ticker) and feature columns
        target_df: DataFrame with MultiIndex (date, ticker) and target column
    """
    # Get common dates
    common_dates = regime.index
    for name, scores in alpha_scores.items():
        common_dates = common_dates.intersection(scores.index)
    common_dates = sorted(common_dates)

    close = panels["Close"].loc[common_dates]
    high = panels["High"].loc[common_dates]
    volume = panels["Volume"].loc[common_dates]
    tickers = close.columns

    # ── Earnings features (from Alpha Vantage) ──
    print("  📊 Loading earnings data...")
    earnings_dict = load_earnings_data(DATA_DIR)
    earnings_lookup = build_earnings_lookup(
        earnings_dict, list(tickers),
        [pd.Timestamp(d) for d in common_dates]
    )
    print(f"     Earnings coverage: {len(earnings_lookup)}/{len(tickers)} tickers")

    # ── Alpha features (cross-sectional rank) ──
    alpha_features = {}
    for name, scores in alpha_scores.items():
        ranked = scores.loc[common_dates, tickers].rank(axis=1, pct=True)
        alpha_features[name] = ranked

    # ── Market features ──
    returns_1d = close.pct_change(1)
    returns_5d = close.pct_change(5)
    returns_20d = close.pct_change(20)
    returns_60d = close.pct_change(60)

    # Realized volatility (20-day)
    vol_20d = returns_1d.rolling(20).std()

    # Volume rank (cross-sectional)
    volume_rank = volume.rank(axis=1, pct=True)

    # Volume trend (today / 20-day average)
    vol_avg_20 = volume.rolling(20).mean()
    vol_trend = volume / (vol_avg_20 + 1e-10)

    # RSI-like (14-day): percentage of up days
    up_days = (returns_1d > 0).astype(float).rolling(14).mean()

    # Distance from 20-day high
    high_20d = high.rolling(20).max()
    dist_from_high = (close - high_20d) / (high_20d + 1e-10)

    # ── Regime as numeric ──
    regime_map = {BULL: 2, NEUTRAL: 1, BEAR: 0, CRISIS: -1}
    regime_numeric = regime.loc[common_dates].map(regime_map).fillna(1)

    import config
    from config import TARGET_HORIZON
    horizon = getattr(config, 'TARGET_HORIZON', 5)

    # ── Target: forward N-day return > 0 ──
    fwd_return = close.shift(-horizon) / close - 1
    target = (fwd_return > 0).astype(int)

    # ── Stack into long format ──
    print("  📊 Building feature matrix...")

    records = []
    # Skip first 252 days (warm-up) and last 5 days (no target)
    valid_dates = common_dates[252:-5]

    for date in valid_dates:
        for ticker in tickers:
            row = {"date": date, "ticker": ticker}

            # Alpha features
            for name in alpha_scores.keys():
                val = alpha_features[name].loc[date, ticker]
                if pd.isna(val):
                    break
                row[f"alpha_{name}"] = val
            else:
                # Market features
                row["regime"] = regime_numeric.get(date, 1)
                row["vol_20d"] = vol_20d.loc[date, ticker]
                row["volume_rank"] = volume_rank.loc[date, ticker]
                row["momentum_20d"] = returns_20d.loc[date, ticker]
                row["momentum_60d"] = returns_60d.loc[date, ticker]
                row["rsi_14"] = up_days.loc[date, ticker]
                row["vol_trend"] = vol_trend.loc[date, ticker]
                row["dist_from_high"] = dist_from_high.loc[date, ticker]

                # Earnings features
                if ticker in earnings_lookup:
                    erow = earnings_lookup[ticker]
                    if date in erow.index:
                        row["earnings_surprise"] = erow.loc[date, "earnings_surprise"]
                        row["days_since_earnings"] = erow.loc[date, "days_since_earnings"]
                        row["earnings_beat_streak"] = erow.loc[date, "earnings_beat_streak"]
                    else:
                        row["earnings_surprise"] = np.nan
                        row["days_since_earnings"] = np.nan
                        row["earnings_beat_streak"] = np.nan
                else:
                    row["earnings_surprise"] = np.nan
                    row["days_since_earnings"] = np.nan
                    row["earnings_beat_streak"] = np.nan

                # Target
                t_val = target.loc[date, ticker]
                if pd.isna(t_val):
                    continue
                row["target"] = int(t_val)
                row["fwd_return"] = fwd_return.loc[date, ticker]

                records.append(row)

    df = pd.DataFrame(records)
    print(f"  ✅ Feature matrix: {len(df):,} rows × {len(df.columns) - 3} features")
    print(f"     Date range: {valid_dates[0]} → {valid_dates[-1]}")
    print(f"     Target balance: {df['target'].mean()*100:.1f}% positive")

    nan_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    assert nan_pct < 0.02, (
        f"WARNING: {nan_pct:.1%} NaN in features after join — "
        f"check date alignment between regime series and alpha_scores. "
        f"Regime starts later than price data due to 20-day rolling window."
    )

    return df


# ═══════════════════════════════════════════════
# LIGHTGBM MODEL
# ═══════════════════════════════════════════════

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,    # Conservative — avoid overfitting
    "lambda_l1": 1.0,           # L1 regularization
    "lambda_l2": 1.0,           # L2 regularization
    "max_depth": 6,             # Shallow trees — avoid overfitting
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

FEATURE_COLS = None  # Will be set dynamically


def get_feature_cols(df):
    """Get feature column names from the DataFrame."""
    exclude = {"date", "ticker", "target", "fwd_return"}
    return [c for c in df.columns if c not in exclude]


def train_model(train_df, feature_cols, params=None):
    """
    Train a single LightGBM model on the given training data.

    Returns: trained model
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    X = train_df[feature_cols].values
    y = train_df["target"].values

    train_data = lgb.Dataset(X, label=y, feature_name=feature_cols, free_raw_data=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(period=0)],
        )

    return model


def predict_proba(model, test_df, feature_cols):
    """Predict probability of positive forward return."""
    X = test_df[feature_cols].values
    return model.predict(X)


# ═══════════════════════════════════════════════
# PURGED WALK-FORWARD TRAINING
# ═══════════════════════════════════════════════

def purged_walk_forward_train(features_df, results_dir=None):
    """
    Purged expanding-window walk-forward training.

    For each year Y from 2011 to 2025:
      - Train on: [start ... Y-1] (all available history)
      - Purge: 63 trading days gap (prevent label leakage)
      - Test on: year Y

    This ensures the model NEVER sees future data.

    Args:
        features_df: DataFrame from build_features()
        results_dir: optional path to save results

    Returns:
        predictions_df: DataFrame with columns [date, ticker, prob, target, fwd_return]
        models: list of trained models (one per fold)
        fold_metrics: list of per-fold performance dicts
    """
    feature_cols = get_feature_cols(features_df)

    print(f"\n{'═' * 60}")
    print(f"  🧠 PURGED WALK-FORWARD TRAINING (LightGBM)")
    print(f"{'═' * 60}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Total samples: {len(features_df):,}")
    print(f"  Purge gap: 63 trading days\n")

    # Get unique years
    features_df["year"] = pd.to_datetime(features_df["date"]).dt.year
    all_years = sorted(features_df["year"].unique())

    # Start testing from year 4 onwards (need 3+ years training)
    min_train_years = 3
    test_years = all_years[min_train_years:]

    all_predictions = []
    models = []
    fold_metrics = []
    last_model = None

    print(f"  Train years: {all_years[0]} → varies | Test years: {test_years[0]} → {test_years[-1]}")
    print(f"  {'─' * 55}")

    for test_year in test_years:
        # Training: all data before test_year, minus 63-day purge gap
        train_mask = features_df["year"] < test_year
        test_mask = features_df["year"] == test_year

        train_df = features_df[train_mask].copy()
        test_df = features_df[test_mask].copy()

        if len(train_df) < 500 or len(test_df) < 100:
            print(f"  ⚠️ {test_year}: skip (train={len(train_df)}, test={len(test_df)})")
            continue

        # Purge: remove last 63 days of training (forward return leakage zone)
        train_dates = sorted(train_df["date"].unique())
        if len(train_dates) > 63:
            purge_cutoff = train_dates[-63]
            train_df = train_df[train_df["date"] < purge_cutoff]

        # Train model
        model = train_model(train_df, feature_cols)
        models.append(model)
        last_model = model

        # Predict on test set
        test_df = test_df.copy()
        test_df["prob"] = predict_proba(model, test_df, feature_cols)

        # Enforce strict regime block: 0 probability if regime == BEAR (0) or CRISIS (-1)
        test_df.loc[test_df["regime"] <= 0, "prob"] = 0.0

        # Evaluate: use probability > 0.5 as buy signal
        pred_positive = test_df["prob"] > 0.5
        actual_positive = test_df["target"] == 1

        accuracy = (pred_positive == actual_positive).mean() * 100
        precision = actual_positive[pred_positive].mean() * 100 if pred_positive.sum() > 0 else 0
        auc_approx = test_df.groupby("target")["prob"].mean()

        # Profitability check: avg forward return when predicted positive
        avg_return_when_buy = test_df.loc[pred_positive, "fwd_return"].mean() * 100 if pred_positive.sum() > 0 else 0

        fold_info = {
            "year": int(test_year),
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "accuracy": float(round(accuracy, 1)),
            "precision": float(round(precision, 1)),
            "buy_signals": int(pred_positive.sum()),
            "avg_return_when_buy": float(round(avg_return_when_buy, 2)),
        }
        fold_metrics.append(fold_info)

        print(
            f"  {test_year}: "
            f"Train={len(train_df):>6,} | Test={len(test_df):>5,} | "
            f"Acc={accuracy:.1f}% | Prec={precision:.1f}% | "
            f"Buys={pred_positive.sum():>4d} | "
            f"AvgReturn={avg_return_when_buy:>+.2f}%"
        )

        all_predictions.append(test_df[["date", "ticker", "prob", "target", "fwd_return"]])

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Summary
    print(f"\n  {'═' * 55}")
    overall_pred = predictions_df["prob"] > 0.5
    overall_actual = predictions_df["target"] == 1
    overall_acc = (overall_pred == overall_actual).mean() * 100
    overall_return = predictions_df.loc[overall_pred, "fwd_return"].mean() * 100

    print(f"  📊 OVERALL: Accuracy={overall_acc:.1f}% | "
          f"Avg Return on Buy={overall_return:+.2f}%")

    # Feature importance
    if last_model:
        importance = last_model.feature_importance(importance_type="gain")
        feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
        print(f"\n  🏆 Top Features (by gain):")
        for name, imp in feat_imp[:8]:
            bar = "█" * max(1, int(imp / max(importance) * 20))
            print(f"     {name:<20s} {bar} {imp:.0f}")

    # Save
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        predictions_df.to_csv(os.path.join(results_dir, "ml_predictions.csv"), index=False)
        with open(os.path.join(results_dir, "ml_fold_metrics.json"), "w") as f:
            json.dump(fold_metrics, f, indent=2)
        if last_model:
            feat_imp_dict = {name: float(imp) for name, imp in feat_imp}
            with open(os.path.join(results_dir, "ml_feature_importance.json"), "w") as f:
                json.dump(feat_imp_dict, f, indent=2)

    return predictions_df, models, fold_metrics


# ═══════════════════════════════════════════════
# ML-BASED PORTFOLIO SELECTION
# ═══════════════════════════════════════════════

def ml_select_portfolio(predictions_df, regime, panels=None, top_n=15, rebalance_days=5):
    """
    Build portfolio holdings using ML predictions instead of fixed alpha weights.

    For each rebalance date:
      - Take top_n stocks by ML probability score
      - Weight by probability (higher confidence = larger position)
      - Scale by regime exposure

    Args:
        predictions_df: DataFrame from purged_walk_forward_train()
        regime: pd.Series of regime labels
        panels: unused (kept for API compat)
        top_n: number of stocks to hold
        rebalance_days: how often to rebalance

    Returns:
        holdings: DataFrame (dates × tickers) of portfolio weights
    """
    from config import REGIME_EXPOSURE

    dates = sorted(predictions_df["date"].unique())
    tickers = sorted(predictions_df["ticker"].unique())

    holdings = pd.DataFrame(0.0, index=dates, columns=tickers)

    rebalance_counter = 0
    current_holdings = {}

    for date in dates:
        rebalance_counter += 1
        day_regime = regime.get(date, NEUTRAL)
        exposure = REGIME_EXPOSURE.get(day_regime, 0.5)

        # CRISIS override: force 0% exposure immediately
        if day_regime == CRISIS:
            current_holdings = {}
            holdings.loc[date] = 0.0
            continue

        if rebalance_counter >= rebalance_days:
            rebalance_counter = 0

            # Get today's predictions
            day_preds = predictions_df[predictions_df["date"] == date].copy()

            if len(day_preds) == 0:
                continue

            # Only consider stocks with prob > 0.5 (model thinks positive)
            bullish = day_preds[day_preds["prob"] > 0.5].sort_values("prob", ascending=False)

            if len(bullish) == 0:
                current_holdings = {}
                continue

            # Take top N by probability
            top = bullish.head(top_n)

            # Weight by probability (normalize to sum = exposure)
            probs = top["prob"].values
            weights = probs / probs.sum() * exposure
            current_holdings = dict(zip(top["ticker"], weights))

        # Apply current holdings
        for ticker, weight in current_holdings.items():
            if ticker in holdings.columns:
                holdings.loc[date, ticker] = weight

    # Summary
    avg_positions = (holdings > 0).sum(axis=1).mean()
    avg_exposure = holdings.sum(axis=1).mean()
    print(f"\n  ✅ ML Portfolio: avg {avg_positions:.1f} positions, "
          f"avg {avg_exposure:.1%} exposure")

    return holdings

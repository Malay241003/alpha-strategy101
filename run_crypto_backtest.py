"""
run_crypto_backtest.py — Main entry point for the Crypto Alpha Strategy.

Full pipeline:
  1. Load/download crypto data (Yahoo Finance → cache)
  2. Compute all 10 alphas (same formulas, universal across asset classes)
  3. Compute BTC regime filter (EMA200 + SSRN momentum quality)
  4. Build ML features (alpha ranks + market features)
  5. Purged walk-forward LightGBM training
  6. ML-based portfolio selection
  7. Run portfolio backtest
  8. Run per-stock engine backtest (SL/TP)
  9. Statistical screening
  10. Generate analysis report

NOTE: User runs this manually. Results saved to results_crypto/.
"""

import sys
import os
import json
import warnings
import time

# IMPORTANT: crypto_config must be imported BEFORE config
# We monkey-patch the config module so all imports use crypto values
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override config module with crypto_config values
import crypto_config
import config
# Patch all config values with crypto equivalents
for attr in dir(crypto_config):
    if not attr.startswith('_'):
        setattr(config, attr, getattr(crypto_config, attr))

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", category=FutureWarning)

from crypto_config import (
    RESULTS_DIR, START_DATE, END_DATE, IN_SAMPLE_END, OUT_SAMPLE_START,
    SPY_TICKER, REGIME_TICKER
)
from src.crypto_data_loader import load_all_crypto_data
from src.alphas import compute_all_alphas
from src.crypto_regime_filter import compute_crypto_regime
from src.scorer import compute_composite_score, select_portfolio
from src.backtester import run_backtest
from src.engine import run_engine_all_stocks
from src.ml_scorer import build_features, purged_walk_forward_train, ml_select_portfolio
from src.statistical_screen import compute_per_stock_stats, run_statistical_screen
from src.analysis import generate_full_report

import pandas as pd
import numpy as np


def main():
    """Run the full crypto alpha strategy pipeline with ML scoring."""
    start_time = time.time()

    print("=" * 70)
    print("  🪙 CRYPTO ALPHA STRATEGY — ML PIPELINE (LightGBM)")
    print("=" * 70)
    print(f"  Data Range:  {START_DATE} → {END_DATE}")
    print(f"  In-Sample:   {START_DATE} → {IN_SAMPLE_END}")
    print(f"  Out-Sample:  {OUT_SAMPLE_START} → {END_DATE}")
    print(f"  Regime:      BTC EMA200 + Momentum Quality")
    print(f"  Fees:        CoinDCX Futures (0.118% + spread + slippage)")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Load Data ──
    print("\n📦 STEP 1: Loading crypto data...")
    panels, btc_data, valid_tickers = load_all_crypto_data()

    if len(valid_tickers) < 10:
        print(f"❌ Only {len(valid_tickers)} assets with sufficient history. Need at least 10.")
        return

    # ── Step 2: Compute Alphas ──
    print("\n🧮 STEP 2: Computing alphas...")
    alpha_scores = compute_all_alphas(panels)

    # ── Step 3: BTC Regime Filter ──
    print("\n🏛️ STEP 3: Computing BTC regime filter...")
    regime = compute_crypto_regime(btc_data)

    # ── Step 4: Build ML Features ──
    print("\n📊 STEP 4: Building ML feature matrix...")
    features_df = build_features(alpha_scores, panels, regime)

    # ── Step 5: Purged Walk-Forward LightGBM Training ──
    print("\n🧠 STEP 5: Training LightGBM (purged walk-forward)...")
    predictions_df, models, fold_metrics = purged_walk_forward_train(
        features_df, results_dir=RESULTS_DIR
    )

    # ── Step 6: ML Portfolio Selection ──
    print("\n💼 STEP 6: ML-based portfolio selection...")
    ml_holdings = ml_select_portfolio(predictions_df, regime, panels,
                                      top_n=15, rebalance_days=3)

    # ── Step 7: Run Portfolio Backtest with ML holdings ──
    print("\n📈 STEP 7: Running portfolio backtest with ML signals...")
    results = run_backtest(ml_holdings, panels, regime)

    # Also run rule-based for comparison
    print("\n📊 STEP 7b: Running rule-based backtest for comparison...")
    composite = compute_composite_score(alpha_scores, regime)
    rule_holdings = select_portfolio(composite, regime)
    rule_results = run_backtest(rule_holdings, panels, regime)

    # Compare
    print(f"\n{'═' * 60}")
    print(f"  📊 ML vs RULE-BASED COMPARISON (CRYPTO)")
    print(f"{'═' * 60}")

    def _extract_metrics(res_dict):
        eq = res_dict.get("equity_curve", pd.Series([100000]))
        if isinstance(eq, pd.Series) and len(eq) > 0:
            total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
            daily_r = eq.pct_change().dropna()
            if len(daily_r) > 10 and daily_r.std() > 0:
                sharpe = (daily_r.mean() / daily_r.std()) * (252 ** 0.5)
            else:
                sharpe = 0
            running_max = eq.cummax()
            dd = (eq - running_max) / running_max
            max_dd = dd.min() * 100
            return total_ret, sharpe, max_dd
        return 0, 0, 0

    ml_ret, ml_sharpe, ml_dd = _extract_metrics(results)
    rule_ret, rule_sharpe, rule_dd = _extract_metrics(rule_results)

    print(f"  {'Metric':<20s} {'ML Model':>12s} {'Rule-Based':>12s}")
    print(f"  {'─'*20} {'─'*12} {'─'*12}")
    print(f"  {'Total Return':<20s} {ml_ret:>+11.1f}% {rule_ret:>+11.1f}%")
    print(f"  {'Sharpe Ratio':<20s} {ml_sharpe:>12.2f} {rule_sharpe:>12.2f}")
    print(f"  {'Max Drawdown':<20s} {ml_dd:>11.1f}% {rule_dd:>11.1f}%")

    # ── Step 8: Engine Backtest (per-stock SL/TP) ──
    print("\n🔧 STEP 8: Running per-asset engine backtest...")

    # Override engine config for crypto costs
    from crypto_config import (
        FEE_PCT, SLIPPAGE_PCT, TP_R, SL_ATR_MULT, MAX_BARS_IN_TRADE,
        WARM_UP_BARS, SIGNAL_THRESHOLD, EXIT_THRESHOLD, MIN_SL_PCT
    )
    from src.engine import ENGINE_CONFIG
    ENGINE_CONFIG["FEE_PCT"] = FEE_PCT
    ENGINE_CONFIG["SLIPPAGE_PCT"] = SLIPPAGE_PCT
    ENGINE_CONFIG["TP_R"] = TP_R
    ENGINE_CONFIG["SL_ATR_MULT"] = SL_ATR_MULT
    ENGINE_CONFIG["MAX_BARS_IN_TRADE"] = MAX_BARS_IN_TRADE
    ENGINE_CONFIG["WARM_UP_BARS"] = WARM_UP_BARS
    ENGINE_CONFIG["SIGNAL_THRESHOLD"] = SIGNAL_THRESHOLD
    ENGINE_CONFIG["EXIT_THRESHOLD"] = EXIT_THRESHOLD
    ENGINE_CONFIG["MIN_SL_PCT"] = MIN_SL_PCT

    # Use ML probabilities as composite for engine
    ml_composite = predictions_df.pivot_table(
        index="date", columns="ticker", values="prob", aggfunc="first"
    ).fillna(0)

    # Pass the overrides implicitly through the modified ENGINE_CONFIG
    engine_trades_df, stock_results = run_engine_all_stocks(
        panels, ml_composite, regime
    )

    # ── Step 9: Statistical Screening ──
    print("\n🧪 STEP 9: Running statistical screening...")
    stock_stats = []
    for result in stock_results:
        if result["trades"]:
            stats = compute_per_stock_stats(
                result["ticker"], result["trades"], regime
            )
            if stats:
                stock_stats.append(stats)

    if stock_stats:
        screened_stocks, screen_report = run_statistical_screen(stock_stats)
        with open(os.path.join(RESULTS_DIR, "screened_universe.json"), "w") as f:
            json.dump(screened_stocks, f, indent=2)

    # ── Step 10: Generate Analysis Report ──
    print("\n📊 STEP 10: Generating analysis report...")
    btc_close = btc_data["Close"] if btc_data is not None else None
    generate_full_report(results, benchmark_equity=btc_close)

    # Save engine trades
    if len(engine_trades_df) > 0:
        engine_trades_df.to_csv(
            os.path.join(RESULTS_DIR, "engine_trades_detailed.csv"), index=False
        )

    # Save fold metrics summary
    with open(os.path.join(RESULTS_DIR, "ml_comparison.json"), "w") as f:
        json.dump({
            "ml": {"total_return": round(ml_ret, 2), "sharpe": round(ml_sharpe, 2), "max_dd": round(ml_dd, 2)},
            "rule": {"total_return": round(rule_ret, 2), "sharpe": round(rule_sharpe, 2), "max_dd": round(rule_dd, 2)},
        }, f, indent=2)

    # Save trade summary
    if len(engine_trades_df) > 0:
        from src.engine import compute_stock_metrics
        agg = compute_stock_metrics(engine_trades_df.to_dict("records"))
        if agg:
            with open(os.path.join(RESULTS_DIR, "trade_summary.json"), "w") as f:
                json.dump(agg, f, indent=2)

    # ── Done ──
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total pipeline time: {elapsed:.1f} seconds")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print("=" * 70)
    print("  ✅ CRYPTO ML PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

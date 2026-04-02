"""
run_backtest.py — Main entry point for the Alpha Combination Strategy.

Full pipeline:
  1. Load/download data (Yahoo Finance → cache, full 2008-2025)
  2. Compute all 10 alphas
  3. Compute regime filter
  4. Build ML features (alpha ranks + market features)
  5. Purged walk-forward LightGBM training
  6. ML-based portfolio selection
  7. Run engine backtest (per-stock SL/TP)
  8. Statistical screening (4-stage filter)
  9. Walk-forward validation
  10. Generate analysis report
"""

import sys
import os
import json
import warnings
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", category=FutureWarning)

from config import SPY_TICKER, RESULTS_DIR, START_DATE, END_DATE, IN_SAMPLE_END, OUT_SAMPLE_START
from src.data_loader import load_all_data
from src.alphas import compute_all_alphas
from src.regime_filter import compute_regime
from src.scorer import compute_composite_score, select_portfolio
from src.backtester import run_backtest
from src.engine import run_engine_all_stocks
from src.ml_scorer import build_features, purged_walk_forward_train, ml_select_portfolio
from src.statistical_screen import compute_per_stock_stats, run_statistical_screen
from src.sector_risk_model import run_sector_risk_model
from src.portfolio_optimizer import run_portfolio_optimization
from src.walk_forward import walk_forward_all_stocks
from src.wf_evaluator import evaluate_all_stocks
from src.analysis import generate_full_report

import pandas as pd


def main():
    """Run the full alpha strategy pipeline with ML scoring."""
    start_time = time.time()

    print("=" * 70)
    print("  🧠 ALPHA STRATEGY — ML PIPELINE (LightGBM)")
    print("=" * 70)
    print(f"  Data Range:  {START_DATE} → {END_DATE}")
    print(f"  In-Sample:   {START_DATE} → {IN_SAMPLE_END}")
    print(f"  Out-Sample:  {OUT_SAMPLE_START} → {END_DATE}")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Load Data ──
    print("\n📦 STEP 1: Loading data...")
    panels, market_data = load_all_data()

    # ── Step 2: Compute Alphas ──
    print("\n🧮 STEP 2: Computing alphas...")
    alpha_scores = compute_all_alphas(panels)

    # ── Step 3: Regime Filter ──
    print("\n🏛️ STEP 3: Computing regime filter...")
    regime = compute_regime(market_data)

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
    ml_holdings = ml_select_portfolio(predictions_df, regime, panels, top_n=5, rebalance_days=5)

    # ── Step 7: Run Portfolio Backtest with ML holdings ──
    print("\n📈 STEP 7: Running portfolio backtest with ML signals...")
    results = run_backtest(ml_holdings, panels, regime)

    # Also run the old rule-based for comparison
    print("\n📊 STEP 7b: Running rule-based backtest for comparison...")
    composite = compute_composite_score(alpha_scores, regime)
    rule_holdings = select_portfolio(composite, regime)
    rule_results = run_backtest(rule_holdings, panels, regime)

    # Compare
    print(f"\n{'═' * 60}")
    print(f"  📊 ML vs RULE-BASED COMPARISON")
    print(f"{'═' * 60}")

    def _extract_metrics(res_dict):
        eq = res_dict.get("equity_curve", pd.Series([100000]))
        if isinstance(eq, pd.Series) and len(eq) > 0:
            total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
            # Simplified Sharpe from daily returns
            daily_r = eq.pct_change().dropna()
            if len(daily_r) > 10 and daily_r.std() > 0:
                sharpe = (daily_r.mean() / daily_r.std()) * (252 ** 0.5)
            else:
                sharpe = 0
            # Max drawdown
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

    # ── Step 8: Engine Backtest (per-stock SL/TP) with ML signals ──
    print("\n🔧 STEP 8: Running per-stock engine backtest...")
    # Use ML probability as score for engine
    ml_composite = predictions_df.pivot_table(
        index="date", columns="ticker", values="prob", aggfunc="first"
    ).fillna(0)

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
    spy_close = market_data.get(SPY_TICKER)
    benchmark = spy_close["Close"] if spy_close is not None else None
    generate_full_report(results, benchmark_equity=benchmark)

    # Save engine trades
    if len(engine_trades_df) > 0:
        engine_trades_df.to_csv(
            os.path.join(RESULTS_DIR, "engine_trades_detailed.csv"), index=False
        )

    # ── Done ──
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total pipeline time: {elapsed:.1f} seconds")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print("=" * 70)
    print("  ✅ ML PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

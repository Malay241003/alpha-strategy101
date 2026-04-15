"""
run_regime_comparison.py — Head-to-head: Heuristic vs Markov regime in the ML pipeline.

Runs the FULL ML pipeline (LightGBM walk-forward) TWICE:
  1) With heuristic regime (MA + VIX) — your current production model
  2) With Markov Chain HMM regime — the new probabilistic model

Train/Test split:
  Train (calibration): 2008 → 2016  (Markov calibration + ML walk-forward)
  Test  (backtest):    2017 → now   (equity curves start here)

Usage:
    python run_regime_comparison.py
"""

import sys
import os
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    SPY_TICKER, RESULTS_DIR, START_DATE, END_DATE,
    INITIAL_CAPITAL,
    BULL, NEUTRAL, BEAR, CRISIS, REGIME_EXPOSURE
)
from src.data_loader import load_all_data
from src.alphas import compute_all_alphas
from src.regime_filter import compute_regime
from src.markov_regime import compute_regime_markov
from src.ml_scorer import build_features, purged_walk_forward_train, ml_select_portfolio
from src.backtester import run_backtest

# ─── TRAIN / TEST SPLIT ───
TRAIN_END  = "2016-12-31"
TEST_START = "2017-01-01"


def compute_metrics(equity_curve, daily_returns, label=""):
    """Compute performance metrics from an equity curve."""
    if equity_curve is None or len(equity_curve) < 10:
        return {}

    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    n_years = len(equity_curve) / 252
    cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1) * 100

    daily_r = daily_returns.dropna()
    if len(daily_r) > 10 and daily_r.std() > 0:
        sharpe = (daily_r.mean() / daily_r.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    downside = daily_r[daily_r < 0]
    if len(downside) > 5:
        sortino = (daily_r.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    max_dd = dd.min() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "label": label,
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "final_equity": equity_curve.iloc[-1],
    }


def regime_stats(regime_series, label=""):
    """Compute regime distribution and transition statistics."""
    counts = regime_series.value_counts()
    total = len(regime_series)
    transitions = (regime_series != regime_series.shift(1)).sum() - 1

    stats = {"label": label, "total_days": total, "transitions": transitions}
    for r in [BULL, NEUTRAL, BEAR, CRISIS]:
        n = counts.get(r, 0)
        stats[f"{r.lower()}_days"] = n
        stats[f"{r.lower()}_pct"] = n / total * 100

    stats["avg_regime_duration"] = total / max(transitions, 1)
    return stats


def plot_comparison(results_heur, results_markov, regime_heur, regime_markov,
                    exposure_markov, benchmark, save_dir):
    """Generate comparison charts and save to results directory."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), facecolor='#0d1117')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, alpha=0.15, color='#30363d', linestyle='--')

    # ─── Chart 1: Equity Curves ───
    ax1 = axes[0]
    eq_h = results_heur["equity_curve"]
    eq_m = results_markov["equity_curve"]

    ax1.plot(eq_h.index, eq_h.values, color='#f97316', linewidth=1.5,
             label='Heuristic (MA+VIX)', alpha=0.9)
    ax1.plot(eq_m.index, eq_m.values, color='#3b82f6', linewidth=1.5,
             label='Markov Chain HMM', alpha=0.9)
    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0] * INITIAL_CAPITAL
        bench_common = bench_norm.reindex(eq_h.index, method='ffill').dropna()
        ax1.plot(bench_common.index, bench_common.values, color='#6b7280',
                 linewidth=1, label='SPY (buy & hold)', alpha=0.6, linestyle='--')

    ax1.set_title(f'ML Equity Curves — Test Period ({TEST_START} onwards)', color='#c9d1d9',
                   fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', color='#8b949e')
    ax1.legend(loc='upper left', facecolor='#21262d', edgecolor='#30363d',
               labelcolor='#c9d1d9')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # ─── Chart 2: Regime Timeline ───
    ax2 = axes[1]
    regime_numeric_h = regime_heur.map({BULL: 3, NEUTRAL: 2, BEAR: 1, CRISIS: 0})
    regime_numeric_m = regime_markov.map({BULL: 3, NEUTRAL: 2, BEAR: 1, CRISIS: 0})
    common_dates = regime_numeric_h.index.intersection(regime_numeric_m.index)

    ax2.fill_between(common_dates, 0, 2, where=(regime_heur.loc[common_dates] == BULL),
                     color='#3fb950', alpha=0.3, label='BULL')
    ax2.fill_between(common_dates, 0, 2, where=(regime_heur.loc[common_dates] == NEUTRAL),
                     color='#d29922', alpha=0.3, label='NEUTRAL')
    ax2.fill_between(common_dates, 0, 2, where=(regime_heur.loc[common_dates] == BEAR),
                     color='#f85149', alpha=0.3, label='BEAR')
    ax2.fill_between(common_dates, 0, 2, where=(regime_heur.loc[common_dates] == CRISIS),
                     color='#8b5cf6', alpha=0.3, label='CRISIS')

    rh = regime_numeric_h.loc[common_dates]
    rm = regime_numeric_m.loc[common_dates]
    ax2.plot(common_dates, rh.values * 2 / 3 + 1, color='#f97316', linewidth=0.8,
             alpha=0.8, label='Heuristic')
    ax2.plot(common_dates, rm.values * 2 / 3 + 1, color='#3b82f6', linewidth=0.8,
             alpha=0.8, label='Markov')

    ax2.set_title('Regime Classification (Test Period)', color='#c9d1d9',
                   fontsize=14, fontweight='bold')
    ax2.set_ylabel('Regime', color='#8b949e')
    ax2.set_yticks([0.33, 1.0, 1.67, 2.33])
    ax2.set_yticklabels(['CRISIS', 'BEAR', 'NEUTRAL', 'BULL'])
    ax2.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d',
               labelcolor='#c9d1d9', fontsize=8, ncol=3)

    # ─── Chart 3: Exposure Score ───
    ax3 = axes[2]
    if exposure_markov is not None:
        ax3.fill_between(exposure_markov.index, 0, exposure_markov.values,
                         color='#3b82f6', alpha=0.3)
        ax3.plot(exposure_markov.index, exposure_markov.values,
                 color='#3b82f6', linewidth=0.8, alpha=0.8)
        heur_exposure = regime_heur.map(REGIME_EXPOSURE)
        ax3.step(heur_exposure.index, heur_exposure.values, color='#f97316',
                 linewidth=1.0, alpha=0.8, where='post', label='Heuristic (step)')
        ax3.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d',
                   labelcolor='#c9d1d9')

    ax3.set_title('Exposure Score — Markov (smooth) vs Heuristic (step)',
                   color='#c9d1d9', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Exposure (0=cash, 1=full)', color='#8b949e')
    ax3.set_ylim(-0.05, 1.05)

    # ─── Chart 4: Drawdown ───
    ax4 = axes[3]
    for res, color, label in [
        (results_heur, '#f97316', 'Heuristic'),
        (results_markov, '#3b82f6', 'Markov')
    ]:
        eq = res["equity_curve"]
        running_max = eq.cummax()
        dd = (eq - running_max) / running_max * 100
        ax4.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color)
        ax4.plot(dd.index, dd.values, color=color, linewidth=0.8, label=label)

    ax4.set_title('Drawdown Comparison', color='#c9d1d9', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Drawdown (%)', color='#8b949e')
    ax4.legend(loc='lower left', facecolor='#21262d', edgecolor='#30363d',
               labelcolor='#c9d1d9')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))

    plt.tight_layout(pad=2.0)
    save_path = os.path.join(save_dir, "regime_comparison.png")
    fig.savefig(save_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close(fig)
    print(f"\n📊 Comparison chart saved to: {save_path}")


def run_ml_pipeline(panels, alpha_scores, regime, label=""):
    """
    Run the FULL ML pipeline with a given regime filter.

    This mirrors exactly what run_backtest.py does:
      1. build_features (regime is a feature)
      2. purged_walk_forward_train (LightGBM)
      3. ml_select_portfolio (regime controls exposure)
      4. run_backtest (portfolio simulation)

    Returns: backtest results dict
    """
    print(f"\n  📊 Building ML features with {label} regime...")
    features_df = build_features(alpha_scores, panels, regime)

    print(f"\n  🧠 Training LightGBM (walk-forward) with {label} regime...")
    predictions_df, models, fold_metrics = purged_walk_forward_train(features_df)

    print(f"\n  💼 ML portfolio selection with {label} regime...")
    ml_holdings = ml_select_portfolio(predictions_df, regime, panels, top_n=5, rebalance_days=5)

    print(f"\n  📈 Running backtest with {label} regime...")
    results = run_backtest(ml_holdings, panels, regime)

    return results, predictions_df, fold_metrics


def main():
    """Run head-to-head ML comparison with strict train/test split."""
    start_time = time.time()

    print("=" * 70)
    print("  ⚔️  ML REGIME COMPARISON: Heuristic vs Markov Chain HMM")
    print("=" * 70)
    print(f"  Data Range:    {START_DATE} → {END_DATE}")
    print(f"  Train Period:  {START_DATE} → {TRAIN_END}  (Markov calibration)")
    print(f"  Test Period:   {TEST_START} → {END_DATE}   (equity curves)")
    print(f"  ML Model:      LightGBM (purged walk-forward)")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Load Data ──
    print("\n📦 STEP 1: Loading data...")
    panels, market_data = load_all_data()

    # ── Step 2: Compute Alphas ──
    print("\n🧮 STEP 2: Computing alphas...")
    alpha_scores = compute_all_alphas(panels)

    # ── Step 3a: Heuristic Regime ──
    print("\n" + "─" * 70)
    print("  📐 METHOD A: Heuristic Regime (MA + VIX thresholds)")
    print("─" * 70)
    regime_heur = compute_regime(market_data)

    # ── Step 3b: Markov Chain Regime ──
    print("\n" + "─" * 70)
    print("  🔮 METHOD B: Markov Chain HMM Regime")
    print(f"      Calibration: {START_DATE} → {TRAIN_END}")
    print("─" * 70)
    regime_markov_full, exposure_markov_full = compute_regime_markov(
        market_data, calibration_end=TRAIN_END
    )

    # ── Step 4: Run FULL ML pipeline with EACH regime ──
    print("\n" + "═" * 70)
    print("  🧠 PIPELINE A: ML + Heuristic Regime")
    print("═" * 70)
    results_heur, preds_h, folds_h = run_ml_pipeline(
        panels, alpha_scores, regime_heur, label="Heuristic"
    )

    print("\n" + "═" * 70)
    print("  🧠 PIPELINE B: ML + Markov Regime")
    print("═" * 70)
    results_markov, preds_m, folds_m = run_ml_pipeline(
        panels, alpha_scores, regime_markov_full, label="Markov"
    )

    # ── Step 5: Filter equity curves to TEST PERIOD ONLY ──
    test_start_ts = pd.Timestamp(TEST_START)

    eq_h = results_heur["equity_curve"]
    eq_m = results_markov["equity_curve"]
    dr_h = results_heur["daily_returns"]
    dr_m = results_markov["daily_returns"]

    # Filter to test period
    eq_h_test = eq_h[eq_h.index >= test_start_ts]
    eq_m_test = eq_m[eq_m.index >= test_start_ts]
    dr_h_test = dr_h[dr_h.index >= test_start_ts]
    dr_m_test = dr_m[dr_m.index >= test_start_ts]

    # Normalize both to start at INITIAL_CAPITAL at test start
    if len(eq_h_test) > 0 and len(eq_m_test) > 0:
        h_start = eq_h_test.iloc[0]
        m_start = eq_m_test.iloc[0]
        eq_h_test = eq_h_test / h_start * INITIAL_CAPITAL
        eq_m_test = eq_m_test / m_start * INITIAL_CAPITAL

        # Replace in results for charting
        results_heur_test = dict(results_heur)
        results_heur_test["equity_curve"] = eq_h_test
        results_heur_test["daily_returns"] = dr_h_test

        results_markov_test = dict(results_markov)
        results_markov_test["equity_curve"] = eq_m_test
        results_markov_test["daily_returns"] = dr_m_test
    else:
        results_heur_test = results_heur
        results_markov_test = results_markov

    # ── Step 6: Compute Metrics (test period only) ──
    metrics_h = compute_metrics(eq_h_test, dr_h_test, "Heuristic")
    metrics_m = compute_metrics(eq_m_test, dr_m_test, "Markov HMM")

    # ── Step 7: Regime Stats (test period) ──
    regime_heur_test = regime_heur[regime_heur.index >= test_start_ts]
    regime_markov_test = regime_markov_full[regime_markov_full.index >= test_start_ts]
    exposure_markov_test = exposure_markov_full[exposure_markov_full.index >= test_start_ts] \
        if exposure_markov_full is not None else None

    rstats_h = regime_stats(regime_heur_test, "Heuristic")
    rstats_m = regime_stats(regime_markov_test, "Markov")

    # Align for agreement calc
    common = regime_heur_test.index.intersection(regime_markov_test.index)
    agreement = (regime_heur_test.loc[common] == regime_markov_test.loc[common]).mean() * 100

    # ═══ PRINT RESULTS ═══

    print("\n" + "═" * 70)
    print(f"  ⚔️  ML PERFORMANCE — TEST ({TEST_START} → {END_DATE})")
    print("═" * 70)
    print(f"  {'Metric':<25s} {'Heuristic':>15s} {'Markov HMM':>15s} {'Delta':>12s}")
    print(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*12}")

    comparisons = [
        ("Total Return",     "total_return",  "%",  1),
        ("CAGR",             "cagr",          "%",  1),
        ("Sharpe Ratio",     "sharpe",        "",   2),
        ("Sortino Ratio",    "sortino",       "",   2),
        ("Max Drawdown",     "max_drawdown",  "%",  1),
        ("Calmar Ratio",     "calmar",        "",   2),
        ("Final Equity",     "final_equity",  "$",  0),
    ]

    for name, key, unit, decimals in comparisons:
        h_val = metrics_h.get(key, 0)
        m_val = metrics_m.get(key, 0)
        delta = m_val - h_val
        if unit == "$":
            print(f"  {name:<25s} ${h_val:>13,.0f} ${m_val:>13,.0f} ${delta:>+10,.0f}")
        elif unit == "%":
            print(f"  {name:<25s} {h_val:>14.{decimals}f}% {m_val:>14.{decimals}f}% {delta:>+11.{decimals}f}%")
        else:
            print(f"  {name:<25s} {h_val:>15.{decimals}f} {m_val:>15.{decimals}f} {delta:>+12.{decimals}f}")

    # Regime Distribution
    print(f"\n{'═' * 70}")
    print(f"  📊 REGIME DISTRIBUTION (Test Period)")
    print(f"{'═' * 70}")
    print(f"  {'Regime':<10s} {'Heuristic':>18s} {'Markov HMM':>18s}")
    print(f"  {'─'*10} {'─'*18} {'─'*18}")
    for r in [BULL, NEUTRAL, BEAR, CRISIS]:
        h_d = rstats_h.get(f"{r.lower()}_days", 0)
        h_p = rstats_h.get(f"{r.lower()}_pct", 0)
        m_d = rstats_m.get(f"{r.lower()}_days", 0)
        m_p = rstats_m.get(f"{r.lower()}_pct", 0)
        print(f"  {r:<10s} {h_d:>8d} ({h_p:>5.1f}%) {m_d:>8d} ({m_p:>5.1f}%)")

    print(f"\n  Transitions:  Heuristic={rstats_h['transitions']}, "
          f"Markov={rstats_m['transitions']}")
    print(f"  Avg duration: Heuristic={rstats_h['avg_regime_duration']:.0f} days, "
          f"Markov={rstats_m['avg_regime_duration']:.0f} days")
    print(f"  Agreement:    {agreement:.1f}% of days same regime")

    # ── Step 8: Charts ──
    print("\n🎨 Generating comparison charts...")
    spy_data = market_data.get(SPY_TICKER)
    benchmark = spy_data["Close"] if spy_data is not None else None
    if benchmark is not None:
        benchmark = benchmark[benchmark.index >= test_start_ts]

    plot_comparison(
        results_heur_test, results_markov_test,
        regime_heur_test, regime_markov_test,
        exposure_markov_test, benchmark,
        RESULTS_DIR
    )

    comparison_df = pd.DataFrame([metrics_h, metrics_m])
    comparison_df.to_csv(
        os.path.join(RESULTS_DIR, "regime_comparison_metrics.csv"), index=False
    )

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total comparison time: {elapsed:.1f} seconds")
    print("=" * 70)
    print("  ✅ ML REGIME COMPARISON COMPLETE")
    print("=" * 70)

    return {
        "heuristic": {"metrics": metrics_h, "regime": regime_heur, "results": results_heur},
        "markov": {"metrics": metrics_m, "regime": regime_markov_full, "results": results_markov,
                   "exposure": exposure_markov_full},
    }


if __name__ == "__main__":
    main()

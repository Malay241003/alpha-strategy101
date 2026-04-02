"""
run_fx_macro_strategy.py — Main runner for the Forex Systematic Macro Pipeline.

Executes:
  1. Fetch 20+ years of Daily FX Data via yfinance
  2. Compute Multi-horizon Macro Features
  3. Prepare Target (>0.05% forward 5-Day return)
  4. Train LightGBM via Purged Walk-Forward (1 Year chunks)
  5. Backtest Daily Engine with typical retail spread.
"""

import sys, os
import pandas as pd
import warnings
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

# Importers
from fx_config import RESULTS_DIR, FX_UNIVERSE
from src.fx_data_loader import fetch_yfinance_forex_data
from src.fx_features import build_fx_features
from src.fx_ml_model import prepare_fx_ml_data, fx_purged_walk_forward
from src.fx_engine import run_fx_engine

def run_fx_pipeline():
    start_time = time.time()
    
    print("=" * 70)
    print("  🚀 FX SYSTEMATIC MACRO STRATEGY (NLTSMOM)")
    print("=" * 70)
    print("  Data:       Yahoo Finance (Decades of Daily History)")
    print("  Features:   Multi-horizon Momentum, Rolling Volatility, XS Rank")
    print("  Horizon:    1-Week (5 Trading Days) Forward Return")
    print("  Costs:      1.5 Pip Spread Round-Trip")
    print("=" * 70)
    
    # ── 1. Fetch Data ──
    panels, valid_tickers = fetch_yfinance_forex_data(FX_UNIVERSE)
    if not valid_tickers:
        print("❌ Data fetching failed.")
        return
        
    # ── 2. Feature Engineering ──
    features_df = build_fx_features(panels)
    
    # ── 3. ML Target Prep ──
    ml_df = prepare_fx_ml_data(features_df, panels["Close"])
    
    # ── 4. Train LightGBM Walk-Forward ──
    preds_df, models, metrics = fx_purged_walk_forward(ml_df)
    
    # ── 5. Backtest Engine ──
    trades_df = run_fx_engine(panels, preds_df)
    
    # ── 6. Final Summary ──
    if not trades_df.empty:
        path = os.path.join(RESULTS_DIR, "fx_macro_trades.csv")
        trades_df.to_csv(path, index=False)
        print(f"\n💾 Saved FX trade logs: {path}")
        
    print(f"\n⏱️ Pipeline completed in {int(time.time() - start_time)} seconds.")
    print("=" * 70)

if __name__ == "__main__":
    run_fx_pipeline()

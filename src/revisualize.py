"""
revisualize.py — Manually regenerate all visualization plots from the latest backtest results.
"""

import os
import pandas as pd
import numpy as np
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, SPY_TICKER, NEUTRAL, CRISIS, BULL, BEAR, TOP_N_STOCKS
from src.data_loader import load_all_data
from src.regime_filter import compute_regime
from src.ml_scorer import ml_select_portfolio
from src.backtester import run_backtest
from src.analysis import generate_full_report

def main():
    print("🎨 Manually regenerating visuals...")
    
    # 1. Load Data
    panels, market_data = load_all_data(force=False)
    regime = compute_regime(market_data)
    
    # 2. Load ML Predictions
    preds_path = os.path.join(RESULTS_DIR, "ml_predictions.csv")
    if not os.path.exists(preds_path):
        print(f"❌ Cannot find {preds_path}")
        return
        
    print(f"📂 Loading predictions from {preds_path}...")
    predictions_df = pd.read_csv(preds_path, parse_dates=['date'])
    
    # 3. Re-simulate Portfolio (Fast)
    print(f"📈 Simulating portfolio holdings (Top {TOP_N_STOCKS} stocks) from predictions...")
    holdings = ml_select_portfolio(predictions_df, regime, top_n=TOP_N_STOCKS)
    results = run_backtest(holdings, panels, regime)
    
    # 4. Generate Report
    print("📊 Calling generate_full_report...")
    spy_close = market_data.get(SPY_TICKER)
    benchmark = spy_close["Close"] if spy_close is not None else None
    
    generate_full_report(results, benchmark_equity=benchmark)
    
    print("\n✨ All visuals have been updated in the results/ folder.")

if __name__ == "__main__":
    main()

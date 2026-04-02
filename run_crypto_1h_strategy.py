"""
run_crypto_1h_strategy.py — Main runner for the Crypto-Native 1H Strategy.

This script executes the following pipeline:
  1. Fetch/Load Binance 1H data (Top 25 liquid pairs for speed)
  2. Compute Crypto-Native Microstructure Features (Order Flow, Liquidity Shocks)
  3. Prepare ML target (>0.3% next 12H return)
  4. Train LightGBM via Purged Walk-Forward (1 Year chunks)
  5. Backtest intra-candle (1H) execution engine with complete CoinDCX costs.
"""

import sys, os
import pandas as pd
import warnings
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

# Importers
from crypto_config import DATA_DIR, RESULTS_DIR, UNIVERSE_TICKERS
from src.crypto_binance_loader import fetch_binance_universe_data, get_binance_symbol
from src.crypto_features import build_crypto_features
from src.crypto_ml_model import prepare_ml_data, crypto_purged_walk_forward
from src.crypto_engine_1h import run_crypto_1h_engine

def run_crypto_pipeline():
    start_time = time.time()
    
    print("=" * 70)
    print("  🚀 CRYPTO-NATIVE ML STRATEGY (1H TIMEFRAME)")
    print("=" * 70)
    print("  Data:       Binance direct API")
    print("  Features:   Order Flow Imbalance, Volatility, Momentum")
    print("  Horizon:    12-Hour Forward Return")
    print("  Costs:      CoinDCX Futures Tier (0.118% x 2 + Spread)")
    print("=" * 70)

    # Output Dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Use top 15 most liquid tokens from universe to ensure we have long matching history 
    # (Binance maintains consistent history for large caps since 2020)
    target_tokens = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", 
        "ADAUSDT", "DOGEUSDT", "TRXUSDT", "LINKUSDT", "MATICUSDT",
        "LTCUSDT", "BCHUSDT", "ATOMUSDT", "AVAXUSDT", "DOTUSDT"
    ]
    
    # ── 1. Fetch Binance 1H Data (2020-2025) ──
    print(f"\n📦 STEP 1: Fetching 5 Years of 1H Data ({len(target_tokens)} Assets)...")
    start_str = "2020-01-01"  # 5 years of history (2020 to 2025)
    end_str = "2025-01-01"
    
    panels, valid_tickers = fetch_binance_universe_data(
        target_tokens, start_str, end_str, DATA_DIR, "1h"
    )
    
    if len(valid_tickers) < 2:
        print("❌ Not enough valid data downloaded. Aborting.")
        return
        
    # Get BTC specifically for regime logic
    btc_df = panels["Close"][["BTCUSDT"]] if "BTCUSDT" in panels["Close"] else None
    
    # ── 2. Feature Engineering ──
    print("\n📊 STEP 2: Creating Crypto Microstructure Features...")
    btc_panels = {"Close": btc_df} if btc_df is not None else None
    features_df = build_crypto_features(panels, btc_panels)
    
    # ── 3. ML Target Prep ──
    print("\n🎯 STEP 3: Preparing ML Target Variable...")
    # Target = 12H forward return > 0.3%
    ml_df = prepare_ml_data(features_df, panels["Close"], horizon=12)
    
    # ── 4. Train LightGBM Walk-Forward ──
    print("\n🧠 STEP 4: Purged Walk-Forward Training...")
    preds_df, models, metrics = crypto_purged_walk_forward(ml_df, RESULTS_DIR)
    
    # ── 5. Backtest 1H Engine ──
    print("\n⚙️ STEP 5: Running Engine Backtest with Fees...")
    trades_df = run_crypto_1h_engine(panels, preds_df, btc_panels)
    
    # ── 6. Final Summary ──
    if not trades_df.empty:
        # Save exact trades for inspection
        path = os.path.join(RESULTS_DIR, "crypto_1h_trades.csv")
        trades_df.to_csv(path, index=False)
        print(f"\n💾 Saved 1H trade logs: {path}")
        
    print(f"\n⏱️ Pipeline completed in {int(time.time() - start_time)} seconds.")
    print("=" * 70)

if __name__ == "__main__":
    run_crypto_pipeline()

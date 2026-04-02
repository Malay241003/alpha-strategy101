"""
fx_config.py — Configuration settings for the Forex Systematic Macro Pipeline.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_DIR = os.path.join(BASE_DIR, "macro")
DATA_DIR = os.path.join(MACRO_DIR, "data")
RESULTS_DIR = os.path.join(MACRO_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Major USD Pairs (High Liquidity, Deep History) ──
# Note: yfinance format for Forex is BaseCurrency + QuoteCurrency + "=X"
FX_UNIVERSE = [
    "EURUSD=X",  # Euro
    "GBPUSD=X",  # British Pound
    "JPY=X",     # Japanese Yen (Note: USDJPY)
    "AUDUSD=X",  # Australian Dollar
    "NZDUSD=X",  # New Zealand Dollar
    "CAD=X",     # Canadian Dollar (Note: USDCAD)
    "CHF=X",     # Swiss Franc (Note: USDCHF)
]

# ── Data & Modeling Parameters ──
DATA_START_DATE = "2000-01-01"  # 25+ years of data for deep macro context
TARGET_HORIZON_DAYS = 5         # Predict 1-week forward return sign

# ── 1D Engine Constraints ──
ENGINE_CONFIG_FX = {
    "TP_R": 2.0,                  # Take-profit in R
    "SL_ATR_MULT": 1.5,           # Stop-loss buffer
    "ATR_PERIOD": 14,             # Daily ATR
    "MAX_BARS_IN_TRADE": 20,      # Max hold roughly 1 trading month
    "SPREAD_PIPS": 1.5,           # Typical retail spread for majors (in pips)
    "RISK_PCT": 1.0,              # Risk 1% per trade
}

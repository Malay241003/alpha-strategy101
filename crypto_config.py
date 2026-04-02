"""
crypto_config.py — Configuration for the Crypto Alpha Strategy.

Universe: Binance × CoinDCX intersection (from tradeBot's fetchTop100Universe.js)
Parameters: Ported from tradeBot's backtest/config.js
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "crypto")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_crypto")

# ─────────────────────────────────────────────
# Date Range
# More data = better ML training. Many alts only go back to 2017-2018.
# ─────────────────────────────────────────────
START_DATE = "2017-01-01"
END_DATE = "2025-12-31"

# In-sample / Out-of-sample split
IN_SAMPLE_END = "2022-12-31"    # 2017-2022 = in-sample (train)
OUT_SAMPLE_START = "2023-01-01"  # 2023-2025 = out-of-sample (validate)

# Minimum history required for a coin (in trading days)
# Coins with less history are excluded from the universe
MIN_HISTORY_DAYS = 500  # ~2 years

# ─────────────────────────────────────────────
# Crypto Universe — Binance ∩ CoinDCX Top 100
# Source: tradeBot/bot/universes/crypto_top100.js
# Generated: 2026-02-24 via fetchTop100Universe.js
# ─────────────────────────────────────────────
UNIVERSE = [
    # ── Top 20 by volume ──
    "BTC", "ETH", "SOL", "XRP", "BNB",
    "DOGE", "BCH", "TRX", "PEPE", "ZEC",
    "SUI", "ADA", "LINK", "AVAX", "LTC",
    "AAVE", "UNI", "TAO", "ENA", "WLD",

    # ── 21-40 ──
    "ARB", "NEAR", "HBAR", "FIL", "DOT",
    "SEI", "ICP", "DASH", "INJ", "APT",
    "POL", "CAKE", "TON", "WIF", "EIGEN",

    # ── 41-60 ──
    "XLM", "ONDO", "SHIB", "UMA", "ETC",
    "FET", "CRV", "BONK", "SNX", "ZK",
    "RENDER", "CHZ", "PENDLE", "ATOM", "AR",

    # ── 61-80 ──
    "COTI", "TIA", "AXS", "YGG", "VET",
    "ZEN", "FLOKI", "VTHO", "STORJ", "RUNE",
    "ROSE", "JTO", "DUSK", "GAS", "LPT",
]

# yfinance tickers: append "-USD" to each
UNIVERSE_TICKERS = [f"{coin}-USD" for coin in UNIVERSE]

# BTC as regime proxy (replaces SPY in stock strategy)
REGIME_TICKER = "BTC-USD"

# ─────────────────────────────────────────────
# Regime Filter — from tradeBot's engine.js
# TradeBot uses 1H EMA200 macro filter:
#   LONG: close > EMA200
#   SHORT: close < EMA200
# We use daily BTC SMA to define 3 regimes
# ─────────────────────────────────────────────
BULL = "BULL"
NEUTRAL = "NEUTRAL"
BEAR = "BEAR"

# BTC regime: EMA-based (matching tradeBot's approach)
REGIME_FAST_PERIOD = 50     # Short-term trend
REGIME_SLOW_PERIOD = 200    # Long-term trend (tradeBot's EMA200)

# ─────────────────────────────────────────────
# Regime Exposure — how much capital to deploy per regime
# ─────────────────────────────────────────────
REGIME_EXPOSURE = {
    BULL:    1.0,    # Full exposure in bull market
    NEUTRAL: 0.5,   # Half exposure in neutral
    BEAR:    0.0,    # No new entries in bear (matching tradeBot)
}

# ─────────────────────────────────────────────
# Alpha Weights (for rule-based fallback — ML overrides these)
# ─────────────────────────────────────────────
ALPHA_WEIGHTS = {
    "alpha_101": {BULL: 0.10, NEUTRAL: 0.10, BEAR: 0.05},
    "alpha_9":   {BULL: 0.08, NEUTRAL: 0.12, BEAR: 0.15},
    "alpha_32":  {BULL: 0.12, NEUTRAL: 0.08, BEAR: 0.05},
    "alpha_23":  {BULL: 0.15, NEUTRAL: 0.15, BEAR: 0.15},
    "alpha_52":  {BULL: 0.12, NEUTRAL: 0.10, BEAR: 0.08},
    "alpha_7":   {BULL: 0.10, NEUTRAL: 0.10, BEAR: 0.10},
    "alpha_37":  {BULL: 0.08, NEUTRAL: 0.08, BEAR: 0.10},
    "alpha_36":  {BULL: 0.10, NEUTRAL: 0.10, BEAR: 0.12},
    "alpha_19":  {BULL: 0.10, NEUTRAL: 0.10, BEAR: 0.10},
    "alpha_12":  {BULL: 0.05, NEUTRAL: 0.07, BEAR: 0.10},
}

TOP_N_STOCKS = 15   # Actually top_n crypto now

# ─────────────────────────────────────────────
# Portfolio & Risk — from tradeBot's config.js
# ─────────────────────────────────────────────
INITIAL_CAPITAL = 100000
RISK_PER_TRADE_PCT = 0.5     # 0.5% per trade (from tradeBot PROP_FIRM config)
REBALANCE_EVERY_N_DAYS = 3   # Crypto moves faster, rebalance more often
COMMISSION_PER_SHARE = 0.0   # N/A for crypto, using percentage fees

# ─────────────────────────────────────────────
# Costs — from tradeBot's backtest/config.js
# CoinDCX Futures (USDT-M) fee structure
# ─────────────────────────────────────────────
FEE_PCT = 0.00118             # 0.118% commission per side
SPREAD_PCT = 0.0010           # 0.10% spread
SLIPPAGE_PCT = 0.0008         # 0.08% slippage per side
FUNDING_PER_8H = 0.0001       # 0.01% funding rate per 8 hours
MIN_SL_PCT = 0.003            # 0.30% min stop distance (from tradeBot)

# ─────────────────────────────────────────────
# Engine — from tradeBot's backtest/config.js (Adjusted for daily ML)
# ─────────────────────────────────────────────
TP_R = 3.0                    # Take-profit in R (crypto is volatile, 3:1 is plenty for daily)
SL_ATR_MULT = 3.5             # SL buffer (wider for daily candles to avoid noise)
MAX_BARS_IN_TRADE = 90        # Daily bars = ~3 months (tradeBot uses 672 x 15m = 7 days)
ATR_PERIOD = 14               # Standard ATR period
WARM_UP_BARS = 200            # Need EMA200 warm-up (matching tradeBot)
SIGNAL_THRESHOLD = 0.75       # Top 25% score to enter
EXIT_THRESHOLD = 0.35         # Exit below 35th percentile

# Market Hours — crypto is 24/7
MARKET_HOURS_FILTER = False   # No market hours filter for crypto

# ─────────────────────────────────────────────
# SPY Ticker — replaced by BTC for crypto
# ─────────────────────────────────────────────
SPY_TICKER = "BTC-USD"  # Used as market proxy for benchmarking

# ─────────────────────────────────────────────
# ML Settings
# ─────────────────────────────────────────────
TARGET_HORIZON = 3  # Predict 3-day forward returns for crypto (faster decay)

# ─────────────────────────────────────────────
# VWAP Proxy
# ─────────────────────────────────────────────
def vwap_proxy(high, low, close):
    return (high + low + 2 * close) / 4

# Volumes moving averages
ADV_PERIODS = [5, 10, 15, 20, 30, 40, 50, 60, 120, 180]

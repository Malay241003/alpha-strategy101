"""
config.py — Central configuration for the Alpha Combination Strategy.

All constants, universe, regime thresholds, alpha weights, and portfolio rules.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ─────────────────────────────────────────────
# Date Range
# ─────────────────────────────────────────────
START_DATE = "2008-01-01"
END_DATE = "2026-04-01"

# In-sample / Out-of-sample split
IN_SAMPLE_END = "2019-12-31"   # 2008-2019 = in-sample (tune)
OUT_SAMPLE_START = "2020-01-01"  # 2020-2025 = out-of-sample (validate)

# ─────────────────────────────────────────────
# Stock Universe — 100 S&P 500 Stocks (Diversified Across Sectors)
# ─────────────────────────────────────────────
UNIVERSE = [
    # ── Technology (20) ──
    "AAPL", "MSFT", "NVDA", "AVGO", "ADBE",
    "CRM", "CSCO", "ACN", "TXN", "INTC",
    "AMD", "QCOM", "IBM", "AMAT", "NOW",
    "ORCL", "INTU", "MU", "LRCX", "KLAC",

    # ── Healthcare (15) ──
    "UNH", "JNJ", "LLY", "MRK", "ABBV",
    "PFE", "TMO", "ABT", "DHR", "AMGN",
    "BMY", "GILD", "MDT", "ISRG", "VRTX",

    # ── Financials (15) ──
    "JPM", "V", "MA", "BAC", "GS",
    "MS", "BRK-B", "SPGI", "BLK", "AXP",
    "C", "SCHW", "MMC", "CB", "PGR",

    # ── Consumer Discretionary (10) ──
    "AMZN", "TSLA", "HD", "MCD", "LOW",
    "NKE", "SBUX", "TJX", "BKNG", "CMG",

    # ── Consumer Staples (8) ──
    "PG", "KO", "PEP", "COST", "WMT",
    "CL", "MDLZ", "PM",

    # ── Communication Services (7) ──
    "GOOGL", "META", "NFLX", "CMCSA", "DIS",
    "TMUS", "VZ",

    # ── Energy (7) ──
    "XOM", "CVX", "COP", "SLB", "EOG",
    "MPC", "PSX",

    # ── Industrials (8) ──
    "HON", "UNP", "CAT", "BA", "GE",
    "RTX", "DE", "LMT",

    # ── Utilities (4) ──
    "NEE", "DUK", "SO", "AEP",

    # ── Real Estate (3) ──
    "PLD", "AMT", "SPG",

    # ── Materials (3) ──
    "LIN", "APD", "SHW",
]

# Market data tickers (for regime filter)
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"

# ─────────────────────────────────────────────
# VWAP Proxy
# ─────────────────────────────────────────────
# True VWAP requires intraday data. We use a weighted typical price.
# (High + Low + 2*Close) / 4 — gives close double weight since most
# volume concentrates near open and close.
def vwap_proxy(high, low, close):
    return (high + low + 2 * close) / 4

# ─────────────────────────────────────────────
# Regime Filter Thresholds
# ─────────────────────────────────────────────
REGIME_SPY_FAST_MA = 50     # Fast moving average period (days)
REGIME_SPY_SLOW_MA = 200    # Slow moving average period (days)
REGIME_VIX_BULL_MAX = 20    # VIX below this + uptrend = BULL
REGIME_VIX_BEAR_MIN = 28    # VIX above this + downtrend = BEAR (slow)
REGIME_VIX_CRISIS = 35      # VIX above this = CRISIS override (any trend)
REGIME_VIX_BEAR_FAST = 25   # VIX above this + Close < 200MA = fast BEAR

# Regime labels
BULL = "BULL"
NEUTRAL = "NEUTRAL"
BEAR = "BEAR"
CRISIS = "CRISIS"

# ─────────────────────────────────────────────
# Alpha Weights by Regime
# ─────────────────────────────────────────────
# Keys are alpha function names, values are weights per regime.
# Weights must sum to 1.0 for each regime.

ALPHA_WEIGHTS = {
    #                    BULL   NEUTRAL  BEAR
    "alpha_101": {BULL: 0.20, NEUTRAL: 0.10, BEAR: 0.05},   # Intraday momentum
    "alpha_52":  {BULL: 0.15, NEUTRAL: 0.10, BEAR: 0.05},   # Long-term momentum
    "alpha_19":  {BULL: 0.15, NEUTRAL: 0.10, BEAR: 0.00},   # Secular trend
    "alpha_9":   {BULL: 0.10, NEUTRAL: 0.15, BEAR: 0.15},   # Adaptive MR/Mom
    "alpha_32":  {BULL: 0.10, NEUTRAL: 0.15, BEAR: 0.10},   # Multi-factor
    "alpha_23":  {BULL: 0.05, NEUTRAL: 0.10, BEAR: 0.20},   # Breakout fade
    "alpha_12":  {BULL: 0.05, NEUTRAL: 0.10, BEAR: 0.20},   # Volume reversal
    "alpha_37":  {BULL: 0.05, NEUTRAL: 0.05, BEAR: 0.15},   # Gap + trend MR
    "alpha_7":   {BULL: 0.10, NEUTRAL: 0.10, BEAR: 0.05},   # Volume conditional
    "alpha_36":  {BULL: 0.05, NEUTRAL: 0.05, BEAR: 0.05},   # Multi-factor diversifier
}

# ─────────────────────────────────────────────
# Portfolio Construction
# ─────────────────────────────────────────────
TOP_N_STOCKS = 5           # Number of stocks to hold (top N by composite score)
MAX_WEIGHT_PER_STOCK = 0.08 # Max 8% in any single stock
REBALANCE_EVERY_N_DAYS = 5  # Rebalance frequency (trading days) — weekly reduces turnover

# Exposure by regime
REGIME_EXPOSURE = {
    BULL:    1.00,   # 100% invested
    NEUTRAL: 0.70,   # 70% invested, 30% cash
    BEAR:    0.30,   # 30% invested, 70% cash
    CRISIS:  0.00,   # 0% invested, 100% cash — full exit
}

# ─────────────────────────────────────────────
# Markov Chain Regime Model
# ─────────────────────────────────────────────
# Method: "heuristic" = old MA+VIX rules, "markov" = Markov Chain HMM
MARKOV_REGIME_METHOD = "markov"

# Markov model calibration uses data up to IN_SAMPLE_END (no look-ahead)
# Rolling window for adaptive calibration within the IS period
MARKOV_CALIBRATION_WINDOW = 252  # 1 year of trading days

# ─────────────────────────────────────────────
# Risk & R-Multiple Configuration
# ─────────────────────────────────────────────
RISK_PER_TRADE_PCT = 0.5    # 0.5% risk per trade (defines 1R)
# 1R = INITIAL_CAPITAL * RISK_PER_TRADE_PCT / 100

# ─────────────────────────────────────────────
# Transaction Costs
# ─────────────────────────────────────────────
COMMISSION_PER_SHARE = 0.005   # $0.005 per share (half a cent)
SLIPPAGE_BPS = 5               # 5 basis points slippage
INITIAL_CAPITAL = 100_000      # Starting capital in USD

# ─────────────────────────────────────────────
# ADV Lookback Periods (for adv{d} computation)
# ─────────────────────────────────────────────
ADV_PERIODS = [5, 10, 15, 20, 30, 40, 50, 60, 120, 180]

# ─────────────────────────────────────────────
# ML Target
# ─────────────────────────────────────────────
TARGET_HORIZON = 5  # Predict 5-day forward return sign

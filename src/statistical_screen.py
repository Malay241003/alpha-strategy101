"""
statistical_screen.py — 4-stage institutional statistical filter.

Ported from tradeBot/scripts_us_stocks/statisticalScreenStocks.js

Stages:
  1. Minimum viability (trades ≥ 20, expectancy > 0, maxDD < 15R)
  2. Risk-adjusted quality (annualized Sharpe ≥ 0.25)
  3. Regime stability (profitable in ≥ 2/3 time periods)
  4. Sector concentration cap (max 5 stocks per GICS sector)

References:
  - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
  - Harvey, Liu & Zhu (2016) "...and the Cross-Section of Expected Returns"
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from config import BULL, NEUTRAL, BEAR, CRISIS

# ═══════════════════════════════════════════════
# GICS SECTOR MAPPING (100 stocks)
# ═══════════════════════════════════════════════
SECTOR_MAP = {
    # Technology (20)
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "CSCO": "Technology", "ACN": "Technology", "TXN": "Technology",
    "INTC": "Technology", "AMD": "Technology", "QCOM": "Technology",
    "IBM": "Technology", "AMAT": "Technology", "NOW": "Technology",
    "ORCL": "Technology", "INTU": "Technology", "MU": "Technology",
    "LRCX": "Technology", "KLAC": "Technology",
    # Healthcare (15)
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "MRK": "Healthcare", "ABBV": "Healthcare", "PFE": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "AMGN": "Healthcare", "BMY": "Healthcare", "GILD": "Healthcare",
    "MDT": "Healthcare", "ISRG": "Healthcare", "VRTX": "Healthcare",
    # Financials (15)
    "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "BRK-B": "Financials", "SPGI": "Financials", "BLK": "Financials",
    "AXP": "Financials", "C": "Financials", "SCHW": "Financials",
    "MMC": "Financials", "CB": "Financials", "PGR": "Financials",
    # Consumer Discretionary (10)
    "AMZN": "Consumer Disc.", "TSLA": "Consumer Disc.", "HD": "Consumer Disc.",
    "MCD": "Consumer Disc.", "LOW": "Consumer Disc.", "NKE": "Consumer Disc.",
    "SBUX": "Consumer Disc.", "TJX": "Consumer Disc.", "BKNG": "Consumer Disc.",
    "CMG": "Consumer Disc.",
    # Consumer Staples (8)
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "CL": "Consumer Staples",
    "MDLZ": "Consumer Staples", "PM": "Consumer Staples",
    # Communication Services (7)
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
    "CMCSA": "Communication", "DIS": "Communication", "TMUS": "Communication",
    "VZ": "Communication",
    # Energy (7)
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy", "PSX": "Energy",
    # Industrials (8)
    "HON": "Industrials", "UNP": "Industrials", "CAT": "Industrials",
    "BA": "Industrials", "GE": "Industrials", "RTX": "Industrials",
    "DE": "Industrials", "LMT": "Industrials",
    # Utilities (4)
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities",
    # Real Estate (3)
    "PLD": "Real Estate", "AMT": "Real Estate", "SPG": "Real Estate",
    # Materials (3)
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
}

# ═══════════════════════════════════════════════
# SCREENING THRESHOLDS
# ═══════════════════════════════════════════════
MIN_TRADES = 20
MIN_EXPECTANCY = 0.0
MAX_DD_R = 15.0
MIN_SHARPE = 0.25
MIN_PROFITABLE_PERIODS = 2  # out of 3
MAX_PER_SECTOR = 5


def normal_cdf(x):
    """Normal CDF approximation (Abramowitz & Stegun)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = -1 if x < 0 else 1
    x = abs(x) / np.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def compute_per_stock_stats(ticker, trades, regime_series):
    """
    Compute screening statistics for a single stock from its trade list.

    Returns dict with: trades, winRate, expectancy, netR, maxDD, sharpe,
                        calmar, profitablePeriods, periodExpectancies, sector
    """
    if not trades:
        return None

    r_values = [t["R"] for t in trades]
    n = len(r_values)
    won = [r for r in r_values if r > 0]
    lost = [r for r in r_values if r <= 0]

    win_rate = len(won) / n * 100 if n > 0 else 0
    net_r = sum(r_values)

    # Expectancy: WR × avgWin - (1-WR) × avgLoss
    avg_win = np.mean(won) if won else 0
    avg_loss = np.mean([abs(r) for r in lost]) if lost else 0
    wr = len(won) / n if n > 0 else 0
    expectancy = wr * avg_win - (1 - wr) * avg_loss

    # Max drawdown in R
    cumsum = np.cumsum(r_values)
    running_max = np.maximum.accumulate(cumsum)
    dd = cumsum - running_max
    max_dd = abs(dd.min()) if len(dd) > 0 else 0

    # Annualized Sharpe from trade R-values
    # Approximate: treat each trade as independent observation
    if n > 1 and np.std(r_values) > 0:
        trades_per_year = n / max(1, len(set(
            t.get("entryDate", t.get("entry_date", None)) for t in trades
            if t.get("entryDate") or t.get("entry_date")
        )) / 252) if n > 10 else 50
        sharpe = (np.mean(r_values) / np.std(r_values)) * np.sqrt(min(trades_per_year, 252))
    else:
        sharpe = 0.0

    calmar = net_r / max_dd if max_dd > 0 else 0.0

    # Regime stability: split trades into 3 equal periods
    period_size = n // 3
    period_expectations = []
    for p in range(3):
        start = p * period_size
        end = (p + 1) * period_size if p < 2 else n
        period_r = r_values[start:end]
        if period_r:
            period_expectations.append(np.mean(period_r))
        else:
            period_expectations.append(0.0)

    profitable_periods = sum(1 for e in period_expectations if e > 0)

    return {
        "ticker": ticker,
        "sector": SECTOR_MAP.get(ticker, "Unknown"),
        "trades": n,
        "winRate": round(win_rate, 2),
        "expectancy": round(expectancy, 4),
        "netR": round(net_r, 2),
        "maxDD": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "profitablePeriods": profitable_periods,
        "periodExpectancies": [round(e, 4) for e in period_expectations],
    }


def run_statistical_screen(stock_stats_list):
    """
    Run the full 4-stage statistical screen.

    Args:
        stock_stats_list: list of per-stock stat dicts from compute_per_stock_stats()

    Returns:
        screened: list of stocks that passed all 4 stages
        report: dict with stage-by-stage results
    """
    total = len(stock_stats_list)
    report = {"total": total, "stages": {}}

    print(f"\n{'═' * 60}")
    print(f"  🧪 STATISTICAL SCREENING PIPELINE")
    print(f"{'═' * 60}")
    print(f"  Input: {total} stocks\n")

    # ── STAGE 1: Minimum Viability ──
    print("── STAGE 1: Minimum Viability ──")
    print(f"   Rules: trades ≥ {MIN_TRADES}, expectancy > {MIN_EXPECTANCY}, maxDD < {MAX_DD_R}R\n")

    stage1 = []
    stage1_rejected = []
    for s in stock_stats_list:
        pass_trades = s["trades"] >= MIN_TRADES
        pass_exp = s["expectancy"] > MIN_EXPECTANCY
        pass_dd = s["maxDD"] < MAX_DD_R

        if pass_trades and pass_exp and pass_dd:
            stage1.append(s)
        else:
            reasons = []
            if not pass_trades: reasons.append(f"trades={s['trades']}")
            if not pass_exp: reasons.append(f"exp={s['expectancy']:.2f}")
            if not pass_dd: reasons.append(f"dd={s['maxDD']:.1f}")
            stage1_rejected.append((s["ticker"], reasons))
            print(f"   ❌ {s['ticker']:>6s} {', '.join(reasons)}")

    print(f"\n   ✅ Stage 1 survivors: {len(stage1)}/{total}\n")
    report["stages"]["stage1"] = {"passed": len(stage1), "rejected": len(stage1_rejected)}

    # ── STAGE 2: Risk-Adjusted Quality ──
    print("── STAGE 2: Risk-Adjusted Quality ──")
    print(f"   Rule: Annualized Sharpe ≥ {MIN_SHARPE}\n")

    stage2 = []
    for s in stage1:
        # Deflated Sharpe Ratio (informational)
        num_trials = total
        approx_days = max(s["trades"] * 5, 252)
        expected_max_sr = np.sqrt(2 * np.log(max(num_trials, 2)))
        var_sr = 1 / approx_days
        z_score = (s["sharpe"] - expected_max_sr) / np.sqrt(var_sr)
        dsr = normal_cdf(z_score)
        s["dsr"] = round(dsr, 3)

        if s["sharpe"] >= MIN_SHARPE:
            stage2.append(s)
            print(f"   ✅ {s['ticker']:>6s} SR={s['sharpe']:.2f} (DSR={s['dsr']:.3f}) [{s['sector']}]")
        else:
            print(f"   ❌ {s['ticker']:>6s} SR={s['sharpe']:.2f} < {MIN_SHARPE}")

    print(f"\n   ✅ Stage 2 survivors: {len(stage2)}/{len(stage1)}\n")
    report["stages"]["stage2"] = {"passed": len(stage2)}

    # ── STAGE 3: Regime Stability ──
    print(f"── STAGE 3: Regime Stability ({MIN_PROFITABLE_PERIODS}/3 periods profitable) ──\n")

    stage3 = []
    for s in stage2:
        if s["profitablePeriods"] >= MIN_PROFITABLE_PERIODS:
            stage3.append(s)
            exp_str = ", ".join(f"{'+' if e > 0 else ''}{e:.2f}" for e in s["periodExpectancies"])
            print(f"   ✅ {s['ticker']:>6s} {s['profitablePeriods']}/3 periods [{exp_str}]")
        else:
            print(f"   ❌ {s['ticker']:>6s} {s['profitablePeriods']}/3 — regime dependent")

    print(f"\n   ✅ Stage 3 survivors: {len(stage3)}/{len(stage2)}\n")
    report["stages"]["stage3"] = {"passed": len(stage3)}

    # ── STAGE 4: Sector Concentration Cap ──
    print(f"── STAGE 4: Sector Concentration Cap (max {MAX_PER_SECTOR} per sector) ──\n")

    # Sort by Sharpe within sector to keep the best
    stage3_sorted = sorted(stage3, key=lambda x: x["sharpe"], reverse=True)
    sector_counts = {}
    stage4 = []

    for s in stage3_sorted:
        sector = s["sector"]
        sector_counts.setdefault(sector, 0)

        if sector_counts[sector] < MAX_PER_SECTOR:
            sector_counts[sector] += 1
            stage4.append(s)
            print(f"   ✅ {s['ticker']:>6s} [{sector}] ({sector_counts[sector]}/{MAX_PER_SECTOR})")
        else:
            print(f"   ❌ {s['ticker']:>6s} [{sector}] — sector cap hit")

    print(f"\n   ✅ Stage 4 survivors: {len(stage4)}/{len(stage3)}")

    # Sort final by expectancy
    stage4.sort(key=lambda x: x["expectancy"], reverse=True)

    # Summary table
    print(f"\n{'═' * 80}")
    print(f"  🏆 FINAL SCREENED UNIVERSE: {len(stage4)} stocks")
    print(f"{'═' * 80}\n")

    print("  #  Stock   Sector            Trades  WinRate  Expect.   NetR    MaxDD   Sharpe")
    print("  ─  ─────   ──────            ──────  ───────  ───────   ────    ─────   ──────")
    for i, s in enumerate(stage4):
        print(
            f"{i+1:>3}  {s['ticker']:>6s}  {s['sector']:<16s}  {s['trades']:>5d}  "
            f"{s['winRate']:>5.1f}%  {'+' if s['expectancy'] > 0 else ''}{s['expectancy']:>6.2f}  "
            f"{'+' if s['netR'] > 0 else ''}{s['netR']:>7.1f}  {s['maxDD']:>6.1f}  {s['sharpe']:>6.2f}"
        )

    report["stages"]["stage4"] = {"passed": len(stage4)}
    report["screened_tickers"] = [s["ticker"] for s in stage4]

    return stage4, report

"""
analysis.py — Performance metrics, Monte Carlo simulation, stress testing, and visualization.

Incorporates analysis patterns from the user's existing us_stocks_long.ipynb notebook:
  - Equity curve with regime overlay
  - Drawdown analysis
  - Monte Carlo simulation (IID, block bootstrap, stress)
  - R-distribution with fat-tail detection
  - Stress testing (execution shock)
  - Monthly returns heatmap
  - Rolling Sharpe
  - In-sample vs Out-of-sample comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from config import RESULTS_DIR, INITIAL_CAPITAL, BULL, NEUTRAL, BEAR, CRISIS


# ═══════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════

def compute_metrics(equity_curve, daily_returns=None, label="Full Period"):
    """
    Compute comprehensive performance metrics.

    Returns: dict of metrics
    """
    if daily_returns is None:
        daily_returns = equity_curve.pct_change().dropna()

    n_days = len(daily_returns)
    n_years = n_days / 252

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe & Sortino
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

    downside_ret = daily_returns[daily_returns < 0]
    downside_std = downside_ret.std()
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin()

    # Max DD duration
    dd_duration = 0
    max_dd_dur = 0
    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            dd_duration += 1
            max_dd_dur = max(max_dd_dur, dd_duration)
        else:
            dd_duration = 0

    # Win rate (daily)
    win_days = (daily_returns > 0).sum()
    total_days = len(daily_returns)
    win_rate = win_days / total_days * 100 if total_days > 0 else 0

    # Average win / loss
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else np.inf

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    metrics = {
        "label": label,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_dd_duration_days": max_dd_dur,
        "calmar": calmar,
        "win_rate_daily": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "n_days": n_days,
        "n_years": n_years,
        "final_equity": equity_curve.iloc[-1],
    }

    return metrics


def print_metrics(metrics):
    """Print metrics in a formatted table."""
    print(f"\n{'═' * 50}")
    print(f"  📊 {metrics['label']}")
    print(f"{'═' * 50}")
    print(f"  Total Return:     {metrics['total_return']:>10.1%}")
    print(f"  CAGR:             {metrics['cagr']:>10.1%}")
    print(f"  Sharpe Ratio:     {metrics['sharpe']:>10.2f}")
    print(f"  Sortino Ratio:    {metrics['sortino']:>10.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']:>10.1%}")
    print(f"  Max DD Duration:  {metrics['max_dd_duration_days']:>10d} days")
    print(f"  Calmar Ratio:     {metrics['calmar']:>10.2f}")
    print(f"  Daily Win Rate:   {metrics['win_rate_daily']:>10.1f}%")
    print(f"  Avg Win:          {metrics['avg_win']:>10.4f}")
    print(f"  Avg Loss:         {metrics['avg_loss']:>10.4f}")
    print(f"  Profit Factor:    {metrics['profit_factor']:>10.2f}")
    print(f"  Final Equity:     ${metrics['final_equity']:>10,.0f}")
    print(f"{'═' * 50}")


# ═══════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════

def plot_equity_with_regime(equity_curve, regime, benchmark=None, save_path=None):
    """
    Plot equity curve with colored regime overlay.

    Green background = BULL, Yellow = NEUTRAL, Red = BEAR
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[3, 1, 1])

    ax1, ax2, ax3 = axes

    # ─── Regime colored background ───
    regime_colors = {BULL: "#E8F5E9", NEUTRAL: "#FFF8E1", BEAR: "#FFEBEE", CRISIS: "#B71C1C"}
    prev_regime = regime.iloc[0]
    start_date = regime.index[0]

    for date, r in regime.items():
        if r != prev_regime:
            ax1.axvspan(start_date, date, alpha=0.3,
                        color=regime_colors.get(prev_regime, "#FFFFFF"))
            ax2.axvspan(start_date, date, alpha=0.3,
                        color=regime_colors.get(prev_regime, "#FFFFFF"))
            start_date = date
            prev_regime = r
    # Last period
    ax1.axvspan(start_date, regime.index[-1], alpha=0.3,
                color=regime_colors.get(prev_regime, "#FFFFFF"))
    ax2.axvspan(start_date, regime.index[-1], alpha=0.3,
                color=regime_colors.get(prev_regime, "#FFFFFF"))

    # ─── Equity curve ───
    ax1.plot(equity_curve.index, equity_curve.values,
             color="#1565C0", linewidth=1.5, label="Strategy")
    if benchmark is not None:
        common = equity_curve.index.intersection(benchmark.index)
        if len(common) > 0:
            # Normalize benchmark to start at same value as equity curve ON the common start date
            bench_start_val = benchmark.loc[common[0]]
            equity_start_val = equity_curve.loc[common[0]]
            bench_norm = benchmark.loc[common] / bench_start_val * equity_start_val
            ax1.plot(common, bench_norm.values,
                     color="#757575", linewidth=1, alpha=0.7, label="SPY (Buy & Hold)")
    ax1.set_title("Equity Curve with Regime Overlay", fontsize=15, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ─── Drawdown ───
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#EF5350", alpha=0.5)
    ax2.set_title("Drawdown (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)

    # ─── Daily returns ───
    daily_ret = equity_curve.pct_change()
    colors = ["#66BB6A" if r > 0 else "#EF5350" for r in daily_ret.values]
    ax3.bar(daily_ret.index, daily_ret.values * 100, color=colors, alpha=0.5, width=1)
    ax3.set_title("Daily Returns (%)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Return %")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  💾 Saved: {save_path}")
    # plt.show() # Disabled for background execution to prevent hanging


def plot_monthly_returns_heatmap(equity_curve, save_path=None):
    """Monthly returns heatmap (year × month)."""
    daily_ret = equity_curve.pct_change()
    monthly = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values * 100
    })
    pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 9, "fontweight": "bold"},
                cbar_kws={"label": "Monthly Return %"})
    ax.set_title("Monthly Returns Heatmap (%)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show() # Disabled for background execution to prevent hanging


def plot_rolling_sharpe(equity_curve, window=252, save_path=None):
    """Rolling 12-month Sharpe ratio."""
    daily_ret = equity_curve.pct_change()
    rolling_mean = daily_ret.rolling(window).mean()
    rolling_std = daily_ret.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="#1565C0", linewidth=1.5)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.axhline(1, color="green", linestyle="--", alpha=0.3, label="Sharpe = 1")
    ax.axhline(2, color="green", linestyle=":", alpha=0.3, label="Sharpe = 2")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values > 0, color="#66BB6A", alpha=0.2)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values < 0, color="#EF5350", alpha=0.2)
    ax.set_title(f"Rolling {window}-day Sharpe Ratio", fontsize=15, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show() # Disabled for background execution to prevent hanging


# ═══════════════════════════════════════════════
# MONTE CARLO SIMULATION (from user's notebook pattern)
# ═══════════════════════════════════════════════

def monte_carlo_simulation(daily_returns, n_sims=5000, n_days=None):
    """
    Monte Carlo simulation with multiple models:
      - IID Shuffle: randomly resample daily returns
      - Block Bootstrap: resample in blocks of 20 days (preserves autocorrelation)
      - Stress Injected: IID but inject -3% shock days randomly

    Returns: dict with paths, drawdowns, and stats for each model
    """
    if n_days is None:
        n_days = len(daily_returns)
    returns = daily_returns.dropna().values

    models = {}

    # ─── IID Shuffle ───
    print("  Running IID MC...")
    iid_paths = []
    iid_drawdowns = []
    for _ in range(n_sims):
        sampled = np.random.choice(returns, size=n_days, replace=True)
        equity = INITIAL_CAPITAL * np.cumprod(1 + sampled)
        iid_paths.append(equity)
        cummax = np.maximum.accumulate(equity)
        dd = (equity - cummax) / cummax
        iid_drawdowns.append(dd.min())

    models["iid"] = _mc_stats(iid_paths, iid_drawdowns, "IID Shuffle")

    # ─── Block Bootstrap ───
    print("  Running Block Bootstrap MC...")
    block_size = 20
    block_paths = []
    block_drawdowns = []
    for _ in range(n_sims):
        sampled = []
        while len(sampled) < n_days:
            start = np.random.randint(0, max(1, len(returns) - block_size))
            sampled.extend(returns[start:start + block_size])
        sampled = np.array(sampled[:n_days])
        equity = INITIAL_CAPITAL * np.cumprod(1 + sampled)
        block_paths.append(equity)
        cummax = np.maximum.accumulate(equity)
        dd = (equity - cummax) / cummax
        block_drawdowns.append(dd.min())

    models["block"] = _mc_stats(block_paths, block_drawdowns, "Block Bootstrap")

    # ─── Stress Injected ───
    print("  Running Stress MC...")
    stress_paths = []
    stress_drawdowns = []
    shock_prob = 0.02  # 2% chance of shock day
    shock_magnitude = -0.03  # -3% shock
    for _ in range(n_sims):
        sampled = np.random.choice(returns, size=n_days, replace=True)
        # Inject shocks
        shocks = np.random.random(n_days) < shock_prob
        sampled[shocks] = shock_magnitude
        equity = INITIAL_CAPITAL * np.cumprod(1 + sampled)
        stress_paths.append(equity)
        cummax = np.maximum.accumulate(equity)
        dd = (equity - cummax) / cummax
        stress_drawdowns.append(dd.min())

    models["stress"] = _mc_stats(stress_paths, stress_drawdowns, "Stress Injected")

    return models


def _mc_stats(paths, drawdowns, name):
    """Helper to compute MC stats for a model."""
    finals = [p[-1] for p in paths]
    return {
        "name": name,
        "paths": paths,
        "drawdowns": drawdowns,
        "stats": {
            "medianFinal": np.median(finals),
            "pct5Final": np.percentile(finals, 5),
            "pct95Final": np.percentile(finals, 95),
            "medianDD": np.median(drawdowns),
            "pct5DD": np.percentile(drawdowns, 5),
            "pct95DD": np.percentile(drawdowns, 95),
            "medianCAGR": ((np.median(finals) / INITIAL_CAPITAL) **
                           (252 / len(paths[0])) - 1) * 100,
        }
    }


def plot_monte_carlo(mc_results, save_path=None):
    """Plot MC equity fan charts for each model."""
    n_models = len(mc_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 7))
    if n_models == 1:
        axes = [axes]

    colors = {"iid": "#66BB6A", "block": "#42A5F5", "stress": "#EF5350"}

    for ax, (key, model) in zip(axes, mc_results.items()):
        color = colors.get(key, "#888888")
        paths = model["paths"]

        # Plot sample paths
        for path in paths[:200]:  # Only plot 200 for readability
            ax.plot(path, color=color, alpha=0.03, linewidth=0.3)

        # Percentile bands
        max_len = max(len(p) for p in paths)
        padded = np.array([np.pad(p, (0, max_len - len(p)),
                                   constant_values=p[-1]) for p in paths])
        median = np.median(padded, axis=0)
        p5 = np.percentile(padded, 5, axis=0)
        p95 = np.percentile(padded, 95, axis=0)

        x = range(len(median))
        ax.plot(median, color=color, linewidth=2.5, label="Median")
        ax.fill_between(x, p5, p95, alpha=0.12, color=color, label="5th–95th %ile")
        ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_title(f"{model['name']}\nMedian: ${model['stats']['medianFinal']:,.0f}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(fontsize=9, loc="upper left")

    plt.suptitle("Monte Carlo — Equity Fan Charts (5,000 sims)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show() # Disabled for background execution to prevent hanging


# ═══════════════════════════════════════════════
# STRESS TEST (from user's notebook)
# ═══════════════════════════════════════════════

def stress_test(daily_returns, shock_values=[0.001, 0.002, 0.003], save_path=None):
    """
    Test strategy robustness by subtracting execution shock from all returns.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    original_mean = daily_returns.mean()
    sns.histplot(daily_returns, bins=40, color="green", stat="density",
                 alpha=0.4, label=f"Original (Mean: {original_mean:.4f})", ax=ax)

    for shock in shock_values:
        stressed = daily_returns - shock
        stressed_mean = stressed.mean()
        label = f"Stressed -{shock*100:.1f}bps (Mean: {stressed_mean:.4f})"
        status = "✅" if stressed_mean > 0 else "❌"
        print(f"   {status} Shock {shock*100:.1f}bps: Mean return = {stressed_mean:.6f}")
        sns.histplot(stressed, bins=40, stat="density", alpha=0.3, label=label, ax=ax)

    ax.axvline(0, color="black", linestyle="--")
    ax.legend()
    ax.set_title("Stress Test: Impact of Execution Shock", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show() # Disabled for background execution to prevent hanging


# ═══════════════════════════════════════════════
# R-DISTRIBUTION (from user's notebook)
# ═══════════════════════════════════════════════

def plot_return_distribution(daily_returns, save_path=None):
    """
    Plot daily return distribution with normal overlay for fat-tail detection.
    """
    data = daily_returns.dropna()
    mu, std = data.mean(), data.std()
    kurt = data.kurtosis()
    skew = data.skew()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Histogram
    n, bins, patches = ax.hist(data, bins=60, density=True,
                               color="#4FC3F7", edgecolor="white", alpha=0.7,
                               label="Actual Distribution")

    # Bell curve overlay
    x = np.linspace(data.min() - 0.005, data.max() + 0.005, 500)
    bell = norm.pdf(x, mu, std)
    ax.plot(x, bell, color="#1565C0", linewidth=3, label="Normal Distribution")
    ax.fill_between(x, bell, alpha=0.15, color="#1565C0")

    # Annotations
    ax.axvline(mu, color="black", linestyle="-", linewidth=1.2, alpha=0.6,
               label=f"Mean = {mu:.5f}")
    ax.axvline(data.median(), color="#FF6F00", linestyle="--", linewidth=1.5,
               label=f"Median = {data.median():.5f}")

    # Stats box
    stats_text = (f"Excess Kurtosis: {kurt:.2f}\n"
                  f"Skewness: {skew:.2f}\n"
                  f"Fat-Tailed: {'YES' if kurt > 0 else 'NO'}")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      alpha=0.9, edgecolor="gray"))

    ax.set_title("Daily Return Distribution — Fat Tail Detection",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show() # Disabled for background execution to prevent hanging


# ═══════════════════════════════════════════════
# R-MULTIPLE TRADE ANALYSIS
# ═══════════════════════════════════════════════

def compute_trade_summary_r(trade_log):
    """
    Compute comprehensive trade-level statistics in R-multiples.

    Matches the user's existing backtest output format from their JS pipeline.
    Returns a dict with the same structure as their existing system.
    """
    import json

    if trade_log is None or len(trade_log) == 0:
        print("  ⚠️ No completed trades to analyze")
        return None

    df = trade_log.copy()
    r_values = df["R"]

    # Basic stats
    trades = len(df)
    won = df[r_values > 0]
    lost = df[r_values <= 0]
    won_count = len(won)
    lost_count = len(lost)
    win_rate = (won_count / trades * 100) if trades > 0 else 0
    expectancy = r_values.mean() if trades > 0 else 0

    # Drawdown in R (running sum of R)
    cumsum_r = r_values.cumsum()
    running_max = cumsum_r.cummax()
    dd_r = cumsum_r - running_max
    max_dd_r = abs(dd_r.min()) if len(dd_r) > 0 else 0

    # Profit/Loss in R
    total_profit_r = won["R"].sum() if len(won) > 0 else 0
    total_loss_r = abs(lost["R"].sum()) if len(lost) > 0 else 0
    net_profit_r = r_values.sum()

    # Avg time in trade
    avg_time = df["holding_days"].mean() if "holding_days" in df.columns else 0

    # Win distribution (top 5 + count of others)
    win_dist = {}
    if len(won) > 0:
        win_r_counts = won["R"].value_counts().sort_values(ascending=False)
        for i, (r_val, count) in enumerate(win_r_counts.items()):
            if i < 5:
                pct = count / won_count * 100
                win_dist[f"+{r_val:.2f}R"] = f"{count} ({pct:.2f}%)"
            else:
                remaining = len(win_r_counts) - 5
                win_dist["..."] = f"({remaining} other targets)"
                break

    # Loss distribution (top 5 + count of others)
    loss_dist = {}
    if len(lost) > 0:
        loss_r_counts = lost["R"].value_counts().sort_values(ascending=False)
        for i, (r_val, count) in enumerate(loss_r_counts.items()):
            if i < 5:
                pct = count / lost_count * 100
                loss_dist[f"{r_val:.2f}R"] = f"{count} ({pct:.2f}%)"
            else:
                remaining = len(loss_r_counts) - 5
                loss_dist["..."] = f"({remaining} other targets)"
                break

    summary = {
        "trades": trades,
        "winRate": f"{win_rate:.2f}",
        "expectancy": f"{expectancy:.2f}",
        "maxDrawdownR": f"{max_dd_r:.2f}",
        "avgTimeInTradeBars": f"{avg_time:.1f}",
        "wonTrades": won_count,
        "lostTrades": lost_count,
        "totalProfit": f"{total_profit_r:.2f}",
        "totalLoss": f"{total_loss_r:.2f}",
        "netProfit": f"{net_profit_r:.2f}",
        "winDistribution": win_dist,
        "lossDistribution": loss_dist,
    }

    # Print summary
    print(f"\n{'═' * 55}")
    print(f"  📊 TRADE SUMMARY (R-Multiples)")
    print(f"{'═' * 55}")
    print(f"  Trades:           {trades}")
    print(f"  Won / Lost:       {won_count} / {lost_count}")
    print(f"  Win Rate:         {win_rate:.2f}%")
    print(f"  Expectancy:       {expectancy:.2f}R")
    print(f"  Max Drawdown:     {max_dd_r:.2f}R")
    print(f"  Avg Holding:      {avg_time:.1f} days")
    print(f"  Total Profit:     +{total_profit_r:.2f}R")
    print(f"  Total Loss:       -{total_loss_r:.2f}R")
    print(f"  Net Profit:       {net_profit_r:.2f}R")
    print(f"{'═' * 55}")

    # Print JSON for easy copy-paste
    print(f"\n📋 JSON Summary:")
    print(json.dumps(summary, indent=2))

    return summary


def plot_trade_r_distribution(trade_log, save_path=None):
    """
    Plot the R-multiple distribution of completed trades.
    """
    if trade_log is None or len(trade_log) == 0:
        return

    r_vals = trade_log["R"].dropna()
    mu, std = r_vals.mean(), r_vals.std()
    kurt = r_vals.kurtosis()
    skew = r_vals.skew()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # ─── Left: R histogram with bell curve ───
    ax = axes[0]
    n, bins, patches = ax.hist(r_vals, bins=50, density=True,
                               color="#4FC3F7", edgecolor="white", alpha=0.7,
                               label="Actual Distribution")
    x = np.linspace(r_vals.min() - 0.5, r_vals.max() + 0.5, 500)
    bell = norm.pdf(x, mu, std)
    ax.plot(x, bell, color="#1565C0", linewidth=3, label="Normal Distribution")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(mu, color="black", linestyle="-", linewidth=1.2, alpha=0.6,
               label=f"Mean = {mu:.2f}R")
    ax.axvline(r_vals.median(), color="#FF6F00", linestyle="--", linewidth=1.5,
               label=f"Median = {r_vals.median():.2f}R")

    stats_text = (f"Kurtosis: {kurt:.2f}\nSkewness: {skew:.2f}\n"
                  f"Fat-Tailed: {'YES' if kurt > 0 else 'NO'}")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      alpha=0.9, edgecolor="gray"))
    ax.set_title("Trade R-Multiple Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("R-Multiple")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    # ─── Right: Cumulative R over time ───
    ax2 = axes[1]
    cum_r = r_vals.cumsum()
    ax2.plot(range(len(cum_r)), cum_r.values, color="#1565C0", linewidth=1.5)
    ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax2.fill_between(range(len(cum_r)), cum_r.values, 0,
                     where=cum_r.values >= 0, color="#66BB6A", alpha=0.2)
    ax2.fill_between(range(len(cum_r)), cum_r.values, 0,
                     where=cum_r.values < 0, color="#EF5350", alpha=0.2)
    ax2.set_title("Cumulative R-Multiple (Equity in R)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Cumulative R")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show() # Disabled for background execution to prevent hanging


# ═══════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════

def compare_periods(metrics_list):
    """
    Print a comparison table for in-sample vs out-of-sample (or any list of metrics).
    """
    df = pd.DataFrame(metrics_list)
    df = df.set_index("label")

    # Format
    fmt = {
        "total_return": "{:.1%}",
        "cagr": "{:.1%}",
        "sharpe": "{:.2f}",
        "sortino": "{:.2f}",
        "max_drawdown": "{:.1%}",
        "calmar": "{:.2f}",
        "win_rate_daily": "{:.1f}%",
        "profit_factor": "{:.2f}",
        "final_equity": "${:,.0f}",
    }

    display_cols = ["total_return", "cagr", "sharpe", "sortino", "max_drawdown",
                    "calmar", "win_rate_daily", "profit_factor", "final_equity"]

    print(f"\n{'═' * 80}")
    print(f"  📊 Period Comparison")
    print(f"{'═' * 80}")

    header = f"{'Metric':<20}"
    for m in metrics_list:
        header += f"  {m['label']:>20}"
    print(header)
    print(f"{'─' * 80}")

    for col in display_cols:
        row = f"{col:<20}"
        for m in metrics_list:
            val = m.get(col, 0)
            if col in fmt:
                row += f"  {fmt[col].format(val):>20}"
            else:
                row += f"  {val:>20}"
        print(row)

    print(f"{'═' * 80}")


# ═══════════════════════════════════════════════
# MASTER REPORT
# ═══════════════════════════════════════════════

def generate_full_report(results, benchmark_equity=None):
    """
    Generate the complete analysis report with all plots.
    """
    import json as json_mod

    os.makedirs(RESULTS_DIR, exist_ok=True)

    equity = results["equity_curve"]
    daily_ret = results["daily_returns"]
    regime = results["regime"]
    trade_log = results.get("trade_log", None)

    print("\n" + "=" * 60)
    print("  📊 GENERATING FULL ANALYSIS REPORT")
    print("=" * 60)

    # 1. Full period metrics
    full_metrics = compute_metrics(equity, daily_ret, label="Full (2008-2025)")
    print_metrics(full_metrics)

    # 2. In-sample / Out-of-sample metrics
    from src.backtester import split_in_out_sample
    in_sample_eq, out_sample_eq = split_in_out_sample(results)

    metrics_list = [full_metrics]
    if len(in_sample_eq) > 10:
        is_metrics = compute_metrics(in_sample_eq, label="In-Sample (2008-2019)")
        print_metrics(is_metrics)
        metrics_list.append(is_metrics)

    if len(out_sample_eq) > 10:
        os_metrics = compute_metrics(out_sample_eq, label="Out-of-Sample (2020-2025)")
        print_metrics(os_metrics)
        metrics_list.append(os_metrics)

    if len(metrics_list) > 1:
        compare_periods(metrics_list)

    # 3. R-Multiple Trade Analysis
    print("\n📊 Computing R-multiple trade statistics...")
    trade_summary = compute_trade_summary_r(trade_log)
    if trade_summary:
        # Save JSON
        json_path = os.path.join(RESULTS_DIR, "trade_summary.json")
        with open(json_path, "w") as f:
            json_mod.dump(trade_summary, f, indent=2)
        print(f"  💾 Saved: {json_path}")

        # Save detailed trade log as CSV
        if trade_log is not None and len(trade_log) > 0:
            csv_path = os.path.join(RESULTS_DIR, "trades_detailed.csv")
            trade_log.to_csv(csv_path, index=False)
            print(f"  💾 Saved: {csv_path}")

    # 4. Equity curve with regime overlay
    print("\n📈 Plotting equity curve with regime overlay...")
    plot_equity_with_regime(
        equity, regime, benchmark=benchmark_equity,
        save_path=os.path.join(RESULTS_DIR, "equity_curve.png")
    )

    # 5. Trade R-distribution
    print("\n📊 Plotting trade R-distribution...")
    plot_trade_r_distribution(
        trade_log, save_path=os.path.join(RESULTS_DIR, "trade_r_distribution.png")
    )

    # 6. Monthly returns heatmap
    print("\n📅 Plotting monthly returns heatmap...")
    plot_monthly_returns_heatmap(
        equity, save_path=os.path.join(RESULTS_DIR, "monthly_heatmap.png")
    )

    # 7. Rolling Sharpe
    print("\n📉 Plotting rolling Sharpe...")
    plot_rolling_sharpe(
        equity, save_path=os.path.join(RESULTS_DIR, "rolling_sharpe.png")
    )

    # 8. Return distribution
    print("\n📊 Plotting return distribution...")
    plot_return_distribution(
        daily_ret, save_path=os.path.join(RESULTS_DIR, "return_distribution.png")
    )

    # 9. Stress test
    print("\n🔬 Running stress tests...")
    stress_test(
        daily_ret, save_path=os.path.join(RESULTS_DIR, "stress_test.png")
    )

    # 10. Monte Carlo
    print("\n🎲 Running Monte Carlo simulation (5,000 sims × 3 models)...")
    mc_results = monte_carlo_simulation(daily_ret)
    plot_monte_carlo(
        mc_results, save_path=os.path.join(RESULTS_DIR, "monte_carlo.png")
    )

    print(f"\n✅ Full report generated. Plots saved to: {RESULTS_DIR}")
    return full_metrics

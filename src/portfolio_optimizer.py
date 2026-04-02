"""
portfolio_optimizer.py — Institutional portfolio weight optimization.

Ported from tradeBot/scripts_us_stocks/portfolioOptimizerStocks.js

Implements 4 methods:
  1. Minimum Variance (Markowitz 1952)
  2. Risk Parity / Equal Risk Contribution (Maillard, Roncalli & Teïletche 2010)
  3. Maximum Diversification Ratio (Choueifaty & Coignard 2008)
  4. Hierarchical Risk Parity (López de Prado 2016)

Plus:
  - Sector weight cap enforcement (no GICS sector > 35%)
  - Composite scoring to auto-pick the best method
  - Simplex projection for constrained optimization

References:
  - Markowitz (1952) "Portfolio Selection"
  - Maillard, Roncalli, Teïletche (2010) "Equally Weighted Risk Contribution"
  - Choueifaty & Coignard (2008) "Toward Maximum Diversification"
  - López de Prado (2016) "Building Diversified Portfolios that Outperform OOS"
  - Duchi et al. (2008) simplex projection
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.statistical_screen import SECTOR_MAP

MAX_SECTOR_WEIGHT = 0.35


# ═══════════════════════════════════════════════
# SIMPLEX PROJECTION (Duchi et al. 2008)
# ═══════════════════════════════════════════════

def project_simplex(v):
    """Project vector v onto the probability simplex (sum=1, all ≥ 0)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cumsum = np.cumsum(u)
    rho = 0
    for j in range(n):
        if u[j] - (cumsum[j] - 1) / (j + 1) > 0:
            rho = j + 1
    theta = (cumsum[rho - 1] - 1) / rho
    return np.maximum(v - theta, 0)


# ═══════════════════════════════════════════════
# METHOD 1: MINIMUM VARIANCE (Markowitz 1952)
# ═══════════════════════════════════════════════

def minimum_variance(cov_matrix):
    n = cov_matrix.shape[0]
    w = np.full(n, 1.0 / n)
    lr = 0.01

    for _ in range(5000):
        grad = 2 * cov_matrix @ w
        w = w - lr * grad
        w = project_simplex(w)

    return w


# ═══════════════════════════════════════════════
# METHOD 2: RISK PARITY (Maillard et al. 2010)
# ═══════════════════════════════════════════════

def risk_parity(cov_matrix):
    n = cov_matrix.shape[0]
    w = np.full(n, 1.0 / n)
    budgets = np.full(n, 1.0 / n)

    for _ in range(3000):
        sigma_w = cov_matrix @ w
        total_risk = np.sqrt(w @ sigma_w)
        if total_risk < 1e-15:
            break

        mrc = sigma_w / total_risk
        rc = w * mrc
        total_rc = rc.sum()
        target_rc = budgets * total_rc

        new_w = np.where(mrc > 1e-15, w * target_rc / (rc + 1e-15), w)
        w = new_w / new_w.sum()

        if np.max(np.abs(rc - target_rc)) < 1e-10:
            break

    return w


# ═══════════════════════════════════════════════
# METHOD 3: MAX DIVERSIFICATION (Choueifaty 2008)
# ═══════════════════════════════════════════════

def max_diversification(cov_matrix):
    n = cov_matrix.shape[0]
    vols = np.sqrt(np.diag(cov_matrix))
    w = np.full(n, 1.0 / n)
    lr = 0.005

    for _ in range(5000):
        sigma_w = cov_matrix @ w
        portfolio_vol = np.sqrt(w @ sigma_w)
        weighted_avg_vol = w @ vols
        if portfolio_vol < 1e-15:
            break

        grad = vols / portfolio_vol - weighted_avg_vol * sigma_w / (portfolio_vol ** 3)
        w = w + lr * grad
        w = project_simplex(w)

    return w


# ═══════════════════════════════════════════════
# METHOD 4: HRP (López de Prado 2016)
# ═══════════════════════════════════════════════

def _single_linkage_clustering(dist_matrix, n):
    """Single-linkage agglomerative clustering."""
    clusters = {i: [i] for i in range(n)}
    dist = dist_matrix.copy()
    active = set(range(n))
    merges = []
    merge_clusters = {}
    next_id = n

    while len(active) > 1:
        min_dist = np.inf
        best_i, best_j = -1, -1
        active_list = sorted(active)

        for a_idx in range(len(active_list)):
            for b_idx in range(a_idx + 1, len(active_list)):
                i, j = active_list[a_idx], active_list[b_idx]
                if dist[i, j] < min_dist:
                    min_dist = dist[i, j]
                    best_i, best_j = i, j

        new_cluster = clusters.get(best_i, []) + clusters.get(best_j, [])
        clusters[next_id] = new_cluster
        merge_clusters[next_id] = {"left": best_i, "right": best_j}
        merges.append((best_i, best_j, min_dist, len(new_cluster)))

        # Expand dist matrix
        new_size = next_id + 1
        if new_size > dist.shape[0]:
            new_dist = np.full((new_size, new_size), np.inf)
            new_dist[:dist.shape[0], :dist.shape[1]] = dist
            dist = new_dist

        for k in active:
            if k == best_i or k == best_j:
                continue
            d = min(dist[k, best_i], dist[k, best_j])
            dist[k, next_id] = d
            dist[next_id, k] = d

        active.discard(best_i)
        active.discard(best_j)
        active.add(next_id)
        clusters.pop(best_i, None)
        clusters.pop(best_j, None)
        next_id += 1

    return merges, merge_clusters, n


def _get_quasi_diag_order(merge_clusters, n, total_merges):
    """Get quasi-diagonal ordering from hierarchical clustering."""
    if total_merges == 0:
        return list(range(n))

    root_id = n + total_merges - 1

    def get_leaves(node_id):
        if node_id < n:
            return [node_id]
        node = merge_clusters.get(node_id)
        if not node:
            return [node_id] if node_id < n else []
        return get_leaves(node["left"]) + get_leaves(node["right"])

    return get_leaves(root_id)


def _cluster_variance(cov_matrix, indices):
    """Compute variance of a cluster using inverse-variance weights."""
    if len(indices) == 1:
        return cov_matrix[indices[0], indices[0]]

    variances = np.array([cov_matrix[i, i] for i in indices])
    inv_var = 1.0 / (variances + 1e-15)
    total_inv = inv_var.sum()
    w = inv_var / total_inv

    port_var = 0.0
    for a in range(len(indices)):
        for b in range(len(indices)):
            port_var += w[a] * w[b] * cov_matrix[indices[a], indices[b]]
    return port_var


def _recursive_bisection(cov_matrix, sorted_idx, weights):
    """Recursive bisection — allocate weights based on inverse variance."""
    if len(sorted_idx) <= 1:
        return

    mid = len(sorted_idx) // 2
    left = sorted_idx[:mid]
    right = sorted_idx[mid:]

    left_var = _cluster_variance(cov_matrix, left)
    right_var = _cluster_variance(cov_matrix, right)

    total_inv = 1 / (left_var + 1e-15) + 1 / (right_var + 1e-15)
    left_alloc = (1 / (left_var + 1e-15)) / total_inv
    right_alloc = (1 / (right_var + 1e-15)) / total_inv

    for i in left:
        weights[i] *= left_alloc
    for i in right:
        weights[i] *= right_alloc

    if len(left) > 1:
        _recursive_bisection(cov_matrix, left, weights)
    if len(right) > 1:
        _recursive_bisection(cov_matrix, right, weights)


def hrp(cov_matrix, corr_matrix):
    """Hierarchical Risk Parity (López de Prado 2016)."""
    n = cov_matrix.shape[0]

    # Distance matrix from correlation
    dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    np.fill_diagonal(dist_matrix, 0)

    # Clustering
    merges, merge_clusters, _ = _single_linkage_clustering(dist_matrix, n)

    # Quasi-diagonal ordering
    sorted_idx = _get_quasi_diag_order(merge_clusters, n, len(merges))

    # Recursive bisection
    weights = np.ones(n)
    _recursive_bisection(cov_matrix, sorted_idx, weights)

    # Normalize
    return weights / weights.sum()


# ═══════════════════════════════════════════════
# SECTOR WEIGHT CAP ENFORCEMENT
# ═══════════════════════════════════════════════

def enforce_sector_cap(weights, tickers):
    """Iterative proportional fitting to cap sector weights at MAX_SECTOR_WEIGHT."""
    w = weights.copy()
    n = len(w)

    sector_indices = {}
    for i, t in enumerate(tickers):
        sec = SECTOR_MAP.get(t, "Unknown")
        sector_indices.setdefault(sec, []).append(i)

    for _ in range(10):
        capped = False
        for sec, indices in sector_indices.items():
            total_sec_w = sum(w[i] for i in indices)
            if total_sec_w > MAX_SECTOR_WEIGHT:
                scale = MAX_SECTOR_WEIGHT / total_sec_w
                excess = total_sec_w - MAX_SECTOR_WEIGHT

                for i in indices:
                    w[i] *= scale

                other_indices = [i for i in range(n) if i not in indices]
                other_total = sum(w[i] for i in other_indices)
                if other_total > 0:
                    for i in other_indices:
                        w[i] += excess * (w[i] / other_total)
                capped = True

        if not capped:
            break

    return w / w.sum()


# ═══════════════════════════════════════════════
# PORTFOLIO METRICS
# ═══════════════════════════════════════════════

def portfolio_metrics(weights, returns_dict, tickers):
    """Compute portfolio-level metrics for a given weight vector."""
    n = len(tickers)
    min_len = min(len(returns_dict.get(t, [])) for t in tickers)
    returns = [returns_dict[t][:min_len] for t in tickers]

    # Combined daily returns
    portfolio_daily = []
    for t in range(min_len):
        day_r = sum(weights[i] * returns[i][t] for i in range(n))
        portfolio_daily.append(day_r)

    # Equity curve
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    equity_curve = []

    for r in portfolio_daily:
        equity += r
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualized √252)
    avg_r = np.mean(portfolio_daily)
    std_r = np.std(portfolio_daily, ddof=0)
    sharpe = (avg_r / std_r) * np.sqrt(252) if std_r > 0 else 0

    calmar = equity / max_dd if max_dd > 0 else 0

    # Sortino
    downside = [r for r in portfolio_daily if r < 0]
    downside_dev = np.sqrt(sum(r ** 2 for r in downside) / len(portfolio_daily)) if downside else 0
    sortino = (avg_r / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0

    # Diversification ratio
    asset_vols = [np.std(returns[i], ddof=0) for i in range(n)]
    weighted_avg_vol = sum(weights[i] * asset_vols[i] for i in range(n))
    div_ratio = weighted_avg_vol / std_r if std_r > 0 else 1

    # Equity curve smoothness (R² of linear regression)
    n_points = len(equity_curve)
    if n_points > 2:
        x_mean = (n_points - 1) / 2
        y_mean = np.mean(equity_curve)
        ss_xy = sum((i - x_mean) * (equity_curve[i] - y_mean) for i in range(n_points))
        ss_xx = sum((i - x_mean) ** 2 for i in range(n_points))
        ss_yy = sum((equity_curve[i] - y_mean) ** 2 for i in range(n_points))
        r2 = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0
    else:
        r2 = 0

    return {
        "totalR": round(equity, 2),
        "maxDD": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "divRatio": round(div_ratio, 2),
        "smoothness": round(r2, 3),
        "days": min_len,
    }


def composite_score(metrics):
    """
    Score a portfolio method (higher = better).
    Weights: 35% Sharpe + 25% Calmar + 25% Smoothness + 15% DivRatio
    """
    return (
        0.35 * metrics["sharpe"] +
        0.25 * metrics["calmar"] +
        0.25 * (metrics["smoothness"] * 5) +
        0.15 * metrics["divRatio"]
    )


# ═══════════════════════════════════════════════
# MAIN OPTIMIZER
# ═══════════════════════════════════════════════

def run_portfolio_optimization(risk_model):
    """
    Run all 4 portfolio optimization methods and pick the best.

    Args:
        risk_model: dict from run_sector_risk_model()

    Returns:
        best_method: name of best method
        best_weights: dict of {ticker: weight}
        all_results: dict of all methods' results
    """
    tickers = risk_model["tickers"]
    cov_matrix = risk_model["covMatrix"]
    corr_matrix = risk_model["corrMatrix"]
    returns_dict = risk_model["returnsDict"]
    n = len(tickers)

    if n < 3:
        print("  ⚠️ Too few stocks for optimization. Using equal weights.")
        w = {t: 1.0 / n for t in tickers}
        return "Equal Weight", w, {}

    print(f"\n{'═' * 60}")
    print(f"  📐 PORTFOLIO OPTIMIZATION ({n} stocks)")
    print(f"{'═' * 60}\n")

    methods = [
        ("Minimum Variance", lambda: minimum_variance(cov_matrix)),
        ("Risk Parity", lambda: risk_parity(cov_matrix)),
        ("Max Diversification", lambda: max_diversification(cov_matrix)),
        ("HRP (López de Prado)", lambda: hrp(cov_matrix, corr_matrix)),
    ]

    results = {}

    for name, fn in methods:
        print(f"  ━━━ {name.upper()} ━━━")

        raw_weights = fn()
        weights = enforce_sector_cap(raw_weights, tickers)
        metrics = portfolio_metrics(weights, returns_dict, tickers)
        score = composite_score(metrics)

        # Non-zero allocations
        allocations = [
            {
                "ticker": tickers[i],
                "sector": SECTOR_MAP.get(tickers[i], "Unknown"),
                "weight": round(float(weights[i]), 4),
            }
            for i in range(n)
            if weights[i] > 0.005
        ]
        allocations.sort(key=lambda x: x["weight"], reverse=True)

        # Sector summary
        sector_weights = {}
        for a in allocations:
            sector_weights[a["sector"]] = sector_weights.get(a["sector"], 0) + a["weight"]

        print(f"    Stocks: {len(allocations)} | Sharpe: {metrics['sharpe']:.2f} | "
              f"MaxDD: {metrics['maxDD']:.2f}R | Calmar: {metrics['calmar']:.2f} | "
              f"Score: {score:.3f}\n")

        results[name] = {
            "weights": weights,
            "allocations": allocations,
            "metrics": metrics,
            "compositeScore": round(score, 3),
            "sectorWeights": sector_weights,
        }

    # Compare methods
    print(f"\n{'═' * 80}")
    print("  METHOD COMPARISON")
    print(f"{'═' * 80}")
    print("  Method                 Sharpe  Sortino  MaxDD   Calmar  Smooth  DivR   Score  Stocks")
    print("  ──────                 ──────  ───────  ─────   ──────  ──────  ────   ─────  ──────")

    for name, data in results.items():
        m = data["metrics"]
        n_stocks = len(data["allocations"])
        print(
            f"  {name:<22s} {m['sharpe']:>6.2f}  {m['sortino']:>7.2f}  {m['maxDD']:>6.1f}  "
            f"{m['calmar']:>6.2f}  {m['smoothness']:>6.3f}  {m['divRatio']:>4.2f}   "
            f"{data['compositeScore']:>5.3f}  {n_stocks:>4d}"
        )

    # Pick best
    best_name = max(results, key=lambda k: results[k]["compositeScore"])
    best_data = results[best_name]

    print(f"\n  🏆 RECOMMENDED: {best_name} (Composite Score: {best_data['compositeScore']})")

    # Equal-weight baseline
    eq_w = np.full(n, 1.0 / n)
    eq_metrics = portfolio_metrics(eq_w, returns_dict, tickers)
    print(f"  📊 Equal-Weight Baseline: Sharpe={eq_metrics['sharpe']} MaxDD={eq_metrics['maxDD']}R")

    # Build weight dict
    best_weights = {tickers[i]: float(best_data["weights"][i]) for i in range(n)}

    return best_name, best_weights, results

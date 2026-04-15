"""
markov_regime.py — Markov Chain Regime Switching model for market regime detection.

Adapted from Quant Guild's IB live-streaming bot into a daily-bar backtesting
framework. Uses a 4-state Hidden Markov Model with Gaussian emissions calibrated
on historical SPY + VIX data.

Key adaptations from the reference code:
  - 4 states (BULL/NEUTRAL/BEAR/CRISIS) instead of 3 (LOW/MED/HIGH)
  - Multi-feature emissions (realized vol + VIX) instead of single bar range
  - Rolling 1-year calibration window (adaptive)
  - Vectorized daily-bar processing instead of tick-by-tick
  - Outputs both discrete labels AND continuous exposure scores
  - Strict in-sample calibration (no look-ahead bias)

Reference: romanmichaelpaolucci/Quant-Guild-Library
  "74. How to Build a Markov Chain Regime Switching Bot in Python"
"""

import numpy as np
import pandas as pd
from config import (
    BULL, NEUTRAL, BEAR, CRISIS,
    IN_SAMPLE_END, SPY_TICKER, VIX_TICKER
)


# ═══════════════════════════════════════════════════════════════════════
# MARKOV CHAIN REGIME MODEL
# ═══════════════════════════════════════════════════════════════════════

class MarkovRegimeModel:
    """
    4-state Hidden Markov Model for market regime classification.

    States:
        0 = BULL    (low volatility, strong uptrend)
        1 = NEUTRAL (moderate volatility, no clear trend)
        2 = BEAR    (elevated volatility, downtrend)
        3 = CRISIS  (extreme volatility, market panic)

    The model uses:
        - Transition matrix: P(next_state | current_state) — captures regime persistence
        - Emission distributions: P(observed_features | state) — Gaussian per feature per state
        - Bayesian filtering: posterior ∝ prior × likelihood — sequential belief update

    Calibration uses percentile-based clustering on historical volatility features
    to estimate emission parameters and transition probabilities.
    """

    STATE_NAMES = [BULL, NEUTRAL, BEAR, CRISIS]
    N_STATES = 4

    def __init__(self):
        self.n_states = self.N_STATES

        # Current belief: P(state | all observations so far)
        # Start uniform — no prior information
        self.state_probs = np.array([0.25, 0.25, 0.25, 0.25])

        # Current MAP (maximum a posteriori) state
        self.current_state = 0  # BULL

        # ─── Transition Matrix (Markov property) ───
        # T[i, j] = P(next_state = j | current_state = i)
        # High diagonal = regimes are "sticky" (persist over multiple days)
        # Hard to jump from BULL→CRISIS directly (goes through NEUTRAL/BEAR)
        self.transition_matrix = np.array([
            #  BULL    NEUT    BEAR   CRISIS
            [0.92,   0.06,   0.015,  0.005],   # From BULL
            [0.08,   0.84,   0.06,   0.02 ],   # From NEUTRAL
            [0.02,   0.08,   0.85,   0.05 ],   # From BEAR
            [0.005,  0.025,  0.07,   0.90 ],   # From CRISIS
        ])

        # ─── Emission Parameters (Gaussian) ───
        # Each state emits observed features from a Gaussian distribution.
        # We use 2 features: [realized_vol_20d, vix_normalized]
        #
        # emission_means[state] = [mean_vol, mean_vix]
        # emission_stds[state]  = [std_vol,  std_vix]
        #
        # These are DEFAULTS — will be overwritten by calibration
        self.n_features = 2
        self.emission_means = np.array([
            [0.008, 0.30],   # BULL:    low vol, low VIX
            [0.014, 0.45],   # NEUTRAL: moderate
            [0.022, 0.65],   # BEAR:    elevated
            [0.040, 0.85],   # CRISIS:  extreme
        ])
        self.emission_stds = np.array([
            [0.004, 0.10],   # BULL
            [0.005, 0.12],   # NEUTRAL
            [0.008, 0.15],   # BEAR
            [0.015, 0.20],   # CRISIS
        ])

        # Calibration metadata
        self.is_calibrated = False
        self.calibration_date = None

    def calibrate(self, features_df):
        """
        Calibrate emission parameters and transition matrix from historical features.

        Uses percentile-based clustering to assign regime labels, then estimates
        Gaussian emission parameters and transition probabilities.

        Args:
            features_df: DataFrame with columns ['realized_vol', 'vix_norm']
                         indexed by date. Should be IN-SAMPLE data only.
        """
        if len(features_df) < 50:
            print("  ⚠️ Insufficient data for Markov calibration")
            return

        vol = features_df['realized_vol'].values
        vix = features_df['vix_norm'].values

        # Remove NaN rows
        valid = np.isfinite(vol) & np.isfinite(vix)
        vol = vol[valid]
        vix = vix[valid]

        if len(vol) < 50:
            return

        # ─── Step 1: Percentile-based regime assignment ───
        # Use a combined stress score (average of vol and VIX percentile ranks)
        # to assign each day to a regime
        vol_pct = np.searchsorted(np.sort(vol), vol) / len(vol)
        vix_pct = np.searchsorted(np.sort(vix), vix) / len(vix)
        stress_score = 0.5 * vol_pct + 0.5 * vix_pct

        # Assign regimes using ASYMMETRIC percentiles that reflect
        # historical market regime distribution:
        #   ~60% BULL, ~20% NEUTRAL, ~13% BEAR, ~7% CRISIS
        # Equal quartiles (25/25/25/25) would massively over-classify stress
        p60, p80, p93 = np.percentile(stress_score, [60, 80, 93])
        regime_assignments = np.zeros(len(stress_score), dtype=int)
        regime_assignments[stress_score >= p60] = 1   # NEUTRAL
        regime_assignments[stress_score >= p80] = 2   # BEAR
        regime_assignments[stress_score >= p93] = 3   # CRISIS

        # ─── Step 2: Estimate emission parameters per regime ───
        features = np.column_stack([vol, vix])

        for state in range(self.n_states):
            mask = regime_assignments == state
            state_features = features[mask]

            if len(state_features) >= 5:
                self.emission_means[state] = np.mean(state_features, axis=0)
                self.emission_stds[state] = np.maximum(
                    np.std(state_features, axis=0), 1e-6
                )

        # Ensure emission means are monotonically increasing (stress-ordered)
        # Sort by the average of both feature means
        avg_stress = self.emission_means.mean(axis=1)
        sort_idx = np.argsort(avg_stress)
        self.emission_means = self.emission_means[sort_idx]
        self.emission_stds = self.emission_stds[sort_idx]

        # ─── Step 3: Estimate transition matrix from regime sequence ───
        transition_counts = np.zeros((self.n_states, self.n_states))
        for t in range(1, len(regime_assignments)):
            prev = regime_assignments[t - 1]
            curr = regime_assignments[t]
            transition_counts[prev, curr] += 1

        # Normalize with Laplace smoothing (prevents zero probabilities)
        for i in range(self.n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                smoothing = 0.1
                self.transition_matrix[i] = (
                    (transition_counts[i] + smoothing) /
                    (row_sum + smoothing * self.n_states)
                )

        # Reset belief state after calibration
        self.state_probs = np.array([0.25, 0.25, 0.25, 0.25])
        self.is_calibrated = True
        self.calibration_date = features_df.index[-1] if hasattr(features_df.index, '__len__') else None

        print(f"  ✅ Markov model calibrated on {len(vol)} days")
        print(f"     Emission means (vol, vix):")
        for i, name in enumerate(self.STATE_NAMES):
            print(f"       {name:>8s}: vol={self.emission_means[i,0]:.4f}, "
                  f"vix={self.emission_means[i,1]:.2f}")
        print(f"     Transition matrix diagonal: "
              f"{[f'{self.transition_matrix[i,i]:.2f}' for i in range(4)]}")

    def _gaussian_likelihood(self, obs, state):
        """
        Compute the joint probability density of observing feature vector `obs`
        given `state`, assuming independent Gaussian features.

        P(obs | state) = Π_k N(obs_k; μ_k, σ_k)

        Args:
            obs: np.array of shape (n_features,) — observed feature values
            state: int — regime index (0-3)

        Returns:
            float — joint likelihood (product of per-feature Gaussian PDFs)
        """
        likelihood = 1.0
        for k in range(self.n_features):
            mean = self.emission_means[state, k]
            std = self.emission_stds[state, k]

            # Gaussian PDF
            coeff = 1.0 / (std * np.sqrt(2.0 * np.pi))
            exponent = -0.5 * ((obs[k] - mean) / std) ** 2
            pdf = coeff * np.exp(exponent)

            # Clamp to prevent numerical underflow
            likelihood *= max(pdf, 1e-300)

        return likelihood

    def filter_step(self, observation):
        """
        Perform one step of Bayesian filtering (predict → observe → update).

        This is the core inference step of the HMM, applied sequentially
        to each new observation (daily bar).

        Args:
            observation: np.array of shape (n_features,) — [realized_vol, vix_norm]

        Returns:
            dict with:
                - state: int (MAP regime index)
                - state_name: str (BULL/NEUTRAL/BEAR/CRISIS)
                - state_probs: np.array (posterior probabilities)
                - exposure_score: float (probability-weighted exposure, 0.0-1.0)
        """
        # Skip if observation has NaN
        if np.any(~np.isfinite(observation)):
            return {
                'state': self.current_state,
                'state_name': self.STATE_NAMES[self.current_state],
                'state_probs': self.state_probs.copy(),
                'exposure_score': self._compute_exposure(self.state_probs),
            }

        # ─── Step 1: PREDICTION (Time Update) ───
        # prior[j] = Σ_i T[i,j] × state_probs[i]
        # "Where might we be, given where we were?"
        prior_probs = self.transition_matrix.T @ self.state_probs

        # ─── Step 2: EMISSION LIKELIHOODS ───
        # likelihoods[j] = P(observation | state = j)
        likelihoods = np.array([
            self._gaussian_likelihood(observation, j)
            for j in range(self.n_states)
        ])

        # ─── Step 3: UPDATE (Bayes' Rule) ───
        # posterior ∝ prior × likelihood
        posterior = prior_probs * likelihoods

        # Normalize
        total = posterior.sum()
        if total > 0:
            posterior = posterior / total
        else:
            # Fallback: use prior if all likelihoods are ~0
            posterior = prior_probs

        # Save updated belief
        self.state_probs = posterior

        # ─── Step 4: MAP Estimate ───
        self.current_state = int(np.argmax(posterior))

        return {
            'state': self.current_state,
            'state_name': self.STATE_NAMES[self.current_state],
            'state_probs': posterior.copy(),
            'exposure_score': self._compute_exposure(posterior),
        }

    def _compute_exposure(self, probs):
        """
        Compute a continuous exposure score from regime probabilities.

        Exposure = weighted sum of probabilities × regime exposure values:
            E = P(BULL)×1.0 + P(NEUTRAL)×0.7 + P(BEAR)×0.3 + P(CRISIS)×0.0

        This replaces the discrete REGIME_EXPOSURE lookup with a smooth,
        probability-weighted version that avoids abrupt exposure changes.

        Returns: float in [0.0, 1.0]
        """
        exposure_weights = np.array([1.0, 0.7, 0.3, 0.0])
        return float(np.dot(probs, exposure_weights))

    def reset(self):
        """Reset the belief state to uniform (for a fresh filtering pass)."""
        self.state_probs = np.array([0.25, 0.25, 0.25, 0.25])
        self.current_state = 0


# ═══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def compute_markov_features(spy_close, vix_close):
    """
    Compute the feature matrix for the Markov regime model.

    Features (starting simple — 2 features):
        1. realized_vol: 20-day rolling standard deviation of SPY log returns
        2. vix_norm: VIX close normalized to [0, 1] range using historical min/max

    Args:
        spy_close: pd.Series of SPY closing prices
        vix_close: pd.Series of VIX closing prices

    Returns:
        pd.DataFrame with columns ['realized_vol', 'vix_norm'], indexed by date
    """
    # Align dates
    common_dates = spy_close.index.intersection(vix_close.index)
    spy = spy_close.loc[common_dates].sort_index()
    vix = vix_close.loc[common_dates].sort_index()

    # Feature 1: 20-day realized volatility of SPY returns
    spy_returns = np.log(spy / spy.shift(1))
    realized_vol = spy_returns.rolling(20).std()

    # Feature 2: VIX normalized to [0, 1] using fixed domain-appropriate scale
    # VIX historically ranges from ~9 (extreme calm) to ~80+ (panic)
    # Using a FIXED scale avoids look-ahead bias and prevents the expanding
    # min/max from compressing moderate readings into the "stressed" zone.
    # Floor=9 (historical low), Ceiling=80 (covers all but Black Monday)
    VIX_FLOOR = 9.0
    VIX_CEILING = 80.0
    vix_norm = (vix - VIX_FLOOR) / (VIX_CEILING - VIX_FLOOR)
    vix_norm = vix_norm.clip(0.0, 1.0)

    features = pd.DataFrame({
        'realized_vol': realized_vol,
        'vix_norm': vix_norm,
    }, index=common_dates)

    return features.dropna()


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE-COMPATIBLE REGIME FILTER
# ═══════════════════════════════════════════════════════════════════════

def compute_regime_markov(market_data, calibration_end=None):
    """
    Compute regime labels using the Markov Chain model.

    Drop-in replacement for compute_regime() in regime_filter.py.
    Returns the same output format: pd.Series with values BULL/NEUTRAL/BEAR/CRISIS.

    Also returns an exposure_scores Series for graduated position sizing.

    Calibration workflow:
        1. Compute features from SPY + VIX
        2. Calibrate model on in-sample data (up to calibration_end)
        3. Run Bayesian filtering over ALL dates (sequentially, no look-ahead)

    Args:
        market_data: dict with keys SPY_TICKER and VIX_TICKER,
                     each containing a DataFrame with 'Close' column.
        calibration_end: str or Timestamp. Calibration uses data up to this date.
                         Defaults to IN_SAMPLE_END from config.

    Returns:
        tuple of (regime, exposure_scores):
            - regime: pd.Series of regime labels (BULL/NEUTRAL/BEAR/CRISIS)
            - exposure_scores: pd.Series of float exposure values (0.0-1.0)
    """
    if calibration_end is None:
        calibration_end = IN_SAMPLE_END

    calibration_end = pd.Timestamp(calibration_end)

    spy_close = market_data[SPY_TICKER]["Close"]
    vix_close = market_data[VIX_TICKER]["Close"]

    # ─── Step 1: Compute features ───
    print("  Computing Markov features (realized_vol, vix_norm)...")
    features = compute_markov_features(spy_close, vix_close)

    if len(features) < 100:
        raise ValueError(f"Insufficient feature data: {len(features)} days")

    # ─── Step 2: Calibrate on in-sample data ───
    is_features = features[features.index <= calibration_end]
    print(f"  Calibrating on {len(is_features)} in-sample days "
          f"({is_features.index[0].strftime('%Y-%m-%d')} → "
          f"{is_features.index[-1].strftime('%Y-%m-%d')})...")

    model = MarkovRegimeModel()
    model.calibrate(is_features)

    if not model.is_calibrated:
        raise RuntimeError("Markov model calibration failed")

    # ─── Step 3: Run Bayesian filtering over all dates ───
    # The filter is strictly causal — only uses past observations.
    # Parameters are frozen from IS calibration (no re-estimation in OOS).
    model.reset()

    regime_labels = []
    exposure_values = []
    dates = features.index.tolist()

    for date in dates:
        obs = features.loc[date].values  # [realized_vol, vix_norm]
        result = model.filter_step(obs)
        regime_labels.append(result['state_name'])
        exposure_values.append(result['exposure_score'])

    regime = pd.Series(regime_labels, index=dates, name="regime")
    exposure_scores = pd.Series(exposure_values, index=dates, name="exposure")

    # ─── Summary ───
    counts = regime.value_counts()
    total = len(regime)
    print(f"\n📊 Markov Regime classification ({total} days):")
    emojis = {BULL: "🟢", NEUTRAL: "🟡", BEAR: "🔴", CRISIS: "🚨"}
    for r in [BULL, NEUTRAL, BEAR, CRISIS]:
        n = counts.get(r, 0)
        pct = n / total * 100
        emoji = emojis.get(r, "❓")
        print(f"   {emoji} {r}: {n} days ({pct:.1f}%)")

    # Transition count (how often regime changes)
    transitions = (regime != regime.shift(1)).sum() - 1
    print(f"   🔄 Total transitions: {transitions} "
          f"(avg {total / max(transitions, 1):.0f} days per regime)")

    avg_exposure = exposure_scores.mean()
    print(f"   📊 Avg exposure score: {avg_exposure:.2f}")

    return regime, exposure_scores


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_loader import download_market_data

    print("=" * 60)
    print("  🧪 MARKOV REGIME MODEL — STANDALONE TEST")
    print("=" * 60)

    market_data = download_market_data()
    regime, exposure = compute_regime_markov(market_data)

    print(f"\nFirst 10 regime values:")
    print(regime.head(10))
    print(f"\nFirst 10 exposure values:")
    print(exposure.head(10))

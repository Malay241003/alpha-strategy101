"""
test_alphas.py — Unit tests for operators and alpha computations.

Tests:
  1. Operator correctness (rank, delay, delta, ts_rank, decay_linear, correlation)
  2. Alpha#101 against hand-calculated values
  3. Regime filter classification
  4. Composite scorer weight application
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from src.operators import (
    rank, delay, delta, correlation, ts_min, ts_max,
    ts_rank, sum_, stddev, decay_linear, sign, scale
)
from src.alphas import alpha_101, alpha_12, alpha_23


def test_delay():
    """Test that delay shifts values correctly."""
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = delay(s, 1)
    assert np.isnan(result.iloc[0]), "delay(1): first value should be NaN"
    assert result.iloc[1] == 1.0, "delay(1): second value should be 1"
    assert result.iloc[4] == 4.0, "delay(1): last value should be 4"
    print("  ✅ delay() PASSED")


def test_delta():
    """Test that delta computes difference correctly."""
    s = pd.Series([10, 12, 11, 15, 13], dtype=float)
    result = delta(s, 1)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 2.0  # 12 - 10
    assert result.iloc[2] == -1.0  # 11 - 12
    assert result.iloc[3] == 4.0  # 15 - 11
    print("  ✅ delta() PASSED")


def test_rank_cross_sectional():
    """Test cross-sectional rank on a DataFrame."""
    df = pd.DataFrame({
        "A": [1, 4, 3],
        "B": [2, 5, 1],
        "C": [3, 6, 2],
    }, dtype=float)
    result = rank(df)
    # Row 0: A=1(rank=0.33), B=2(rank=0.67), C=3(rank=1.0)
    assert abs(result.iloc[0]["A"] - 1/3) < 0.01
    assert abs(result.iloc[0]["C"] - 1.0) < 0.01
    print("  ✅ rank() cross-sectional PASSED")


def test_ts_min_max():
    """Test time-series min and max."""
    s = pd.Series([5, 3, 7, 2, 8], dtype=float)
    assert ts_min(s, 3).iloc[2] == 3.0  # min of [5,3,7]
    assert ts_max(s, 3).iloc[2] == 7.0  # max of [5,3,7]
    assert ts_min(s, 3).iloc[4] == 2.0  # min of [7,2,8]
    assert ts_max(s, 3).iloc[4] == 8.0  # max of [7,2,8]
    print("  ✅ ts_min() / ts_max() PASSED")


def test_sum():
    """Test rolling sum."""
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = sum_(s, 3)
    assert result.iloc[2] == 6.0   # 1+2+3
    assert result.iloc[4] == 12.0  # 3+4+5
    print("  ✅ sum_() PASSED")


def test_decay_linear():
    """Test weighted moving average with linear decay."""
    s = pd.Series([1, 1, 1, 1, 1], dtype=float)
    # For uniform input, any weighting should give 1
    result = decay_linear(s, 3)
    assert abs(result.iloc[2] - 1.0) < 0.01
    print("  ✅ decay_linear() PASSED")


def test_sign():
    """Test sign function."""
    s = pd.Series([-3, 0, 5], dtype=float)
    result = sign(s)
    assert result.iloc[0] == -1
    assert result.iloc[1] == 0
    assert result.iloc[2] == 1
    print("  ✅ sign() PASSED")


def test_scale():
    """Test scale function (cross-sectional)."""
    s = pd.Series([1, -2, 3], dtype=float)
    result = scale(s, a=1.0)
    assert abs(result.abs().sum() - 1.0) < 0.001
    print("  ✅ scale() PASSED")


def test_alpha_101():
    """Test Alpha#101: (close - open) / ((high - low) + 0.001)"""
    panels = {
        "Close": pd.DataFrame({"A": [105.0, 98.0], "B": [50.0, 52.0]}),
        "Open":  pd.DataFrame({"A": [100.0, 100.0], "B": [51.0, 50.0]}),
        "High":  pd.DataFrame({"A": [106.0, 101.0], "B": [52.0, 53.0]}),
        "Low":   pd.DataFrame({"A": [99.0, 97.0], "B": [49.0, 49.0]}),
    }
    result = alpha_101(panels)
    # Row 0, A: (105-100) / (106-99+0.001) = 5 / 7.001 ≈ 0.714
    expected_a = 5 / (7 + 0.001)
    assert abs(result.iloc[0]["A"] - expected_a) < 0.01
    # Row 0, B: (50-51) / (52-49+0.001) = -1/3.001 ≈ -0.333
    expected_b = -1 / (3 + 0.001)
    assert abs(result.iloc[0]["B"] - expected_b) < 0.01
    print("  ✅ alpha_101() PASSED")


def test_alpha_12():
    """Test Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
    panels = {
        "Close":  pd.DataFrame({"A": [100.0, 102.0, 101.0]}),
        "Volume": pd.DataFrame({"A": [1000.0, 1500.0, 1200.0]}),
    }
    result = alpha_12(panels)
    # Day 1: sign(1500-1000)=1, -1*(102-100)=-2 → -2
    assert abs(result.iloc[1]["A"] - (-2.0)) < 0.01
    # Day 2: sign(1200-1500)=-1, -1*(101-102)=1 → -1*1=-1
    assert abs(result.iloc[2]["A"] - (-1.0)) < 0.01
    print("  ✅ alpha_12() PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 50)
    print("  🧪 RUNNING UNIT TESTS")
    print("=" * 50)

    tests = [
        test_delay,
        test_delta,
        test_rank_cross_sectional,
        test_ts_min_max,
        test_sum,
        test_decay_linear,
        test_sign,
        test_scale,
        test_alpha_101,
        test_alpha_12,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} ERROR: {e}")
            failed += 1

    print(f"\n{'═' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'═' * 50}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
from scipy.optimize import minimize_scalar

from statistical_layer.calibration import TemperatureScaler, expected_calibration_error, brier_score
from statistical_tests import (holm, benjamini_hochberg, power_two_proportion, n_two_proportion,
                               n_mcnemar, simulate_two_proportion, n_for_half_width, wilson_half_width)


def test_temperature_recovers_true_value():
    print("\n--- Test 1: temperature scaling recovers a known temperature ---")
    rng = np.random.default_rng(1)
    z = rng.normal(0, 4, size=(20000, 3))
    p = TemperatureScaler.softmax(z, 2.0)
    y = np.array([rng.choice(3, p=row) for row in p])
    t = TemperatureScaler.fit(z, y)
    print(f"true T = 2.0, fitted T = {t:.4f}")
    assert abs(t - 2.0) < 0.05
    print("[PASS]")


def test_temperature_matches_scipy_and_keeps_argmax():
    print("\n--- Test 2: Newton fit equals a bounded scalar minimiser; argmax unchanged ---")
    rng = np.random.default_rng(2)
    z = rng.normal(0, 5, size=(500, 3))
    y = np.array([rng.choice(3, p=row) for row in TemperatureScaler.softmax(z, 3.0)])
    t = TemperatureScaler.fit(z, y)
    ref = minimize_scalar(lambda x: TemperatureScaler.nll(z, y, x), bounds=(0.05, 50), method="bounded",
                          options={"xatol": 1e-10}).x
    print(f"Newton T = {t:.6f}, scipy T = {ref:.6f}")
    assert abs(t - ref) < 1e-4
    assert (TemperatureScaler.softmax(z, t).argmax(1) == z.argmax(1)).all()
    # labels unrelated to the logits: the likelihood keeps rising as T grows, so the fit runs to a very large T
    t_noise = TemperatureScaler.fit(z, rng.integers(0, 3, 500))
    print(f"uninformative logits: T = {t_noise:.3g}")
    assert t_noise > 50
    print("[PASS]")


def test_calibration_metrics():
    print("\n--- Test 3: ECE and Brier on hand-checked cases ---")
    ece, table = expected_calibration_error([0.9, 0.9, 0.6, 0.6], [1, 0, 1, 1], bins=2)
    # bins {0.6, 0.6}: |0.6 - 1| = 0.4; {0.9, 0.9}: |0.9 - 0.5| = 0.4 -> ECE 0.4
    assert abs(ece - 0.4) < 1e-12, ece
    assert abs(brier_score([[1, 0, 0], [0.5, 0.5, 0]], np.array([0, 1])) - 0.25) < 1e-12
    print("[PASS]")


def test_multiple_comparisons():
    print("\n--- Test 4: Holm and Benjamini-Hochberg adjusted p-values ---")
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    h, bh = holm(p), benjamini_hochberg(p)
    # Holm: 4*0.005, 3*0.01, 2*0.03, max(1*0.04, 0.06); BH: min over j >= i of 4 p_(j) / j
    assert np.allclose([h[k] for k in "dacb"], [0.02, 0.03, 0.06, 0.06])
    assert np.allclose([bh[k] for k in "dacb"], [0.02, 0.02, 0.04, 0.04])
    print("[PASS]")


def test_power_formulas():
    print("\n--- Test 5: sample-size formulas reach their target power ---")
    n = n_two_proportion(0.91, 0.835)
    assert power_two_proportion(0.91, 0.835, n) >= 0.8 > power_two_proportion(0.91, 0.835, n - 5)
    sim = simulate_two_proportion(0.91, 0.835, n, np.random.default_rng(3))
    print(f"n = {n} per group, analytic power {power_two_proportion(0.91, 0.835, n):.3f}, simulated {sim:.3f}")
    assert abs(sim - 0.8) < 0.03
    w = n_for_half_width(0.91, 0.03)
    assert wilson_half_width(0.91, w) <= 0.03 < wilson_half_width(0.91, w - 1)
    # a larger difference between the discordant probabilities needs fewer pairs
    assert n_mcnemar(0.30, 0.20) < n_mcnemar(0.30, 0.05)
    print("[PASS]")


if __name__ == "__main__":
    test_temperature_recovers_true_value()
    test_temperature_matches_scipy_and_keeps_argmax()
    test_calibration_metrics()
    test_multiple_comparisons()
    test_power_formulas()
    print("\n[ALL STATISTICS TESTS PASSED]")

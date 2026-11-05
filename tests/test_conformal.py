import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conformal import (  # noqa: E402
    absolute_residual_scores,
    aci_thresholds,
    block_bootstrap_interval,
    conformal_threshold,
    covered,
    cqr_scores,
    fixed_thresholds,
    rolling_thresholds,
    worst_block_coverage,
)

ALPHA = 0.1


def drifting_scores(rng, n_cal=1000, n_test=4000, factor=2.0):
    """Scores whose scale grows after calibration, so a fixed threshold under-covers."""
    calibration = np.abs(rng.normal(size=n_cal))
    test = np.abs(rng.normal(size=n_test)) * factor
    return calibration, test


def test_threshold_uses_finite_sample_correction():
    assert conformal_threshold(np.arange(1, 10) / 10, ALPHA) == 0.9


def test_threshold_is_infinite_when_calibration_is_too_small():
    assert conformal_threshold(np.array([0.1, 0.2]), ALPHA) == float("inf")


def test_scores_have_expected_values():
    assert absolute_residual_scores(np.array([3.0]), np.array([1.0]))[0] == 2.0
    y = np.array([0.0, 5.0, 12.0])
    lower, upper = np.full(3, 2.0), np.full(3, 10.0)
    assert cqr_scores(y, lower, upper).tolist() == [2.0, -3.0, 2.0]


def test_fixed_threshold_covers_exchangeable_data():
    rng = np.random.default_rng(0)
    calibration, test = np.abs(rng.normal(size=5000)), np.abs(rng.normal(size=20000))
    hits = covered(test, fixed_thresholds(calibration, len(test), ALPHA))
    assert abs(hits.mean() - 0.9) < 0.015


def test_fixed_threshold_undercovers_after_drift():
    rng = np.random.default_rng(1)
    calibration, test = drifting_scores(rng)
    assert covered(test, fixed_thresholds(calibration, len(test), ALPHA)).mean() < 0.7


def test_rolling_recovers_coverage_after_drift():
    rng = np.random.default_rng(2)
    calibration, test = drifting_scores(rng)
    thresholds = rolling_thresholds(calibration, test, ALPHA, window=500, refresh=50)
    assert covered(test, thresholds)[1000:].mean() > 0.87


def test_rolling_does_not_look_ahead():
    rng = np.random.default_rng(3)
    calibration, test = drifting_scores(rng, n_test=400)
    before = rolling_thresholds(calibration, test, ALPHA, window=200, refresh=50)
    changed = test.copy()
    changed[100:] += 100.0
    after = rolling_thresholds(calibration, changed, ALPHA, window=200, refresh=50)
    assert np.array_equal(before[:100], after[:100])
    assert not np.array_equal(before[100:], after[100:])


def test_rolling_uses_only_the_window():
    calibration, test = np.zeros(100), np.arange(1.0, 101.0)
    thresholds = rolling_thresholds(calibration, test, ALPHA, window=10, refresh=50)
    assert thresholds[0] == 0.0
    assert thresholds[50] == 50.0  # the last 10 scores are 41 to 50


def test_aci_does_not_look_ahead_and_reacts_to_drift():
    rng = np.random.default_rng(4)
    calibration, test = drifting_scores(rng)
    fixed = covered(test, fixed_thresholds(calibration, len(test), ALPHA)).mean()
    thresholds = aci_thresholds(calibration, test, ALPHA, gamma=0.01)
    assert covered(test, thresholds).mean() > fixed + 0.1
    changed = test.copy()
    changed[100:] += 100.0
    assert np.array_equal(thresholds[:101], aci_thresholds(calibration, changed, ALPHA, gamma=0.01)[:101])


def test_worst_block_finds_the_lowest_block():
    hits = np.array([1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    assert worst_block_coverage(hits, 2) == 0.0


def test_bootstrap_interval_contains_the_mean():
    rng = np.random.default_rng(5)
    hits = rng.random(5000) < 0.9
    low, high = block_bootstrap_interval(hits, 100)
    assert low < hits.mean() < high

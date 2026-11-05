"""Conformal prediction intervals for regression when data arrive over time.

All functions use plain NumPy, so they can be tested without a model. Scores and
thresholds are always in time order.
"""
from __future__ import annotations

import math

import numpy as np


def absolute_residual_scores(y: np.ndarray, median: np.ndarray) -> np.ndarray:
    """Score for intervals of fixed half-width around the median."""
    return np.abs(y - median)


def cqr_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Conformalized quantile regression score: how far y falls outside the band."""
    return np.maximum(lower - y, y - upper)


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of the scores."""
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return math.inf
    return float(np.sort(scores)[rank - 1])


def fixed_thresholds(calibration: np.ndarray, n_test: int, alpha: float) -> np.ndarray:
    """One threshold from the calibration block, never updated."""
    return np.full(n_test, conformal_threshold(calibration, alpha))


def rolling_thresholds(
    calibration: np.ndarray, test: np.ndarray, alpha: float, window: int, refresh: int
) -> np.ndarray:
    """Recompute the threshold every `refresh` steps from the latest `window` scores.

    A threshold only uses scores from before the start of its refresh block.
    """
    history = np.concatenate([calibration, test])
    start = len(calibration)
    thresholds = np.empty(len(test))
    for block_start in range(0, len(test), refresh):
        end = start + block_start
        recent = history[max(0, end - window) : end]
        thresholds[block_start : block_start + refresh] = conformal_threshold(recent, alpha)
    return thresholds


def aci_thresholds(calibration: np.ndarray, test: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Adaptive conformal inference (Gibbs and Candes, 2021).

    The miscoverage level moves after each step: up after a covered point and down
    after a miss. Levels outside [0, 1] are clipped when reading the quantile, so a
    threshold stays between the smallest and largest calibration score.
    """
    ordered = np.sort(calibration)
    n = len(ordered)
    level = alpha
    thresholds = np.empty(len(test))
    for t, score in enumerate(test):
        rank = math.ceil((n + 1) * (1 - level))
        thresholds[t] = ordered[min(max(rank, 1), n) - 1]
        missed = float(score > thresholds[t])
        level += gamma * (alpha - missed)
    return thresholds


def covered(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """A point is covered when its score does not exceed its threshold."""
    return scores <= thresholds


def worst_block_coverage(hits: np.ndarray, block: int) -> float:
    """Lowest coverage over consecutive complete blocks."""
    usable = len(hits) // block
    if usable == 0:
        return float(hits.mean())
    return float(hits[: usable * block].reshape(usable, block).mean(axis=1).min())


def block_bootstrap_interval(
    hits: np.ndarray, block: int, repeats: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for coverage, resampling whole blocks to respect time dependence."""
    usable = max(len(hits) // block, 1)
    block_size = len(hits) // usable
    means = hits[: usable * block_size].reshape(usable, block_size).mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, size=(repeats, usable), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)

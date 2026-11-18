"""Small, dependency-light statistics used by calibration and the regression gate."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for ``stat`` over items. Returns (point, lo, hi)."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for a metric of paired (label, prediction) arrays, resampling items."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, size=n)
        if len(np.unique(y_true[i])) < 2:
            continue
        boots.append(metric(y_true[i], y_score[i]))
    return (
        float(metric(y_true, y_score)),
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap of mean(candidate - baseline) over the same items.

    Returns the delta, its CI and a one-sided p-value for "candidate is worse"
    (fraction of bootstrap deltas >= 0, i.e. how often the drop disappears).
    """
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(candidate, dtype=float)
    if b.shape != c.shape:
        raise ValueError("baseline and candidate must be scored on the same items")
    d = c - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "delta": float(d.mean()),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "p_worse": float((boots >= 0).mean()),
    }


def mcnemar_exact(baseline_pass: Sequence[bool], candidate_pass: Sequence[bool]) -> dict[str, float]:
    """Exact (binomial) McNemar test on paired pass/fail outcomes."""
    b = np.asarray(baseline_pass, dtype=bool)
    c = np.asarray(candidate_pass, dtype=bool)
    n01 = int((b & ~c).sum())  # regressions: baseline passed, candidate failed
    n10 = int((~b & c).sum())  # fixes
    n = n01 + n10
    if n == 0:
        return {"regressions": 0, "fixes": 0, "p_value": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"regressions": n01, "fixes": n10, "p_value": float(min(1.0, 2 * p))}

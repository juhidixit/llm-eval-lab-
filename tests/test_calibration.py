import numpy as np
import pytest

from evallab import calibration as cal


def test_confusion_and_f1():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([1, 0, 0, 1, 1])
    assert cal.confusion(y, p) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    prec, rec, f = cal.precision_recall_f1(y, p)
    assert prec == pytest.approx(2 / 3) and rec == pytest.approx(2 / 3) and f == pytest.approx(2 / 3)


def test_kappa_perfect_and_chance():
    y = np.array([0, 1] * 50)
    assert cal.cohen_kappa(y, y) == pytest.approx(1.0)
    assert cal.cohen_kappa(y, 1 - y) == pytest.approx(-1.0)
    assert cal.cohen_kappa(y, np.ones_like(y)) == pytest.approx(0.0)


def test_roc_auc_matches_bruteforce_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200) + 0.3 * y, 1)  # coarse rounding creates ties
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert cal.roc_auc(y, s) == pytest.approx(brute)


def test_ece_zero_for_perfectly_calibrated_bins():
    y = np.array([1] * 8 + [0] * 2 + [1] * 2 + [0] * 8)
    p = np.array([0.8] * 10 + [0.2] * 10)
    assert cal.expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_platt_scaling_reduces_ece_of_overconfident_scores():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.2, 0.8, 4000)
    y = rng.random(4000) < true_p
    overconfident = 1 / (1 + np.exp(-4 * np.log(true_p / (1 - true_p))))  # pushes towards 0/1
    before = cal.expected_calibration_error(y, overconfident)
    scaler = cal.PlattScaler().fit(overconfident[:2000], y[:2000], iters=3000, lr=0.5)
    after = cal.expected_calibration_error(y[2000:], scaler.predict(overconfident[2000:]))
    assert after < before / 2


def test_best_threshold_and_report():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.55, 0.7, 0.9])
    t = cal.best_f1_threshold(y, s)
    assert cal.f1(y, s >= t) == pytest.approx(cal.f1(y, s >= 0.55))
    rep = cal.agreement_report(y, s >= 0.5, s, n_boot=200)
    assert rep["n"] == 6 and 0 <= rep["roc_auc"] <= 1
    assert rep["kappa_ci"][0] <= rep["kappa"] <= rep["kappa_ci"][1] + 1e-9

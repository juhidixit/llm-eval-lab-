"""Judge calibration: how far can an LLM judge be trusted, measured against human labels?

Two different questions, both answered here:

1. **Agreement** - does the judge's verdict match the humans? (accuracy, precision/recall/F1 on
   the "problem" class, Cohen's kappa, ROC-AUC of the continuous score).
2. **Probability calibration** - when the judge says 80%, is it right 80% of the time?
   (Expected Calibration Error, reliability bins, Brier score; fixed with Platt scaling fitted on
   a calibration split and evaluated on a disjoint split.)

All functions take ``y_true`` (1 = humans flagged a problem, e.g. a hallucination) and either
binary predictions or continuous ``scores`` in [0, 1] where higher means "more likely problem".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evallab.stats import bootstrap_metric_ci


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": int((y_true & y_pred).sum()),
        "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()),
        "tn": int((~y_true & ~y_pred).sum()),
    }


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    c = confusion(y_true, y_pred)
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return precision_recall_f1(y_true, y_pred)[2]


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    c = confusion(y_true, y_pred)
    tpr = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    tnr = c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else 0.0
    return (tpr + tnr) / 2


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    po = float((y_true == y_pred).mean())
    pe = sum(float((y_true == k).sum() * (y_pred == k).sum()) for k in (0, 1)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U formulation with tie handling (average ranks)."""
    y_true = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_bins(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append(
                {"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())}
            )
    return out


def brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs, float) - np.asarray(y_true, float)) ** 2))


@dataclass
class PlattScaler:
    """Logistic regression on the logit of the judge score: p = sigmoid(a * logit(s) + b)."""

    a: float = 1.0
    b: float = 0.0

    @staticmethod
    def _logit(s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s))

    def fit(self, scores: np.ndarray, y_true: np.ndarray, iters: int = 500, lr: float = 0.1) -> PlattScaler:
        x = self._logit(scores)
        y = np.asarray(y_true, float)
        a, b = 1.0, 0.0
        for _ in range(iters):  # Newton-free gradient descent is plenty for 2 parameters
            p = 1 / (1 + np.exp(-(a * x + b)))
            g = p - y
            a -= lr * float((g * x).mean())
            b -= lr * float(g.mean())
        self.a, self.b = a, b
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * self._logit(scores) + self.b)))


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    s = np.asarray(scores, float)
    candidates = np.unique(s)
    best_t, best_f = 0.5, -1.0
    for t in candidates:
        f = f1(y_true, s >= t)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def agreement_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
    n_boot: int = 2000,
) -> dict[str, float | list[float]]:
    """Agreement of binary predictions (and optionally scores) with human labels, with 95% CIs."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f = precision_recall_f1(y_true, y_pred)
    k = bootstrap_metric_ci(y_true, y_pred, cohen_kappa, n_boot=n_boot)
    fci = bootstrap_metric_ci(y_true, y_pred, f1, n_boot=n_boot)
    rep: dict[str, float | list[float]] = {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "pred_positive_rate": float(y_pred.mean()),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f,
        "f1_ci": [fci[1], fci[2]],
        "kappa": k[0],
        "kappa_ci": [k[1], k[2]],
    }
    if scores is not None and len(np.unique(scores)) > 1:
        a = bootstrap_metric_ci(y_true, np.asarray(scores, float), roc_auc, n_boot=n_boot)
        rep["roc_auc"] = a[0]
        rep["roc_auc_ci"] = [a[1], a[2]]
    return rep

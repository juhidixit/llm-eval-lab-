"""Analyse judge scores from judge_calibration.py against the human labels.

For each judge:
* agreement at the default threshold (0.5),
* agreement after tuning the threshold (cross-fitted: tuned on one half, scored on the other, so
  the number is not optimistic),
* probability calibration: ECE / Brier of the raw score vs after Platt scaling (also cross-fitted),
* ops: tokens, latency per judgement.

Writes results/judge_calibration/summary.json and summary.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evallab import calibration as cal
from evallab.datasets.ragtruth import download, load_ragtruth
from evallab.metrics.deterministic import lexical_support

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results/judge_calibration"
STACK = ["lexical-overlap", "qwen2.5-3b_direct", "gemma3-4b_direct"]


def stacked(data: dict[str, list[dict]], members: list[str], folds: int = 5, seed: int = 0) -> list[dict]:
    """Cross-fitted logistic regression over member scores: every example is scored by a model
    that never saw it. Shows whether the LLM judges add signal on top of the lexical heuristic."""
    base = data[members[0]]
    idx = [{r["id"]: r for r in data[m]} for m in members]
    x = np.array([[float(d[r["id"]]["score"]) for d in idx] for r in base])
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    x = np.hstack([x, np.ones((len(x), 1))])
    y = np.array([float(r["label"]) for r in base])
    fold = np.random.default_rng(seed).permutation(len(y)) % folds
    out = np.zeros(len(y))
    for k in range(folds):
        tr = fold != k
        w = np.zeros(x.shape[1])
        for _ in range(3000):
            p = 1 / (1 + np.exp(-x[tr] @ w))
            w -= 0.1 * (x[tr].T @ (p - y[tr]) / tr.sum() + 1e-3 * w)
        out[~tr] = 1 / (1 + np.exp(-x[~tr] @ w))
    return [
        {
            **r,
            "score": float(out[i]),
            "prompt_tokens": sum(d[r["id"]]["prompt_tokens"] for d in idx),
            "completion_tokens": sum(d[r["id"]]["completion_tokens"] for d in idx),
            "latency_s": sum(d[r["id"]]["latency_s"] for d in idx),
            "calls": len(members) - 1,
        }
        for i, r in enumerate(base)
    ]


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DIR / f"scores_{name}.jsonl").open()]


def crossfit(y: np.ndarray, s: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """2-fold cross-fitted tuned predictions and Platt-calibrated probabilities."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % 2
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y))
    thresholds = []
    for k in (0, 1):
        tr, te = fold != k, fold == k
        t = cal.best_f1_threshold(y[tr], s[tr])
        thresholds.append(t)
        pred[te] = s[te] >= t
        prob[te] = cal.PlattScaler().fit(s[tr], y[tr], iters=3000, lr=0.5).predict(s[te])
    return pred, prob, thresholds


def analyse(name: str, rows: list[dict]) -> dict:
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([float(r["score"]) for r in rows])
    default = cal.agreement_report(y, s >= 0.5, s)
    tuned_pred, platt, thresholds = crossfit(y, s)
    tuned = cal.agreement_report(y, tuned_pred)
    gens = sorted({r["generator"] for r in rows})
    by_gen = {}
    for g in gens:
        m = np.array([r["generator"] == g for r in rows])
        by_gen[g] = {
            "n": int(m.sum()),
            "prevalence": float(y[m].mean()),
            "accuracy@0.5": float(((s[m] >= 0.5) == y[m]).mean()),
        }
    lat = [r["latency_s"] for r in rows]
    return {
        "judge": name,
        "n": len(rows),
        "default_threshold": default,
        "tuned_threshold": {**tuned, "thresholds": thresholds},
        "calibration": {
            "ece_raw": cal.expected_calibration_error(y, s),
            "ece_platt": cal.expected_calibration_error(y, platt),
            "brier_raw": cal.brier(y, s),
            "brier_platt": cal.brier(y, platt),
            "reliability_raw": cal.reliability_bins(y, s, 5),
            "reliability_platt": cal.reliability_bins(y, platt, 5),
        },
        "by_generator": by_gen,
        "ops": {
            "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in rows])),
            "completion_tokens_mean": float(np.mean([r["completion_tokens"] for r in rows])),
            "latency_mean_s": float(np.mean(lat)) if lat else 0.0,
            "latency_p95_s": float(np.quantile(lat, 0.95)) if lat else 0.0,
            "calls_mean": float(np.mean([r.get("calls", 0) for r in rows])),
        },
    }


def main() -> None:
    names = sorted(p.stem.removeprefix("scores_") for p in DIR.glob("scores_*.jsonl"))
    data = {n: rows for n in names if (rows := load(n))}
    n_full = len(data["lexical-overlap"])
    if all(m in data and len(data[m]) == n_full for m in STACK):
        data["stacked(lexical+qwen2.5-3b+gemma3-4b)"] = stacked(data, STACK)
    # Sentence-level judges: the default score is the max over sentences; also report the mean.
    for n in [n for n in data if n.endswith("_sentences")]:
        data[n + "(mean)"] = [{**r, "score": float(np.mean(r["p_unsupported"]))} for r in data[n]]
    # Partial runs (e.g. CoT on a subset) are compared against the other judges on the same subset.
    out = {}
    for n, rows in data.items():
        out[n] = analyse(n, rows)
        if len(rows) < len(data["lexical-overlap"]):
            ids = {r["id"] for r in rows}
            out[n]["same_subset_reference"] = {
                m: analyse(m, [r for r in data[m] if r["id"] in ids])["default_threshold"]
                for m in data
                if m != n and len(data[m]) == len(data["lexical-overlap"])
            }
    # The lexical heuristic (incl. its short-refusal rule) was designed while looking at this
    # sample, so it is re-checked on all 900 labelled responses of RAGTruth's held-out QA test split.
    test = load_ragtruth(download(ROOT / "data/raw/ragtruth"), split="test", task_types=("QA",))
    test_rows = [
        {
            "id": e.id,
            "label": e.labels["hallucinated"],
            "generator": e.meta["generator"],
            "score": 1 - lexical_support(e.input["answer"], e.input["contexts"]),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": 0.0,
        }
        for e in test
    ]
    out["lexical-overlap (held-out test split)"] = analyse("lexical-overlap (held-out test split)", test_rows)
    (DIR / "summary.json").write_text(json.dumps(out, indent=1))

    def fmt(x: float, ci: list[float] | None = None) -> str:
        return f"{x:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")

    lines = [
        "| judge | n | ROC-AUC | F1 @0.5 | κ @0.5 | F1 tuned* | κ tuned* | ECE raw → Platt* | tokens in/out | s / judgement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, r in sorted(out.items(), key=lambda kv: -kv[1]["default_threshold"].get("roc_auc", 0)):
        d, t, c, o = r["default_threshold"], r["tuned_threshold"], r["calibration"], r["ops"]
        lines.append(
            f"| {n} | {r['n']} | {fmt(d.get('roc_auc', float('nan')), d.get('roc_auc_ci'))} | {fmt(d['f1'])} | "
            f"{fmt(d['kappa'], d['kappa_ci'])} | {fmt(t['f1'])} | {fmt(t['kappa'])} | "
            f"{c['ece_raw']:.2f} → {c['ece_platt']:.2f} | {o['prompt_tokens_mean']:.0f} / {o['completion_tokens_mean']:.0f} | "
            f"{o['latency_mean_s']:.1f} |"
        )
    (DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

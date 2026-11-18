import copy

import pytest

from evallab.gate import evaluate_gate, render_markdown
from evallab.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_delta


def test_bootstrap_ci_contains_mean():
    pt, lo, hi = bootstrap_ci([0, 1] * 50)
    assert lo < pt == 0.5 < hi


def test_paired_bootstrap_detects_real_drop_and_ignores_noise():
    base = [1.0] * 100
    worse = [1.0] * 80 + [0.0] * 20
    assert paired_bootstrap_delta(base, worse)["p_worse"] < 0.01
    same = paired_bootstrap_delta(base, base)
    assert same["delta"] == 0.0


def test_mcnemar_exact():
    r = mcnemar_exact([True] * 10, [False] * 10)
    assert r["regressions"] == 10 and r["p_value"] == pytest.approx(2 / 2**10)
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def _run(values, version="v1", latency=1.0):
    return {
        "suite": "s",
        "dataset": {"path": "d.jsonl", "version": version, "n": len(values)},
        "system": {"hash": "h"},
        "ops": {"latency_p95_s": latency},
        "examples": [{"id": str(i), "metrics": {"faithfulness": v}} for i, v in enumerate(values)],
    }


RULES = {"faithfulness": {"min": 0.7, "max_drop": 0.03}, "latency_p95_s": {"max_increase_pct": 20}}


def test_gate_passes_identical_runs():
    b = _run([1.0] * 90 + [0.0] * 10)
    checks = evaluate_gate(b, copy.deepcopy(b), RULES)
    assert all(c.passed for c in checks)
    assert "PASS" in render_markdown(checks, b, b)


def test_gate_fails_significant_regression_and_latency():
    b = _run([1.0] * 90 + [0.0] * 10)
    c = _run([1.0] * 70 + [0.0] * 30, latency=1.5)
    checks = {(ch.metric, ch.rule.split()[0]): ch for ch in evaluate_gate(b, c, RULES)}
    assert not checks[("faithfulness", "drop")].passed
    assert checks[("faithfulness", ">=")].passed  # 0.70 still meets the floor
    assert not checks[("latency_p95_s", "+20%")].passed


def test_gate_tolerates_small_noisy_change():
    b = _run([1.0] * 90 + [0.0] * 10)
    c = _run([1.0] * 89 + [0.0] * 11)  # one flip: below max_drop
    assert all(ch.passed for ch in evaluate_gate(b, c, RULES))


def test_gate_refuses_different_dataset_versions():
    with pytest.raises(ValueError, match="dataset version"):
        evaluate_gate(_run([1.0]), _run([1.0], version="v2"), RULES)

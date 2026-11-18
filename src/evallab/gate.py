"""Regression gate: compare a candidate run against a baseline run on the same golden set.

A gate rule per metric can combine:

* ``min``        - absolute floor for the candidate mean ("faithfulness must be >= 0.85").
* ``max_drop``   - largest allowed drop vs baseline in absolute points, *and* the drop must be
  statistically real (paired bootstrap one-sided p < ``alpha``) before it fails the build.
  This stops CI from flaking on noise while still catching true regressions.
* ``max``        - absolute ceiling (for "lower is better" ops metrics like latency).
* ``max_increase_pct`` - relative ceiling vs baseline for ops metrics (cost, latency).

The gate refuses to compare runs made on different dataset versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evallab.stats import mcnemar_exact, paired_bootstrap_delta

OPS_LOWER_IS_BETTER = {
    "latency_p50_s",
    "latency_p95_s",
    "prompt_tokens_mean",
    "completion_tokens_mean",
    "projected_cost_per_1k_usd",
}


@dataclass
class Check:
    metric: str
    rule: str
    baseline: float | None
    candidate: float
    passed: bool
    detail: str


def _per_example(run: dict[str, Any], metric: str) -> dict[str, float]:
    return {r["id"]: r["metrics"][metric] for r in run["examples"] if metric in r["metrics"]}


def evaluate_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    rules: dict[str, dict[str, float]],
    alpha: float = 0.05,
) -> list[Check]:
    if baseline["dataset"]["version"] != candidate["dataset"]["version"]:
        raise ValueError(
            f"dataset version mismatch: baseline {baseline['dataset']['version']} vs "
            f"candidate {candidate['dataset']['version']}; re-run the baseline on the new golden set"
        )
    checks: list[Check] = []
    for metric, rule in rules.items():
        if metric in OPS_LOWER_IS_BETTER:
            b, c = baseline["ops"].get(metric), candidate["ops"].get(metric)
            if c is None:
                continue
            if "max" in rule:
                checks.append(Check(metric, f"<= {rule['max']}", b, c, c <= rule["max"], ""))
            if "max_increase_pct" in rule and b:
                inc = (c - b) / b * 100
                checks.append(
                    Check(
                        metric,
                        f"+{rule['max_increase_pct']}% max",
                        b,
                        c,
                        inc <= rule["max_increase_pct"],
                        f"{inc:+.1f}%",
                    )
                )
            continue
        cb, cc = _per_example(baseline, metric), _per_example(candidate, metric)
        if not cc:
            continue
        cmean = sum(cc.values()) / len(cc)
        bmean = sum(cb.values()) / len(cb) if cb else None
        if "min" in rule:
            checks.append(Check(metric, f">= {rule['min']}", bmean, cmean, cmean >= rule["min"], ""))
        if "max_drop" in rule and cb:
            ids = sorted(set(cb) & set(cc))
            bvals, cvals = [cb[i] for i in ids], [cc[i] for i in ids]
            pb = paired_bootstrap_delta(bvals, cvals)
            detail = f"Δ {pb['delta']:+.3f} [{pb['lo']:+.3f}, {pb['hi']:+.3f}], p_worse={pb['p_worse']:.3f}"
            if all(v in (0.0, 1.0) for v in bvals + cvals):
                mc = mcnemar_exact([v == 1.0 for v in bvals], [v == 1.0 for v in cvals])
                detail += f", {mc['regressions']} regressed / {mc['fixes']} fixed"
            failed = (-pb["delta"] > rule["max_drop"]) and pb["p_worse"] < alpha
            checks.append(Check(metric, f"drop <= {rule['max_drop']}", bmean, cmean, not failed, detail))
    return checks


def render_markdown(checks: list[Check], baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    ok = all(c.passed for c in checks)
    lines = [
        f"### Eval gate: {'✅ PASS' if ok else '❌ FAIL'}",
        "",
        f"Suite `{candidate['suite']}` · golden set `{candidate['dataset']['path']}` "
        f"(v{candidate['dataset']['version']}, n={candidate['dataset']['n']}) · "
        f"baseline system `{baseline['system']['hash']}` → candidate `{candidate['system']['hash']}`",
        "",
        "| metric | rule | baseline | candidate | result | detail |",
        "|---|---|---:|---:|:---:|---|",
    ]
    for c in checks:
        b = "–" if c.baseline is None else f"{c.baseline:.3f}"
        lines.append(
            f"| {c.metric} | {c.rule} | {b} | {c.candidate:.3f} | {'✅' if c.passed else '❌'} | {c.detail} |"
        )
    return "\n".join(lines) + "\n"

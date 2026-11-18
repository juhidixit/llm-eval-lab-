"""Static, dependency-free HTML dashboard for eval results (open the file; no server needed).

Reads ``results/judge_calibration/summary.json`` and ``results/runs/*.json`` and renders:
judge leaderboard (ROC-AUC with 95% CI), reliability diagram before/after Platt scaling,
the regression-gate table per candidate, and cost/latency per system variant.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ee;--text-primary:#0b0b0b;--text-secondary:#52514e;
--grid:#e4e3de;--series-1:#2a78d6;--series-2:#eb6834;--good:#008300;--critical:#d03b3b;
font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--surface-1);color:var(--text-primary);
max-width:1100px;margin:0 auto;padding:24px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
--surface-2:#242423;--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#3a3a37;--series-1:#3987e5;--series-2:#d95926;
--good:#0ca30c;--critical:#e66767}}
body{margin:0;background:var(--surface-1)}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 6px}p.sub{color:var(--text-secondary);margin:0 0 12px}
.card{background:var(--surface-2);border-radius:10px;padding:16px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600}
svg text{fill:var(--text-secondary);font-size:12px}svg .lbl{fill:var(--text-primary)}
.pass{color:var(--good);font-weight:600}.fail{color:var(--critical);font-weight:600}
details{margin-top:8px}summary{cursor:pointer;color:var(--text-secondary)}
.legend{display:flex;gap:16px;color:var(--text-secondary);font-size:13px;margin:4px 0}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
"""


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _auc_chart(summary: dict[str, Any]) -> str:
    rows = sorted(
        ((k, v["default_threshold"]) for k, v in summary.items() if "roc_auc" in v["default_threshold"]),
        key=lambda kv: -kv[1]["roc_auc"],
    )
    if not rows:
        return ""
    w, left, bar, gap = 760, 300, 18, 10
    h = len(rows) * (bar + gap) + 40
    lo_ax = 0.3
    x = lambda v: left + (max(v, lo_ax) - lo_ax) / (1 - lo_ax) * (w - left - 40)  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Judge ROC-AUC vs human labels">'
    ]
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        style = (
            'stroke="var(--text-secondary)" stroke-dasharray="4 4"' if t == 0.5 else 'stroke="var(--grid)"'
        )
        parts.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{h - 24}" {style}/>'
            f'<text x="{x(t):.1f}" y="{h - 8}" text-anchor="middle">{t:.1f}</text>'
        )
    for i, (name, d) in enumerate(rows):
        y = i * (bar + gap) + 6
        lo, hi = d["roc_auc_ci"]
        parts.append(
            f"<g><title>{_esc(name)}: ROC-AUC {d['roc_auc']:.3f} (95% CI {lo:.3f}–{hi:.3f}), n={d['n']}</title>"
            f'<text class="lbl" x="{left - 8}" y="{y + bar * 0.7}" text-anchor="end">{_esc(name)}</text>'
            f'<rect x="{x(lo_ax):.1f}" y="{y}" width="{max(0, x(d["roc_auc"]) - x(lo_ax)):.1f}" height="{bar}" rx="4" fill="var(--series-1)"/>'
            f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y + bar / 2}" y2="{y + bar / 2}" stroke="var(--text-primary)" stroke-width="2"/>'
            f'<text x="{x(max(hi, d["roc_auc"])) + 6:.1f}" y="{y + bar * 0.7}">{d["roc_auc"]:.2f}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _reliability(cal: dict[str, Any]) -> str:
    w = h = 300
    pad = 36
    sx = lambda v: pad + v * (w - pad - 10)  # noqa: E731
    sy = lambda v: h - pad - v * (h - pad - 10)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" width="320" role="img" aria-label="Reliability diagram">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line x1="{sx(0)}" x2="{sx(1)}" y1="{sy(t)}" y2="{sy(t)}" stroke="var(--grid)"/>'
            f'<text x="{pad - 6}" y="{sy(t) + 4}" text-anchor="end">{t:g}</text>'
            f'<text x="{sx(t)}" y="{h - pad + 16}" text-anchor="middle">{t:g}</text>'
        )
    parts.append(
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="var(--text-secondary)" stroke-dasharray="4 4"/>'
    )
    for key, var in (("reliability_raw", "--series-2"), ("reliability_platt", "--series-1")):
        pts = cal[key]
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(p['mean_pred']):.1f},{sy(p['frac_pos']):.1f}"
            for i, p in enumerate(pts)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="var({var})" stroke-width="2"/>')
        for p in pts:
            parts.append(
                f'<circle cx="{sx(p["mean_pred"]):.1f}" cy="{sy(p["frac_pos"]):.1f}" r="5" fill="var({var})" '
                f'stroke="var(--surface-2)" stroke-width="2"><title>{key.split("_")[1]}: predicted {p["mean_pred"]:.2f}, '
                f"observed {p['frac_pos']:.2f} (n={p['n']})</title></circle>"
            )
    parts.append(
        f'<text x="{w / 2 + 10}" y="{h - 4}" text-anchor="middle">judge P(hallucination)</text></svg>'
    )
    return "".join(parts)


def _hbar(items: list[tuple[str, float]], unit: str, label: str) -> str:
    if not items:
        return ""
    w, left, bar, gap = 560, 170, 16, 8
    h = len(items) * (bar + gap) + 8
    mx = max(v for _, v in items) or 1
    parts = [
        f'<p class="sub">{_esc(label)}</p><svg viewBox="0 0 {w} {h}" width="{w}" role="img" aria-label="{_esc(label)}">'
    ]
    for i, (name, v) in enumerate(items):
        y = i * (bar + gap) + 4
        bw = v / mx * (w - left - 110)
        parts.append(
            f"<g><title>{_esc(name)}: {v:,.2f} {unit}</title>"
            f'<text class="lbl" x="{left - 8}" y="{y + 12}" text-anchor="end">{_esc(name)}</text>'
            f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bar}" rx="4" fill="var(--series-1)"/>'
            f'<text x="{left + bw + 6:.1f}" y="{y + 12}">{v:,.1f} {unit}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def build_dashboard(results: Path, out: Path) -> None:
    body = [
        '<h1>llm-eval-lab results</h1><p class="sub">Every number on this page was measured; see '
        "results/RESULTS.md for methodology. Hover any mark for exact values.</p>"
    ]
    summ_path = results / "judge_calibration/summary.json"
    if summ_path.exists():
        summary = json.loads(summ_path.read_text())
        body.append(
            '<h2>1 · Which judge can you trust?</h2><p class="sub">ROC-AUC of each judge’s P(hallucination) '
            "against RAGTruth human labels (bar) with 95% bootstrap CI (whisker). Dashed line = chance (0.5).</p>"
        )
        body.append(f'<div class="card">{_auc_chart(summary)}</div>')
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{v['n']}</td><td>{v['default_threshold'].get('roc_auc', float('nan')):.3f}</td>"
            f"<td>{v['default_threshold']['f1']:.3f}</td><td>{v['default_threshold']['kappa']:.3f}</td>"
            f"<td>{v['tuned_threshold']['f1']:.3f}</td><td>{v['tuned_threshold']['kappa']:.3f}</td>"
            f"<td>{v['calibration']['ece_raw']:.3f}</td><td>{v['calibration']['ece_platt']:.3f}</td>"
            f"<td>{v['ops']['latency_mean_s']:.1f}</td></tr>"
            for k, v in sorted(summary.items(), key=lambda kv: -kv[1]["default_threshold"].get("roc_auc", 0))
        )
        body.append(
            "<details><summary>Table view</summary><table><tr><th>judge</th><th>n</th><th>ROC-AUC</th><th>F1@0.5</th>"
            "<th>κ@0.5</th><th>F1 tuned</th><th>κ tuned</th><th>ECE raw</th><th>ECE Platt</th><th>s/judgement</th></tr>"
            f"{rows}</table></details>"
        )
        best = max(
            (k for k in summary if not k.startswith("lexical")),
            key=lambda k: summary[k]["default_threshold"].get("roc_auc", 0),
            default=None,
        )
        if best:
            body.append(
                f'<h2>2 · Are its probabilities honest?</h2><p class="sub">Reliability of <b>{_esc(best)}</b>: '
                "raw score vs after Platt scaling (cross-fitted). Diagonal = perfectly calibrated.</p>"
                '<div class="legend"><span><span class="sw" style="background:var(--series-2)"></span>raw score</span>'
                '<span><span class="sw" style="background:var(--series-1)"></span>Platt-scaled</span></div>'
                f'<div class="card">{_reliability(summary[best]["calibration"])}</div>'
            )
    runs_dir = results / "runs"
    if runs_dir.exists():
        runs = {p.stem: json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))}
        if runs:
            body.append(
                '<h2>3 · Regression gate on the RAG app</h2><p class="sub">Each candidate change is compared '
                "with the baseline on the same golden set; the gate fails only on statistically real drops.</p>"
            )
            metrics = sorted({m for r in runs.values() for m in r["summary"] if m != "p_hallucination"})
            head = "".join(f"<th>{_esc(m)}</th>" for m in metrics)
            trs = []
            for name, r in runs.items():
                gate_md = results / f"gates/{name}.md"
                status = ""
                if gate_md.exists():
                    ok = "PASS" in gate_md.read_text().splitlines()[0]
                    status = (
                        f'<span class="{"pass" if ok else "fail"}">{"✅ pass" if ok else "❌ blocked"}</span>'
                    )
                elif name == "baseline":
                    status = "baseline"
                cells = "".join(
                    f"<td>{r['summary'][m]['mean']:.3f}</td>" if m in r["summary"] else "<td>–</td>"
                    for m in metrics
                )
                trs.append(
                    f"<tr><td>{_esc(name)}</td>{cells}<td>{r['ops']['prompt_tokens_mean']:.0f}</td>"
                    f"<td>{r['ops']['latency_p95_s']:.1f}</td><td>{status}</td></tr>"
                )
            body.append(
                f'<div class="card"><table><tr><th>system</th>{head}<th>prompt tok</th><th>p95 s</th><th>gate</th></tr>'
                f"{''.join(trs)}</table></div>"
            )
            body.append(
                '<h2>4 · Cost and latency per request</h2><div class="card">'
                + _hbar(
                    [
                        (n, r["ops"]["prompt_tokens_mean"] + r["ops"]["completion_tokens_mean"])
                        for n, r in runs.items()
                    ],
                    "tokens",
                    "Tokens per request",
                )
                + _hbar([(n, r["ops"]["latency_p95_s"]) for n, r in runs.items()], "s p95", "p95 latency")
                + "</div>"
            )
    page = (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width,initial-scale=1"><title>llm-eval-lab dashboard</title><style>{CSS}</style>'
        f'</head><body><main class="viz-root">{"".join(body)}</main></body></html>'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)

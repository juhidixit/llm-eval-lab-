"""``evallab`` command line.

evallab run suites/rag_qa.yaml --cache replay --out runs/candidate.json
evallab gate --baseline results/runs/baseline.json --candidate runs/candidate.json --suite suites/rag_qa.yaml
evallab dataset info data/golden/rag_qa_golden.jsonl
evallab report --out results/dashboard.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evallab.datasets import Dataset


def _cmd_run(a: argparse.Namespace) -> int:
    from evallab.runner import Suite, run_suite, save_run

    overrides = dict(kv.split("=", 1) for kv in a.set or [])
    suite = Suite.load(a.suite, overrides)
    run = run_suite(suite, cache_mode=a.cache, limit=a.limit, progress=True)
    save_run(run, a.out)
    s = run["summary"]
    print(f"run saved to {a.out}  (dataset v{run['dataset']['version']}, system {run['system']['hash']})")
    for k, v in s.items():
        print(f"  {k:<18} {v['mean']:.3f}  [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]  n={v['n']}")
    for k, v in run["ops"].items():
        print(f"  {k:<26} {v if isinstance(v, str) else round(v, 4)}")
    return 0


def _cmd_gate(a: argparse.Namespace) -> int:
    from evallab.gate import evaluate_gate, render_markdown
    from evallab.runner import Suite

    baseline = json.loads(Path(a.baseline).read_text())
    candidate = json.loads(Path(a.candidate).read_text())
    rules = Suite.load(a.suite).gates
    checks = evaluate_gate(baseline, candidate, rules)
    md = render_markdown(checks, baseline, candidate)
    print(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(md + "\n")
    if a.markdown_out:
        Path(a.markdown_out).write_text(md)
    passed = all(c.passed for c in checks)
    if a.expect == "fail":  # gate self-test: a known-bad change MUST be blocked
        return 0 if not passed else 1
    return 0 if passed else 1


def _cmd_dataset(a: argparse.Namespace) -> int:
    ds = Dataset.load(a.path)
    tags: dict[str, int] = {}
    for e in ds:
        for t in e.tags:
            tags[t] = tags.get(t, 0) + 1
    print(json.dumps({"name": ds.name, "n": len(ds), "version": ds.version, "tags": tags}, indent=2))
    return 0


def _cmd_report(a: argparse.Namespace) -> int:
    from evallab.report import build_dashboard

    build_dashboard(Path(a.results), Path(a.out))
    print(f"wrote {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="evallab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a suite and save a run record")
    r.add_argument("suite")
    r.add_argument("--cache", default="replay", choices=["replay", "record", "auto", "off"])
    r.add_argument("--limit", type=int)
    r.add_argument("--set", action="append", help="override a suite key, e.g. system.prompt=prompts/x.txt")
    r.add_argument("--out", default="runs/run.json")
    r.set_defaults(fn=_cmd_run)

    g = sub.add_parser("gate", help="compare candidate vs baseline; non-zero exit on regression")
    g.add_argument("--baseline", required=True)
    g.add_argument("--candidate", required=True)
    g.add_argument("--suite", required=True)
    g.add_argument("--markdown-out")
    g.add_argument("--expect", choices=["pass", "fail"], default="pass")
    g.set_defaults(fn=_cmd_gate)

    d = sub.add_parser("dataset", help="dataset utilities")
    d.add_argument("action", choices=["info"])
    d.add_argument("path")
    d.set_defaults(fn=_cmd_dataset)

    rep = sub.add_parser("report", help="build the static HTML dashboard")
    rep.add_argument("--results", default="results")
    rep.add_argument("--out", default="results/dashboard.html")
    rep.set_defaults(fn=_cmd_report)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())

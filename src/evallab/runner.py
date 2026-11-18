"""Run an eval suite: system under test -> metrics -> a self-describing run record.

A *suite* (YAML) pins the golden dataset, the system configuration, the judge and the metric
list. A *run* (JSON) stores per-example outputs and scores plus a summary with bootstrap CIs,
token usage, latency percentiles and projected cost. Runs are what the regression gate compares.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evallab.apps.rag_qa import RagQAApp
from evallab.datasets import Dataset
from evallab.judges import FaithfulnessJudge
from evallab.metrics import deterministic as det
from evallab.metrics.rag import RagMetrics
from evallab.providers import CachedProvider, Provider, make_provider
from evallab.providers.pricing import cost_usd
from evallab.stats import bootstrap_ci


@dataclass
class Suite:
    name: str
    dataset: str
    system: dict[str, Any]
    provider: dict[str, Any]
    judge: dict[str, Any] | None
    metrics: list[str]
    gates: dict[str, dict[str, float]] = field(default_factory=dict)
    cassette_dir: str = "cassettes"
    price_as: str | None = None
    # minimum lexical support for the deterministic `grounded` metric (chosen in experiment 1)
    grounded_min_support: float = 0.70
    root: Path = Path(".")

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> Suite:
        path = Path(path)
        cfg = yaml.safe_load(path.read_text())
        for k, v in (overrides or {}).items():  # "system.prompt=prompts/x.txt" style overrides
            node = cfg
            keys = k.split(".")
            for kk in keys[:-1]:
                node = node.setdefault(kk, {})
            node[keys[-1]] = yaml.safe_load(v) if isinstance(v, str) else v
        root = path.resolve().parent
        while root != root.parent and not (root / "pyproject.toml").exists():
            root = root.parent
        return cls(root=root, **{k: cfg[k] for k in cfg if k in cls.__dataclass_fields__})

    def system_hash(self) -> str:
        sys_cfg = dict(self.system)
        if "prompt" in sys_cfg:
            sys_cfg["prompt_text"] = (self.root / sys_cfg["prompt"]).read_text()
        return hashlib.sha256(json.dumps(sys_cfg, sort_keys=True).encode()).hexdigest()[:12]


def _live_provider(cfg: dict[str, Any], root: Path) -> Provider:
    kwargs = {k: v for k, v in cfg.items() if k != "kind"}
    if cfg["kind"] == "llamacpp_local":
        kwargs["models"] = {k: root / v for k, v in kwargs.get("models", {}).items()}
    return make_provider(cfg["kind"], **kwargs)


def _p(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else float("nan")


def run_suite(
    suite: Suite, cache_mode: str = "replay", limit: int | None = None, progress: bool = False
) -> dict[str, Any]:
    root = suite.root
    ds = Dataset.load(root / suite.dataset)
    examples = ds.examples[:limit] if limit else ds.examples
    live = None if cache_mode == "replay" else _live_provider(suite.provider, root)
    cdir = root / suite.cassette_dir
    gen = CachedProvider(live, cdir / f"{suite.name}__generator.jsonl", mode=cache_mode)  # type: ignore[arg-type]
    jprov = CachedProvider(live, cdir / f"{suite.name}__judge.jsonl", mode=cache_mode)  # type: ignore[arg-type]
    app = RagQAApp.from_config(gen, suite.system, root)
    judge = None
    if suite.judge:
        judge = FaithfulnessJudge(
            jprov,
            suite.judge["model"],
            strategy=suite.judge.get("strategy", "direct"),
            threshold=float(suite.judge.get("threshold", 0.5)),
        )
    rag = RagMetrics(jprov, suite.judge["model"]) if suite.judge else None

    rows = []
    for i, ex in enumerate(examples):
        q, ctxs = ex.input["question"], ex.input["contexts"]
        out = app(q, ctxs)
        answer = out.text.strip()
        refused = det.is_refusal(answer)
        unanswerable = "unanswerable" in ex.tags
        m: dict[str, float] = {}
        jt = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
        if "refusal_correct" in suite.metrics:
            m["refusal_correct"] = float(refused == unanswerable)
        if "refused_unanswerable" in suite.metrics and unanswerable:
            m["refused_unanswerable"] = float(refused)
        if "answered" in suite.metrics and not unanswerable:
            m["answered"] = float(not refused)
        if "grounded" in suite.metrics:
            support = det.lexical_support(answer, app.contexts_used(ctxs))
            m["grounded"] = float(support >= suite.grounded_min_support)
            m["lexical_support"] = support
        if "token_f1" in suite.metrics and ex.reference and not unanswerable:
            m["token_f1"] = det.token_f1(answer, ex.reference)
        if judge and "faithfulness" in suite.metrics:
            r = judge(q, app.contexts_used(ctxs), answer)
            m["faithfulness"] = 1.0 - float(r.verdict)  # 1 = judged faithful
            m["p_hallucination"] = r.score
            jt = {
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "latency_s": r.latency_s,
            }
        if rag and "context_recall" in suite.metrics and ex.reference and not unanswerable:
            m["context_recall"] = rag.context_recall(ex.reference, app.contexts_used(ctxs))
        rows.append(
            {
                "id": ex.id,
                "tags": ex.tags,
                "answer": answer,
                "metrics": m,
                "prompt_tokens": out.prompt_tokens,
                "completion_tokens": out.completion_tokens,
                "latency_s": round(out.latency_s, 4),
                "judge": jt,
            }
        )
        if progress and (i + 1) % 10 == 0:
            print(f"  {suite.name}: {i + 1}/{len(examples)}", flush=True)

    summary: dict[str, Any] = {}
    names = sorted({k for r in rows for k in r["metrics"]})
    for name in names:
        vals = [r["metrics"][name] for r in rows if name in r["metrics"]]
        pt, lo, hi = bootstrap_ci(vals)
        summary[name] = {"mean": pt, "ci": [lo, hi], "n": len(vals)}
    lat = [r["latency_s"] for r in rows]
    ptok = [r["prompt_tokens"] for r in rows]
    ctok = [r["completion_tokens"] for r in rows]
    ops = {
        "latency_p50_s": _p(lat, 0.5),
        "latency_p95_s": _p(lat, 0.95),
        "prompt_tokens_mean": float(np.mean(ptok)),
        "completion_tokens_mean": float(np.mean(ctok)),
        "judge_latency_total_s": float(sum(r["judge"]["latency_s"] for r in rows)),
    }
    if suite.price_as:
        per = [cost_usd(suite.price_as, p, c) for p, c in zip(ptok, ctok, strict=True)]
        ops["projected_cost_per_1k_usd"] = float(np.mean(per) * 1000)
        ops["priced_as"] = suite.price_as  # type: ignore[assignment]
    return {
        "suite": suite.name,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"path": suite.dataset, "version": ds.version, "n": len(examples)},
        "system": {**suite.system, "hash": suite.system_hash()},
        "judge": suite.judge,
        "summary": summary,
        "ops": ops,
        "cache": {
            "generator_hits": gen.hits,
            "generator_misses": gen.misses,
            "judge_hits": jprov.hits,
            "judge_misses": jprov.misses,
        },
        "examples": rows,
    }


def save_run(run: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False))

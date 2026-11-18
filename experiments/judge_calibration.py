"""Experiment 1: how well do small open-weight LLM judges agree with human hallucination labels?

Scores every (judge model x strategy) on a fixed, stratified sample of RAGTruth QA responses and
writes one JSONL of per-example scores per judge. All LLM calls go through a record/replay
cassette, so ``--cache replay`` recomputes everything from the committed cassettes without any
model weights. Analysis lives in ``analyze_calibration.py``.

    python experiments/judge_calibration.py --judges qwen2.5-3b:direct gemma3-4b:direct
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evallab.datasets.ragtruth import download, load_ragtruth
from evallab.judges import FaithfulnessJudge
from evallab.metrics.deterministic import lexical_support
from evallab.providers import CachedProvider
from evallab.providers.llamacpp_local import LlamaCppProvider

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen2.5-3b": "models/qwen2.5-3b-q4km.gguf",
    "gemma3-4b": "models/gemma3-4b-q4km.gguf",
    "qwen2.5-7b": "models/qwen2.5-7b-q4km.gguf",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=["qwen2.5-3b:direct"])
    ap.add_argument("--raw", default="data/raw/ragtruth")
    ap.add_argument("--n-pos", type=int, default=120)
    ap.add_argument("--n-neg", type=int, default=180)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N examples")
    ap.add_argument("--cache", default="auto", choices=["auto", "record", "replay"])
    ap.add_argument("--out", default="results/judge_calibration")
    args = ap.parse_args()

    raw = download(ROOT / args.raw)
    ds = load_ragtruth(raw, split="train", task_types=("QA",), n_pos=args.n_pos, n_neg=args.n_neg, seed=0)
    ds.save(ROOT / "data/golden/ragtruth_qa_calibration.jsonl")
    print(f"dataset {ds.name} n={len(ds)} version={ds.version}")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    examples = ds.examples[: args.limit] if args.limit else ds.examples

    # Non-LLM baseline: lexical support (fraction of answer content words found in the context).
    with (out / "scores_lexical-overlap.jsonl").open("w") as f:
        for ex in ds:
            s = 1 - lexical_support(ex.input["answer"], ex.input["contexts"])
            f.write(
                json.dumps(
                    {
                        "id": ex.id,
                        "label": ex.labels["hallucinated"],
                        "score": s,
                        "generator": ex.meta["generator"],
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_s": 0.0,
                    }
                )
                + "\n"
            )

    for spec in args.judges:
        # "model:strategy" or "model:strategy:N" (N = score only the first N examples; used for
        # the more expensive judges, which were run on a subset)
        model, strategy, *n = spec.split(":")
        todo = examples[: int(n[0])] if n else examples
        live = None if args.cache == "replay" else LlamaCppProvider({model: ROOT / MODELS[model]})
        provider = CachedProvider(live, ROOT / f"cassettes/judge_calibration/{model}.jsonl", mode=args.cache)
        judge = FaithfulnessJudge(provider, model, strategy=strategy)
        path = out / f"scores_{model}_{strategy}.jsonl"
        t0 = time.time()
        with path.open("w") as f:
            for i, ex in enumerate(todo):
                r = judge(ex.input["question"], ex.input["contexts"], ex.input["answer"])
                f.write(
                    json.dumps(
                        {
                            "id": ex.id,
                            "label": ex.labels["hallucinated"],
                            "score": r.score,
                            "generator": ex.meta["generator"],
                            "prompt_tokens": r.prompt_tokens,
                            "completion_tokens": r.completion_tokens,
                            "latency_s": r.latency_s,
                            "calls": r.calls,
                            "p_unsupported": r.details.get("p_unsupported"),
                            "raw": r.raw[-400:],
                        }
                    )
                    + "\n"
                )
                f.flush()
                if (i + 1) % 25 == 0:
                    print(f"  {spec}: {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        print(
            f"{spec}: done in {time.time() - t0:.0f}s (cache hits {provider.hits}, misses {provider.misses})"
        )


if __name__ == "__main__":
    main()

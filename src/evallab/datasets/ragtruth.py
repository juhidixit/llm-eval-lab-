"""RAGTruth (Niu et al., ACL 2024): ~18k LLM responses with word-level human hallucination labels.

Source: https://github.com/ParticleMedia/RAGTruth (MIT). We use the response-level label
"humans marked at least one hallucinated span" as ground truth for faithfulness judges.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

from evallab.datasets.schema import Dataset, Example

BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
FILES = ("response.jsonl", "source_info.jsonl")


def download(raw_dir: str | Path) -> Path:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw / name
        if not target.exists():
            urllib.request.urlretrieve(f"{BASE}/{name}", target)  # noqa: S310 - fixed https URL
    return raw


def _split_passages(text: str) -> list[str]:
    parts = re.split(r"passage \d+:", text)
    return [p.strip() for p in parts if p.strip()]


def _context_for(src: dict) -> tuple[str, list[str], str]:
    """Return (question/task, list of context chunks, task_type)."""
    info = src["source_info"]
    task = src["task_type"]
    if task == "QA":
        return info["question"], _split_passages(info["passages"]), task
    if task == "Summary":
        return "Summarize the following news.", [info], task
    # Data2txt: structured business record
    return "Write an objective overview of the business.", [json.dumps(info, ensure_ascii=False)], task


def load_ragtruth(
    raw_dir: str | Path,
    split: str = "train",
    task_types: tuple[str, ...] = ("QA",),
    n_pos: int | None = None,
    n_neg: int | None = None,
    seed: int = 0,
    exclude_models: tuple[str, ...] = (),
) -> Dataset:
    """Labelled examples for judge calibration.

    ``n_pos`` / ``n_neg`` draw a stratified sample of hallucinated / faithful responses (None =
    all). At most one response per source is kept per class so a sample isn't dominated by a few
    contexts.
    """
    raw = Path(raw_dir)
    sources = {}
    with (raw / "source_info.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            sources[d["source_id"]] = d
    pos: list[Example] = []
    neg: list[Example] = []
    seen: dict[tuple[str, bool], int] = {}
    rows = [json.loads(line) for line in (raw / "response.jsonl").open()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    for r in rows:
        src = sources[r["source_id"]]
        if r["split"] != split or src["task_type"] not in task_types or r["model"] in exclude_models:
            continue
        halluc = bool(r["labels"])
        key = (r["source_id"], halluc)
        if seen.get(key):
            continue
        seen[key] = 1
        question, contexts, task = _context_for(src)
        ex = Example(
            id=f"ragtruth-{r['id']}",
            input={"question": question, "contexts": contexts, "answer": r["response"]},
            labels={
                "hallucinated": halluc,
                "label_types": sorted({lab["label_type"] for lab in r["labels"]}),
                "n_spans": len(r["labels"]),
            },
            tags=[task, r["model"]] + (["hallucinated"] if halluc else ["faithful"]),
            meta={
                "source_id": r["source_id"],
                "source": src["source"],
                "generator": r["model"],
                "split": r["split"],
            },
        )
        (pos if halluc else neg).append(ex)
    if n_pos is not None:
        pos = pos[:n_pos]
    if n_neg is not None:
        neg = neg[:n_neg]
    exs = pos + neg
    rng.shuffle(exs)
    return Dataset(f"ragtruth-{split}-{'+'.join(task_types)}", exs)

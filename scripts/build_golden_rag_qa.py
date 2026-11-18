"""Build the golden set for the example RAG-QA suite from RAGTruth's *test* split.

* 80 answerable questions (MS MARCO question + its 3 retrieved passages). The reference answer
  is the GPT-4 response that RAGTruth annotators marked hallucination-free.
* 20 unanswerable questions: the question is paired with *another* question's passages, so the
  correct behaviour is to refuse. This is the standard "negative context" robustness slice.

The calibration sample (experiments/judge_calibration.py) comes from the *train* split, so the
judge is never calibrated on the questions it later gates.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evallab.datasets import Dataset, Example  # noqa: E402
from evallab.datasets.ragtruth import _split_passages, download  # noqa: E402


def main(n_answerable: int = 80, n_unanswerable: int = 20, seed: int = 7) -> None:
    raw = download(ROOT / "data/raw/ragtruth")
    sources = {json.loads(row)["source_id"]: json.loads(row) for row in (raw / "source_info.jsonl").open()}
    refs: dict[str, str] = {}
    by_src: dict[str, list[dict]] = defaultdict(list)
    for line in (raw / "response.jsonl").open():
        r = json.loads(line)
        if r["split"] == "test" and sources[r["source_id"]]["task_type"] == "QA":
            by_src[r["source_id"]].append(r)
    for sid, rs in by_src.items():
        for pref in ("gpt-4-0613", "gpt-3.5-turbo-0613"):
            ok = [r for r in rs if r["model"] == pref and not r["labels"]]
            if ok and "unable to answer" not in ok[0]["response"].lower():
                refs[sid] = ok[0]["response"]
                break
    ids = sorted(refs)
    rng = random.Random(seed)
    rng.shuffle(ids)
    answerable, donors = ids[:n_answerable], ids[n_answerable : n_answerable + n_unanswerable]
    pool = sorted(set(sources) & set(by_src) - set(answerable) - set(donors))
    examples = []
    for sid in answerable:
        info = sources[sid]["source_info"]
        examples.append(
            Example(
                id=f"rqa-{sid}",
                input={"question": info["question"], "contexts": _split_passages(info["passages"])},
                reference=refs[sid],
                tags=["answerable"],
                meta={"source_id": sid, "source": "RAGTruth/MS MARCO test"},
            )
        )
    for sid in donors:
        q = sources[sid]["source_info"]["question"]
        other = rng.choice(pool)
        examples.append(
            Example(
                id=f"rqa-neg-{sid}",
                input={"question": q, "contexts": _split_passages(sources[other]["source_info"]["passages"])},
                reference="Unable to answer based on given passages.",
                tags=["unanswerable"],
                meta={
                    "source_id": sid,
                    "context_from": other,
                    "source": "RAGTruth/MS MARCO test (swapped context)",
                },
            )
        )
    ds = Dataset("rag_qa_golden", examples)
    ds.validate()
    ds.save(ROOT / "data/golden/rag_qa_golden.jsonl")
    print(f"wrote {len(ds)} examples, version {ds.version}")


if __name__ == "__main__":
    main()

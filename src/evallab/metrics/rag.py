"""RAG metrics: retrieval quality (context precision / recall) and answer quality.

Definitions follow the RAGAS paper (Es et al., 2023) so numbers are comparable with the tools
teams already use, but each metric here is a small, readable function over a yes/no judge:

* **context precision@k** - are the *relevant* chunks ranked first? Average precision over the
  ranked chunks, where relevance of each chunk is judged against the question (+ reference).
* **context recall** - can every sentence of the *reference* answer be attributed to the
  retrieved context? Needs a reference; tells you if retrieval missed something.
* **answer relevance** - does the answer actually address the question (refusals are scored
  separately so they don't get rewarded for being "safe").

Faithfulness lives in :mod:`evallab.judges.faithfulness`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evallab.judges.base import ask_yes_no
from evallab.providers.base import Provider

CHUNK_RELEVANCE = """QUESTION: {question}
{reference_block}
PASSAGE:
{chunk}

Does this PASSAGE contain information that helps answer the QUESTION? Reply with exactly one word: Yes or No."""

ATTRIBUTION = """CONTEXT:
{context}

STATEMENT: {sentence}

Can the STATEMENT be inferred from the CONTEXT? Reply with exactly one word: Yes or No."""

ANSWER_RELEVANCE = """QUESTION: {question}

ANSWER: {answer}

Does the ANSWER directly address the QUESTION (even if it might be factually wrong)? \
Reply with exactly one word: Yes or No."""


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def average_precision(relevance: list[bool]) -> float:
    """AP over a ranked list of binary relevance judgements (0 if nothing relevant)."""
    hits, total = 0, 0.0
    for k, rel in enumerate(relevance, 1):
        if rel:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


@dataclass
class RagMetrics:
    provider: Provider
    model: str
    threshold: float = 0.5

    def chunk_relevance(self, question: str, chunks: list[str], reference: str | None = None) -> list[float]:
        ref = f"REFERENCE ANSWER: {reference}\n" if reference else ""
        return [
            ask_yes_no(
                self.provider,
                self.model,
                None,
                CHUNK_RELEVANCE.format(question=question, reference_block=ref, chunk=c[:3000]),
            )[0]
            for c in chunks
        ]

    def context_precision(self, question: str, chunks: list[str], reference: str | None = None) -> float:
        probs = self.chunk_relevance(question, chunks, reference)
        return average_precision([p >= self.threshold for p in probs])

    def context_recall(self, reference: str, chunks: list[str]) -> float:
        sentences = split_sentences(reference)
        if not sentences:
            return 1.0
        ctx = "\n\n".join(chunks)[:6000]
        probs = [
            ask_yes_no(self.provider, self.model, None, ATTRIBUTION.format(context=ctx, sentence=s))[0]
            for s in sentences
        ]
        return sum(p >= self.threshold for p in probs) / len(sentences)

    def answer_relevance(self, question: str, answer: str) -> float:
        return ask_yes_no(
            self.provider, self.model, None, ANSWER_RELEVANCE.format(question=question, answer=answer)
        )[0]

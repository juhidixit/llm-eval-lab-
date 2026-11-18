"""Faithfulness (groundedness) judges for RAG answers.

Three strategies, all scored as P(answer contains unsupported / contradicted content):

* ``direct``  - one yes/no question, score from first-token log-probs. Cheapest (1 call,
  ~1 output token).
* ``cot``     - the judge reasons first, then ends with a verdict line. Costs many output tokens.
* ``claims``  - RAGAS-style: extract atomic claims, then verify each against the context.
  Score = fraction of unsupported claims; verdict = any unsupported claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evallab.judges.base import JudgeResult, ask_yes_no
from evallab.metrics.rag import split_sentences
from evallab.providers.base import Provider, Request, chat

SYSTEM = (
    "You are a meticulous fact-checker. You compare an ANSWER against a CONTEXT and decide whether "
    "the answer contains information that is not supported by the context."
)


def format_context(contexts: list[str], max_chars: int = 6000) -> str:
    out, used = [], 0
    for i, c in enumerate(contexts, 1):
        c = c[: max(0, max_chars - used)]
        used += len(c)
        out.append(f"[{i}] {c}")
        if used >= max_chars:
            break
    return "\n\n".join(out)


DIRECT_TEMPLATE = """CONTEXT:
{context}

QUESTION / TASK: {question}

ANSWER:
{answer}

Does the ANSWER contain any claim that is NOT supported by the CONTEXT, or that contradicts it? \
General knowledge that is not stated in the context counts as unsupported. A refusal such as \
"unable to answer" is supported.
Reply with exactly one word: Yes or No."""

# Same question with positive polarity ("is everything supported?"). Small models are strongly
# biased towards answering "Yes" to negated questions, so this framing behaves very differently.
SUPPORTED_TEMPLATE = """CONTEXT:
{context}

QUESTION / TASK: {question}

ANSWER:
{answer}

Is every claim in the ANSWER directly supported by the CONTEXT? Information that is not stated in \
the context counts as unsupported. A refusal such as "unable to answer" counts as supported.
Reply with exactly one word: Yes or No."""

# Sentence-level check: the context (and full answer, for reference) form a shared prefix, so with
# KV-cache reuse each extra sentence costs only a few dozen tokens.
SENTENCE_TEMPLATE = """CONTEXT:
{context}

QUESTION / TASK: {question}

FULL ANSWER (for reference):
{answer}

Now check one sentence of that answer.
SENTENCE: {sentence}

Is everything stated in this SENTENCE supported by the CONTEXT? Details that are not in the \
context count as unsupported. A refusal counts as supported.
Reply with exactly one word: Yes or No."""

COT_TEMPLATE = """CONTEXT:
{context}

QUESTION / TASK: {question}

ANSWER:
{answer}

Check the ANSWER sentence by sentence against the CONTEXT. Briefly note any claim that is not \
supported by the context or contradicts it (general knowledge not in the context counts as \
unsupported). Then finish with a final line that is exactly "VERDICT: HALLUCINATED" or \
"VERDICT: FAITHFUL"."""

CLAIMS_EXTRACT = """Break the following ANSWER into short, self-contained factual claims, one per line, \
numbered "1.", "2.", ... Write at most 8 claims. If the answer is a refusal or contains no factual \
claims, write "NONE".

ANSWER:
{answer}"""

CLAIMS_VERIFY = """CONTEXT:
{context}

For each numbered CLAIM below, decide if it is supported by the CONTEXT. Answer with one line per \
claim in the form "<number>. SUPPORTED" or "<number>. UNSUPPORTED".

CLAIMS:
{claims}"""


@dataclass
class FaithfulnessJudge:
    provider: Provider
    model: str
    strategy: str = "direct"  # direct | supported | sentences | cot | claims
    threshold: float = 0.5
    max_context_chars: int = 6000

    @property
    def name(self) -> str:
        return f"faithfulness/{self.strategy}@{self.model}"

    def __call__(self, question: str, contexts: list[str], answer: str) -> JudgeResult:
        ctx = format_context(contexts, self.max_context_chars)
        if self.strategy == "direct":
            return self._direct(ctx, question, answer)
        if self.strategy == "supported":
            p, c = ask_yes_no(
                self.provider,
                self.model,
                SYSTEM,
                SUPPORTED_TEMPLATE.format(context=ctx, question=question, answer=answer),
            )
            return JudgeResult(
                1 - p, 1 - p >= self.threshold, c.text, c.prompt_tokens, c.completion_tokens, c.latency_s
            )
        if self.strategy == "sentences":
            return self._sentences(ctx, question, answer)
        if self.strategy == "cot":
            return self._cot(ctx, question, answer)
        if self.strategy == "claims":
            return self._claims(ctx, answer)
        raise ValueError(f"unknown strategy {self.strategy!r}")

    def _direct(self, ctx: str, question: str, answer: str) -> JudgeResult:
        p, c = ask_yes_no(
            self.provider,
            self.model,
            SYSTEM,
            DIRECT_TEMPLATE.format(context=ctx, question=question, answer=answer),
        )
        return JudgeResult(p, p >= self.threshold, c.text, c.prompt_tokens, c.completion_tokens, c.latency_s)

    def _sentences(self, ctx: str, question: str, answer: str) -> JudgeResult:
        sents = split_sentences(answer)[:12] or [answer]
        probs, pt, ct, lat = [], 0, 0, 0.0
        for sent in sents:
            p, c = ask_yes_no(
                self.provider,
                self.model,
                SYSTEM,
                SENTENCE_TEMPLATE.format(context=ctx, question=question, answer=answer, sentence=sent),
            )
            probs.append(1 - p)
            pt, ct, lat = pt + c.prompt_tokens, ct + c.completion_tokens, lat + c.latency_s
        score = max(probs)
        return JudgeResult(
            score,
            score >= self.threshold,
            f"per-sentence P(unsupported)={[round(x, 3) for x in probs]}",
            pt,
            ct,
            lat,
            calls=len(sents),
            details={"sentences": sents, "p_unsupported": probs},
        )

    def _cot(self, ctx: str, question: str, answer: str) -> JudgeResult:
        c = self.provider.complete(
            Request(
                model=self.model,
                messages=chat(SYSTEM, COT_TEMPLATE.format(context=ctx, question=question, answer=answer)),
                max_tokens=384,
            )
        )
        m = re.findall(r"VERDICT:\s*\**\s*(HALLUCINATED|FAITHFUL)", c.text, re.I)
        if m:
            score = 1.0 if m[-1].upper() == "HALLUCINATED" else 0.0
        else:  # truncated or malformed: fall back to keyword presence
            score = 1.0 if re.search(r"not supported|unsupported|contradict", c.text, re.I) else 0.0
        return JudgeResult(
            score,
            score >= self.threshold,
            c.text,
            c.prompt_tokens,
            c.completion_tokens,
            c.latency_s,
            details={"parsed_verdict": bool(m)},
        )

    def _claims(self, ctx: str, answer: str) -> JudgeResult:
        c1 = self.provider.complete(
            Request(
                model=self.model, messages=chat(None, CLAIMS_EXTRACT.format(answer=answer)), max_tokens=256
            )
        )
        claims = parse_numbered(c1.text)
        if not claims:
            return JudgeResult(
                0.0,
                False,
                c1.text,
                c1.prompt_tokens,
                c1.completion_tokens,
                c1.latency_s,
                details={"claims": []},
            )
        numbered = "\n".join(f"{i}. {cl}" for i, cl in enumerate(claims, 1))
        c2 = self.provider.complete(
            Request(
                model=self.model,
                messages=chat(SYSTEM, CLAIMS_VERIFY.format(context=ctx, claims=numbered)),
                max_tokens=12 * len(claims) + 16,
            )
        )
        verdicts = parse_claim_verdicts(c2.text, len(claims))
        unsupported = sum(1 for v in verdicts if v is False)
        score = unsupported / len(claims)
        return JudgeResult(
            score,
            unsupported > 0,
            c1.text + "\n---\n" + c2.text,
            c1.prompt_tokens + c2.prompt_tokens,
            c1.completion_tokens + c2.completion_tokens,
            c1.latency_s + c2.latency_s,
            calls=2,
            details={"claims": claims, "verdicts": verdicts},
        )


def parse_numbered(text: str) -> list[str]:
    if re.match(r"^\W*none\b", text.strip(), re.I):
        return []
    items = re.findall(r"^\s*\d+[.)]\s*(.+?)\s*$", text, re.M)
    return [i for i in items if i][:8]


def parse_claim_verdicts(text: str, n: int) -> list[bool | None]:
    out: list[bool | None] = [None] * n
    for num, verdict in re.findall(
        r"(\d+)[.)]?\s*[:\-]?\s*\**\s*(UNSUPPORTED|SUPPORTED|NOT SUPPORTED)", text, re.I
    ):
        i = int(num) - 1
        if 0 <= i < n:
            out[i] = verdict.upper() == "SUPPORTED"
    return out

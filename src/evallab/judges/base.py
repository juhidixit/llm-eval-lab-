"""Shared judge plumbing: the verdict-with-probability pattern.

Judges ask a yes/no question and read the log-probabilities of the *first* answer token. The
normalised P("yes") is a continuous score, which gives three things a plain text verdict can't:
a ROC curve, a tunable decision threshold, and something to calibrate (Platt scaling) so that
"0.8" really means 80%. When a backend does not expose log-probs (e.g. the Anthropic API) the
score degrades gracefully to 0/1 parsed from the text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from evallab.providers.base import Completion, Provider, Request, chat


@dataclass
class JudgeResult:
    score: float  # P(problem) in [0, 1]; higher = more likely the thing the judge looks for
    verdict: bool  # score >= threshold
    raw: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    calls: int = 1
    details: dict = field(default_factory=dict)


def yes_probability(c: Completion) -> float | None:
    """Normalised P(yes) from first-token top log-probs, or None if neither token is present."""
    p_yes = p_no = 0.0
    for tok, lp in c.first_token_logprobs.items():
        t = tok.strip().lower().lstrip("*\"'")
        if not t:
            continue
        if t.startswith("yes") or t == "y":
            p_yes += math.exp(lp)
        elif t.startswith("no") or t == "n":
            p_no += math.exp(lp)
    if p_yes + p_no == 0:
        return None
    return p_yes / (p_yes + p_no)


_YES = re.compile(r"^\W*(yes)\b", re.I)
_NO = re.compile(r"^\W*(no)\b", re.I)


def parse_yes_no(text: str) -> bool | None:
    if _YES.search(text):
        return True
    if _NO.search(text):
        return False
    return None


def ask_yes_no(
    provider: Provider,
    model: str,
    system: str | None,
    user: str,
    max_tokens: int = 2,
) -> tuple[float, Completion]:
    """Ask a yes/no question; return (P(yes), completion)."""
    c = provider.complete(
        Request(model=model, messages=chat(system, user), max_tokens=max_tokens, top_logprobs=20)
    )
    p = yes_probability(c)
    if p is None:
        parsed = parse_yes_no(c.text)
        p = 0.5 if parsed is None else float(parsed)
    return p, c

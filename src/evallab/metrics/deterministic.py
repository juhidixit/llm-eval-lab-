"""Deterministic (non-LLM) metrics. Cheap, reproducible, and the baseline every judge must beat."""

from __future__ import annotations

import json
import re
import string
from collections import Counter

_STOP = set(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "as",
        "into",
        "about",
        "than",
        "then",
        "there",
        "their",
        "they",
        "he",
        "she",
        "his",
        "her",
        "them",
        "we",
        "you",
        "your",
        "our",
        "i",
        "not",
        "no",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "also",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "how",
        "based",
        "passage",
        "passages",
        "given",
    ]
)
_REFUSAL = re.compile(
    r"(unable to answer|cannot answer|can't answer|not (?:enough|sufficient) information|"
    r"do(?:es)? not (?:contain|provide|mention)|no information|not mentioned)",
    re.I,
)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    p, r = normalize(prediction).split(), normalize(reference).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def contains_any(prediction: str, needles: list[str]) -> float:
    low = prediction.lower()
    return float(any(n.lower() in low for n in needles))


def regex_match(prediction: str, pattern: str) -> float:
    return float(re.search(pattern, prediction) is not None)


def json_valid(prediction: str, required_keys: list[str] | None = None) -> float:
    try:
        obj = json.loads(prediction)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    if required_keys and (not isinstance(obj, dict) or any(k not in obj for k in required_keys)):
        return 0.0
    return 1.0


def is_refusal(prediction: str) -> bool:
    return bool(_REFUSAL.search(prediction))


def content_words(text: str) -> list[str]:
    return [w for w in normalize(text).split() if w not in _STOP and len(w) > 1]


def lexical_support(answer: str, contexts: list[str]) -> float:
    """Fraction of the answer's content words that appear anywhere in the contexts (1 = all).

    A short, pure refusal ("unable to answer ...") makes no claims, so it is fully supported by
    definition. Longer answers that refuse *and* add content are scored on their words.
    """
    if is_refusal(answer) and len(answer.split()) <= 25:
        return 1.0
    words = content_words(answer)
    if not words:
        return 1.0
    ctx = set(content_words(" ".join(contexts)))
    return sum(w in ctx for w in words) / len(words)

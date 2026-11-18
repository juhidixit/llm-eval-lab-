"""Provider interface shared by every model backend.

A provider turns a chat request into a :class:`Completion` that always carries token counts and
wall-clock latency, so cost and latency are tracked for every call without extra bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Request:
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int = 256
    seed: int | None = 0
    # Return the top-k log-probabilities of the *first* generated token. Judges use this to turn
    # a "yes"/"no" verdict into a continuous score that can be thresholded and calibrated.
    top_logprobs: int = 0
    stop: tuple[str, ...] = ()

    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [asdict(m) for m in self.messages],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "top_logprobs": self.top_logprobs,
                "stop": list(self.stop),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str
    # token -> logprob for the first generated position (empty unless requested)
    first_token_logprobs: dict[str, float] = field(default_factory=dict)
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Completion:
        return cls(**d)


class Provider(Protocol):
    name: str

    def complete(self, request: Request) -> Completion: ...


def chat(system: str | None, user: str) -> tuple[Message, ...]:
    msgs = [Message("system", system)] if system else []
    msgs.append(Message("user", user))
    return tuple(msgs)

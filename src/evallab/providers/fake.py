"""Deterministic scripted provider for unit tests."""

from __future__ import annotations

from collections.abc import Callable

from evallab.providers.base import Completion, Request


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        responder: Callable[[Request], str],
        logprobs: Callable[[Request], dict[str, float]] | None = None,
    ) -> None:
        self.responder = responder
        self.logprobs = logprobs
        self.calls: list[Request] = []

    def complete(self, request: Request) -> Completion:
        self.calls.append(request)
        text = self.responder(request)
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text.split()),
            latency_s=0.001,
            model=request.model,
            first_token_logprobs=self.logprobs(request) if self.logprobs else {},
        )

"""Claude via the official SDK (``pip install llm-eval-lab[anthropic]``)."""

from __future__ import annotations

import time

from evallab.providers.base import Completion, Request


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic  # imported lazily so the core install stays light

        self._client = anthropic.Anthropic()

    def complete(self, request: Request) -> Completion:
        system = "\n".join(m.content for m in request.messages if m.role == "system") or None
        messages = [{"role": m.role, "content": m.content} for m in request.messages if m.role != "system"]
        kwargs: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if request.stop:
            kwargs["stop_sequences"] = list(request.stop)
        t0 = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        latency = time.perf_counter() - t0
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        # The Messages API does not expose token log-probabilities, so judges fall back to
        # parsing the verdict text (score is then 0/1 and threshold tuning is a no-op).
        return Completion(
            text=text,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            latency_s=latency,
            model=request.model,
        )

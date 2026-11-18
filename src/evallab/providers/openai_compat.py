"""Any OpenAI-compatible chat endpoint: llama.cpp server, vLLM, Ollama, LM Studio, OpenAI.

Uses only the standard library so the core package has no heavy dependencies.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from evallab.providers.base import Completion, Request


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = (base_url or os.environ.get("EVALLAB_BASE_URL", "http://localhost:8080/v1")).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("EVALLAB_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network
                last = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"request to {self.base_url} failed: {last}")

    def complete(self, request: Request) -> Completion:
        payload: dict = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.top_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = request.top_logprobs
        t0 = time.perf_counter()
        data = self._post(payload)
        latency = time.perf_counter() - t0
        choice = data["choices"][0]
        first: dict[str, float] = {}
        lp = (choice.get("logprobs") or {}).get("content") or []
        if lp:
            for alt in lp[0].get("top_logprobs", []):
                first[alt["token"]] = float(alt["logprob"])
        usage = data.get("usage") or {}
        return Completion(
            text=choice["message"]["content"] or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=latency,
            model=request.model,
            first_token_logprobs=first,
        )

"""The *system under test* for the example suite: a minimal retrieval-augmented QA app.

It is deliberately small - a prompt template, a context budget and a model - because those are
exactly the knobs that change in real pull requests ("make the answers friendlier", "send fewer
chunks to save tokens", "switch to a cheaper model"). The eval suite measures what each change
does to faithfulness, refusals, latency and cost before it ships.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evallab.providers.base import Completion, Provider, Request, chat


@dataclass
class RagQAApp:
    provider: Provider
    model: str
    prompt_template: str
    max_contexts: int = 3
    max_tokens: int = 200

    @classmethod
    def from_config(cls, provider: Provider, cfg: dict, root: Path) -> RagQAApp:
        template = (root / cfg["prompt"]).read_text()
        return cls(
            provider=provider,
            model=cfg["model"],
            prompt_template=template,
            max_contexts=int(cfg.get("max_contexts", 3)),
            max_tokens=int(cfg.get("max_tokens", 200)),
        )

    def contexts_used(self, contexts: list[str]) -> list[str]:
        return contexts[: self.max_contexts]

    def __call__(self, question: str, contexts: list[str]) -> Completion:
        used = self.contexts_used(contexts)
        passages = "\n\n".join(f"passage {i}: {c}" for i, c in enumerate(used, 1))
        user = self.prompt_template.format(question=question, passages=passages, n=len(used))
        return self.provider.complete(
            Request(model=self.model, messages=chat(None, user), max_tokens=self.max_tokens, temperature=0.0)
        )

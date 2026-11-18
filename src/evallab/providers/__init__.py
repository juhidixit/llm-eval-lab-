from __future__ import annotations

from evallab.providers.base import Completion, Message, Provider, Request, chat
from evallab.providers.cache import CachedProvider, CacheMiss


def make_provider(kind: str, **kwargs) -> Provider:
    if kind in ("openai_compat", "llamacpp", "vllm", "ollama", "openai"):
        from evallab.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(**kwargs)
    if kind == "llamacpp_local":
        from evallab.providers.llamacpp_local import LlamaCppProvider

        return LlamaCppProvider(**kwargs)
    if kind == "anthropic":
        from evallab.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    raise ValueError(f"unknown provider {kind!r}")


__all__ = [
    "CacheMiss",
    "CachedProvider",
    "Completion",
    "Message",
    "Provider",
    "Request",
    "chat",
    "make_provider",
]

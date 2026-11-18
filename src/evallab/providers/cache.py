"""Record / replay cache ("cassettes") for LLM calls.

* ``record`` mode calls the wrapped provider and appends every completion to a JSONL cassette.
* ``replay`` mode never touches the network: a cache miss is an error. CI runs in replay mode,
  so the regression gate is deterministic, free and needs no secrets or GPU, while still being
  computed from real model outputs.
* ``auto`` replays hits and records misses (handy during development).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from evallab.providers.base import Completion, Provider, Request

Mode = Literal["record", "replay", "auto", "off"]


class CacheMiss(KeyError):
    pass


class CachedProvider:
    def __init__(self, inner: Provider | None, cassette: str | Path, mode: Mode = "auto") -> None:
        self.inner = inner
        self.path = Path(cassette)
        self.mode = mode
        self.name = f"cached({getattr(inner, 'name', 'none')})"
        self._store: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._store[rec["key"]] = rec["completion"]
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def complete(self, request: Request) -> Completion:
        key = request.cache_key()
        if self.mode != "off" and self.mode != "record" and key in self._store:
            self.hits += 1
            c = Completion.from_dict(self._store[key])
            c.cached = True
            return c
        if self.mode == "replay":
            raise CacheMiss(
                f"no recorded completion for request {key[:12]} in {self.path} "
                "(re-record with --cache record)"
            )
        if self.inner is None:
            raise CacheMiss("cache miss and no live provider configured")
        self.misses += 1
        comp = self.inner.complete(request)
        if self.mode != "off":
            self._store[key] = comp.to_dict()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "completion": comp.to_dict()}, ensure_ascii=False) + "\n")
        return comp

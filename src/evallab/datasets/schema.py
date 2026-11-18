"""Golden datasets: versioned, content-hashed JSONL.

Every example is one JSON object per line::

    {"id": "...", "input": {...}, "reference": "...", "labels": {...}, "tags": [...], "meta": {...}}

* ``input``     - whatever the system under test consumes (e.g. question + contexts).
* ``reference`` - optional gold answer (used by reference-based metrics such as context recall).
* ``labels``    - optional human labels (used to calibrate judges, never shown to them).
* ``tags``      - slices to report metrics by (e.g. ``"unanswerable"``, ``"numeric"``).

The dataset *version* is the SHA-256 of its canonicalised content. Runs record it, and the
regression gate refuses to compare runs made on different versions of the golden set, so a
silently edited dataset cannot masquerade as a model improvement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Example:
    id: str
    input: dict[str, Any]
    reference: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass
class Dataset:
    name: str
    examples: list[Example]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    @property
    def version(self) -> str:
        h = hashlib.sha256()
        for ex in sorted(self.examples, key=lambda e: e.id):
            h.update(ex.to_json().encode())
            h.update(b"\n")
        return h.hexdigest()[:16]

    def validate(self) -> None:
        ids = [e.id for e in self.examples]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate example ids in {self.name}: {sorted(dupes)[:5]}")
        for e in self.examples:
            if not isinstance(e.input, dict) or not e.input:
                raise ValueError(f"example {e.id} has an empty input")

    def slice(self, tag: str) -> Dataset:
        return Dataset(f"{self.name}[{tag}]", [e for e in self.examples if tag in e.tags])

    def head(self, n: int) -> Dataset:
        return Dataset(self.name, self.examples[:n])

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        path = Path(path)
        examples = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    examples.append(Example(**json.loads(line)))
        ds = cls(path.stem, examples)
        ds.validate()
        return ds

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for e in self.examples:
                f.write(e.to_json() + "\n")

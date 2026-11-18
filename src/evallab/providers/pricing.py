"""Token prices for cost tracking.

Prices are USD per million tokens and are *list prices entered by hand*; check your provider's
pricing page before trusting a projection. Local models are priced at $0 and tracked by latency
instead. Override or extend with ``EVALLAB_PRICES=path/to/prices.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


# Public list prices (USD / 1M tokens) used only for *projected* cost columns.
DEFAULT_PRICES: dict[str, Price] = {
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-sonnet-4-5": Price(3.00, 15.00),
    "gpt-4o-mini": Price(0.15, 0.60),
    "gpt-4o": Price(2.50, 10.00),
}


def load_prices() -> dict[str, Price]:
    prices = dict(DEFAULT_PRICES)
    path = os.environ.get("EVALLAB_PRICES")
    if path:
        with open(path) as f:
            for k, v in json.load(f).items():
                prices[k] = Price(float(v["input"]), float(v["output"]))
    return prices


def cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int, prices: dict[str, Price] | None = None
) -> float:
    p = (prices or load_prices()).get(model)
    if p is None:
        return 0.0
    return (prompt_tokens * p.input_per_mtok + completion_tokens * p.output_per_mtok) / 1e6

import json

import pytest

from evallab.datasets import Dataset, Example
from evallab.providers import CachedProvider, CacheMiss, Request, chat
from evallab.providers.fake import FakeProvider
from evallab.providers.pricing import cost_usd


def test_cache_records_then_replays(tmp_path):
    cassette = tmp_path / "c.jsonl"
    fake = FakeProvider(lambda r: "hello")
    req = Request(model="m", messages=chat("s", "u"))
    rec = CachedProvider(fake, cassette, mode="record")
    assert rec.complete(req).text == "hello"
    replay = CachedProvider(None, cassette, mode="replay")
    out = replay.complete(req)
    assert out.text == "hello" and out.cached and replay.hits == 1
    with pytest.raises(CacheMiss):
        replay.complete(Request(model="m", messages=chat("s", "different prompt")))


def test_cache_key_sensitive_to_every_field():
    a = Request(model="m", messages=chat(None, "u"))
    assert a.cache_key() != Request(model="m2", messages=chat(None, "u")).cache_key()
    assert a.cache_key() != Request(model="m", messages=chat(None, "u"), max_tokens=3).cache_key()


def test_cost():
    assert cost_usd("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.0)
    assert cost_usd("local-model", 10, 10) == 0.0


def test_dataset_roundtrip_and_version(tmp_path):
    ds = Dataset("d", [Example("1", {"q": "a"}, tags=["x"]), Example("2", {"q": "b"})])
    p = tmp_path / "d.jsonl"
    ds.save(p)
    back = Dataset.load(p)
    assert back.version == ds.version and len(back.slice("x")) == 1
    lines = p.read_text().splitlines()
    edited = json.loads(lines[1])
    edited["input"]["q"] = "changed"
    p.write_text(lines[0] + "\n" + json.dumps(edited) + "\n")
    assert Dataset.load(p).version != ds.version


def test_dataset_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        Dataset("d", [Example("1", {"q": "a"}), Example("1", {"q": "b"})]).validate()

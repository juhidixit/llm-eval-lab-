"""End-to-end: suite -> run -> gate, with a scripted model standing in for the LLMs."""

import math
from pathlib import Path

import yaml

from evallab import runner
from evallab.cli import main
from evallab.datasets import Dataset, Example
from evallab.providers.fake import FakeProvider


def _project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts/good.txt").write_text("Q: {question}\n{passages}\nstrict ({n})")
    (tmp_path / "prompts/bad.txt").write_text("Q: {question}\n{passages}\nbe creative ({n})")
    exs = [
        Example(
            f"a{i}",
            {"question": f"q{i}", "contexts": ["the sky is blue"]},
            reference="blue",
            tags=["answerable"],
        )
        for i in range(30)
    ]
    exs += [
        Example(
            f"u{i}",
            {"question": f"u{i}", "contexts": ["unrelated"]},
            reference="Unable to answer",
            tags=["unanswerable"],
        )
        for i in range(10)
    ]
    Dataset("golden", exs).save(tmp_path / "golden.jsonl")
    suite = {
        "name": "t",
        "dataset": "golden.jsonl",
        "system": {"model": "gen", "prompt": "prompts/good.txt", "max_contexts": 1},
        "provider": {"kind": "fake"},
        "judge": {"model": "judge", "strategy": "direct", "threshold": 0.5},
        "metrics": ["faithfulness", "refusal_correct", "answered", "token_f1"],
        "gates": {"faithfulness": {"min": 0.8, "max_drop": 0.03}, "refusal_correct": {"max_drop": 0.03}},
    }
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite))
    return tmp_path


def _fake(req):
    text = req.messages[-1].content
    if "be creative" in text:
        return "The sky is blue because of Rayleigh scattering discovered in 1871."
    if "strict" in text:
        return "Unable to answer based on given passages." if "unrelated" in text else "blue"
    return "Yes"


def _lp(req):
    hall = "Rayleigh" in req.messages[-1].content
    return {"Yes": math.log(0.9 if hall else 0.05), "No": math.log(0.1 if hall else 0.95)}


def test_suite_run_and_gate(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    monkeypatch.setattr(runner, "_live_provider", lambda cfg, root: FakeProvider(_fake, _lp))
    monkeypatch.chdir(root)
    assert main(["run", "suite.yaml", "--cache", "record", "--out", "base.json"]) == 0
    # Replay must reproduce the run from the cassette alone.
    assert main(["run", "suite.yaml", "--cache", "replay", "--out", "replay.json"]) == 0
    assert (
        main(["gate", "--baseline", "base.json", "--candidate", "replay.json", "--suite", "suite.yaml"]) == 0
    )
    # A worse prompt must be caught.
    assert (
        main(
            [
                "run",
                "suite.yaml",
                "--cache",
                "record",
                "--set",
                "system.prompt=prompts/bad.txt",
                "--out",
                "bad.json",
            ]
        )
        == 0
    )
    assert main(["gate", "--baseline", "base.json", "--candidate", "bad.json", "--suite", "suite.yaml"]) == 1
    assert (
        main(
            [
                "gate",
                "--baseline",
                "base.json",
                "--candidate",
                "bad.json",
                "--suite",
                "suite.yaml",
                "--expect",
                "fail",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "FAIL" in out and "faithfulness" in out

import math

import pytest

from evallab.judges import FaithfulnessJudge, yes_probability
from evallab.judges.faithfulness import parse_claim_verdicts, parse_numbered
from evallab.metrics import deterministic as det
from evallab.metrics.rag import RagMetrics, average_precision, split_sentences
from evallab.providers.base import Completion
from evallab.providers.fake import FakeProvider


def _c(lp):
    return Completion("x", 1, 1, 0.0, "m", lp)


def test_yes_probability_normalises_variants():
    p = yes_probability(_c({"Yes": math.log(0.6), " yes": math.log(0.1), "No": math.log(0.1), "Maybe": -1}))
    assert p == pytest.approx(0.7 / 0.8)
    assert yes_probability(_c({"Hmm": -0.1})) is None


def test_direct_judge_uses_logprobs_and_threshold():
    fake = FakeProvider(lambda r: "Yes", lambda r: {"Yes": math.log(0.3), "No": math.log(0.7)})
    j = FaithfulnessJudge(fake, "m", strategy="direct", threshold=0.25)
    r = j("q", ["ctx"], "ans")
    assert r.score == pytest.approx(0.3) and r.verdict is True
    assert fake.calls[0].top_logprobs == 20 and fake.calls[0].max_tokens == 2


def test_direct_judge_falls_back_to_text_without_logprobs():
    j = FaithfulnessJudge(FakeProvider(lambda r: "No."), "m")
    assert j("q", ["c"], "a").score == 0.0


def test_cot_judge_parses_last_verdict():
    text = "Sentence 1 ok. VERDICT: FAITHFUL? no wait.\nVERDICT: **HALLUCINATED**"
    r = FaithfulnessJudge(FakeProvider(lambda r: text), "m", strategy="cot")("q", ["c"], "a")
    assert r.score == 1.0 and r.details["parsed_verdict"]


def test_claims_judge_scores_fraction_unsupported():
    def responder(req):
        if "Break the following" in req.messages[-1].content:
            return "1. Paris is in France.\n2. Paris has 90 million people.\n3. It is a capital."
        return "1. SUPPORTED\n2. UNSUPPORTED\n3. SUPPORTED"

    r = FaithfulnessJudge(FakeProvider(responder), "m", strategy="claims")("q", ["c"], "a")
    assert r.score == pytest.approx(1 / 3) and r.verdict and r.calls == 2


def test_claim_parsers():
    assert parse_numbered("NONE") == []
    assert parse_numbered("1) a\n2. b") == ["a", "b"]
    assert parse_claim_verdicts("2. UNSUPPORTED\n1: SUPPORTED", 3) == [True, False, None]


def test_deterministic_metrics():
    assert det.exact_match("The Eiffel Tower!", "eiffel tower") == 1.0
    assert det.token_f1("paris france", "paris") == pytest.approx(2 / 3)
    assert det.json_valid('{"a": 1}', ["a"]) == 1.0 and det.json_valid("{", None) == 0.0
    assert det.is_refusal("Unable to answer based on given passages.")
    assert not det.is_refusal("The phone number is (510) 889-8690.")
    assert det.lexical_support("Paris is lovely", ["Paris is the capital"]) == pytest.approx(0.5)


def test_rag_metrics_with_fake_judge():
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, False]) == 0.0
    assert split_sentences("One fact here. Two facts here! Three?") == [
        "One fact here.",
        "Two facts here!",
        "Three?",
    ]

    def lp(req):
        good = "RELEVANT" in req.messages[-1].content
        return {"Yes": math.log(0.9 if good else 0.1), "No": math.log(0.1 if good else 0.9)}

    rm = RagMetrics(FakeProvider(lambda r: "Yes", lp), "m")
    assert rm.context_precision("q", ["noise", "RELEVANT chunk", "noise"]) == pytest.approx(0.5)
    assert rm.context_recall("A. B.", ["RELEVANT"]) == 1.0

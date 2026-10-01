"""Promotion graph loading and evaluation contract tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import promote_model as pm
import pytest
from promote_model import Config, PromotionError
from promotion_helpers import (
    LABELS,
    FakeConfig,
    FakeSession,
    FakeTokenizer,
    make_cfg,
    make_repo,
    metrics,
    restore_eval,
)


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


@pytest.fixture
def cfg(repo):
    return make_cfg(repo)


@pytest.fixture(autouse=True)
def _restore_eval():
    with restore_eval():
        yield


def test_evaluate_graph_attaches_the_config_that_keeps_label_order(
    repo: Path, monkeypatch
) -> None:
    """The gate must read id2label, not infer it.

    An ORT session has no ``.config``, so without this line ``evaluate_on_split``
    falls back to alphabetical label order -- negative, neutral, positive --
    against a model whose ids are positive=0, negative=1, neutral=2. Every
    metric would still be computed and would still look plausible, with Neutral
    and Negative silently swapped: the worst possible failure for a gate, since
    it produces a confident wrong answer rather than an error.
    """
    import transformers

    from raay.training import evaluate as ev

    session = FakeSession()
    seen: dict = {}

    def fake_eval(frame, sess, tokenizer, model_name, max_length):
        seen["config"] = sess.config
        return metrics()

    seen_options: dict = {}

    def fake_session_load(path, *, session_options=None):
        seen_options["options"] = session_options
        return session

    monkeypatch.setattr(ev, "load_onnx_session", fake_session_load)
    monkeypatch.setattr(ev, "evaluate_on_split", fake_eval)
    monkeypatch.setattr(ev, "dialect_breakdown", lambda *a, **k: {})
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: FakeConfig()
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )

    cfg = dataclasses.replace(
        Config(
            test_split=repo / "test.csv",
            tokenizer_dir=repo / "tok",
            dvc_lock=repo / "dvc.lock",
        ),
        latency_runs=2,
        latency_warmup=1,
    )
    frame = pd.DataFrame({"text": ["مراجعة جيدة", "خدمة سيئة", "عادي"]})

    out = pm.evaluate_graph(cfg, str(repo / "prod.onnx"), frame)

    assert isinstance(seen["config"], FakeConfig), "id2label was never attached"
    # The session being timed is also the session that was scored, so this is
    # the only place the gate's session configuration can be pinned. Spinning
    # workers are what made two sessions in one process look 25% apart.
    options = seen_options["options"]
    assert options is not None, "the gate scored the graph on default options"
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    # Spinning off, pool intact. Pinning intra_op_num_threads=1 was the obvious
    # "reduce the noise" move and it makes this graph ~3x slower (measured here:
    # 242 ms against 86 ms per call), which would change what the gate is
    # measuring rather than stabilise it.
    assert options.intra_op_num_threads == 0, "the thread pool must stay at its default"
    assert [seen["config"].id2label[i] for i in range(3)] == list(LABELS)
    assert out["label_names"] == list(LABELS)
    assert out["id2label"] == {"0": "positive", "1": "negative", "2": "neutral"}
    # 1 warmup call is made and then discarded, so 3 calls yield 2 samples --
    # keeping the warmup in the percentile would understate p95.
    assert session.runs == 3, "the latency measurement did not actually run"
    assert out["latency"]["runs"] == 2, "the warmup call leaked into the samples"


def test_the_warmup_is_long_enough_to_reach_steady_state() -> None:
    """Pinned because the evidence for it is a one-off measurement, not a law.

    Three warm-up calls were measured leaving the first of the three series
    still paying for the other two graphs' pools coming up, which is a
    systematic bias rather than noise -- and systematic bias is exactly what a
    rotating order does not cancel. Ten costs about a second.
    """
    assert Config().latency_warmup == 10
    assert Config().latency_runs == 50


def test_the_shared_session_loader_keeps_its_default_behaviour(monkeypatch) -> None:
    """Passing options must be opt-in, or this changes the parity checker too.

    ``load_onnx_session`` is also used by ``quantize_onnx`` and ``evaluate``.
    The gate needs tuned options; those callers do not, and silently giving them
    no-spinning sessions would change what every other latency number in the
    repo means.
    """
    import onnxruntime as ort

    from raay.training import evaluate as ev

    calls: list[dict] = []

    def fake_session(path, **kwargs):
        calls.append(kwargs)
        return "session"

    monkeypatch.setattr(ort, "InferenceSession", fake_session)

    ev.load_onnx_session("model.onnx")
    assert "sess_options" not in calls[0], "the default path started passing options"

    options = ort.SessionOptions()
    ev.load_onnx_session("model.onnx", session_options=options)
    assert calls[1]["sess_options"] is options
    assert calls[1]["providers"] == ["CPUExecutionProvider"]


def test_evaluate_graph_refuses_a_missing_graph(repo: Path) -> None:
    cfg = Config(test_split=repo / "test.csv")
    with pytest.raises(PromotionError, match="graph not found"):
        pm.evaluate_graph(cfg, str(repo / "absent.onnx"), pd.DataFrame({"text": ["x"]}))

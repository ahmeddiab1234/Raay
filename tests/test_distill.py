from types import SimpleNamespace

import numpy as np
import torch

from raay.training.distill import (
    _cached_teacher_logits,
    _resolve_cache_path,
    distill_loss,
)


def test_distill_loss_zero_when_teacher_and_student_agree():
    torch.manual_seed(0)
    # Very confident, label-aligned logits: softmax is ~one-hot so both the CE
    # and KL terms vanish.
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    labels = torch.tensor([0, 1])

    loss = distill_loss(logits, logits.clone(), labels, alpha=0.4, temperature=4.0)
    assert loss.ndim == 0
    assert loss.item() < 1e-4


def test_distill_loss_blends_both_terms():
    torch.manual_seed(0)
    student = torch.tensor([[2.0, 1.0, 0.1], [0.5, 3.0, 1.0]])
    teacher = torch.tensor([[1.0, 2.0, 0.5], [2.0, 0.5, 3.0]])
    labels = torch.tensor([0, 0])

    alpha = 0.4
    loss = distill_loss(student, teacher, labels, alpha=alpha, temperature=4.0)

    ce = torch.nn.functional.cross_entropy(student, labels)
    t = 4.0
    kl = (
        torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student / t, dim=-1),
            torch.nn.functional.softmax(teacher / t, dim=-1),
            reduction="batchmean",
        )
        * t
        * t
    )
    expected = alpha * ce + (1.0 - alpha) * kl

    assert torch.allclose(loss, expected)


def test_distill_loss_alpha_edge_cases():
    torch.manual_seed(0)
    student = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    teacher = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    labels = torch.tensor([0, 1])

    # alpha=1 => pure hard-label CE (no teacher influence).
    alpha1 = distill_loss(student, teacher, labels, alpha=1.0, temperature=4.0)
    pure_ce = torch.nn.functional.cross_entropy(student, labels)
    assert torch.allclose(alpha1, pure_ce)

    # alpha=0 => pure temperature-scaled KL (labels irrelevant).
    alpha0 = distill_loss(student, teacher, labels, alpha=0.0, temperature=4.0)
    t = 4.0
    pure_kl = (
        torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student / t, dim=-1),
            torch.nn.functional.softmax(teacher / t, dim=-1),
            reduction="batchmean",
        )
        * t
        * t
    )
    assert torch.allclose(alpha0, pure_kl)


def test_resolve_cache_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(teacher_logits_cache="models/distilled/teacher_logits")
    path = _resolve_cache_path("train", cfg)
    assert str(path) == str(
        tmp_path / "models" / "distilled" / "teacher_logits" / "train_logits.npy"
    )


def test_resolve_cache_path_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(teacher_logits_cache="")
    assert _resolve_cache_path("train", cfg) is None


def test_cached_teacher_logits_reuses_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(
        teacher_logits_cache="models/distilled/teacher_logits",
        batch_size=2,
        max_length=8,
    )
    expected = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    cache_path = _resolve_cache_path("train", cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, expected)

    calls = []

    def fake_teacher(model, tokenizer, texts, cfg_):
        calls.append(len(texts))
        return np.zeros((len(texts), 3))

    monkeypatch.setattr("raay.training.distill._teacher_logits", fake_teacher)
    logits = _cached_teacher_logits(None, None, ["a", "b"], "train", cfg)

    # Cache hit: the (expensive) teacher forward must not run.
    assert calls == []
    np.testing.assert_array_equal(logits, expected)


def test_cached_teacher_logits_recomputes_on_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(
        teacher_logits_cache="models/distilled/teacher_logits",
        batch_size=2,
        max_length=8,
    )
    # 1-row cache does not match the 2-row split => recompute + overwrite.
    cache_path = _resolve_cache_path("train", cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.zeros((1, 3)))

    def fake_teacher(model, tokenizer, texts, cfg_):
        return np.tile(np.arange(3, dtype=float), (len(texts), 1))

    monkeypatch.setattr("raay.training.distill._teacher_logits", fake_teacher)
    logits = _cached_teacher_logits(None, None, ["a", "b"], "train", cfg)

    np.testing.assert_array_equal(logits, np.tile(np.arange(3.0), (2, 1)))
    np.testing.assert_array_equal(np.load(cache_path), logits)

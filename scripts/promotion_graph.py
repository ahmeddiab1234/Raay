"""Scoring one graph on the frozen split, the way the service would score it.

``AutoConfig`` is pinned onto the ORT session here on purpose. A session has no
``.config``, and ``evaluate_on_split`` reads id2label from there, so without the
pin the labels fall back to alphabetical order (negative, neutral, positive)
while the model's ids are positive=0, negative=1, neutral=2 -- no exception, a
plausible confusion matrix, and Neutral and Negative silently swapped. That line
has been deleted once and the whole suite still passed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from promotion_types import Config, Graph, PromotionError, _round


def load_graph(cfg: Config, onnx_path: str, frame: pd.DataFrame) -> Graph:
    """Score one graph on the frozen split the way the service would.

    Latency is deliberately *not* measured here. Two graphs have to be timed
    against each other in one interleaved loop to get a number that means
    anything, and that is :func:`promotion_timing.measure_latency_pair`'s job.
    """
    from promotion_timing import latency_session_options
    from transformers import AutoConfig, AutoTokenizer

    from raay.training.evaluate import (
        dialect_breakdown,
        evaluate_on_split,
        load_onnx_session,
    )

    if not Path(onnx_path).exists():
        raise PromotionError(f"graph not found: {onnx_path}")
    session = load_onnx_session(onnx_path, session_options=latency_session_options())

    session.config = AutoConfig.from_pretrained(cfg.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)

    report = evaluate_on_split(
        frame, session, tokenizer, cfg.model_name, cfg.max_length
    )
    report["dialect_breakdown"] = (
        dialect_breakdown(frame, session, tokenizer, cfg.model_name, cfg.max_length)
        if cfg.dialect_breakdown
        else None
    )
    id2label = {
        int(k): v for k, v in (getattr(session.config, "id2label", None) or {}).items()
    }
    size_mb = Path(onnx_path).stat().st_size / 1_048_576
    metrics = {
        "onnx_path": onnx_path,
        "size_mb": _round(size_mb, 2),
        "label_names": report["label_names"],
        "id2label": {str(k): v for k, v in id2label.items()},
        "accuracy": _round(report["accuracy"]),
        "f1_macro": _round(report["f1_macro"]),
        "f1_weighted": _round(report["f1_weighted"]),
        "per_class": {
            name: {
                "precision": _round(values["precision"]),
                "recall": _round(values["recall"]),
                "f1": _round(values["f1"]),
                "support": values["support"],
            }
            for name, values in report["per_class"].items()
        },
        "confusion_matrix": report["confusion_matrix"],
        "sample_size": report["sample_size"],
        "metadata": report.get("metadata", {}),
    }
    return Graph(metrics=metrics, session=session, tokenizer=tokenizer)


def evaluate_graph(cfg: Config, onnx_path: str, frame: pd.DataFrame) -> dict[str, Any]:
    """``load_graph`` plus a standalone timing of that one graph."""
    from promotion_timing import measure_latency

    graph = load_graph(cfg, onnx_path, frame)
    measure_latency(graph, frame["text"].tolist()[:64], cfg)
    return graph.metrics


def read_parity(cfg: Config) -> dict[str, Any]:
    """The existing ONNX-vs-PyTorch logit check, loaded rather than re-derived.

    Re-exporting the graph here would need the 540 MB fp32 checkpoint and a
    torch install, neither of which belongs in a promotion gate. The report is a
    committed artifact of the export step, so it is read -- and its absence is a
    gate failure rather than a silent skip.
    """
    if not cfg.parity_report.exists():
        raise PromotionError(
            f"{cfg.parity_report} is missing; run "
            "`python -m raay.inference.quantize_onnx` to produce it"
        )
    return json.loads(cfg.parity_report.read_text())


def _parity_section(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten the report's shape: it is either flat or keyed by variant."""
    if "max_abs_diff" in report:
        return report
    for value in report.values():
        if isinstance(value, dict) and "max_abs_diff" in value:
            return value
    raise PromotionError("parity report has no max_abs_diff")

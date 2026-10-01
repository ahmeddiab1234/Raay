"""Shared fixtures for the nightly drift tests.

Hermetic by design (AGENTS.md rule: unit tests only): no ONNX graph, no MLflow,
no Airflow, no encoder weights. The duck-typed fakes in ``conftest.py`` stand in
for the INT8 session and the AraBERT encoder; drift checks run on tiny synthetic
DataFrames.
"""

from __future__ import annotations

import pandas as pd
from conftest import FakeEmbedder, FakeTokenizer, ascii_vocab, identity, scored_frame

from raay.inference.drift_engine import EngineeredDriftSpec
from raay.inference.drift_features import DriftFeatureBuilder, save_basis


def pool_csv(tmp_path, n: int = 60) -> str:
    df = pd.DataFrame({"text": [f"مراجعة {i}" for i in range(n)]})
    path = tmp_path / "pool.csv"
    df.to_csv(path, index=False)
    return str(path)


def write_frame(tmp_path, frame: pd.DataFrame, name: str) -> str:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


def engineered_spec(
    tmp_path, n_components: int = 3, reference: pd.DataFrame | None = None
):
    """An ``EngineeredDriftSpec`` wired to the fakes, with a fitted basis on disk.

    Returns ``(spec, builder)``. The basis is fitted on ``reference`` (or a
    default panel) so ``drift_check`` has a frozen projection to load, exactly
    as ``init-reference`` would have written.
    """
    builder = DriftFeatureBuilder(
        FakeEmbedder(),
        FakeTokenizer(ascii_vocab()),
        n_components=n_components,
        model_dir="models/baseline/final",
        max_length=128,
        preprocess=identity,
    )
    ref = reference if reference is not None else scored_frame(60)
    basis = builder.fit_basis(builder.embed(ref))
    pca_path = str(tmp_path / "pca_basis.joblib")
    save_basis(basis, pca_path)
    spec = EngineeredDriftSpec(
        builder=builder,
        pca_path=pca_path,
        reference_engineered=str(tmp_path / "reference_engineered.csv"),
        current_engineered=str(tmp_path / "current_engineered.csv"),
    )
    return spec, builder

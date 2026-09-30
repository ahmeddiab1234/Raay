"""Nightly batch re-scoring job with an Evidently PSI drift check.

Phase 5 step 3: a day's worth of reviews is scored offline in LARGE batches
directly against the INT8 ONNX graph (``serve.predict_probs``, in-process --
no REST call), the predictions are written to ``data/scoring/output/`` and an
Evidently PSI drift check is run against a scored reference set. Airflow
orchestrates the three steps nightly (see ``airflow/dags/raay_nightly_batch_scoring.py``).

Modes
-----
- ``make-input``: emulate "a day's worth of reviews" by deterministically
  sampling ``data/processed/test.csv`` (seeded by date) into
  ``data/scoring/input/{date}.csv``. There is no live scraper in this repo, so
  the pool stands in for new production reviews.
- ``score``: score an input CSV in ``--batch-size`` chunks and write
  ``data/scoring/output/{date}.csv`` with per-class probabilities, the argmax
  label and its score. Enforces the ``--min-samples`` (default 1000) floor.
- ``init-reference``: build the drift reference once by scoring a seeded slice
  of the pool into ``data/scoring/reference/reference.csv``. It then writes two
  more artifacts: ``reference_engineered.csv`` (the reference plus the Phase-6
  engineered input features) and ``pca_basis.joblib`` (the PCA projection
  frozen on the reference embeddings).
- ``drift``: Evidently PSI (``ColumnDriftMetric`` + ``psi_stat_test``) on the
  current day vs the reference; each column is PASS (<0.1) / WARN (<0.2) /
  FAIL (>=0.2) and the verdicts are written to ``reports/drift/{date}.json``.

Phase 6 step 1 extended the drift columns from the model's own outputs
(``predicted_label``, ``positive``) to the **inputs** as well: AraBERT
embedding principal components, the ``[UNK]`` rate, the dialect mix, review
length and confidence. Those come from ``raay.inference.drift_features`` and
need the fine-tuned encoder, so they are opt-out via ``--no-engineer``.

The report is a list of per-column ``ColumnDriftMetric``s rather than an
Evidently ``DataDriftPreset``: a preset reports one dataset-level score and
per-column detail only for a handful of auto-detected columns, and this job
needs a separate PASS/WARN/FAIL verdict plus a PSI for *each* of the 16
engineered features. Per-column metrics are what makes that possible; the
preset would only add ``drift_share``, which is computed directly here.

Every run also logs params/metrics/artifacts to the ``raay_batch`` MLflow
experiment (unless ``--no-mlflow``).

Run (from the repo root any of these):

    uv run python -m raay.inference.batch_score --mode init-reference --samples 2000
    uv run python -m raay.inference.batch_score --mode make-input --date 2026-09-24
    uv run python -m raay.inference.batch_score --mode score --date 2026-09-24
    uv run python -m raay.inference.batch_score --mode drift --date 2026-09-24

``init-reference`` is a setup step, not a nightly one (the Airflow DAG runs the
other three): the reference and its PCA basis are frozen on purpose.

Honest caveat: the pool is a single dataset, so real-world PSI drift will be ~0
by construction; the mechanism (and the Phase-5 monitoring hook it provides) is
the deliverable. The engineered columns do not escape this -- they compare
slices of one corpus, not production traffic.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DataColumns, DefaultPaths, Experiments, Models
from raay.inference.drift_features import (
    COL_CONFIDENCE,
    COL_DIALECT_LABEL,
    COL_OOV_BUCKET,
    COL_OOV_RATE,
    COL_TEXT_LENGTH,
    DriftFeatureBuilder,
    ProjectionBasis,
    default_drift_columns,
    dialect_mix,
    dialect_total_variation,
    feature_means,
    load_basis,
    save_basis,
)
from raay.serving.serve import predict_probs


def _date_seed(date: str) -> int:
    """Deterministic uint32 seed for a date string (same day -> same sample)."""
    return zlib.crc32(date.encode("utf-8"))


@dataclass
class Scorer:
    """Lazy INT8 ONNX scorer sharing ``serve.predict_probs``.

    ``score`` returns the full row-softmax probability matrix so the nightly
    job can persist every class probability, not just the argmax.
    """

    onnx_path: str = DefaultPaths.ONNX_INT8_MODEL.value
    tokenizer_dir: str = DefaultPaths.BASELINE_MODEL.value
    model_name: str = Models.TEACHER.value
    max_length: int = 128
    batch_size: int = 64
    _session: Any = field(init=False, repr=False, default=None)
    _tokenizer: Any = field(init=False, repr=False, default=None)
    _id2label: dict[int, str] = field(init=False, repr=False, default_factory=dict)
    _loaded: bool = field(init=False, repr=False, default=False)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._session = ort.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        config = AutoConfig.from_pretrained(self.tokenizer_dir)
        raw = getattr(config, "id2label", None) or {}
        self._id2label = {int(k): v for k, v in raw.items()}
        self._loaded = True
        logger.info(
            f"Batch scorer loaded {self.onnx_path} labels={self._id2label} "
            f"providers={self._session.get_providers()}"
        )

    @property
    def label_columns(self) -> list[str]:
        return [self._id2label[i] for i in sorted(self._id2label)]

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.label_columns)))
        self._ensure_loaded()
        assert self._session is not None and self._tokenizer is not None
        return predict_probs(
            self._session,
            self._tokenizer,
            texts,
            self.model_name,
            self.max_length,
            batch_size=self.batch_size,
        )


def make_input(
    pool_csv: str,
    date: str,
    samples: int,
    out_csv: str,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample ``samples`` reviews from the pool (seeded by date) into ``out_csv``."""
    df = pd.read_csv(pool_csv)
    if len(df) < samples:
        raise ValueError(
            f"pool has {len(df)} rows, need at least {samples}; pick a smaller --samples"
        )
    rng = np.random.default_rng(seed if seed is not None else _date_seed(date))
    sample = df.sample(n=samples, random_state=rng).copy()
    sample.insert(0, "day", date)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(sample)} reviews to {out_csv}")
    return sample


def score_input(
    input_csv: str,
    out_csv: str,
    scorer: Scorer,
    min_samples: int,
    target_label: str = "label",
) -> dict[str, Any]:
    """Score every text in ``input_csv`` in batches and persist predictions."""
    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError(
            f"input CSV must have a 'text' column (got {list(df.columns)})"
        )
    n = len(df)
    if n < min_samples:
        raise ValueError(
            f"input has {n} rows; nightly batch should re-score at least "
            f"{min_samples} (got --min-samples {min_samples})"
        )
    t0 = time.perf_counter()
    probs = scorer.score(df["text"].astype(str).tolist())
    elapsed = time.perf_counter() - t0
    cols = scorer.label_columns
    for i, col in enumerate(cols):
        df[col] = probs[:, i]
    arg = np.argmax(probs, axis=1)
    df["predicted_label"] = [cols[int(i)] for i in arg]
    df["predicted_score"] = probs[np.arange(n), arg]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Scored {n} reviews in {elapsed:.2f}s -> {out_csv}")
    prop = df["predicted_label"].value_counts().to_dict()
    return {
        "n_reviews": n,
        "elapsed_sec": elapsed,
        "batch_size": scorer.batch_size,
        "label_columns": cols,
        "class_proportions": {k: int(v) for k, v in prop.items()},
        "target_label": target_label,
    }


@dataclass(frozen=True)
class EngineeredDriftSpec:
    """Everything the drift check needs to add the engineered input features.

    Bundled into one object (and injected as ``None`` for the legacy path)
    rather than threaded through six parameters: ``drift_check`` has to keep
    working byte-identically when the features are off, which is what the
    existing unit tests exercise, and a boolean flag is the easy way to
    accidentally flip that default later.
    """

    builder: DriftFeatureBuilder
    pca_path: str
    reference_engineered: str
    current_engineered: str

    @property
    def n_components(self) -> int:
        return self.builder.n_components


def init_reference(
    pool_csv: str,
    samples: int,
    out_csv: str,
    scorer: Scorer,
    engineered: EngineeredDriftSpec | None = None,
) -> pd.DataFrame:
    """Score a fixed slice of the pool once; the drift reference for all days.

    With ``engineered``, also embeds the reference, fits the PCA basis on those
    embeddings and writes the engineered reference panel. The basis is fitted
    here and only here -- it is the reference distribution's projection, and
    refitting it per day would rotate the axes under the comparison.
    """
    make_input(pool_csv, "reference", samples, out_csv, seed=0)
    score_input(out_csv, out_csv, scorer, min_samples=1)
    frame = pd.read_csv(out_csv)
    if engineered is None:
        return frame
    embeddings = engineered.builder.embed(frame)
    basis = engineered.builder.fit_basis(embeddings)
    save_basis(basis, engineered.pca_path)
    featured = engineered.builder.engineer(frame, basis, embeddings)
    Path(engineered.reference_engineered).parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(engineered.reference_engineered, index=False)
    logger.info(
        f"Wrote engineered reference ({len(featured)} rows, "
        f"{basis.n_components} PCs) to {engineered.reference_engineered}"
    )
    return featured


def _engineered_frames(
    reference_csv: str,
    current_csv: str,
    spec: EngineeredDriftSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, ProjectionBasis, bool]:
    """Reference/current frames with the engineered columns attached.

    The reference is read from the cached ``reference_engineered.csv`` when it
    exists (that is the frozen side -- recomputing it nightly would be wasted
    encoder work and would silently re-derive the basis if the panel changed).
    Otherwise it is engineered on the fly from the scored reference, which
    costs a full extra encoder pass but keeps the check runnable against a
    reference built by an older version.

    Returns ``(ref, cur, basis, from_cache)``. The current frame is always
    recomputed and written out, because that is the panel a human opens when a
    gate fails.
    """
    basis = load_basis(spec.pca_path)
    basis.check_compatible(
        model_dir=spec.builder.model_dir,
        max_length=spec.builder.max_length,
        pooling=spec.builder.pooling,
    )
    if Path(spec.reference_engineered).exists():
        ref = pd.read_csv(spec.reference_engineered)
        from_cache = True
    else:
        logger.warning(
            f"{spec.reference_engineered} is missing; engineering the reference "
            f"from {reference_csv} on the fly (one extra encoder pass)"
        )
        ref = spec.builder.engineer(pd.read_csv(reference_csv), basis)
        from_cache = False
    cur = spec.builder.engineer(pd.read_csv(current_csv), basis)
    Path(spec.current_engineered).parent.mkdir(parents=True, exist_ok=True)
    cur.to_csv(spec.current_engineered, index=False)
    logger.info(f"Wrote engineered current panel to {spec.current_engineered}")
    return ref, cur, basis, from_cache


# Below this reference std a numeric column is treated as having no spread at
# all; see _uncomparable_reason.
_MIN_COMPARABLE_STD = 1e-8


def _uncomparable_reason(reference: pd.Series) -> dict[str, Any] | None:
    """Why this reference column has no distribution to compare, or ``None``.

    A numeric column whose reference std is below ``_MIN_COMPARABLE_STD`` cannot
    be binned by Evidently and cannot drift: the tail components of a PCA basis
    sit at float-noise around the component mean. The threshold is absolute
    because PCA components of mean-pooled AraBERT embeddings are O(0.1-10), so
    anything under ``1e-8`` is seven-plus orders of magnitude below the leading
    components -- numerically indistinguishable from the mean, not a small but
    real signal.

    Categorical columns are never skipped here: a constant *categorical* column
    (every row the same dialect) is still a comparable distribution, because
    Evidently bins categories rather than a value range.
    """
    if not pd.api.types.is_numeric_dtype(reference):
        return None
    values = reference.dropna()
    if values.empty:
        return {"reason": "reference column is entirely NaN", "reference_std": 0.0}
    std = float(values.std())
    if std >= _MIN_COMPARABLE_STD:
        return None
    return {
        "reason": (
            f"reference std {std:.3e} is below {_MIN_COMPARABLE_STD:g}, so the "
            f"column has no spread to bin or to drift in"
        ),
        "reference_std": std,
    }


def _psi_per_column(
    columns: list[str],
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    thresholds: tuple[float, float],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Run PSI one column at a time; a column that cannot be binned is ERRORed.

    Two failure modes, deliberately reported differently:

    **SKIPPED (uncomparable).** The reference column has no usable spread, so
    there is no distribution to compare. This is not hypothetical: the tail
    principal components of a PCA basis routinely land at ``~1e-16`` std, i.e.
    floating-point noise around the component mean, and Evidently's
    ``numpy.histogram_bin_edges(bins="sturges")`` then raises ``Too many bins
    for data range``. Gating those would fail the report for a component that
    by construction cannot move, so they are recorded with their observed std
    and left out of the severity calculation -- with the number visible, so a
    reader can see *why* the column was not checked.

    **ERROR (escalates).** Anything else, e.g. a binning failure on a column
    that does have spread. That is a bug or an unanticipated input shape, and a
    monitoring gate that silently stops checking a column is worse than one that
    pages someone, so it is recorded with ``decision: "ERROR"`` and lifts the
    overall verdict.

    Running per column rather than as one ``Report`` is what makes either
    outcome survivable: a single exception from the shared report would take
    down every column, not just the broken one.
    """
    from evidently.legacy.calculations.stattests.psi import psi_stat_test
    from evidently.legacy.metrics import ColumnDriftMetric
    from evidently.legacy.report import Report

    lo, hi = thresholds
    verdicts: dict[str, dict[str, Any]] = {}
    errored: list[str] = []
    uncomparable: list[str] = []
    for col in columns:
        degenerate = _uncomparable_reason(ref[col])
        if degenerate is not None:
            logger.warning(
                f"Skipping drift column {col!r}: {degenerate['reason']} "
                f"(observed std={degenerate['reference_std']:.3e})"
            )
            verdicts[col] = {
                "drift_score": None,
                "stattest": "PSI",
                "stattest_threshold": None,
                "drift_detected": None,
                "decision": "SKIPPED",
                **degenerate,
            }
            uncomparable.append(col)
            continue
        try:
            report = Report(
                metrics=[ColumnDriftMetric(column_name=col, stattest=psi_stat_test)]
            )
            report.run(reference_data=ref, current_data=cur)
            result = report.as_dict()["metrics"][0]["result"]
            score = float(result["drift_score"])
            verdicts[col] = {
                "drift_score": round(score, 4),
                "stattest": result["stattest_name"],
                "stattest_threshold": float(result["stattest_threshold"]),
                "drift_detected": bool(result["drift_detected"]),
                "decision": "PASS"
                if score < lo
                else ("WARN" if score < hi else "FAIL"),
            }
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            logger.warning(f"PSI failed for column {col!r}: {exc!r}")
            verdicts[col] = {
                "drift_score": None,
                "stattest": "PSI",
                "stattest_threshold": None,
                "drift_detected": None,
                "decision": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            errored.append(col)
    return verdicts, errored, uncomparable


def drift_check(
    reference_csv: str,
    current_csv: str,
    out_json: str,
    date: str,
    thresholds: tuple[float, float] = (0.1, 0.2),
    drift_columns: tuple[str, ...] | None = None,
    engineered: EngineeredDriftSpec | None = None,
) -> dict[str, Any]:
    """Evidently PSI drift between reference and a scored day; write verdicts.

    With ``engineered`` the comparison widens from the model's outputs to the
    inputs as well (embedding PCs, ``[UNK]`` rate, dialect mix, length,
    confidence). Requested columns that are not present in the frames are
    skipped and named in ``skipped_columns`` rather than being silently
    dropped or crashing the job -- a clamped PCA basis legitimately has fewer
    components than requested.
    """

    ref = pd.read_csv(reference_csv)
    cur = pd.read_csv(current_csv)
    reference_label, current_label = reference_csv, current_csv
    basis: ProjectionBasis | None = None
    reference_from_cache = False
    features_applied = False

    if engineered is not None:
        if DataColumns.TEXT.value not in ref.columns:
            # No text means no embeddings, no OOV, no recomputed dialect. Say so
            # and fall through to the output-only comparison rather than
            # reporting engineered columns that were never computed.
            logger.warning(
                f"{reference_csv} has no 'text' column, so the engineered input "
                f"features are unavailable; falling back to the output-only "
                f"drift columns"
            )
        else:
            ref, cur, basis, reference_from_cache = _engineered_frames(
                reference_csv, current_csv, engineered
            )
            reference_label = (
                engineered.reference_engineered
                if reference_from_cache
                else reference_csv
            )
            current_label = engineered.current_engineered
            features_applied = True

    if drift_columns is None:
        # mypy cannot narrow `engineered` from `features_applied`, so assert the
        # pairing here rather than reaching through an Optional.
        drift_columns = (
            default_drift_columns(engineered.n_components)  # type: ignore[union-attr]
            if features_applied and engineered is not None
            else ("predicted_label", "positive")
        )

    present = [c for c in drift_columns if c in ref.columns and c in cur.columns]
    skipped = [c for c in drift_columns if c not in present]
    if skipped:
        logger.warning(
            f"Skipping {len(skipped)} drift columns absent from the frames: {skipped}"
        )
    if not present:
        raise ValueError(
            f"none of the requested drift columns {list(drift_columns)} are present "
            f"in both {reference_csv} {list(ref.columns)} and {current_csv} "
            f"{list(cur.columns)}"
        )

    lo, hi = thresholds
    columns, errored, uncomparable = _psi_per_column(present, ref, cur, thresholds)

    severity = {"PASS": 0, "WARN": 1, "FAIL": 2, "ERROR": 3}
    # SKIPPED columns are excluded from the rollup: a column with no reference
    # spread cannot drift, so letting it into the maximum could only ever
    # downgrade a real verdict, never upgrade one. If *nothing* was checkable
    # the rollup is SKIPPED rather than PASS -- a gate that ran no checks must
    # not report itself healthy.
    severity_decisions = [
        columns[c]["decision"] for c in columns if columns[c]["decision"] in severity
    ]
    overall = (
        max(severity_decisions, key=severity.__getitem__)
        if severity_decisions
        else "SKIPPED"
    )
    checked = [c for c in columns.values() if c["drift_score"] is not None]
    verdict: dict[str, Any] = {
        "date": date,
        "thresholds": {"warn": lo, "fail": hi},
        "reference": reference_label,
        "current": current_label,
        "n_reference": len(ref),
        "n_current": len(cur),
        "columns": columns,
        "n_columns_checked": len(checked),
        "skipped_columns": skipped,
        "uncomparable_columns": uncomparable,
        "errored_columns": errored,
        "drift_share": round(
            (sum(1 for c in checked if c["drift_detected"]) / len(checked))
            if checked
            else 0.0,
            4,
        ),
        "overall": overall,
    }
    if features_applied:
        verdict["engineered"] = True
        verdict["reference_from_cache"] = reference_from_cache
        if basis is not None:
            verdict["projection"] = basis.summary()
        verdict["feature_means"] = feature_means(
            cur, (COL_TEXT_LENGTH, COL_CONFIDENCE, COL_OOV_RATE)
        )
        verdict["oov_bucket_mix"] = dialect_mix(cur, COL_OOV_BUCKET)
        if COL_DIALECT_LABEL in cur.columns:
            verdict["dialect_mix"] = dialect_total_variation(ref, cur)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    logger.info(f"Drift check {overall}: {columns}")
    return verdict


def _log_run(
    run_name: str,
    metrics: dict[str, float],
    artifacts: list[tuple[str, str]],
    run_type: str,
) -> None:
    """Idempotently log a batch/drift run to the ``raay_batch`` experiment."""
    import mlflow

    tracking_uri = mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(Experiments.BATCH.value)
    if experiment is None:
        experiment_id = mlflow.create_experiment(Experiments.BATCH.value)
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tag("run_type", run_type)
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        for artifact_path, local_path in artifacts:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.info(f"Logged raay_batch run {run.info.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["make-input", "score", "init-reference", "drift"],
        default="score",
    )
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")
    parser.add_argument("--pool", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--current", default=None, help="drift current-day CSV")
    parser.add_argument("--reference", default=DefaultPaths.SCORING_REFERENCE.value)
    parser.add_argument("--drift-out", default=None)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--onnx-path", default=DefaultPaths.ONNX_INT8_MODEL.value)
    parser.add_argument("--tokenizer-dir", default=DefaultPaths.BASELINE_MODEL.value)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument("--report-out", default=None)
    # Phase 6 step 1: engineered input features (embedding PCs, OOV, dialect).
    parser.add_argument(
        "--no-engineer",
        action="store_true",
        help=(
            "Skip the engineered input features and gate only the model outputs "
            "(predicted_label, positive). Needs no encoder weights."
        ),
    )
    parser.add_argument(
        "--pca-path",
        default=DefaultPaths.DRIFT_PCA_BASIS.value,
        help="Frozen PCA basis fitted by init-reference (never refit per day).",
    )
    parser.add_argument(
        "--reference-engineered",
        default=DefaultPaths.SCORING_REFERENCE_ENGINEERED.value,
        help="Cached engineered reference panel written by init-reference.",
    )
    parser.add_argument(
        "--engineered-current",
        default=None,
        help="Where to write the engineered current panel (default alongside it).",
    )
    parser.add_argument(
        "--n-pcs", type=int, default=10, help="PCA components kept (10-20 is sane)."
    )
    parser.add_argument(
        "--pooling", choices=["mean", "cls"], default="mean", help="Encoder pooling."
    )
    parser.add_argument(
        "--drift-columns",
        default=None,
        help="Comma-separated override of the gated columns.",
    )
    args = parser.parse_args()

    load_environment()
    from datetime import datetime

    day = args.date or datetime.now().astimezone().date().isoformat()
    input_csv = args.input or str(Path(DefaultPaths.SCORING_INPUT.value) / f"{day}.csv")
    output_csv = args.output or str(
        Path(DefaultPaths.SCORING_OUTPUT.value) / f"{day}.csv"
    )
    stats_json = args.report_out or f"reports/batch_score_{day}.json"
    drift_json = args.drift_out or f"reports/drift/{day}.json"
    scorer = Scorer(
        onnx_path=args.onnx_path,
        tokenizer_dir=args.tokenizer_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # Built only for the two modes that actually read it. `make-input` samples a
    # CSV and `score` runs the INT8 graph; neither has any use for the encoder,
    # and loading it costs ~2 GB of resident weights plus a few seconds that
    # the nightly Airflow DAG would pay twice per day.
    needs_features = args.mode in ("init-reference", "drift")
    engineered: EngineeredDriftSpec | None = None
    if needs_features and not args.no_engineer:
        current_csv_for_engineering = args.current or output_csv
        engineered = EngineeredDriftSpec(
            builder=DriftFeatureBuilder.from_config(
                tokenizer_dir=args.tokenizer_dir,
                n_components=args.n_pcs,
                max_length=args.max_length,
                pooling=args.pooling,
                batch_size=args.batch_size,
            ),
            pca_path=args.pca_path,
            reference_engineered=args.reference_engineered,
            current_engineered=args.engineered_current
            or str(
                Path(current_csv_for_engineering).with_name(
                    Path(current_csv_for_engineering).stem + "_engineered.csv"
                )
            ),
        )

    if args.mode == "make-input":
        make_input(args.pool, day, args.samples, input_csv)
        if not args.no_mlflow:
            _log_run(
                f"make-input-{day}",
                {"n_reviews": float(args.samples)},
                [(f"input/{day}", input_csv)],
                "make-input",
            )
        return

    if args.mode == "init-reference":
        frame = init_reference(
            args.pool, args.samples, args.reference, scorer, engineered=engineered
        )
        if not args.no_mlflow:
            metrics = {"n_reference": float(len(frame))}
            artifacts: list[tuple[str, str]] = [("reference", args.reference)]
            if engineered is not None:
                artifacts += [
                    ("reference_engineered", engineered.reference_engineered),
                    ("pca_basis", engineered.pca_path),
                ]
                metrics |= {
                    f"reference_mean_{col}": value
                    for col, value in feature_means(
                        frame, (COL_TEXT_LENGTH, COL_OOV_RATE)
                    ).items()
                }
            _log_run("init-reference", metrics, artifacts, "init-reference")
        return

    if args.mode == "score":
        stats = score_input(input_csv, output_csv, scorer, min_samples=args.min_samples)
        Path(stats_json).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        if not args.no_mlflow:
            _log_run(
                f"score-{day}",
                {
                    "n_reviews": float(stats["n_reviews"]),
                    "elapsed_sec": stats["elapsed_sec"],
                    "mean_predicted_score": float(
                        pd.read_csv(output_csv)["predicted_score"].mean()
                    ),
                },
                [(f"score/{day}", output_csv), ("stats_today", stats_json)],
                "score",
            )
        return

    drift_columns_override = (
        tuple(c.strip() for c in args.drift_columns.split(",") if c.strip())
        if args.drift_columns
        else None
    )
    verdict = drift_check(
        args.reference,
        args.current or output_csv,
        drift_json,
        day,
        drift_columns=drift_columns_override,
        engineered=engineered,
    )
    if not args.no_mlflow:
        # Only columns that actually produced a score: an ERRORed column has
        # drift_score None, and logging a placeholder 0.0 for it would draw a
        # healthy line on a chart for a check that never ran.
        metrics = {
            col: data["drift_score"]
            for col, data in verdict["columns"].items()
            if data["drift_score"] is not None
        }
        metrics["drift_share"] = verdict["drift_share"]
        metrics["n_columns_errored"] = float(len(verdict["errored_columns"]))
        metrics["n_current"] = float(verdict["n_current"])
        # Trendable input-feature means and mixes, so a rising OOV rate is a
        # visible series in the raay_batch experiment rather than something you
        # have to open each JSON to find.
        metrics |= {
            f"mean_{col}": value
            for col, value in verdict.get("feature_means", {}).items()
        }
        metrics |= {
            f"oov_share_{bucket}": share
            for bucket, share in verdict.get("oov_bucket_mix", {}).items()
        }
        metrics |= {
            f"dialect_share_{dialect}": share
            for dialect, share in verdict.get("dialect_mix", {})
            .get("current", {})
            .items()
        }
        _log_run(f"drift-{day}", metrics, [("drift", drift_json)], "drift")


if __name__ == "__main__":
    main()

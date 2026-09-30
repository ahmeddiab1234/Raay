"""Engineered *input* drift features for the nightly monitoring job.

Phase 6 step 1. ``batch_score.drift_check`` already compares the model's own
outputs (``predicted_label``, per-class probabilities) against a frozen
reference. That is **output** drift: it can only tell you the model changed
*after* its inputs changed, and on this corpus the reference and the day are
slices of the same pool, so it reads PASS by construction.

This module adds the input side, which is what actually moves first:

- ``embedding_pc1..embedding_pcN`` -- the review encoded by the **fine-tuned**
  AraBERT encoder, projected onto a PCA basis **frozen on the reference
  panel**. Raw 768-dim PSI is not interpretable: Evidently bins numerics on 30
  reference quantiles, so 768 columns means 768 mostly-empty bin comparisons
  and a report nobody reads. 10-20 dims is the standard reduction and 10 is
  the default here.
- ``oov_rate`` -- fraction of AraBERT subword tokens that fall back to
  ``[UNK]``. This is the *earliest* signal available: ``[UNK]`` is a hard
  boundary, so new slang or transliterated foreign words show up here before
  the encoder has smeared them into their nearest neighbours.
- ``dialect_label`` -- the heuristic MSA/Egyptian/Gulf/Levantine/Maghrebi/
  Arabizi mix, so a seasonal vocabulary shift (Ramadan, a campaign that only
  runs in one region) is legible as a proportion move.
- ``text_length`` and ``confidence_score`` -- the cheap shape signals.

Two deliberate choices worth knowing before changing anything here:

**The projection is frozen, never refit.** A PCA refit per day would rotate the
axes under the comparison and every column would drift for arithmetic reasons.
``fit_basis`` therefore runs once (at ``init-reference`` time) and
``ProjectionBasis`` is persisted with joblib; the nightly job only
``transform``s. The basis also records the encoder dir, ``max_length`` and
pooling it was fitted with, and a mismatch on the latter two **raises** --
comparing day N against a projection that was fitted with different
preprocessing produces numbers that look fine and mean nothing.

**Dialect labels are recomputed from ``text``, never read from the stored
``dialect`` column.** ``Dialects`` is a ``str, Enum``, so
``str(Dialects.ARABIZI) == "Dialects.ARABIZI"``: a frame round-tripped
through CSV carries that prefix (the same quirk that puts
``"Dialects.ARABIZI"`` keys in ``reports/split_metrics.json``). Recomputing
gives clean, canonical labels on both sides of the comparison.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

from raay.data.dialect import detect_dialect_scored
from raay.enums.constants import DataColumns, DefaultPaths, Models

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None  # type: ignore[assignment]

# Engineered column names. Kept as module constants because the drift column
# list, the feature means in the verdict and the tests all have to agree.
COL_TEXT_LENGTH = "text_length"
COL_CONFIDENCE = "confidence_score"
COL_OOV_RATE = "oov_rate"
COL_OOV_BUCKET = "oov_bucket"
COL_DIALECT_LABEL = "dialect_label"
EMBEDDING_PREFIX = "embedding_pc"

# Per-class probability columns ``score_input`` writes; used to derive
# ``confidence_score`` when ``predicted_score`` is absent.
_LABEL_PROB_COLUMNS = ("positive", "negative", "neutral")

# ``embedding_pc1`` .. ``embedding_pcN``.
_PCS = 10
# Files whose presence means "this directory holds real encoder weights".
# ``model.safetensors.index.json`` ends in ``.json`` and is correctly excluded.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
# Tokenizer batch for the OOV pass. The embedding pass is far heavier, so this
# is only about not building a giant python list of ids at once.
_OOV_BATCH = 256
_POOLERS = ("mean", "cls")


def embedding_pc_columns(n_components: int = _PCS) -> tuple[str, ...]:
    """``("embedding_pc1", ..., "embedding_pcN")`` in ascending order."""
    return tuple(f"{EMBEDDING_PREFIX}{i}" for i in range(1, n_components + 1))


def default_drift_columns(n_components: int = _PCS) -> tuple[str, ...]:
    """Every column the engineered drift check gates, in report order.

    ``positive`` stays in the list even though ``confidence_score`` is the row
    max: the max masks a drop in ``positive`` that leaves another class on top,
    so the two are not redundant.
    """
    return (
        "predicted_label",
        "positive",
        COL_CONFIDENCE,
        COL_TEXT_LENGTH,
        COL_OOV_BUCKET,
        COL_OOV_RATE,
        COL_DIALECT_LABEL,
        *embedding_pc_columns(n_components),
    )


# Upper edges of the OOV buckets, in increasing order, with ``>0.10`` last.
_OOV_EDGES = (0.0, 0.02, 0.05, 0.10)
_OOV_LABELS = ("none", "low", "moderate", "high")


# arabert instantiation pulls in pyarabic and is expensive; mirror serve.py and
# cache one processor per model name.
_PREPROCESSORS: dict[str, Any] = {}


def _preprocess(text: str, model_name: str) -> str:
    """Same normalization ``serve.predict_probs`` applies before tokenizing."""
    if ArabertPreprocessor is None:
        return str(text)
    proc = _PREPROCESSORS.get(model_name)
    if proc is None:
        proc = _PREPROCESSORS.setdefault(
            model_name, ArabertPreprocessor(model_name=model_name)
        )
    return proc.preprocess(text)


def encoder_weights_present(model_dir: str) -> bool:
    """Whether ``model_dir`` actually holds fine-tuned encoder weights.

    Only the three JSON files under ``models/baseline/final/`` are DVC-tracked,
    so a fresh clone has a config and a tokenizer but no weights -- and
    ``AutoModel.from_pretrained`` on that directory raises an error about
    missing weights rather than about the thing the operator actually needs to
    know. Checked up front so the failure is actionable.
    """
    directory = Path(model_dir)
    if not directory.is_dir():
        return False
    return any(
        p.is_file() and p.suffix in _WEIGHT_SUFFIXES for p in directory.iterdir()
    )


class AraBertEmbedder:
    """Pooled sentence embeddings from the fine-tuned AraBERT encoder.

    The ONNX graph cannot do this job: it takes ``input_ids``/``attention_mask``
    and emits 3 logits, nothing else. Drift features need the pooled hidden
    state, so this loads the encoder the classification head sits on top of
    (``AutoModel``, not ``BertForSequenceClassification`` -- the ``classifier.*``
    weights come back as unexpected keys and are dropped).

    Pooling defaults to **mean over non-padding tokens**. ``cls`` is available
    and is what the classifier itself reads, but the mean is far less
    sensitive to truncation at ``max_length``, and a review truncated to 128
    subwords would otherwise be represented mostly by its first tokens.
    """

    def __init__(
        self,
        model_dir: str = DefaultPaths.BASELINE_MODEL.value,
        model_name: str = Models.TEACHER.value,
        max_length: int = 128,
        batch_size: int = 32,
        pooling: str = "mean",
    ) -> None:
        if pooling not in _POOLERS:
            raise ValueError(f"pooling must be one of {_POOLERS}, got {pooling!r}")
        self.model_dir = model_dir
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._hidden_size: int | None = None

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        return self._tokenizer

    @property
    def hidden_size(self) -> int:
        self._ensure_loaded()
        assert self._hidden_size is not None
        return self._hidden_size

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not encoder_weights_present(self.model_dir):
            raise FileNotFoundError(
                f"no encoder weights in {self.model_dir} "
                f"(expected a *{', *'.join(_WEIGHT_SUFFIXES)} file). The embedding "
                f"drift features need the fine-tuned AraBERT encoder, not just the "
                f"tokenizer; only the JSON files there are DVC-tracked, so a fresh "
                f"clone cannot compute them. Restore the weights, or point "
                f"--tokenizer-dir at a directory that has them."
            )
        import torch
        from transformers import AutoConfig, AutoModel

        self._torch = torch
        self._model = AutoModel.from_pretrained(self.model_dir, local_files_only=True)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self._hidden_size = int(AutoConfig.from_pretrained(self.model_dir).hidden_size)
        logger.info(
            f"Embedding encoder loaded from {self.model_dir} "
            f"hidden_size={self._hidden_size} pooling={self.pooling} "
            f"max_length={self.max_length}"
        )

    def _pool(self, last_hidden_state: Any, attention_mask: Any) -> np.ndarray:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :].float().cpu().numpy()
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        # A fully padded row cannot happen (the tokenizer always emits at least
        # one real token), but dividing by 0 would poison the panel silently.
        counts = mask.sum(dim=1).clamp(min=1.0)
        return (summed / counts).float().cpu().numpy()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), hidden_size)`` float64 embeddings, preprocessed first."""
        self._ensure_loaded()
        torch = self._torch
        if len(texts) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float64)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [
                    _preprocess(t, self.model_name)
                    for t in texts[i : i + self.batch_size]
                ]
                enc = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                out = self._model(**enc)
                chunks.append(self._pool(out.last_hidden_state, enc["attention_mask"]))
        return np.concatenate(chunks, axis=0).astype(np.float64)


def oov_rates(
    texts: Sequence[str],
    tokenizer: Any,
    model_name: str = Models.TEACHER.value,
    batch_size: int = _OOV_BATCH,
    preprocess: Callable[[str], str] | None = None,
) -> np.ndarray:
    """Per-text fraction of AraBERT subword tokens that are ``[UNK]``.

    Measured on the *preprocessed* text, which is what the model actually
    tokenizes: measuring the raw string would count ``[UNK]``s that
    normalization then removes, so the rate would not describe what the
    encoder sees. ``preprocess`` is injectable so tests can bypass pyarabic,
    which rewrites text in ways that have nothing to do with OOV.

    Deliberately **not** truncated at ``max_length``: the signal is "how much of
    this review is unspellable to the tokenizer", and clipping to the first 128
    subwords would hide novelty that only appears late in a long review. The
    embedding pass does truncate, because the model has to.
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        raise ValueError(
            "tokenizer exposes no unk_token_id, so the OOV rate is undefined; "
            "refusing to report a constant 0.0 as if it were a measurement"
        )
    normalize = preprocess or (lambda t: _preprocess(t, model_name))
    rates = np.zeros(len(texts), dtype=np.float64)
    for i in range(0, len(texts), batch_size):
        chunk = [normalize(t) for t in texts[i : i + batch_size]]
        encoded = tokenizer(chunk, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        for j, ids in enumerate(encoded):
            if len(ids) == 0:
                continue
            rates[i + j] = sum(1 for t in ids if t == unk_id) / len(ids)
    return rates


def oov_bucket(rate: float) -> str:
    """Bucket label for an ``oov_rate``: none / low / moderate / high.

    This exists because **PSI on the raw rate is not safe to gate**. Evidently
    picks its binning from the data: with more than 20 distinct values it hands
    the combined series to ``numpy.histogram_bin_edges(bins="sturges")``, and
    when the values span a tiny range numpy raises ``Too many bins for data
    range``. ``oov_rate`` is precisely the column that hits this -- most real
    reviews sit at 0.0, so the series is a near-spike -- and an exception there
    takes down the whole nightly report.

    The bucket has at most four distinct values, which keeps Evidently on its
    "few unique values" path (no histogram, no range to collapse), and it is
    the form a human actually wants anyway: "6% of reviews are now >10%
    unspellable" rather than a PSI over 30 quantile slices of a rate.
    """
    if rate > _OOV_EDGES[-1]:
        return _OOV_LABELS[-1]
    for edge, label in zip(_OOV_EDGES, _OOV_LABELS, strict=True):
        if rate <= edge:
            return label
    return _OOV_LABELS[-1]  # pragma: no cover - the zip above always returns


def dialect_labels(texts: Sequence[str]) -> list[str]:
    """Canonical (``Dialects.value``) dialect label per text -- see module docs."""
    return [detect_dialect_scored(t).dialect.value for t in texts]


def text_lengths(texts: Sequence[str]) -> np.ndarray:
    """Review length in characters, the cheapest distribution there is."""
    return np.array([len(t) for t in texts], dtype=np.float64)


def confidence_scores(frame: pd.DataFrame) -> np.ndarray:
    """Row confidence: ``predicted_score`` if present, else max class prob.

    Raises rather than filling zeros when neither exists -- a constant 0.0
    column would pass PSI forever and read as "no drift" in every report.
    """
    if "predicted_score" in frame.columns:
        return frame["predicted_score"].to_numpy(dtype=np.float64)
    present = [c for c in _LABEL_PROB_COLUMNS if c in frame.columns]
    if not present:
        raise ValueError(
            f"cannot derive {COL_CONFIDENCE}: frame has neither 'predicted_score' "
            f"nor any of {list(_LABEL_PROB_COLUMNS)} (columns={list(frame.columns)})"
        )
    return frame[present].to_numpy(dtype=np.float64).max(axis=1)


@dataclass(frozen=True)
class ProjectionBasis:
    """A frozen PCA projection fitted on the reference panel's embeddings.

    Persisted with joblib. The metadata is not decoration: ``check_compatible``
    refuses to compare a day against a basis that was fitted with a different
    ``max_length`` or pooling, because that is a numerically valid but
    meaningless comparison, and nothing downstream would flag it.
    """

    pca: PCA
    n_components: int
    requested_components: int
    explained_variance_ratio: list[float]
    encoder_dir: str
    model_name: str
    max_length: int
    pooling: str
    n_reference: int

    def check_compatible(
        self, *, model_dir: str, max_length: int, pooling: str
    ) -> None:
        mismatches = []
        if self.max_length != max_length:
            mismatches.append(f"max_length {self.max_length} != {max_length}")
        if self.pooling != pooling:
            mismatches.append(f"pooling {self.pooling!r} != {pooling!r}")
        if mismatches:
            raise ValueError(
                "the frozen PCA basis at this path was fitted with different "
                f"preprocessing ({'; '.join(mismatches)}), so the projection would "
                "not match the reference. Re-run `--mode init-reference` to refit it."
            )
        if Path(self.encoder_dir) != Path(model_dir):
            # Warned, not raised: the same encoder can legitimately live at two
            # paths (absolute vs repo-relative, a copy on the staging host).
            logger.warning(
                f"PCA basis was fitted on encoder {self.encoder_dir} but this run "
                f"is using {model_dir}; the two must be the same fine-tuned model"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "requested_components": self.requested_components,
            "explained_variance_ratio": [
                round(v, 4) for v in self.explained_variance_ratio
            ],
            "explained_variance_total": round(
                float(sum(self.explained_variance_ratio)), 4
            ),
            "encoder_dir": self.encoder_dir,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "n_reference": self.n_reference,
        }


def fit_basis(
    embeddings: np.ndarray,
    *,
    requested_components: int = _PCS,
    model_dir: str = DefaultPaths.BASELINE_MODEL.value,
    model_name: str = Models.TEACHER.value,
    max_length: int = 128,
    pooling: str = "mean",
) -> ProjectionBasis:
    """Fit the PCA basis on reference embeddings, clamped to what they support.

    ``sklearn`` raises if ``n_components`` exceeds ``min(n_samples,
    n_features)``. A short reference panel (the smoke-test sizes) would
    therefore crash the nightly job on a config problem, so the count is
    clamped and the reduction is recorded in ``requested_components`` -- the
    drift column list is then intersected with the columns that exist rather
    than being asked for values that were never computed.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    n_rows, n_dims = embeddings.shape
    n_components = max(1, min(requested_components, n_rows, n_dims))
    if n_components != requested_components:
        logger.warning(
            f"reference has {n_rows} rows x {n_dims} dims; clamping PCA to "
            f"{n_components} components (asked for {requested_components})"
        )
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(embeddings)
    logger.info(
        f"Fitted PCA basis {n_components} components on {n_rows} reference "
        f"embeddings, explaining {float(pca.explained_variance_ratio_.sum()):.3f} "
        f"of the variance"
    )
    return ProjectionBasis(
        pca=pca,
        n_components=n_components,
        requested_components=requested_components,
        explained_variance_ratio=[float(v) for v in pca.explained_variance_ratio_],
        encoder_dir=model_dir,
        model_name=model_name,
        max_length=max_length,
        pooling=pooling,
        n_reference=n_rows,
    )


def save_basis(basis: ProjectionBasis, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(basis, path)
    logger.info(f"Saved PCA basis to {path}")


def load_basis(path: str) -> ProjectionBasis:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"no frozen PCA basis at {path}. The engineered drift columns cannot be "
            f"compared against a projection that is refit per day (it would rotate "
            f"the axes under the comparison). Run `--mode init-reference` to build it."
        )
    basis = joblib.load(path)
    if not isinstance(basis, ProjectionBasis):
        raise TypeError(f"{path} holds {type(basis).__name__}, not a ProjectionBasis")
    return basis


class DriftFeatureBuilder:
    """Turns a scored reviews frame into the engineered drift feature frame.

    Takes its embedder and tokenizer by injection rather than building them, so
    the unit tests drive the whole pipeline with a tiny fake encoder and never
    touch a real model graph. Use :meth:`from_config` for the real thing.
    """

    def __init__(
        self,
        embedder: Any,
        tokenizer: Any,
        *,
        n_components: int = _PCS,
        model_dir: str = DefaultPaths.BASELINE_MODEL.value,
        model_name: str = Models.TEACHER.value,
        max_length: int = 128,
        preprocess: Callable[[str], str] | None = None,
    ) -> None:
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.n_components = n_components
        self.model_dir = model_dir
        self.model_name = model_name
        self.max_length = max_length
        self.preprocess = preprocess

    @classmethod
    def from_config(
        cls,
        *,
        tokenizer_dir: str = DefaultPaths.BASELINE_MODEL.value,
        model_name: str = Models.TEACHER.value,
        n_components: int = _PCS,
        max_length: int = 128,
        pooling: str = "mean",
        batch_size: int = 32,
    ) -> DriftFeatureBuilder:
        embedder = AraBertEmbedder(
            model_dir=tokenizer_dir,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            pooling=pooling,
        )
        return cls(
            embedder,
            embedder.tokenizer,
            n_components=n_components,
            model_dir=tokenizer_dir,
            model_name=model_name,
            max_length=max_length,
        )

    @property
    def pooling(self) -> str:
        return getattr(self.embedder, "pooling", "mean")

    def embed(self, frame: pd.DataFrame) -> np.ndarray:
        """Embed a frame's ``text`` column (validated against the row count)."""
        if DataColumns.TEXT.value not in frame.columns:
            raise ValueError(
                f"frame must have a {DataColumns.TEXT.value!r} column to embed "
                f"(got {list(frame.columns)})"
            )
        return self.embedder.embed(frame[DataColumns.TEXT.value].astype(str).tolist())

    def fit_basis(self, embeddings: np.ndarray) -> ProjectionBasis:
        return fit_basis(
            embeddings,
            requested_components=self.n_components,
            model_dir=self.model_dir,
            model_name=self.model_name,
            max_length=self.max_length,
            pooling=self.pooling,
        )

    def engineer(
        self,
        frame: pd.DataFrame,
        basis: ProjectionBasis,
        embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Return a copy of ``frame`` plus every engineered drift column.

        ``embeddings`` may be passed when the caller already has them (fitting
        the basis needs the same matrix), which halves the encoder passes.
        """
        out = frame.copy()
        texts = out[DataColumns.TEXT.value].astype(str).tolist()
        if embeddings is None:
            embeddings = self.embedder.embed(texts)
        if embeddings.shape[0] != len(out):
            raise ValueError(
                f"got {embeddings.shape[0]} embeddings for {len(out)} rows; the "
                f"embedding and feature frames must be row-aligned"
            )
        pcs = basis.pca.transform(embeddings)
        if pcs.shape[1] != basis.n_components:
            raise ValueError(
                f"basis projects to {basis.n_components} components but the "
                f"transform returned {pcs.shape[1]}"
            )
        for i in range(pcs.shape[1]):
            out[f"{EMBEDDING_PREFIX}{i + 1}"] = pcs[:, i]
        out[COL_TEXT_LENGTH] = text_lengths(texts)
        out[COL_CONFIDENCE] = confidence_scores(out)
        out[COL_OOV_RATE] = oov_rates(
            texts, self.tokenizer, self.model_name, preprocess=self.preprocess
        )
        out[COL_OOV_BUCKET] = [oov_bucket(v) for v in out[COL_OOV_RATE]]
        out[COL_DIALECT_LABEL] = dialect_labels(texts)
        return out


def dialect_mix(
    frame: pd.DataFrame, column: str = COL_DIALECT_LABEL
) -> dict[str, float]:
    """Label -> share of rows, in every label seen (0.0 for the ones absent)."""
    if column not in frame.columns:
        raise ValueError(f"frame has no {column!r} column")
    counts = frame[column].astype(str).value_counts(normalize=True)
    return {str(k): round(float(v), 6) for k, v in counts.items()}


def dialect_total_variation(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column: str = COL_DIALECT_LABEL,
) -> dict[str, Any]:
    """How far the dialect mix moved, as shares plus total variation distance.

    PSI on the categorical column is the gate; this is the human-readable
    version of the same movement ("5 points of share went from Egyptian to
    Gulf"), which is what you need to decide whether the shift matters.

    Labels present on only one side are counted as a zero share on the other
    rather than being left out of the union, so a dialect that *appears* for
    the first time moves this number -- the whole point of tracking the mix.
    """
    ref_mix = dialect_mix(reference, column)
    cur_mix = dialect_mix(current, column)
    labels = sorted(set(ref_mix) | set(cur_mix))
    union = {
        k: {"reference": ref_mix.get(k, 0.0), "current": cur_mix.get(k, 0.0)}
        for k in labels
    }
    tvd = 0.5 * sum(
        abs(entry["current"] - entry["reference"]) for entry in union.values()
    )
    return {
        "column": column,
        "reference": ref_mix,
        "current": cur_mix,
        "shares": union,
        "total_variation": round(float(tvd), 6),
    }


def feature_means(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    """Mean of the numeric engineered features, for trending in MLflow."""
    means: dict[str, float] = {}
    for col in columns:
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            means[col] = round(float(frame[col].mean()), 6)
    return means

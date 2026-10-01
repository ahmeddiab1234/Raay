"""Customer-service feedback: QA review and train-set merge (Phase 6 step 4).

The capture endpoint (``raay.serving.feedback_service``) writes raw assertions to
``data/feedback/raw/{date}.csv``. This module turns them into training data in
two steps, deliberately separated:

``--mode review``
    Raw rows -> ``data/feedback/reviewed/overrides.csv`` with a ``status`` per
    row. **Not a DVC stage**: it reads a directory a running service appends to,
    which is not a dependency DVC can hash. An operator runs it, checks the
    disputed rows, then ``dvc add`` + ``dvc push`` the result -- the same ritual
    as ``data/raw/Final_Data.csv``.

``--mode merge``
    Reviewed rows -> ``data/processed/train_feedback.csv``. This one *is* a DVC
    stage, and it is the only piece that touches training data.

Why the merge is a train-only sidecar
--------------------------------------

Feedback must not re-enter ``data/interim/normalized.csv``. If it did, ``split``
would re-run, ``data/processed/test.csv`` would change, and
``scripts/promote_model.py:check_frozen_split`` would exit 2 with no report --
taking the whole 14-gate promotion story down with it. So merged rows go to their
own file that ``train.py`` concatenates onto **train only**. The frozen test
split never moves, and ``promote_model.py`` never reads ``train.csv`` anyway.

Why an override is not ground truth
------------------------------------

A support agent's correction is a human opinion formed under time pressure during
a dispute, and disputes are frequently about shipping, refunds or seller conduct
rather than sentiment -- exactly the scope violation
``docs/labeling_guidelines.md`` section 2 warns about. The brief is right that a
mislabeled override is worse than no label, so nothing reaches the training set on
a single assertion:

1. two distinct agents must agree on the same text, or
2. an adjudicator must have ruled (``adjudicated_label`` wins over both).

That mirrors the 2-annotator + adjudicator process in section 4 of the
guidelines, applied to a population where the second annotator is cheap because
the text already exists.

**Operational prerequisite, not a bug:** corroboration needs an agent roster with
at least two real identities. Until one exists, ``--mode review`` legitimately
returns 100% ``single_agent`` and the merge accepts nothing.

What this measures that nothing else can
-----------------------------------------

``model_label`` on a production review is a *free* labelled error record. Every
other Phase 6 signal is a seeded draw from ``data/processed/test.csv`` and says
so in its own ``caveat`` field; this is actual traffic. The
``production_error_rate`` block of ``reports/feedback_metrics.json`` is therefore
the only production-measured error rate in the project -- and it is only
interpretable if confirmations are posted too, since they are the denominator.

Implementation lives in siblings -- ``feedback_schema`` (statuses, routes,
columns, config), ``feedback_text`` (normalization + label coercion),
``feedback_qa`` (the ladder), ``feedback_review`` (``--mode review``),
``feedback_merge`` (``--mode merge``), ``feedback_metrics`` (the report) and
``feedback_cli`` (argument parsing + dispatch) -- re-exported here so
``python -m raay.data.feedback`` and ``raay.data.feedback.<name>`` keep one
import path. **The ``feedback`` stage in ``dvc.yaml`` lists every one of those
files in its ``deps``**; a module missing from that list means editing it does
not invalidate the stage.
"""

from __future__ import annotations

from raay.data.feedback_cli import (
    _read_reviewed_for_report,
    _review_for_report,
    _run_review,
    load_params,
    main,
    parse_args,
)
from raay.data.feedback_merge import (
    build_merged,
    label_proportions,
    merge_reviewed,
    read_merged,
)
from raay.data.feedback_metrics import (
    build_report,
    has_new_feedback,
    log_run,
    production_error_rate,
    write_report,
)
from raay.data.feedback_qa import (
    classify_row,
    corroboration_groups,
    find_test_overlap,
    route_for_confidence,
)
from raay.data.feedback_review import read_raw, review_rows, write_reviewed
from raay.data.feedback_schema import (
    ALL_STATUSES,
    MERGED_COLUMNS,
    RAW_REQUIRED,
    REVIEWED_COLUMNS,
    ROUTE_ADJUDICATE_FIRST,
    ROUTE_METRICS_ONLY,
    ROUTE_TRAIN,
    ROUTES_NEEDS_ADJUDICATOR,
    ROUTES_NEEDS_SECOND_AGENT,
    STATUS_ADJUDICATED,
    STATUS_CONFIRMATION,
    STATUS_CORROBORATED,
    STATUS_DISPUTED,
    STATUS_DUPLICATE,
    STATUS_LEAK,
    STATUS_NEAR_EMPTY,
    STATUS_SINGLE_AGENT,
    TRAIN_BASE_COLUMNS,
    TRAINABLE_STATUSES,
    FeedbackConfig,
    ReviewResult,
)
from raay.data.feedback_text import (
    coerce_label,
    normalize_text,
    numeric_or_none,
    text_key,
)

__all__ = [
    "ALL_STATUSES",
    "MERGED_COLUMNS",
    "REVIEWED_COLUMNS",
    "STATUS_ADJUDICATED",
    "STATUS_CONFIRMATION",
    "STATUS_CORROBORATED",
    "STATUS_DISPUTED",
    "STATUS_DUPLICATE",
    "STATUS_LEAK",
    "STATUS_NEAR_EMPTY",
    "STATUS_SINGLE_AGENT",
    "TRAINABLE_STATUSES",
    "FeedbackConfig",
    "ReviewResult",
    "build_merged",
    "build_report",
    "classify_row",
    "coerce_label",
    "corroboration_groups",
    "find_test_overlap",
    "has_new_feedback",
    "label_proportions",
    "main",
    "merge_reviewed",
    "normalize_text",
    "numeric_or_none",
    "production_error_rate",
    "read_merged",
    "read_raw",
    "review_rows",
    "route_for_confidence",
    "text_key",
    "write_report",
    "write_reviewed",
]

# Underscored names the original module exposed; still importable from here.
_PRIVATE_REEXPORTS = (
    RAW_REQUIRED,
    ROUTE_ADJUDICATE_FIRST,
    ROUTE_METRICS_ONLY,
    ROUTE_TRAIN,
    ROUTES_NEEDS_ADJUDICATOR,
    ROUTES_NEEDS_SECOND_AGENT,
    TRAIN_BASE_COLUMNS,
    _read_reviewed_for_report,
    _review_for_report,
    _run_review,
    log_run,
    load_params,
    parse_args,
)


if __name__ == "__main__":
    raise SystemExit(main())

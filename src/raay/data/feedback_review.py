"""``--mode review``: raw assertions -> ``reviewed/overrides.csv``.

Not a DVC stage: it reads a directory a running service appends to, which is not
a dependency DVC can hash. An operator runs it, checks the disputed rows, then
``dvc add`` + ``dvc push`` the result.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from raay.data.feedback_qa import (
    classify_row,
    corroboration_groups,
    find_test_overlap,
    route_for_confidence,
)
from raay.data.feedback_schema import (
    ALL_STATUSES,
    RAW_REQUIRED,
    REVIEWED_COLUMNS,
    STATUS_DUPLICATE,
    STATUS_LEAK,
    STATUS_NEAR_EMPTY,
    FeedbackConfig,
    ReviewResult,
)
from raay.data.feedback_text import (
    _text_key_only,
    coerce_label,
    normalize_text,
)
from raay.data.preprocess import flag_near_empty
from raay.enums.constants import DefaultPaths


def read_raw(raw_dir: str) -> pd.DataFrame:
    """Concatenate every ``data/feedback/raw/*.csv``.

    An empty directory yields an empty frame rather than an error: before any
    customer service exists there is nothing to review, and that is a normal
    state for the nightly job to be in.
    """
    paths = sorted(Path(raw_dir).glob("*.csv"))
    if not paths:
        return pd.DataFrame(columns=list(REVIEWED_COLUMNS))
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    raw = pd.concat(frames, ignore_index=True)
    missing = [column for column in RAW_REQUIRED if column not in raw.columns]
    if missing:
        raise ValueError(
            f"{raw_dir} is missing required column(s) {missing}; the capture sink "
            f"writes {list(REVIEWED_COLUMNS[:11])}"
        )
    return raw


def review_rows(
    raw: pd.DataFrame,
    config: FeedbackConfig,
    test_csv: str = DefaultPaths.TEST_SPLIT.value,
    dedup_threshold: float = 0.9,
    already_merged: pd.DataFrame | None = None,
    elongation_max_repeat: int = 2,
) -> ReviewResult:
    """Apply the QA ladder; return the reviewed frame and the status counts."""
    if raw.empty:
        return ReviewResult(frame=raw.copy(), counts={})

    frame = raw.copy()
    # Filled by hand downstream, and a CSV round trip must not turn an absent
    # corroborating agent into the string "nan".
    for column in (
        "model_score",
        "second_agent_id",
        "second_label",
        "adjudicator_id",
        "adjudicated_label",
        "override_id",
        "captured_at",
        "model_version",
        "company",
        "note",
        "guideline_version",
    ):
        if column not in frame.columns:
            frame[column] = ""

    # Normalize first: every comparison below then runs on the same footing as
    # the training corpus and the test split (see normalize_text).
    frame["text"] = frame["text"].map(
        lambda t: normalize_text(t, elongation_max_repeat)
    )
    frame["model_label"] = frame["model_label"].map(coerce_label)
    frame["corrected_label"] = frame["corrected_label"].map(coerce_label)
    frame["_key"] = frame["text"].map(lambda t: " ".join(str(t).split()))
    frame["is_correction"] = frame["corrected_label"] != frame["model_label"]
    # Resolved once, on the coerced values, so corroboration cannot be fooled by
    # a label the ladder has already rejected as out-of-vocabulary.
    frame["_label"] = frame["corrected_label"]

    groups = corroboration_groups(frame, config.min_corroborating_agents)
    statuses: list[str] = []
    routes: list[str] = []
    second_agents: list[str] = []
    corroborating: list[str] = []
    adjudicated_labels: list[str] = []
    adjudicators: list[str] = []
    for _, row in frame.iterrows():
        group = groups[str(row["_key"])]
        status, route = classify_row(row, group, config)
        statuses.append(status)
        routes.append(
            route_for_confidence(status, row.get("model_score"), config, route)
        )
        voters = sorted(
            str(a).strip()
            for a in group["voters"]
            if str(a).strip() != str(row["agent_id"]).strip()
        )
        second_agents.append(voters[0] if voters else "")
        # Every agent that defended the agreed label, so the merge can record the
        # full set on the single row it keeps.
        corroborating.append(";".join(sorted(group["voters"])))
        # Propagate the group's ruling onto every row, so the reviewed CSV records
        # the resolution once per assertion and `build_merged` can read it without
        # re-running the ladder.
        adjudicated_labels.append(group["adjudicated_label"])
        adjudicators.append(
            str(row.get("adjudicator_id", "")).strip()
            if group["adjudicated_label"]
            else ""
        )
    frame["status"] = statuses
    frame["route"] = routes
    frame["second_agent_id"] = second_agents
    frame["corroborating_agents"] = corroborating
    frame["adjudicated_label"] = adjudicated_labels
    frame["adjudicator_id"] = adjudicators
    frame["second_label"] = frame["_key"].map(
        lambda key: groups[str(key)]["agreed_label"]
    )

    # Exclusions run *after* the ladder so the counters still show how many
    # assertions were disputed, not merely how many rows survived: an override
    # rejected for overlapping the test split is still a production error.
    #
    # Precedence matters. Each exclusion overwrites the previous one, so the order
    # is the priority ladder: a row that is both a leak and a duplicate must report
    # as a leak (train-on-test is the serious one, and a duplicate is the milder
    # explanation), while a near-empty row still reports as near-empty because
    # that is why it can never be trusted.
    leaked = find_test_overlap(frame, test_csv, dedup_threshold)
    if leaked:
        frame.loc[frame["_key"].isin(leaked), "status"] = STATUS_LEAK

    # Rows merged by an earlier run are duplicates of work already done -- this is
    # what makes merge_reviewed idempotent after a crash.
    duplicate = pd.Series(False, index=frame.index)
    if (
        already_merged is not None
        and not already_merged.empty
        and "text" in already_merged
    ):
        prior = {_text_key_only(t) for t in already_merged["text"]}
        duplicate = frame["_key"].isin(prior)
    frame.loc[duplicate, "status"] = STATUS_DUPLICATE

    frame.loc[
        frame["text"].map(lambda t: flag_near_empty(t, config.min_char_length)),
        "status",
    ] = STATUS_NEAR_EMPTY

    reviewed = frame.drop(columns=["_key", "_label"])
    counts = {
        status: int((reviewed["status"] == status).sum()) for status in ALL_STATUSES
    }
    return ReviewResult(frame=reviewed, counts=counts)


def write_reviewed(result: ReviewResult, out_csv: str) -> pd.DataFrame:
    frame = result.frame.copy()
    for column in REVIEWED_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    ordered = frame[list(REVIEWED_COLUMNS)].copy()
    ordered["is_correction"] = ordered["is_correction"].map(
        lambda v: str(bool(v)).lower() if v not in ("", None) else ""
    )
    # Sorted by a content hash, not by capture time: a row's position must not
    # depend on the order files happened to be read in, because this file is
    # git-tracked and DVC-hashed.
    ordered = ordered.sort_values("override_id", kind="stable").reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(out_csv, index=False)
    return ordered

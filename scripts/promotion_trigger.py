"""Trigger provenance: why does this version exist at all?

Read from the git-tracked Phase 6 step 3 reports and stamped onto the registry
*only* at the Production flip, so a rejected candidate or a ``--dry-run`` never
carries a tag implying a promotion that did not happen.

A clean night's report (``reason: none``) is a non-trigger and emits no tags:
tagging it would assert a drift-motivated promotion that never occurred, which
is worse than no tag. An absent report takes the same path, so an untagged
promotion means exactly one thing.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from promotion_types import _MODEL, Config, RegistryClient


def trigger_report_path(cfg: Config) -> Path | None:
    """Newest Phase 6 step 3 trigger report, or ``None`` when there is none.

    The trigger reports are git-tracked (unlike the per-run data-refresh
    reports), so on a fresh checkout the promoted version can be stamped with
    the reason it was retrained. Absent is normal: a promotion driven by hand,
    or one predating this step, simply carries no trigger tag.
    """
    reports = cfg.report_dir / "retrain_trigger"
    if not reports.is_dir():
        return None
    candidates = sorted(reports.glob("*.json"))
    return candidates[-1] if candidates else None


def trigger_version_tags(cfg: Config) -> dict[str, str]:
    """Registry tags recording *why* a version was retrained.

    Deliberately best-effort and total: an unreadable or half-written trigger
    report yields no tags rather than raising, because losing the audit trail
    is not a reason to block a promotion that already passed 14 gates.
    """
    path = trigger_report_path(cfg)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not payload.get("triggered"):
        return {}
    tags: dict[str, str] = {}
    reason = payload.get("reason")
    if reason:
        tags["trigger_reason"] = str(reason)
    when = payload.get("date")
    if when:
        tags["trigger_date"] = str(when)
    worst = (payload.get("psi") or {}).get("worst") or {}
    if worst.get("drift_score") is not None:
        tags["trigger_psi"] = str(worst["drift_score"])
    if worst.get("column"):
        tags["trigger_psi_column"] = str(worst["column"])
    return tags


def apply_trigger_tags(
    client: RegistryClient, version: str, tags: dict[str, str]
) -> list[str]:
    """Stamp the trigger provenance onto a registered version.

    Returns the keys that stuck. Each tag is attempted independently: a single
    rejection must not abort the rest, and the caller records what landed.
    """
    applied: list[str] = []
    for key, value in tags.items():
        try:
            client.set_model_version_tag(_MODEL, version, key, value)
            applied.append(key)
        except Exception as error:  # noqa: BLE001 - best-effort provenance
            logger.warning(f"Could not set {key} on version {version}: {error}")
    return applied

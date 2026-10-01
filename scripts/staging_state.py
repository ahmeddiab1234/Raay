"""The deploy state file.

It lives at ``/var/lib/raay-staging/deploy-state.json``, deliberately *outside*
the checkout: the CD job scp's the tools to ``/opt/raay-staging/`` on every
deploy, and a state file inside that directory would be destroyed by the copy
that precedes the deploy it is supposed to inform.

``last_known_good`` is the only field that changes what a rollback does, so it is
written atomically and read leniently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from staging_types import HISTORY_LIMIT, _now


def read_state(path: Path) -> dict[str, Any]:
    """Load the deploy state, tolerating a missing or corrupt file.

    A corrupt state file must not block a deploy; it only costs the remembered
    rollback target, which ``Config.label`` can recover from the live container.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Write atomically: a host that dies mid-write must not lose the record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _record(state: dict[str, Any], sha: str, outcome: str) -> None:
    """Append one outcome to the bounded history, timestamped."""
    history = [entry for entry in state.get("history", []) if isinstance(entry, dict)]
    history.append({"sha": sha, "outcome": outcome, "at": _now()})
    state["history"] = history[-HISTORY_LIMIT:]
    state["updated_at"] = _now()

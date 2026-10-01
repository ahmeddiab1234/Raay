"""The rollout's state file and its atomic write.

``reports/canary_state.json`` (git-ignored) is the single source of truth for
where a rollout is and what Production pointed at before it started. Writes go
through a temp file and ``replace`` so a crash mid-write cannot leave a
truncated state that claims a rollback already happened.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from canary_phases import CanaryState


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(content)
    tmp.replace(path)


def _load_state(path: Path) -> CanaryState:
    if not path.exists():
        raise SystemExit(f"No rollout state at {path}. Start with `--mode shadow`.")
    return CanaryState(**json.loads(path.read_text()))


def _save_state(state: CanaryState, path: Path) -> None:
    _write_text(path, json.dumps(asdict(state), indent=2) + "\n")

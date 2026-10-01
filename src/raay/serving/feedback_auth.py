"""Token resolution for the feedback endpoint.

Same shape as ``raay.inference.retrain_trigger.resolve_token``: the secret
travels through a path or the environment, never argv, so it cannot land in
``ps`` output on a shared host.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from raay.enums.constants import DefaultPaths

_TOKEN_ENV = "RAAY_FEEDBACK_TOKEN"
_TOKEN_FILE_ENV = "RAAY_FEEDBACK_TOKEN_FILE"
_ALLOW_ANON_ENV = "RAAY_FEEDBACK_ALLOW_ANON"
_DIR_ENV = "RAAY_FEEDBACK_DIR"


def resolve_token(explicit: str | None = None, env: Any = None) -> str | None:
    """Token from an explicit value, a secret file, or the environment.

    The env var wins over the default file path so a container can point
    anywhere; the default file (``airflow_runtime/secrets/feedback_token``)
    matches the convention ``github_dispatch_token`` already set.
    """
    env = os.environ if env is None else env
    if explicit and explicit.strip():
        return explicit.strip()
    path = env.get(_TOKEN_FILE_ENV, "").strip() or DefaultPaths.FEEDBACK_SECRETS.value
    if Path(path).exists():
        return Path(path).read_text().strip() or None
    return env.get(_TOKEN_ENV, "").strip() or None

"""Load the repo ``.env`` and resolve runtime environment variables.

Entry points should call :func:`load_environment` once before reading env vars
so values from ``.env`` (repo root) are honoured while process-level variables
(Kaggle notebook cells) keep precedence. ``.env`` is git-ignored; see
``.env.example`` for the committed template.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from raay.enums.constants import EnvVar


def load_environment() -> None:
    """Load ``.env`` from the repo root, without overriding real env vars."""
    load_dotenv(verbose=False)


def env_str(name: EnvVar, default: str = "") -> str:
    """Read an environment variable by :class:`EnvVar` name."""
    return os.environ.get(name.value, default)


def mlflow_tracking_uri(default: str = "file:./mlruns") -> str:
    """Resolve the MLflow tracking URI, enabling the file-store workaround.

    MLflow >= 3 refuses ``file:`` tracking URIs unless
    ``MLFLOW_ALLOW_FILE_STORE=true``; set it here when the resolved URI is a
    file store so callers don't have to.
    """
    uri = env_str(EnvVar.MLFLOW_TRACKING_URI, default)
    if uri.startswith("file:"):
        os.environ.setdefault(EnvVar.MLFLOW_ALLOW_FILE_STORE.value, "true")
    return uri

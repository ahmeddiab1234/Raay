"""GitHub ``repository_dispatch`` for a fired retrain, stdlib-only."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

#: GitHub REST endpoint for a repository_dispatch. The token goes in an
#: Authorization header, never argv (same discipline as deploy_staging.py).
_GITHUB_API = "https://api.github.com"

#: Default receiver. This is the *git remote* slug, which is what dispatch
#: needs -- not the container-registry name the CD workflow publishes under.
_DEFAULT_REPOSITORY = "ahmeddiab1234/Raay"


def read_token_file(path: str | Path) -> str:
    """Read the dispatch token from a file.

    Mirrors ``scripts/deploy_staging.py --registry-token-file``: the token
    travels through a path, never argv, so it cannot land in ``ps`` output.
    """
    return Path(path).read_text().strip()


def resolve_token(explicit: str | None = None) -> str | None:
    """Token from ``--token-file``/env, or ``None`` when unprovisioned.

    The trigger is fully functional without one: the decision, the report and
    the MLflow provenance are all produced, and the nightly job still exits 0.
    Only the GitHub dispatch is skipped. That is deliberate -- a missing secret
    must not take the nightly pipeline down or hide a real breach.
    """
    path = explicit or os.environ.get("RAAY_GITHUB_DISPATCH_TOKEN_FILE", "")
    if path:
        token_path = Path(path)
        if not token_path.exists():
            logger.warning(
                f"No dispatch token at {token_path}; GitHub dispatch will be skipped."
            )
            return None
        return read_token_file(token_path)
    inline = os.environ.get("RAAY_GITHUB_DISPATCH_TOKEN", "")
    return inline.strip() or None


def dispatch_retrain(
    payload: dict[str, Any],
    token: str,
    repository: str = _DEFAULT_REPOSITORY,
    event_type: str = "retrain",
    api_url: str = _GITHUB_API,
    timeout: float = 30.0,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """POST the retrain event to GitHub. Never raises; returns a result dict.

    ``urllib.request`` rather than ``requests``: the latter is only a
    transitive dependency here, and the sibling tools (``deploy_staging``,
    ``canary_promote``) are stdlib-only for the same reason. The token travels
    in an Authorization header, never argv.

    A 204 with no body is success. 401/403 (bad or under-scoped token) and 404
    (wrong repo) are reported with the API's own message, so the failure is
    diagnosable from the report rather than a bare "dispatch failed".
    """
    url = f"{api_url.rstrip('/')}/repos/{repository}/dispatches"
    body = json.dumps({"event_type": event_type, "client_payload": payload}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "raay-retrain-trigger",
        },
    )
    open_url = opener or urllib.request.urlopen
    result: dict[str, Any] = {
        "attempted": True,
        "ok": False,
        "status": None,
        "repository": repository,
        "event_type": event_type,
        "error": None,
    }
    try:
        with open_url(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            result["status"] = status
            result["ok"] = status in (200, 202, 204)
            if not result["ok"]:
                result["error"] = f"unexpected status {status}"
    except urllib.error.HTTPError as error:
        result["status"] = int(error.code)
        detail = ""
        try:
            detail = error.read().decode("utf-8", "replace")
        except OSError:
            detail = ""
        result["error"] = f"HTTP {error.code}: {detail[:300]}"
    except urllib.error.URLError as error:
        result["error"] = f"URLError: {error.reason}"
    except OSError as error:
        result["error"] = f"OSError: {error}"
    return result

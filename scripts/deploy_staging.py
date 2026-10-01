"""Deploy a SHA-tagged image to staging and roll back automatically on failure.

This module is the thin CLI façade over ``staging_*`` siblings; the heavy logic
lives there so the test harness can import the pieces it patches. The contract,
design points, and the three smoke fixtures are documented in the original module
and preserved across the split.

Entry point remains ``python scripts/deploy_staging.py`` (the CD job calls it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from staging_deploy import deploy
from staging_smoke import (
    SMOKE_CASES,
    check_one_prediction,
)
from staging_state import read_state, write_state
from staging_types import (
    DEFAULT_BASE_URL,
    DEFAULT_COMPOSE_FILE,
    DEFAULT_HEALTH_TIMEOUT_S,
    DEFAULT_STATE_FILE,
    VERSION_LABEL,
    Config,
    DeployError,
    Report,
)

__all__ = [
    "SMOKE_CASES",
    "VERSION_LABEL",
    "Config",
    "DeployError",
    "Report",
    "check_one_prediction",
    "main",
    "read_state",
    "write_state",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--registry", default="ghcr.io")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--compose-file", default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT_S)
    parser.add_argument(
        "--expect-model-version",
        default=None,
        help="cross-check the reported model_version against this value",
    )
    parser.add_argument("--registry-user", default="github-actions")
    parser.add_argument(
        "--registry-token-file",
        default=None,
        help="read the registry token from this file instead of RAAY_REGISTRY_TOKEN",
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="assume the host is already logged in (local rehearsals)",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("RAAY_REGISTRY_TOKEN", "")
    if args.registry_token_file:
        token = Path(args.registry_token_file).read_text().strip()

    from staging_types import Config

    cfg = Config(
        registry=args.registry,
        repository=args.repository,
        compose_file=args.compose_file,
        state_file=Path(args.state_file),
        base_url=args.base_url,
        health_timeout_s=args.health_timeout,
        expect_model_version=args.expect_model_version,
    )
    try:
        report = deploy(
            cfg,
            args.target_sha,
            registry_user=args.registry_user,
            registry_token=token,
            skip_login=args.skip_login,
        )
    except DeployError as exc:
        print(json.dumps({"outcome": "failed", "errors": [str(exc)]}, indent=2))
        return 1

    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    if report.logs:
        print("\n--- staging container logs (tail) ---", file=sys.stderr)
        print(report.logs, file=sys.stderr)
    if report.ok:
        return 0
    print(
        f"\nDeploy FAILED: staging is now on "
        f"{report.rolled_back_to or 'nothing working'}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

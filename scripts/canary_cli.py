"""argv dispatch for the canary rollout.

Kept separate from the commands so the commands stay callable from a test with a
fake client, which is the only way the transition rules get exercised without a
docker host and a live Prometheus.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from canary_declare import declare
from canary_modes import cmd_advance, cmd_rollback, cmd_shadow
from canary_phases import _PHASES
from canary_registry import _MODEL

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import Experiments

_DEFAULT_COMPOSE_FILE = "docker-compose.canary.yml"
_DEFAULT_CONF_PATH = "deploy/nginx_canary.conf"
_DEFAULT_STATE_PATH = "reports/canary_state.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow-then-canary rollout")
    parser.add_argument(
        "--mode",
        choices=["declare", "shadow", "advance", "rollback"],
        required=True,
    )
    parser.add_argument(
        "--to",
        choices=list(_PHASES[1:]),
        default=None,
        help="target stage for --mode advance",
    )
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--conf-path", default=_DEFAULT_CONF_PATH)
    parser.add_argument("--compose-file", default=_DEFAULT_COMPOSE_FILE)
    parser.add_argument("--state-file", default=_DEFAULT_STATE_PATH)
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--window", default="5m")
    parser.add_argument("--min-requests", type=int, default=500)
    parser.add_argument("--hold-seconds", type=int, default=600)
    parser.add_argument("--agreement-min", type=float, default=0.99)
    parser.add_argument("--error-rate-max", type=float, default=0.005)
    parser.add_argument("--p95-ratio-max", type=float, default=1.10)
    args = parser.parse_args()

    if args.mode == "advance" and not args.to:
        parser.error("--mode advance requires --to canary-5|canary-25|canary-50|full")

    load_environment()
    tracking_uri = (
        args.tracking_uri
        if args.tracking_uri
        else mlflow_tracking_uri(default="file:./mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    conf_path = Path(args.conf_path)
    state_path = Path(args.state_file)

    if args.mode == "declare":
        registered = declare(client, args.experiment)
        print(f"Declared canary: {_MODEL} v{registered.version} (alias 'Canary')")
    elif args.mode == "shadow":
        cmd_shadow(
            client,
            conf_path=conf_path,
            compose_file=args.compose_file,
            state_path=state_path,
            prometheus_url=args.prometheus_url,
        )
    elif args.mode == "advance":
        cmd_advance(
            client,
            to=args.to,
            conf_path=conf_path,
            compose_file=args.compose_file,
            state_path=state_path,
            window=args.window,
            min_requests=args.min_requests,
            hold_seconds=args.hold_seconds,
            agreement_min=args.agreement_min,
            error_rate_max=args.error_rate_max,
            p95_ratio_max=args.p95_ratio_max,
        )
    elif args.mode == "rollback":
        cmd_rollback(
            client,
            conf_path=conf_path,
            compose_file=args.compose_file,
            state_path=state_path,
        )

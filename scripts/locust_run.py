"""Before/after load test: same BentoML service, FP32 ONNX vs INT8 ONNX.

For each variant it starts ``bentoml serve`` with ``RAAY_ONNX_PATH`` pointed at
that graph, waits for readiness, runs a headless Locust sweep against
``/predict``, then tears the server down by pid (never ``pkill -f`` — that
matches our own shell command line). Writes an HTML + CSV locust report per
variant and prints a p50/p95/p99 comparison line:

    uv run python scripts/locust_run.py [--users 8] [--run-time 90] [--port 3031]

Produces (in reports/):
    locust_fp32.html / locust_fp32_stats.csv
    locust_int8.html / locust_int8_stats.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVE_MODULE = "src/raay/serving/serve.py:svc"
_VARIANTS: tuple[tuple[str, str], ...] = (
    ("fp32", "models/onnx/model.onnx"),
    ("int8", "models/onnx/model_int8.onnx"),
)


def _wait_ready(port: int, timeout_s: int = 180) -> None:
    url = f"http://127.0.0.1:{port}/healthz"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    logger.info(f"Server ready on port {port}")
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Server on port {port} did not become ready in {timeout_s}s")


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    logger.info(f"Stopped bentoml (pid {proc.pid})")


def _run_locust(
    port: int, name: str, users: int, spawn_rate: int, run_time: int
) -> None:
    reports = _REPO_ROOT / "reports"
    cmd = [
        "uv",
        "run",
        "locust",
        "-f",
        str(_REPO_ROOT / "scripts" / "locustfile.py"),
        "--host",
        f"http://127.0.0.1:{port}",
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "--run-time",
        f"{run_time}s",
        "--html",
        str(reports / f"locust_{name}.html"),
        "--csv",
        str(reports / f"locust_{name}"),
        "--csv-full-history",
        "--only-summary",
    ]
    logger.info(f"Running locust ({users} users, {run_time}s) for {name}")
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)
    logger.info(f"Locust {name} report -> reports/locust_{name}.html")


def _read_summary_csv(name: str) -> dict[str, Any]:
    path = _REPO_ROOT / "reports" / f"locust_{name}_stats.csv"
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            raw = dict(zip(header, line.strip().split(",")))
            if "POST" in raw.get("Type", "") and "/predict" in raw.get("Name", ""):
                return {
                    "req_per_s": raw.get("Requests/s", "?"),
                    "median_ms": raw.get("Median Response Time", "?"),
                    "avg_ms": raw.get("Average Response Time", "?"),
                    "p95_ms": raw.get("95%", "?"),
                    "p99_ms": raw.get("99%", "?"),
                    "failures": raw.get("Failure Count", "?"),
                }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=8)
    parser.add_argument("--spawn-rate", type=int, default=2)
    parser.add_argument("--run-time", type=int, default=90)
    parser.add_argument("--port", type=int, default=3031)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip testing; just print the summary from existing reports/csv.",
    )
    args = parser.parse_args()

    (Path(_REPO_ROOT / "reports")).mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, Any]] = {}
    for name, onnx_path in _VARIANTS:
        if args.report_only:
            summary[name] = _read_summary_csv(name)
            continue
        log_path = _REPO_ROOT / "reports" / f"serve_{onnx_path.split('/')[-1]}.log"
        with open(log_path, "w") as log_file:
            env = os.environ.copy()
            env["RAAY_ONNX_PATH"] = onnx_path
            env["RAAY_MAX_LENGTH"] = "128"
            proc = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "bentoml",
                    "serve",
                    _SERVE_MODULE,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                ],
                cwd=_REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            logger.info(f"Started bentoml (pid {proc.pid}) with {onnx_path}")
            try:
                _wait_ready(args.port)
                _run_locust(args.port, name, args.users, args.spawn_rate, args.run_time)
                summary[name] = _read_summary_csv(name)
            finally:
                _stop_server(proc)

    print("\nLocust before/after summary (per /predict call):")
    for name, onnx_path in _VARIANTS:
        row = summary.get(name, {})
        print(
            f"{name:6s} ({onnx_path}):  "
            f"req/s={row.get('req_per_s', '?')}  "
            f"median={row.get('median_ms', '?')}ms  "
            f"avg={row.get('avg_ms', '?')}ms  "
            f"95%={row.get('p95_ms', '?')}ms  "
            f"99%={row.get('p99_ms', '?')}ms  "
            f"failures={row.get('failures', '?')}"
        )

    sys.exit(0)


if __name__ == "__main__":
    main()

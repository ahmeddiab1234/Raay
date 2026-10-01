"""The post-deploy smoke test.

Three fixed Arabic reviews with measured score floors, an exact schema check, and
a 422 probe. The last two are the assertions that catch a *stale but healthy*
image: a container that answers ``/health`` with the wrong graph is otherwise
indistinguishable from one serving correctly, and only the predictions and the
pydantic validation tell them apart.

The score floors sit well below the observed scores so quantisation or CPU
differences cannot fail a healthy release, while a wrong label or a swapped
graph still does.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from staging_health import wait_for_health
from staging_types import (
    PREDICTION_KEYS,
    RESPONSE_KEYS,
    VALID_LABELS,
    Config,
    DeployError,
    Report,
)


@dataclass(frozen=True)
class SmokeCase:
    """One fixed review and the label it must produce."""

    text: str
    label: str
    min_score: float


#: Verified against the shipped int8 graph (see the module docstring).
SMOKE_CASES = (
    SmokeCase("المنتج ممتاز وسريع التوصيل", "positive", 0.85),
    SmokeCase("الطلبية وصلت متأخرة جدا وتلفت البضاعة", "negative", 0.85),
    SmokeCase("المنتج بحجم متوسط", "neutral", 0.50),
)


def check_health_body(
    body: dict[str, Any], expected_version: str | None, report: Report
) -> None:
    if body.get("status") != "healthy":
        raise DeployError(
            f"/health reported status={body.get('status')!r}, expected 'healthy'"
        )
    report.checks.append("/health status is healthy")
    reported = body.get("model_version")
    if expected_version is None:
        report.checks.append(f"/health model_version={reported!r} (unverified)")
        return
    if reported != expected_version:
        raise DeployError(
            f"/health reports model_version={reported!r}, but the image is "
            f"labelled {expected_version!r}"
        )
    report.checks.append(
        f"/health model_version matches the image label ({expected_version})"
    )


def check_one_prediction(case: SmokeCase, prediction: Any) -> None:
    if not isinstance(prediction, dict):
        raise DeployError(f"prediction for {case.text!r} is not an object")
    keys = set(prediction)
    if keys != PREDICTION_KEYS:
        raise DeployError(
            f"prediction keys are {sorted(keys)}, expected {sorted(PREDICTION_KEYS)}"
        )
    label = prediction["label"]
    if label not in VALID_LABELS:
        raise DeployError(f"label {label!r} is not one of {list(VALID_LABELS)}")
    if label != case.label:
        raise DeployError(
            f"{case.text!r} was labelled {label!r}, expected {case.label!r}"
        )
    score = prediction["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise DeployError(f"score {score!r} is not a number")
    if not 0.0 <= float(score) <= 1.0:
        raise DeployError(f"score {score!r} is outside [0, 1]")
    if float(score) < case.min_score:
        raise DeployError(
            f"{case.text!r} scored {float(score):.4f}, below the "
            f"{case.min_score:.2f} floor for {case.label!r}"
        )


def check_predict_payload(
    body: Any, expected_version: str | None, report: Report
) -> None:
    if not isinstance(body, dict):
        raise DeployError(
            f"/predict returned {type(body).__name__}, expected an object"
        )
    keys = set(body)
    if keys != RESPONSE_KEYS:
        raise DeployError(
            f"/predict response keys are {sorted(keys)}, expected {sorted(RESPONSE_KEYS)}"
        )
    predictions = body["predictions"]
    if not isinstance(predictions, list):
        raise DeployError("/predict 'predictions' is not a list")
    if len(predictions) != len(SMOKE_CASES):
        raise DeployError(
            f"/predict returned {len(predictions)} predictions, "
            f"expected {len(SMOKE_CASES)}"
        )
    for case, prediction in zip(SMOKE_CASES, predictions, strict=True):
        check_one_prediction(case, prediction)
        report.checks.append(f"{case.label}: {case.text[:24]}...")
    if expected_version is not None and body["model_version"] != expected_version:
        raise DeployError(
            f"/predict reports model_version={body['model_version']!r}, "
            f"expected {expected_version!r}"
        )
    report.checks.append("/predict labels, scores and schema are correct")


def check_validation_rejects_non_strings(cfg: Config, report: Report) -> None:
    """A 422 here means the pydantic wiring survived the image build."""
    status, _ = cfg.http(f"{cfg.base_url}/predict", "POST", {"texts": [1]}, 10)
    if status != 422:
        raise DeployError(f"a non-string review returned HTTP {status}, expected 422")
    report.checks.append("non-string review is rejected with 422")


def run_smoke(
    cfg: Config,
    image: str,
    report: Report,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    expected = cfg.expect_model_version
    check_health_body(
        wait_for_health(cfg, image, report, sleep, monotonic), expected, report
    )
    status, payload = cfg.http(
        f"{cfg.base_url}/predict",
        "POST",
        {"texts": [case.text for case in SMOKE_CASES]},
        60,
    )
    if status != 200:
        raise DeployError(f"/predict returned HTTP {status}, expected 200")
    check_predict_payload(payload, expected, report)
    check_validation_rejects_non_strings(cfg, report)

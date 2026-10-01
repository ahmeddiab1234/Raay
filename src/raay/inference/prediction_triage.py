"""Pairing the input-drift and output-drift verdicts into a triage label."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

#: Verdicts the input-drift half of the triage can hand us.
_TRIAGE_PASSTHROUGH = {"PASS": "PASS", "SKIPPED": "PASS"}


def classify_triage(input_verdict: str | None, output_verdict: str) -> str:
    """Name *which half* moved. The triage answer the brief is asking for.

    - inputs drifted, outputs did not: the world changed and the model coped.
    - inputs held, outputs did not: the model itself moved -- degradation, or
      an inference/serving change rather than a data change.
    - both: genuinely ambiguous, and a human should look.
    - neither: stable.

    A missing input report is ``None``, not a silent PASS: without the input
    half there is no way to attribute anything, so the answer is
    ``indeterminate``.
    """
    if input_verdict is None:
        return "indeterminate"
    if input_verdict == "PASS":
        return "stable" if output_verdict == "PASS" else "model_degraded"
    return "world_changed" if output_verdict == "PASS" else "ambiguous"


def _read_input_verdict(path: str | Path | None) -> tuple[str | None, str | None]:
    """``(verdict, source)`` from the input-drift report, tolerating its absence."""
    if not path:
        return None, None
    report = Path(path)
    if not report.exists():
        logger.warning(
            f"No input-drift report at {report}; triage will be 'indeterminate' "
            "rather than assuming the inputs held"
        )
        return None, None
    try:
        payload = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read input-drift report {report}: {exc}")
        return None, None
    verdict = payload.get("overall")
    if verdict is None:
        logger.warning(
            f"{report} has no 'overall' key; treating input drift as unknown"
        )
        return None, None
    return str(verdict), str(report)

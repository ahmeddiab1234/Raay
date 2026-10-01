"""Should tonight's batch trigger a retrain? (Phase 6 step 3)

Steps 1 and 2 watch the world and the model. Neither of them *acts*: both are
report-only and exit 0 (see ``batch_score --mode drift`` / ``--mode
predict-drift``). This step is the seam they were pointing at -- the nightly
job now asks a question and records the answer, so "the gate fired and nobody
looked" stops being possible.

The decision is deliberately small and boring, because the cost of being wrong
is asymmetric: a needless retrain burns a Kaggle GPU sweep and a human
promotion, while a missed one is caught by the next night's gate.

    manual        an operator asked for it; bypasses detection entirely
    psi_breach    a gated column PSI >= 0.2 in either drift report
    scheduled     a confirmed seasonal event falls in the lead window
    none          nothing fired

Three deliberate non-firers, each of which would otherwise be a false alarm:

- **WARN does not fire.** The brief puts action at 0.2-0.25; WARN (0.1-0.2) is
  a heads-up, not a retrain. A model sitting permanently at ~0.02 (its Neutral
  weakness) must never reach a trigger.
- **SKIPPED and ERROR columns do not fire.** ``oov_rate`` is structurally 0.0
  on this corpus and reports SKIPPED with a reason; treating a null score as a
  breach would retrain nightly over a check that never ran.
- **Unconfirmed calendar dates do not fire.** Ramadan and Eid dates are set by
  moon sighting. A stale or guessed date that fired would be a retrain nobody
  asked for, so an event must be explicitly ``confirmed: true`` to arm.

The one signal deliberately *excluded* from firing is ``escalate`` (falling
confidence AND rising class PSI), which ``prediction_drift`` reports as its own
field. It is stronger evidence than either half alone, but it is a coupled
signal deliberately kept out of ``overall``; letting it trigger on its own would
invent a threshold below the agreed 0.2. It is recorded as evidence and can be
opted into with ``--escalate-fires``.

**This step does not retrain anything.** Fine-tuning needs a GPU and happens in
``scripts/kaggle_train_runs.py`` on Kaggle; ``promote_model.py`` is the only
code allowed to move the Production alias. What happens here is a decision, a
report, and MLflow provenance.

The seasonal trigger is the one part that cannot be validated. Nothing in this
corpus carries a timestamp, so no retrain this job triggers can ever be checked
after the fact for whether firing before Ramadan helped. It is calendar-only and
is documented as untested rather than presented as a feedback loop.

Implementation lives in siblings -- ``retrain_calendar`` (seasonal events),
``retrain_signals`` (the PSI signal), ``retrain_decision`` (precedence),
``retrain_report`` (report + MLflow), ``retrain_dispatch`` (GitHub POST),
``retrain_args``/``retrain_cli`` (the CLI) -- re-exported here so callers keep
the one import path.
"""

from __future__ import annotations

from raay.inference.retrain_calendar import (
    SeasonalEvent,
    evaluate_calendar_trigger,
    load_calendar,
    resolve_event_date,
)
from raay.inference.retrain_cli import main
from raay.inference.retrain_decision import (
    TRIGGER_REASONS,
    RetrainDecision,
    decide,
    evaluate_trigger,
)
from raay.inference.retrain_dispatch import (
    dispatch_retrain,
    read_token_file,
    resolve_token,
)
from raay.inference.retrain_report import build_report, mlflow_metrics, mlflow_tags
from raay.inference.retrain_signals import SignalEvidence, evaluate_psi_trigger

__all__ = [
    "TRIGGER_REASONS",
    "RetrainDecision",
    "SeasonalEvent",
    "SignalEvidence",
    "build_report",
    "decide",
    "dispatch_retrain",
    "evaluate_calendar_trigger",
    "evaluate_psi_trigger",
    "evaluate_trigger",
    "load_calendar",
    "main",
    "mlflow_metrics",
    "mlflow_tags",
    "read_token_file",
    "resolve_event_date",
    "resolve_token",
]


if __name__ == "__main__":
    raise SystemExit(main())

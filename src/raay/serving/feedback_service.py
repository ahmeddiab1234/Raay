"""Customer-service override capture (Phase 6 step 4).

A small internal endpoint where a support agent records that the model got a
review's sentiment wrong during a dispute. It is the closest thing this project
has to production traffic, and therefore the only source of signal that is not a
seeded draw from ``data/processed/test.csv`` -- see the ``caveat`` field on
every drift report, which says exactly that.

Runs as its own process beside the model, the same shape as
``raay.serving.canary_agent`` and for the same reason: a feedback endpoint
inside the BentoML service would drag an auth story, a durable write path and a
second failure mode into the request path that serves ``/predict``.

    python -m uvicorn raay.serving.feedback_service:app --host 127.0.0.1 --port 9101

Endpoints:

* ``POST /feedback`` -- one agent override (see :class:`FeedbackOverride`).
* ``GET /health``   -- liveness.
* ``GET /stats``    -- capture counters.

Two decisions that shape everything downstream
----------------------------------------------

**A confirmation is stored, not rejected.** ``model_label == corrected_label``
returns 200 and writes a row with ``is_correction=False``. It is never training
data -- those rows would only re-teach what the model already does -- but they
are the *denominator* of ``production_error_rate`` in
``reports/feedback_metrics.json``. A tool that only posts disputes produces an
uninterpretable error rate, because the number of times the model was right is
never observed. Making a confirmation as cheap to post as a dispute is what
makes that measurement exist at all.

**Auth is on by default and the endpoint refuses to serve without it.** This is
an unauthenticated write path into a file that ends up in a training set, so it
is a label-poisoning vector, not an internal convenience. The token travels
through a path or the environment, never argv, and is compared with
``hmac.compare_digest``. ``RAAY_FEEDBACK_ALLOW_ANON=1`` is the single escape
hatch and exists for tests.

Durability and concurrency
--------------------------

The sink is an append-only CSV per UTC day under ``data/feedback/raw/``. It is
durable -- a restart loses nothing -- and it is **single-writer**: one uvicorn
worker, guarded by a lock. That is the honest limit of a CSV. Several workers
means a real database; pretending a lock makes a CSV concurrent is how rows get
silently interleaved into unparseable lines.

Implementation lives in siblings -- ``feedback_schema`` (request model + column
list), ``feedback_sink`` (the append-only writer), ``feedback_auth`` (token
resolution) and ``feedback_app`` (routes, counters, the lazy ASGI app) --
re-exported here so ``raay.serving.feedback_service:app`` keeps one import path.
"""

from __future__ import annotations

from raay.serving.feedback_app import FeedbackService, app
from raay.serving.feedback_auth import (
    resolve_token,
)
from raay.serving.feedback_schema import AGENT_ID_RE, RAW_COLUMNS, FeedbackOverride
from raay.serving.feedback_sink import FeedbackSink

__all__ = [
    "AGENT_ID_RE",
    "RAW_COLUMNS",
    "FeedbackOverride",
    "FeedbackService",
    "FeedbackSink",
    "app",
    "resolve_token",
]

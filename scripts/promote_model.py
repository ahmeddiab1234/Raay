"""Gate a candidate model before it is allowed to become Production.

The registry is not a place where a decision should be made by hand, because
the interesting decisions are comparisons: is this model better than the one
that is currently serving, and is it still above the floor we agreed to? Both
are easy to eyeball and get wrong -- a 0.5% macro-F1 regression is invisible
next to an 85% accuracy headline, and accuracy is exactly the number that hides
a model which has learned to ignore the minority classes. The held-out Neutral
class is 5% of the test set, so a model can lose *all* of it and still report a
comfortable accuracy.

The flow is therefore deliberately one-directional (see ``promotion_flow``):

1. register the candidate under the ``Candidate`` alias -- never Production;
2. evaluate it on the frozen, DVC-pinned test split, using the same inference
   path the service uses, on the same machine as the production graph so the
   latency comparison is like-for-like;
3. run every gate, all-or-nothing;
4. only then move the ``Production`` alias, and only if a human approved it.

On any failure the aliases are left exactly as they were and the exit code is
non-zero, so CI fails instead of quietly shipping. The decision, every gate's
observed value and its threshold, and the provenance of each number are written
to ``reports/promotion_<version>.json`` either way.

Local rehearsal without a registry:

    python scripts/promote_model.py --candidate-version 5 --skip-registry

Exit codes: ``0`` gates passed, ``1`` a gate failed, ``2`` the gate could not run
at all. ``.github/workflows/promote.yml`` reads all three.

Implementation lives in siblings -- ``promotion_types`` (config, gate results,
registry protocol), ``promotion_split`` (the frozen-split guard),
``promotion_graph`` (scoring one graph), ``promotion_timing`` (the interleaved
A/B + control measurement), ``promotion_gates`` / ``promotion_gates_host`` (the
fourteen gates), ``promotion_report`` (the report) and
``promotion_trigger`` (why this version exists),
``promotion_flow`` (the flow) and ``promotion_cli`` (argv + exit codes) --
re-exported here so ``python scripts/promote_model.py`` and
``import promote_model`` keep one import path. ``scripts/`` is not a package;
the siblings import each other by bare name because running the file puts its own
directory on ``sys.path``.

Two properties of the latency measurement are pinned **structurally** in
``tests/test_promote_model.py`` against this file and its siblings, because the
fake session makes them statistically unobservable: which list the control
percentile is summarised from, and whether the noise spread is divided by the
fastest or the slowest series. A symmetric loop cannot distinguish either.
"""

from __future__ import annotations

# Re-exported so the timing tests can patch the shared clock through this module.
import time

from promotion_cli import main, parse_args
from promotion_flow import promote
from promotion_gates import (
    evaluate_all_gates,
    gate_accuracy,
    gate_f1_macro,
    gate_f1_macro_floor,
    gate_full_split,
    gate_label_order,
    gate_recall,
)
from promotion_gates_host import gate_latency, gate_parity, gate_size
from promotion_graph import (
    _parity_section,
    evaluate_graph,
    load_graph,
    read_parity,
)
from promotion_report import build_payload, format_table, report_path, write_report
from promotion_split import check_frozen_split, load_split, locked_test_split_md5
from promotion_timing import (
    _noise_pct,
    _one_call,
    _summarize,
    latency_session_options,
    measure_latency,
    measure_latency_pair,
)
from promotion_trigger import (
    apply_trigger_tags,
    trigger_report_path,
    trigger_version_tags,
)
from promotion_types import (
    _ALIAS_CANDIDATE,
    _ALIAS_PRODUCTION,
    _MODEL,
    _TOOL_TAG,
    _WATCHED_CLASSES,
    Config,
    Decision,
    GateResult,
    Graph,
    PromotionError,
    RegistryClient,
    _round,
    file_md5,
)

__all__ = [
    "ALIASES",
    "Config",
    "Decision",
    "GateResult",
    "Graph",
    "PromotionError",
    "RegistryClient",
    "apply_trigger_tags",
    "build_payload",
    "check_frozen_split",
    "evaluate_all_gates",
    "evaluate_graph",
    "file_md5",
    "format_table",
    "gate_accuracy",
    "gate_f1_macro",
    "gate_f1_macro_floor",
    "gate_full_split",
    "gate_label_order",
    "gate_latency",
    "gate_parity",
    "gate_recall",
    "gate_size",
    "latency_session_options",
    "load_graph",
    "load_split",
    "locked_test_split_md5",
    "main",
    "measure_latency",
    "measure_latency_pair",
    "parse_args",
    "promote",
    "read_parity",
    "report_path",
    "time",
    "trigger_report_path",
    "trigger_version_tags",
    "write_report",
]

#: The two alias names the flow is allowed to move, for a reader checking the
#: one-directional guarantee without reading ``promotion_flow``.
ALIASES = (_ALIAS_CANDIDATE, _ALIAS_PRODUCTION)

# Names the original module exposed but this facade does not re-export above.
_PRIVATE_REEXPORTS = (
    _MODEL,
    _TOOL_TAG,
    _WATCHED_CLASSES,
    _noise_pct,
    _one_call,
    _parity_section,
    _round,
    _summarize,
)


if __name__ == "__main__":
    raise SystemExit(main())

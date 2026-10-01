"""The one-directional promotion flow.

1. register the candidate under the ``Candidate`` alias -- never Production;
2. evaluate it on the frozen, DVC-pinned test split, using the same inference
   path the service uses, on the same machine as the production graph so the
   latency comparison is like-for-like;
3. run every gate, all-or-nothing;
4. only then move the ``Production`` alias, and only if a human approved it.

On any failure the aliases are left exactly as they were, so CI fails instead of
quietly shipping.
"""

from __future__ import annotations

import json

# Imported as modules, not as names: `promote()` calls them through the module
# attribute so a test that replaces `promote_model.load_graph` (which *is* this
# same function object) actually reaches the flow. A `from x import y` here
# would copy the reference and leave every such fixture patching a dead name.
import promotion_gates
import promotion_graph
import promotion_report
import promotion_split
import promotion_timing
import promotion_trigger
from promotion_types import (
    _ALIAS_CANDIDATE,
    _ALIAS_PRODUCTION,
    _MODEL,
    Config,
    Decision,
    PromotionError,
    RegistryClient,
)


def promote(
    cfg: Config,
    version: str,
    candidate_onnx: str,
    production_onnx: str,
    client: RegistryClient | None = None,
    dry_run: bool = False,
) -> Decision:
    """Register the candidate, gate it, and promote only if everything passed.

    The ``Production`` alias is the last thing touched, and only on a clean
    sweep, so a failure anywhere above leaves the registry serving what it was
    serving before. ``dry_run`` measures everything and sets ``Candidate`` but
    stops there, which is what the measuring half of the workflow runs.
    """
    previous_production = None
    if client is not None:
        try:
            previous_production = client.get_model_version_by_alias(
                _MODEL, _ALIAS_PRODUCTION
            )
        except Exception:  # noqa: BLE001 - see below
            # No Production alias yet means this is the first promotion, not a
            # reason to abort. There is also nothing to roll back to, which the
            # report records as a null previous_production_version.
            #
            # The catch is deliberately broad: the registry sits behind a
            # Protocol, and each implementation signals "this alias does not
            # exist" its own way (MlflowClient raises MlflowException, a stub
            # may raise anything). Only this one lookup is guarded, so a real
            # error later in the flow still surfaces.
            previous_production = None
        # The candidate is registered under its own alias, never directly as
        # Production, so a version is always inspectable before it can serve.
        client.set_registered_model_alias(_MODEL, _ALIAS_CANDIDATE, version)

    # Verified before anything is scored. The frozen-split gate also appears in
    # the report, but by the time evaluate_all_gates reaches it both graphs have
    # already been scored against a split the gate was about to reject -- twenty
    # minutes of CPU, and on a contended runner enough scheduling noise to
    # produce a latency verdict nobody should act on.
    #
    # Only a *mismatch* short-circuits. A split with no dvc.lock entry at all is
    # a known, tolerated configuration that still produces a truthful report
    # with a failing pin gate, and turning that into an exception would take
    # away the report that explains it.
    frozen = promotion_split.check_frozen_split(cfg)
    if frozen.observed is not None and not frozen.passed:
        raise PromotionError(
            f"the test split does not match {cfg.dvc_lock} "
            f"(observed md5 {frozen.observed}, locked {frozen.threshold}); "
            "re-run the gate against the frozen split"
        )

    frame = promotion_split.load_split(cfg)
    candidate_graph = promotion_graph.load_graph(cfg, candidate_onnx, frame)
    production_graph = promotion_graph.load_graph(cfg, production_onnx, frame)
    promotion_timing.measure_latency_pair(
        candidate_graph, production_graph, frame["text"].tolist()[:64], cfg
    )
    candidate = candidate_graph.metrics
    production = production_graph.metrics
    baseline = json.loads(cfg.floor_report.read_text())

    gates = promotion_gates.evaluate_all_gates(cfg, candidate, production, baseline)
    passed = all(gate.passed for gate in gates)
    promoted = False
    trigger_tags: list[str] = []
    if passed and client is not None and not dry_run:
        client.set_registered_model_alias(_MODEL, _ALIAS_PRODUCTION, version)
        promoted = True
        # Provenance for *why* this version exists, stamped only once the alias
        # has actually moved -- a rejected candidate is not part of the served
        # history, so tagging it would imply a promotion that did not happen.
        trigger_tags = promotion_trigger.apply_trigger_tags(
            client, version, promotion_trigger.trigger_version_tags(cfg)
        )
    elif passed and dry_run:
        # The measuring job must not be able to promote. This is the whole
        # reason --dry-run is a separate flag from --skip-registry: the
        # workflow's gate job needs a real registry connection to resolve and
        # stage the candidate graph, so "no registry" cannot be how it is kept
        # from flipping Production before a human approves.
        print("dry run: Candidate alias set, Production alias not touched")

    payload = promotion_report.build_payload(
        cfg,
        version,
        candidate,
        production,
        baseline,
        gates,
        promoted,
        previous_production,
        dry_run=dry_run,
        trigger_tags=trigger_tags,
    )
    return Decision(promoted=promoted, gates=gates, payload=payload)

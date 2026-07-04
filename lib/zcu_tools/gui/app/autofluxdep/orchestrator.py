"""Workflow orchestrator — the pure requirement resolver (see CONTEXT.md).

Sweeps flux × the user-ordered providers. It is a **requirement resolver**, NOT
an ordering / topological resolver: execution order is whatever sequence it is
handed (the user's GUI list, with the predictor Service prepended), and it just
runs it. Per flux point, for each provider in order it:

1. resolves the provider's declared ``requires`` / ``optional`` / module deps
   against the current info/module state into a read-only ``Snapshot``
   (latest-available, with skip/fallback) — ``resolve_provider_snapshot``;
2. builds the provider's Node for this point via ``builder.build_node(env)``,
   currying in the execution environment (flux, soc, tools, this provider's
   Result, the round_hook) so ``produce`` exposes only "requirements in, Patch
   out";
3. calls ``node.produce(snapshot) -> Patch`` and merges the Patch by
   ``provides`` / ``provides_modules``.

It knows nothing of drawing, tools, acquire, fit, Result, soc, or round_hook —
all that is curried into the Node by its Builder. It does not distinguish a
measurement Node from a Service Node: both go through the same ``build_node`` →
``produce`` path, zero ``isinstance``. A Service simply has no Result/Plotter
(its Builder's factories return None), so nothing is curried for it and the UI
builds no figure — that asymmetry needs no orchestrator branch.

Two carry-over stores per running sweep:

- ``point`` — values produced this flux point (raw Node outputs + derived).
- ``prev``  — snapshot of ``point`` from the previous flux point.

Smoothed values live in their own ``point_smoothed`` / ``prev_smoothed`` stores
so a smoothing consumer reads the smoothed estimate while a plain consumer of
the same key reads the raw value. The first-point baseline and the smoothing
recursion seed are NOT here — they are the predictor service's and the
SmoothingService's own internal state, not dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from zcu_tools.gui.app.autofluxdep.derivation import (
    DerivationService,
    SmoothingService,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import (
    Snapshot,
    validate_patch,
)
from zcu_tools.gui.app.autofluxdep.profiling import PerfStats, elapsed_ms, perf_now
from zcu_tools.gui.app.autofluxdep.tools import Tools

logger = logging.getLogger(__name__)
_PRODUCE_PERF = PerfStats("worker.node_produce", logger, slow_ms=500.0)


class ModuleSource(Protocol):
    """The ml library's read-only module lookup.

    ``get_module(name)`` returns the named preset module, or None if absent —
    the orchestrator then falls back to a provider's declared module default.
    """

    def get_module(self, name: str) -> Any: ...


class DepDeclaring(Protocol):
    """The declaration surface ``resolve_provider_snapshot`` reads off a provider.

    A ``PlacedNode`` satisfies this (it delegates to its Builder). Kept as a
    Protocol so the resolver depends only on the declarations, not on the
    placement type.
    """

    def all_dependencies(self) -> tuple[Any, ...]: ...
    def all_module_deps(self) -> tuple[Any, ...]: ...


# notify(provider_name, flux_idx): the row-updated notification the round_hook
# fires — the main thread redraws that provider's Plotter. Pure data (a name +
# an index), no figure crosses the thread (ADR-0017).
Notify = Callable[[str, int], None]
OnPoint = Callable[[int, float, "InfoStore"], None]
# on_node(provider_name, flux_idx): fired when a provider is about to run (after
# it resolved — a skipped provider does not fire). The UI uses it to auto-follow
# (select the running Node + show its run tab). Pure data, like Notify.
OnNode = Callable[[str, int], None]
OnSkip = Callable[[str, int, "SkipReason"], None]
OnNodeRow = Callable[[str, int, Any, "InfoStore"], None]
OnNodeFailed = Callable[[str, int, Exception, str], None]
OnFluxCommitted = Callable[[int, float, "InfoStore"], None]


@dataclass(frozen=True)
class SkipReason:
    """Structured resolver skip reason for run artifact journal events."""

    missing_info_keys: tuple[str, ...] = ()
    missing_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class SnapshotResolution:
    """Resolver output: either a Snapshot or a machine-readable skip reason."""

    snapshot: Snapshot | None
    skip_reason: SkipReason | None = None


@dataclass
class InfoStore:
    """The orchestrator's master information container for one running sweep.

    Holds the most information; Nodes never touch it directly. The orchestrator
    projects a per-provider ``Snapshot`` out of it before each provider and
    merges the Node's ``Patch`` back in after.

    Two parallel spaces. **Info values**: ``point``/``prev`` (raw + derived) and
    ``point_smoothed``/``prev_smoothed`` (smoothed projections built after the
    Nodes). **Modules**: ``module_point``/``module_prev`` hold Node-produced
    modules (e.g. ro_optimize's tuned readout); a module dep resolves
    module_point → module_prev → ml preset → declared default.
    """

    point: dict[str, Any] = field(default_factory=dict)
    prev: dict[str, Any] = field(default_factory=dict)
    point_smoothed: dict[str, Any] = field(default_factory=dict)
    prev_smoothed: dict[str, Any] = field(default_factory=dict)
    module_point: dict[str, Any] = field(default_factory=dict)
    module_prev: dict[str, Any] = field(default_factory=dict)

    def begin_point(self) -> None:
        """Snapshot the just-finished point into ``prev`` then clear for next."""
        if self.point:
            self.prev = dict(self.point)
            if self.point_smoothed:
                self.prev_smoothed = dict(self.point_smoothed)
            self.module_prev = dict(self.module_point)
        self.point = {}
        self.point_smoothed = {}
        self.module_point = {}

    def record_smoothed(self, smoothed: Mapping[str, Any]) -> None:
        """Project this point's smoothed values with last-good carry-forward."""
        self.point_smoothed = dict(self.prev_smoothed)
        self.point_smoothed.update(smoothed)

    def latest(self, key: str, *, smoothed: bool) -> Any:
        """Latest available value for ``key``, or the sentinel ``_MISSING``.

        Raw: this point if present, else the previous point. Smoothed: ONLY the
        previous point — a smoothed value is necessarily the previous point's
        estimate, because the SmoothingService derives this point's smoothed
        value only AFTER every Node has run (it needs all raw outputs). So while
        Nodes execute, smoothing consumers read the latest trusted carried-over
        estimate.
        """
        if smoothed:
            return self.prev_smoothed.get(key, _MISSING)
        if key in self.point:
            return self.point[key]
        return self.prev.get(key, _MISSING)

    def latest_module(self, name: str) -> Any:
        """Latest Node-produced module for ``name``: this point, else previous.

        Returns ``_MISSING`` if no Node produced it — the caller then falls back
        to the ml preset, then the declared default.
        """
        if name in self.module_point:
            return self.module_point[name]
        return self.module_prev.get(name, _MISSING)


_MISSING = object()


def _resolve_module(
    name: str, aliases: tuple[str, ...], info: InfoStore, ml: ModuleSource | None
) -> Any:
    """Latest-available module: Node-produced (this/prev point), else ml preset.

    Returns ``_MISSING`` if neither a Node nor the ml library provides it.
    """
    produced = info.latest_module(name)
    if produced is not _MISSING:
        return produced
    if ml is not None:
        for module_name in aliases or (name,):
            preset = ml.get_module(module_name)
            if preset is not None:
                return preset
    return _MISSING


def resolve_provider_snapshot(
    provider: DepDeclaring, info: InfoStore, ml: ModuleSource | None = None
) -> SnapshotResolution:
    """Project a provider Snapshot, preserving structured skip reasons."""
    name = getattr(provider, "name", "?")
    deps = provider.all_dependencies()
    resolved: dict[str, Any] = {}
    missing_info_keys: list[str] = []
    for d in deps:
        value = info.latest(d.key, smoothed=d.smooth is not None)
        if value is _MISSING:
            if d.default is not None:
                resolved[d.key] = d.default()
            elif not d.is_optional:
                logger.debug(
                    "skip %r: required info %r unavailable everywhere", name, d.key
                )
                missing_info_keys.append(str(d.key))
        else:
            resolved[d.key] = value

    modules: dict[str, Any] = {}
    missing_modules: list[str] = []
    for m in provider.all_module_deps():
        mod = _resolve_module(m.name, m.aliases, info, ml)
        if mod is _MISSING:
            if m.default is not None:
                modules[m.name] = m.default()
            elif not m.is_optional:
                logger.debug(
                    "skip %r: required module %r unavailable everywhere", name, m.name
                )
                missing_modules.append(str(m.name))
        else:
            modules[m.name] = mod

    if missing_info_keys or missing_modules:
        return SnapshotResolution(
            snapshot=None,
            skip_reason=SkipReason(
                missing_info_keys=tuple(missing_info_keys),
                missing_modules=tuple(missing_modules),
            ),
        )

    return SnapshotResolution(Snapshot(resolved, modules))


@dataclass
class Orchestrator:
    """Sweeps flux × the user-ordered providers — a pure requirement resolver.

    ``providers`` run in the given order (predictor Service prepended, then the
    user's GUI layout) — no topo sort. Each provider is a ``PlacedNode``
    (Builder + params).

    ``tools`` is the sweep-lived stateful service container injected into Nodes
    (predictor, feedback runtime), owned for the whole sweep so state persists.

    ``ml`` is the read-only module library: when a declared module is not
    produced by any Node, the snapshot falls back to ``ml.get_module(name)``.

    ``soc`` is the connected board curried into measurement Nodes. ``results``
    maps a provider's name → its pre-allocated sweep Result (built on the main
    thread at Run start); ``notify`` is the row-updated notification a Node's
    round_hook fires so the main thread redraws. Both are None/empty for a
    headless dry run.

    Smoothing is collected automatically: every dependency that sets ``smooth``
    is gathered across all providers, deduped by key, and turned into one
    SmoothingService run after the Nodes each point. ``derivations`` are any
    extra non-Node producers to run after that.
    """

    providers: list[PlacedNode]
    tools: Tools = field(default_factory=Tools)
    ml: ModuleSource | None = None
    soc: Any = None
    soccfg: Any = None
    md: Any = None
    # The name of the connected device the flux value is applied through (the
    # user's flux-source pick). Curried into each RunEnv so a real-acquire Node
    # writes ``flux`` into ``cfg.dev[flux_device]``. None when no source is picked.
    flux_device: str | None = None
    results: Mapping[str, Any] = field(default_factory=dict)
    notify: Notify | None = None
    derivations: list[DerivationService] = field(default_factory=list)

    def __post_init__(self) -> None:
        # auto-collect consumer-declared smoothing into one service
        specs = [s for p in self.providers for s in p.smooth_specs()]
        self._smoothing: SmoothingService | None = (
            SmoothingService.from_specs(specs) if specs else None
        )
        # The run's cooperative cancel poll, set at the start of ``run`` so
        # ``_make_env`` can curry it into each RunEnv (a real acquire threads it
        # into ``stop_checkers``). None until ``run`` is entered.
        self._should_stop: Callable[[], bool] | None = None
        # The exception a Node's ``produce`` raised mid-sweep, if any. ``run``
        # catches it (a real acquire can Fast-Fail on an unconfigured Node), stops
        # the sweep, and exposes it here so the caller turns it into a terminal
        # RUN_FAILED instead of letting it abort the run worker's QThread.
        self.run_error: Exception | None = None
        self.run_error_node: str | None = None
        self.run_error_flux_idx: int | None = None
        self.run_error_stage: str | None = None

    def _make_env(self, provider: PlacedNode, idx: int, flux: float) -> RunEnv:
        """Curry this point's execution environment for ``provider`` into a RunEnv.

        A provider with no Result (a Service) gets ``result=None`` /
        ``round_hook=None`` — nothing to fill or notify; the Node ignores them.
        """
        result = self.results.get(provider.name)
        round_hook: Callable[[Any], None] | None = None
        if result is not None and self.notify is not None:
            name = provider.name
            notify = self.notify

            def _hook(payload: Any) -> None:
                del payload  # the trace is in the Result already; notify carries idx
                notify(name, idx)

            round_hook = _hook

        return RunEnv(
            flux=flux,
            flux_idx=idx,
            schema=provider.schema,
            soc=self.soc,
            soccfg=self.soccfg,
            ml=self.ml,
            md=self.md,
            tools=self.tools,
            feedback=self.tools.feedback.view_for(provider.name),
            flux_device=self.flux_device,
            result=result,
            round_hook=round_hook,
            should_stop=self._should_stop,
        )

    def run(
        self,
        flux_values: list[float],
        on_point: OnPoint | None = None,
        on_node: OnNode | None = None,
        on_skip: OnSkip | None = None,
        on_node_row: OnNodeRow | None = None,
        on_node_failed: OnNodeFailed | None = None,
        on_flux_committed: OnFluxCommitted | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> InfoStore:
        """Sweep flux × providers in order.

        Per flux point, each provider is projected → built → produced → merged.
        ``on_node`` fires just before a *resolved* provider runs (a skipped one
        does not fire) — the UI auto-follows to that Node's run tab. After all
        providers, the auto-built SmoothingService derives smoothed values into
        ``point_smoothed``, then any extra ``derivations`` run, then ``on_point``
        (the place to react to a finished point).

        ``should_stop`` is polled before each flux point (and before each
        provider) for cooperative cancellation; when it returns True the sweep
        stops and returns the InfoStore as-is.
        """
        logger.info(
            "sweep: %d provider(s) %s over %d flux point(s)",
            len(self.providers),
            [p.name for p in self.providers],
            len(flux_values),
        )
        # Stash for ``_make_env`` to curry into each RunEnv (a real acquire threads
        # it into ``stop_checkers``).
        self._should_stop = should_stop
        self.run_error = None
        self.run_error_node = None
        self.run_error_flux_idx = None
        self.run_error_stage = None
        info = InfoStore()
        for idx, flux in enumerate(flux_values):
            if should_stop is not None and should_stop():
                logger.info("sweep stopped before flux idx %d", idx)
                break
            info.begin_point()
            info.point["flux_value"] = flux
            info.point["flux_idx"] = idx
            logger.debug("flux point %d: value=%g", idx, flux)
            provider_loop_completed = True
            for provider in self.providers:
                if should_stop is not None and should_stop():
                    logger.info("sweep stopped within flux idx %d", idx)
                    provider_loop_completed = False
                    break
                resolution = resolve_provider_snapshot(provider, info, self.ml)
                if resolution.snapshot is None:
                    if on_skip is not None and resolution.skip_reason is not None:
                        on_skip(provider.name, idx, resolution.skip_reason)
                    continue  # skipped this point (a required dep/module missing)
                snapshot = resolution.snapshot
                if on_node is not None:
                    on_node(provider.name, idx)
                node = provider.builder.build_node(self._make_env(provider, idx, flux))
                profile_start = perf_now()
                try:
                    patch = node.produce(snapshot)
                except Exception as exc:  # a real acquire can Fast-Fail (e.g.
                    # unconfigured Node / no flux device). Record it as the run's
                    # terminal error and stop the sweep gracefully — never let it
                    # propagate out of ``run`` and abort the run worker's QThread.
                    logger.exception(
                        "produce failed for %r at flux idx %d", provider.name, idx
                    )
                    _PRODUCE_PERF.record(
                        elapsed_ms(profile_start),
                        detail=f"node={provider.name} idx={idx} failed=1",
                    )
                    self._record_run_error(provider.name, idx, "produce", exc)
                    if on_node_failed is not None:
                        on_node_failed(provider.name, idx, exc, "produce")
                    return info
                _PRODUCE_PERF.record(
                    elapsed_ms(profile_start),
                    detail=f"node={provider.name} idx={idx}",
                )
                if should_stop is not None and should_stop():
                    logger.info(
                        "sweep stopped after %r produced at flux idx %d; "
                        "row not committed",
                        provider.name,
                        idx,
                    )
                    provider_loop_completed = False
                    break
                try:
                    validate_patch(
                        patch, provider.provides, provider.provides_modules
                    )  # provides / provides_modules = the output contract
                except Exception as exc:
                    self._record_run_error(provider.name, idx, "validate_patch", exc)
                    if on_node_failed is not None:
                        on_node_failed(provider.name, idx, exc, "validate_patch")
                    else:
                        raise
                    return info
                try:
                    if on_node_row is not None:
                        on_node_row(provider.name, idx, patch, info)
                except Exception as exc:
                    self._record_run_error(provider.name, idx, "row_write", exc)
                    if on_node_failed is not None:
                        on_node_failed(provider.name, idx, exc, "row_write")
                    else:
                        raise
                    return info
                info.point.update(patch.values())
                info.module_point.update(patch.modules())
                logger.debug(
                    "  %s produced: %s%s",
                    provider.name,
                    patch.values(),
                    f" modules={list(patch.modules())}" if patch.modules() else "",
                )
            if self._smoothing is not None:
                smoothed = self._smoothing.derive(info.point)
                info.record_smoothed(smoothed)
                if smoothed:
                    logger.debug("  smoothed: %s", smoothed)
            for svc in self.derivations:
                info.point.update(svc.derive(info.point))
            if provider_loop_completed and on_flux_committed is not None:
                on_flux_committed(idx, flux, info)
            if on_point is not None:
                on_point(idx, flux, info)
        return info

    def _record_run_error(
        self, provider_name: str, flux_idx: int, stage: str, exc: Exception
    ) -> None:
        self.run_error = exc
        self.run_error_node = provider_name
        self.run_error_flux_idx = flux_idx
        self.run_error_stage = stage

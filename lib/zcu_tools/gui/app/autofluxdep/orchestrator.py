"""Workflow orchestrator — the pure requirement resolver (see CONTEXT.md).

Sweeps flux × the user-ordered providers. It is a **requirement resolver**, NOT
an ordering / topological resolver: execution order is whatever sequence it is
handed (the user's GUI list, with the predictor Service prepended), and it just
runs it. Per flux point, for each provider in order it:

1. projects the provider's declared ``requires`` / ``optional`` / module deps
   against the current info/module state into a read-only ``Snapshot``
   (latest-available, with skip/fallback) — ``project_snapshot``;
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

from dataclasses import dataclass, field

from typing_extensions import Any, Callable, Mapping, Optional, Protocol

from zcu_tools.gui.app.autofluxdep.derivation import (
    DerivationService,
    SmoothingService,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import (
    Snapshot,
    validate_patch,
)
from zcu_tools.gui.app.autofluxdep.tools import Tools


class ModuleSource(Protocol):
    """The ml library's read-only module lookup (a ModuleLibrary in Phase B).

    ``get_module(name)`` returns the named preset module, or None if absent —
    the orchestrator then falls back to a provider's declared module default.
    """

    def get_module(self, name: str) -> Any: ...


class DepDeclaring(Protocol):
    """The declaration surface ``project_snapshot`` reads off a provider.

    A ``PlacedNode`` satisfies this (it delegates to its Builder). Kept as a
    Protocol so the resolver depends only on the declarations, not on the
    placement type.
    """

    def all_dependencies(self) -> tuple[Any, ...]: ...
    def all_module_deps(self) -> tuple[Any, ...]: ...


# notify(provider_name, flux_idx): the row-updated notification the round_hook
# fires — the main thread redraws that provider's Plotter. Pure data (a name +
# an index), no figure crosses the thread (ADR-0018).
Notify = Callable[[str, int], None]
OnPoint = Callable[[int, float, "InfoStore"], None]


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
            self.prev_smoothed = dict(self.point_smoothed)
            self.module_prev = dict(self.module_point)
        self.point = {}
        self.point_smoothed = {}
        self.module_point = {}

    def latest(self, key: str, *, smoothed: bool) -> Any:
        """Latest available value for ``key``, or the sentinel ``_MISSING``.

        Raw: this point if present, else the previous point. Smoothed: ONLY the
        previous point — a smoothed value is necessarily the previous point's
        estimate, because the SmoothingService derives this point's smoothed
        value only AFTER every Node has run (it needs all raw outputs). So while
        Nodes execute, ``point_smoothed`` is still empty; a smoothing consumer
        inherently reads the carried-over estimate.
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


def _resolve_module(name: str, info: InfoStore, ml: Optional[ModuleSource]) -> Any:
    """Latest-available module: Node-produced (this/prev point), else ml preset.

    Returns ``_MISSING`` if neither a Node nor the ml library provides it.
    """
    produced = info.latest_module(name)
    if produced is not _MISSING:
        return produced
    if ml is not None:
        preset = ml.get_module(name)
        if preset is not None:
            return preset
    return _MISSING


def project_snapshot(
    provider: DepDeclaring, info: InfoStore, ml: Optional[ModuleSource] = None
) -> Optional[Snapshot]:
    """Project the master container into a Snapshot for ``provider``, or None to skip.

    Info values: each declared key resolved latest-available (this point, else
    previous; a ``smooth`` dep reads the smoothed projection under the same key),
    defaults filled. Modules: each declared module resolved module_point →
    module_prev → ml preset → declared default. A required info key or module
    that resolves to nothing anywhere (and has no default) → None (skip the
    provider this point).
    """
    deps = provider.all_dependencies()
    resolved: dict[str, Any] = {}
    for d in deps:
        value = info.latest(d.key, smoothed=d.smooth is not None)
        if value is _MISSING:
            if d.default is not None:
                resolved[d.key] = d.default()
            elif not d.is_optional:
                return None  # required, no value anywhere, no default → skip
        else:
            resolved[d.key] = value

    modules: dict[str, Any] = {}
    for m in provider.all_module_deps():
        mod = _resolve_module(m.name, info, ml)
        if mod is _MISSING:
            if m.default is not None:
                modules[m.name] = m.default()
            elif not m.is_optional:
                return None  # required module unavailable everywhere → skip
        else:
            modules[m.name] = mod

    return Snapshot(resolved, modules)


@dataclass
class Orchestrator:
    """Sweeps flux × the user-ordered providers — a pure requirement resolver.

    ``providers`` run in the given order (predictor Service prepended, then the
    user's GUI layout) — no topo sort. Each provider is a ``PlacedNode``
    (Builder + params).

    ``tools`` is the sweep-lived stateful service injected into Nodes (the
    adaptive predictor), owned for the whole sweep so its state persists.

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
    ml: Optional[ModuleSource] = None
    soc: Any = None
    results: Mapping[str, Any] = field(default_factory=dict)
    notify: Optional[Notify] = None
    derivations: list[DerivationService] = field(default_factory=list)

    def __post_init__(self) -> None:
        # auto-collect consumer-declared smoothing into one service
        specs = [s for p in self.providers for s in p.smooth_specs()]
        self._smoothing: Optional[SmoothingService] = (
            SmoothingService.from_specs(specs) if specs else None
        )

    def _make_env(self, provider: PlacedNode, idx: int, flux: float) -> RunEnv:
        """Curry this point's execution environment for ``provider`` into a RunEnv.

        A provider with no Result (a Service) gets ``result=None`` /
        ``round_hook=None`` — nothing to fill or notify; the Node ignores them.
        """
        result = self.results.get(provider.name)
        round_hook: Optional[Callable[[Any], None]] = None
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
            params=dict(provider.params),
            soc=self.soc,
            tools=self.tools,
            result=result,
            round_hook=round_hook,
        )

    def run(
        self,
        flux_values: list[float],
        on_point: Optional[OnPoint] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> InfoStore:
        """Sweep flux × providers in order.

        Per flux point, each provider is projected → built → produced → merged.
        After all providers, the auto-built SmoothingService derives smoothed
        values into ``point_smoothed``, then any extra ``derivations`` run, then
        ``on_point`` (the place to react to a finished point).

        ``should_stop`` is polled before each flux point (and before each
        provider) for cooperative cancellation; when it returns True the sweep
        stops and returns the InfoStore as-is.
        """
        info = InfoStore()
        for idx, flux in enumerate(flux_values):
            if should_stop is not None and should_stop():
                break
            info.begin_point()
            info.point["flux_value"] = flux
            info.point["flux_idx"] = idx
            for provider in self.providers:
                if should_stop is not None and should_stop():
                    break
                snapshot = project_snapshot(provider, info, self.ml)
                if snapshot is None:
                    continue  # skipped this point (a required dep/module missing)
                node = provider.builder.build_node(self._make_env(provider, idx, flux))
                patch = node.produce(snapshot)
                validate_patch(
                    patch, provider.provides, provider.provides_modules
                )  # provides / provides_modules = the output contract
                info.point.update(patch.values())
                info.module_point.update(patch.modules())
            if self._smoothing is not None:
                info.point_smoothed.update(self._smoothing.derive(info.point))
            for svc in self.derivations:
                info.point.update(svc.derive(info.point))
            if on_point is not None:
                on_point(idx, flux, info)
        return info

"""Workflow orchestrator — the runner-free replacement for FluxDepExecutor.

Sweeps flux × the user-ordered Nodes. The orchestrator owns one master
information container (``InfoStore``); per Node it projects a read-only
``Snapshot`` of just that Node's declared keys (``project_snapshot``), runs the
Node, then validates and merges the Node's ``Patch`` back into the master
container. Nodes never touch the master container — they see only their snapshot
in and return a patch out. This skeleton does NOT touch hardware: the per-Node
"run" is injected as a callback (Phase B wires the real acquire + fit).

There is no topological sort and no DAG: Node order is whatever the user laid
out (the GUI list). Dependency resolution is "latest available" — this flux
point's value if present, else the previous point's, else the optional default.
So a Node running before its producer simply reads the previous point's value;
running after, it reads this point's. Both are "latest available", which is why
the consumer needn't say which.

Two carry-over stores per running sweep:

- ``point`` — values produced this flux point (raw Node outputs + derived).
- ``prev``  — snapshot of ``point`` from the previous flux point.

Smoothed values live in their own ``point_smoothed`` / ``prev_smoothed`` stores
so a smoothing consumer reads the smoothed estimate while a plain consumer of
the same key reads the raw value. The first-point baseline (notebook's
``info.first``) and the smoothing recursion seed are NOT here — they are the
pre-step's and the SmoothingService's own internal state, not dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Callable, Optional, Protocol

from zcu_tools.gui.app.autofluxdep.derivation import (
    DerivationService,
    SmoothingService,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import (
    Patch,
    Snapshot,
    validate_patch,
)
from zcu_tools.gui.app.autofluxdep.nodes.spec import NodeInstance
from zcu_tools.gui.app.autofluxdep.tools import Tools


class ModuleSource(Protocol):
    """The ml library's read-only module lookup (a ModuleLibrary in Phase B).

    ``get_module(name)`` returns the named preset module, or None if absent —
    the orchestrator then falls back to a Node's declared module default.
    """

    def get_module(self, name: str) -> Any: ...


@dataclass
class InfoStore:
    """The orchestrator's master information container for one running sweep.

    Holds the most information; Nodes never touch it directly. The orchestrator
    projects a per-Node ``Snapshot`` out of it before each Node and merges the
    Node's ``Patch`` back in after. Replaces ``FluxDepInfoDict``.

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
        inherently reads the carried-over estimate. (This matches the notebook,
        where the smoothing seed is always ``info.last``.)
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
    node: NodeInstance, info: InfoStore, ml: Optional[ModuleSource] = None
) -> Optional[Snapshot]:
    """Project the master container into a Snapshot for ``node``, or None to skip.

    Info values: each declared key resolved latest-available (this point, else
    previous; a ``smooth`` dep reads the smoothed projection under the same key),
    defaults filled. Modules: each declared module resolved module_point →
    module_prev → ml preset → declared default. A required info key or module
    that resolves to nothing anywhere (and has no default) → None (skip the Node
    this point).
    """
    resolved: dict[str, Any] = {}
    for d in node.spec.requires:
        value = info.latest(d.key, smoothed=d.smooth is not None)
        if value is _MISSING:
            if d.default is not None:
                resolved[d.key] = d.default()
            else:
                return None  # required, no value anywhere, no default → skip
        else:
            resolved[d.key] = value
    for d in node.spec.optional:
        value = info.latest(d.key, smoothed=d.smooth is not None)
        if value is not _MISSING:
            resolved[d.key] = value
        elif d.default is not None:
            resolved[d.key] = d.default()

    modules: dict[str, Any] = {}
    for m in node.spec.requires_modules:
        mod = _resolve_module(m.name, info, ml)
        if mod is _MISSING:
            if m.default is not None:
                modules[m.name] = m.default()
            else:
                return None  # required module unavailable everywhere → skip
        else:
            modules[m.name] = mod
    for m in node.spec.optional_modules:
        mod = _resolve_module(m.name, info, ml)
        if mod is not _MISSING:
            modules[m.name] = mod
        elif m.default is not None:
            modules[m.name] = m.default()

    return Snapshot(resolved, modules)


# A Node's per-point execution: given its Snapshot (read-only projection of its
# declared keys) + the sweep-lived tools, it returns a Patch of the keys it
# produced. The orchestrator validates the Patch against ``provides`` and merges
# it into the master container. Phase B replaces the body with build_cfg →
# acquire → fit → post-process (where tools.predictor.update_bias happens); the
# skeleton injects a fake.
RunNode = Callable[[NodeInstance, Snapshot, Tools], Patch]

# The per-point pre-step also gets tools (the notebook's before_each reads
# predictor.predict_matrix_element / predict_freq to seed cur_m / predict_freq).
PrePoint = Callable[[int, float, InfoStore, Tools], None]
OnPoint = Callable[[int, float, InfoStore], None]


@dataclass
class Orchestrator:
    """Sweeps flux × the user-ordered Nodes. Hardware-free in this skeleton.

    ``nodes`` run in the given order (the user's GUI layout) — no topo sort.

    ``tools`` is the sweep-lived stateful service injected into Nodes (the
    adaptive predictor), owned for the whole sweep so its state persists.

    ``ml`` is the read-only module library: when a Node's declared module is not
    produced by any Node, the snapshot falls back to ``ml.get_module(name)``.

    Smoothing is collected automatically: every dependency that sets ``smooth``
    is gathered across all Nodes, deduped by key (conflicts →
    ``SmoothConflictError``), and turned into one SmoothingService run after the
    Nodes each point. Its output goes into ``info.point_smoothed`` so smoothing
    consumers read the smoothed value under the same key. ``derivations`` are
    any *extra* non-Node producers to run after that.
    """

    nodes: list[NodeInstance]
    run_node: RunNode
    tools: Tools = field(default_factory=Tools)
    ml: Optional[ModuleSource] = None
    derivations: list[DerivationService] = field(default_factory=list)

    def __post_init__(self) -> None:
        # auto-collect consumer-declared smoothing into one service
        specs = [s for n in self.nodes for s in n.spec.smooth_specs()]
        self._smoothing: Optional[SmoothingService] = (
            SmoothingService.from_specs(specs) if specs else None
        )

    def run(
        self,
        flux_values: list[float],
        pre_point: Optional[PrePoint] = None,
        on_point: Optional[OnPoint] = None,
    ) -> InfoStore:
        """Sweep flux × Nodes in order.

        ``pre_point`` runs once per flux point BEFORE any Node — the place to
        seed values the Nodes depend on (the notebook's before_each seeding
        predict_freq / cur_m via ``tools.predictor``). After all Nodes, the
        auto-built SmoothingService derives smoothed values into
        ``point_smoothed``, then any extra ``derivations`` run, then
        ``on_point`` (the place to refresh plots).
        """
        info = InfoStore()
        for idx, flux in enumerate(flux_values):
            info.begin_point()
            info.point["flux_value"] = flux
            info.point["flux_idx"] = idx
            if pre_point is not None:
                pre_point(idx, flux, info, self.tools)
            for node in self.nodes:
                snapshot = project_snapshot(node, info, self.ml)
                if snapshot is None:
                    continue  # skipped this point (a required dep/module missing)
                patch = self.run_node(node, snapshot, self.tools)
                validate_patch(
                    patch, node.spec.provides, node.spec.provides_modules
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

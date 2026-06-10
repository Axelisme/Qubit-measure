"""Builder / Node / Provider — the execution abstraction (see CONTEXT.md).

The orchestrator sees only three things on a provider: ``provides``,
``requires``, and (per flux point) a ``Node`` with ``produce``. It is a pure
requirement resolver — it does not know about drawing, tools, acquire, fit, or
Result.

- A **Builder** is the kind of provider, one subclass per experiment, stateless.
  It declares ``provides`` / ``requires`` / ``provides_modules`` / module deps,
  holds the sweep-lived factories (``make_init_result`` / ``make_plotter`` called
  once at Run start), and a per-flux-point factory ``build_node`` that **curries
  the execution environment** (this point's Result / round_hook / soc / tools …)
  into the returned Node.
- A **Node** is what a Builder produces for one flux point — short-lived, holding
  that point's state with the environment closed in. Its only orchestrator-facing
  surface is ``produce(snapshot) -> Patch``.
- **Service** is a Builder whose Node produces by pure computation (no hardware);
  its ``build_node`` curries no soc/Result/round_hook. Same ``produce`` interface
  — the orchestrator does not distinguish it from a measurement Builder.

The execution environment a Builder may curry in is bundled in ``RunEnv`` so
``build_node`` has a stable signature across Builders (a Service simply ignores
the fields it does not use).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep

# round_hook(whole_trace): called each acquire round with the running-averaged
# trace; the Node fills its Result row + the env notifies the main thread.
RoundHook = Callable[[Any], None]


@dataclass
class RunEnv:
    """The per-flux-point execution environment a Builder curries into a Node.

    ``flux`` / ``flux_idx`` — this point. ``params`` — the placed provider's
    user-tuned params. ``soc`` / ``soccfg`` / ``ml`` / ``tools`` — sweep
    resources: the connected board + its QICK config, the active ModuleLibrary
    (the Builder lowers it into the run cfg, Phase B), and the stateful tools.
    ``result`` — the sweep-lived Result this Node fills (its row ``flux_idx``);
    None for pure-compute Nodes. ``round_hook`` — called by acquire each round
    (fill row + notify); None for pure-compute Nodes.
    """

    flux: float
    flux_idx: int
    params: Mapping[str, Any]
    soc: Any = None
    soccfg: Any = None
    ml: Any = None
    md: Any = None
    tools: Any = None
    result: Any = None
    round_hook: RoundHook | None = None


class Node(ABC):
    """One flux point's executable unit, with its environment curried in."""

    @abstractmethod
    def produce(self, snapshot: Snapshot) -> Patch:
        """Resolve → measure/compute → fit → return the Patch of provides.

        The Result (if any) is filled in place; only the Patch is returned. May
        return a partial Patch (omit a provides key when its fit is poor) — the
        downstream then reads the latest-available value.
        """
        ...


class Provider(Protocol):
    """What the orchestrator sees: a name + declared deps + a per-point factory.

    Both a measurement Builder and a Service implement this; the orchestrator
    never touches their other capabilities.
    """

    name: str
    provides: tuple[str, ...]
    provides_modules: tuple[str, ...]

    def all_dependencies(self) -> tuple[Dependency, ...]: ...
    def all_module_deps(self) -> tuple[ModuleDep, ...]: ...
    def smooth_specs(self) -> tuple[tuple[str, Any], ...]: ...
    def build_node(self, env: RunEnv) -> Node: ...


class Builder(ABC):
    """The kind of provider — one subclass per experiment, stateless.

    Subclasses set the declaration class-attrs and implement ``build_node`` (the
    per-point Node factory). Measurement Builders also implement the sweep-lived
    factories ``make_init_result`` / ``make_plotter``; pure-compute Services
    leave them as the no-op defaults.
    """

    name: str = ""
    provides: tuple[str, ...] = ()
    requires: tuple[Dependency, ...] = ()
    optional: tuple[Dependency, ...] = ()
    requires_modules: tuple[ModuleDep, ...] = ()
    optional_modules: tuple[ModuleDep, ...] = ()
    provides_modules: tuple[str, ...] = ()
    base_params: tuple[str, ...] = ()

    # --- declaration helpers (the orchestrator reads these) ---

    def all_dependencies(self) -> tuple[Dependency, ...]:
        return self.requires + self.optional

    def all_module_deps(self) -> tuple[ModuleDep, ...]:
        return self.requires_modules + self.optional_modules

    def smooth_specs(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (d.key, d.smooth) for d in self.all_dependencies() if d.smooth is not None
        )

    # --- sweep-lived factories (Run start; no-op for pure-compute Services) ---

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> Any:
        """Pre-allocate the empty sweep Result. None = no Result.

        ``flux`` is the full (n_flux,) flux axis — known at Run start, so the
        Result fills its flux axis up front (the trailing signal/fit fields stay
        nan until each ``produce`` fills its row). The Plotter needs the complete
        flux axis as its colormap/line x, which is why the whole array is passed
        (not just the length).
        """
        del params, flux  # base is a no-op; measurement Builders override
        return None

    def make_plotter(self, figure: Any) -> Any:
        """Build the sweep-lived Plotter bound to ``figure``. None = no plot."""
        del figure  # base is a no-op; measurement Builders override
        return None

    # --- per-flux-point factory (curries the environment in) ---

    @abstractmethod
    def build_node(self, env: RunEnv) -> Node:
        """Produce the Node for this flux point, closing ``env`` into it."""
        ...


@dataclass
class PlacedNode:
    """A provider placed in a workflow: its Builder + a name + the user's params.

    This is the unit State holds and the GUI edits — distinct from the Builder
    (the stateless kind): the same Builder can be placed twice with different
    params and *different names* (e.g. two ``mist`` placements named ``g_mist`` /
    ``e_mist``). The user tunes ``params`` (the Builder's ``base_params``), the
    list order, and the instance ``name``; ``produce``/Result/Plotter are domain
    code they never edit.

    ``name`` is the **instance identity** — the display label, the key into
    ``run_results`` / the Plotter map, the auto-follow / remove target. It
    defaults to the Builder's type name and is unique within a workflow (the
    controller de-dups). It is distinct from what the placement *provides*: the
    ``provides`` / ``requires`` info keys come from the Builder unchanged (two
    ``mist`` placements both provide ``success`` — info keys are flat, not
    instance-scoped), so renaming changes identity, not the dependency wiring.

    The declaration helpers delegate to the Builder so a PlacedNode satisfies the
    orchestrator's ``Provider`` view directly.
    """

    builder: Builder
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.builder.name

    @property
    def type_name(self) -> str:
        """The Builder's type name (e.g. ``mist``) — the registry key."""
        return self.builder.name

    @property
    def provides(self) -> tuple[str, ...]:
        return self.builder.provides

    @property
    def provides_modules(self) -> tuple[str, ...]:
        return self.builder.provides_modules

    def all_dependencies(self) -> tuple[Dependency, ...]:
        return self.builder.all_dependencies()

    def all_module_deps(self) -> tuple[ModuleDep, ...]:
        return self.builder.all_module_deps()

    def smooth_specs(self) -> tuple[tuple[str, Any], ...]:
        return self.builder.smooth_specs()

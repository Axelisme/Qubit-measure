"""Node dependency model — the front-half skeleton's core abstraction.

This replaces the runner-based ``autofluxdep`` module's ``cfg_maker`` lambda +
``ctx.env["info"]`` walrus chains with an explicit, declarative dependency
model. See ``task_plans/tool_gui/autofluxdep_gui_assessment.md`` §3 for the
analysis this is derived from.

A measurement is a **Node**. Each Node declares:

- ``provides``  — the information keys it writes after a successful run.
- ``requires``  — dependencies that MUST be present, else the orchestrator
  skips the Node for that flux point ((1) "缺則 skip").
- ``optional``  — dependencies that fall back to a default when absent
  ((2) "缺則用 default").

A dependency is just a **key** (the quantity) plus an optional **smooth** flag.
There is deliberately NO time scope: resolution is "give me the latest available
value" — the orchestrator looks this flux point first, then falls back to the
previous point, then to the optional default. The notebook's ``info`` vs
``info.last`` distinction was never a user choice; it only reflected execution
order (a Node running after the producer sees this point's value, one running
before sees the previous point's). Both are "latest available", so the consumer
need not say which.

Two time semantics that *aren't* "latest available" live OUTSIDE the dependency
system, as internal state of whoever owns them:

- the smoothing recursion seed (``prev_smooth``) is the SmoothingService's own
  history, not a dependency (see ``autofluxdep.tools`` / ``derivation``);
- the first-point baseline (the notebook's ``info.first["cur_m"]`` for
  ``m_ratio``) is the per-point pre-step's own state, not a dependency.

When a dependency sets ``smooth``, the consumer reads the *smoothed* value under
the SAME key: it declares ``Dependency("t1", smooth="ewma")`` and reads
``deps["t1"]`` — the resolver projects the smoothed estimate in under the raw
key, so the Node never knows nor cares whether it got raw or smoothed. The
orchestrator collects every smoothing declaration, dedups, and runs one
SmoothingService.

The orchestrator resolves every dependency into a plain ``deps`` dict (defaults
already applied) and hands it to the Node's ``build_cfg`` callback, which does
the (3) "拿到值後任意運算" step — the only part of the old ``cfg_maker`` that
survives, now reduced to pure arithmetic with no walrus / ``.get(k, default)``.
``build_cfg`` is authored per Node type (domain logic); the GUI user only tunes
``base_params`` and wiring, never edits ``build_cfg``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Callable, Literal, Mapping, Optional, Protocol

SmoothMode = Literal["ewma", "step_weighted"]


@dataclass(frozen=True)
class Dependency:
    """A single declared dependency of a Node.

    ``key`` is the quantity to read (the latest available value — this point,
    else the previous point, else the default). ``smooth``, when set, means
    "read the smoothed estimate of ``key`` under the same name": a
    SmoothingService smooths the raw ``key`` with the given mode and the
    resolver projects it in under ``key`` — the Node still reads ``deps[key]``
    and never knows it is smoothed.

    ``default`` only applies to ``optional`` dependencies. It is a zero-arg
    callable rather than a bare value so a fallback can reference external
    constants (the notebook's ``md.qf_w / 0.05`` case) and be evaluated lazily
    — never a mutable shared object captured at declaration time.
    """

    key: str
    smooth: Optional[SmoothMode] = None
    default: Optional[Callable[[], Any]] = None

    @property
    def is_optional(self) -> bool:
        return self.default is not None


@dataclass(frozen=True)
class ModuleDep:
    """A declared module dependency of a Node.

    ``name`` names the module the Node wants (a cfg component such as a readout).
    It is resolved latest-available across producers, then falls back to the ml
    library's same-named preset, then to ``default``:

        Node-produced this point → produced previous point → ml preset → default

    ``default`` is a zero-arg callable returning a module (lazy, like
    ``Dependency.default``). A required module dep (``default is None``) that
    resolves to nothing anywhere skips the Node for that point.
    """

    name: str
    default: Optional[Callable[[], Any]] = None

    @property
    def is_optional(self) -> bool:
        return self.default is not None


# A resolved dependency bundle handed to build_cfg: key -> value, with
# optional-and-absent keys already filled by their default (or omitted if the
# default itself is None-producing — build_cfg decides what that means).
Deps = Mapping[str, Any]


class BuildCfg(Protocol):
    """Per-Node-type config builder — the surviving (3) of the old cfg_maker.

    Receives the ``snapshot`` (read-only projection of the Node's declared info
    values AND modules — ``snapshot[key]`` / ``snapshot.module(name)``, the
    modules already resolved Node-produced-or-ml-preset), the user-tuned
    ``params``, and ``tools`` (the orchestrator-owned predictor). ``build_cfg``
    typically only *reads* tools; the predictor write (update_bias) happens in
    result post-processing. Returns the experiment cfg, or ``None`` to skip this
    Node for this flux point.

    The concrete cfg type is intentionally ``Any`` at the skeleton layer — each
    Node type narrows it. Phase B wires the real ``ExpCfgModel`` subclasses.
    """

    def __call__(
        self, snapshot: Any, params: Mapping[str, Any], tools: Any
    ) -> Optional[Any]: ...


@dataclass(frozen=True)
class NodeSpec:
    """Static declaration of one measurement Node type.

    ``base_params`` lists the user-tunable attribute keys (detune_sweep,
    num_expts, reps, rounds, earlystop_snr, ...) — what the GUI property form
    exposes. The runtime values live on the Node instance / State, not here.
    """

    name: str
    provides: tuple[str, ...]
    requires: tuple[Dependency, ...] = ()
    optional: tuple[Dependency, ...] = ()
    requires_modules: tuple[ModuleDep, ...] = ()
    optional_modules: tuple[ModuleDep, ...] = ()
    provides_modules: tuple[str, ...] = ()
    base_params: tuple[str, ...] = ()
    build_cfg: Optional[BuildCfg] = None

    def all_dependencies(self) -> tuple[Dependency, ...]:
        return self.requires + self.optional

    def all_module_deps(self) -> tuple[ModuleDep, ...]:
        return self.requires_modules + self.optional_modules

    def smooth_specs(self) -> tuple[tuple[str, SmoothMode], ...]:
        """Every (key, mode) this Node's deps want smoothed.

        The orchestrator collects these across all Nodes, dedups by key, and
        builds the SmoothingService. ``key`` is both the raw quantity smoothed
        and the name the smoothed value is projected in under (the Node reads
        ``deps[key]`` either way).
        """
        return tuple(
            (d.key, d.smooth) for d in self.all_dependencies() if d.smooth is not None
        )


@dataclass
class NodeInstance:
    """A Node placed in a workflow: its spec + the user's tuned params.

    Distinct from ``NodeSpec`` (the type) — multiple instances of the same type
    could coexist with different params, and the instance is what State holds
    and the GUI edits.
    """

    spec: NodeSpec
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

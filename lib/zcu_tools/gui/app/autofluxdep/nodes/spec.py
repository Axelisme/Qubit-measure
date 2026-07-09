"""Dependency declarations — the quantities a provider consumes.

This is the *declaration vocabulary* shared by every provider (a measurement
``Builder`` or a ``Service`` — see ``nodes/builder.py``). It replaces the
runner-based ``autofluxdep`` module's ``cfg_maker`` lambda + ``ctx.env["info"]``
walrus chains with an explicit, declarative dependency model. See ``CONTEXT.md``
and ADR-0018 for the active boundary.

A provider declares:

- ``provides``  — the information keys it writes after a successful run.
- ``requires``  — dependencies that MUST be present, else the orchestrator
  skips the provider for that flux point ((1) "缺則 skip").
- ``optional``  — dependencies that fall back to a default when absent
  ((2) "缺則用 default").

A dependency is a **key** (the quantity), an optional **smooth** flag, and a
source policy. The default source policy is ``Need.LATEST``: the orchestrator
looks this flux point first, then falls back to the previous point, then to the
optional default. A provider may instead declare ``Need.NOW`` when stale values
would be physically misleading; then only values produced earlier in the same
flux point are accepted.

Two time semantics that *aren't* "latest available" live OUTSIDE the dependency
system, as internal state of whoever owns them:

- the smoothing recursion seed (``prev_smooth``) is the SmoothingService's own
  history, not a dependency (see ``autofluxdep.tools`` / ``derivation``);
- the first-point baseline (the notebook's ``info.first["cur_m"]`` for
  ``m_ratio``) is the predictor service's own state, not a dependency.

When a dependency sets ``smooth``, the consumer reads the *smoothed* value under
the SAME key: it declares ``Dependency("t1", smooth="ewma")`` and reads
``snapshot["t1"]`` — the resolver projects the smoothed estimate in under the
raw key, so the provider never knows nor cares whether it got raw or smoothed.
The orchestrator collects every smoothing declaration, dedups, and runs one
SmoothingService.

How a resolved snapshot becomes a cfg (the (3) "拿到值後任意運算" step) is now
the ``Node.produce`` body's job (``nodes/builder.py``) — pure arithmetic with no
walrus / ``.get(k, default)``. ``produce`` is authored per experiment (domain
logic); the GUI user only tunes the provider's params and wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

SmoothMode = Literal["ewma", "step_weighted"]


class Need(str, Enum):
    """Which produced value/module generation a dependency may read."""

    LATEST = "latest"
    NOW = "now"


class ModuleFallback(str, Enum):
    """Fallback policy after node-produced modules are unavailable."""

    LIBRARY = "library"
    NONE = "none"


@dataclass(frozen=True)
class Dependency:
    """A single declared dependency of a provider.

    ``key`` is the quantity to read. ``need`` controls whether the resolver may
    use the latest available value or only a current-point value. ``smooth``,
    when set, means
    "read the smoothed estimate of ``key`` under the same name": a
    SmoothingService smooths the raw ``key`` with the given mode and the
    resolver projects it in under ``key`` — the provider still reads
    ``snapshot[key]`` and never knows it is smoothed.

    ``default`` only applies to ``optional`` dependencies. It is a zero-arg
    callable rather than a bare value so a fallback can reference external
    constants (the notebook's ``md.qf_w / 0.05`` case) and be evaluated lazily
    — never a mutable shared object captured at declaration time.
    """

    key: str
    smooth: SmoothMode | None = None
    default: Callable[[], Any] | None = None
    need: Need = Need.LATEST

    def __post_init__(self) -> None:
        object.__setattr__(self, "need", Need(self.need))
        if self.smooth is not None and self.need is Need.NOW:
            raise ValueError("smoothed dependencies cannot require need=NOW")

    @property
    def is_optional(self) -> bool:
        return self.default is not None


@dataclass(frozen=True)
class ModuleDep:
    """A declared module dependency of a provider.

    ``name`` names the module the provider wants (a cfg component such as a
    readout). It is resolved from node-produced modules according to ``need``.
    When ``fallback`` allows it, resolution then falls back to the ml library's
    same-named preset or declared aliases, then to ``default``:

        Node-produced this point → maybe produced previous point → maybe ml preset/alias → default

    ``default`` is a zero-arg callable returning a module (lazy, like
    ``Dependency.default``). A required module dep (``default is None``) that
    resolves to nothing anywhere skips the provider for that point.

    ``aliases`` only applies to the library fallback and, when provided, is the
    full preferred lookup order. Produced modules remain keyed by ``name`` so
    the workflow's in-memory data contract stays stable even when persisted
    ModuleLibrary entries use app-specific calibrated names.
    """

    name: str
    default: Callable[[], Any] | None = None
    aliases: tuple[str, ...] = ()
    need: Need = Need.LATEST
    fallback: ModuleFallback = ModuleFallback.LIBRARY

    def __post_init__(self) -> None:
        object.__setattr__(self, "need", Need(self.need))
        object.__setattr__(self, "fallback", ModuleFallback(self.fallback))

    @property
    def is_optional(self) -> bool:
        return self.default is not None

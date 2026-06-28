"""Dependency declarations — the quantities a provider consumes.

This is the *declaration vocabulary* shared by every provider (a measurement
``Builder`` or a ``Service`` — see ``nodes/builder.py``). It replaces the
runner-based ``autofluxdep`` module's ``cfg_maker`` lambda + ``ctx.env["info"]``
walrus chains with an explicit, declarative dependency model. See
``.agent_state/plans/tool_gui/autofluxdep_gui_assessment.md`` §3 for the analysis.

A provider declares:

- ``provides``  — the information keys it writes after a successful run.
- ``requires``  — dependencies that MUST be present, else the orchestrator
  skips the provider for that flux point ((1) "缺則 skip").
- ``optional``  — dependencies that fall back to a default when absent
  ((2) "缺則用 default").

A dependency is just a **key** (the quantity) plus an optional **smooth** flag.
There is deliberately NO time scope: resolution is "give me the latest available
value" — the orchestrator looks this flux point first, then falls back to the
previous point, then to the optional default. The notebook's ``info`` vs
``info.last`` distinction was never a user choice; it only reflected execution
order (a provider running after the producer sees this point's value, one
running before sees the previous point's). Both are "latest available", so the
consumer need not say which.

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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

SmoothMode = Literal["ewma", "step_weighted"]


@dataclass(frozen=True)
class Dependency:
    """A single declared dependency of a provider.

    ``key`` is the quantity to read (the latest available value — this point,
    else the previous point, else the default). ``smooth``, when set, means
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

    @property
    def is_optional(self) -> bool:
        return self.default is not None


@dataclass(frozen=True)
class ModuleDep:
    """A declared module dependency of a provider.

    ``name`` names the module the provider wants (a cfg component such as a
    readout). It is resolved latest-available across producers, then falls back
    to the ml library's same-named preset or declared aliases, then to
    ``default``:

        Node-produced this point → produced previous point → ml preset/alias → default

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

    @property
    def is_optional(self) -> bool:
        return self.default is not None


# A resolved dependency bundle: key -> value, with optional-and-absent keys
# already filled by their default. Handed to a Node as part of its Snapshot.
Deps = Mapping[str, Any]

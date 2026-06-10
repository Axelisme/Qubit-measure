"""Derivation services — non-Node information producers.

A Node honestly reports what it *measured* this flux point (raw t1, raw kappa).
Anything *derived* across points — smoothing, ratios, running estimates — is the
job of a **DerivationService**, the symmetric counterpart of a Node: a Node
produces keys from hardware, a DerivationService produces keys from other
producers' output.

Smoothing is consumer-driven: a Node that wants the smoothed estimate of a
quantity declares it on the dependency itself — ``Dependency("t1",
smooth="ewma")``. The orchestrator collects every such declaration across all
Nodes, dedups by key (``SmoothingService.from_specs``), and builds one
SmoothingService. Two Nodes asking for the same key with conflicting modes is a
``SmoothConflictError``. The smoothed value is keyed under the SAME name as the
raw quantity (``t1``), but stored in a separate smoothed space so a plain
consumer of ``t1`` still reads the raw value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Iterable, Mapping, Protocol

from zcu_tools.gui.app.autofluxdep.nodes.spec import SmoothMode
from zcu_tools.gui.app.autofluxdep.tools import Smoother


class SmoothConflictError(ValueError):
    """Two consumers declared smoothing for the same key with different modes."""


@dataclass(frozen=True)
class SmoothRule:
    """Smooth the raw quantity ``key`` with ``mode``; emit under the same key."""

    key: str
    mode: SmoothMode


class DerivationService(Protocol):
    """A non-Node producer. Given this point's raw info, returns the keys it
    derives. Implementations hold their own cross-point state."""

    def provides(self) -> tuple[str, ...]: ...
    def derive(self, point: Mapping[str, Any]) -> dict[str, Any]: ...


@dataclass
class SmoothingService:
    """Derives smoothed quantities from raw Node outputs, across flux points.

    Built from the consumer-declared (key, mode) pairs, deduped by key. A rule
    fires only when its key is present this point (the Node ran and succeeded);
    a skipped/failed Node leaves the raw absent, so the smoother is not advanced
    for that point. Owns a single ``Smoother`` whose per-key history is the
    sweep-lived state.
    """

    rules: tuple[SmoothRule, ...]
    smoother: Smoother = field(default_factory=Smoother)

    @classmethod
    def from_specs(cls, specs: Iterable[tuple[str, SmoothMode]]) -> "SmoothingService":
        """Build from (key, mode) pairs, deduping by key.

        Same key+mode declared twice collapses to one rule; same key with a
        different mode raises ``SmoothConflictError``.
        """
        by_key: dict[str, SmoothRule] = {}
        for key, mode in specs:
            rule = SmoothRule(key=key, mode=mode)
            existing = by_key.get(key)
            if existing is not None and existing != rule:
                raise SmoothConflictError(
                    f"Conflicting smoothing for '{key}': {existing} vs {rule}"
                )
            by_key[key] = rule
        return cls(rules=tuple(by_key.values()))

    def provides(self) -> tuple[str, ...]:
        return tuple(r.key for r in self.rules)

    def derive(self, point: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        idx = int(point.get("flux_idx", 0))
        for r in self.rules:
            if r.key not in point:
                continue  # Node didn't produce it this point (skip/fail)
            out[r.key] = self.smoother.update(r.key, idx, point[r.key], r.mode)
        return out

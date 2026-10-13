"""Factory contracts injected by measure adapter authors; no role defaults."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.cfg import CfgSectionSpec, ReferenceValue

# A blank factory always produces a value (never None); a ref factory's optional
# path may return None (the disabled-optional ref).
BlankFactory = Callable[["SessionEnv"], "ReferenceValue"]
RefFactory = Callable[..., "ReferenceValue | None"]


@dataclass(frozen=True)
class RoleFactorySpec:
    """Injected factories for one authored role, with no role vocabulary/defaults.

    kind is module or waveform, and shape returns an isolated context-free spec.
    blank(SessionEnv) returns a fresh ReferenceValue without library lookup.
    ref is None for inline-only roles; otherwise ref(SessionEnv, optional=bool)
    returns an adopted/fallback reference or None for a disabled optional ref.
    Factory failures propagate through definition instantiation. The builder
    validates kind/shape before resolving any value tree."""

    kind: Literal["module", "waveform"]
    shape: Callable[[], CfgSectionSpec]
    blank: BlankFactory
    ref: RefFactory | None = None

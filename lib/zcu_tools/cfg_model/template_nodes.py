"""Frontend-neutral authored nodes shared by templates and measure forms.

These declarations contain no source resolution or GUI behavior. TemplateSeed
is the closed seed vocabulary; consumers validate paths against their schemas.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

type TemplateLiteral = str | int | float | bool | complex | None


@dataclass(frozen=True, slots=True)
class Expr:
    """Authored expression text; unlike a plain string, consumers evaluate it.

    ``text`` is the expression body, without a wire wrapper. Template expansion
    substitutes component placeholders but does not parse or evaluate the body.
    """

    text: str


@dataclass(frozen=True, slots=True)
class Template:
    """Request a fresh inline module from an injected template catalog.

    ``id`` names a TemplateDef in that catalog. ``set`` maps dotted paths relative
    to the module root to initial literal, Expr or Slot seeds. ``lock`` maps
    relative scalar paths to fixed literals; forms reapply locks on branch
    switches, but apply set only at initialization. ``qubit`` and ``resonator``
    bind component placeholders to authored paths, defaulting to primary aliases.

    Construction detaches the mappings. Consumers validate the ID, module family,
    path compatibility and placeholder syntax; this node does not read sources.
    """

    id: str
    set: Mapping[str, TemplateSeed] = field(default_factory=dict)
    lock: Mapping[str, TemplateLiteral] = field(default_factory=dict)
    qubit: str = "$qubit"
    resonator: str = "$resonator"

    def __post_init__(self) -> None:
        object.__setattr__(self, "set", MappingProxyType(dict(self.set)))
        object.__setattr__(self, "lock", MappingProxyType(dict(self.lock)))


@dataclass(frozen=True, slots=True)
class Slot:
    """Request a library module slot/root, such as ``$qubit.module.readout``.

    ``path`` is an authored slot or library root, optionally containing component
    placeholders. Forms resolve it at Reset to a direct key. ``copy=False`` keeps
    a reference; ``copy=True`` copies the module inline. ``set`` maps relative
    dotted paths to initial literal, Expr or Slot seeds; ``lock`` maps relative
    scalar paths to fixed literals and forces inline form content. ``fallback``
    optionally requests a fresh Template when the slot is unset, dangling or its
    alias is unbound. Without fallback, forms leave the input absent: required
    core fields are Invalid; nullable core fields with a None default remain
    disabled and Valid. Set is initialization-only; locks are reapplied on form
    branch switches.

    Construction detaches mappings and rejects an empty path with ValueError.
    TemplateDef expansion changes only path placeholders, not Form options or
    references. The library creation catalog accepts only Slot(path); it rejects
    nondefault copy/set/lock/fallback with TypeError and does not use Form fallback
    policy. Each consumer owns source lookup and schema compatibility failures.
    """

    path: str
    copy: bool = False
    set: Mapping[str, TemplateSeed] = field(default_factory=dict)
    lock: Mapping[str, TemplateLiteral] = field(default_factory=dict)
    fallback: Template | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("Slot path must not be empty")
        object.__setattr__(self, "set", MappingProxyType(dict(self.set)))
        object.__setattr__(self, "lock", MappingProxyType(dict(self.lock)))


type TemplateSeed = TemplateLiteral | Expr | Slot

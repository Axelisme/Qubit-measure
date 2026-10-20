"""Frontend-neutral authored nodes shared by templates and measure forms.

These declarations contain no source resolution or GUI behavior. TemplateSeed
is the closed seed vocabulary; consumers validate paths against their schemas.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from pydantic_core import PydanticUndefined, PydanticUndefinedType

type TemplateLiteral = str | int | float | bool | complex | None


@dataclass(frozen=True, slots=True)
class Expr:
    """Authored expression text; unlike a plain string, consumers evaluate it.

    ``text`` is the expression body, without a wire wrapper. Template expansion
    substitutes component placeholders but does not parse or evaluate the body.
    """

    text: str


@dataclass(frozen=True, slots=True)
class Hidden:
    """Omit a field from input and supply a literal during cfg conversion.

    value is the literal in the core field's units, still subject to core
    validation. With no argument, conversion restores the core default;
    a required field then fails at Form construction. Explicit None supplies
    None rather than restoring a default. Hidden does not infer a value source.
    """

    # Undefined distinguishes Hidden() from an explicit nullable Hidden(None).
    value: TemplateLiteral | PydanticUndefinedType = PydanticUndefined


@dataclass(frozen=True, slots=True)
class Template:
    """Request a fresh inline module from an injected template catalog.

    ``id`` names a TemplateDef in that catalog. ``set`` maps dotted paths relative
    to the module root to initial literal, Expr, Slot, Hidden(value), or typed model-list seeds.
    List elements map original Spec field names to seeds; consumers validate
    their shape and fields against that Spec. ``bindings`` maps
    plain placeholder names to authored replacement text. Explicit bindings
    override captured registry defaults and may use unregistered names; omitted
    bindings are empty.

    Construction detaches the mappings. Consumers validate the ID, module family,
    path compatibility and placeholder syntax; this node does not read sources.
    """

    id: str
    set: Mapping[str, TemplateSeed] = field(default_factory=dict)
    bindings: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "set", MappingProxyType(dict(self.set)))
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))


@dataclass(frozen=True, slots=True)
class Slot:
    """Request a library module slot/root, such as ``$qubit.module.readout``.

    ``path`` is an authored slot or library root, optionally containing component
    placeholders. Forms resolve it at Reset to a direct key. ``copy=False`` keeps
    a reference; ``copy=True`` copies the module inline. ``set`` maps relative
    dotted paths to initial literal, Expr, Slot, Hidden(value), or typed
    model-list seeds validated against the original Spec. Hidden
    supplies a validated literal and hides that inline field. When the slot
    is unset, dangling or its alias is unbound, forms leave input absent: required
    core fields are Invalid; nullable core fields with a None default remain
    disabled and Valid. Set is initialization-only.

    Construction detaches mappings and rejects an empty path with ValueError.
    TemplateDef expansion changes only path placeholders, not Form options or
    references. The library creation catalog accepts Slot(path), but rejects
    Slot.copy/set with TypeError. Bound unset/dangling sources keep nested
    structural defaults; unbound aliases stay linked empty. Each consumer owns
    source lookup and schema compatibility failures. Form authors choose alternate
    sources in make_form(ctx), not in this declaration.
    """

    path: str
    copy: bool = False
    set: Mapping[str, TemplateSeed] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("Slot path must not be empty")
        object.__setattr__(self, "set", MappingProxyType(dict(self.set)))


type TemplateSeed = (
    TemplateLiteral | Expr | Slot | Hidden | list[dict[str, TemplateSeed]]
)

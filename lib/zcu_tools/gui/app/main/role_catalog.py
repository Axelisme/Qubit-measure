"""Catalog of named "experiment-role" templates for blank ml entries.

A ``RoleEntry`` pairs a human-facing role (e.g. "Resonator probe readout") with
its eval-aware value factory — the existing ``make_<role>_default`` builders.
The GUI defines this interface; ``experiment/v2_gui`` populates it at startup
(mirroring ``Registry`` / ``register_all``), keeping the dependency direction
correct (gui must not import the adapters package).

Picking a role and a name seeds a blank ml module/waveform whose defaults are
"the thing to define" (md-linked values) rather than structural zero-values. It
is a one-shot create: editing the entry afterwards goes through the normal
modify path (inspect modify / ``editor.new(from_name=...)``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from zcu_tools.gui.cfg import CfgSectionSpec, LiteralSpec, ReferenceValue
from zcu_tools.gui.measure_cfg import PROGRAM_SHAPES, UnknownProgramShapeError

from .adapter import ExpContext

logger = logging.getLogger(__name__)

RoleItemKind: TypeAlias = Literal["module", "waveform"]
RoleShapeFactory: TypeAlias = Callable[[], CfgSectionSpec]
RoleValueFactory: TypeAlias = Callable[[ExpContext], ReferenceValue]


@dataclass(frozen=True, slots=True)
class RoleEntry:
    """One named experiment-role template.

    ``make_value`` is the eval-aware default factory for this role: given the
    live ``ExpContext`` it returns a ``ReferenceValue`` whose inner value carries
    md-linked ``EvalValue`` defaults.

    ``shape`` builds the matching context-free, fresh canonical Spec. Registration
    validates it without calling ``make_value``; create calls both factories fresh.

    ``default_name`` is a naming-convention suggestion for the create dialog's
    name field (e.g. ``"readout_rf"``); empty means "no suggestion" (blank roles
    leave the user to name the entry).
    """

    role_id: str
    label: str
    item_kind: RoleItemKind
    shape: RoleShapeFactory
    make_value: RoleValueFactory
    default_name: str = ""


class RoleCatalog:
    """Ordered registry of ``RoleEntry`` (insertion order = dropdown order)."""

    def __init__(self) -> None:
        self._entries: dict[str, RoleEntry] = {}

    def register(self, entry: RoleEntry) -> None:
        logger.debug("register role: id=%r kind=%r", entry.role_id, entry.item_kind)
        if entry.role_id in self._entries:
            raise ValueError(f"Role {entry.role_id!r} is already registered")
        _validate_entry_shape(entry)
        self._entries[entry.role_id] = entry

    def entries_for(self, item_kind: RoleItemKind) -> list[RoleEntry]:
        return [e for e in self._entries.values() if e.item_kind == item_kind]

    def get(self, role_id: str) -> RoleEntry:
        if role_id not in self._entries:
            raise KeyError(
                f"Role {role_id!r} not found; available: {list(self._entries)}"
            )
        return self._entries[role_id]

    def has(self, role_id: str) -> bool:
        return role_id in self._entries

    def list_meta(self) -> list[dict[str, str]]:
        """Wire-friendly metadata for every role (agent discovery)."""
        return [
            {
                "role_id": e.role_id,
                "label": e.label,
                "item_kind": e.item_kind,
                "default_name": e.default_name,
            }
            for e in self._entries.values()
        ]


def _validate_entry_shape(entry: RoleEntry) -> None:
    spec = entry.shape()
    if not isinstance(spec, CfgSectionSpec):
        raise TypeError(
            f"Role {entry.role_id!r} shape factory must return CfgSectionSpec, "
            f"got {type(spec).__name__}"
        )
    key = "type" if entry.item_kind == "module" else "style"
    other_kind: RoleItemKind = "waveform" if entry.item_kind == "module" else "module"
    other_key = "style" if entry.item_kind == "module" else "type"
    if key in spec.fields and other_key in spec.fields:
        raise ValueError(
            f"Role {entry.role_id!r} shape must declare exactly one root discriminator"
        )
    literal = spec.fields.get(key)
    if not isinstance(literal, LiteralSpec) or not isinstance(literal.value, str):
        other_literal = spec.fields.get(other_key)
        if isinstance(other_literal, LiteralSpec) and isinstance(
            other_literal.value, str
        ):
            raise TypeError(
                f"Role {entry.role_id!r} declares kind {entry.item_kind!r} but "
                f"shape root kind is {other_kind!r}"
            )
        raise ValueError(
            f"Role {entry.role_id!r} shape has no string literal discriminator {key!r}"
        )
    discriminator = literal.value
    try:
        PROGRAM_SHAPES.get(entry.item_kind, discriminator)
    except UnknownProgramShapeError:
        try:
            PROGRAM_SHAPES.get(other_kind, discriminator)
        except UnknownProgramShapeError as exc:
            raise ValueError(
                f"Role {entry.role_id!r} has unknown {entry.item_kind} shape "
                f"{discriminator!r}"
            ) from exc
        raise TypeError(
            f"Role {entry.role_id!r} declares kind {entry.item_kind!r} but "
            f"shape root kind is {other_kind!r}"
        ) from None

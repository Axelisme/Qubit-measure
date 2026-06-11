"""Catalog of named "experiment-role" templates for blank ml entries.

A ``RoleEntry`` pairs a human-facing role (e.g. "Resonator probe readout") with
its eval-aware value factory — the existing ``make_<role>_default`` builders.
The GUI defines this interface; ``experiment/v2_gui`` populates it at startup
(mirroring ``Registry`` / ``register_all``), keeping the dependency direction
correct (gui must not import the adapters package).

Picking a role and a name seeds a blank ml module/waveform whose defaults are
"the thing to define" (md-linked values) rather than structural zero-values. It
is a one-shot create: editing the entry afterwards goes through the normal
modify path (inspect modify / ``editor.open(from_name=...)``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal, TypeAlias, Union

from .adapter import ExpContext, ModuleRefValue, WaveformRefValue

logger = logging.getLogger(__name__)

RoleItemKind: TypeAlias = Literal["module", "waveform"]
RoleValueFactory: TypeAlias = Callable[[ExpContext], ModuleRefValue | WaveformRefValue]


class RoleEntry:
    """One named experiment-role template.

    ``make_value`` is the eval-aware default factory for this role: given the
    live ``ExpContext`` it returns a ``ModuleRefValue`` (modules) or
    ``WaveformRefValue`` (waveforms) whose inner value carries md-linked
    ``EvalValue`` defaults.

    ``default_name`` is a naming-convention suggestion for the create dialog's
    name field (e.g. ``"readout_rf"``); empty means "no suggestion" (blank roles
    leave the user to name the entry).
    """

    __slots__ = ("role_id", "label", "item_kind", "make_value", "default_name")

    def __init__(
        self,
        role_id: str,
        label: str,
        item_kind: RoleItemKind,
        make_value: RoleValueFactory,
        default_name: str = "",
    ) -> None:
        self.role_id = role_id
        self.label = label
        self.item_kind = item_kind
        self.make_value = make_value
        self.default_name = default_name


class RoleCatalog:
    """Ordered registry of ``RoleEntry`` (insertion order = dropdown order)."""

    def __init__(self) -> None:
        self._entries: dict[str, RoleEntry] = {}

    def register(self, entry: RoleEntry) -> None:
        logger.debug("register role: id=%r kind=%r", entry.role_id, entry.item_kind)
        if entry.role_id in self._entries:
            raise ValueError(f"Role {entry.role_id!r} is already registered")
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

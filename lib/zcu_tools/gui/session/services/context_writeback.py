"""Bind selected-point commands to the ordered writeback execution port."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from zcu_tools.format_version import YamlMap, YamlValue
    from zcu_tools.gui.session.context_types import SavedPoint
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.resources.entry import ResultEntry
    from zcu_tools.resources.entry.ledger_models import JsonObject
    from zcu_tools.resources.entry.writeback import LibraryUpdate, ParameterUpdate


class ContextWritebackTarget:
    """Bind the context's target setters to the shared ordered execution loop."""

    def __init__(
        self, context: ContextService, target: SavedPoint, entry: ResultEntry
    ) -> None:
        self._context = context
        self._target = target
        self._entry = entry
        self.entry_id = entry.entry_id
        self.ledger = entry.ledger
        self.point = target.point

    def set_parameters(
        self, updates: tuple[ParameterUpdate, ...]
    ) -> tuple[YamlValue, ...]:
        return self._context.set_parameters(self._target, updates)

    def set_modules(self, updates: tuple[LibraryUpdate, ...]) -> tuple[YamlMap, ...]:
        return self._context.set_modules(self._target, updates)

    def append_sample(
        self,
        component: str,
        fields: JsonObject,
        *,
        run_id: str,
        at: datetime,
        source: str,
    ) -> JsonObject:
        return self._entry.samples(component).append(
            run_id=run_id, point=self.point, at=at, source=source, **fields
        )

"""Entry and point management for shared session UI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

# ADR-0067: shared session ports do not depend on app or Qt owners.
if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.context_types import (
        DocumentSetRef,
        EntryInfo,
        EntryRequest,
        PointCreateRequest,
        WorkingPointInfo,
    )
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.services.project_settings import (
        DialogPrefill,
        ProjectSettingsService,
    )


class EntryControlPort(Protocol):
    """Entry/point surface for the shared Entry dialog.

    devices exposes the already-injected DeviceControlPort. Entry uses its
    State-only list/value/unit queries to suggest an editable working-point
    label; it does not poll hardware or bind a point to a device.
    """

    devices: DeviceControlPort

    def get_bus(self) -> BaseEventBus:
        """Return the session fact bus; the caller owns its subscriptions."""
        ...

    def get_dialog_prefill(self) -> DialogPrefill:
        """Read cached Entry/SoC input values without opening or connecting."""
        ...

    def list_entries(self, *, refresh: bool = False) -> tuple[str, ...]:
        """List cached entry names; True explicitly refreshes via resources."""
        ...

    def open_entry(self, request: EntryRequest) -> EntryInfo:
        """Open through resources and leave point unselected; errors preserve the binding."""
        ...

    def new_entry(self, request: EntryRequest) -> EntryInfo:
        """Create an entry without selecting a point; installation failure retains it for Open."""
        ...

    def get_entry_info(self) -> EntryInfo | None:
        """Read cached opened entry identity/paths, or None."""
        ...

    def current_point_label(self) -> str | None:
        """Read the selected saved point label, or None when none is selected; no I/O."""
        ...

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        """Read cached point descriptions; True explicitly refreshes through owner."""
        ...

    def use_point(self, point: str | None = None) -> DocumentSetRef:
        """Select global for None/omitted, or any saved point label in the entry.

        No opened entry raises no_entry. Preparation failure preserves binding,
        versions, revisions and captures; alias format/syntax errors are deferred
        to consumers, following ContextService.use_point.
        """
        ...

    def new_point(self, request: PointCreateRequest) -> DocumentSetRef:
        """Create with inherit/empty/clone then select; binding failure retains destination."""
        ...


class EntryControlFacet(EntryControlPort):
    """Composition facade over the services used by EntryDialog."""

    def __init__(
        self,
        *,
        bus: BaseEventBus,
        settings: ProjectSettingsService,
        context: ContextControlPort,
        device: DeviceControlPort,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._context = context
        self.devices = device

    def get_bus(self) -> BaseEventBus:
        return self._bus

    def get_dialog_prefill(self) -> DialogPrefill:
        return self._settings.get_dialog_prefill()

    def list_entries(self, *, refresh: bool = False) -> tuple[str, ...]:
        return self._settings.list_entries(refresh=refresh)

    def open_entry(self, request: EntryRequest) -> EntryInfo:
        return self._settings.open_entry(request)

    def new_entry(self, request: EntryRequest) -> EntryInfo:
        return self._settings.new_entry(request)

    def get_entry_info(self) -> EntryInfo | None:
        return self._context.get_entry_info()

    def current_point_label(self) -> str | None:
        return self._context.current_point_label()

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        return self._context.list_points(refresh=refresh)

    def use_point(self, point: str | None = None) -> DocumentSetRef:
        """Select global for None/omitted, or any saved label; owner errors propagate.

        Preparation failure preserves binding, versions, revisions and captures.
        Alias format/syntax errors are deferred to consumers.
        """
        return self._context.use_point(point)

    def new_point(self, request: PointCreateRequest) -> DocumentSetRef:
        return self._context.new_point(request)

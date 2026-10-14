"""Read-only session chips shown above the workflow Window's two columns."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import ContextSourceUnavailableError
from zcu_tools.gui.session.events import (
    AliasesChangedPayload,
    ContextSwitchedPayload,
    DeviceChangedPayload,
    MdChangedPayload,
    SessionPayload,
    SocChangedPayload,
)
from zcu_tools.gui.session.services.build import SessionServices
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.resources.entry import RoleResolutionError

_CHIP_STYLE = (
    "QLabel { background: #eef2f7; border: 1px solid #c9d3e0;"
    " border-radius: 9px; padding: 2px 8px; color: #1f2937; }"
)
_WARN_CHIP_STYLE = (
    "QLabel { background: #fff4e5; border: 1px solid #f0c27a;"
    " border-radius: 9px; padding: 2px 8px; color: #7a4a00; }"
)
# Long diagnostics are elided so a chip never widens the Window; the tooltip
# keeps the full text. The text room excludes the style's padding and border.
_CHIP_MAX_WIDTH = 420
_CHIP_TEXT_WIDTH = _CHIP_MAX_WIDTH - 2 * (8 + 1)


class WorkflowSessionStrip(QWidget):
    """Show Entry/Point, Qubit, connected devices and ZCU state as chips.

    Each chip is a QLabel whose text starts with its title, in this order:
    ``workflow.session.entry`` shows ``Entry: <entry> / <point>`` (``no point``
    before selection, ``Entry: none`` without an entry);
    ``workflow.source.status`` shows ``Qubit: <component> (<kind>)`` for the
    global ``$qubit`` alias, ``Qubit: no context`` for EMPTY, or
    ``Qubit unavailable: <reason>``; ``workflow.session.devices`` lists the
    registered device names or ``none``; ``workflow.session.zcu`` shows
    ``simulator``, ``<address>:<port>``, ``connected`` or ``not connected``.
    Text wider than the chip is elided and the full text is the tooltip.

    The strip reads services only; it never opens entries, resolves forms or
    touches hardware. It refreshes on context, alias, point-content, SoC and
    device events from the session bus, and whenever refresh is called.
    detach drops those subscriptions; destroying the widget also drops them.
    """

    def __init__(
        self, services: SessionServices, parent: QWidget | None = None
    ) -> None:
        """Build the chips on the Qt owner and show the current session once.

        services is the Window's shared session bundle; parent owns Qt lifetime.
        """
        super().__init__(parent)
        self.setObjectName("workflow.session")
        self._services = services
        self._entry = self._chip("workflow.session.entry")
        self._qubit = self._chip("workflow.source.status")
        self._devices = self._chip("workflow.session.devices")
        self._zcu = self._chip("workflow.session.zcu")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for chip in (self._entry, self._qubit, self._devices, self._zcu):
            layout.addWidget(chip)
        self._subscriptions = EventSubscriptions()
        bus = services.setup_control.get_bus()
        for payload in (
            ContextSwitchedPayload,
            AliasesChangedPayload,
            MdChangedPayload,
            SocChangedPayload,
            DeviceChangedPayload,
        ):
            self._subscriptions.subscribe(bus, payload, self._session_changed)
        self.destroyed.connect(self._subscriptions.unsubscribe_all)
        self.refresh()

    def _chip(self, name: str) -> QLabel:
        chip = QLabel(self)
        chip.setObjectName(name)
        chip.setStyleSheet(_CHIP_STYLE)
        chip.setMaximumWidth(_CHIP_MAX_WIDTH)
        make_label_selectable(chip)
        return chip

    def refresh(self, *_args: object) -> None:
        """Re-read every chip from services; accepts and ignores Qt signal args."""
        self._show(self._entry, self._entry_text())
        qubit, available = self._qubit_text()
        self._show(self._qubit, qubit)
        self._qubit.setStyleSheet(_CHIP_STYLE if available else _WARN_CHIP_STYLE)
        names = tuple(self._services.device.get_connected_devices())
        self._show(self._devices, f"Devices: {', '.join(names) if names else 'none'}")
        self._show(self._zcu, f"ZCU: {self._zcu_state()}")

    def detach(self) -> None:
        """Stop following session events; the chips keep their last text."""
        self._subscriptions.unsubscribe_all()

    def _session_changed(self, _payload: SessionPayload) -> None:
        self.refresh()

    def _entry_text(self) -> str:
        control = self._services.context_control
        entry = control.get_entry_info()
        if entry is None:
            return "Entry: none"
        point = control.current_point_label()
        return f"Entry: {entry.name} / {point if point is not None else 'no point'}"

    def _qubit_text(self) -> tuple[str, bool]:
        try:
            parameters = self._services.context.read_context()
            name = parameters.resolve_path("$qubit")
            kind = parameters.lookup_point("$qubit.kind")
        # ContextSourceUnavailableError subclasses FailedPreconditionError, so it
        # must be caught first: an unreadable source is not an EMPTY context.
        except (RoleResolutionError, ContextSourceUnavailableError) as error:
            return f"Qubit unavailable: {error}", False
        except FailedPreconditionError:
            # EMPTY is expected at fresh app startup; status must not block it.
            return "Qubit: no context", True
        if not kind.present:
            return f"Qubit unavailable: {name} has no kind", False
        return f"Qubit: {name} ({kind.value})", True

    def _zcu_state(self) -> str:
        connection = self._services.soc_connection
        if not connection.has_soc():
            return "not connected"
        if connection.is_mock_soc():
            return "simulator"
        endpoint = connection.connected_endpoint()
        if endpoint["address"] is None:
            return "connected"
        return f"{endpoint['address']}:{endpoint['port']}"

    def _show(self, chip: QLabel, text: str) -> None:
        shown = chip.fontMetrics().elidedText(
            text, Qt.TextElideMode.ElideRight, _CHIP_TEXT_WIDTH
        )
        chip.setText(shown)
        chip.setToolTip(text if shown != text else "")

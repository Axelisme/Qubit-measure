"""Read-only session chip shown above the workflow Window's two columns."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.session.events import ContextSwitchedPayload
from zcu_tools.gui.session.services.build import SessionServices

_CHIP_STYLE = (
    "QLabel { background: #eef2f7; border: 1px solid #c9d3e0;"
    " border-radius: 9px; padding: 2px 8px; color: #1f2937; }"
)
# Long names are elided so the chip never widens the Window; the tooltip
# keeps the full text. The text room excludes the style's padding and border.
_CHIP_MAX_WIDTH = 420
_CHIP_TEXT_WIDTH = _CHIP_MAX_WIDTH - 2 * (8 + 1)


class WorkflowSessionStrip(QWidget):
    """Show the session's Entry and Point as one chip.

    The chip is the QLabel ``workflow.session.entry`` showing
    ``Entry: <entry> / <point>`` (``no point`` before selection,
    ``Entry: none`` without an entry). Text wider than the chip is elided and
    the full text is the tooltip.

    The strip reads services only; it never opens entries or touches hardware.
    It refreshes on context switches from the session bus, and whenever
    refresh is called. detach drops that subscription; destroying the widget
    also drops it.
    """

    def __init__(
        self, services: SessionServices, parent: QWidget | None = None
    ) -> None:
        """Build the chip on the Qt owner and show the current session once.

        services is the Window's shared session bundle; parent owns Qt lifetime.
        """
        super().__init__(parent)
        self.setObjectName("workflow.session")
        self._services = services
        self._entry = QLabel(self)
        self._entry.setObjectName("workflow.session.entry")
        self._entry.setStyleSheet(_CHIP_STYLE)
        self._entry.setMaximumWidth(_CHIP_MAX_WIDTH)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._entry)
        self._subscriptions = EventSubscriptions()
        self._subscriptions.subscribe(
            services.setup_control.get_bus(), ContextSwitchedPayload, self.refresh
        )
        self.destroyed.connect(self._subscriptions.unsubscribe_all)
        self.refresh()

    def refresh(self, *_args: object) -> None:
        """Re-read the chip from services; accepts and ignores signal/event args."""
        text = self._entry_text()
        shown = self._entry.fontMetrics().elidedText(
            text, Qt.TextElideMode.ElideRight, _CHIP_TEXT_WIDTH
        )
        self._entry.setText(shown)
        self._entry.setToolTip(text if shown != text else "")

    def detach(self) -> None:
        """Stop following session events; the chip keeps its last text."""
        self._subscriptions.unsubscribe_all()

    def _entry_text(self) -> str:
        control = self._services.context_control
        entry = control.get_entry_info()
        if entry is None:
            return "Entry: none"
        point = control.current_point_label()
        return f"Entry: {entry.name} / {point if point is not None else 'no point'}"

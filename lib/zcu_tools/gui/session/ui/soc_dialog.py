"""Shared SoC connection form and configuration browser."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from qtpy.QtGui import QColor, QShowEvent
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.services.project_settings import ConnectionPreferences
from zcu_tools.gui.widgets import error_colour, set_foreground, success_colour
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.program import describe_soc

if TYPE_CHECKING:
    from zcu_tools.gui.session.setup_control import SocControlPort


def _breakable(text: str) -> str:
    """Return text with a zero-width space after each path separator.

    Paths contain no spaces, so a word-wrapped QLabel could not break them and
    would widen the dialog instead; the invisible break opportunities fix that.
    """
    return text.replace("/", "/\u200b").replace("\\", "\\\u200b")


def _wrapping_label(object_name: str = "") -> QLabel:
    """Return a selectable, word-wrapped label whose text never sets its width."""
    label = QLabel("")
    if object_name:
        label.setObjectName(object_name)
    label.setWordWrap(True)
    make_label_selectable(label)
    # Minimum height: wrapped lines are never squeezed out by the layout.
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
    return label


@runtime_checkable
class SocConfigProtocol(Protocol):
    """Protocol for QICK Soc configuration objects (dict-style field access)."""

    def __getitem__(self, key: str) -> object: ...


class SocDialog(QDialog):
    """Connect a remote SoC and browse its configuration through SocControlPort.

    Construct and use on the Qt owner thread. Showing the dialog prefills cached
    connection preferences and synchronizes connection state/configuration from
    the session cache without accessing hardware. Connect remembers the
    submitted address and starts the asynchronous operation; admission and
    terminal failures remain visible for retry. The host controls write access
    with set_read_only. Close hides the dialog without cancelling the operation.
    """

    def __init__(
        self, controller: SocControlPort, parent: QWidget | None = None
    ) -> None:
        """Build the form over a session's SoC facet and optional Qt parent."""
        super().__init__(parent)
        self._ctrl = controller
        self._read_only = False
        self._connecting = False
        self.setWindowTitle("SoC")
        self.resize(600, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_connection_panel(), stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._prefill_from_preferences()
        self._maybe_show_current_cfg()
        controller.get_bus().subscribe(SocChangedPayload, self._on_bus_soc_changed)
        self._bus_subs_active = True
        self.finished.connect(self._cleanup_bus_subscriptions)
        self.destroyed.connect(self._cleanup_bus_subscriptions)

    def _build_connection_panel(self) -> QWidget:
        # Connection form and read-only configuration browser.
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        conn_group = QGroupBox("ZCU Connection")
        conn_form = QFormLayout(conn_group)

        self._ip_edit = QLineEdit("192.168.10.1")
        self._ip_edit.setObjectName("soc_ip")
        # Keep any IPv4 address fully visible when the dialog is narrow.
        self._ip_edit.setMinimumWidth(
            self._ip_edit.fontMetrics().horizontalAdvance("255.255.255.255 ") + 12
        )
        conn_form.addRow("IP address:", self._ip_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setObjectName("soc_port")
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(8887)
        conn_form.addRow("Port:", self._port_spin)

        right_layout.addWidget(conn_group)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setDefault(True)
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        right_layout.addWidget(self._connect_btn)

        self._conn_status = _wrapping_label()
        right_layout.addWidget(self._conn_status)

        # soccfg description — hidden until connection succeeds
        self._cfg_text = QPlainTextEdit()
        self._cfg_text.setReadOnly(True)
        self._cfg_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._cfg_text.setVisible(False)
        right_layout.addWidget(self._cfg_text, stretch=1)

        right_layout.addStretch()
        return right_widget

    def set_read_only(self, *, read_only: bool) -> None:
        """Gate connection edits while preserving configuration browsing and Close."""
        self._read_only = read_only
        self._ip_edit.setEnabled(not read_only)
        self._port_spin.setEnabled(not read_only)
        self._refresh_connect_enabled()

    def _refresh_connect_enabled(self) -> None:
        self._connect_btn.setEnabled(not self._read_only and not self._connecting)

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        if self._bus_subs_active:
            self._ctrl.get_bus().unsubscribe(
                SocChangedPayload, self._on_bus_soc_changed
            )
            self._bus_subs_active = False

    def showEvent(self, a0: QShowEvent | None) -> None:
        """Sync cached inputs and connection on explicit show, preserving minimise drafts."""
        if a0 is not None:
            super().showEvent(a0)
            if a0.spontaneous():
                return
        if not self._bus_subs_active:
            self._ctrl.get_bus().subscribe(SocChangedPayload, self._on_bus_soc_changed)
            self._bus_subs_active = True
        self._prefill_from_preferences()
        self._maybe_show_current_cfg()

    def _prefill_from_preferences(self) -> None:
        data = self._ctrl.get_setup_preferences()
        if data.ip:
            self._ip_edit.setText(data.ip)
        if data.port:
            self._port_spin.setValue(data.port)

    def _on_bus_soc_changed(self, payload: object) -> None:
        from zcu_tools.gui.session.events import SocChangedPayload

        self._refresh_connect_enabled()
        if isinstance(payload, SocChangedPayload) and payload.soc is not None:
            self._set_conn_status("Connected", error=False)
            self._maybe_show_current_cfg()
        else:
            self._set_conn_status("Disconnected", error=False)
            self._cfg_text.clear()
            self._cfg_text.setVisible(False)

    @staticmethod
    def _show_status(label: QLabel, msg: str, color: QColor) -> None:
        label.setText(_breakable(msg))
        label.setToolTip(msg)
        set_foreground(label, color)

    def _maybe_show_current_cfg(self) -> None:
        soccfg = self._ctrl.get_soccfg()
        if isinstance(soccfg, SocConfigProtocol):
            self._show_cfg(describe_soc(soccfg))
            self._set_conn_status("Currently connected", error=False)
        else:
            self._cfg_text.clear()
            self._cfg_text.setVisible(False)
            self._set_conn_status("Disconnected", error=False)

    def _on_connect_clicked(self) -> None:
        if self._read_only or self._connecting:
            return
        from zcu_tools.gui.session.services.connection import ConnectRemoteRequest

        self._ctrl.bind_connection_outcome(
            self._on_connect_finished, self._on_connect_failed
        )
        self._ctrl.remember_connection(
            ConnectionPreferences(
                ip=self._ip_edit.text().strip(), port=self._port_spin.value()
            )
        )
        self._connecting = True
        self._refresh_connect_enabled()
        self._set_conn_status("Connecting…", error=False)
        try:
            self._ctrl.start_connect(
                ConnectRemoteRequest(
                    ip=self._ip_edit.text().strip(), port=self._port_spin.value()
                )
            )
        except Exception as exc:  # noqa: BLE001 — restore the button on admission rejection
            self._on_connect_failed(str(exc))

    def _on_connect_finished(self) -> None:
        self._connecting = False
        self._refresh_connect_enabled()
        self._set_conn_status("Connected", error=False)
        soccfg = self._ctrl.get_soccfg()
        if isinstance(soccfg, SocConfigProtocol):
            self._show_cfg(describe_soc(soccfg))

    def _on_connect_failed(self, message: str) -> None:
        self._connecting = False
        self._refresh_connect_enabled()
        self._set_conn_status(message, error=True)

    def _show_cfg(self, text: str) -> None:
        self._cfg_text.setPlainText(text)
        self._cfg_text.setVisible(True)

    def _set_conn_status(self, msg: str, *, error: bool = False) -> None:
        self._show_status(
            self._conn_status,
            msg,
            error_colour(self) if error else success_colour(self),
        )

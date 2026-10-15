"""SetupDialog — explicit result entry binding and ZCU connection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QDesktopServices, QResizeEvent, QShowEvent
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryInfo,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.services.project_settings import ConnectionPreferences
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.program import describe_soc

if TYPE_CHECKING:
    from zcu_tools.gui.session.setup_control import SetupControlPort

logger = logging.getLogger(__name__)


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


class _ElidedPathLabel(QLabel):
    """Show one filesystem path elided in the middle to the current width.

    The last segment (the entry name) stays visible; the tooltip and path()
    always hold the full path. An empty path clears the label.
    """

    def __init__(self, object_name: str) -> None:
        super().__init__("")
        self.setObjectName(object_name)
        make_label_selectable(self)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._path = ""

    def path(self) -> str:
        """Return the full path last given to set_path, or ''."""
        return self._path

    def set_path(self, path: str) -> None:
        """Store path, show it in the tooltip and elide the visible text."""
        self._path = path
        self.setToolTip(path)
        self._update_text()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._update_text()

    def _update_text(self) -> None:
        self.setText(
            self.fontMetrics().elidedText(
                self._path, Qt.TextElideMode.ElideMiddle, max(self.width(), 0)
            )
        )


@runtime_checkable
class SocConfigProtocol(Protocol):
    """Protocol for QICK Soc configuration objects (dict-style field access)."""

    def __getitem__(self, key: str) -> object: ...


class SetupDialog(QDialog):
    """Resizable result-entry and connection form over SetupControlPort.

    Showing/reopening only prefills cached preferences and queries; it does not
    open an entry or connect hardware. New/Open publish the owner binding and
    display its paths, elided in the middle with the full path as tooltip and
    an Open folder action. The working-point list shows labels only, marks the
    current point with a bold "● " prefix and keeps the selection on it; the
    selected point's description shows below the list. Switch/Create submit
    the exact label, description and same-entry clone choice to the owner;
    a successful Create clears the form. Failures remain visible in the
    wrapped status label without retry; a missing result root is shown as a
    hint because New entry creates it. Track device only suggests an
    editable label from cached value/unit, preserving manual edits and None.
    """

    def __init__(
        self,
        controller: SetupControlPort,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Setup")
        self.resize(900, 680)

        root_layout = QVBoxLayout(self)

        # ── horizontal splitter: left = project, right = connection ──────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, stretch=1)

        # ── Left panel: Project ──────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        left_layout.addWidget(self._build_entry_group())

        left_layout.addWidget(self._build_point_list(), stretch=1)

        left_layout.addWidget(self._build_new_point_group())

        # project status label
        splitter.addWidget(left_widget)

        splitter.addWidget(self._build_connection_panel())

        splitter.setSizes([450, 450])

        # ── Close button ─────────────────────────────────────────────────
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        # Full-width status keeps long project messages to as few lines as possible.
        self._project_status = _wrapping_label("setup_status")
        root_layout.addWidget(self._project_status)
        root_layout.addWidget(btn_box)

        # initialise
        self._refresh_entries()
        self._prefill_from_preferences()
        self._refresh_device_list()
        self._refresh_context_list()
        self._maybe_show_current_cfg()

        # EventBus subscriptions for live updates
        from zcu_tools.gui.session.events import (
            ContextSwitchedPayload,
            DeviceChangedPayload,
            SocChangedPayload,
        )

        bus = controller.get_bus()
        bus.subscribe(ContextSwitchedPayload, self._on_bus_context_switched)
        bus.subscribe(SocChangedPayload, self._on_bus_soc_changed)
        bus.subscribe(DeviceChangedPayload, self._on_bus_device_changed)
        self._bus_subs_active = True
        self.finished.connect(self._cleanup_bus_subscriptions)
        self.destroyed.connect(self._cleanup_bus_subscriptions)

    def _build_entry_group(self) -> QGroupBox:
        # Entry binding is explicit; changing the selection only edits the form.
        entry_group = QGroupBox("Result entry")
        entry_form = QFormLayout(entry_group)
        entry_row = QHBoxLayout()
        self._entry_combo = QComboBox()
        self._entry_combo.setObjectName("result_entry")
        self._entry_combo.setMinimumWidth(260)
        entry_row.addWidget(self._entry_combo)
        refresh_entries_btn = QPushButton("↻")
        refresh_entries_btn.setFixedWidth(28)
        refresh_entries_btn.setToolTip("Refresh result entries")
        refresh_entries_btn.clicked.connect(
            lambda checked=False: self._refresh_entries(refresh=True)
        )
        entry_row.addWidget(refresh_entries_btn)
        entry_form.addRow("Entry:", entry_row)

        entry_actions = QHBoxLayout()
        new_entry_btn = QPushButton("New entry…")
        new_entry_btn.clicked.connect(self._on_new_entry_clicked)
        entry_actions.addWidget(new_entry_btn)
        open_entry_btn = QPushButton("Open")
        open_entry_btn.clicked.connect(self._on_open_entry_clicked)
        entry_actions.addWidget(open_entry_btn)
        entry_form.addRow("", entry_actions)
        self._result_path = _ElidedPathLabel("result_path")
        self._result_open = self._open_folder_button(self._result_path)
        entry_form.addRow(
            "Result:", self._path_row(self._result_path, self._result_open)
        )
        self._database_path = _ElidedPathLabel("database_path")
        self._database_open = self._open_folder_button(self._database_path)
        entry_form.addRow(
            "Database:", self._path_row(self._database_path, self._database_open)
        )
        return entry_group

    def _build_point_list(self) -> QGroupBox:
        # context list
        ctx_group = QGroupBox("Working points")
        ctx_layout = QVBoxLayout(ctx_group)
        self._ctx_list = QListWidget()
        self._ctx_list.setObjectName("working_points")
        self._ctx_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._ctx_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._ctx_list.currentItemChanged.connect(self._show_selected_description)
        rows = self._ctx_list.fontMetrics().height() + 4
        self._ctx_list.setMinimumHeight(4 * rows + 2 * self._ctx_list.frameWidth())
        ctx_layout.addWidget(self._ctx_list, stretch=1)
        self._point_descriptions: dict[str, str] = {}
        self._selected_description = _wrapping_label("selected_point_description")
        ctx_layout.addWidget(self._selected_description)

        switch_row = QHBoxLayout()
        self._switch_btn = QPushButton("Switch to selected")
        self._switch_btn.clicked.connect(self._on_switch_clicked)
        switch_row.addWidget(self._switch_btn)
        ctx_layout.addLayout(switch_row)
        return ctx_group

    def _build_new_point_group(self) -> QGroupBox:
        # new context
        new_group = QGroupBox("New working point")
        new_form = QFormLayout(new_group)

        device_row = QHBoxLayout()
        self._device_combo = QComboBox()
        self._device_combo.setObjectName("track_device")
        self._device_combo.setMinimumWidth(160)
        self._device_combo.currentTextChanged.connect(self._on_device_changed)
        device_row.addWidget(self._device_combo)
        refresh_dev_btn = QPushButton("↻")
        refresh_dev_btn.setFixedWidth(28)
        refresh_dev_btn.setToolTip("Refresh device list")
        refresh_dev_btn.clicked.connect(self._refresh_device_list)
        device_row.addWidget(refresh_dev_btn)
        new_form.addRow("Track device:", device_row)

        self._label_edit = QLineEdit()
        self._label_edit.setObjectName("point_label")
        new_form.addRow("Label:", self._label_edit)

        self._clone_combo = QComboBox()
        self._clone_combo.setObjectName("clone_from")
        self._clone_combo.setMinimumWidth(160)
        new_form.addRow("Clone from:", self._clone_combo)

        self._description_edit = QLineEdit()
        self._description_edit.setObjectName("point_description")
        new_form.addRow("Description:", self._description_edit)

        self._new_ctx_btn = QPushButton("Create")
        self._new_ctx_btn.clicked.connect(self._on_new_ctx_clicked)
        new_form.addRow("", self._new_ctx_btn)
        return new_group

    def _build_connection_panel(self) -> QWidget:
        # ── Right panel: Connection ──────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        conn_group = QGroupBox("ZCU Connection")
        conn_form = QFormLayout(conn_group)

        self._ip_edit = QLineEdit("192.168.10.1")
        # Keep any IPv4 address fully visible when the dialog is narrow.
        self._ip_edit.setMinimumWidth(
            self._ip_edit.fontMetrics().horizontalAdvance("255.255.255.255 ") + 12
        )
        conn_form.addRow("IP address:", self._ip_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(8887)
        conn_form.addRow("Port:", self._port_spin)

        right_layout.addWidget(conn_group)

        self._connect_btn = QPushButton("Connect")
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

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        if not self._bus_subs_active:
            logger.debug("_cleanup_bus_subscriptions called but already inactive")
            return
        from zcu_tools.gui.session.events import (
            ContextSwitchedPayload,
            DeviceChangedPayload,
            SocChangedPayload,
        )

        bus = self._ctrl.get_bus()
        bus.unsubscribe(ContextSwitchedPayload, self._on_bus_context_switched)
        bus.unsubscribe(SocChangedPayload, self._on_bus_soc_changed)
        bus.unsubscribe(DeviceChangedPayload, self._on_bus_device_changed)
        self._bus_subs_active = False

    def showEvent(self, a0: QShowEvent | None) -> None:
        """Re-seed entry/connection inputs from cached preferences on show.

        Reopening the dialog after a close (``show()`` on the kept instance)
        replaces any previously-typed-but-never-applied draft with the
        remembered values from ``get_setup_preferences``.  The first show is
        harmless because ``_prefill_from_preferences`` already ran in
        ``__init__``.

        Spontaneous show events (window-system restore after minimise) keep the
        draft — the user never closed the dialog, so wiping mid-typing input
        would be hostile.  ``a0`` may be ``None`` in unit tests that call
        ``showEvent`` directly to simulate a reopen without a real Qt event loop.
        """
        if a0 is not None:
            super().showEvent(a0)
            if a0.spontaneous():
                return
        self._refresh_entries()
        self._prefill_from_preferences()
        self._refresh_context_list()

    def _on_bus_context_switched(self, payload: object) -> None:
        del payload
        self._refresh_context_list()
        info = self._ctrl.get_entry_info()
        if info is not None:
            self._select_entry_name(info.name)
            self._show_entry_paths(info)

    def _on_bus_soc_changed(self, payload: object) -> None:
        from zcu_tools.gui.session.events import SocChangedPayload

        self._connect_btn.setEnabled(True)
        if isinstance(payload, SocChangedPayload) and payload.soc is not None:
            self._set_conn_status("Connected", error=False)
            self._maybe_show_current_cfg()
        else:
            self._set_conn_status("Disconnected", error=False)
            self._cfg_text.setVisible(False)

    def _on_bus_device_changed(self, payload: object) -> None:
        del payload
        self._refresh_device_list()

    # ------------------------------------------------------------------
    # Project panel handlers
    # ------------------------------------------------------------------

    def _prefill_from_preferences(self) -> None:
        data = self._ctrl.get_setup_preferences()
        if data.entry_name:
            self._select_entry_name(data.entry_name)
        if data.ip:
            self._ip_edit.setText(data.ip)
        if data.port:
            self._port_spin.setValue(data.port)
        info = self._ctrl.get_entry_info()
        if info is not None:
            self._show_entry_paths(info)

    def _select_entry_name(self, name: str) -> None:
        idx = self._entry_combo.findText(name)
        if idx < 0:
            # Remembered preferences may outlive the cached discovery result.
            self._entry_combo.addItem(name)
            idx = self._entry_combo.count() - 1
        self._entry_combo.setCurrentIndex(idx)

    def _refresh_entries(self, *, refresh: bool = False) -> None:
        try:
            names = self._ctrl.list_entries(refresh=refresh)
        except FileNotFoundError as exc:
            # A missing result root is the normal first-run state, not a failure.
            root = exc.filename if exc.filename is not None else "the result root"
            self._entry_combo.clear()
            self._set_project_hint(f"No result entries yet. New entry… creates {root}")
            return
        except Exception as exc:
            # Keep discovery failures visible without escaping the Qt callback.
            logger.exception("Failed to list result entries")
            self._set_project_status(
                f"Failed to refresh result entries: {exc}", error=True
            )
            return
        current = self._entry_combo.currentText()
        self._entry_combo.clear()
        self._entry_combo.addItems(list(names))
        idx = self._entry_combo.findText(current)
        if idx >= 0:
            self._entry_combo.setCurrentIndex(idx)

    def _open_folder_button(self, label: _ElidedPathLabel) -> QPushButton:
        """Return a disabled Open folder button that opens label's full path."""
        button = QPushButton("Open folder")
        button.setEnabled(False)
        button.clicked.connect(lambda checked=False: self._open_folder(label.path()))
        return button

    @staticmethod
    def _path_row(label: _ElidedPathLabel, button: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(label, stretch=1)
        row.addWidget(button)
        return row

    def _show_entry_paths(self, info: EntryInfo) -> None:
        self._result_path.set_path(str(info.result_path))
        self._database_path.set_path(str(info.database_path))
        self._result_open.setEnabled(True)
        self._database_open.setEnabled(True)

    def _open_folder(self, path: str) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _refresh_device_list(self) -> None:
        entries = self._ctrl.devices.list_devices()

        current = self._device_combo.currentText()
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._device_combo.addItem("(none)")
        for entry in entries:
            # Only offer live drivers in the flux-device combo, keeping the prior
            # is_connected bool semantics (connected / disconnecting / setting_up,
            # not the transient connecting state; DeviceEntry.status, FC7).
            if entry.status in ("connected", "disconnecting", "setting_up"):
                self._device_combo.addItem(
                    f"{entry.name}  [{entry.type_name}]", userData=entry.name
                )
        self._device_combo.blockSignals(False)

        idx = self._device_combo.findText(current)
        self._device_combo.setCurrentIndex(max(idx, 0))
        self._on_device_changed(self._device_combo.currentText())

    def _on_device_changed(self, _text: str) -> None:
        if self._label_edit.isModified():
            return
        name: str | None = self._device_combo.currentData()
        if name is None:
            return
        value = self._ctrl.devices.get_cached_device_value(name)
        if value is None:
            return
        unit = self._ctrl.devices.get_device_unit(name)
        self._label_edit.setText(f"{value}{unit}")

    def _on_new_entry_clicked(self) -> None:
        name, accepted = QInputDialog.getText(self, "New entry", "Entry name:")
        if not accepted:
            return
        try:
            info = self._ctrl.new_entry(EntryRequest(name))
        except Exception as exc:
            # Isolate the Qt callback; the owner preserves binding and recovery.
            logger.exception("Failed to create result entry")
            self._set_project_status(str(exc), error=True)
            return
        self._entry_installed(info)

    def _on_open_entry_clicked(self) -> None:
        name = self._entry_combo.currentText()
        if not name:
            self._set_project_status("Select an entry first", error=True)
            return
        try:
            info = self._ctrl.open_entry(EntryRequest(name))
        except Exception as exc:
            # Isolate the Qt callback without hiding the owner's failure.
            logger.exception("Failed to open result entry")
            self._set_project_status(str(exc), error=True)
            return
        self._entry_installed(info)

    def _entry_installed(self, info: EntryInfo) -> None:
        self._refresh_entries()
        self._select_entry_name(info.name)
        self._show_entry_paths(info)
        self._refresh_context_list()
        self._set_project_status(f"Opened entry: {info.name}")

    def _on_switch_clicked(self) -> None:
        item = self._ctx_list.currentItem()
        if item is None:
            self._set_project_status("Select a working point first", error=True)
            return
        label = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        try:
            self._ctrl.use_point(PointUseRequest(label))
        except Exception as exc:
            # Keep owner rejection visible without escaping the Qt callback.
            logger.exception("Failed to switch working point")
            self._set_project_status(
                f"Failed to switch working point: {exc}", error=True
            )
            return
        self._refresh_context_list()
        self._set_project_status(f"Switched to working point: {label}")

    def _on_new_ctx_clicked(self) -> None:
        request = PointCreateRequest(
            label=self._label_edit.text(),
            description=self._description_edit.text(),
            origin=CommandOrigin("gui"),
            clone_from=self._clone_combo.currentData(),
        )
        try:
            self._ctrl.new_point(request)
        except Exception as exc:
            # PointBindingError may retain a created resource; do not retry New.
            logger.exception("Failed to create working point")
            self._refresh_context_list()
            self._set_project_status(
                f"Failed to create working point: {exc}", error=True
            )
            return
        self._refresh_context_list()
        self._clear_new_point_form()
        self._set_project_status(f"Created working point: {request.label}")

    def _clear_new_point_form(self) -> None:
        self._label_edit.clear()
        self._label_edit.setModified(False)
        self._description_edit.clear()
        self._on_device_changed(self._device_combo.currentText())

    def _refresh_context_list(self) -> None:
        self._ctx_list.clear()
        points = self._ctrl.list_points()
        current = self._ctrl.current_point_label()
        self._point_descriptions = {p.label: p.description for p in points}
        current_item: QListWidgetItem | None = None
        for point in points:
            is_current = point.label == current
            item = QListWidgetItem(f"● {point.label}" if is_current else point.label)
            item.setData(Qt.ItemDataRole.UserRole, point.label)
            item.setToolTip(point.description)
            if is_current:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                current_item = item
            self._ctx_list.addItem(item)
        if current_item is None:
            self._ctx_list.setCurrentRow(-1)
        else:
            self._ctx_list.setCurrentItem(current_item)
        self._show_selected_description(current_item, None)
        self._refresh_clone_combo([point.label for point in points])

    def _show_selected_description(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            self._selected_description.setText("")
            return
        description = self._point_descriptions[current.data(Qt.ItemDataRole.UserRole)]
        if description:
            self._selected_description.setText(f"Description: {description}")
            self._selected_description.setStyleSheet("")
        else:
            self._selected_description.setText("No description")
            self._selected_description.setStyleSheet("color: gray;")

    def _refresh_clone_combo(self, labels: list[str]) -> None:
        # "(empty)" uses the owner's setup seed and empty library, not unsaved session parameters.
        prev: str | None = self._clone_combo.currentData()
        self._clone_combo.blockSignals(True)
        self._clone_combo.clear()
        self._clone_combo.addItem("(empty)", userData=None)
        for label in labels:
            self._clone_combo.addItem(label, userData=label)
        self._clone_combo.blockSignals(False)
        idx = self._clone_combo.findData(prev) if prev is not None else 0
        self._clone_combo.setCurrentIndex(max(idx, 0))

    def _set_project_status(self, msg: str, *, error: bool = False) -> None:
        self._show_status(self._project_status, msg, "red" if error else "green")

    def _set_project_hint(self, msg: str) -> None:
        self._show_status(self._project_status, msg, "gray")

    @staticmethod
    def _show_status(label: QLabel, msg: str, color: str) -> None:
        label.setText(_breakable(msg))
        label.setToolTip(msg)
        label.setStyleSheet(f"color: {color};")

    # ------------------------------------------------------------------
    # Connection panel handlers
    # ------------------------------------------------------------------

    def _maybe_show_current_cfg(self) -> None:
        soccfg = self._ctrl.get_soccfg()
        if isinstance(soccfg, SocConfigProtocol):
            self._show_cfg(describe_soc(soccfg))
            self._set_conn_status("Currently connected", error=False)

    def _on_connect_clicked(self) -> None:
        from zcu_tools.gui.session.services.connection import ConnectRemoteRequest

        self._ctrl.bind_connection_outcome(
            self._on_connect_finished, self._on_connect_failed
        )
        self._ctrl.remember_connection(
            ConnectionPreferences(
                ip=self._ip_edit.text().strip(), port=self._port_spin.value()
            )
        )
        self._connect_btn.setEnabled(False)
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
        self._connect_btn.setEnabled(True)
        self._set_conn_status("Connected", error=False)
        soccfg = self._ctrl.get_soccfg()
        if isinstance(soccfg, SocConfigProtocol):
            self._show_cfg(describe_soc(soccfg))

    def _on_connect_failed(self, message: str) -> None:
        self._connect_btn.setEnabled(True)
        self._set_conn_status(message, error=True)

    def _show_cfg(self, text: str) -> None:
        self._cfg_text.setPlainText(text)
        self._cfg_text.setVisible(True)

    def _set_conn_status(self, msg: str, *, error: bool = False) -> None:
        self._show_status(self._conn_status, msg, "red" if error else "green")

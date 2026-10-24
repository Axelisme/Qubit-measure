"""SetupDialog — explicit result entry and working-point binding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QColor, QDesktopServices, QResizeEvent, QShowEvent
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
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
from zcu_tools.gui.widgets import (
    error_colour,
    reset_foreground,
    secondary_colour,
    set_foreground,
    success_colour,
)
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

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


class SetupDialog(QDialog):
    """Resizable result-entry and working-point form over SetupControlPort.

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
        self._read_only = False
        self.setWindowTitle("Setup")
        self.resize(620, 640)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self._build_entry_group())
        root_layout.addWidget(self._build_point_list(), stretch=1)
        root_layout.addWidget(self._build_new_point_group())

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

        # EventBus subscriptions for live updates
        from zcu_tools.gui.session.events import (
            ContextSwitchedPayload,
            DeviceChangedPayload,
        )

        bus = controller.get_bus()
        bus.subscribe(ContextSwitchedPayload, self._on_bus_context_switched)
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
        self._entry_combo.setMinimumWidth(160)
        entry_row.addWidget(self._entry_combo, stretch=1)
        refresh_entries_btn = QPushButton("↻")
        refresh_entries_btn.setFixedWidth(28)
        refresh_entries_btn.setToolTip("Refresh result entries")
        refresh_entries_btn.clicked.connect(
            lambda checked=False: self._refresh_entries(refresh=True)
        )
        entry_row.addWidget(refresh_entries_btn)
        self._new_entry_btn = QPushButton("New entry…")
        self._new_entry_btn.clicked.connect(self._on_new_entry_clicked)
        entry_row.addWidget(self._new_entry_btn)
        self._open_entry_btn = QPushButton("Open")
        self._open_entry_btn.clicked.connect(self._on_open_entry_clicked)
        entry_row.addWidget(self._open_entry_btn)
        entry_form.addRow("Entry:", entry_row)
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
        switch_row = QHBoxLayout()
        switch_row.addWidget(self._selected_description, stretch=1)
        self._switch_btn = QPushButton("Switch to selected")
        self._switch_btn.clicked.connect(self._on_switch_clicked)
        switch_row.addWidget(self._switch_btn)
        ctx_layout.addLayout(switch_row)
        return ctx_group

    def _build_new_point_group(self) -> QGroupBox:
        # new context
        new_group = QGroupBox("New working point")
        new_form = QGridLayout(new_group)
        new_form.setColumnStretch(1, 1)
        new_form.setColumnStretch(3, 1)

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
        track_label = QLabel("Track device:")
        track_label.setBuddy(self._device_combo)
        new_form.addWidget(track_label, 0, 0)
        new_form.addLayout(device_row, 0, 1)

        self._label_edit = QLineEdit()
        self._label_edit.setObjectName("point_label")
        label_label = QLabel("Label:")
        label_label.setBuddy(self._label_edit)
        new_form.addWidget(label_label, 0, 2)
        new_form.addWidget(self._label_edit, 0, 3, 1, 2)

        self._clone_combo = QComboBox()
        self._clone_combo.setObjectName("clone_from")
        self._clone_combo.setMinimumWidth(160)
        clone_label = QLabel("Clone from:")
        clone_label.setBuddy(self._clone_combo)
        new_form.addWidget(clone_label, 1, 0)
        new_form.addWidget(self._clone_combo, 1, 1)

        self._description_edit = QLineEdit()
        self._description_edit.setObjectName("point_description")
        description_label = QLabel("Description:")
        description_label.setBuddy(self._description_edit)
        new_form.addWidget(description_label, 1, 2)
        new_form.addWidget(self._description_edit, 1, 3)

        self._new_ctx_btn = QPushButton("Create")
        self._new_ctx_btn.clicked.connect(self._on_new_ctx_clicked)
        new_form.addWidget(self._new_ctx_btn, 1, 4)
        return new_group

    def set_read_only(self, *, read_only: bool) -> None:
        """Gate entry and point edits while preserving browsing and Close.

        The host owns this policy; cached event refreshes preserve it.
        """
        self._read_only = read_only
        for button in (
            self._new_entry_btn,
            self._open_entry_btn,
            self._new_ctx_btn,
            self._switch_btn,
        ):
            button.setEnabled(not read_only)
        for field in (self._label_edit, self._description_edit):
            field.setReadOnly(read_only)
        for combo in (self._entry_combo, self._device_combo, self._clone_combo):
            combo.setEnabled(not read_only)

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        if not self._bus_subs_active:
            logger.debug("_cleanup_bus_subscriptions called but already inactive")
            return
        from zcu_tools.gui.session.events import (
            ContextSwitchedPayload,
            DeviceChangedPayload,
        )

        bus = self._ctrl.get_bus()
        bus.unsubscribe(ContextSwitchedPayload, self._on_bus_context_switched)
        bus.unsubscribe(DeviceChangedPayload, self._on_bus_device_changed)
        self._bus_subs_active = False

    def showEvent(self, a0: QShowEvent | None) -> None:
        """Re-seed entry inputs from cached preferences on show.

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
            self._refresh_entries()
            self._select_entry_name(info.name)
            self._show_entry_paths(info)
            self._set_project_status(f"Opened entry: {info.name}")

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
        self._set_project_hint("")
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
        if self._read_only:
            return
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
        if self._read_only:
            return
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
        if self._read_only:
            return
        item = self._ctx_list.currentItem()
        if item is None:
            self._set_project_status("Select a working point first", error=True)
            return
        label = item.data(Qt.ItemDataRole.UserRole)
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
        if self._read_only:
            return
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
            reset_foreground(self._selected_description)
        else:
            self._selected_description.setText("No description")
            set_foreground(self._selected_description, secondary_colour(self))

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
        self._show_status(
            self._project_status,
            msg,
            error_colour(self) if error else success_colour(self),
        )

    def _set_project_hint(self, msg: str) -> None:
        self._show_status(self._project_status, msg, secondary_colour(self))

    @staticmethod
    def _show_status(label: QLabel, msg: str, color: QColor) -> None:
        label.setText(_breakable(msg))
        label.setToolTip(msg)
        set_foreground(label, color)

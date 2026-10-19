"""DeviceDialog — register devices and inspect/control selected device."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.session.devices import device_adapters, get_device_adapter
from zcu_tools.gui.session.devices.base import DevicePanelProtocol
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DeviceSnapshot,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.ui.progress_stack import ProgressStack
from zcu_tools.gui.widgets import (
    accent_colour,
    error_colour,
    secondary_colour,
    set_foreground,
    success_colour,
)
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

if TYPE_CHECKING:
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.parameters import ParameterContext


logger = logging.getLogger(__name__)

_ACTIVITY_MARKER_PREFIX = "● "
_ACTIVITY_MARKER_ROLE = Qt.ItemDataRole.UserRole + 1
_ACTIVITY_MARKER_TOOLTIP = "Operating or recently active"


class _MemoryDevicePanel(QWidget):
    """Read-only info panel for a remembered-but-not-connected device."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)

        self._type_label = QLabel()
        make_label_selectable(self._type_label)
        form.addRow("Type:", self._type_label)

        self._name_label = QLabel()
        make_label_selectable(self._name_label)
        form.addRow("Name:", self._name_label)

        self._addr_label = QLabel()
        make_label_selectable(self._addr_label)
        form.addRow("Address:", self._addr_label)

        note = QLabel("Not connected. Press Reconnect to connect.")
        set_foreground(note, secondary_colour(self))
        form.addRow(note)

    def load_memory(self, type_name: str, name: str, address: str) -> None:
        self._type_label.setText(type_name)
        self._name_label.setText(name)
        self._addr_label.setText(address or "(none)")


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------


class DeviceDialog(QDialog):
    """Edit device settings through the supplied DeviceControlPort.

    parameter_provider returns one fixed ParameterContext for each preview or
    expression-bearing Apply. None permits pure calculations but not parameter
    paths. Direct-only Apply does not call it. Provider or expression failures
    display an error and prevent setup; valid Apply delegates setup to device.
    before_stop, when supplied, permits Stop in inspection mode and runs before
    the cancellation request. The application uses it to reclaim authority;
    this shared dialog does not own the holder or decide control transitions.
    """

    def __init__(
        self,
        device: DeviceControlPort,
        parameter_provider: Callable[[], ParameterContext | None],
        parent: QWidget | None = None,
        *,
        before_stop: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._before_stop = before_stop
        self._dev = device
        self._parameter_provider = parameter_provider
        self._read_only = False
        self.setWindowTitle("Manage Hardware Devices")
        self.resize(800, 500)

        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_device_list())

        splitter.addWidget(self._build_device_details())
        splitter.setSizes([300, 500])

        # Status & Progress
        self._progress = ProgressStack()
        layout.addWidget(self._progress)

        self._event_unsubs = [
            self._dev.on_device_changed(self._on_device_changed),
            self._dev.on_device_setup_started(self._on_setup_started),
            self._dev.on_device_setup_finished(self._on_setup_finished),
        ]
        self._event_subs_active = True
        # Phase C: concurrent setups. Progress is now multi-owner — every device
        # currently setting up has its own ProgressService subscription, keyed by
        # device name. _sync_progress_subscriptions diffs the live setup set
        # against these on every setup start/finish; _on_progress_changed merges
        # all owners' bars into the single ProgressStack.
        self._progress_unsubs: dict[str, Callable[[], None]] = {}
        self.finished.connect(self._cleanup_event_subscriptions)
        self.destroyed.connect(self._cleanup_event_subscriptions)

        # Dialog-scoped + selection-scoped live poller (Phase 2): while the
        # dialog is visible AND a device is selected, tick once a second and
        # best-effort off-main read the selected device's real driver values,
        # including setup/ramp current-value refreshes. The result flows
        # back through DEVICE_CHANGED (only when it actually moved), which the
        # Phase 1 repaint+preserve-selection path absorbs. Built stopped;
        # _update_poll_timer (re)decides on every selection change and on
        # show/hide.
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._on_poll_tick)
        # Activity is a lock-free observation, independent of selected-value I/O.
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(200)
        self._activity_timer.timeout.connect(self._refresh_activity_markers)

        # Tracks the device name most recently loaded into the right panel. Used
        # in _on_selection_changed to distinguish a poll repaint of the SAME device
        # (name unchanged → preserve eval expressions, R3) from a device SWITCH
        # (name changed → reset eval fields to direct mode before loading).
        self._last_panel_device_name: str | None = None

        self._refresh_list()
        # Phase C: subscribe progress for every device already setting up (the
        # dialog may open mid-setup, possibly with several concurrent setups).
        self._sync_progress_subscriptions()

    def _build_device_list(self) -> QWidget:
        # --- Left side: List + Management ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_selection_changed)
        left_layout.addWidget(QLabel("Registered Devices:"))
        self._empty_hint = QLabel("No devices registered")
        self._empty_hint.setObjectName("devices_empty_hint")
        set_foreground(self._empty_hint, secondary_colour(self))
        left_layout.addWidget(self._empty_hint)
        left_layout.addWidget(self._list, stretch=1)
        # Drop acts on the selected list row, so it sits directly under the list.
        self._drop_btn = QPushButton("Drop")
        self._drop_btn.setStyleSheet("color: red;")
        self._drop_btn.clicked.connect(self._on_forget_clicked)
        drop_row = QHBoxLayout()
        drop_row.addWidget(self._drop_btn)
        drop_row.addStretch()
        left_layout.addLayout(drop_row)

        self._rename_edit = QLineEdit()
        self._rename_edit.setObjectName("device_rename_name")
        self._rename_edit.setPlaceholderText("New name")
        self._rename_btn = QPushButton("Rename")
        self._rename_btn.clicked.connect(self._on_rename_clicked)
        rename_row = QHBoxLayout()
        rename_row.addWidget(self._rename_edit)
        rename_row.addWidget(self._rename_btn)
        left_layout.addLayout(rename_row)

        # Add device form
        self._add_box = QGroupBox("Register New Device")
        add_form = QFormLayout(self._add_box)

        self._type_combo = QComboBox()
        self._type_combo.addItems(
            adapter.driver_cls.__name__
            for adapter in device_adapters()
            if adapter.capabilities.user_connectable
        )
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        add_form.addRow("Type:", self._type_combo)

        self._name_edit = QLineEdit()
        add_form.addRow("Name:", self._name_edit)

        self._addr_edit = QLineEdit()
        self._addr_edit.setPlaceholderText("TCPIP::192.168.1.1::INSTR")
        add_form.addRow("Address:", self._addr_edit)

        self._add_btn = QPushButton("Add Device")
        self._add_btn.clicked.connect(self._on_add_clicked)
        add_form.addRow(self._add_btn)

        self._add_status = QLabel("")
        make_label_selectable(self._add_status)
        self._add_status.setWordWrap(True)
        add_form.addRow(self._add_status)

        left_layout.addWidget(self._add_box)
        return left_widget

    def _build_device_details(self) -> QWidget:
        # --- Right side: Detail Panel Stack ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Panels receive parameter_provider so their EvalNumericField widgets can show
        # a live ghost preview while the user composes an expression.
        self._stack = QStackedWidget()
        self._device_panels: dict[str, QWidget] = {}
        self._stack.addWidget(QLabel("Select a device to configure."))  # Page 0: Idle
        self._memory_panel = _MemoryDevicePanel()
        self._stack.addWidget(self._memory_panel)
        right_layout.addWidget(self._stack, stretch=1)

        # Bottom buttons for right side
        btn_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.clicked.connect(self._on_apply_or_stop_clicked)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row.addStretch()
        btn_row.addWidget(self._refresh_btn)
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(close_btn)
        right_layout.addLayout(btn_row)

        return right_widget

    def set_read_only(self, *, read_only: bool) -> None:
        """Switch the dialog to inspection mode without blocking read paths."""
        self._read_only = bool(read_only)
        self._refresh_button_states(self._selected_device_name())

    def _cleanup_event_subscriptions(self, *_args: object) -> None:
        if not self._event_subs_active:
            return
        for unsubscribe in reversed(self._event_unsubs):
            unsubscribe()
        self._event_unsubs.clear()
        self._detach_all_progress()
        self._poll_timer.stop()
        self._activity_timer.stop()
        self._event_subs_active = False

    # --- live-value poller (dialog-scoped + selection-scoped) ----------------

    def showEvent(self, a0: Any) -> None:
        super().showEvent(a0)
        self._update_poll_timer()
        self._refresh_activity_markers()
        if self._event_subs_active:
            self._activity_timer.start()

    def hideEvent(self, a0: Any) -> None:
        super().hideEvent(a0)
        self._update_poll_timer()
        self._activity_timer.stop()

    def _update_poll_timer(self) -> None:
        """(Re)decide whether the live poller should be running.

        Dialog-scoped + selection-scoped: poll only while the dialog is visible
        AND a device is selected. A memory-only / transient selection still
        keeps the timer running: DeviceService cheaply skips non-live and
        non-setup mutation states, while setup/ramp can still refresh current
        values. The timer stops when the dialog hides or nothing is selected,
        so there is no always-on / all-device polling and no leak after close.
        """
        should_run = self.isVisible() and self._selected_device_name() is not None
        if should_run:
            if not self._poll_timer.isActive():
                self._poll_timer.start()
        elif self._poll_timer.isActive():
            self._poll_timer.stop()

    def _on_poll_tick(self) -> None:
        name = self._selected_device_name()
        if name is None:
            # Selection vanished between decisions — stop rather than poll None.
            self._poll_timer.stop()
            return
        self._dev.try_poll_device_info(name)

    def _refresh_list(self, select_name: str | None = None) -> None:
        # Rebuilding the list clears the selection; preserve the user's current
        # selection across the rebuild so an unrelated device change (e.g. a poll
        # tick or another device's status update) never steals focus from the
        # device the user is inspecting. ``select_name`` is an explicit override
        # for the cases that *want* to move the selection (e.g. surfacing a newly
        # connected device when nothing is selected yet).
        current_name = self._selected_device_name()
        self._list.clear()
        entries = self._dev.list_devices()
        for entry in entries:
            # DeviceEntry and DeviceSnapshot are both State projections. Prefer
            # the snapshot so a reopened dialog and every repaint read the
            # authoritative per-device status directly; the entry status keeps
            # lightweight test/fallback ports usable when no snapshot exists.
            snapshot = self._dev.get_device_snapshot(entry.name)
            status = (
                snapshot.status if snapshot is not None else DeviceStatus(entry.status)
            )
            is_live = status in {
                DeviceStatus.CONNECTED,
                DeviceStatus.DISCONNECTING,
                DeviceStatus.SETTING_UP,
            }
            label = f"{entry.name} ({entry.type_name})"
            if is_live:
                item = QListWidgetItem(label)
            else:
                item = QListWidgetItem(f"{label} [not connected]")
                item.setForeground(secondary_colour(self))
            item.setData(Qt.ItemDataRole.UserRole, entry.name)  # type: ignore[attr-defined]
            self._list.addItem(item)
        self._empty_hint.setVisible(self._list.count() == 0)
        self._refresh_activity_markers()

        target = select_name if select_name is not None else current_name
        if target is not None:
            for row in range(self._list.count()):
                it = self._list.item(row)
                if it is not None and it.data(Qt.ItemDataRole.UserRole) == target:  # type: ignore[attr-defined]
                    self._list.setCurrentRow(row)
                    break

        self._on_selection_changed(self._list.currentRow())
        # Keep the submitted name while connecting or failed so editing only
        # the address retries that device. Suggest a new name after success.
        name = self._name_edit.text()
        if not name or any(
            entry.name == name and entry.status == DeviceStatus.CONNECTED.value
            for entry in entries
        ):
            dtype = self._type_combo.currentText()
            existing = {e.name for e in entries}
            self._name_edit.setText(self._unique_name(dtype.lower(), existing))

    def _refresh_activity_markers(self) -> None:
        activity = self._dev.activity_snapshot()
        now = time.monotonic()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            observed = activity.get(item.data(Qt.ItemDataRole.UserRole))
            active = observed is not None and (
                observed.in_operation
                or (
                    observed.last_finished_monotonic is not None
                    and now - observed.last_finished_monotonic < 1.0
                )
            )
            marked = item.data(_ACTIVITY_MARKER_ROLE) is True
            if active == marked:
                continue
            item.setData(_ACTIVITY_MARKER_ROLE, active)
            if active:
                item.setText(f"{_ACTIVITY_MARKER_PREFIX}{item.text()}")
                item.setForeground(accent_colour(self))
                item.setToolTip(_ACTIVITY_MARKER_TOOLTIP)
            else:
                item.setText(item.text().removeprefix(_ACTIVITY_MARKER_PREFIX))
                item.setForeground(
                    secondary_colour(self)
                    if item.text().endswith(" [not connected]")
                    else self._list.palette().text()
                )
                item.setToolTip("")

    def _selected_device_name(self) -> str | None:
        item = self._list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]

    def _on_selection_changed(self, _row: int) -> None:
        # Selection-scoped poller decision: a change in what is selected (or to
        # nothing selected) re-evaluates whether the live poller runs.
        self._update_poll_timer()
        item = self._list.currentItem()
        if item is None:
            self._stack.setCurrentIndex(0)
            # Per-device button refresh still runs; no device selected means all
            # action buttons are disabled regardless of setup state.
            self._refresh_button_states(None)
            return

        name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        snapshot = self._dev.get_device_snapshot(name)
        if snapshot is None:
            return

        # Recompute button states per-device: the setup lock is scoped to each
        # device's own setup status, not the whole dialog. Pass the already-
        # fetched snapshot to avoid a second controller call.
        self._refresh_button_states(name, snapshot)

        is_memory = snapshot.status is DeviceStatus.MEMORY_ONLY
        if is_memory:
            self._memory_panel.load_memory(
                snapshot.type_name, snapshot.name, snapshot.address
            )
            self._stack.setCurrentWidget(self._memory_panel)
            self._apply_current_panel_editing_state(snapshot)
            return

        info = snapshot.info
        if info is None:
            return

        adapter = get_device_adapter(info.type)
        if adapter is None:
            self._stack.setCurrentIndex(0)
            return
        panel = self._device_panels.get(info.type)
        if panel is None:
            panel = adapter.panel_cls(self._parameter_provider)
            self._device_panels[info.type] = panel
            self._stack.addWidget(panel)
        self._stack.setCurrentWidget(panel)
        if isinstance(panel, DevicePanelProtocol):
            # R3: reset eval fields when a different device is loaded into this panel
            # so expressions from one device do not persist into another device's context.
            # Same-device repaints (poll tick, status update) preserve the expression.
            if name != self._last_panel_device_name:
                panel.reset_eval_fields()
            self._last_panel_device_name = name
            panel.load(info)
            self._apply_current_panel_editing_state(snapshot)

    def _setup_device_names(self) -> set[str]:
        """The set of devices currently setting up (Phase C, possibly several).

        Derived from per-device snapshot status (SETTING_UP): DeviceService runs
        setups for different devices concurrently, and its State-owned status is
        the SSOT for "which devices are mid-setup". The remote contract exposes
        the same set via the ``device.active_setups`` enumerator, but the dialog
        reads snapshot status directly rather than going through it."""
        names: set[str] = set()
        for entry in self._dev.list_devices():
            snapshot = self._dev.get_device_snapshot(entry.name)
            if snapshot is not None and snapshot.status is DeviceStatus.SETTING_UP:
                names.add(entry.name)
        return names

    def _refresh_button_states(
        self,
        selected_name: str | None,
        snapshot: DeviceSnapshot | None = None,
    ) -> None:
        """Recompute button states for the currently-selected device (Phase C).

        Per-device locking: the lock is scoped to *each* device's own setup, and
        setups of different devices run concurrently. The selected device is
        locked (Apply -> red Stop, edit fields disabled) only when *it* is
        setting up; a different device being set up no longer blocks this one —
        the user can Apply it to start a concurrent setup.

        ``snapshot`` may be supplied by the caller (e.g. _on_selection_changed)
        to avoid a redundant controller call; if omitted and selected_name is
        not None, it is fetched once here.
        """
        if selected_name is None:
            # Nothing selected — all device-action buttons disabled. The Add-box
            # stays enabled: registering a new device is a connect on a fresh
            # name, which never conflicts with an in-flight mutation (the gate is
            # resource-aware in Phase C).
            self._drop_btn.setEnabled(False)
            self._drop_btn.setText("Drop")
            self._rename_edit.setEnabled(False)
            self._rename_btn.setEnabled(False)
            self._rename_btn.setToolTip("Select a disconnected device to rename.")
            self._refresh_btn.setEnabled(False)
            self._apply_btn.setEnabled(False)
            self._apply_btn.setText("Apply Changes")
            self._apply_btn.setStyleSheet("")
            self._apply_btn.setToolTip("")
            self._set_add_box_enabled(enabled=True)
            self._apply_current_panel_editing_state(None)
            return

        if snapshot is None:
            snapshot = self._dev.get_device_snapshot(selected_name)
        is_memory = snapshot is not None and snapshot.status is DeviceStatus.MEMORY_ONLY
        is_busy = snapshot is not None and snapshot.status not in {
            DeviceStatus.MEMORY_ONLY,
            DeviceStatus.CONNECTED,
        }
        is_setting_up = snapshot is not None and (
            snapshot.status is DeviceStatus.SETTING_UP
        )

        # --- Add-box: always enabled (a new-name connect never conflicts) ---
        self._set_add_box_enabled(enabled=True)

        # --- Drop button ---
        # Dropping the selected device disconnects it; that conflicts with its
        # OWN in-flight mutation (a busy device, incl. setting up) but not with a
        # different device's setup. So: enable iff the selected device is idle.
        self._drop_btn.setEnabled(not self._read_only and not is_busy)
        self._drop_btn.setText("Forget" if is_memory else "Drop")

        can_rename = not self._read_only and is_memory
        self._rename_edit.setEnabled(can_rename)
        self._rename_btn.setEnabled(can_rename)
        self._rename_btn.setToolTip(
            "Rename the disconnected device. Aliases are not updated."
            if is_memory
            else "Disconnect the device before renaming."
        )

        # --- Refresh button ---
        # refresh reads get_device_info; the read is gate-guarded against the
        # selected device's own mutation, so disable it only when the selected
        # device is memory-only or busy (incl. its own setup). A different
        # device's concurrent setup is irrelevant here.
        self._refresh_btn.setEnabled(not is_memory and not is_busy)

        # --- Apply button ---
        if is_setting_up:
            # The selected device is mid-setup: show its red Stop button.
            self._apply_btn.setEnabled(
                not self._read_only or self._before_stop is not None
            )
            self._apply_btn.setText("Stop")
            self._apply_btn.setStyleSheet("color: red;")
            self._apply_btn.setToolTip("Cancel the in-progress setup.")
        else:
            # Idle (or another device setting up — no longer blocks this one):
            # enable Apply based on the selected device's own state.
            self._apply_btn.setEnabled(not self._read_only and not is_busy)
            self._apply_btn.setText("Reconnect" if is_memory else "Apply Changes")
            self._apply_btn.setStyleSheet("")
            self._apply_btn.setToolTip("")
        self._apply_current_panel_editing_state(snapshot)

    def _set_add_box_enabled(self, *, enabled: bool) -> None:
        effective = bool(enabled) and not self._read_only
        self._add_box.setEnabled(effective)
        self._type_combo.setEnabled(effective)
        self._name_edit.setEnabled(effective)
        self._addr_edit.setEnabled(effective)
        self._add_btn.setEnabled(effective)

    def _apply_current_panel_editing_state(
        self, snapshot: DeviceSnapshot | None
    ) -> None:
        panel = self._stack.currentWidget()
        if not isinstance(panel, DevicePanelProtocol):
            return
        panel.set_editing_enabled(
            enabled=not self._read_only
            and snapshot is not None
            and snapshot.status is DeviceStatus.CONNECTED
        )

    @staticmethod
    def _unique_name(base: str, existing: set[str]) -> str:
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def _on_type_changed(self, dtype: str) -> None:
        existing = {e.name for e in self._dev.list_devices()}
        self._name_edit.setText(self._unique_name(dtype.lower(), existing))

    def _on_add_clicked(self) -> None:
        if self._read_only:
            return
        dtype = self._type_combo.currentText()
        name = self._name_edit.text().strip() or dtype.lower()
        addr = self._addr_edit.text().strip()
        self._add_status.setText("")

        self._dev.start_connect_device(
            ConnectDeviceRequest(type_name=dtype, name=name, address=addr)
        )
        set_foreground(self._add_status, secondary_colour(self))
        self._add_status.setText(f"Connecting {name}...")

    def _on_rename_clicked(self) -> None:
        if self._read_only:
            return
        name = self._selected_device_name()
        if name is None:
            return
        new_name = self._rename_edit.text()
        try:
            self._dev.rename_device(name, new_name)
        except ExpectedError as exc:
            self._show_error(str(exc))
            return
        self._refresh_list(select_name=new_name)
        self._rename_edit.clear()
        set_foreground(self._add_status, success_colour(self))
        self._add_status.setText(f"Device name: {new_name}. Aliases are unchanged.")

    def _on_forget_clicked(self) -> None:
        if self._read_only:
            return
        item = self._list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if self._dev.is_memory_device(name):
            # Remove from memory entirely — won't appear after restart
            self._dev.forget_device(name)
        else:
            # Disconnect only — keep as a remembered device so it reappears as gray on next launch
            self._dev.start_disconnect_device(DisconnectDeviceRequest(name=name))

    def _on_refresh_clicked(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
            self._dev.get_device_info(name)
        self._on_selection_changed(self._list.currentRow())

    def _on_apply_or_stop_clicked(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]

        # Phase C: the button means Stop only for the selected device's OWN
        # in-flight setup (the same device that shows the red Stop). A different
        # device's concurrent setup never makes this button a Stop.
        snapshot = self._dev.get_device_snapshot(name)
        if snapshot is not None and snapshot.status is DeviceStatus.SETTING_UP:
            if self._read_only and self._before_stop is None:
                return
            if self._before_stop is not None:
                self._before_stop()
            self._dev.cancel_device_operation(name)
            return

        if self._read_only:
            return
        if self._dev.is_memory_device(name):
            self._add_status.setText("")
            self._dev.start_reconnect_device(name)
            set_foreground(self._add_status, secondary_colour(self))
            self._add_status.setText(f"Reconnecting {name}...")
            return

        self._apply_panel(name)

    def _apply_panel(self, name: str) -> None:
        panel = self._stack.currentWidget()
        if not isinstance(panel, DevicePanelProtocol):
            return

        updates = panel.read()

        resolved_updates = self._resolve_panel_updates(name, updates)
        if resolved_updates is None:
            return

        info = self._dev.get_device_info(name)
        if info is None:
            return
        new_info = info.with_updates(**resolved_updates)
        self._dev.start_setup_device(SetupDeviceRequest(name=name, info=new_info))

    def _resolve_panel_updates(
        self, name: str, updates: Mapping[str, object]
    ) -> dict[str, object] | None:
        """Resolve once for Apply; display input failures and return None to reject submission."""
        # Resolve any EvalRef markers against the current ParameterContext (Design 1:
        # apply-time resolve once, not per-keystroke). Only touch the controller's
        # ParameterContext when at least one field is in eval mode — direct-only applies
        # do not depend on an active context.
        from zcu_tools.gui.session.expression import (
            EvalRef,
            coerce_eval_result,
            evaluate_numeric_expr,
        )

        if any(isinstance(v, EvalRef) for v in updates.values()):
            try:
                parameters = self._parameter_provider()
            except Exception as exc:
                self._show_error(f"{name}: {exc}")
                return None
            resolved_updates: dict[str, object] = {}
            for key, val in updates.items():
                if isinstance(val, EvalRef):
                    try:
                        coerced = coerce_eval_result(
                            evaluate_numeric_expr(val.expr, parameters), val.type_
                        )
                    except Exception as exc:
                        self._show_error(f"{name}: field '{key}': {exc}")
                        return None
                    # Bounds check: eval must respect the same inclusive range as
                    # the direct spinbox path — prevents e.g. rampstep=0 reaching
                    # the driver where it would cause ZeroDivisionError.
                    if not (val.minimum <= float(coerced) <= val.maximum):
                        self._show_error(
                            f"{name}: field '{key}': value {coerced} out of range"
                            f" [{val.minimum}, {val.maximum}]"
                        )
                        return None
                    resolved_updates[key] = coerced
                else:
                    resolved_updates[key] = val
        else:
            # All fields are direct values — pass through unchanged, no parameters access.
            resolved_updates = dict(updates)

        return resolved_updates

    def _show_error(self, message: str) -> None:
        set_foreground(self._add_status, error_colour(self))
        self._add_status.setText(message)

    def _on_device_changed(self, payload: DeviceChangedPayload) -> None:
        name = payload.name
        # Keep the user's current selection across the refresh; only surface the
        # changed device when nothing is selected yet (e.g. the first device just
        # connected). When the changed device IS the selected one, _refresh_list's
        # trailing _on_selection_changed repaints the right panel with the fresh
        # cached info, so a value that moved underneath us shows up immediately.
        select = name if self._selected_device_name() is None else None
        self._refresh_list(select_name=select)
        if name is None:
            return
        snapshot = self._dev.get_device_snapshot(name)
        if snapshot is None:
            return
        if snapshot.error is not None:
            self._show_error(snapshot.error)
        elif snapshot.status is DeviceStatus.CONNECTED:
            set_foreground(self._add_status, success_colour(self))
            self._add_status.setText(f"Connected {name}.")
        elif snapshot.status is DeviceStatus.MEMORY_ONLY:
            set_foreground(self._add_status, secondary_colour(self))
            self._add_status.setText(f"Disconnected {name}.")
        elif snapshot.status in {
            DeviceStatus.CONNECTING,
            DeviceStatus.DISCONNECTING,
        }:
            set_foreground(self._add_status, secondary_colour(self))
            self._add_status.setText(
                f"{snapshot.status.value.replace('_', ' ').title()}: {name}..."
            )

    def _on_setup_started(self, payload: DeviceSetupStartedPayload) -> None:
        del payload
        # The start event is also a repaint boundary: derive every row from the
        # State-owned status so a dialog opened before the event shows the marker
        # without maintaining a second busy cache.
        self._refresh_list()
        self._sync_progress_subscriptions()
        # A concurrent setup may have started on a device other than the selected
        # one; recompute the selected device's buttons so e.g. its Apply stays
        # available (Phase C no longer globally blocks Apply during any setup).
        self._refresh_button_states(self._selected_device_name())

    def _on_setup_finished(self, payload: DeviceSetupFinishedPayload) -> None:
        del payload
        # Rebuild the list from State on every terminal outcome. The selected
        # device is preserved by _refresh_list, while each row clears or keeps its
        # own marker independently.
        self._refresh_list()
        self._sync_progress_subscriptions()

    def _sync_progress_subscriptions(self) -> None:
        """Diff the live setup set against current subscriptions (Phase C).

        Every device currently setting up gets its own ProgressService
        subscription (keyed by device name); subscriptions for devices that have
        finished are disposed. The single ProgressStack then shows the merged
        bars of all live setups. Subscriptions survive a dialog reopen because
        the container lives in ProgressService, not here."""
        live = self._setup_device_names()
        # Surface a setup owner only when nothing is selected (dialog opened
        # mid-setup with no prior selection); otherwise keep the user's choice.
        if self._selected_device_name() is None and live:
            owner = next(iter(sorted(live)))
            for row in range(self._list.count()):
                item = self._list.item(row)
                if (
                    item is not None and item.data(Qt.ItemDataRole.UserRole) == owner  # type: ignore[attr-defined]
                ):
                    self._list.setCurrentRow(row)
                    break
        # Subscribe newly-started setups.
        for name in live - set(self._progress_unsubs):
            self._progress_unsubs[name] = self._dev.attach_progress(
                name, self._on_progress_changed
            )
        # Dispose finished ones.
        for name in set(self._progress_unsubs) - live:
            self._progress_unsubs.pop(name)()
        if self._progress_unsubs:
            self._progress.show()
        else:
            self._progress.hide()
        self._on_progress_changed()  # render whatever is already live

    def _detach_all_progress(self) -> None:
        for dispose in self._progress_unsubs.values():
            dispose()
        self._progress_unsubs.clear()

    def _on_progress_changed(self) -> None:
        # Merge every subscribed owner's live bars into the single stack
        # (sorted by name so the order is stable across ticks).
        models: list[Any] = []
        for name in sorted(self._progress_unsubs):
            models.extend(m for _, m in self._dev.progress_bars(name))
        self._progress.render_models(tuple(models))

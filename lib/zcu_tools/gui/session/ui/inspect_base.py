"""InspectDialogBase — app-agnostic context inspector (components + ModuleLibrary).

The session-core half of the inspect dialog: the component tab (read + edit) and
the ModuleLibrary tab's *view / rename / delete*. App-specific actions live in
subclasses; this base exposes template-method hooks for the top toolbar and the
ModuleLibrary button row so the base never learns app-local concerns.

Depends only on ``ContextControlPort`` + the shared event bus, so every
measurement-session app can reuse it; measure adds ml-edit in its subclass.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import yaml
from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QFont  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    MdChangedPayload,
    MlChangedPayload,
    SessionPayload,
)
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.resources.context.library import LibrarySnapshot

logger = logging.getLogger(__name__)

_MONO_FONT = QFont("Monospace")
_MONO_FONT.setStyleHint(QFont.StyleHint.Monospace)


class InspectDialogBase(QDialog):
    """Non-modal dialog showing components and ModuleLibrary contents."""

    def __init__(
        self,
        ctrl: ContextControlPort,
        bus: BaseEventBus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._bus = bus
        self.setWindowTitle("Inspect Context")
        self.resize(700, 500)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.Window  # type: ignore[attr-defined]
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- top toolbar: status on the left, app actions + Refresh + Close on the right ---
        toolbar = QHBoxLayout()
        self._status_label = QLabel("No context")
        toolbar.addWidget(self._status_label)
        toolbar.addStretch()
        self._build_extra_toolbar_buttons(toolbar)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(self._refresh_btn)
        toolbar.addWidget(close_btn)
        layout.addLayout(toolbar)

        # --- tabs ---
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_md_tab(), "Parameters")
        self._tabs.addTab(self._build_ml_tab(), "Modules")
        layout.addWidget(self._tabs)

        # Subscribe to EventBus for auto-refresh
        bus.subscribe(ContextSwitchedPayload, self._on_bus_refresh)
        bus.subscribe(MdChangedPayload, self._on_bus_refresh)
        bus.subscribe(MlChangedPayload, self._on_bus_refresh)
        self._bus_subs_active = True
        self.finished.connect(self._cleanup_bus_subscriptions)
        self.destroyed.connect(self._cleanup_bus_subscriptions)

        self.refresh()

    # ------------------------------------------------------------------
    # Template-method hooks (the measure subclass adds ml create/modify)
    # ------------------------------------------------------------------

    def _build_extra_toolbar_buttons(self, toolbar: QHBoxLayout) -> None:
        """Hook: subclasses prepend app-specific toolbar buttons before Refresh."""
        del toolbar

    def _build_extra_ml_buttons(self, btn_layout: QHBoxLayout) -> None:
        """Hook: subclasses prepend ml create/modify buttons (default: none)."""
        del btn_layout

    def _on_ml_selection_changed(self, enabled: bool) -> None:
        """Hook: subclasses toggle their own ml-edit buttons (default: none)."""
        del enabled

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_md_tab(self) -> QWidget:
        self._component_tree = ComponentTreeWidget(self._ctrl)
        return self._component_tree

    def _build_ml_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)  # type: ignore[attr-defined]
        splitter.setChildrenCollapsible(False)

        # left: tree (modules / waveforms groups)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._ml_tree = QTreeWidget()
        self._ml_tree.setHeaderHidden(True)
        self._ml_tree.setRootIsDecorated(True)
        self._ml_tree.currentItemChanged.connect(self._on_ml_item_changed)
        left_layout.addWidget(self._ml_tree)

        btn_layout = QHBoxLayout()
        # The subclass injects Create/Modify (cfg-editor edit path) ahead of the
        # app-agnostic Rename/Delete buttons, preserving the original order.
        self._build_extra_ml_buttons(btn_layout)
        self._rename_ml_btn = QPushButton("Rename...")
        self._rename_ml_btn.setEnabled(False)
        self._del_ml_btn = QPushButton("Delete")
        self._del_ml_btn.setEnabled(False)
        btn_layout.addWidget(self._rename_ml_btn)
        btn_layout.addWidget(self._del_ml_btn)
        left_layout.addLayout(btn_layout)

        self._rename_ml_btn.clicked.connect(self._on_rename_ml_clicked)
        self._del_ml_btn.clicked.connect(self._on_delete_ml_clicked)

        splitter.addWidget(left_panel)

        # right: YAML text
        self._ml_text = QPlainTextEdit()
        self._ml_text.setReadOnly(True)
        self._ml_text.setFont(_MONO_FONT)
        splitter.addWidget(self._ml_text)

        splitter.setSizes([200, 450])
        layout.addWidget(splitter)
        return w

    # ------------------------------------------------------------------
    # Populate helpers
    # ------------------------------------------------------------------

    def _populate_ml(self, ml: LibrarySnapshot | None) -> None:
        # Remember current selection to restore after rebuild
        prev_item = self._ml_tree.currentItem()
        prev_data = (
            prev_item.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
            if prev_item is not None
            else None
        )

        self._ml_tree.blockSignals(True)
        self._ml_tree.clear()

        if ml is None:
            self._ml_tree.blockSignals(False)
            self._ml_text.setPlainText("")
            return

        bold = QFont()
        bold.setBold(True)

        restore_item: QTreeWidgetItem | None = None
        for group_label, store in [
            ("modules", ml.module_keys()),
            ("waveforms", ml.waveform_keys()),
        ]:
            if not store:
                continue
            group_item = QTreeWidgetItem(self._ml_tree, [group_label])
            group_item.setFont(0, bold)
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # type: ignore[attr-defined]
            group_item.setExpanded(True)
            for name in store:
                child = QTreeWidgetItem(group_item, [name])
                child.setData(0, Qt.ItemDataRole.UserRole, (group_label, name))  # type: ignore[attr-defined]
                if prev_data == (group_label, name):
                    restore_item = child

        self._ml_tree.blockSignals(False)

        if restore_item is not None:
            self._ml_tree.setCurrentItem(restore_item)
        else:
            self._ml_text.setPlainText("")

    def _on_ml_item_changed(
        self, current: QTreeWidgetItem | None, _previous: Any
    ) -> None:
        self._rename_ml_btn.setEnabled(False)
        self._del_ml_btn.setEnabled(False)
        self._on_ml_selection_changed(False)
        if current is None:
            self._ml_text.setPlainText("")
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if data is None:
            self._ml_text.setPlainText("")
            return

        self._rename_ml_btn.setEnabled(True)
        self._del_ml_btn.setEnabled(True)
        self._on_ml_selection_changed(True)

        group, name = data
        ml = self._ctrl.read_context().library
        value = ml.entry_value(
            name, kind="module" if group == "modules" else "waveform"
        )
        text = yaml.dump(value, allow_unicode=True, sort_keys=False)
        self._ml_text.setPlainText(text)

    def _current_ml_item_data(self) -> tuple[str, str] | None:
        current = self._ml_tree.currentItem()
        if current is None:
            return None
        data = current.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if data is None:
            return None
        group, name = data
        if group not in {"modules", "waveforms"}:
            raise RuntimeError(f"Unsupported ModuleLibrary group {group!r}")
        if not isinstance(name, str):
            raise RuntimeError(f"Invalid ModuleLibrary item name {name!r}")
        return group, name

    def _on_delete_ml_clicked(self) -> None:
        data = self._current_ml_item_data()
        if data is None:
            return

        group, name = data
        ans = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {group[:-1]} '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        if group == "modules":
            self._ctrl.del_ml_module(name)
        else:
            self._ctrl.del_ml_waveform(name)

    def _on_rename_ml_clicked(self) -> None:
        data = self._current_ml_item_data()
        if data is None:
            return
        group, name = data
        new, ok = QInputDialog.getText(
            self, "Rename", f"New name for {group[:-1]} '{name}':", text=name
        )
        new = new.strip()
        if not ok or not new or new == name:
            return
        try:
            if group == "modules":
                self._ctrl.rename_ml_module(name, new)
            else:
                self._ctrl.rename_ml_waveform(name, new)
        except Exception as exc:  # noqa: BLE001 — surface failure to the user
            QMessageBox.critical(self, "Rename failed", str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        if not self._bus_subs_active:
            logger.debug("_cleanup_bus_subscriptions called but already inactive")
            return
        self._bus.unsubscribe(ContextSwitchedPayload, self._on_bus_refresh)
        self._bus.unsubscribe(MdChangedPayload, self._on_bus_refresh)
        self._bus.unsubscribe(MlChangedPayload, self._on_bus_refresh)
        self._bus_subs_active = False

    def _on_bus_refresh(self, payload: SessionPayload) -> None:
        """EventBus subscriber wrapper; payload is ignored, delegates to refresh()."""
        del payload
        self.refresh()

    def refresh(self) -> None:
        self._component_tree.refresh()
        try:
            parameters = self._ctrl.read_context()
        except FailedPreconditionError:
            self._populate_ml(None)
            self._status_label.setText("No context")
            return
        self._populate_ml(parameters.library)
        now = datetime.now().strftime("%H:%M:%S")
        self._status_label.setText(f"Last updated: {now}")
        logger.debug("InspectDialog: refreshed")

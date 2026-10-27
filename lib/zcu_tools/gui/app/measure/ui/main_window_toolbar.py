"""Toolbar coordinator for the measure-gui main window."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from qtpy.QtCore import QEvent, QObject
from qtpy.QtGui import QAction
from qtpy.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.lifetime import LifetimeScope


@dataclass(frozen=True)
class AdapterMenuItem:
    """One adapter action in the hierarchical new-tab menu."""

    adapter_name: str
    label: str
    parent_path: tuple[str, ...]


def adapter_menu_items(adapter_names: Iterable[str]) -> tuple[AdapterMenuItem, ...]:
    """Project slash-separated adapter names into menu actions.

    Preserves input order and the previous label/grouping semantics:
    ``twotone/rabi/amp_rabi`` becomes action label ``amp_rabi`` under submenu
    path ``("twotone", "rabi")``.
    """
    items: list[AdapterMenuItem] = []
    for adapter_name in adapter_names:
        parts = tuple(adapter_name.split("/"))
        items.append(
            AdapterMenuItem(
                adapter_name=adapter_name,
                label=parts[-1],
                parent_path=parts[:-1],
            )
        )
    return tuple(items)


class MainWindowToolbarHost(Protocol):
    """Narrow host surface required by ``MainWindowToolbar``."""

    def list_adapter_names(self) -> list[str]: ...
    def create_tab(self, adapter_name: str) -> None: ...
    def open_dialog(self, name: DialogName) -> None: ...
    def reload_experiments(self) -> None: ...
    def retry_skipped_experiment_tabs(self) -> None: ...
    def open_results_browser(self) -> None: ...
    def toggle_workflows(self) -> None: ...


class MainWindowToolbar:
    """Owns the top toolbar widgets and new-tab menu orchestration."""

    def __init__(self, host: MainWindowToolbarHost, *, parent: QWidget) -> None:
        self._scope = LifetimeScope(parent)
        self._host = host
        self._parent = parent
        self._writable = True
        self._new_tab_menus: tuple[QMenu, ...] = ()
        self._layout = QHBoxLayout()
        self._new_tab_btn = QPushButton("New Tab ▾")
        self._scope.connect(self._new_tab_btn.clicked, self.show_new_tab_menu)
        self._layout.addWidget(self._new_tab_btn)
        self._layout.addStretch()
        self._add_dialog_button("Setup…", DialogName.SETUP)
        self._add_dialog_button("SoC…", DialogName.SOC)
        self._add_dialog_button("Devices…", DialogName.DEVICE)
        self._add_dialog_button("Inspect…", DialogName.INSPECT)
        self._more_btn = QPushButton("More ▾")
        self._more_btn.setObjectName("main.more")
        self._scope.connect(self._more_btn.clicked, self._show_more_menu)
        self._layout.addWidget(self._more_btn)
        self._build_more_menu()
        self._promoted: list[tuple[QPushButton, QAction]] = []
        for action in (self._predictor_action, self._results_action):
            button = QPushButton(action.text())
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            self._scope.connect(
                button.clicked, lambda _checked=False, a=action: a.trigger()
            )
            button.setMaximumWidth(button.sizeHint().width())
            self._layout.insertWidget(self._layout.count() - 1, button, stretch=1)
            self._promoted.append((button, action))
        self._scope.install_event_filter(parent, self._on_window_event)
        self._refresh_overflow()

    def dispose(self) -> None:
        """Retire toolbar callbacks before the host deletes its Qt children."""
        self._scope.close()

    def _on_window_event(self, _owner: QObject | None, event: QEvent | None) -> bool:
        if event is not None and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
        ):
            self._refresh_overflow()
        return False

    def _refresh_overflow(self) -> None:
        widgets = [
            item.widget()
            for index in range(self._layout.count())
            if (item := self._layout.itemAt(index)) is not None
            and item.widget() is not None
        ]
        margins = self._layout.contentsMargins()
        required = (
            sum(widget.sizeHint().width() for widget in widgets if widget is not None)
            + max(0, len(widgets) - 1) * max(0, self._layout.spacing())
            + margins.left()
            + margins.right()
        )
        overflow = required > self._parent.width()
        for button, action in self._promoted:
            button.setVisible(not overflow)
            action.setVisible(overflow)

    @property
    def layout(self) -> QHBoxLayout:
        return self._layout

    @property
    def new_tab_button(self) -> QPushButton:
        """Return the new-tab button for focused UI tests."""
        return self._new_tab_btn

    def set_new_tab_enabled(self, *, enabled: bool) -> None:
        self._new_tab_btn.setEnabled(enabled and self._writable)

    def set_writable(self, *, writable: bool) -> None:
        """Gate commands, not buttons that merely open inspection dialogs."""
        self._writable = writable
        self.set_new_tab_enabled(enabled=True)
        self._reload_action.setEnabled(writable)
        self._retry_tabs_action.setEnabled(writable)
        if not writable:
            for menu in reversed(self._new_tab_menus):
                for action in menu.actions():
                    action.setEnabled(False)
                menu.close()

    def set_reload_recovery(self, *, retry_catalog: bool, skipped_count: int) -> None:
        self._reload_action.setText(
            "Retry reload" if retry_catalog else "Reload experiments"
        )
        self._retry_tabs_action.setText(f"Retry skipped tabs ({skipped_count})")
        self._retry_tabs_action.setVisible(skipped_count > 0)

    def set_workflow_mode(self, *, workflow: bool) -> None:
        """Label the switch with the tab family it will reveal next."""
        self._workflow_action.setText(
            "Experiment tabs" if workflow else "Workflow tabs"
        )

    def _add_dialog_button(self, label: str, name: DialogName) -> None:
        button = QPushButton(label)
        self._scope.connect(button.clicked, lambda: self._host.open_dialog(name))
        self._layout.addWidget(button)

    def _build_more_menu(self) -> None:
        self._more_menu = QMenu(self._parent)
        self._reload_action = self._add_more_action(
            "Reload experiments", "reload", self._host.reload_experiments
        )
        self._retry_tabs_action = self._add_more_action(
            "Retry skipped tabs", "retry", self._host.retry_skipped_experiment_tabs
        )
        self._retry_tabs_action.setVisible(False)
        self._workflow_action = self._add_more_action(
            "Workflow tabs", "workflow", self._host.toggle_workflows
        )
        self._predictor_action = self._add_more_action(
            "Predictor…",
            "predictor",
            lambda: self._host.open_dialog(DialogName.PREDICTOR),
        )
        self._results_action = self._add_more_action(
            "Results browser…", "results", self._host.open_results_browser
        )

    def _add_more_action(
        self, label: str, name: str, callback: Callable[[], None]
    ) -> QAction:
        action = QAction(label, self._more_menu)
        action.setObjectName(f"main.more.{name}")
        self._scope.connect(action.triggered, callback)
        self._more_menu.addAction(action)
        return action

    def _show_more_menu(self) -> None:
        self._more_menu.exec(
            self._more_btn.mapToGlobal(self._more_btn.rect().bottomLeft())
        )

    def show_new_tab_menu(self) -> None:
        """Open the adapter menu and create the selected tab."""
        menu = QMenu(self._parent)
        submenus: dict[tuple[str, ...], QMenu] = {}

        def _get_or_create_submenu(path: tuple[str, ...]) -> QMenu:
            cached = submenus.get(path)
            if cached is not None:
                return cached
            parent_menu = menu if len(path) == 1 else _get_or_create_submenu(path[:-1])
            sub_menu = parent_menu.addMenu(path[-1])
            if sub_menu is None:
                raise RuntimeError(f"Failed to create submenu: {'/'.join(path)}")
            submenus[path] = sub_menu
            return sub_menu

        for item in adapter_menu_items(self._host.list_adapter_names()):
            target_menu = (
                menu
                if not item.parent_path
                else _get_or_create_submenu(item.parent_path)
            )
            action = target_menu.addAction(item.label)
            if action is None:
                raise RuntimeError(f"Failed to create adapter action: {item.label}")
            action.setData(item.adapter_name)

        self._new_tab_menus = (menu, *submenus.values())
        try:
            action = menu.exec(
                self._new_tab_btn.mapToGlobal(self._new_tab_btn.rect().bottomLeft())
            )
        finally:
            self._new_tab_menus = ()
            menu.deleteLater()
        if action is None or not action.isEnabled():
            return
        adapter_name = action.data()
        if not adapter_name:
            return
        self._host.create_tab(str(adapter_name))


__all__ = [
    "AdapterMenuItem",
    "MainWindowToolbar",
    "MainWindowToolbarHost",
    "adapter_menu_items",
]

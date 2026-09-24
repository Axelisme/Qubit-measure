"""Toolbar coordinator for the measure-gui main window."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QHBoxLayout,
    QMenu,
    QPushButton,
    QWidget,
)

from zcu_tools.gui.app.main.services.remote.dialogs import DialogName


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


class MainWindowToolbar:
    """Owns the top toolbar widgets and new-tab menu orchestration."""

    def __init__(self, host: MainWindowToolbarHost, *, parent: QWidget) -> None:
        self._host = host
        self._parent = parent
        self._layout = QHBoxLayout()
        self._new_tab_btn = QPushButton("New Tab ▾")
        self._new_tab_btn.clicked.connect(self.show_new_tab_menu)
        self._layout.addWidget(self._new_tab_btn)
        self._reload_btn = QPushButton("Reload experiments")
        self._reload_btn.clicked.connect(lambda: self._host.reload_experiments())
        self._layout.addWidget(self._reload_btn)
        self._retry_tabs_btn = QPushButton("Retry skipped tabs")
        self._retry_tabs_btn.clicked.connect(
            lambda: self._host.retry_skipped_experiment_tabs()
        )
        self._retry_tabs_btn.setVisible(False)
        self._layout.addWidget(self._retry_tabs_btn)
        self._layout.addStretch()
        self._add_dialog_button("Setup…", DialogName.SETUP)
        self._add_dialog_button("Devices…", DialogName.DEVICE)
        self._add_dialog_button("Predictor…", DialogName.PREDICTOR)
        self._add_dialog_button("Inspect…", DialogName.INSPECT)

    @property
    def layout(self) -> QHBoxLayout:
        return self._layout

    @property
    def new_tab_button(self) -> QPushButton:
        """Return the new-tab button for focused UI tests."""
        return self._new_tab_btn

    def set_new_tab_enabled(self, enabled: bool) -> None:
        self._new_tab_btn.setEnabled(enabled)

    def set_reload_recovery(self, *, retry_catalog: bool, skipped_count: int) -> None:
        self._reload_btn.setText(
            "Retry reload" if retry_catalog else "Reload experiments"
        )
        self._retry_tabs_btn.setText(f"Retry skipped tabs ({skipped_count})")
        self._retry_tabs_btn.setVisible(skipped_count > 0)

    def _add_dialog_button(self, label: str, name: DialogName) -> None:
        button = QPushButton(label)
        button.clicked.connect(lambda _checked=False, n=name: self._host.open_dialog(n))
        self._layout.addWidget(button)

    def show_new_tab_menu(self) -> None:
        """Open the adapter menu and create the selected tab."""
        menu = QMenu(self._parent)
        submenus: dict[tuple[str, ...], QMenu] = {}

        def _get_or_create_submenu(path: tuple[str, ...]) -> QMenu:
            cached = submenus.get(path)
            if cached is not None:
                return cached
            if len(path) == 1:
                parent_menu = menu
            else:
                parent_menu = _get_or_create_submenu(path[:-1])
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

        action = menu.exec(
            self._new_tab_btn.mapToGlobal(self._new_tab_btn.rect().bottomLeft())
        )
        if action is None:
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

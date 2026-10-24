"""Bounded saved-document trees with lossless section copying."""

from __future__ import annotations

import json

from pydantic import JsonValue
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import (
    QApplication,
    QHeaderView,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from zcu_tools.datafile import JsonObject
from zcu_tools.gui.app.measure.services.results_index import ResultRun, ResultsIndex


class ResultsDetailsTree(QTreeWidget):
    """Present saved sections as a two-column, collapsible Key/Value tree.

    Values descend three levels below a section and display at most 200
    characters per leaf. Copy JSON copies the selected section's complete saved
    JSON, not the bounded preview. Headers start at content width and remain
    mouse-resizable; the last column stretches to use spare space.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderLabels(["Key", "Value"])
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._copy_menu)
        header = self.header()
        if header is None:
            raise RuntimeError("Details tree has no header")
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self._sized = False

    def present_metadata(self, index: ResultsIndex, run: ResultRun | None) -> None:
        """Show captured identity and raw cfg sections; retain per-file errors.

        None clears a previous run and displays the selection hint. No schema
        migration, adapter lookup or experiment import is performed.
        """
        self.clear()
        if run is None:
            self._section("Select a run to inspect cfg and analysis", None)
        else:
            roles: JsonObject = {}
            for key, value in run.roles.items():
                roles[key] = value if isinstance(value, str) else list[JsonValue](value)
            metadata: JsonObject = {
                "entry": run.entry,
                "point": run.point,
                "experiment": run.experiment,
                "run": run.run_path.name,
                "time": run.at,
                "registration": "Registered" if run.registered else "Unregistered",
                "roles": roles,
                "issues": list(run.issues),
            }
            self._section("Identity", metadata)
            if run.data_path is not None:
                try:
                    document = index.read_metadata(run.key)
                except (OSError, ValueError, KeyError) as error:
                    self._section("Metadata error", str(error))
                else:
                    self._section("Metadata", document, expanded=False)
                try:
                    cfg = index.read_cfg(run.key)
                except (OSError, ValueError, KeyError) as error:
                    self._section("Cfg error", str(error))
                else:
                    if "values" in cfg:
                        identity: JsonObject = {
                            field: value
                            for field, value in cfg.items()
                            if field != "values"
                        }
                        self._section("Cfg identity", identity)
                        self._section("Cfg values", cfg["values"])
                    else:
                        self._section("Cfg", cfg)
            self._finish()

    def present_analysis(self, index: ResultsIndex, run: ResultRun | None) -> None:
        """Show raw summary and parameters per saved JSON; retain read errors."""
        self.clear()
        if run is not None:
            for path in run.analysis_paths:
                try:
                    document = index.read_analysis(run.key, path.name)
                except (OSError, ValueError, KeyError) as error:
                    self._section(path.name, str(error))
                else:
                    self._section(
                        f"Summary · {path.name}", document.get("summary", document)
                    )
                    if "parameters" in document:
                        self._section(
                            f"Parameters · {path.name}", document["parameters"]
                        )
        self._finish()

    def _section(self, name: str, value: JsonValue, *, expanded: bool = True) -> None:
        root = QTreeWidgetItem(self, [name, ""])
        root.setData(
            0, Qt.ItemDataRole.UserRole, json.dumps(value, ensure_ascii=False, indent=2)
        )
        if isinstance(value, dict):
            for key, child in value.items():
                self._append(root, key, child, 1)
        else:
            self._append(root, "Value", value, 1)
        root.setExpanded(expanded)

    def _append(
        self, parent: QTreeWidgetItem, key: str, value: JsonValue, depth: int
    ) -> None:
        item = QTreeWidgetItem(parent, [key, ""])
        if isinstance(value, dict) and value and depth < 3:
            for name, child in value.items():
                self._append(item, name, child, depth + 1)
            item.setExpanded(True)
        elif isinstance(value, list) and value and depth < 3:
            for number, child in enumerate(value):
                self._append(item, f"[{number}]", child, depth + 1)
            item.setExpanded(True)
        else:
            text = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else "—"
                if value is None
                else str(value)
            )
            item.setText(1, text[:200] + (" …" if len(text) > 200 else ""))

    def _finish(self) -> None:
        if not self._sized and self.topLevelItemCount():
            self.resizeColumnToContents(0)
            self.resizeColumnToContents(1)
            self._sized = True

    def _copy_menu(self, point: QPoint) -> None:
        item = self.currentItem()
        if item is None:
            item = self.itemAt(point)
        if item is None:
            return
        while (parent := item.parent()) is not None:
            item = parent
        menu = QMenu(self)
        copy = menu.addAction("Copy JSON")
        viewport = self.viewport()
        assert viewport is not None
        if menu.exec(viewport.mapToGlobal(point)) is copy:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(str(item.data(0, Qt.ItemDataRole.UserRole)))
        menu.deleteLater()

"""MainWindow — the fluxdep-gui shell.

Layout (Q2 decision): a left spectrum list + a right single editing area, not
tabs (spectra are members of one collection — list + single editor reads more
naturally than parallel tabs).

  ┌────────────┬───────────────────────────┐
  │ spectra    │  editing area (stacked)   │
  │  • R1 ...  │   per active spectrum's   │
  │  • Q1 ...  │   pipeline stage          │
  │ [load]     │   (interactive widgets    │
  │ [remove]   │    land here in a later   │
  │ [export]   │    batch — placeholder    │
  └────────────┴───────────────────────────┘

This batch is the shell: the list + buttons + an empty editing-area placeholder,
wired to the Controller and refreshing on EventBus signals. The interactive
line-picker / point-selector widgets fill the editing area later.
"""

from __future__ import annotations

import logging
from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.event_bus import (
    ActiveSpectrumChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.fluxdep_gui.state import SpecType

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """The fluxdep analysis window shell."""

    def __init__(self, ctrl: Controller) -> None:
        super().__init__()
        self._ctrl = ctrl
        self.setWindowTitle("fluxdep-gui")
        self.resize(1100, 700)

        self._build_ui()
        self._subscribe_events()
        self._refresh_list()

    # --- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)

        # Left panel: spectrum list + actions.
        left = QVBoxLayout()
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_list_selection)
        left.addWidget(QLabel("Spectra"))
        left.addWidget(self._list, stretch=1)

        self._load_btn = QPushButton("Load…")
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        self._export_btn = QPushButton("Export spectrums.hdf5")
        self._export_btn.clicked.connect(self._on_export_clicked)
        for btn in (self._load_btn, self._remove_btn, self._export_btn):
            left.addWidget(btn)

        left_panel = QWidget()
        left_panel.setLayout(left)
        left_panel.setFixedWidth(280)

        # Right panel: editing area (stacked; placeholder until widgets land).
        self._editor_stack = QStackedWidget()
        self._placeholder = QLabel("Select or load a spectrum to begin.")
        self._placeholder.setEnabled(False)
        self._editor_stack.addWidget(self._placeholder)

        root.addWidget(left_panel)
        root.addWidget(self._editor_stack, stretch=1)
        self.setCentralWidget(central)

    def _subscribe_events(self) -> None:
        bus = self._ctrl.bus
        bus.subscribe(SpectrumAddedPayload, lambda _p: self._refresh_list())
        bus.subscribe(SpectrumRemovedPayload, lambda _p: self._refresh_list())
        bus.subscribe(SpectrumChangedPayload, lambda _p: self._refresh_list())
        bus.subscribe(ActiveSpectrumChangedPayload, self._on_active_changed)

    # --- list refresh ----------------------------------------------------

    def _refresh_list(self) -> None:
        """Rebuild the spectrum list from State (label = name + type + stage)."""
        active = self._ctrl.state.active_spectrum
        self._list.blockSignals(True)
        self._list.clear()
        for name in self._ctrl.list_spectrums():
            entry = self._ctrl.state.spectrums[name]
            stage = (
                "✓pts"
                if entry.points_selected
                else "✓align"
                if entry.aligned
                else "new"
            )
            item = QListWidgetItem(f"{name}  [{entry.spec_type}·{stage}]")
            item.setData(256, name)  # Qt.UserRole == 256; stash the bare name
            self._list.addItem(item)
            if name == active:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)

    # --- actions ---------------------------------------------------------

    def _on_list_selection(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            return
        name = current.data(256)
        if name and name != self._ctrl.state.active_spectrum:
            self._ctrl.set_active_spectrum(name)

    def _on_active_changed(self, _payload: ActiveSpectrumChangedPayload) -> None:
        # Editing area swap lands with the interactive widgets; shell shows the
        # placeholder for now.
        self._editor_stack.setCurrentWidget(self._placeholder)

    def _on_load_clicked(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load spectrum", filter="HDF5 (*.hdf5 *.h5);;All files (*)"
        )
        if not filepath:
            return
        spec_type = self._ask_spec_type()
        if spec_type is None:
            return
        try:
            self._ctrl.load_spectrum(filepath, spec_type)
        except Exception as exc:  # noqa: BLE001 — surface load errors, don't crash the shell
            logger.exception("load_spectrum failed")
            self._show_error("Load failed", str(exc))

    def _ask_spec_type(self) -> Optional[SpecType]:
        choice, ok = QInputDialog.getItem(
            self, "Spectrum type", "Type:", ["OneTone", "TwoTone"], 0, False
        )
        if not ok:
            return None
        return "OneTone" if choice == "OneTone" else "TwoTone"

    def _on_remove_clicked(self) -> None:
        name = self._ctrl.state.active_spectrum
        if name is not None:
            self._ctrl.remove_spectrum(name)

    def _on_export_clicked(self) -> None:
        try:
            path = self._ctrl.export_spectrums(mode="w")
            self._show_info("Exported", f"Wrote {path}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("export failed")
            self._show_error("Export failed", str(exc))

    # --- dialogs ---------------------------------------------------------

    def _show_error(self, title: str, message: str) -> None:
        from qtpy.QtWidgets import QMessageBox  # type: ignore[attr-defined]

        QMessageBox.critical(self, title, message)

    def _show_info(self, title: str, message: str) -> None:
        from qtpy.QtWidgets import QMessageBox  # type: ignore[attr-defined]

        QMessageBox.information(self, title, message)

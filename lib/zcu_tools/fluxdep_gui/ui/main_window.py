"""MainWindow — the fluxdep-gui shell.

Layout (Q2 decision): a left spectrum list + a right single editing area, not
tabs (spectra are members of one collection — list + single editor reads more
naturally than parallel tabs).

  ┌────────────┬───────────────────────────┐
  │ spectra    │  editing area (stacked)   │
  │  • R1 ...  │   the active spectrum's   │
  │  • Q1 ...  │   pipeline-stage widget:  │
  │ [load]     │   not-aligned → LinePicker│
  │ [remove]   │   aligned → point selector│
  │ [export]   │   (OneTone / FindPoints)  │
  └────────────┴───────────────────────────┘

The editing area swaps the interactive widget by the active spectrum's stage; a
widget's ``finished`` signal writes its result back through the Controller, which
advances the stage and re-triggers the swap. "Re-align active" reopens the line
picker; "Cross-spectrum filter…" opens the joint-point-cloud selector over every
spectrum that has selected points.
"""

from __future__ import annotations

import logging
from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QHBoxLayout,
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
from zcu_tools.fluxdep_gui.state import SpectrumEntry
from zcu_tools.fluxdep_gui.ui.interactive.base import InteractiveMplWidget
from zcu_tools.fluxdep_gui.ui.interactive.find_points import FindPointsWidget
from zcu_tools.fluxdep_gui.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.fluxdep_gui.ui.interactive.onetone import OneToneWidget
from zcu_tools.fluxdep_gui.ui.interactive.selector import SelectorWidget

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
        self._realign_btn = QPushButton("Re-align active")
        self._realign_btn.clicked.connect(self._on_realign_clicked)
        self._filter_btn = QPushButton("Cross-spectrum filter…")
        self._filter_btn.clicked.connect(self._on_filter_clicked)
        self._export_btn = QPushButton("Export spectrums.hdf5")
        self._export_btn.clicked.connect(self._on_export_clicked)
        for btn in (
            self._load_btn,
            self._remove_btn,
            self._realign_btn,
            self._filter_btn,
            self._export_btn,
        ):
            left.addWidget(btn)

        left_panel = QWidget()
        left_panel.setLayout(left)
        left_panel.setFixedWidth(280)

        # Right panel: editing area (stacked). The active spectrum's pipeline
        # stage decides which interactive widget is shown; _current_editor is
        # the live one (replaced on stage/active change).
        self._editor_stack = QStackedWidget()
        self._placeholder = QLabel("Select or load a spectrum to begin.")
        self._placeholder.setEnabled(False)
        self._editor_stack.addWidget(self._placeholder)
        self._current_editor: Optional[InteractiveMplWidget] = None

        root.addWidget(left_panel)
        root.addWidget(self._editor_stack, stretch=1)
        self.setCentralWidget(central)

    def _subscribe_events(self) -> None:
        bus = self._ctrl.bus
        bus.subscribe(SpectrumAddedPayload, lambda _p: self._refresh_list())
        bus.subscribe(SpectrumRemovedPayload, self._on_spectrum_removed)
        bus.subscribe(SpectrumChangedPayload, self._on_spectrum_changed)
        bus.subscribe(ActiveSpectrumChangedPayload, self._on_active_changed)

    def _on_spectrum_removed(self, _payload: SpectrumRemovedPayload) -> None:
        self._refresh_list()
        self._rebuild_editor()

    def _on_spectrum_changed(self, payload: SpectrumChangedPayload) -> None:
        self._refresh_list()
        # A stage change (aligned / points_selected) on the active spectrum must
        # advance its editing widget (line-picker → point-selector → done).
        if payload.name == self._ctrl.state.active_spectrum:
            self._rebuild_editor()

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
        self._rebuild_editor()

    # --- editing area (stage-driven interactive widget) ------------------

    def _clear_editor(self) -> None:
        if self._current_editor is not None:
            self._editor_stack.removeWidget(self._current_editor)
            self._current_editor.deleteLater()
            self._current_editor = None
        self._editor_stack.setCurrentWidget(self._placeholder)

    def _rebuild_editor(self) -> None:
        """Show the interactive widget for the active spectrum's current stage.

        Stages: not-aligned → LinePicker; aligned-not-selected → OneTone/FindPoints
        (by spec_type); selected → placeholder (use Re-align / Cross-spectrum
        filter to continue). The widget's ``finished`` signal writes its result
        back through the Controller, which advances the stage and re-triggers this.
        """
        self._clear_editor()
        name = self._ctrl.state.active_spectrum
        if name is None:
            return
        entry = self._ctrl.state.spectrums[name]

        if not entry.aligned:
            self._mount_line_picker(entry)
        elif not entry.points_selected:
            self._mount_point_selector(entry)
        # else: selected — leave the placeholder; the points are set.

    def _mount(self, widget: InteractiveMplWidget) -> None:
        self._current_editor = widget
        self._editor_stack.addWidget(widget)
        self._editor_stack.setCurrentWidget(widget)

    def _mount_line_picker(self, entry: SpectrumEntry) -> None:
        widget = LinePickerWidget(
            entry.raw["signals"], entry.raw["dev_values"], entry.raw["freqs"]
        )
        name = entry.name

        def _on_finish() -> None:
            half, integer = widget.get_result()
            self._ctrl.set_alignment(name, half, integer)

        widget.finished.connect(_on_finish)
        self._mount(widget)

    def _mount_point_selector(self, entry: SpectrumEntry) -> None:
        if entry.spec_type == "OneTone":
            widget: InteractiveMplWidget = OneToneWidget(
                entry.raw["signals"], entry.raw["dev_values"], entry.raw["freqs"]
            )
        else:
            widget = FindPointsWidget(
                entry.raw["signals"], entry.raw["dev_values"], entry.raw["freqs"]
            )
        name = entry.name

        def _on_finish() -> None:
            dev_values, freqs = widget.get_result()  # type: ignore[attr-defined]
            self._ctrl.set_points(name, dev_values, freqs)

        widget.finished.connect(_on_finish)
        self._mount(widget)

    def _on_realign_clicked(self) -> None:
        """Re-open the line picker for the active spectrum (redo alignment)."""
        name = self._ctrl.state.active_spectrum
        if name is None:
            return
        self._clear_editor()
        self._mount_line_picker(self._ctrl.state.spectrums[name])

    def _on_filter_clicked(self) -> None:
        """Open the cross-spectrum selector over all spectra with selected points."""
        from zcu_tools.notebook.persistance import SpectrumResult

        spectrums: dict[str, SpectrumResult] = {
            n: SpectrumResult(
                type=e.spec_type,
                flux_half=e.flux_half,
                flux_int=e.flux_int,
                flux_period=e.flux_period,
                spectrum=e.raw,
                points=e.points,
            )
            for n, e in self._ctrl.state.spectrums.items()
            if e.points_selected
        }
        if not spectrums:
            self._show_error("No points", "No spectrum has selected points yet.")
            return
        selector = SelectorWidget(
            spectrums, selected=self._ctrl.state.selection.selected
        )

        def _on_finish() -> None:
            _fluxs, _freqs, selected = selector.get_result()
            self._ctrl.set_selection(selected)
            self._clear_editor()

        selector.finished.connect(_on_finish)
        self._mount(selector)

    def _on_load_clicked(self) -> None:
        from zcu_tools.fluxdep_gui.ui.load_dialog import LoadSpectrumDialog

        dialog = LoadSpectrumDialog(self._ctrl.list_spectrums(), parent=self)
        if dialog.exec() != int(dialog.DialogCode.Accepted):
            return
        req = dialog.result_request()
        if req is None:
            return
        try:
            name = self._ctrl.load_spectrum(
                req.filepath,
                req.spec_type,
                inherit_from=req.inherit_from,
                transpose_axes=req.transpose_axes,
            )
            self._ctrl.set_active_spectrum(name)  # jump straight into line picking
        except Exception as exc:  # noqa: BLE001 — surface load errors, don't crash the shell
            logger.exception("load_spectrum failed")
            self._show_error("Load failed", str(exc))

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

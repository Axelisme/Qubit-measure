"""Controller — the fluxdep-gui façade.

Holds the State + EventBus and the domain services, and is the single command
surface for both Views (the Qt MainWindow and, later, the RemoteControlAdapter).
Services stay pure (they mutate State and bump versions, Qt-free, independently
testable); the Controller is the coordination layer that calls a service and
then emits the corresponding EventBus event so Views can react.

It deliberately has NO measure concepts (run / analyze / writeback / context /
device / tab) — only the fluxdep pipeline actions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from zcu_tools.fluxdep_gui.event_bus import (
    ActiveSpectrumChangedPayload,
    EventBus,
    ProjectChangedPayload,
    SelectionChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.fluxdep_gui.services.alignment import AlignmentService, PointsService
from zcu_tools.fluxdep_gui.services.export import ExportService
from zcu_tools.fluxdep_gui.services.load import LoadService
from zcu_tools.fluxdep_gui.services.store import SelectionService, SpectrumStore
from zcu_tools.fluxdep_gui.state import FluxDepState, ProjectInfo, SpecType

logger = logging.getLogger(__name__)


class Controller:
    """Command façade over the fluxdep pipeline services."""

    def __init__(self, state: FluxDepState, bus: Optional[EventBus] = None) -> None:
        self._state = state
        self._bus = bus if bus is not None else EventBus()
        self._load = LoadService(state)
        self._alignment = AlignmentService(state)
        self._points = PointsService(state)
        self._store = SpectrumStore(state)
        self._selection = SelectionService(state)
        self._export = ExportService(state)

    @property
    def state(self) -> FluxDepState:
        return self._state

    @property
    def bus(self) -> EventBus:
        return self._bus

    # --- project ---------------------------------------------------------

    def setup_project(self, project: ProjectInfo) -> None:
        self._state.set_project(project)
        self._bus.emit(ProjectChangedPayload())

    # --- spectrum collection --------------------------------------------

    def load_spectrum(
        self,
        filepath: str,
        spec_type: SpecType,
        inherit_from: Optional[str] = None,
        transpose_axes: bool = False,
    ) -> str:
        name = self._load.load_spectrum(
            filepath, spec_type, inherit_from, transpose_axes
        )
        self._bus.emit(SpectrumAddedPayload(name=name))
        return name

    def remove_spectrum(self, name: str) -> None:
        self._store.remove_spectrum(name)
        self._bus.emit(SpectrumRemovedPayload(name=name))

    def set_active_spectrum(self, name: Optional[str]) -> None:
        self._store.set_active(name)
        self._bus.emit(ActiveSpectrumChangedPayload(name=name))

    def list_spectrums(self) -> list[str]:
        return self._store.list_spectrums()

    # --- alignment / points ---------------------------------------------

    def set_alignment(self, name: str, flux_half: float, flux_int: float) -> None:
        self._alignment.set_alignment(name, flux_half, flux_int)
        self._bus.emit(SpectrumChangedPayload(name=name))

    def set_points(
        self, name: str, dev_values: NDArray[np.float64], freqs: NDArray[np.float64]
    ) -> None:
        self._points.set_points(name, dev_values, freqs)
        self._bus.emit(SpectrumChangedPayload(name=name))

    # --- cross-spectrum selection ---------------------------------------

    def derive_pointcloud(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._selection.derive_pointcloud()

    def set_selection(self, selected: NDArray[np.bool_]) -> None:
        self._selection.set_selection(selected)
        self._bus.emit(SelectionChangedPayload())

    # --- export ----------------------------------------------------------

    def export_spectrums(self, filepath: Optional[str] = None, mode: str = "x") -> str:
        return self._export.export_spectrums(filepath, mode)

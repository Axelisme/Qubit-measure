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

from zcu_tools.gui.app.fluxdep.event_bus import (
    ActiveSpectrumChangedPayload,
    EventBus,
    FitChangedPayload,
    ProjectChangedPayload,
    SelectionChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.services.alignment import AlignmentService, PointsService
from zcu_tools.gui.app.fluxdep.services.export import ExportService
from zcu_tools.gui.app.fluxdep.services.fit import FitService, PbarFactory, SearchResult
from zcu_tools.gui.app.fluxdep.services.load import LoadService
from zcu_tools.gui.app.fluxdep.services.store import SelectionService, SpectrumStore
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpecType
from zcu_tools.gui.controller_base import BaseController
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.notebook.persistance import TransitionDict

logger = logging.getLogger(__name__)


class Controller(BaseController[FluxDepState, EventBus]):
    """Command façade over the fluxdep pipeline services."""

    def __init__(
        self,
        state: FluxDepState,
        bus: Optional[EventBus] = None,
        project_root: Optional[str] = None,
    ) -> None:
        super().__init__(state, bus if bus is not None else EventBus(), project_root)
        self._load = LoadService(state)
        self._alignment = AlignmentService(state)
        self._points = PointsService(state)
        self._store = SpectrumStore(state)
        self._selection = SelectionService(state)
        self._export = ExportService(state)
        self._fit = FitService(state)

    # --- project ---------------------------------------------------------

    def setup_project(self, project: ProjectInfo) -> None:
        self._state.set_project(project)
        self._emit(ProjectChangedPayload())

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
        self._emit(SpectrumAddedPayload(name=name))
        return name

    def load_processed_spectrums(self, filepath: str) -> list[str]:
        """Restore a processed spectrums.hdf5 (aligned + selected spectra)."""
        names = self._load.load_processed_spectrums(filepath)
        for name in names:
            self._emit(SpectrumAddedPayload(name=name))
        return names

    def remove_spectrum(self, name: str) -> None:
        self._store.remove_spectrum(name)
        self._emit(SpectrumRemovedPayload(name=name))

    def set_active_spectrum(self, name: Optional[str]) -> None:
        self._store.set_active(name)
        self._emit(ActiveSpectrumChangedPayload(name=name))

    def list_spectrums(self) -> list[str]:
        return self._store.list_spectrums()

    # --- alignment / points ---------------------------------------------

    def set_alignment(self, name: str, flux_half: float, flux_int: float) -> None:
        self._alignment.set_alignment(name, flux_half, flux_int)
        self._emit(SpectrumChangedPayload(name=name))

    def set_points(
        self, name: str, dev_values: NDArray[np.float64], freqs: NDArray[np.float64]
    ) -> None:
        self._points.set_points(name, dev_values, freqs)
        self._emit(SpectrumChangedPayload(name=name))

    # --- cross-spectrum selection ---------------------------------------

    def derive_pointcloud(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._selection.derive_pointcloud()

    def set_selection(
        self, selected: NDArray[np.bool_], min_distance: float = 0.0
    ) -> None:
        self._selection.set_selection(selected, min_distance)
        self._emit(SelectionChangedPayload())

    # --- export ----------------------------------------------------------

    def export_spectrums(self, filepath: Optional[str] = None, mode: str = "x") -> str:
        return self._export.export_spectrums(filepath, mode)

    # --- database-search fit (v2) ---------------------------------------

    def set_fit_params(
        self,
        database_path: str,
        EJb: tuple[float, float],
        ECb: tuple[float, float],
        ELb: tuple[float, float],
        transitions: TransitionDict,
        r_f: Optional[float],
        sample_f: Optional[float],
    ) -> None:
        self._fit.set_params(database_path, EJb, ECb, ELb, transitions, r_f, sample_f)
        self._emit(FitChangedPayload(has_result=self._state.fit.has_result))

    def selected_pointcloud(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._fit.selected_pointcloud()

    def compute_search(
        self,
        *,
        pbar_factory: Optional[PbarFactory] = None,
        plot: bool = False,
    ) -> SearchResult:
        """Run the search WITHOUT touching State (safe on a worker thread).

        Pair with ``record_search_result`` on the main thread. The GUI worker
        calls this off-main, then marshals the result to the main thread to
        record it; the RPC convenience ``search_database`` does both in sequence
        on the main thread.
        """
        return self._fit.compute_search(pbar_factory=pbar_factory, plot=plot)

    def record_search_result(self, result: SearchResult) -> None:
        """Write a computed search result onto State (MAIN THREAD only)."""
        self._fit.record_result(result)
        self._emit(FitChangedPayload(has_result=True))

    def search_database(
        self,
        *,
        pbar_factory: Optional[PbarFactory] = None,
        plot: bool = False,
    ) -> SearchResult:
        """Main-thread convenience: compute the search then record it (RPC path).

        Runs the blocking search inline on the calling (main) thread — used by the
        RPC dispatch, where momentary GUI unresponsiveness is acceptable and the
        State write must stay on the main thread anyway. The GUI uses the split
        ``compute_search`` / ``record_search_result`` to keep the search off-main.
        """
        result = self._fit.compute_search(pbar_factory=pbar_factory, plot=plot)
        self.record_search_result(result)
        return result

    def export_params(self, savepath: Optional[str] = None) -> str:
        return self._fit.export_params(savepath)

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
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import DatabaseSearchResult
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
from zcu_tools.gui.app.fluxdep.interactive import (
    FluxDepInteractiveOwner,
    FluxDepInteractivePorts,
)
from zcu_tools.gui.app.fluxdep.services.alignment import AlignmentService, PointsService
from zcu_tools.gui.app.fluxdep.services.export import ExportService
from zcu_tools.gui.app.fluxdep.services.fit import FitService, PbarFactory
from zcu_tools.gui.app.fluxdep.services.load import LoadService
from zcu_tools.gui.app.fluxdep.services.store import SelectionService, SpectrumStore
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpecType
from zcu_tools.gui.controller_base import BaseController
from zcu_tools.gui.interactive.plugin import BackgroundSubmitter
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.ports import OwnerScheduler

logger = logging.getLogger(__name__)


class Controller(BaseController[FluxDepState, EventBus]):
    """Command façade over the fluxdep pipeline services."""

    def __init__(
        self,
        state: FluxDepState,
        bus: EventBus | None = None,
        project_root: str | None = None,
        *,
        interactive_owner: OwnerScheduler | None = None,
        interactive_background: BackgroundSubmitter | None = None,
    ) -> None:
        """Compose services and a picker owner.

        interactive_owner defines the state thread; omitted uses a manual owner
        for headless callers. interactive_background delivers completion on that
        thread; omitted rejects asynchronous alignment. Project root is the
        optional export/path context, not an input-data location.
        """
        super().__init__(state, bus if bus is not None else EventBus(), project_root)
        self._load = LoadService(state)
        self._alignment = AlignmentService(state)
        self._points = PointsService(state)
        self._store = SpectrumStore(state)
        self._selection = SelectionService(state)
        self._export = ExportService(state)
        self._fit = FitService(state)
        self._interactive = FluxDepInteractiveOwner(
            state,
            self.bus,
            interactive_owner
            if interactive_owner is not None
            else ManualOwnerScheduler(),
            ports=FluxDepInteractivePorts(
                background=interactive_background,
                publish_alignment=self.set_alignment,
                publish_points=self.set_points,
                derive_pointcloud=self.derive_pointcloud,
                publish_selection=self.set_selection,
            ),
        )

    @property
    def interactive(self) -> FluxDepInteractiveOwner:
        """Return the owner shared by GUI and command-driven line picking."""
        return self._interactive

    # --- project ---------------------------------------------------------

    def setup_project(self, project: ProjectInfo) -> None:
        self._state.set_project(project)
        self._emit(ProjectChangedPayload())

    # --- spectrum collection --------------------------------------------

    def load_spectrum(
        self,
        filepath: str,
        spec_type: SpecType,
        inherit_from: str | None = None,
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

    def set_active_spectrum(self, name: str | None) -> None:
        self._store.set_active(name)
        self._emit(ActiveSpectrumChangedPayload(name=name))

    def list_spectrums(self) -> list[str]:
        return self._store.list_spectrums()

    # --- alignment / points ---------------------------------------------

    def set_alignment(self, name: str, flux_half: float, flux_int: float) -> None:
        self._alignment.set_alignment(name, flux_half, flux_int)
        self._emit(SpectrumChangedPayload(name=name))

    def reset_alignment(self, name: str) -> None:
        """Reopen alignment, keeping points/completion; emit one change on success.

        Owner-thread command. Unknown name raises KeyError without publication.
        """
        self._alignment.reset_alignment(name)
        self._emit(SpectrumChangedPayload(name=name))

    def reset_points(self, name: str) -> None:
        """Clear points/completion and reopen picking; emit one change on success.

        Owner-thread command. Unknown name raises KeyError; unaligned spectrum
        raises ValueError. Failures do not publish.
        """
        self._points.reset_points(name)
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

    def export_spectrums(self, filepath: str | None = None, mode: str = "x") -> str:
        return self._export.export_spectrums(filepath, mode)

    # --- database-search fit (v2) ---------------------------------------

    def set_fit_params(
        self,
        database_path: str,
        EJb: tuple[float, float],
        ECb: tuple[float, float],
        ELb: tuple[float, float],
        transitions: TransitionDict,
        r_f: float | None,
        sample_f: float | None,
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
        pbar_factory: PbarFactory | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> DatabaseSearchResult:
        """Run the search WITHOUT touching State (safe on a worker thread).

        Pair with ``record_search_result`` on the main thread. The GUI worker
        calls this off-main, then marshals the result to the main thread to
        record it; the synchronous convenience ``search_database`` does both in
        sequence on the calling thread. pbar_factory optionally supplies worker
        progress; cancel_requested is a quick worker-safe predicate. True at a
        kernel checkpoint raises SearchCancelled without a result or State
        write. None disables cancellation. Other failures propagate, and an
        in-flight HDF5/Numba call may delay the next cancellation checkpoint.
        """
        return self._fit.compute_search(
            pbar_factory=pbar_factory, cancel_requested=cancel_requested
        )

    def record_search_result(self, result: DatabaseSearchResult) -> None:
        """Write a computed search result onto State (MAIN THREAD only)."""
        self._fit.record_result(result)
        self._emit(FitChangedPayload(has_result=True))

    def search_database(
        self,
        *,
        pbar_factory: PbarFactory | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> DatabaseSearchResult:
        """Synchronous convenience: compute the search then record it.

        Runs the blocking search inline on the calling thread, which must be the
        main thread because of the State write. No remote method triggers it; the
        GUI worker uses the split ``compute_search`` / ``record_search_result`` to
        keep the search off-main. pbar_factory optionally supplies progress;
        cancel_requested is a quick worker-safe predicate. True at a kernel
        checkpoint raises SearchCancelled and does not publish a fit result or
        fact. None disables cancellation. Other failures propagate. In-flight
        HDF5/Numba work may delay cancellation observation.
        """
        result = self._fit.compute_search(
            pbar_factory=pbar_factory, cancel_requested=cancel_requested
        )
        self.record_search_result(result)
        return result

    def export_params(self, savepath: str | None = None) -> str:
        return self._fit.export_params(savepath)

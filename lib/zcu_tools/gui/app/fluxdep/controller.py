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
from zcu_tools.gui.app.fluxdep.search import FluxDepSearchOwner, FluxDepSearchRuntime
from zcu_tools.gui.app.fluxdep.services.alignment import AlignmentService, PointsService
from zcu_tools.gui.app.fluxdep.services.export import ExportService
from zcu_tools.gui.app.fluxdep.services.fit import FitService, PbarFactory, SearchInput
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
        search_runtime: FluxDepSearchRuntime | None = None,
    ) -> None:
        """Compose services and shared picker/search owners.

        interactive_owner schedules both owners on State's constructing thread;
        omitted uses a manual headless owner. interactive_background delivers
        picker work there; omitted rejects asynchronous alignment. Search uses
        its own search_runtime background/progress pair; None rejects operation
        start but permits synchronous search_database.
        project_root is optional export/path context, not an input-data location.
        """
        super().__init__(state, bus if bus is not None else EventBus(), project_root)
        self._load = LoadService(state)
        self._alignment = AlignmentService(state)
        self._points = PointsService(state)
        self._store = SpectrumStore(state)
        self._selection = SelectionService(state)
        self._export = ExportService(state)
        self._fit = FitService(state)
        owner = (
            interactive_owner
            if interactive_owner is not None
            else ManualOwnerScheduler()
        )
        self._search = FluxDepSearchOwner(
            state,
            self.bus,
            owner,
            self._fit,
            self.record_search_result,
            runtime=search_runtime,
        )
        self._interactive = FluxDepInteractiveOwner(
            state,
            self.bus,
            owner,
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

    @property
    def search(self) -> FluxDepSearchOwner:
        """Return the app-owned search lifecycle shared by GUI and RPC."""
        return self._search

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

        Owner-thread command. Unknown name raises InvalidInputError
        (unknown_spectrum) without publication.
        """
        self._alignment.reset_alignment(name)
        self._emit(SpectrumChangedPayload(name=name))

    def reset_points(self, name: str) -> None:
        """Clear points/completion and reopen picking; emit one change on success.

        Owner-thread command. Unknown name raises InvalidInputError
        (unknown_spectrum); unaligned raises FailedPreconditionError
        (spectrum_not_aligned). Failures do not publish.
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
        """Write the native spectrum collection and return its output path.

        Owner-thread command. None uses
        <project.result_dir>/data/fluxdep/spectrums.hdf5.
        mode is the native h5py mode: x creates only, w replaces an existing
        file. Empty collection raises FailedPreconditionError (no_spectrums)
        before I/O. I/O errors propagate and may leave partial output.
        State and resource observations do not change.
        """
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
        """Replace search inputs and clear any result; emit FitChanged once.

        Owner-thread command. database_path is the search database file, not
        the project's raw-data directory. EJb/ECb/ELb are lower/upper bounds
        in GHz. transitions holds native integer level-pair categories and
        optional r_f/sample_f values in GHz; separate frequencies are injected
        by the search kernel according to transitions_with_freqs. None clears
        a separate frequency. Domain validation happens when searching.
        This command neither checks the database nor starts a search.
        """
        self._fit.set_params(database_path, EJb, ECb, ELb, transitions, r_f, sample_f)
        self._emit(FitChangedPayload(has_result=self._state.fit.has_result))

    def selected_pointcloud(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._fit.selected_pointcloud()

    def capture_search(self) -> SearchInput:
        """Capture detached search inputs on State's owner; invalid inputs raise."""
        return self._fit.capture_search()

    def compute_search(
        self,
        inputs: SearchInput,
        *,
        pbar_factory: PbarFactory | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> DatabaseSearchResult:
        """Compute only from detached inputs returned by capture_search.

        Safe on a worker; no live State reads/writes or publication. Use
        record_search_result on the owner to commit. pbar_factory optionally
        supplies progress. cancel_requested is a quick thread-safe predicate;
        True at a kernel checkpoint raises SearchCancelled without a result.
        None disables cancellation. Other failures propagate. In-flight
        HDF5/Numba work may delay the next cancellation checkpoint.
        """
        return self._fit.compute_search(
            inputs, pbar_factory=pbar_factory, cancel_requested=cancel_requested
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

        Captures inputs, computes and commits inline on State's owner thread,
        without an operation token. GUI/RPC use search.start instead.
        pbar_factory optionally supplies progress;
        cancel_requested is a quick worker-safe predicate. True at a kernel
        checkpoint raises SearchCancelled and does not publish a fit result or
        fact. None disables cancellation. Other failures propagate. In-flight
        HDF5/Numba work may delay cancellation observation.
        """
        result = self._fit.compute_search(
            self.capture_search(),
            pbar_factory=pbar_factory,
            cancel_requested=cancel_requested,
        )
        self.record_search_result(result)
        return result

    def export_params(self, savepath: str | None = None) -> str:
        """Merge the current fit into params.json and return its output path.

        Owner-thread command. None uses params.json under project.result_dir.
        Independent sections are preserved. Calibration comes from the first
        aligned spectrum. FailedPreconditionError reports no_fit_result or
        no_aligned_spectrum before I/O; native I/O errors propagate. State and
        resource observations do not change.
        """
        return self._fit.export_params(savepath)

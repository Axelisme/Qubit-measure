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
from matplotlib.figure import Figure
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
from zcu_tools.gui.app.fluxdep.services.fit import (
    FitExportReceipt,
    FitService,
    PbarFactory,
    SearchInput,
)
from zcu_tools.gui.app.fluxdep.services.load import LoadService
from zcu_tools.gui.app.fluxdep.services.store import SelectionService, SpectrumStore
from zcu_tools.gui.app.fluxdep.services.viz import (
    FitDisplay,
    derive_fit_auto_limits,
    render_fit_result,
    render_selection_result,
)
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpecType
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.controller_base import BaseController
from zcu_tools.gui.expected_error import FailedPreconditionError
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
        self.control_identity = ControlIdentity()
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

    def set_alignment(
        self,
        name: str,
        flux_half: float,
        flux_int: float,
        inherited_from: str | None = None,
    ) -> list[str]:
        """Publish native half/integer lines and return downstream invalidated labels.

        The picker passes inherited_from only when finishing an untouched
        inherited draft. Other callers omit it for manually determined alignment.
        The owner service maps raw/point flux axes and publishes provenance under
        the same version bump. Invalid separation/mapped values fail before writes.
        """
        before = self.state.pipeline_snapshot()
        self._alignment.set_alignment(name, flux_half, flux_int, inherited_from)
        invalidated = self.state.invalidated(before)
        self._emit(SpectrumChangedPayload(name=name))
        return invalidated

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
    ) -> list[str]:
        """Publish native device/GHz points and return this publication's invalidated labels.

        PointsService sorts/maps arrays. Validation failures publish nothing.
        Capture downstream facts before emitting; reentrant subscribers cannot
        change this receipt. State and events remain committed if an emit fails.
        """
        before = self.state.pipeline_snapshot()
        self._points.set_points(name, dev_values, freqs)
        invalidated = self.state.invalidated(before)
        self._emit(SpectrumChangedPayload(name=name))
        return invalidated

    # --- cross-spectrum selection ---------------------------------------

    def derive_pointcloud(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._selection.derive_pointcloud()

    def set_selection(
        self, selected: NDArray[np.bool_], min_distance: float = 0.0
    ) -> list[str]:
        """Publish a joint bool mask and return this publication's invalidated labels.

        min_distance is the normalized downsample threshold. Shape errors fail
        before publication. Capture facts before emitting, independent of
        synchronous subscriber writes. No rollback is claimed on emit failure.
        """
        before = self.state.pipeline_snapshot()
        self._selection.set_selection(selected, min_distance)
        invalidated = self.state.invalidated(before)
        self._emit(SelectionChangedPayload())
        return invalidated

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
        search_db: str,
        EJb: tuple[float, float],
        ECb: tuple[float, float],
        ELb: tuple[float, float],
        transitions: TransitionDict,
        r_f: float | None,
        sample_f: float | None,
    ) -> None:
        """Replace search inputs and clear any result; emit FitChanged once.

        Owner-thread command. search_db is the search database file, not
        the project's raw-data directory. EJb/ECb/ELb are lower/upper bounds
        in GHz. transitions holds native integer level-pair categories and
        optional r_f/sample_f values in GHz; separate frequencies are injected
        by the search kernel according to transitions_with_freqs. None clears
        a separate frequency. Unknown categories fail before I/O; nonempty
        pairs or dynamic categories read DB metadata to check level indices.
        Empty basic choices need no file. Invalid choices raise InvalidInputError;
        I/O errors propagate before publication. Frequency/bounds validation
        remains search-owned. This command does not start a search.
        """
        self._fit.set_params(search_db, EJb, ECb, ELb, transitions, r_f, sample_f)
        self._emit(FitChangedPayload(has_result=self._state.fit.has_result))

    def selected_pointcloud(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._fit.selected_pointcloud()

    def render_selection(self, figure: Figure) -> None:
        """Draw published Filter evidence into caller-owned figure without a Session.

        Uses every spectrum\'s published sampled points and Filter mask. No state,
        draft or display selection is changed. Stored fit params overlay model
        curves, including stale/params-only fits, without residual markings;
        rendering failures propagate.
        """
        render_selection_result(figure, self.state.spectrums, fit=self.state.fit)

    def fit_auto_limits(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return Show's ((flux_min, flux_max), (GHz_min, GHz_max)) on owner.

        Recorded search uses only its detached capture backgrounds/calibration
        and original search points, even when outdated. Before search or with a
        params-only result, use current published backgrounds and selected points.
        Empty sources yield a unit box, expanded by published frequency references.
        No State or Figure changes. Foreign threads and missing recorded capture
        raise RuntimeError; a fit result is not required.
        """
        self.state.assert_owner_thread()
        fluxs, freqs = self.selected_pointcloud()
        return derive_fit_auto_limits(
            self.state.fit, self.state.spectrums, fluxs, freqs
        )

    def render_fit(
        self,
        figure: Figure,
        *,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        show_const_freq: bool = True,
        plot_transitions: TransitionDict | None = None,
    ) -> None:
        """Draw the native Show overlay into caller-owned figure on owner thread.

        Uses the recorded search's detached backgrounds/calibration and original
        points, even when stale; params-only results use current published sources
        and selected points. Does not publish state.
        Limits are normalized flux and GHz; None derives native auto limits.
        plot_transitions overrides the displayed lines only; None uses fit choices.
        show_const_freq toggles reference lines. Figure contents are replaced.
        Recorded search points keep their original coordinates and attribution;
        later selection changes cannot shift their colors/outlier markers. The
        title lists State's stale reasons. Missing result raises
        FailedPreconditionError(no_fit_result); simulation
        and render errors propagate. GUI and RPC use this same rendering path.
        """
        self.state.assert_owner_thread()
        fit = self.state.fit
        if fit.params is None:
            raise FailedPreconditionError(
                "no fit result to render", reason_code="no_fit_result"
            )
        fluxs, freqs = self.selected_pointcloud()
        display = FitDisplay(
            xlim=xlim,
            ylim=ylim,
            show_const_freq=show_const_freq,
            plot_transitions=plot_transitions,
            stale_reasons=tuple(self.state.fit_stale_reasons),
        )
        render_fit_result(figure, fit, self.state.spectrums, fluxs, freqs, display)

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

    def record_search_result(
        self, result: DatabaseSearchResult, inputs: SearchInput
    ) -> None:
        """Publish result with the exact SearchInput used to compute it, on owner.

        inputs carries detached original normalized-flux/GHz coordinates, point
        order and captured dependency versions. It is not reconstructed from
        live State. Late direct publication stays stale; search owner rejects
        changed inputs before delivery. Shape failures raise without a fact.
        """
        self._fit.record_result(result, inputs)
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
        inputs = self.capture_search()
        result = self._fit.compute_search(
            inputs,
            pbar_factory=pbar_factory,
            cancel_requested=cancel_requested,
        )
        self.record_search_result(result, inputs)
        return result

    def export_params(self, savepath: str | None = None) -> str:
        """Merge the current fit into params.json and return its output path.

        Owner-thread command. None uses params.json under project.result_dir.
        Independent sections are preserved. Calibration comes from the first
        aligned spectrum. FailedPreconditionError reports no_fit_result,
        fit_stale (with root-change reasons) or no_aligned_spectrum before I/O;
        native I/O errors propagate. State and resource observations do not change.
        """
        return self._fit.export_params(savepath)

    def export_params_receipt(self, savepath: str | None = None) -> FitExportReceipt:
        """Merge the fit into params.json and return captured output facts.

        Owner-thread command. None uses params.json under project.result_dir.
        Independent sections are preserved. The receipt holds full-precision
        (EJ, EC, EL) GHz and the first aligned spectrum used for calibration.
        FailedPreconditionError reports no_fit_result, fit_stale (with root-change
        reasons) or no_aligned_spectrum before I/O. Native I/O failures propagate;
        no receipt is returned on failure. State and observations do not change.
        """
        return self._fit.export_params_receipt(savepath)

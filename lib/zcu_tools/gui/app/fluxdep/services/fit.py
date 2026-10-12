"""FitService — database-search fit over the selected joint point cloud (v2).

The v2 pipeline tail: take the cross-spectrum selection's *selected* (flux, freq)
points, search a precomputed fluxonium database for the best (EJ, EC, EL), record
it on State, and export it as ``params.json``.

Pure, Qt-free, synchronous — like every fluxdep service. The slow ``search`` is
wrapped in a worker thread by the GUI (``ui/analyze_panel``); the remote interface
does not trigger a search.
``search`` accepts an optional progress-bar factory so the GUI worker can inject
a Qt-signalling ``BaseProgressBar`` via ``use_pbar_factory``; without one,
``search_database`` falls back to its tqdm default.

The search kernel and its numeric result are shared with the Notebook wrapper.
The GUI owns diagnostic rendering after recording the result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import (
    DatabaseSearchResult,
    ParamBounds,
    SearchExecution,
    database_info,
    search_database,
)
from zcu_tools.analysis.fluxdep.search_models import (
    transition_requires_levels,
    validate_transition_levels,
)
from zcu_tools.gui.app.fluxdep.state import (
    FluxDepState,
    PipelineStage,
    transitions_with_freqs,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.progress_bar import BaseProgressBar, use_pbar_factory
from zcu_tools.resources.qubit_params import (
    FluxDepFit,
    ParamsProject,
    QubitParams,
    params_path_for_result_dir,
)

logger = logging.getLogger(__name__)

PbarFactory = Callable[..., BaseProgressBar]


def default_params_path(result_dir: str) -> str:
    """The notebook-layout default params.json path (``<result_dir>/params.json``).

    ``result_dir`` is the project's result dir, which is always a concrete path
    (``ProjectInfo`` derives it from chip/qubit eagerly).
    """
    return params_path_for_result_dir(result_dir)


@dataclass(frozen=True)
class SearchInput:
    """Detached numeric input, captured on the State owner before worker use.

    search_db is the HDF5 database filename. fluxs is a copied 1-D cloud
    in Φ/Φ0; freqs is its copied GHz cloud. transitions is a deep copy with
    optional r_f/sample_f GHz values injected. bounds holds EJ/EC/EL GHz limits.
    The arrays and transition lists belong to this snapshot, never live State.
    """

    search_db: str
    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    transitions: TransitionDict
    bounds: ParamBounds
    # Owner-captured dependency graph/counters, including zero-point sources.
    stages: tuple[PipelineStage, ...]


class FitService:
    """Database-search fit over the selected cross-spectrum point cloud."""

    def __init__(self, state: FluxDepState) -> None:
        self._state = state

    # --- inputs ----------------------------------------------------------

    def set_params(
        self,
        search_db: str,
        EJb: tuple[float, float],
        ECb: tuple[float, float],
        ELb: tuple[float, float],
        transitions: TransitionDict,
        r_f: float | None,
        sample_f: float | None,
    ) -> None:
        """Validate transition choices, then replace inputs and clear results.

        Unknown categories raise InvalidInputError before I/O. Nonempty pairs
        and dynamic categories (even empty) read DB metadata to validate levels;
        basic empty categories do not require a file. I/O errors propagate.
        Missing transition frequencies remain search-time validation.
        """
        try:
            needs_levels = transition_requires_levels(transitions)
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
        if needs_levels:
            levels = database_info(search_db).levels
            try:
                validate_transition_levels(transitions, levels)
            except ValueError as exc:
                raise InvalidInputError(str(exc)) from exc
        self._state.set_fit_params(search_db, EJb, ECb, ELb, transitions, r_f, sample_f)

    # --- derived ---------------------------------------------------------

    def selected_pointcloud(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The *selected* subset of the joint (flux, freq) cloud — the search input.

        Concatenates every spectrum's selected points (insertion order), then
        applies the per-spectrum Filter masks projected in the same order.
        With no mask set, every point is included. Republished points are
        all-selected without changing any other spectrum's exclusions.
        """
        flux_parts: list[NDArray[np.float64]] = []
        freq_parts: list[NDArray[np.float64]] = []
        for entry in self._state.spectrums.values():
            flux_parts.append(np.asarray(entry.points["fluxs"], dtype=np.float64))
            freq_parts.append(np.asarray(entry.points["freqs"], dtype=np.float64))
        if flux_parts:
            fluxs = np.concatenate(flux_parts)
            freqs = np.concatenate(freq_parts)
        else:
            fluxs = np.empty(0, dtype=np.float64)
            freqs = np.empty(0, dtype=np.float64)

        mask = self._state.selection_mask()
        if mask is None:
            return fluxs, freqs
        return fluxs[mask], freqs[mask]

    # --- search ----------------------------------------------------------

    def capture_search(self) -> SearchInput:
        """Capture detached inputs on the State owner.

        Rejects foreign threads, missing database path and empty selected cloud.
        Kernel validation of bounds, transitions and database contents happens
        during compute_search.
        FailedPreconditionError reasons are no_search_db or no_selected_points.
        Foreign threads raise RuntimeError.
        """
        self._state.assert_owner_thread()
        fit = self._state.fit
        if not fit.search_db:
            raise FailedPreconditionError(
                "no database path set (call set_params first)",
                reason_code="no_search_db",
            )
        fluxs, freqs = self.selected_pointcloud()
        if fluxs.size == 0:
            raise FailedPreconditionError(
                "no selected points to fit (select points first)",
                reason_code="no_selected_points",
            )
        return SearchInput(
            search_db=fit.search_db,
            fluxs=fluxs.copy(),
            freqs=freqs.copy(),
            transitions=deepcopy(
                transitions_with_freqs(fit.transitions, fit.r_f, fit.sample_f)
            ),
            bounds=ParamBounds(EJ=fit.EJb, EC=fit.ECb, EL=fit.ELb),
            stages=self._state.pipeline_snapshot(),
        )

    def compute_search(
        self,
        inputs: SearchInput,
        *,
        pbar_factory: PbarFactory | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> DatabaseSearchResult:
        """Compute from explicit detached inputs without reading or writing State.

        inputs comes from capture_search on the owner. Safe on a worker thread;
        result carries numeric diagnostics, never a Figure. pbar_factory installs
        worker progress (None uses tqdm). cancel_requested is a quick thread-safe
        predicate; True at a kernel checkpoint raises SearchCancelled without a
        result. None disables cancellation. Other kernel failures propagate.
        In-flight HDF5/Numba work may delay the next cancellation checkpoint.
        """

        def _run() -> DatabaseSearchResult:
            return search_database(
                inputs.fluxs,
                inputs.freqs,
                inputs.search_db,
                inputs.transitions,
                inputs.bounds,
                execution=SearchExecution(cancel_requested=cancel_requested),
            )

        if pbar_factory is not None:
            with use_pbar_factory(pbar_factory):
                result = _run()
        else:
            result = _run()

        logger.debug("compute_search: params=%s", result.params)
        return result

    def record_result(self, result: DatabaseSearchResult) -> None:
        """Write a computed search result onto State (MAIN THREAD only).

        Separated from ``compute_search`` so the heavy search can run on a worker
        thread while this single State write happens on the Qt main thread, per
        the main-thread State invariant. ``search_database`` raises if no
        candidate is feasible, so a ``DatabaseSearchResult`` here always carries a real
        result.
        """
        self._state.set_fit_result(result.params, result=result)
        logger.debug("record_result: params=%s", result.params)

    # --- export ----------------------------------------------------------

    def export_params(self, savepath: str | None = None) -> str:
        """Write the fit result to ``params.json`` and return the path.

        Fast-fails without a search result. The flux alignment written is taken
        from the first aligned spectrum (the notebook stores a single
        flux_half/int/period; in a multi-spectrum session every spectrum is
        aligned to the same flux coordinate, so the first aligned one is
        representative). Before file I/O, FailedPreconditionError reports
        no_fit_result or no_aligned_spectrum. Native I/O failures propagate.
        """
        fit = self._state.fit
        if fit.params is None:
            raise FailedPreconditionError(
                "no fit result to export (run search first)",
                reason_code="no_fit_result",
            )

        reasons = self._state.fit_stale_reasons
        if reasons:
            raise FailedPreconditionError(
                f"fit is stale: {'; '.join(reasons)}",
                reason_code="fit_stale",
            )

        aligned = next((e for e in self._state.spectrums.values() if e.aligned), None)
        if aligned is None:
            raise FailedPreconditionError(
                "no aligned spectrum (align one before exporting)",
                reason_code="no_aligned_spectrum",
            )

        project = self._state.project
        path = (
            savepath
            if savepath is not None
            else default_params_path(project.result_dir)
        )

        EJ, EC, EL = fit.params
        params_file = QubitParams(path)
        params_file.ensure_project(
            ParamsProject(chip_name=project.chip_name, qub_name=project.qub_name)
        )
        params_file.set_fluxdep_fit(
            FluxDepFit(
                EJ=EJ,
                EC=EC,
                EL=EL,
                flux_half=aligned.flux_half,
                flux_int=aligned.flux_int,
                flux_period=aligned.flux_period,
                plot_transitions=fit.transitions,
            )
        )
        logger.debug("export_params: %r -> %r", fit.params, path)
        return path

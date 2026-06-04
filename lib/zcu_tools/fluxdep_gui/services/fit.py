"""FitService — database-search fit over the selected joint point cloud (v2).

The v2 pipeline tail: take the cross-spectrum selection's *selected* (flux, freq)
points, search a precomputed fluxonium database for the best (EJ, EC, EL), record
it on State, and export it as ``params.json``.

Pure, Qt-free, synchronous — like every fluxdep service. The slow ``search`` is
wrapped in a worker thread by the GUI (``ui/analyze_panel``); the RPC path runs it on
the main thread under a wider timeout (see the gui AI_NOTE for that trade-off).
``search`` accepts an optional progress-bar factory so the GUI worker can inject
a Qt-signalling ``BaseProgressBar`` via ``use_pbar_factory``; without one,
``search_in_database`` falls back to its tqdm default.

The numerical cores are reused verbatim from the notebook:
``search_in_database`` (database search) and ``dump_result`` (params.json).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.fluxdep_gui.state import FluxDepState, transitions_with_freqs
from zcu_tools.notebook.analysis.fluxdep.fitting import search_in_database
from zcu_tools.notebook.persistance import (
    FluxDepFitResult,
    TransitionDict,
    dump_result,
)
from zcu_tools.progress_bar import BaseProgressBar, use_pbar_factory

logger = logging.getLogger(__name__)

PbarFactory = Callable[..., BaseProgressBar]


@dataclass(frozen=True)
class SearchResult:
    """A completed database search — a pure value, no State touched.

    Returned by ``compute_search`` (runnable off the main thread) and handed to
    ``record_result`` (main thread) to write onto State. The diagnostic Figure is
    None on the RPC path (``plot=False``) and present in the GUI worker path.
    """

    params: tuple[float, float, float]  # (EJ, EC, EL)
    figure: Optional[Figure] = None


def default_params_path(result_dir: str, chip_name: str, qub_name: str) -> str:
    """The notebook-layout default params.json path (``<result_dir>/params.json``).

    Falls back to the chip/qubit-derived result dir (``result/<chip>/<qubit>``)
    when ``result_dir`` is unset, so the path is always well-formed (a bare
    ``params.json`` would make ``dump_result``'s ``makedirs(dirname)`` fail on the
    empty dirname).
    """
    from zcu_tools.fluxdep_gui.services.export import default_result_dir

    root = result_dir or default_result_dir(chip_name, qub_name)
    return os.path.join(root, "params.json")


class FitService:
    """Database-search fit over the selected cross-spectrum point cloud."""

    def __init__(self, state: FluxDepState) -> None:
        self._state = state

    # --- inputs ----------------------------------------------------------

    def set_params(
        self,
        database_path: str,
        EJb: tuple[float, float],
        ECb: tuple[float, float],
        ELb: tuple[float, float],
        transitions: TransitionDict,
        r_f: Optional[float],
        sample_f: Optional[float],
    ) -> None:
        """Record the search inputs (clears any stale result)."""
        self._state.set_fit_params(
            database_path, EJb, ECb, ELb, transitions, r_f, sample_f
        )

    # --- derived ---------------------------------------------------------

    def selected_pointcloud(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The *selected* subset of the joint (flux, freq) cloud — the search input.

        Concatenates every spectrum's selected points (insertion order), then
        applies the cross-spectrum selection mask. With no mask set, every point
        is included (the selector defaults to all-selected). Fast-fails if a
        stored mask disagrees with the current cloud size.
        """
        flux_parts: list[NDArray[np.float64]] = []
        freq_parts: list[NDArray[np.float64]] = []
        for entry in self._state.spectrums.values():
            flux_parts.append(np.asarray(entry.points["fluxs"], dtype=np.float64))
            freq_parts.append(np.asarray(entry.points["freqs"], dtype=np.float64))
        if not flux_parts:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty.copy()
        fluxs = np.concatenate(flux_parts)
        freqs = np.concatenate(freq_parts)

        mask = self._state.selection.selected
        if mask is None:
            return fluxs, freqs
        if mask.shape[0] != fluxs.shape[0]:
            raise ValueError(
                f"selection mask length {mask.shape[0]} != joint point cloud "
                f"size {fluxs.shape[0]} (re-run the cross-spectrum filter)"
            )
        return fluxs[mask], freqs[mask]

    # --- search ----------------------------------------------------------

    def compute_search(
        self,
        *,
        pbar_factory: Optional[PbarFactory] = None,
        plot: bool = False,
    ) -> SearchResult:
        """Run the database search and return its result — WITHOUT touching State.

        This is the pure, runnable-anywhere core: it snapshots the inputs and the
        selected point cloud off State *before* doing any work (a fast read on the
        caller's thread), then calls ``search_in_database``. It performs NO State
        write, so it is safe to run on a worker thread — the result is recorded
        separately on the main thread via ``record_result``.

        ``plot=True`` asks the search for its native diagnostic Figure (frequency
        comparison + per-parameter distance scatter). ``pbar_factory`` installs a
        custom progress-bar factory for the duration (a Qt-signalling bar in the
        GUI worker); without one, tqdm is used. Fast-fails when no database path
        is set or the selected cloud is empty.
        """
        fit = self._state.fit
        if not fit.database_path:
            raise ValueError("no database path set (call set_params first)")
        s_fluxs, s_freqs = self.selected_pointcloud()
        if s_fluxs.size == 0:
            raise ValueError("no selected points to fit (select points first)")

        # Snapshot every State-derived input now (this call may run on a worker
        # thread; reading State here is fine, writing it is not). Inject the
        # r_f / sample_f keys the transition model needs (only when provided).
        database_path = fit.database_path
        transitions = transitions_with_freqs(fit.transitions, fit.r_f, fit.sample_f)
        EJb, ECb, ELb = fit.EJb, fit.ECb, fit.ELb

        def _run() -> tuple[tuple[float, float, float], Optional[Figure]]:
            return search_in_database(
                s_fluxs,
                s_freqs,
                database_path,
                transitions,
                EJb,
                ECb,
                ELb,
                plot=plot,
            )

        if pbar_factory is not None:
            with use_pbar_factory(pbar_factory):
                params, figure = _run()
        else:
            params, figure = _run()

        logger.debug("compute_search: params=%s", params)
        return SearchResult(params=params, figure=figure)

    def record_result(self, result: SearchResult) -> None:
        """Write a computed search result onto State (MAIN THREAD only).

        Separated from ``compute_search`` so the heavy search can run on a worker
        thread while this single State write happens on the Qt main thread, per
        the main-thread State invariant. ``search_in_database`` raises if no
        candidate is feasible, so a ``SearchResult`` here always carries a real
        result; ``best_dist`` is not surfaced separately, so it is recorded as NaN.
        """
        self._state.set_fit_result(result.params, best_dist=float("nan"))
        logger.debug("record_result: params=%s", result.params)

    # --- export ----------------------------------------------------------

    def export_params(self, savepath: Optional[str] = None) -> str:
        """Write the fit result to ``params.json`` and return the path.

        Fast-fails without a search result. The flux alignment written is taken
        from the first aligned spectrum (the notebook stores a single
        flux_half/int/period; in a multi-spectrum session every spectrum is
        aligned to the same flux coordinate, so the first aligned one is
        representative). Fast-fails if no spectrum is aligned.
        """
        fit = self._state.fit
        if fit.params is None:
            raise ValueError("no fit result to export (run search first)")

        aligned = next((e for e in self._state.spectrums.values() if e.aligned), None)
        if aligned is None:
            raise ValueError("no aligned spectrum (align one before exporting)")

        project = self._state.project
        path = (
            savepath
            if savepath is not None
            else default_params_path(
                project.result_dir, project.chip_name, project.qub_name
            )
        )
        if not path:
            raise ValueError("no result_dir set and no savepath given")

        EJ, EC, EL = fit.params
        result = FluxDepFitResult(
            params={"EJ": EJ, "EC": EC, "EL": EL},
            flux_half=aligned.flux_half,
            flux_int=aligned.flux_int,
            flux_period=aligned.flux_period,
            plot_transitions=fit.transitions,
        )
        name = f"{project.chip_name}/{project.qub_name}"
        dump_result(path, name, fluxdep_fit=result)
        logger.debug("export_params: %r -> %r", fit.params, path)
        return path

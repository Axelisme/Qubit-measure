"""Exact fluxonium database search and cached database loading."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from h5py import File
from numba import set_num_threads
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.models import TransitionDict, energy2linearform
from zcu_tools.progress_bar import make_pbar

from .search_models import compile_transitions


class SearchCancelled(RuntimeError):
    """A supplied cancellation predicate stopped search without a partial result."""


@dataclass(frozen=True)
class SearchExecution:
    """Caller-owned worker policy shared by GUI and Notebook search callers.

    n_jobs: Numba thread count; <=0 selects available CPUs, default is 1.
    cancel_requested: Optional quick, worker-safe predicate. True at a search
    checkpoint raises SearchCancelled without returning partial results.
    Predicate exceptions propagate; None disables cooperative cancellation.
    """

    n_jobs: int = 1
    cancel_requested: Callable[[], bool] | None = None


@dataclass(frozen=True)
class ParamBounds:
    """EJ, EC and EL search intervals in GHz."""

    EJ: tuple[float, float]
    EC: tuple[float, float]
    EL: tuple[float, float]


@dataclass(frozen=True)
class DatabaseSearchResult:
    """Best fit and per-entry diagnostics; arrays are shared, not copied."""

    params: tuple[float, float, float]
    best_distance: float
    best_scale: float
    best_index: int
    entry_results: NDArray[np.float64]
    entry_params: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    predicted_freqs: NDArray[np.float64]
    bounds: ParamBounds


@lru_cache(maxsize=4)
def _load_database_cached(
    datapath: str, _mtime: float, _size: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load (fluxs, params, energies) from a fluxonium database, cached on the file.

    The database file (~290 MB for the full grid) is re-read on every search; the
    GUI re-runs the search against the *same* database many times (each parameter
    tweak), so reading it once and serving the cached arrays cuts ~0.13 s off every
    repeat call. The cache key includes the file's mtime and size, so editing or
    regenerating the database invalidates the entry rather than serving stale data.
    The returned arrays are the cache's own copies — callers must NOT mutate them.
    """
    with File(datapath, "r") as file:
        if "fluxs" in file:
            f_fluxs = file["fluxs"][:]  # (P,) # type: ignore[index]
        elif "flxs" in file:  # accepted misspelling in existing database files
            f_fluxs = file["flxs"][:]  # (P,) # type: ignore[index]
        else:
            raise KeyError("Database file must contain 'fluxs' or 'flxs' dataset.")
        f_params = file["params"][:]  # (N, 3) # type: ignore[index]
        f_energies = file["energies"][:]  # (N, P, M) # type: ignore[index]
    assert isinstance(f_fluxs, np.ndarray)
    assert isinstance(f_params, np.ndarray)
    assert isinstance(f_energies, np.ndarray)
    return (
        np.ascontiguousarray(f_fluxs, dtype=np.float64),
        np.ascontiguousarray(f_params, dtype=np.float64),
        np.ascontiguousarray(f_energies, dtype=np.float64),
    )


def load_database(
    datapath: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load (f_fluxs, f_params, f_energies) from a fluxonium database, file-cached.

    Stats the file for its (mtime, size) and serves a cached load when unchanged
    (see ``_load_database_cached``). Returns float64 C-contiguous arrays shared with
    the cache — treat them as read-only.
    """
    stat = os.stat(datapath)
    return _load_database_cached(datapath, stat.st_mtime, stat.st_size)


@dataclass(frozen=True)
class _PreparedSearch:
    fluxs: NDArray[np.float64]
    params: NDArray[np.float64]
    energies: NDArray[np.float64]
    pairs: NDArray[np.int32]
    coeffs: NDArray[np.float64]
    offsets: NDArray[np.float64]
    idxs: NDArray[np.int64]
    ws: NDArray[np.float64]
    interpolated: NDArray[np.float64]


def _check_cancellation(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise SearchCancelled("database search cancelled")


def _prepare_search(
    fluxs: NDArray[np.float64],
    datapath: str,
    transitions: TransitionDict,
    cancel_requested: Callable[[], bool] | None,
) -> _PreparedSearch:
    """Load and interpolate only levels referenced by compiled transitions."""
    from .search_njit import _apply_interp, _interp_weights

    # Load the database (file-cached: the GUI re-runs the search against the same
    # database on every parameter tweak, so the ~290 MB read is amortised).
    f_fluxs, f_params, f_energies = load_database(datapath)
    _check_cancellation(cancel_requested)
    M = f_energies.shape[2]

    # Pre-compile transitions once so the hot loop calls only nogil njit code.
    tr_pairs, tr_coeffs, tr_offsets = compile_transitions(transitions, M)

    # Only the energy levels actually referenced by the transitions enter the
    # linear form, so interpolate (and carry into the parallel kernel) just those
    # levels instead of all M. This shrinks the interpolated array — for the usual
    # 0->1/0->2/1->2 set that is 3 of 15 levels, a ~5x smaller working set — which
    # both speeds the interpolation and lets the bandwidth-bound parallel kernel
    # scale better. The transition pairs are remapped to the reduced level index
    # space so the linear form is numerically identical.
    used_levels = np.unique(tr_pairs.reshape(-1)) if tr_pairs.size else np.arange(M)
    level_pos = np.full(M, -1, dtype=np.int64)
    level_pos[used_levels] = np.arange(used_levels.shape[0])
    tr_pairs_reduced = level_pos[tr_pairs].astype(np.int32)

    # Interpolate points. f_fluxs is strictly increasing and shared across all
    # entries, so precompute (idx, w) once then apply with a parallel njit
    # kernel — avoids N*M Python-level np.interp calls.
    fluxs = np.mod(fluxs, 1.0)
    fluxs_c = np.ascontiguousarray(fluxs, dtype=np.float64)
    idxs, ws = _interp_weights(fluxs_c, f_fluxs)
    energies_used = np.ascontiguousarray(f_energies[:, :, used_levels])
    sf_energies = _apply_interp(energies_used, idxs, ws)
    _check_cancellation(cancel_requested)

    return _PreparedSearch(
        fluxs,
        f_params,
        f_energies,
        tr_pairs_reduced,
        tr_coeffs,
        tr_offsets,
        idxs,
        ws,
        sf_energies,
    )


def _scan_candidates(
    prepared: _PreparedSearch,
    freqs: NDArray[np.float64],
    bounds: ParamBounds,
    n_jobs: int,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[int, float, float, NDArray[np.float64], NDArray[np.float64], bool]:
    """Scan feasible entries while keeping the progress and interruption behavior."""
    from .search_njit import _lower_bound_kernel, search_one_entry

    f_params, sf_energies = prepared.params, prepared.interpolated
    tr_pairs_reduced, tr_coeffs, tr_offsets = (
        prepared.pairs,
        prepared.coeffs,
        prepared.offsets,
    )
    EJb, ECb, ELb = bounds.EJ, bounds.EC, bounds.EL
    # Initialize variables
    best_idx, best_scale, best_dist = 0, 1.0, np.inf
    best_params = np.full(3, np.nan)
    # results[i] = (mean distance, scale) per entry. The exact path fills only the
    # entries it actually searches (the prune skips provably-worse ones); the rest
    # keep their lower bound (a valid distance floor) for the diagnostic scatter.
    results = np.full((f_params.shape[0], 2), np.nan)

    # Ensure contiguous float64 for njit signature.
    sf_energies_c = np.ascontiguousarray(sf_energies, dtype=np.float64)
    f_params_c = np.ascontiguousarray(f_params, dtype=np.float64)
    freqs_c = np.ascontiguousarray(freqs, dtype=np.float64)

    set_num_threads(n_jobs if n_jobs > 0 else (os.cpu_count() or 1))

    # Exact search with a lower-bound prune. The objective per entry is
    # F(a) = mean_i min_j |A_i - |a*B_ij + C_ij||; ``entry_lower_bound`` gives a
    # valid floor LB(entry) <= min_a F(a) in O(N*K²). Searching entries in
    # increasing-LB order while tracking the incumbent best distance lets us STOP
    # once LB > incumbent — every remaining entry provably cannot beat it, so the
    # winner is IDENTICAL to scanning all entries, but typically only a tiny
    # fraction are fully searched (the true match sorts to the front and drives
    # the incumbent to ~0). The parallel LB pass is cheap; the exact per-entry
    # ``candidate_breakpoint_search`` is the part we avoid for pruned entries.
    lbs = _lower_bound_kernel(
        sf_energies_c,
        f_params_c,
        tr_pairs_reduced,
        tr_coeffs,
        tr_offsets,
        freqs_c,
        EJb[0],
        EJb[1],
        ECb[0],
        ECb[1],
        ELb[0],
        ELb[1],
    )
    _check_cancellation(cancel_requested)
    results[:, 0] = lbs  # unsearched entries keep their LB for the scatter
    order = np.argsort(lbs)
    idx_bar = make_pbar(total=f_params.shape[0], desc="Searching...")
    searched = 0
    interrupted = False
    # Retain scan/progress interrupts, but let a predicate's interrupt propagate.
    checking_cancellation = False
    try:
        for oi in order:
            checking_cancellation = cancel_requested is not None
            _check_cancellation(cancel_requested)
            checking_cancellation = False
            oi = int(oi)
            lb = lbs[oi]
            if not np.isfinite(lb) or lb > best_dist:
                break  # all remaining entries have LB >= this -> cannot win
            p0, p1, p2 = f_params[oi]
            a_min = max(EJb[0] / p0, ECb[0] / p1, ELb[0] / p2)
            a_max = min(EJb[1] / p0, ECb[1] / p1, ELb[1] / p2)
            d, a = search_one_entry(
                sf_energies_c[oi],
                tr_pairs_reduced,
                tr_coeffs,
                tr_offsets,
                freqs_c,
                a_min,
                a_max,
            )
            results[oi] = d, a
            searched += 1
            if searched % 64 == 0:
                idx_bar.update(64)
            if d < best_dist:
                best_dist, best_scale, best_idx = d, a, oi
                best_params = f_params[oi] * a
            checking_cancellation = cancel_requested is not None
            _check_cancellation(cancel_requested)
            checking_cancellation = False
        idx_bar.set_description("Done! ")
    except KeyboardInterrupt:
        if checking_cancellation:
            raise
        interrupted = True
    finally:
        idx_bar.close()

    return best_idx, best_scale, best_dist, best_params, results, interrupted


def _find_close_points(freqs, energies, scale, allows) -> np.ndarray:
    Bs, Cs = energy2linearform(energies, allows)
    fs = np.abs(scale * Bs + Cs)
    dists = np.abs(fs - freqs[:, None])
    min_idx = np.argmin(dists, axis=1)
    return fs[range(len(freqs)), min_idx]


def search_database(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    datapath: str,
    transitions: TransitionDict,
    bounds: ParamBounds,
    *,
    execution: SearchExecution | None = None,
) -> DatabaseSearchResult:
    """Search a fluxonium HDF5 database for exact scale candidates.

    fluxs and freqs are paired one-dimensional flux/GHz arrays. transitions
    describes allowed level pairs; bounds supplies EJ/EC/EL intervals in GHz.
    execution supplies the Numba thread count and optional stop predicate.
    None uses one Numba thread without cooperative cancellation.
    Return the best candidate and numeric diagnostics; infeasible bounds raise
    RuntimeError. File, transition, and numerical failures propagate.

    execution.cancel_requested is a quick worker-safe predicate owned by the caller.
    True at a checkpoint raises SearchCancelled, never a partial result.
    Predicate exceptions propagate. Checks surround preparation, lower-bound
    scanning, exact-entry work and diagnostic reconstruction; an in-flight
    HDF5/Numba call must return first, so cancellation has no fixed latency.
    None disables cooperative cancellation. KeyboardInterrupt during exact
    scanning still returns the best-so-far result with a RuntimeWarning.
    """
    from .search_njit import _apply_interp

    execution = SearchExecution() if execution is None else execution
    cancel_requested = execution.cancel_requested
    _check_cancellation(cancel_requested)
    prepared = _prepare_search(fluxs, datapath, transitions, cancel_requested)
    fluxs = prepared.fluxs
    f_params, f_energies = prepared.params, prepared.energies
    idxs, ws = prepared.idxs, prepared.ws

    best_idx, best_scale, best_dist, best_params, results, interrupted = (
        _scan_candidates(prepared, freqs, bounds, execution.n_jobs, cancel_requested)
    )

    if not np.isfinite(best_dist):
        raise RuntimeError(
            "No valid candidate found in database (all parameter bounds infeasible)."
        )
    if interrupted:
        warnings.warn(
            "Database search was interrupted; returning the best-so-far partial result.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Reconstruct the full transition prediction for the diagnostic builder.
    best_full = _apply_interp(
        np.ascontiguousarray(f_energies[best_idx : best_idx + 1]), idxs, ws
    )[0]
    # An empty transition set has no diagnostic line. The original non-plot
    # search returned successfully in this case; only plotting attempted the
    # invalid empty linear form.
    if any(isinstance(value, list) and value for value in transitions.values()):
        p_freqs = _find_close_points(freqs, best_full, best_scale, transitions)
    else:
        p_freqs = np.empty(0, dtype=np.float64)
    _check_cancellation(cancel_requested)
    return DatabaseSearchResult(
        params=(float(best_params[0]), float(best_params[1]), float(best_params[2])),
        best_distance=float(best_dist),
        best_scale=float(best_scale),
        best_index=best_idx,
        entry_results=results,
        entry_params=f_params,
        fluxs=fluxs,
        freqs=freqs,
        predicted_freqs=p_freqs,
        bounds=bounds,
    )

"""PreprocessService — the one-tone signal-preprocessing pipeline (notebook cells 5-6).

Extracts the normalized phase image the dispersive tuning / fit work against from
the raw complex S-parameter signals: fit + remove the electronic delay, smooth,
fit a common circle centre, take the phase, then differentiate / abs / row-normalize.

Pure and Qt-free. The heavy per-flux ``fit_edelay`` (a 1000-point grid search each)
runs under joblib ``Parallel`` in the loky PROCESS pool (the default) — it is scipy
``linalg.eig``-bound and does NOT release the GIL, so a threading backend is ~4x
slower (measured); only real processes parallelise it. ``return_as="generator"``
yields results in submission order as workers finish, so the progress bar is driven
on the calling (worker) thread while the parallel jobs run the pure, picklable
``fit_edelay`` and never touch the bar (a ``GuiProgressBar``'s QObject channel cannot
survive a fork — same shape as fluxdep's search). The whole ``compute`` is run on a
worker thread by the GUI so it does not block the event loop. Reuses the resonance
fitting primitives verbatim from ``zcu_tools.utils.fitting.resonance``.
"""

from __future__ import annotations

import logging

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from zcu_tools.gui.app.dispersive.state import DispersiveState, PreprocessResult
from zcu_tools.progress_bar import make_pbar
from zcu_tools.utils.fitting.resonance import (
    calc_phase,
    fit_circle_params,
    fit_edelay,
    remove_edelay,
)

logger = logging.getLogger(__name__)

# Smoothing divisors (the notebook's hard-coded factors): the per-row gaussian σ
# is ``n_freq // EDELAY_SMOOTH_DIV`` before the circle fit, and
# ``n_freq // PHASE_SMOOTH_DIV`` before the phase difference. They are part of the
# preprocessing signature so a re-run with different smoothing invalidates a fit.
EDELAY_SMOOTH_DIV = 30
PHASE_SMOOTH_DIV = 10


def _smooth_sigma(n_freq: int, divisor: int) -> int:
    """Gaussian σ = ``n_freq // divisor``, floored at 1.

    ``gaussian_filter1d`` divides by σ², so a σ of 0 (a coarse grid with fewer
    than ``divisor`` frequency points) raises ZeroDivisionError. Flooring at 1 is
    a harmless single-point smooth — the notebook's grids are large enough that
    the floor never binds there, but the GUI must not crash on a small spectrum.
    """
    return max(1, n_freq // divisor)


def compute_preprocess(
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    *,
    n_jobs: int = -1,
) -> PreprocessResult:
    """Run the preprocessing pipeline on a raw one-tone spectrum (pure, off-main-safe).

    ``signals`` is the (n_flux, n_freq) complex grid; ``sp_freqs`` is in GHz. Returns
    the ``PreprocessResult`` (norm_phases + axes + edelay diagnostics). Drives the
    active ``make_pbar`` over the per-flux edelay fits.
    """
    n_flux = signals.shape[0]
    # ``fit_edelay`` is the heavy step (a 1000-point grid search, each point a scipy
    # ``linalg.eig`` circle fit) and does NOT release the GIL — so it must run in
    # the loky PROCESS pool (the joblib default), not a threading backend, to
    # actually parallelise (sharedmem threads are ~4x SLOWER here, measured).
    # ``return_as="generator"`` yields results in submission order as they complete,
    # so the progress bar is driven HERE on the worker thread and the parallel jobs
    # run the pure picklable ``fit_edelay`` and never see the bar (a GuiProgressBar's
    # QObject channel cannot survive a fork — same shape as fluxdep's search).
    pbar = make_pbar(total=n_flux, desc="Fitting edelay")
    try:
        edelays = np.empty(n_flux, dtype=np.float64)
        results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(fit_edelay)(sp_freqs, sig) for sig in signals
        )
        for i, value in enumerate(results):
            edelays[i] = value
            pbar.update(1)
    finally:
        pbar.close()
    edelay = float(np.median(edelays))

    n_freq = int(signals.shape[1])
    rot_signals = remove_edelay(sp_freqs, signals, edelay)
    rot_signals = gaussian_filter1d(
        rot_signals, _smooth_sigma(n_freq, EDELAY_SMOOTH_DIV), axis=1
    )
    rot_signals = np.asarray(rot_signals, dtype=np.complex128)

    circle_param = np.median(
        [fit_circle_params(s.real, s.imag) for s in rot_signals], axis=0
    )
    phases = calc_phase(rot_signals, circle_param[0], circle_param[1], axis=1)

    norm_phases = gaussian_filter1d(
        phases, _smooth_sigma(phases.shape[1], PHASE_SMOOTH_DIV), axis=1
    )
    norm_phases = np.diff(norm_phases, axis=1, prepend=norm_phases[:, :1])
    norm_phases = np.abs(norm_phases)
    norm_phases /= np.max(norm_phases, axis=1, keepdims=True)

    # Data-derived r_f seed: each flux row's peak (the resonance), median over flux
    # (robust to outlier rows). The slider defaults here.
    peak_freqs = sp_freqs[np.argmax(norm_phases, axis=1)]
    median_rf = float(np.median(peak_freqs))

    return PreprocessResult(
        sp_fluxs=sp_fluxs.astype(np.float64),
        sp_freqs=sp_freqs.astype(np.float64),
        norm_phases=norm_phases.astype(np.float64),
        edelays=edelays,
        edelay=edelay,
        median_rf=median_rf,
        signature=(EDELAY_SMOOTH_DIV, PHASE_SMOOTH_DIV, n_flux, n_freq),
    )


class PreprocessService:
    """Runs the preprocessing pipeline on the loaded one-tone, writes the result."""

    def __init__(self, state: DispersiveState) -> None:
        self._state = state

    def compute(self, *, n_jobs: int = -1) -> PreprocessResult:
        """Run the pipeline on the loaded one-tone — pure, off-main-safe (no State write).

        Snapshots the spectrum off State first (a fast read), then runs the heavy
        pipeline. Pair with ``record`` on the main thread. Fast-fails when no
        one-tone is loaded.
        """
        entry = self._state.onetone
        if entry is None:
            raise RuntimeError("no one-tone spectrum loaded (call load_onetone first)")
        raw = entry.raw
        return compute_preprocess(
            raw["fluxs"], raw["freqs"], raw["signals"], n_jobs=n_jobs
        )

    def record(self, result: PreprocessResult) -> None:
        """Write a computed preprocessing result onto State (MAIN THREAD only)."""
        self._state.set_preprocess(result)

    def preprocess(self, *, n_jobs: int = -1) -> PreprocessResult:
        """Compute + record inline (RPC / convenience path, main thread)."""
        result = self.compute(n_jobs=n_jobs)
        self.record(result)
        return result

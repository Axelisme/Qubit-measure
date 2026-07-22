"""Fast per-flux electronic-delay fit (numba), dispersive-gui-local.

A GUI-local, ~14x-faster replacement for looping ``zcu_tools.utils.fitting.resonance.fit_edelay``
over every flux row: the whole (n_flux × grid) double loop is JIT-compiled into one
``@njit(parallel=True)`` kernel with the per-flux outer loop in ``prange``. Two
optimizations vs the utility path:

- the circle fit at each grid point uses an inlined **Kasa** algebraic least-squares
  (a 2×2 solve) instead of the utility's ``scipy.linalg.eig`` — same centre, radius
  within ~2e-3 (measured), ~4x faster per call;
- numba releases the GIL, so the parallelism is real and needs no process fork (so a
  Qt ``GuiProgressBar`` is never pickled across a worker boundary).

This is intentionally NOT in ``zcu_tools.utils`` — it specializes the local circle
refinement for the dispersive preprocessing hot path. Global branch discovery reuses
the shared utility once across all flux rows; the physical cable delay is common, while
the numba kernel retains per-row local refinement.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from zcu_tools.utils.fitting.resonance.base import find_edelay_branch

# Electronic-delay grid: search ±5/(freq span) over this many points (the utility's
# fit_edelay uses 1000; numba makes 1000 cheap enough to keep full precision).
_N_GRID = 1000


@njit(parallel=True, fastmath=True, cache=True)
def _edelay_kernel(
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    seeds: NDArray[np.float64],
    fit_range: float,
) -> NDArray[np.float64]:
    """Per-flux edelay via grid search + inlined Kasa circle fit (parallel over flux)."""
    n_flux, n_freq = signals.shape
    out = np.empty(n_flux, dtype=np.float64)
    grid = np.linspace(-fit_range, fit_range, _N_GRID)
    two_pi = 2.0 * np.pi
    for i in prange(n_flux):
        # remove the shared branch seed once, then search residual delays on the grid
        s2 = np.exp(1j * two_pi * freqs * seeds[i]) * signals[i]
        best_loss = 1e18
        best_j = 0
        xs = np.empty(n_freq, dtype=np.float64)
        ys = np.empty(n_freq, dtype=np.float64)
        for j in range(_N_GRID):
            ed = grid[j]
            for f in range(n_freq):
                c = np.exp(1j * two_pi * freqs[f] * ed) * s2[f]
                xs[f] = c.real
                ys[f] = c.imag
            mx = xs.mean()
            my = ys.mean()
            # Kasa moments (centred)
            Suu = Svv = Suv = 0.0
            Suuu = Svvv = Suvv = Svuu = 0.0
            for f in range(n_freq):
                u = xs[f] - mx
                v = ys[f] - my
                Suu += u * u
                Svv += v * v
                Suv += u * v
                Suuu += u * u * u
                Svvv += v * v * v
                Suvv += u * v * v
                Svuu += v * u * u
            det = Suu * Svv - Suv * Suv
            b0 = 0.5 * (Suuu + Suvv)
            b1 = 0.5 * (Svvv + Svuu)
            uc = (b0 * Svv - Suv * b1) / det
            vc = (Suu * b1 - Suv * b0) / det
            cx = uc + mx
            cy = vc + my
            r0 = np.sqrt(uc * uc + vc * vc + (Suu + Svv) / n_freq)
            # loss = sum (r0 - |signal - centre|)^2
            loss = 0.0
            for f in range(n_freq):
                d = np.sqrt((xs[f] - cx) ** 2 + (ys[f] - cy) ** 2)
                loss += (r0 - d) ** 2
            if loss < best_loss:
                best_loss = loss
                best_j = j
        out[i] = grid[best_j] + seeds[i]
    return out


def fast_edelays(
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    *,
    search_radius: float | None = None,
) -> NDArray[np.float64]:
    """The per-flux electronic delays for a (n_flux, n_freq) signal grid.

    ``freqs`` is the shared frequency axis; ``signals[i]`` is flux row ``i``. Returns
    one edelay per flux row (the median over these is the spectrum's edelay). One
    branch seed is estimated from all rows, then the numba kernel performs the
    local circle refinement for every row in parallel.
    """
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    signals = np.ascontiguousarray(signals, dtype=np.complex128)
    if freqs.ndim != 1 or signals.ndim != 2:
        raise ValueError("fast edelay fit expects a 1D frequency axis and 2D signals")
    if signals.shape[1] != len(freqs):
        raise ValueError("fast edelay frequency and signal lengths must match")
    branch_seed = find_edelay_branch(
        freqs,
        signals,
        search_radius=search_radius,
    )
    seeds = np.full(signals.shape[0], branch_seed, dtype=np.float64)
    fit_range = 5.0 / float(np.ptp(freqs))
    return _edelay_kernel(freqs, signals, seeds, fit_range)

"""Auto-tune g / bare_rf against the sample-flux lines (scipy optimiser).

The user drops a few sample-flux lines; auto-tune searches for the (g, bare_rf) that
makes the predicted ground/excited resonator frequencies fall on the *strongest* part
of the normalised-phase image at those fluxes. The objective is

    score(g, bare_rf) = mean over sample fluxes of
                        max( norm_phase(flux, rf_ground), norm_phase(flux, rf_excited) )

where ``norm_phase`` is read by bilinear interpolation on the (sp_fluxs, sp_freqs)
grid, and ``rf_ground`` / ``rf_excited`` come from the fast dispersive prediction.
norm_phase is large where the dispersive feature is, so we MAXIMISE the score
(``scipy.optimize.minimize`` on its negative).

Pure, Qt-free, no State write — the iterative optimisation runs on a worker thread
(it can take a while); only the resulting (g, bare_rf) is recorded on the main thread.
The dispersive prediction is non-smooth (eigensolve + dressed labelling) and the
bilinear interpolation is only piecewise-linear, so a derivative-free Nelder-Mead with
manual bound clamping is used rather than a gradient method.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .predict import predict_dispersive_at

logger = logging.getLogger(__name__)


def _interp_norm_phase(
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    norm_phases: NDArray[np.float64],
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bilinear ``norm_phases`` lookup at (flux, freq) points, clamped to the grid.

    ``norm_phases[i, j]`` is the value at ``(sp_fluxs[i], sp_freqs[j])``. Query points
    are CLIPPED into the grid range first, so an out-of-band predicted frequency reads
    the nearest edge value — not a linearly-extrapolated (and possibly negative,
    spurious) one, which would mislead the optimiser toward off-grid frequencies.
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (sp_fluxs, sp_freqs),
        norm_phases,
        method="linear",
        bounds_error=False,
    )
    fc = np.clip(fluxs, sp_fluxs[0], sp_fluxs[-1])
    qc = np.clip(freqs, sp_freqs[0], sp_freqs[-1])
    pts = np.column_stack([fc, qc])
    return np.asarray(interp(pts), dtype=np.float64)


def sample_score(
    params: tuple[float, float, float],
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    norm_phases: NDArray[np.float64],
    sample_fluxs: NDArray[np.float64],
    g: float,
    bare_rf: float,
) -> float:
    """The auto-tune objective at one (g, bare_rf): mean over sample fluxes of the
    larger of the ground/excited norm-phase magnitudes. Higher = better match."""
    rf_0, rf_1 = predict_dispersive_at(params, sample_fluxs, g, bare_rf)
    v0 = _interp_norm_phase(sp_fluxs, sp_freqs, norm_phases, sample_fluxs, rf_0)
    v1 = _interp_norm_phase(sp_fluxs, sp_freqs, norm_phases, sample_fluxs, rf_1)
    return float(np.mean(np.maximum(v0, v1)))


def auto_tune(
    params: tuple[float, float, float],
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    norm_phases: NDArray[np.float64],
    sample_fluxs: NDArray[np.float64],
    g0: float,
    bare_rf0: float,
    g_bounds: tuple[float, float],
    rf_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Optimise (g, bare_rf) to maximise ``sample_score`` from the seed (g0, bare_rf0).

    Returns the best (g, bare_rf) found within the bounds. Fast-fails if there are no
    sample fluxes (the objective is undefined). Runnable on a worker thread (no State,
    no Qt). Nelder-Mead is bounded by clamping inside the objective + rejecting
    out-of-range points with a large penalty.
    """
    from scipy.optimize import minimize

    if sample_fluxs.size == 0:
        raise ValueError("no sample fluxes to auto-tune against (add sample lines)")

    g_lo, g_hi = g_bounds
    rf_lo, rf_hi = rf_bounds

    def neg_score(x: NDArray[np.float64]) -> float:
        g, bare_rf = float(x[0]), float(x[1])
        # Reject out-of-bounds with a finite penalty so Nelder-Mead stays in-domain
        # (it has no native bound support).
        if not (g_lo <= g <= g_hi and rf_lo <= bare_rf <= rf_hi):
            return 1e6
        try:
            return -sample_score(
                params, sp_fluxs, sp_freqs, norm_phases, sample_fluxs, g, bare_rf
            )
        except Exception:  # noqa: BLE001 — a bad eval must not abort the optimisation
            logger.exception("auto-tune objective failed at g=%s rf=%s", g, bare_rf)
            return 1e6

    # Initial simplex scaled to the parameter ranges so the search explores both axes.
    g_step = 0.05 * (g_hi - g_lo)
    rf_step = 0.05 * (rf_hi - rf_lo)
    x0 = np.array([g0, bare_rf0], dtype=np.float64)
    simplex = np.array([x0, x0 + [g_step, 0.0], x0 + [0.0, rf_step]], dtype=np.float64)
    res = minimize(
        neg_score,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": simplex,
            "xatol": 1e-5,
            "fatol": 1e-4,
            "maxiter": 400,
        },
    )

    g = float(np.clip(res.x[0], g_lo, g_hi))
    bare_rf = float(np.clip(res.x[1], rf_lo, rf_hi))
    logger.debug("auto_tune: g=%s bare_rf=%s score=%s", g, bare_rf, -res.fun)
    return g, bare_rf

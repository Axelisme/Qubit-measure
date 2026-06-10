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

# The coarse global scan grid (a 2D r_f × g grid evaluated before the local refine).
# This makes auto-tune robust to a far / decoy seed: a purely local optimiser can
# stick in a spurious phase band's basin, but the grid finds the global best region
# first. ~500 single-point predictions (~ms each) ≈ a couple of seconds on a worker.
_COARSE_N_RF = 50
_COARSE_N_G = 10


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


def _coarse_seed(
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
    """The highest-scoring (g, bare_rf) on a coarse grid (plus the current g0/rf0).

    A global pre-scan: evaluate ``sample_score`` on a ``_COARSE_N_RF × _COARSE_N_G``
    grid over the bounds and return the best point, also considering the caller's
    current slider position so a good manual guess is never lost to the grid. This
    becomes the seed for the local refine, so the optimiser starts in the globally
    best region rather than wherever the slider happened to be.
    """
    g_lo, g_hi = g_bounds
    rf_lo, rf_hi = rf_bounds
    gs = np.linspace(g_lo, g_hi, _COARSE_N_G)
    rfs = np.linspace(rf_lo, rf_hi, _COARSE_N_RF)

    best_score = sample_score(
        params, sp_fluxs, sp_freqs, norm_phases, sample_fluxs, g0, bare_rf0
    )
    best = (g0, bare_rf0)
    for rf in rfs:
        for g in gs:
            s = sample_score(
                params,
                sp_fluxs,
                sp_freqs,
                norm_phases,
                sample_fluxs,
                float(g),
                float(rf),
            )
            if s > best_score:
                best_score = s
                best = (float(g), float(rf))
    logger.debug("coarse seed: g=%s bare_rf=%s score=%s", best[0], best[1], best_score)
    return best


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
    """Optimise (g, bare_rf) to maximise ``sample_score``.

    A coarse 2D grid scan over the bounds (plus the current g0/bare_rf0) picks the
    globally best region, then Nelder-Mead refines from there — so the result does not
    depend on the seed being near the answer (a purely local search can stick in a
    spurious phase band's basin). Returns the best (g, bare_rf) within the bounds.
    Fast-fails if there are no sample fluxes. Runnable on a worker thread (no State,
    no Qt). Nelder-Mead is bounded by clamping inside the objective + rejecting
    out-of-range points with a large penalty.
    """
    from scipy.optimize import minimize

    if sample_fluxs.size == 0:
        raise ValueError("no sample fluxes to auto-tune against (add sample lines)")

    g_lo, g_hi = g_bounds
    rf_lo, rf_hi = rf_bounds

    # Global pre-scan: seed the local refine from the best coarse-grid point.
    g0, bare_rf0 = _coarse_seed(
        params,
        sp_fluxs,
        sp_freqs,
        norm_phases,
        sample_fluxs,
        g0,
        bare_rf0,
        g_bounds,
        rf_bounds,
    )

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

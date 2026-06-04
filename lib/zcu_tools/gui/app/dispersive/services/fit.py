"""FitService — scipy auto-fit of the coupling g (and optionally bare_rf).

Ports the notebook's ``auto_fit_dispersive`` (cell 8): maximize the overlap of the
predicted ground/excited dispersive frequencies with the measured one-tone signal,
via ``scipy.optimize.minimize`` (L-BFGS-B). Frequencies are GHz throughout.

Split like fluxdep's FitService: ``compute_autofit`` is pure and off-main-safe
(snapshots its inputs off State, runs the heavy scqubits loss, returns a value);
``record_result`` does the single State write on the Qt main thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.progress_bar import BaseProgressBar, make_pbar, use_pbar_factory
from zcu_tools.simulate.fluxonium import calculate_dispersive_vs_flux

logger = logging.getLogger(__name__)

PbarFactory = Callable[..., BaseProgressBar]

_MAX_ITER = 1000
_MAX_POINTS = 300
# bare_rf is fit within ±2 MHz of its seed (the notebook's bound, GHz).
_BARE_RF_WINDOW = 2e-3


@dataclass(frozen=True)
class AutoFitResult:
    """A completed auto-fit — a pure value, no State touched.

    ``bare_rf`` is None when ``fit_bare_rf`` was False (the seed bare_rf is kept).
    Returned by ``compute_autofit`` (off-main) and handed to ``record_result``.
    """

    g: float  # GHz
    bare_rf: Optional[float]  # GHz, or None when not fit


def auto_fit_dispersive(
    params: tuple[float, float, float],
    bare_rf: float,
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    norm_phases: NDArray[np.float64],
    g_bound: tuple[float, float],
    g_init: float,
    fit_bare_rf: bool,
) -> AutoFitResult:
    """Fit g (and optionally bare_rf) by overlap maximization (pure, off-main-safe).

    Down-samples to ``_MAX_POINTS`` flux points, then minimizes the negative mean
    overlap of the predicted ground/excited frequencies with the (real) signal.
    Drives the active ``make_pbar`` over the optimizer iterations.
    """
    real_signals = np.abs(norm_phases)
    if len(sp_fluxs) > _MAX_POINTS:
        idx = np.round(np.linspace(0, len(sp_fluxs) - 1, _MAX_POINTS)).astype(int)
        sp_fluxs = sp_fluxs[idx]
        real_signals = real_signals[idx]

    pbar = make_pbar(total=_MAX_ITER, desc="Auto fitting g")
    ftol = float(np.max(real_signals)) * 1e-4

    def loss_fn(g: float, rf: float) -> float:
        pbar.update(1)
        rf_0, rf_1 = calculate_dispersive_vs_flux(
            params, sp_fluxs, rf, g, progress=False, res_dim=4
        )
        vals = [
            max(np.interp(a, sp_freqs, sig), np.interp(b, sp_freqs, sig))
            for a, b, sig in zip(rf_0, rf_1, real_signals)
        ]
        return -float(np.mean(vals))

    fit_kwargs = dict(
        method="L-BFGS-B",
        options={"disp": False, "maxiter": _MAX_ITER, "ftol": ftol},
    )
    try:
        if fit_bare_rf:
            res = minimize(
                lambda p: loss_fn(p[0], p[1]),
                x0=[g_init, bare_rf],
                bounds=[
                    g_bound,
                    [bare_rf - _BARE_RF_WINDOW, bare_rf + _BARE_RF_WINDOW],
                ],
                **fit_kwargs,
            )
            x = res if isinstance(res, np.ndarray) else res.x
            return AutoFitResult(g=float(x[0]), bare_rf=float(x[1]))

        res = minimize(
            lambda p: loss_fn(p[0], bare_rf),
            x0=[g_init],
            bounds=[g_bound],
            **fit_kwargs,
        )
        x = res if isinstance(res, np.ndarray) else res.x
        return AutoFitResult(g=float(x[0]), bare_rf=None)
    finally:
        pbar.close()


class FitService:
    """Auto-fit g / bare_rf over the preprocessed one-tone signal."""

    def __init__(self, state: DispersiveState) -> None:
        self._state = state

    def compute_autofit(
        self, *, pbar_factory: Optional[PbarFactory] = None
    ) -> AutoFitResult:
        """Run the auto-fit — pure, off-main-safe (reads State, no write).

        Snapshots every State-derived input now (safe on a worker thread), then
        runs the scipy fit. Fast-fails when no preprocessing result or no tuning
        bare_rf is available. Pair with ``record_result`` on the main thread.
        """
        pp = self._state.preprocess
        if pp is None:
            raise RuntimeError("no preprocessing result (run preprocessing first)")
        inputs = self._state.fit_inputs
        if inputs is None:
            raise RuntimeError("no fluxonium fit inputs (load params.json first)")
        fit = self._state.disp_fit
        if fit.bare_rf is None:
            raise RuntimeError("no bare_rf set (load fit inputs to seed it)")

        params = inputs.params
        bare_rf = fit.bare_rf
        g_bound = fit.g_bound
        g_init = fit.g if fit.g is not None else 0.5 * (g_bound[0] + g_bound[1])

        def _run() -> AutoFitResult:
            return auto_fit_dispersive(
                params,
                bare_rf,
                pp.sp_fluxs,
                pp.sp_freqs,
                pp.norm_phases,
                g_bound,
                g_init,
                fit.fit_bare_rf,
            )

        if pbar_factory is not None:
            with use_pbar_factory(pbar_factory):
                result = _run()
        else:
            result = _run()
        logger.debug("compute_autofit: g=%s bare_rf=%s", result.g, result.bare_rf)
        return result

    def record_result(self, result: AutoFitResult) -> None:
        """Write a computed auto-fit onto State (MAIN THREAD only).

        ``bare_rf`` None means it was not fit — the current tuning bare_rf is kept.
        """
        fit = self._state.disp_fit
        bare_rf = result.bare_rf if result.bare_rf is not None else fit.bare_rf
        assert bare_rf is not None  # compute_autofit fast-fails when bare_rf unset
        self._state.set_disp_result(result.g, bare_rf, auto=True)

    def autofit(self, *, pbar_factory: Optional[PbarFactory] = None) -> AutoFitResult:
        """Compute + record inline (RPC / convenience path, main thread)."""
        result = self.compute_autofit(pbar_factory=pbar_factory)
        self.record_result(result)
        return result

    def set_manual_fit(self, g: float, bare_rf: float) -> None:
        """Record the current slider g / bare_rf as the (manual) result."""
        self._state.set_disp_result(g, bare_rf, auto=False)

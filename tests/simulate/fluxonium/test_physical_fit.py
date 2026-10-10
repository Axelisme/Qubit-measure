"""Public local-fit acceptance and solver-limit rejection contracts."""

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
import zcu_tools.simulate.fluxonium.physical_fit as physical_fit
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares
from zcu_tools.simulate.fluxonium import (
    FluxoniumModelSnapshot,
    fit_local_fluxonium_model,
)


def test_fit_local_fluxonium_model_returns_rejected_result_on_solver_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = FluxoniumModelSnapshot(
        params=(5.0, 1.0, 0.5), flux_half=0.1, flux_period=0.8, flux_bias=0.03
    )
    original = (base.params, base.flux_half, base.flux_period, base.flux_bias)
    values = np.array([0.1, 0.25, 0.4, 0.55])
    measured = base.make_predictor().predict_freq(values) + 100.0
    solver_results: list[OptimizeResult] = []

    def limited_solver(
        fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        x0: NDArray[np.float64],
        *,
        bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
        method: Literal["trf"],
        max_nfev: int,
        x_scale: str,
    ) -> OptimizeResult:
        # Only the evaluation budget changes; retain the real solver outcome.
        result = least_squares(
            fun, x0, bounds=bounds, method=method, max_nfev=1, x_scale=x_scale
        )
        solver_results.append(result)
        return result

    monkeypatch.setattr(physical_fit, "least_squares", limited_solver)
    fit = fit_local_fluxonium_model(base, zip(values, measured, strict=True))

    (solver_result,) = solver_results
    assert solver_result.status == 0
    assert not solver_result.success
    assert np.all(np.isfinite(solver_result.x))
    assert not fit.accepted
    assert fit.reason == str(solver_result.message)
    assert fit.reason == "The maximum number of function evaluations is exceeded."
    assert fit.base is base
    assert (base.params, base.flux_half, base.flux_period, base.flux_bias) == original
    assert fit.fitted is None
    assert fit.predictor is None
    assert fit.n_points == 4
    assert math.isfinite(fit.base_rms_mhz)
    assert fit.base_rms_mhz == pytest.approx(100.0)
    assert math.isnan(fit.fitted_rms_mhz)


def test_fit_local_fluxonium_model_preserves_base_on_exact_data() -> None:
    base = FluxoniumModelSnapshot(
        params=(5.0, 1.0, 0.5), flux_half=0.1, flux_period=0.8, flux_bias=0.03
    )
    original = (base.params, base.flux_half, base.flux_period, base.flux_bias)
    values = np.array([0.1, 0.25, 0.4, 0.55])
    measured = base.make_predictor().predict_freq(values)

    fit = fit_local_fluxonium_model(base, zip(values, measured, strict=True))

    assert fit.accepted
    assert fit.reason == "accepted"
    assert fit.base is base
    assert (base.params, base.flux_half, base.flux_period, base.flux_bias) == original
    assert fit.fitted is not None
    assert fit.fitted.params == pytest.approx(base.params)
    assert fit.fitted.flux_half == base.flux_half
    assert fit.fitted.flux_period == base.flux_period
    assert fit.fitted.flux_bias == pytest.approx(base.flux_bias)
    assert fit.n_points == 4
    assert fit.base_rms_mhz == pytest.approx(0.0, abs=1e-6)
    assert fit.fitted_rms_mhz == pytest.approx(0.0, abs=1e-6)
    assert fit.predictor is not None
    np.testing.assert_allclose(
        fit.predictor.predict_freq(values), measured, rtol=1e-10, atol=1e-6
    )

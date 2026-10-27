"""Public reduced-flux local-fit and snapshot restoration contracts."""

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


def test_fit_local_fluxonium_model_preserves_base_on_exact_data() -> None:
    base = FluxoniumModelSnapshot((5.0, 1.0, 0.5), offset_correction=0.03)
    values = np.array([0.1, 0.25, 0.4, 0.55])
    measured = base.make_predictor().predict_freq(values + 0.03)
    fit = fit_local_fluxonium_model(base, zip(values, measured, strict=True))
    assert fit.accepted
    assert fit.reason == "accepted"
    assert fit.base is base
    assert base.params == (5.0, 1.0, 0.5)
    assert base.offset_correction == 0.03
    assert fit.fitted is not None
    assert fit.fitted.params == pytest.approx(base.params)
    assert fit.fitted.offset_correction == pytest.approx(0.03)
    assert fit.n_points == 4
    assert fit.base_rms_mhz == pytest.approx(0.0, abs=1e-6)
    assert fit.fitted_rms_mhz == pytest.approx(0.0, abs=1e-6)
    restored = fit.fitted.make_predictor()
    np.testing.assert_allclose(
        restored.predict_freq(values + fit.fitted.offset_correction),
        measured,
        rtol=1e-10,
        atol=1e-6,
    )


def test_fit_local_fluxonium_model_returns_rejected_result_on_solver_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = FluxoniumModelSnapshot((5.0, 1.0, 0.5))
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
        result = least_squares(
            fun, x0, bounds=bounds, method=method, max_nfev=1, x_scale=x_scale
        )
        solver_results.append(result)
        return result

    # SciPy's evaluation budget is outside the public fitting contract.
    monkeypatch.setattr(physical_fit, "least_squares", limited_solver)
    fit = fit_local_fluxonium_model(base, zip(values, measured, strict=True))
    (solver_result,) = solver_results
    assert solver_result.status == 0
    assert not fit.accepted
    assert fit.reason == "The maximum number of function evaluations is exceeded."
    assert fit.base is base
    assert fit.fitted is None
    assert fit.predictor is None
    assert fit.n_points == 4
    assert fit.base_rms_mhz == pytest.approx(100.0)
    assert math.isnan(fit.fitted_rms_mhz)

from __future__ import annotations

import numpy as np
import pytest
import zcu_tools.utils.fitting as fitting
import zcu_tools.utils.fitting.base as fitting_base
from numpy.typing import NDArray
from zcu_tools.utils.fitting.shared import (
    FitTrace,
    ParameterSpec,
    fit_shared,
)


def test_public_facades_expose_named_shared_fit_without_batch_compatibility() -> None:
    assert fitting.fit_shared is fit_shared
    assert not hasattr(fitting, "batch_fit_func")
    assert not hasattr(fitting_base, "batch_fit_func")


def linear(
    x: NDArray[np.float64], slope: float, intercept: float
) -> NDArray[np.float64]:
    return slope * x + intercept


def test_fit_shared_uses_named_global_identity_and_covariance() -> None:
    rng = np.random.default_rng(20260901)
    x = np.linspace(-2.0, 2.0, 81)
    errors = np.full_like(x, 0.02)
    first = linear(x, 1.7, -0.4) + rng.normal(0.0, errors)
    second = linear(x, 1.7, 0.8) + rng.normal(0.0, errors)

    result = fit_shared(
        traces=(
            FitTrace(x, first, linear, ("slope", "first_intercept"), errors),
            FitTrace(x, second, linear, ("slope", "second_intercept"), errors),
        ),
        parameters=(
            ParameterSpec("slope", 1.5, limits=(0.0, 3.0)),
            ParameterSpec("first_intercept", -0.4, fixed=True),
            ParameterSpec("second_intercept", 0.5),
        ),
        profile=("slope",),
    )

    assert result.parameter_names == (
        "slope",
        "first_intercept",
        "second_intercept",
    )
    assert result.diagnostics.valid
    assert result.diagnostics.covariance_accurate
    assert result.values["slope"] == pytest.approx(1.7, rel=5e-3)
    assert result.values["first_intercept"] == -0.4
    assert result.values["second_intercept"] == pytest.approx(0.8, rel=5e-3)
    assert result.covariance.shape == (3, 3)
    np.testing.assert_array_equal(result.covariance[1], 0.0)
    np.testing.assert_array_equal(result.covariance[:, 1], 0.0)
    assert result.variance("slope") > 0.0
    expected_projection = (
        result.covariance[0, 0]
        + 4.0 * result.covariance[2, 2]
        - 4.0 * result.covariance[0, 2]
    )
    assert result.projected_variance(
        {"slope": 1.0, "second_intercept": -2.0}
    ) == pytest.approx(expected_projection)
    low, high = result.profile_intervals["slope"]
    assert low < result.values["slope"] < high


def test_fit_shared_accepts_multi_output_trace() -> None:
    x = np.linspace(0.0, 1.0, 21)

    def paired(values: NDArray[np.float64], slope: float) -> NDArray[np.float64]:
        return np.column_stack((slope * values, -slope * values))

    result = fit_shared(
        (FitTrace(x, paired(x, 1.8), paired, ("slope",)),),
        (ParameterSpec("slope", 1.0),),
    )

    assert result.diagnostics.valid
    assert result.values["slope"] == pytest.approx(1.8)


def test_fit_shared_fast_fails_invalid_parameter_declarations() -> None:
    x = np.linspace(0.0, 1.0, 10)
    trace = FitTrace(x, linear(x, 1.0, 0.0), linear, ("slope", "intercept"))

    with pytest.raises(ValueError, match="exactly once"):
        fit_shared(
            (trace,),
            (
                ParameterSpec("slope", 1.0),
                ParameterSpec("slope", 1.1),
                ParameterSpec("intercept", 0.0),
            ),
        )
    with pytest.raises(ValueError, match="unknown parameters"):
        fit_shared((trace,), (ParameterSpec("slope", 1.0),))
    with pytest.raises(ValueError, match="empty limits"):
        fit_shared(
            (trace,),
            (
                ParameterSpec("slope", 1.0, limits=(2.0, 1.0)),
                ParameterSpec("intercept", 0.0),
            ),
        )
    with pytest.raises(ValueError, match="must be finite"):
        fit_shared(
            (trace,),
            (
                ParameterSpec("slope", np.nan),
                ParameterSpec("intercept", 0.0),
            ),
        )
    with pytest.raises(ValueError, match="finite or None"):
        fit_shared(
            (trace,),
            (
                ParameterSpec("slope", 1.0, limits=(np.nan, 2.0)),
                ParameterSpec("intercept", 0.0),
            ),
        )


def test_fit_shared_fast_fails_mismatched_sample_dimension() -> None:
    x = np.linspace(0.0, 1.0, 10)
    y = np.ones((9, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="leading dimension"):
        fit_shared(
            (FitTrace(x, y, linear, ("slope", "intercept")),),
            (ParameterSpec("slope", 1.0), ParameterSpec("intercept", 0.0)),
        )


def test_fit_shared_returns_backend_failure_diagnostics() -> None:
    x = np.linspace(-2.0, 2.0, 101)
    y = linear(x, 1.7, -0.4)

    result = fit_shared(
        (FitTrace(x, y, linear, ("slope", "intercept")),),
        (ParameterSpec("slope", -5.0), ParameterSpec("intercept", 4.0)),
        max_calls=1,
    )

    assert not result.diagnostics.valid
    assert result.diagnostics.edm > 1.0
    assert isinstance(result.diagnostics.reached_call_limit, bool)
    assert set(result.values) == {"slope", "intercept"}
    assert result.covariance.shape == (2, 2)


def test_variance_rejects_unknown_parameter() -> None:
    x = np.linspace(0.0, 1.0, 20)
    result = fit_shared(
        (FitTrace(x, linear(x, 1.0, 0.0), linear, ("slope", "intercept")),),
        (ParameterSpec("slope", 0.9), ParameterSpec("intercept", 0.1)),
    )

    with pytest.raises(KeyError, match="missing"):
        result.variance("missing")

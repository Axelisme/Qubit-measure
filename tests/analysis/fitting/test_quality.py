"""Model-independent fit-quality contract, without invoking an optimizer."""

import json

import numpy as np
import pytest
from zcu_tools.analysis.fitting import compute_fit_quality, decay_time_bounds


@pytest.mark.parametrize(
    ("observations", "fitted", "r2", "normalized_rms"),
    [
        (
            [1, 0.5, 0.25, 0.125],
            [1.05, 0.45, 0.3, 0.075],
            0.9777391304347826,
            0.05714285714285714,
        ),
        ([1, 0, -1, 0, 1], [0.9, 0.1, -0.9, -0.1, 0.9], 0.9821428571428571, 0.05),
    ],
    ids=["exponential", "cosine"],
)
def test_noisy_curves_report_residuals_and_named_relative_errors(
    observations: list[float], fitted: list[float], r2: float, normalized_rms: float
) -> None:
    quality = compute_fit_quality(
        np.array(observations, dtype=np.float64),
        np.array(fitted, dtype=np.float64),
        {"amplitude": 2.0, "decay": -4.0},
        np.diag([0.04, 1.0]),
    )
    assert quality.r2 == pytest.approx(r2)
    assert quality.normalized_residual_rms == pytest.approx(normalized_rms)
    assert quality.relative_parameter_errors == {"amplitude": 0.1, "decay": 0.25}
    assert quality.invalid == ()
    summary = quality.to_summary_dict()
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


@pytest.mark.parametrize(
    ("parameter", "variance", "reason"),
    [
        (0.0, 1.0, "zero_parameter"),
        (0.0, 0.0, "zero_parameter"),
        (0.0, np.nan, "non_finite"),
        (0.0, -1.0, "negative_variance"),
        (1.0, np.inf, "non_finite"),
        (1.0, np.nan, "non_finite"),
        (np.inf, 1.0, "non_finite"),
        (1.0, -1.0, "negative_variance"),
    ],
)
def test_unestimable_parameter_keeps_finite_residual_metrics(
    parameter: float, variance: float, reason: str
) -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        {"bad": parameter, "good": 2.0},
        np.diag([variance, 0.04]),
    )
    assert quality.r2 == 1.0
    assert quality.normalized_residual_rms == 0.0
    assert quality.relative_parameter_errors == {"bad": None, "good": 0.1}
    expected = [{"path": "relative_parameter_errors.bad", "reason": reason}]
    if variance == 0:
        expected.insert(
            0, {"path": "standard_errors.bad", "reason": "zero_standard_error"}
        )
    assert quality.to_summary_dict()["invalid"] == expected
    json.dumps(quality.to_summary_dict(), allow_nan=False)


def test_zero_standard_error_is_invalid_without_changing_numeric_metrics() -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        {"decay_time": 2.0},
        np.zeros((1, 1)),
    )
    assert quality.to_summary_dict()["invalid"] == [
        {"path": "standard_errors.decay_time", "reason": "zero_standard_error"}
    ]
    assert quality.r2 == 1.0
    assert quality.normalized_residual_rms == 0.0
    assert quality.relative_parameter_errors == {"decay_time": 0.0}


@pytest.mark.parametrize("variation", [0.0, 1e-12])
def test_near_constant_predictions_are_invalid_even_with_finite_errors(
    variation: float,
) -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0 - variation, 1.0, 1.0 + variation]),
        {"offset": 1.0},
        np.eye(1),
    )
    assert quality.to_summary_dict()["invalid"] == [
        {"path": "prediction", "reason": "near_constant_prediction"}
    ]
    assert quality.r2 == pytest.approx(0.0, abs=1e-10)
    assert quality.normalized_residual_rms == pytest.approx(0.408248290463863)
    assert quality.relative_parameter_errors == {"offset": 1.0}


@pytest.mark.parametrize(
    ("time", "bound"),
    [
        (0.01, "lower bound 0.1"),
        (0.1, "lower bound 0.1"),
        (0.100000001, "lower bound 0.1"),
        (19.99999999, "upper bound 20"),
        (20.0, "upper bound 20"),
        (100.0, "upper bound 20"),
    ],
)
@pytest.mark.parametrize("name", ["decay_time", "decay_time1", "decay_time2"])
def test_boundary_and_outside_decay_times_are_invalid(
    time: float, name: str, bound: str
) -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        {name: time},
        np.eye(1),
        decay_bounds=(0.1, 20.0),
    )
    assert quality.to_summary_dict()["invalid"] == [
        {
            "path": name,
            "reason": f"{name}={time:.9g} at/outside {bound}: 窗口內無法解析衰減時間",
        }
    ]
    assert quality.r2 == 1.0
    assert quality.relative_parameter_errors[name] == pytest.approx(1.0 / time)


@pytest.mark.parametrize(
    ("delays", "expected"),
    [
        ([0.0, 0.2, 1.0], (0.2, 10.0)),
        ([0.1, 1.0, 2.0], (0.1, 19.0)),
        ([5.0, 0.5, 1.0, 1.0], (0.5, 45.0)),
    ],
)
def test_decay_bounds_use_acquired_delay_resolution(
    delays: list[float], expected: tuple[float, float]
) -> None:
    assert decay_time_bounds(np.array(delays)) == pytest.approx(expected)


def test_interior_decay_and_other_parameter_names_are_not_invalid() -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        {"decay_time": 2.0, "frequency": 100.0},
        np.eye(2),
        decay_bounds=(0.1, 20.0),
    )
    assert quality.invalid == ()


def test_missing_covariance_without_parameters_still_has_a_reason() -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), {}, None
    )
    assert quality.to_summary_dict()["invalid"] == [
        {"path": "covariance", "reason": "covariance_unavailable"}
    ]


def test_fixed_parameter_mask_keeps_free_zero_stderr_invalid() -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        {"phase": 0.0, "decay_time": 2.0},
        np.zeros((2, 2)),
        fixed_parameters=("phase",),
    )
    assert quality.to_summary_dict()["invalid"] == [
        {"path": "standard_errors.decay_time", "reason": "zero_standard_error"}
    ]
    assert quality.relative_parameter_errors == {"phase": None, "decay_time": 0.0}


def test_fixed_parameter_mask_rejects_undeclared_names() -> None:
    with pytest.raises(ValueError, match="fixed_parameters.*declared"):
        compute_fit_quality(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            {"time": 2.0},
            np.eye(1),
            fixed_parameters=("phase",),
        )


def test_missing_covariance_does_not_hide_negative_r2() -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]),
        np.array([2.0, 1.0, 0.0]),
        {"offset": 1.0},
        None,
    )
    assert quality.r2 == -3.0
    assert quality.normalized_residual_rms == pytest.approx(0.816496580927726)
    assert quality.relative_parameter_errors == {"offset": None}
    assert quality.to_summary_dict()["invalid"] == [
        {"path": "relative_parameter_errors.offset", "reason": "covariance_unavailable"}
    ]


@pytest.mark.parametrize("parameter", [0.0, np.nan, np.inf])
def test_missing_covariance_takes_priority_over_parameter_value(
    parameter: float,
) -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]),
        np.array([2.0, 1.0, 0.0]),
        {"fit.offset": parameter, "good": 2.0},
        None,
    )
    assert quality.r2 == -3.0
    assert quality.normalized_residual_rms == pytest.approx(0.816496580927726)
    assert quality.relative_parameter_errors == {"fit.offset": None, "good": None}
    assert quality.to_summary_dict()["invalid"] == [
        {
            "path": "relative_parameter_errors.fit.offset",
            "reason": "covariance_unavailable",
        },
        {"path": "relative_parameter_errors.good", "reason": "covariance_unavailable"},
    ]
    json.dumps(quality.to_summary_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("observations", "fitted", "reasons"),
    [
        ([1.0, 1.0], [1.0, 1.0], ["zero_variance", "zero_range"]),
        ([1.0, np.nan], [1.0, 1.0], ["non_finite", "non_finite"]),
        ([1.0, 2.0], [1.0, np.inf], ["non_finite", "non_finite"]),
    ],
)
def test_unestimable_residual_metrics_keep_parameter_error(
    observations: list[float], fitted: list[float], reasons: list[str]
) -> None:
    quality = compute_fit_quality(
        np.array(observations), np.array(fitted), {"amplitude": 2.0}, np.diag([0.04])
    )
    assert quality.r2 is None
    assert quality.normalized_residual_rms is None
    assert quality.relative_parameter_errors == {"amplitude": 0.1}
    assert quality.to_summary_dict()["invalid"] == [
        {"path": "r2", "reason": reasons[0]},
        {"path": "normalized_residual_rms", "reason": reasons[1]},
    ]
    json.dumps(quality.to_summary_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[1.0, 3.0], [0.0, 4.0]]),
        np.array([[1.0, np.nan], [np.inf, 4.0]]),
    ],
    ids=["asymmetric", "nonfinite-off-diagonal"],
)
def test_marginal_errors_use_only_covariance_diagonal(
    covariance: np.ndarray,
) -> None:
    quality = compute_fit_quality(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        {"first": 2.0, "second": -4.0},
        covariance,
    )
    assert quality.relative_parameter_errors == {"first": 0.5, "second": 0.5}
    assert quality.invalid == ()
    assert quality.r2 == 1.0
    assert quality.normalized_residual_rms == 0.0
    json.dumps(quality.to_summary_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("observations", "fitted"),
    [(np.array([0, 16], dtype=np.uint8), np.array([0, 0], dtype=np.uint8))],
)
def test_integer_observations_are_rejected_before_residual_arithmetic(
    observations: np.ndarray, fitted: np.ndarray
) -> None:
    observations_before = observations.copy()
    fitted_before = fitted.copy()
    with pytest.raises(ValueError, match=r"^y\b.*float64.*uint8"):
        compute_fit_quality(observations, fitted, {}, None)
    np.testing.assert_array_equal(observations, observations_before)
    np.testing.assert_array_equal(fitted, fitted_before)


@pytest.mark.parametrize(
    ("observations", "fitted", "covariance", "name", "dtype"),
    [
        (
            np.array([0, 16], dtype=dtype if name == "y" else np.float64),
            np.array([0, 0], dtype=dtype if name == "y_fit" else np.float64),
            np.eye(1, dtype=dtype if name == "covariance" else np.float64),
            name,
            dtype,
        )
        for name in ("y", "y_fit", "covariance")
        for dtype in ("uint8", "bool", "complex128", "float32")
        if (name, dtype) != ("y", "uint8")  # Covered by the original wrap reproducer.
    ],
)
def test_unsupported_array_dtype_names_the_rejected_argument(
    observations: np.ndarray,
    fitted: np.ndarray,
    covariance: np.ndarray,
    name: str,
    dtype: str,
) -> None:
    observations_before = observations.copy()
    fitted_before = fitted.copy()
    covariance_before = covariance.copy()
    with pytest.raises(ValueError, match=rf"^{name}\b.*float64.*{dtype}"):
        compute_fit_quality(observations, fitted, {"offset": 1.0}, covariance)
    np.testing.assert_array_equal(observations, observations_before)
    np.testing.assert_array_equal(fitted, fitted_before)
    np.testing.assert_array_equal(covariance, covariance_before)


@pytest.mark.parametrize(
    ("observations", "fitted", "parameters", "covariance", "message"),
    [
        (np.array([]), np.array([]), {"p": 1.0}, np.eye(1), "nonempty"),
        (np.ones((1, 2)), np.ones(2), {"p": 1.0}, np.eye(1), "one-dimensional"),
        (np.ones(2), np.ones(3), {"p": 1.0}, np.eye(1), "matching"),
        (np.ones(2), np.ones(2), {"": 1.0}, np.eye(1), "nonempty"),
        (np.ones(2), np.ones(2), {"p": 1.0}, np.eye(2), "covariance shape"),
    ],
)
def test_caller_shape_and_name_mistakes_fail_fast(
    observations: np.ndarray,
    fitted: np.ndarray,
    parameters: dict[str, float],
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_fit_quality(observations, fitted, parameters, covariance)

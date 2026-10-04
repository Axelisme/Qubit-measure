"""Model-independent fit-quality contract, without invoking an optimizer."""

import json

import numpy as np
import pytest
from zcu_tools.analysis.fitting import compute_fit_quality


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
    assert quality.to_summary_dict()["invalid"] == [
        {"path": "relative_parameter_errors.bad", "reason": reason}
    ]
    json.dumps(quality.to_summary_dict(), allow_nan=False)


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
    ("observations", "fitted", "parameters", "covariance", "message"),
    [
        (np.array([]), np.array([]), {"p": 1.0}, np.eye(1), "nonempty"),
        (np.ones((1, 2)), np.ones(2), {"p": 1.0}, np.eye(1), "one-dimensional"),
        (np.ones(2), np.ones(3), {"p": 1.0}, np.eye(1), "matching"),
        (np.ones(2, dtype=complex), np.ones(2), {"p": 1.0}, np.eye(1), "real"),
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

import numpy as np
from zcu_tools.analysis.fitting.anticross import (
    fit_anticross,
    fit_hyperbolic,
    get_predict_ys,
)
from zcu_tools.analysis.fitting.base import retrieve_params


def test_fit_anticross_recovers_center_and_width():
    cx_true, cy_true, width_true = 0.3, 1.5, 0.4
    m1_true, m2_true = 1.0, -1.0
    params = retrieve_params(cx_true, cy_true, width_true, m1_true, m2_true)

    xs = np.linspace(-2, 2, 40) + cx_true
    ys1, ys2 = get_predict_ys(xs, *params)
    mask = np.isfinite(ys1) & np.isfinite(ys2)
    xs, ys1, ys2 = xs[mask], ys1[mask], ys2[mask]

    cx, cy, width, m1, m2, _, _, _ = fit_anticross(xs, ys1, ys2)
    assert abs(cx - cx_true) < 1e-2
    assert abs(cy - cy_true) < 1e-2
    assert abs(width - width_true) / width_true < 5e-2
    assert min(abs(m1 - m1_true), abs(m1 - m2_true)) < 1e-2
    assert min(abs(m2 - m1_true), abs(m2 - m2_true)) < 1e-2


def test_horizontal_hyperbola_fit_preserves_both_branches() -> None:
    coefficients = (0.0, -1.0, 1.0, 1.5, -2.7, 1.76)
    # This conic has real branches on these two intervals, not around x=0.
    xs = np.concatenate((np.linspace(-2.0, -1.0, 20), np.linspace(1.5, 2.0, 20)))
    upper, lower = get_predict_ys(xs, *coefficients)
    assert np.all(np.isfinite(upper))
    assert np.all(np.isfinite(lower))
    assert np.all(upper > lower)

    params = fit_hyperbolic(xs, upper, lower, horizontal_line=True)

    assert isinstance(params, tuple)
    assert len(params) == 6
    assert params[0] == 0.0
    assert np.all(np.isfinite(params))
    np.testing.assert_allclose(np.linalg.norm(params), 1.0, atol=1e-12)
    predicted_upper, predicted_lower = get_predict_ys(xs, *params)
    np.testing.assert_allclose(predicted_upper, upper, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(predicted_lower, lower, rtol=1e-7, atol=1e-7)

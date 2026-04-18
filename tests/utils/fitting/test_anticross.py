import numpy as np

from zcu_tools.utils.fitting.anticross import fit_anticross, get_predict_ys
from zcu_tools.utils.fitting.base import retrieve_params


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

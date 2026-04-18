import numpy as np

from zcu_tools.utils.fitting.singleshot import (
    calc_population_pdf,
    fit_singleshot_p0,
)


def _make_ge_params():
    sg, se, s = -1.0, 1.0, 0.3
    p_avg = 0.5
    length_ratio = 0.1
    return sg, se, s, p_avg, length_ratio


def test_fit_singleshot_p0_recovers_populations():
    sg, se, s, p_avg, length_ratio = _make_ge_params()
    xs = np.linspace(-3, 3, 401)

    true_p0_g, true_p0_e = 0.2, 0.8

    ge_params = (sg, se, s, 1.0, 0.0, p_avg, length_ratio)
    pdf = calc_population_pdf(xs, sg, se, s, true_p0_g, true_p0_e, p_avg, length_ratio)

    (p0_g, p0_e, _), _ = fit_singleshot_p0(
        xs, pdf, init_p0_g=0.5, init_p0_e=0.5, ge_params=ge_params
    )

    ratio_true = true_p0_g / (true_p0_g + true_p0_e)
    ratio_fit = p0_g / (p0_g + p0_e)
    assert abs(ratio_fit - ratio_true) < 0.05

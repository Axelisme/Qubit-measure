import numpy as np
import pytest
from zcu_tools.utils.fitting.singleshot import (
    calc_population_pdf,
    fit_singleshot,
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


def _make_overlapping_histogram(
    xs: np.ndarray,
    sg: float,
    se: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-only g/e histograms (no state-change physics) for simple bounds tests."""
    from scipy.stats import norm

    g_pdf = norm.pdf(xs, loc=sg, scale=sigma)
    e_pdf = norm.pdf(xs, loc=se, scale=sigma)
    g_pdf /= g_pdf.sum()
    e_pdf /= e_pdf.sum()
    return g_pdf, e_pdf


def test_fit_singleshot_no_raise_when_s_exceeds_xs_range():
    # Reproduce the ValueError: "Initial guess is outside of provided bounds" that occurs
    # when the data-derived sigma s = 0.5*(sigma_g + sigma_e) exceeds xs[-1]-xs[0].
    # The narrow xs range (±0.5) forces that violation; the clip fix must prevent it.
    xs = np.linspace(-0.5, 0.5, 51)  # range = 1.0
    # Peaks well-separated relative to the window, sigma intentionally large → s > range
    sg, se, sigma = -0.3, 0.3, 0.8
    g_pdf, e_pdf = _make_overlapping_histogram(xs, sg, se, sigma)

    # Provide explicit fitparams that place s beyond xs[-1]-xs[0] to force the violation.
    # s_init = 1.2 > 1.0 = xs[-1]-xs[0]; before the fix, scipy raises ValueError.
    s_init = 1.2
    fitparams = [float(sg), float(se), s_init, 0.5, 0.5, 0.1, 0.01]

    # Should not raise; returned params must be finite (fit may not converge perfectly).
    pOpt, pCov = fit_singleshot(xs, g_pdf, e_pdf, fitparams=fitparams)
    assert all(np.isfinite(p) for p in pOpt), f"Non-finite params: {pOpt}"


def test_fit_singleshot_no_raise_when_sg_outside_se_bound():
    # Another violation: sg passed explicitly outside the bound derived from se.
    # bounds[0][0] = se (when se < sg) = -0.5; but we pass sg_init = -2.0 < -0.5.
    xs = np.linspace(-1.0, 1.0, 101)
    sg, se = 0.4, -0.4  # sg > se → lower_sg = se = -0.4
    g_pdf, e_pdf = _make_overlapping_histogram(xs, sg, se, sigma=0.15)

    # sg_init = -2.0 < lower bound -0.4 — triggers the pre-fix crash
    fitparams = [-2.0, float(se), 0.15, 0.5, 0.5, 0.1, 0.01]

    pOpt, pCov = fit_singleshot(xs, g_pdf, e_pdf, fitparams=fitparams)
    assert all(np.isfinite(p) for p in pOpt), f"Non-finite params: {pOpt}"

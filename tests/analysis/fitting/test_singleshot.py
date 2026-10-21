import numpy as np
import pytest
from zcu_tools.analysis.fitting import singleshot
from zcu_tools.analysis.fitting.singleshot import (
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


@pytest.mark.parametrize("swap_labels", [False, True])
def test_ge_boundary_peak_recovers_mixed_populations_without_an_outer_tail(swap_labels):
    xs = np.linspace(-1.0, 3.0, 81)
    truth = (-1.0, 1.0, 0.3, 0.65, 0.35, 0.2, 0.3)
    ground = calc_population_pdf(xs, *truth)
    excited = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    if swap_labels:
        ground, excited = excited, ground
    try:
        fitted, covariance = fit_singleshot(xs, ground, excited)
    except ValueError as error:
        pytest.fail(
            f"A finite boundary-peak histogram must fit without NaN width: {error}"
        )
    expected = truth if not swap_labels else (1.0, -1.0, 0.3, 0.65, 0.35, 0.8, 0.3)
    np.testing.assert_allclose(fitted, expected, atol=0.03)
    assert np.isfinite(covariance).all()


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_ge_rejects_nonfinite_supplied_initial_parameters(invalid):
    xs = np.linspace(-3.0, 3.0, 61)
    truth = (-1.0, 1.0, 0.3, 0.65, 0.35, 0.2, 0.3)
    ground = calc_population_pdf(xs, *truth)
    excited = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    with pytest.raises(ValueError, match="initial.*finite"):
        fit_singleshot(
            xs, ground, excited, fitparams=[-1.0, 1.0, invalid, 0.65, 0.35, 0.2, 0.3]
        )


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
    pOpt, covariance = fit_singleshot(xs, g_pdf, e_pdf, fitparams=fitparams)
    assert all(np.isfinite(p) for p in pOpt), f"Non-finite params: {pOpt}"
    assert np.isfinite(covariance).all()


def test_fit_singleshot_no_raise_when_sg_outside_se_bound():
    # Another violation: sg passed explicitly outside the bound derived from se.
    # bounds[0][0] = se (when se < sg) = -0.5; but we pass sg_init = -2.0 < -0.5.
    xs = np.linspace(-1.0, 1.0, 101)
    sg, se = 0.4, -0.4  # sg > se → lower_sg = se = -0.4
    g_pdf, e_pdf = _make_overlapping_histogram(xs, sg, se, sigma=0.15)

    # sg_init = -2.0 < lower bound -0.4 — triggers the pre-fix crash
    fitparams = [-2.0, float(se), 0.15, 0.5, 0.5, 0.1, 0.01]

    pOpt, covariance = fit_singleshot(xs, g_pdf, e_pdf, fitparams=fitparams)
    assert all(np.isfinite(p) for p in pOpt), f"Non-finite params: {pOpt}"
    assert np.isfinite(covariance).all()


@pytest.mark.parametrize("p_avg", [0.15, 0.85])
def test_ge_strong_transition_fit_is_label_symmetric(p_avg):
    xs = np.linspace(-4, 4, 401)
    truth = (-1.0, 1.0, 0.3, 0.9, 0.1, p_avg, 1.2)
    g = calc_population_pdf(xs, *truth)
    e = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    fitted, covariance = fit_singleshot(xs, g, e)
    swapped, swapped_covariance = fit_singleshot(xs, e, g)
    np.testing.assert_allclose(fitted, truth, atol=0.03)
    expected = np.array(fitted)[[1, 0, 2, 3, 4, 5, 6]]
    expected[5] = 1 - expected[5]
    np.testing.assert_allclose(swapped, expected, atol=1e-10)
    transform = np.eye(7)[[1, 0, 2, 3, 4, 5, 6]]
    transform[5, 5] = -1
    np.testing.assert_allclose(swapped_covariance, transform @ covariance @ transform.T)


@pytest.mark.parametrize(
    "fixed_populations", [(None, None), (0.25, None), (None, 0.65), (0.25, 0.65)]
)
def test_joint_ge_respects_fixed_populations_and_simplex(fixed_populations):
    xs = np.linspace(-4, 4, 301)
    truth = (-1.0, 1.0, 0.3, 0.25, 0.65, 0.8, 0.7)
    g = calc_population_pdf(xs, *truth)
    e = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    fixed = [-1.0, 1.0, 0.3, *fixed_populations, 0.8, 0.7]
    fitted, covariance = fit_singleshot(xs, g, e, fixedparams=fixed)
    np.testing.assert_allclose(fitted[3:5], truth[3:5], atol=1e-5)
    assert sum(fitted[3:5]) <= 1
    for index, value in enumerate(fixed):
        if value is not None:
            np.testing.assert_array_equal(covariance[index], 0.0)
            np.testing.assert_array_equal(covariance[:, index], 0.0)


@pytest.mark.parametrize("occupied_population", [0, 1])
def test_ge_fixed_population_can_use_all_probability_mass(occupied_population):
    xs = np.linspace(-4, 4, 301)
    populations = [0.0, 0.0]
    populations[occupied_population] = 1.0
    truth = (-1.0, 1.0, 0.3, populations[0], populations[1], 0.8, 0.7)
    g = calc_population_pdf(xs, *truth)
    e = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    fixed = [-1.0, 1.0, 0.3, None, None, 0.8, 0.7]
    fixed[3 + occupied_population] = 1.0

    fitted, covariance = fit_singleshot(xs, g, e, fixedparams=fixed)

    np.testing.assert_allclose(fitted, truth, atol=1e-12)
    np.testing.assert_array_equal(covariance, np.zeros((7, 7)))


@pytest.mark.parametrize("fit_length_ratio", [False, True])
@pytest.mark.parametrize("length_ratio", [0.0, 0.5])
def test_row_population_is_legal_and_shared_ratio_is_honored(
    fit_length_ratio, length_ratio
):
    xs = np.linspace(-4, 4, 401)
    params = (-1.0, 1.0, 0.3, 0.9, 0.1, 0.8, length_ratio)
    # An over-normalized target used to produce a negative other population.
    pdf = 1.2 * calc_population_pdf(xs, *params)
    fitted, covariance = fit_singleshot_p0(
        xs, pdf, 0.8, 0.2, params, fit_length_ratio=fit_length_ratio
    )
    assert min(fitted[:2]) >= 0
    assert sum(fitted[:2]) <= 1
    if not fit_length_ratio or length_ratio == 0:
        assert fitted[2] == length_ratio
        assert np.all(covariance[2] == 0)


def test_ge_does_not_return_initial_guess_on_optimizer_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("optimizer exhausted")

    # Valid finite histograms cannot reliably exhaust SciPy's unexposed iteration budget.
    monkeypatch.setattr(singleshot, "curve_fit", fail)
    xs = np.linspace(-4, 4, 101)
    g, e = _make_overlapping_histogram(xs, -1, 1, 0.3)
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_singleshot(xs, g, e)


@pytest.mark.parametrize("p_avg", [0.15, 0.85])
@pytest.mark.parametrize("initial_transition", [(0.05, 0.2), (0.95, 2.8)])
def test_ge_recovers_from_misleading_initial_transition(p_avg, initial_transition):
    xs = np.linspace(-4, 4, 201)
    truth = (-1.0, 1.0, 0.3, 0.6, 0.2, p_avg, 1.2)
    g = calc_population_pdf(xs, *truth)
    e = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    initial = [-1.0, 1.0, 0.3, 0.6, 0.2, *initial_transition]

    fitted, covariance = fit_singleshot(xs, g, e, fitparams=initial)

    np.testing.assert_allclose(fitted, truth, atol=0.03)
    predicted_g = calc_population_pdf(xs, *fitted)
    predicted_e = calc_population_pdf(
        xs, *fitted[:3], fitted[4], fitted[3], *fitted[5:]
    )
    residual = np.concatenate([predicted_g - g, predicted_e - e])
    assert np.linalg.norm(residual) < 0.001
    assert np.isfinite(covariance).all()


@pytest.mark.parametrize("failure", ["exhausted", "nan-covariance", "inf-covariance"])
def test_ge_recovers_after_unreliable_optimizer_candidate(monkeypatch, failure):
    from scipy.optimize import curve_fit

    xs = np.linspace(-4, 4, 101)
    truth = (-1.0, 1.0, 0.3, 0.6, 0.2, 0.7, 0.7)
    g = calc_population_pdf(xs, *truth)
    e = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    pending_failure = True

    def optimizer(*args, **options):
        nonlocal pending_failure
        fitted, covariance = curve_fit(*args, **options)
        if pending_failure:
            pending_failure = False
            if failure == "exhausted":
                raise RuntimeError("optimizer exhausted")
            # Keep the real fit's near-zero residual, but make its uncertainty invalid.
            covariance[0, 0] = np.nan if failure == "nan-covariance" else np.inf
        return fitted, covariance

    # Finite histograms cannot reliably yield both valid and invalid SciPy candidates.
    monkeypatch.setattr(singleshot, "curve_fit", optimizer)
    fitted, covariance = fit_singleshot(xs, g, e, fitparams=truth)

    np.testing.assert_allclose(fitted, truth, atol=1e-5)
    assert np.isfinite(covariance).all()


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_ge_rejects_nonfinite_optimizer_parameters(monkeypatch, nonfinite):
    def optimizer(*args, **options):
        fitted = options["p0"].copy()
        fitted[0] = nonfinite
        return fitted, np.eye(len(fitted))

    # SciPy keeps bounded parameters finite for valid finite histogram inputs.
    monkeypatch.setattr(singleshot, "curve_fit", optimizer)
    xs = np.linspace(-4, 4, 101)
    g, e = _make_overlapping_histogram(xs, -1, 1, 0.3)
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_singleshot(xs, g, e)


def test_ge_rejects_unestimable_covariance():
    from scipy.optimize import OptimizeWarning

    # Six observations cannot estimate the covariance of seven free parameters.
    xs = np.array([-1.0, 0.0, 1.0])
    truth = (-0.3, 0.3, 1.2, 0.6, 0.2, 0.7, 0.7)
    g = calc_population_pdf(xs, *truth)
    e = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    with (
        pytest.warns(OptimizeWarning, match="Covariance"),
        pytest.raises(RuntimeError, match="did not converge"),
    ):
        fit_singleshot(xs, g, e, fitparams=truth)


@pytest.mark.parametrize("target_ratio, expected_ratio", [(0.05, 0.35), (2.8, 1.4)])
def test_row_fit_limits_adjustable_ratio(target_ratio, expected_ratio):
    xs = np.linspace(-4, 4, 201)
    params = (-1.0, 1.0, 0.3, 0.6, 0.2, 0.7, 0.7)
    pdf = calc_population_pdf(xs, *params[:6], target_ratio)

    fitted, covariance = fit_singleshot_p0(
        xs, pdf, 0.6, 0.2, params, fit_length_ratio=True
    )

    assert 0.35 <= fitted[2] <= 1.4
    assert fitted[2] == pytest.approx(expected_ratio, abs=0.001)
    assert min(fitted[:2]) >= 0 and sum(fitted[:2]) <= 1
    assert np.isfinite(covariance).all()


@pytest.mark.parametrize(
    "xs, ground, excited",
    [
        ([0, 1], [1, 1], [1, 1]),
        ([[0, 1, 2]], [[1, 1, 1]], [[1, 1, 1]]),
        ([0, 1, 2], [1, 1], [1, 1, 1]),
        ([0, 1, 2], [1, 1, 1], [1, 1]),
        ([0, 1, 1], [1, 1, 1], [1, 1, 1]),
        ([2, 1, 0], [1, 1, 1], [1, 1, 1]),
        ([0, np.nan, 2], [1, 1, 1], [1, 1, 1]),
        ([0, 1, 2], [1, np.inf, 1], [1, 1, 1]),
        ([0, 1, 2], [1, 1, 1], [1, np.nan, 1]),
        ([0, 1, 2], [-1, 2, 2], [1, 1, 1]),
        ([0, 1, 2], [1, 1, 1], [-1, 2, 2]),
        ([0, 1, 2], [0, 0, 0], [1, 1, 1]),
        ([0, 1, 2], [1, 1, 1], [0, 0, 0]),
    ],
    ids=[
        "too-short",
        "not-vector",
        "ground-shape",
        "excited-shape",
        "duplicate-axis",
        "descending-axis",
        "nonfinite-axis",
        "nonfinite-ground",
        "nonfinite-excited",
        "negative-ground",
        "negative-excited",
        "empty-ground",
        "empty-excited",
    ],
)
def test_ge_rejects_invalid_histograms(xs, ground, excited):
    with pytest.raises(
        ValueError, match="finite nonnegative histograms on an increasing axis"
    ):
        fit_singleshot(np.array(xs), np.array(ground), np.array(excited))


@pytest.mark.parametrize(
    "index, value", [(5, np.nan), (6, np.inf), (5, 1.1), (3, -0.1)]
)
def test_ge_rejects_invalid_fixed_values(index, value):
    xs = np.linspace(-4, 4, 101)
    g, e = _make_overlapping_histogram(xs, -1, 1, 0.3)
    fixed: list[float | None] = [None] * 7
    fixed[index] = value
    with pytest.raises(
        ValueError, match="Fixed GE parameters must be finite and within bounds"
    ):
        fit_singleshot(xs, g, e, fixedparams=fixed)


def test_ge_rejects_zero_modeled_population():
    xs = np.linspace(-4, 4, 101)
    g, e = _make_overlapping_histogram(xs, -1, 1, 0.3)
    with pytest.raises(ValueError, match="requires positive modeled population"):
        fit_singleshot(xs, g, e, fixedparams=[None, None, None, 0, 0, None, None])


def test_ge_rejects_impossible_fixed_populations():
    xs = np.linspace(-4, 4, 101)
    g, e = _make_overlapping_histogram(xs, -1, 1, 0.3)
    with pytest.raises(ValueError, match="sum to at most one"):
        fit_singleshot(xs, g, e, fixedparams=[None, None, None, 0.8, 0.8, None, None])

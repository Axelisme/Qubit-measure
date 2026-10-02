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
    fitted, covariance = fit_singleshot_p0(xs, pdf, 0.8, 0.2, params, fit_length_ratio)
    assert min(fitted[:2]) >= 0
    assert sum(fitted[:2]) <= 1
    if not fit_length_ratio or length_ratio == 0:
        assert fitted[2] == length_ratio
        assert np.all(covariance[2] == 0)


def test_ge_does_not_return_initial_guess_on_optimizer_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("optimizer exhausted")

    monkeypatch.setattr(singleshot, "curve_fit", fail)
    xs = np.linspace(-4, 4, 101)
    g, e = _make_overlapping_histogram(xs, -1, 1, 0.3)
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_singleshot(xs, g, e)


@pytest.mark.parametrize("reverse_labels", [False, True])
@pytest.mark.parametrize(
    "fixed_populations, encoded, optimizer_covariance, physical_covariance",
    [
        (
            (None, None),
            [0.8, 0.25, 0.7],
            [[0.04, 0.006, 0.002], [0.006, 0.09, 0.003], [0.002, 0.003, 0.01]],
            [
                [0.0729, -0.0477, -0.0009],
                [-0.0477, 0.0625, 0.0029],
                [-0.0009, 0.0029, 0.01],
            ],
        ),
        (
            (0.6, None),
            [0.5, 0.7],
            [[0.09, 0.003], [0.003, 0.01]],
            [[0.0, 0.0, 0.0], [0.0, 0.0144, 0.0012], [0.0, 0.0012, 0.01]],
        ),
        (
            (None, 0.2),
            [0.75, 0.7],
            [[0.09, 0.003], [0.003, 0.01]],
            [[0.0576, 0.0, 0.0024], [0.0, 0.0, 0.0], [0.0024, 0.0, 0.01]],
        ),
    ],
    ids=["both-free", "ground-fixed", "excited-fixed"],
)
def test_ge_returns_physical_covariance(
    monkeypatch,
    reverse_labels,
    fixed_populations,
    encoded,
    optimizer_covariance,
    physical_covariance,
):
    def optimizer(*args, **kwargs):
        return np.array(encoded), np.array(optimizer_covariance)

    monkeypatch.setattr(singleshot, "curve_fit", optimizer)
    xs = np.linspace(-4, 4, 101)
    truth = (-1.0, 1.0, 0.3, 0.6, 0.2, 0.7, 0.7)
    g = calc_population_pdf(xs, *truth)
    e = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    fixed = [-1.0, 1.0, 0.3, *fixed_populations, None, 0.7]
    expected_values = np.array(truth)
    # Independently calculated physical (p_g, p_e, p_avg) covariance, including
    # correlations with p_avg so reversed labels must also reverse their signs.
    expected_covariance = np.zeros((7, 7))
    expected_covariance[3:6, 3:6] = physical_covariance
    if reverse_labels:
        g, e = e, g
        fixed[0], fixed[1] = fixed[1], fixed[0]
        expected_values[0], expected_values[1] = 1.0, -1.0
        expected_values[5] = 0.3
        expected_covariance[5, :5] *= -1
        expected_covariance[:5, 5] *= -1

    fitted, covariance = fit_singleshot(xs, g, e, fixedparams=fixed)

    np.testing.assert_allclose(fitted, expected_values, atol=1e-12)
    np.testing.assert_allclose(covariance, expected_covariance, atol=1e-12)


@pytest.mark.parametrize("winner", [2, 3], ids=["earlier-valid", "later-valid"])
def test_ge_tries_all_starts_and_selects_lowest_residual(monkeypatch, winner):
    xs = np.linspace(-4, 4, 101)
    truth = (-1.0, 1.0, 0.3, 0.6, 0.2, 0.7, 0.7)
    g = calc_population_pdf(xs, *truth)
    e = calc_population_pdf(xs, *truth[:3], truth[4], truth[3], *truth[5:])
    starts = []
    encoded_truth = np.array([-1.0, 1.0, 0.3, 0.8, 0.25, 0.7, 0.7])

    def optimizer(model, coordinates, data, **options):
        index = len(starts)
        starts.append(options["p0"].copy())
        np.testing.assert_array_equal(coordinates, np.concatenate([xs, xs]))
        np.testing.assert_array_equal(data, np.concatenate([g, e]))
        np.testing.assert_allclose(
            options["bounds"],
            [[-4, -1, xs[1] - xs[0], 0, 0, 0, 0], [1, 4, 8, 1, 1, 1, 3]],
        )
        assert options["max_nfev"] == 5000
        assert options["sigma"] is None
        if index == 0:
            raise RuntimeError("first start exhausted")
        fitted = encoded_truth.copy()
        covariance = np.eye(7) * 0.01
        if index == 1:
            # This candidate has zero residual but cannot be accepted.
            covariance[0, 0] = np.inf
        elif index != winner:
            fitted[4] = 0.4
        return fitted, covariance

    monkeypatch.setattr(singleshot, "curve_fit", optimizer)
    fitted, covariance = fit_singleshot(xs, g, e, fitparams=truth)

    np.testing.assert_allclose(fitted, truth, atol=1e-12)
    assert np.isfinite(covariance).all()
    expected_starts = np.tile(encoded_truth, (4, 1))
    expected_starts[:, 5:] = [[0.7, 0.7], [0.2, 0.5], [0.8, 0.5], [0.5, 1.5]]
    np.testing.assert_allclose(starts, expected_starts, atol=1e-12)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
@pytest.mark.parametrize("component", ["parameters", "covariance"])
def test_ge_rejects_nonfinite_optimizer_results(monkeypatch, component, nonfinite):
    def optimizer(*args, **options):
        fitted = options["p0"].copy()
        covariance = np.eye(len(fitted))
        if component == "parameters":
            fitted[0] = nonfinite
        else:
            covariance[0, 0] = nonfinite
        return fitted, covariance

    monkeypatch.setattr(singleshot, "curve_fit", optimizer)
    xs = np.linspace(-4, 4, 101)
    g, e = _make_overlapping_histogram(xs, -1, 1, 0.3)
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_singleshot(xs, g, e)


@pytest.mark.parametrize("fit_length_ratio", [False, True])
def test_row_fit_uses_histogram_weights_and_ratio_bounds(monkeypatch, fit_length_ratio):
    xs = np.linspace(-4, 4, 101)
    params = (-1.0, 1.0, 0.3, 0.6, 0.2, 0.7, 0.7)
    pdf = calc_population_pdf(xs, *params)
    calls = []

    def optimizer(model, coordinates, data, **options):
        calls.append(options)
        np.testing.assert_array_equal(coordinates, xs)
        np.testing.assert_array_equal(data, pdf)
        return options["p0"], np.eye(len(options["p0"]))

    monkeypatch.setattr(singleshot, "curve_fit", optimizer)
    fitted, _ = fit_singleshot_p0(xs, pdf, 0.6, 0.2, params, fit_length_ratio)

    assert len(calls) == 1
    options = calls[0]
    ground = np.exp(-0.5 * ((xs + 1) / 0.3) ** 2)
    excited = np.exp(-0.5 * ((xs - 1) / 0.3) ** 2)
    weights = 0.6 * ground / ground.sum() + 0.2 * excited / excited.sum()
    np.testing.assert_allclose(options["sigma"], 1 / np.sqrt(weights))
    assert options["max_nfev"] == 5000
    if fit_length_ratio:
        np.testing.assert_allclose(options["p0"], [0.8, 0.25, 0.7])
        np.testing.assert_allclose(options["bounds"], [[0, 0, 0.35], [1, 1, 1.4]])
    else:
        np.testing.assert_allclose(options["p0"], [0.8, 0.25])
        np.testing.assert_allclose(options["bounds"], [[0, 0], [1, 1]])
    np.testing.assert_allclose(fitted, [0.6, 0.2, 0.7])


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

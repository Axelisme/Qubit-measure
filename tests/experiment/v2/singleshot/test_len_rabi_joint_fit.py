from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import zcu_tools.experiment.v2.singleshot.len_rabi_fit as fit_module
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from scipy.special import ndtr
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiExp, LenRabiResult
from zcu_tools.experiment.v2.singleshot.len_rabi_fit import (
    LenRabiPhysicalParams,
    fit_len_rabi_joint,
    model_bin_probabilities,
    multinomial_nll,
    rabi_excited_population,
)
from zcu_tools.utils.fitting.singleshot import (
    calc_fc,
    transition_state_bin_probabilities,
    transition_state_circle_probabilities,
)


def _sample_conditional(
    rng: np.random.Generator,
    *,
    excited: NDArray[np.bool_],
    size: int,
    params: LenRabiPhysicalParams,
) -> NDArray[np.float64]:
    rg = params.p_avg * params.length_ratio
    re = (1.0 - params.p_avg) * params.length_ratio
    grid = np.linspace(0.0005, 0.9995, 2000)
    density_g = calc_fc(grid, rg, re)
    density_e = calc_fc(grid, re, rg)
    density_g /= density_g.sum()
    density_e /= density_e.sum()

    positions = np.empty(size, dtype=np.float64)
    g_indices = np.flatnonzero(~excited)
    e_indices = np.flatnonzero(excited)
    if g_indices.size:
        transitions = rng.random(g_indices.size) >= np.exp(-rg)
        positions[g_indices] = 0.0
        positions[g_indices[transitions]] = rng.choice(
            grid,
            size=int(transitions.sum()),
            p=density_g,
        )
    if e_indices.size:
        transitions = rng.random(e_indices.size) >= np.exp(-re)
        positions[e_indices] = 1.0
        positions[e_indices[transitions]] = 1.0 - rng.choice(
            grid,
            size=int(transitions.sum()),
            p=density_e,
        )
    return params.center_g + (params.center_e - params.center_g) * positions


def _synthetic_raw(
    lengths: NDArray[np.float64],
    params: LenRabiPhysicalParams,
    *,
    shots: int,
    seed: int,
) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    p_e = rabi_excited_population(lengths, params)
    projected = np.empty((lengths.size, shots), dtype=np.float64)
    perpendicular = rng.normal(0.0, params.sigma, size=projected.shape)
    for index, probability in enumerate(p_e):
        excited = rng.random(shots) < probability
        projected[index] = _sample_conditional(
            rng,
            excited=excited,
            size=shots,
            params=params,
        ) + rng.normal(0.0, params.sigma, size=shots)
    origin = 0.35 - 0.22j
    axis = np.exp(0.61j)
    return np.asarray(
        origin + (projected + 1j * perpendicular) * axis, dtype=np.complex128
    )


def test_pca_orientation_uses_initial_dominant_cluster_when_means_tie() -> None:
    initial = np.array([1.0] * 6 + [-1.5] * 4, dtype=np.complex128)
    second = np.array([1.0] * 5 + [-1.0] * 5, dtype=np.complex128)
    projection = fit_module.project_len_rabi_iq(np.stack((initial, second)))

    # The upper cluster dominates the initial row before orientation, while both
    # the initial-row and pooled means are exactly zero.
    assert np.count_nonzero(projection.projected[0] < 0.0) == 6
    assert np.count_nonzero(projection.projected[0] > 0.0) == 4


def test_integrated_bins_and_multinomial_nll_match_direct_calculation() -> None:
    edges = np.array([-3.0, -0.5, 0.25, 3.0])
    qg, qe = transition_state_bin_probabilities(edges, -1.0, 1.0, 0.2, 0.25, 0.4)
    np.testing.assert_allclose(qg.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(qe.sum(), 1.0, atol=1e-14)
    assert np.all(qg > 0.0)
    assert np.all(qe > 0.0)

    gaussian_g, gaussian_e = transition_state_bin_probabilities(
        edges, -1.0, 1.0, 0.2, 0.25, 0.0
    )
    direct_g = np.diff(ndtr((edges + 1.0) / 0.2))
    direct_e = np.diff(ndtr((edges - 1.0) / 0.2))
    np.testing.assert_allclose(gaussian_g, direct_g / direct_g.sum(), atol=1e-14)
    np.testing.assert_allclose(gaussian_e, direct_e / direct_e.sum(), atol=1e-14)

    lengths = np.array([0.0, 0.5])
    params = LenRabiPhysicalParams(0.1, 0.5, -1.0, 1.0, 0.2, 0.25, 0.4, 2.0, np.pi)
    probabilities = model_bin_probabilities(lengths, edges, params)
    p_e = rabi_excited_population(lengths, params)
    expected = (1.0 - p_e[:, None]) * qg + p_e[:, None] * qe
    np.testing.assert_allclose(probabilities, expected, atol=1e-14)

    counts = np.array([[7, 2, 1], [1, 3, 6]], dtype=np.int64)
    expected_nll = -float(np.sum(counts * np.log(expected)))
    assert multinomial_nll(counts, probabilities) == pytest.approx(expected_nll)


def test_conditional_circle_probabilities_are_normalized_and_not_free() -> None:
    g_row, e_row = transition_state_circle_probabilities(-1.0, 1.0, 0.2, 0.25, 0.4, 0.5)
    np.testing.assert_allclose(g_row.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(e_row.sum(), 1.0, atol=1e-14)
    assert g_row[0] > g_row[1]
    assert e_row[1] > e_row[0]
    with pytest.raises(ValueError, match="must not overlap"):
        transition_state_circle_probabilities(-1.0, 1.0, 0.2, 0.25, 0.4, 1.1)


def test_public_analysis_recovers_identifiable_raw_iq_joint_fit() -> None:
    truth = LenRabiPhysicalParams(
        p_e0=0.12,
        p_inf=0.48,
        center_g=-1.0,
        center_e=1.0,
        sigma=0.16,
        p_avg=0.28,
        length_ratio=0.45,
        t_r=3.5,
        omega=3.2,
    )
    lengths = np.linspace(0.05, 3.2, 17)
    signals = _synthetic_raw(lengths, truth, shots=1000, seed=20260901)
    result = LenRabiResult(lengths, np.arange(signals.shape[1]), signals)

    fit, figure = LenRabiExp().analyze(result, max_calls=20_000)

    assert isinstance(figure, Figure)
    assert fit.backend.valid
    assert fit.backend.parameter_names == (
        "p_e0_u",
        "p_inf_u",
        "center_mid",
        "log_separation",
        "log_sigma",
        "p_avg_u",
        "log_length_ratio",
        "log_t_r",
        "log_omega",
    )
    assert fit.backend.covariance.shape == (9, 9)
    assert fit.initial_populations[0] > fit.initial_populations[1]
    assert fit.initial_populations[1] == pytest.approx(truth.p_e0, abs=0.08)
    assert fit.p_inf == pytest.approx(truth.p_inf, abs=0.12)
    assert fit.omega == pytest.approx(truth.omega, abs=0.45)
    assert fit.t_r == pytest.approx(truth.t_r, rel=0.5)
    assert fit.sigma == pytest.approx(truth.sigma, abs=0.08)
    assert fit.p_avg == pytest.approx(truth.p_avg, abs=0.1)
    assert fit.length_ratio == pytest.approx(truth.length_ratio, abs=0.2)
    expected_axis = np.exp(0.61j)
    assert abs(fit.g_center - (0.35 - 0.22j - expected_axis)) < 0.15
    assert abs(fit.e_center - (0.35 - 0.22j + expected_axis)) < 0.15
    assert abs(fit.e_center - fit.g_center) == pytest.approx(2.0, abs=0.3)
    assert 0.0 < fit.radius <= 0.5 * abs(fit.e_center - fit.g_center)
    np.testing.assert_allclose(fit.confusion_matrix.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_array_equal(fit.confusion_matrix[2], [0.0, 0.0, 1.0])
    assert np.isfinite(fit.condition_number)
    assert fit.measured_populations.shape == (lengths.size, 3)
    assert fit.fitted_populations.shape == (lengths.size, 3)
    assert len(figure.axes[0].lines) == 6
    plt.close(figure)


def test_invalid_backend_keeps_diagnostics_but_blocks_finite_calibration() -> None:
    truth = LenRabiPhysicalParams(0.12, 0.48, -1.0, 1.0, 0.16, 0.28, 0.45, 3.5, 3.2)
    lengths = np.linspace(0.05, 3.2, 9)
    signals = _synthetic_raw(lengths, truth, shots=250, seed=9)

    fit = fit_len_rabi_joint(lengths, signals, max_calls=1)

    assert not fit.backend.valid
    assert fit.backend.calls > 0
    assert np.isnan(fit.initial_populations[:2]).all()
    assert np.isnan(fit.g_center.real)
    assert np.isnan(fit.radius)
    assert np.isnan(fit.confusion_matrix).all()


def test_radius_optimizer_failure_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = LenRabiPhysicalParams(0.1, 0.5, -1.0, 1.0, 0.2, 0.25, 0.4, 2.0, 3.0)
    monkeypatch.setattr(
        fit_module,
        "minimize_scalar",
        lambda *args, **kwargs: OptimizeResult(success=False, x=0.4),
    )

    with pytest.raises(RuntimeError, match="radius optimization failed"):
        fit_module._confusion_matrix(params)


def test_fit_failure_keeps_numeric_and_figure_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lengths = np.linspace(0.1, 1.0, 5)
    rng = np.random.default_rng(7)
    signals = rng.normal(size=(5, 30)) + 1j * rng.normal(size=(5, 30))

    def fail_backend(*args: object, **kwargs: object) -> object:
        raise RuntimeError("optimizer unavailable")

    monkeypatch.setattr(fit_module, "_fit_backend", fail_backend)
    result = LenRabiResult(
        lengths,
        np.arange(signals.shape[1]),
        np.asarray(signals, dtype=np.complex128),
    )

    fit, figure = LenRabiExp().analyze(result)

    assert not fit.backend.valid
    assert np.isnan(fit.initial_populations[:2]).all()
    assert np.isnan(fit.confusion_matrix).all()
    assert fit.condition_number == np.inf
    assert fit.measured_populations.shape == (5, 3)
    assert fit.fitted_populations.shape == (5, 3)
    assert figure.axes[0].get_title() == "Len Rabi joint fit invalid"
    plt.close(figure)


@pytest.mark.parametrize(
    "lengths, signals, message",
    [
        (
            np.array([0.1]),
            np.ones((1, 3), dtype=np.complex128),
            "at least two finite lengths",
        ),
        (
            np.array([0.1, 0.2]),
            np.ones((2, 3), dtype=np.complex128),
            "projection axis",
        ),
        (
            np.array([0.1, 0.2]),
            np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]], dtype=np.complex128),
            "finite raw IQ",
        ),
    ],
)
def test_joint_fit_raises_when_projection_or_required_shape_cannot_be_built(
    lengths: NDArray[np.float64],
    signals: NDArray[np.complex128],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_len_rabi_joint(lengths, signals)

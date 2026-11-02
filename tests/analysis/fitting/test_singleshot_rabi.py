from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
import zcu_tools.analysis.fitting.singleshot_rabi as rabi_fit
from zcu_tools.analysis.fitting.singleshot_rabi import (
    RabiPhysicalParams,
    fit_rabi_joint,
    rabi_excited_population,
)


def test_nondecay_rabi_population_has_constant_envelope() -> None:
    gains = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    params = RabiPhysicalParams(
        p_e0=0.05,
        p_inf=0.5,
        center_g=-1.0,
        center_e=1.0,
        sigma=0.2,
        p_avg=0.1,
        length_ratio=0.1,
        t_r=None,
        omega=np.pi,
    )

    np.testing.assert_allclose(
        rabi_excited_population(gains, params),
        [0.5, 0.05, 0.5],
        atol=1e-12,
    )


def test_nondecay_joint_fit_omits_decay_parameter() -> None:
    rng = np.random.default_rng(84)
    gains = np.linspace(0.0, 1.0, 15)
    excited_probability = 0.5 - 0.45 * np.cos(4.0 * np.pi * gains)
    excited = rng.random((gains.size, 300)) < excited_probability[:, None]
    signals = rng.normal(np.where(excited, 1.0, -1.0), 0.18).astype(np.complex128)

    fit = fit_rabi_joint(gains, signals, decay=False)

    assert fit.backend.valid
    assert fit.t_r is None
    assert "log_t_r" not in fit.backend.parameter_names
    assert fit.backend.values.shape == (8,)
    assert fit.phase == 0.0
    assert "phase" not in fit.backend.parameter_names
    np.testing.assert_allclose(fit.zero_length_populations, fit.initial_populations)
    assert fit.backend.covariance.shape == (8, 8)
    np.testing.assert_allclose(
        fit.fitted_populations[:, 1], excited_probability, atol=0.12
    )


def test_decay_joint_fit_retains_decay_parameter() -> None:
    rng = np.random.default_rng(85)
    lengths = np.linspace(0.0, 1.5, 21)
    excited_probability = 0.5 - 0.45 * np.exp(-lengths / 1.5) * np.cos(
        4.0 * np.pi * lengths
    )
    excited = rng.random((lengths.size, 300)) < excited_probability[:, None]
    signals = rng.normal(np.where(excited, 1.0, -1.0), 0.18).astype(np.complex128)

    fit = fit_rabi_joint(lengths, signals)

    assert fit.backend.valid
    assert fit.t_r is not None and fit.t_r > 0.0
    assert "log_t_r" in fit.backend.parameter_names
    assert fit.backend.values.shape == (9,)
    assert fit.backend.covariance.shape == (9, 9)
    np.testing.assert_allclose(
        fit.fitted_populations[:, 1], excited_probability, atol=0.12
    )


@pytest.mark.parametrize("initial_state", ["ground", "excited"])
@pytest.mark.parametrize("decay,start", [(False, 0.0), (False, -0.25), (True, 0.25)])
def test_joint_fit_labels_physical_states_at_zero_drive(
    initial_state: Literal["ground", "excited"], decay: bool, start: float
) -> None:
    rng = np.random.default_rng(238)
    xs = np.linspace(start, start + 1.5, 25)
    p_e0 = 0.08 if initial_state == "ground" else 0.92
    envelope = np.exp(-xs / 2.0) if decay else np.ones_like(xs)
    p_e = 0.5 + (p_e0 - 0.5) * envelope * np.cos(4 * np.pi * xs)
    excited = rng.random((xs.size, 600)) < p_e[:, None]
    g_center, e_center = -1 - 0.4j, 1 + 0.4j
    signals = np.asarray(
        np.where(excited, e_center, g_center)
        + 0.18 * (rng.normal(size=excited.shape) + 1j * rng.normal(size=excited.shape)),
        dtype=np.complex128,
    )
    original = signals.copy()
    fit = fit_rabi_joint(xs, signals, decay=decay, initial_state=initial_state)
    assert fit.backend.valid
    assert fit.initial_populations[1] == pytest.approx(p_e0, abs=0.1)
    assert abs(fit.g_center - g_center) < 0.12
    assert abs(fit.e_center - e_center) < 0.12
    assert fit.omega == pytest.approx(4 * np.pi, rel=0.05)
    np.testing.assert_allclose(fit.fitted_populations[:, 1], p_e, atol=0.1)
    np.testing.assert_allclose(fit.measured_populations[:, 1], p_e, atol=0.1)
    np.testing.assert_allclose(fit.confusion_matrix, np.eye(3), atol=0.1)
    np.testing.assert_array_equal(signals, original)


def test_joint_fit_rejects_unknown_initial_state() -> None:
    invalid_state: Any = "unknown"
    with pytest.raises(ValueError, match="Unknown initial state"):
        fit_rabi_joint(
            np.array([0.0, 1.0]),
            np.zeros((2, 2), dtype=np.complex128),
            initial_state=invalid_state,
        )


def test_joint_fit_rejects_candidate_with_failed_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zcu_tools.analysis.fitting.singleshot_rabi as rabi_fit
    from scipy.optimize import OptimizeResult, minimize_scalar

    rng = np.random.default_rng(84)
    gains = np.linspace(0.0, 1.0, 15)
    p_e = 0.5 - 0.45 * np.cos(4 * np.pi * gains)
    excited = rng.random((gains.size, 300)) < p_e[:, None]
    signals = rng.normal(np.where(excited, 1.0, -1.0), 0.18).astype(np.complex128)

    def optimizer(*args, **options):
        if "options" in options:
            return OptimizeResult(success=False, x=np.nan)
        return minimize_scalar(*args, **options)

    # Valid separated IQ data cannot reliably make SciPy's radius search fail.
    monkeypatch.setattr(rabi_fit, "_minimize_scalar", optimizer)
    fit = fit_rabi_joint(gains, signals, decay=False)
    assert not fit.backend.valid
    assert np.isnan(fit.radius)
    assert np.isnan(fit.confusion_matrix).all()


@pytest.mark.parametrize("initial_state", ["ground", "excited"])
@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("phase", [-0.6, 0.6])
def test_joint_fit_recovers_phase_and_pre_pulse_population(
    initial_state: Literal["ground", "excited"], decay: bool, phase: float
) -> None:
    rng = np.random.default_rng(290)
    lengths = np.linspace(0.2, 1.7, 31)
    p_e0 = 0.08 if initial_state == "ground" else 0.92
    envelope = np.exp(-lengths / 2.0) if decay else np.ones_like(lengths)
    p_e = 0.5 + (p_e0 - 0.5) * envelope * np.cos(4 * np.pi * lengths + phase)
    excited = rng.random((lengths.size, 1000)) < p_e[:, None]
    signals = np.asarray(
        np.where(excited, 1 + 0.4j, -1 - 0.4j)
        + 0.18 * (rng.normal(size=excited.shape) + 1j * rng.normal(size=excited.shape)),
        dtype=np.complex128,
    )
    fit = fit_rabi_joint(
        lengths, signals, decay=decay, fit_phase=True, initial_state=initial_state
    )

    assert fit.backend.valid
    assert fit.phase == pytest.approx(phase, abs=0.12)
    assert fit.omega == pytest.approx(4 * np.pi, rel=0.025)
    assert fit.initial_populations[1] == pytest.approx(p_e0, abs=0.06)
    assert fit.zero_length_populations[1] == pytest.approx(
        0.5 + (p_e0 - 0.5) * np.cos(phase), abs=0.05
    )
    assert abs(fit.zero_length_populations[1] - fit.initial_populations[1]) > 0.02
    assert abs(fit.g_center - (-1 - 0.4j)) < 0.1
    assert abs(fit.e_center - (1 + 0.4j)) < 0.1
    np.testing.assert_allclose(fit.fitted_populations[:, 1], p_e, atol=0.06)
    np.testing.assert_allclose(fit.confusion_matrix, np.eye(3), atol=0.1)
    size = 10 if decay else 9
    assert fit.backend.covariance.shape == (size, size)
    phase_index = fit.backend.parameter_names.index("phase")
    assert np.isfinite(fit.backend.covariance[phase_index]).all()
    assert fit.backend.covariance[phase_index, phase_index] > 0


@pytest.mark.parametrize("fit_phase", [False, True])
@pytest.mark.parametrize("decay", [False, True])
def test_failed_joint_fit_preserves_selected_parameter_schema(
    fit_phase: bool, decay: bool
) -> None:
    rng = np.random.default_rng(123)
    lengths = np.linspace(0, 1, 15)
    p_e = 0.5 - 0.4 * np.cos(4 * np.pi * lengths + 0.5)
    excited = rng.random((15, 100)) < p_e[:, None]
    signals = rng.normal(np.where(excited, 1.0, -1.0), 0.18).astype(np.complex128)
    fit = fit_rabi_joint(
        lengths, signals, decay=decay, fit_phase=fit_phase, max_calls=1
    )
    assert not fit.backend.valid
    assert ("phase" in fit.backend.parameter_names) is fit_phase
    assert ("log_t_r" in fit.backend.parameter_names) is decay
    assert np.isnan(fit.confusion_matrix).all()
    assert np.isnan(fit.zero_length_populations[:2]).all()


@pytest.mark.parametrize("valid", [True, False])
def test_joint_fit_summary_preserves_calibration_and_marginal_errors(
    valid: bool,
) -> None:
    from zcu_tools.gui.app.measure.summary import summarize_analysis

    # A complete numerical result includes many sweep rows and a 9x9 covariance.
    backend = rabi_fit.RabiBackendResult(
        parameter_names=(
            "p_e0",
            "p_inf",
            "g",
            "e",
            "sigma",
            "p_avg",
            "ratio",
            "log_omega",
            "phase",
        ),
        values=np.arange(9, dtype=np.float64),
        covariance=np.diag(np.arange(1, 10, dtype=np.float64) ** 2),
        valid=valid,
        covariance_accurate=True,
        reached_call_limit=False,
        hesse_failed=False,
        edm=1e-8,
        nll=123.0 if valid else np.inf,
        calls=42,
    )
    result = rabi_fit.RabiJointFitResult(
        initial_populations=np.array([0.9, 0.1, 0.0]),
        p_inf=0.5,
        t_r=2.0,
        omega=4 * np.pi,
        projected_g_center=-1.0,
        projected_e_center=1.0,
        sigma=0.2,
        p_avg=0.01,
        length_ratio=0.05,
        g_center=-1 - 0.4j,
        e_center=1 + 0.4j,
        radius=0.7,
        confusion_matrix=np.eye(3),
        condition_number=1.0,
        measured_populations=np.zeros((31, 3)),
        fitted_populations=np.zeros((31, 3)),
        projection=rabi_fit.RabiProjection(
            projected=np.zeros((31, 1000)),
            bin_edges=np.linspace(-2, 2, 101),
            counts=np.zeros((31, 100), dtype=np.int64),
            origin=0j,
            axis=1 + 0j,
            perpendicular_offset=0.0,
        ),
        backend=backend,
        phase=0.6,
    )

    summary = rabi_fit.summarize_rabi_joint_fit(result)

    np.testing.assert_array_equal(summary.initial_populations, [0.9, 0.1, 0.0])
    np.testing.assert_array_equal(summary.confusion_matrix, np.eye(3))
    assert (summary.p_inf, summary.t_r, summary.omega, summary.phase) == (
        0.5,
        2.0,
        4 * np.pi,
        0.6,
    )
    assert (summary.g_center, summary.e_center, summary.radius) == (
        -1 - 0.4j,
        1 + 0.4j,
        0.7,
    )
    assert (summary.projected_g_center, summary.projected_e_center, summary.sigma) == (
        -1.0,
        1.0,
        0.2,
    )
    assert (summary.p_avg, summary.length_ratio, summary.condition_number) == (
        0.01,
        0.05,
        1.0,
    )
    assert summary.backend.parameter_names == backend.parameter_names
    np.testing.assert_array_equal(summary.backend.values, np.arange(9))
    np.testing.assert_array_equal(summary.backend.std, np.arange(1, 10))
    assert summary.backend.valid is valid
    assert summary.backend.nll == (123.0 if valid else np.inf)
    projected = summarize_analysis(summary)
    assert projected.summary["initial_populations"] == [0.9, 0.1, 0.0]
    assert projected.summary["confusion_matrix"] == np.eye(3).tolist()
    assert [issue.path for issue in projected.invalid] == (
        [] if valid else ["summary.backend.nll"]
    )

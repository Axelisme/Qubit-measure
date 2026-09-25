from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pytest
from zcu_tools.experiment.v2.singleshot.amp_rabi import AmpRabiExp, AmpRabiResult
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiExp, LenRabiResult
from zcu_tools.experiment.v2.singleshot.rabi_fit import (
    RabiJointFitResult,
    RabiPhysicalParams,
    fit_rabi_joint,
    rabi_excited_population,
)


@pytest.mark.parametrize("initial_state", ["ground", "excited"])
def test_amp_analysis_always_disables_decay(
    monkeypatch: pytest.MonkeyPatch, initial_state: Literal["ground", "excited"]
) -> None:
    called: list[tuple[bool, str]] = []

    def fake_fit(
        *args: Any, decay: bool, initial_state: str, **kwargs: Any
    ) -> RabiJointFitResult:
        called.append((decay, initial_state))
        raise RuntimeError("fit called")

    monkeypatch.setattr(
        "zcu_tools.experiment.v2.singleshot.amp_rabi.fit_rabi_joint", fake_fit
    )
    result = AmpRabiResult(
        gains=np.array([0.0]),
        shot_indices=np.array([0]),
        signals=np.array([[0.0j]]),
    )
    with pytest.raises(RuntimeError, match="fit called"):
        AmpRabiExp().analyze(result, initial_state=initial_state)
    assert called == [(False, initial_state)]


@pytest.mark.parametrize("initial_state", ["ground", "excited"])
@pytest.mark.parametrize("decay", [False, True])
def test_len_experiment_forwards_decay_to_joint_fit(
    monkeypatch: pytest.MonkeyPatch,
    decay: bool,
    initial_state: Literal["ground", "excited"],
) -> None:
    called: list[tuple[bool, str]] = []

    def fake_fit(
        *args: Any, decay: bool, initial_state: str, **kwargs: Any
    ) -> RabiJointFitResult:
        called.append((decay, initial_state))
        raise RuntimeError("fit called")

    monkeypatch.setattr(
        "zcu_tools.experiment.v2.singleshot.len_rabi.fit_rabi_joint", fake_fit
    )
    result = LenRabiResult(
        lengths=np.array([0.0]),
        shot_indices=np.array([0]),
        signals=np.array([[0.0j]]),
    )
    with pytest.raises(RuntimeError, match="fit called"):
        LenRabiExp().analyze(result, decay=decay, initial_state=initial_state)
    assert called == [(decay, initial_state)]


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
    with pytest.raises(ValueError, match="Unknown initial state"):
        fit_rabi_joint(
            np.array([0.0, 1.0]),
            np.zeros((2, 2), dtype=np.complex128),
            initial_state=cast(Any, "unknown"),
        )


def test_joint_fit_rejects_candidate_with_failed_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zcu_tools.experiment.v2.singleshot import rabi_fit

    rng = np.random.default_rng(84)
    gains = np.linspace(0.0, 1.0, 15)
    p_e = 0.5 - 0.45 * np.cos(4 * np.pi * gains)
    excited = rng.random((gains.size, 300)) < p_e[:, None]
    signals = rng.normal(np.where(excited, 1.0, -1.0), 0.18).astype(np.complex128)

    def fail_calibration(params: RabiPhysicalParams) -> Any:
        raise RuntimeError("radius optimization failed")

    monkeypatch.setattr(rabi_fit, "_confusion_matrix", fail_calibration)
    fit = fit_rabi_joint(gains, signals, decay=False)
    assert not fit.backend.valid
    assert np.isnan(fit.confusion_matrix).all()

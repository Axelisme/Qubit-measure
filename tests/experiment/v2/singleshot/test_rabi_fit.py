from __future__ import annotations

from typing import Any

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


def test_amp_analysis_always_disables_decay(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    def fake_fit(*args: Any, decay: bool, **kwargs: Any) -> RabiJointFitResult:
        called.append(decay)
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
        AmpRabiExp().analyze(result)
    assert called == [False]


@pytest.mark.parametrize("decay", [False, True])
def test_len_experiment_forwards_decay_to_joint_fit(
    monkeypatch: pytest.MonkeyPatch, decay: bool
) -> None:
    called: list[bool] = []

    def fake_fit(*args: Any, decay: bool, **kwargs: Any) -> RabiJointFitResult:
        called.append(decay)
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
        LenRabiExp().analyze(result, decay=decay)
    assert called == [decay]


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

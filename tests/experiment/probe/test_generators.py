"""Recover known parameters from generated arrays using public family fitters."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from zcu_tools.analysis.fitting import fit_decay, fit_decay_fringe, fit_rabi, fitlor
from zcu_tools.analysis.fitting.resonance import HangerModel, TransmissionModel
from zcu_tools.experiment.probe import generators


def test_decay_recovers_truth_with_seeded_noise():
    xs = np.linspace(0, 20, 201)
    ys = generators.generate_decay(
        xs, generators.DecayParameters(0.2, 0.8, 4.0), 0.001, 17
    )
    fit = fit_decay(xs, ys)
    assert_allclose(fit.params, [0.2, 0.8, 4.0], atol=0.01)
    assert fit.decay_time == pytest.approx(4.0, rel=0.005)
    assert_allclose(
        generators.generate_decay(
            xs, generators.DecayParameters(0.2, 0.8, 4.0), 0.001, 17
        ),
        ys,
        atol=0,
        rtol=0,
    )


def test_decay_fringe_recovers_time_frequency_and_phase():
    xs = np.linspace(0, 20, 401)
    ys = generators.generate_decay_fringe(
        xs, generators.OscillationParameters(0.2, 0.8, 0.4, 25.0, 8.0), 0.001, 17
    )
    fit = fit_decay_fringe(xs, ys, fit_params=[0.1, 0.9, 0.39, 20.0, 9.0])
    assert_allclose(fit.params, [0.2, 0.8, 0.4, 25.0, 8.0], rtol=0.01, atol=0.002)


@pytest.mark.parametrize("decay_time", [None, 8.0])
def test_rabi_recovers_frequency_and_pulse_coordinate(decay_time: float | None):
    xs = np.linspace(0, 20, 401)
    ys = generators.generate_rabi(
        xs, generators.OscillationParameters(0.2, 0.8, 0.4, 0.0, decay_time), 0.001, 17
    )
    fit = fit_rabi(xs, ys, decay=decay_time is not None, init_phase=0.0)
    assert fit.freq == pytest.approx(0.4, rel=0.001)
    assert fit.pi_x == pytest.approx(1.25, rel=0.001)
    expected = [0.2, 0.8, 0.4, 0.0]
    if decay_time is not None:
        expected.append(decay_time)
    assert_allclose(fit.params, expected, rtol=0.01, atol=0.002)


def test_lorentzian_recovers_peak_and_sloped_background():
    xs = np.linspace(-5, 5, 401)
    ys = generators.generate_lorentzian(
        xs, generators.LorentzianParameters(0.2, 0.01, 0.8, 0.4, 0.5), 0.001, 17
    )
    params, _ = fitlor(xs, ys)
    assert_allclose(params, [0.2, 0.01, 0.8, 0.4, 0.5], atol=0.003)


@pytest.mark.parametrize("coupling_quality", [None, 1100.0])
def test_resonator_recovers_complex_resonance(coupling_quality: float | None):
    xs = np.linspace(5965.0, 6035.0, 301)
    ys = generators.generate_resonator(
        xs,
        generators.ResonatorParameters(
            6000.0, 700.0, coupling_quality, 0.1, 1.7 + 0.3j, 0.02, 0.002, 0.0001
        ),
        0.00001,
        17,
    )
    model = TransmissionModel if coupling_quality is None else HangerModel
    fit = model.fit(
        xs, ys, edelay=0.02, fit_bg_amp_slope=True, fit_bg_phase_curvature=True
    )
    assert fit["freq"] == pytest.approx(6000.0, abs=0.01)
    assert fit["Ql"] == pytest.approx(700.0, rel=0.01)
    assert fit["a0"] == pytest.approx(1.7 + 0.3j, abs=0.01)
    assert fit["edelay"] == pytest.approx(0.02, abs=1e-5)
    assert fit["bg_amp_slope"] == pytest.approx(0.002, abs=1e-5)
    assert fit["bg_phase_curvature"] == pytest.approx(0.0001, abs=1e-6)

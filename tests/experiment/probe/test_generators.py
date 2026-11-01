"""Recover known parameters from generated arrays using public family fitters."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from zcu_tools.analysis.fitting import (
    fit_decay,
    fit_decay_fringe,
    fit_qubit_freq,
    fit_rabi,
    fitlor,
)
from zcu_tools.analysis.fitting.anticross import fit_anticross
from zcu_tools.analysis.fitting.multi_decay import (
    TransitionRates,
    fit_dual_transition_rates,
)
from zcu_tools.analysis.fitting.resonance import HangerModel, TransmissionModel
from zcu_tools.analysis.fitting.singleshot import fit_singleshot
from zcu_tools.analysis.fitting.singleshot_rabi import (
    RabiPhysicalParams,
    fit_rabi_joint,
)
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


@pytest.mark.parametrize(
    "field, coupling_quality",
    [
        ("frequency", None),
        ("frequency", 1100.0),
        ("loaded_quality", None),
        ("loaded_quality", 1100.0),
        ("coupling_quality", 1100.0),
    ],
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_resonator_rejects_nonfinite_or_nonpositive_frequency_and_quality(
    field: str, coupling_quality: float | None, invalid: float
):
    truth = generators.ResonatorParameters(
        6000.0, 700.0, coupling_quality, 0.1, 1.7 + 0.3j, 0.02, 0.002, 0.0001
    )
    truth = replace(truth, **{field: invalid})
    with pytest.raises(ValueError, match=field):
        generators.generate_resonator(np.array([5990.0, 6000.0, 6010.0]), truth, 0.0, 17)


def test_qubit_frequency_recovers_dip_center_and_linewidth():
    xs = np.linspace(4950.0, 5050.0, 4001)
    ys = generators.generate_qubit_frequency(
        xs, generators.LorentzianParameters(1.0, 0.0, -0.8, 5000.4, 0.5), 0.001, 17
    )
    frequency, _, fwhm, _, _, (params, _) = fit_qubit_freq(xs, ys)
    assert frequency == pytest.approx(5000.4, abs=0.003)
    assert fwhm == pytest.approx(1.0, abs=0.01)
    assert_allclose(params, [1.0, 0.0, -0.8, 5000.4, 0.5], atol=0.003)


def test_singleshot_ge_raw_iq_recovers_centers_width_and_misassignment():
    signals = generators.generate_singleshot_ge(
        np.linspace(-3.0, 3.0, 1201),
        generators.GaussianCloudParameters(-1.0 + 0.4j, 1.0 + 0.4j, 0.2, 100000, 0.05),
        17,
    )
    assert signals.shape == (2, 100000)
    assert signals.dtype == np.complex128
    edges = np.linspace(-3.0, 3.0, 121)
    centers = (edges[:-1] + edges[1:]) / 2
    histograms = np.array([np.histogram(row.real, edges)[0] for row in signals])
    masses = histograms / histograms.sum(axis=1, keepdims=True)
    params, _ = fit_singleshot(
        centers, *masses, fixedparams=[None, None, None, None, None, 0.5, 0.0]
    )
    assert_allclose(params[:5], [-1.0, 1.0, 0.2, 0.95, 0.05], atol=0.01)
    assert_allclose(np.mean(signals.imag, axis=1), [0.4, 0.4], atol=0.005)
    assert_allclose(np.std(signals.imag, axis=1), [0.2, 0.2], atol=0.005)


def test_singleshot_rabi_raw_iq_recovers_population_dynamics():
    lengths = np.linspace(0.0, 1.5, 31)
    truth = RabiPhysicalParams(
        0.08, 0.5, -1.0, 1.0, 0.18, 0.2, 0.3, 1.5, 4 * np.pi, 0.0
    )
    signals = generators.generate_singleshot_rabi(
        lengths, np.linspace(-3, 3, 2401), truth, 4000, 0.18, 17
    )
    assert signals.shape == (31, 4000)
    fit = fit_rabi_joint(lengths, signals)
    assert fit.backend.valid
    assert fit.omega == pytest.approx(4 * np.pi, rel=0.02)
    assert fit.t_r == pytest.approx(1.5, rel=0.1)
    assert fit.initial_populations[1] == pytest.approx(0.08, abs=0.02)
    assert fit.p_inf == pytest.approx(0.5, abs=0.02)
    assert fit.g_center == pytest.approx(-1.0, abs=0.02)
    assert fit.e_center == pytest.approx(1.0, abs=0.02)
    assert fit.sigma == pytest.approx(0.18, abs=0.02)
    assert fit.p_avg == pytest.approx(0.2, abs=0.08)
    assert fit.length_ratio == pytest.approx(0.3, abs=0.06)


def test_anticrossing_recovers_identifiable_center_and_coupling():
    xs = np.linspace(-1.7, 2.3, 81)
    branches = generators.generate_anticrossing(
        xs, generators.ConicParameters(-1.0, 0.6, 1.0, -3.0, 0.0, 2.0), 0.0001, 17
    )
    assert branches.shape == (2, 81)
    center_x, center_y, coupling, m1, m2, *_ = fit_anticross(xs, *branches)
    assert_allclose([center_x, center_y, coupling], [0.3, 1.5, 0.4], atol=0.001)
    assert_allclose(sorted([m1, m2]), [-1.0, 1.0], atol=0.001)


def test_multi_decay_two_initial_states_recover_all_six_rates():
    times = np.linspace(0.0, 20.0, 160)
    rates = TransitionRates(0.1, 0.05, 0.08, 0.04, 0.02, 0.01)
    ground = generators.generate_multi_decay(
        times, generators.MultiDecayParameters(rates, 0.96, 0.03), 0.0002, 17
    )
    excited = generators.generate_multi_decay(
        times, generators.MultiDecayParameters(rates, 0.04, 0.95), 0.0002, 18
    )
    assert ground.shape == excited.shape == (160, 3)
    fit = fit_dual_transition_rates(times, ground, excited)
    assert fit.diagnostics.valid
    assert_allclose(fit.rates.as_array(), rates.as_array(), atol=0.003)
    assert_allclose(fit.initial_populations1, [0.96, 0.03], atol=0.002)
    assert_allclose(fit.initial_populations2, [0.04, 0.95], atol=0.002)

"""Recover known parameters from generated arrays using public family fitters."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from zcu_tools.analysis.fitting import fit_decay, fit_decay_fringe
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
    fit = fit_decay_fringe(xs, ys)
    assert_allclose(fit.params, [0.2, 0.8, 0.4, 25.0, 8.0], rtol=0.01, atol=0.002)

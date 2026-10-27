"""Common-frame dynamics through production Engine routing and strict lab references."""

from contextlib import closing
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine
from zcu_tools.simulate.evolution.kernels import REGISTRY

from .._common_frame import batch, model_source, strict_lab


def evolve(source, timeline):
    with closing(Engine(1)) as engine:
        return engine.evolve(source, (timeline,), cancel=Event(), budget=0.01)


@pytest.mark.parametrize("coherent", [False, True])
def test_complex_envelope_offsets_and_held_endpoint_match_strict_lab(coherent):
    source = model_source(coherent)
    timeline = batch([16, 38, 67], gain=0.6, shaped=True)
    result = evolve(source, timeline)
    expected = strict_lab(source, timeline)
    assert result.paths.segments_by_kernel.get("common_frame", 0) > 0
    # Coherences make endpoint dressing and the DC frame restoration observable.
    np.testing.assert_allclose(result.densities, expected, atol=0.015, rtol=0)
    np.testing.assert_allclose(
        np.trace(result.densities, axis1=1, axis2=2), 1, atol=1e-12
    )


def test_rabi_period_and_decay_follow_lab_curve():
    source = model_source()
    timeline = batch(np.linspace(8, 104, 13), gain=0.6)
    result = evolve(source, timeline)
    expected = strict_lab(source, timeline)
    actual_pop = result.densities[:, 1, 1].real
    expected_pop = expected[:, 1, 1].real
    np.testing.assert_allclose(actual_pop, expected_pop, atol=0.025, rtol=0)
    assert abs(np.argmax(actual_pop) - np.argmax(expected_pop)) <= 1
    assert np.ptp(actual_pop) > 0.15
    assert result.paths.segments_by_kernel.get("common_frame", 0) > 0


@pytest.mark.parametrize(
    "gain,duration,frequency", [(0.5, 32, 0.292), (1.0, 64, 0.294), (2.0, 96, 0.298)]
)
def test_near_carrier_parameter_offsets_remain_admitted(gain, duration, frequency):
    source = model_source()
    timeline = batch([duration] * 4, gain=0.3 * gain, frequency=frequency)
    result = evolve(source, timeline)
    expected_family = "multitone_rwa" if gain == 0.5 else "common_frame"
    assert result.paths.segments_by_kernel == {expected_family: 4}
    np.testing.assert_allclose(
        result.densities.real.diagonal(axis1=1, axis2=2),
        strict_lab(source, timeline).real.diagonal(axis1=1, axis2=2),
        atol=0.1,
        rtol=0,
    )


@pytest.mark.parametrize("gain,frequency", [(15.0, 0.294), (1.0, 0.7)])
def test_over_budget_and_expensive_rows_use_existing_fallback(gain, frequency):
    source = model_source()
    timeline = batch([16], gain=gain, frequency=frequency)
    result = evolve(source, timeline)
    assert result.paths.segments_by_kernel.get("common_frame", 0) == 0
    with closing(
        Engine(1, kernels=tuple(k for k in REGISTRY if k.name != "common_frame"))
    ) as engine:
        original = engine.evolve(source, (timeline,), cancel=Event(), budget=0.01)
    np.testing.assert_array_equal(result.densities, original.densities)

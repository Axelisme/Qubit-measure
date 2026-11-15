"""Common-frame numerical features and default fallback against strict lab references."""

from contextlib import closing
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine

from .._common_frame import batch, model_source, strict_lab


def evolve(source, timeline, *, families: tuple[str, ...] | None = None):
    with closing(Engine(1, families=families)) as engine:
        return engine.evolve(source, (timeline,), cancel=Event(), budget=0.01)


@pytest.mark.parametrize(
    "coherent,gain,families",
    [
        pytest.param(False, 0.6, None, id="mixed-fallback"),
        pytest.param(True, 0.6, None, id="coherent-fallback"),
        pytest.param(
            False, 0.2, ("common_frame", "shaped_sambe"), id="mixed-common-frame"
        ),
        pytest.param(
            True, 0.2, ("common_frame", "shaped_sambe"), id="coherent-common-frame"
        ),
    ],
)
def test_complex_envelope_offsets_and_held_endpoint_match_strict_lab(
    coherent: bool, gain: float, families: tuple[str, ...] | None
):
    source = model_source(coherent)
    timeline = batch([16, 38, 67], gain=gain, shaped=True)
    # The admitted cases need shaped_sambe for the five-ns single-tone prefix.
    result = evolve(source, timeline, families=families)
    expected = strict_lab(source, timeline)
    if families is None:
        assert result.paths.segments_by_kernel.get("common_frame", 0) == 0
    else:
        assert result.paths.segments_by_kernel.get("common_frame", 0) > 0
    # The coherent admitted case exercises endpoint dressing and DC restoration.
    np.testing.assert_allclose(result.densities, expected, atol=0.015, rtol=0)
    np.testing.assert_allclose(
        np.trace(result.densities, axis1=1, axis2=2), 1, atol=1e-12
    )


@pytest.mark.parametrize(
    "gain", [pytest.param(0.6, id="fallback"), pytest.param(0.2, id="common-frame")]
)
def test_rabi_period_and_decay_follow_lab_curve(gain: float):
    source = model_source()
    timeline = batch(np.linspace(8, 104, 13), gain=gain)
    families = ("common_frame",) if gain == 0.2 else None
    result = evolve(source, timeline, families=families)
    expected = strict_lab(source, timeline)
    actual_pop = result.densities[:, 1, 1].real
    expected_pop = expected[:, 1, 1].real
    np.testing.assert_allclose(actual_pop, expected_pop, atol=0.025, rtol=0)
    assert abs(np.argmax(actual_pop) - np.argmax(expected_pop)) <= 1
    assert np.ptp(actual_pop) > 0.15
    if families is None:
        assert result.paths.segments_by_kernel.get("common_frame", 0) == 0
    else:
        assert result.paths.segments_by_kernel.get("common_frame", 0) > 0


@pytest.mark.parametrize(
    "gain,duration,frequency", [(0.5, 32, 0.292), (1.0, 64, 0.294), (2.0, 96, 0.298)]
)
def test_near_carrier_admission_and_fallback_match_strict_lab(
    gain, duration, frequency
):
    source = model_source()
    timeline = batch([duration] * 4, gain=0.3 * gain, frequency=frequency)
    result = evolve(source, timeline)
    if gain == 2.0:
        assert result.paths.segments_by_kernel.get("common_frame", 0) == 0
    else:
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
        Engine(1, families=("multitone_rwa", "multitone_gate", "floquet", "dop853"))
    ) as engine:
        original = engine.evolve(source, (timeline,), cancel=Event(), budget=0.01)
    np.testing.assert_array_equal(result.densities, original.densities)

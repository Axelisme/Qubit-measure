"""Complex sampled single-tone gates at the public Engine seam."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, NoValidKernel

from .accuracy_cases import coherent_source, drive_track, linear_drive_model, timeline


def complex_envelope(shape: str) -> EnvelopeSpec:
    if shape == "imaginary-ramp":
        # The normalized slope is just inside the guard, with no in-phase peak.
        return EnvelopeSpec(
            "sampled", np.array([0.800001j, 0.9000005j, 1j]), np.array([0.0, 4.0, 8.0])
        )
    # At 154 ns this FAST-DRAG envelope is just inside the slope boundary.
    duration_ns = 128 if shape == "drag" else 154
    times = np.arange(duration_ns + 1, dtype=float)
    if shape == "drag":
        coordinate = (times - duration_ns / 2) / (duration_ns / 4)
        in_phase = np.exp(-0.5 * coordinate**2)
        derivative = -coordinate * in_phase / (duration_ns / 4)
    else:
        # FAST-DRAG uses a cosine series for I and its derivative for Q.
        angular = 2 * np.pi / duration_ns
        angle = angular * times
        in_phase = 0.5 * (1 - np.cos(angle)) + 0.1 * (1 - np.cos(2 * angle))
        derivative = angular * (0.5 * np.sin(angle) + 0.2 * np.sin(2 * angle))
    samples = np.asarray(in_phase + 12j * derivative, dtype=np.complex128)
    return EnvelopeSpec("sampled", samples, times)


@pytest.mark.sim
@pytest.mark.parametrize("shape", ["drag", "fast-drag", "imaginary-ramp"])
def test_smooth_complex_gate_matches_full_density(shape: str) -> None:
    envelope = complex_envelope(shape)
    batch = timeline(2, float(envelope.sample_ns[-1]))
    batch = replace(
        batch, tracks=(drive_track(batch, envelope, frequency=0.6, gain=1.0),)
    )
    budget = 2e-3
    with (
        closing(coherent_source(linear_drive_model())) as source,
        closing(Engine(1, families=("single_tone_gate",), backend="cpu")) as gate,
        closing(Engine(1, families=("dop853",), backend="cpu")) as direct,
    ):
        actual = gate.evolve(source, (batch,), budget=budget)
        expected = direct.evolve(source, (batch,), budget=1e-7)
    assert actual.paths.segments_by_kernel == {"single_tone_gate": 2}
    np.testing.assert_allclose(
        actual.densities, expected.densities, atol=budget, rtol=0
    )


@pytest.mark.sim
@pytest.mark.parametrize(
    "samples",
    [
        pytest.param([0.85 - 0.15j, 0.925 - 0.075j, 1.0], id="slope"),
        pytest.param([1 + 0.035j, 1.0, 1 + 0.035j], id="curvature"),
    ],
)
def test_rough_complex_envelope_is_rejected(samples: list[complex]) -> None:
    # The slope case exceeds the limit only in modulus; the curvature case
    # keeps slopes within range, with a sharp turn in the Q component.
    envelope = EnvelopeSpec("sampled", np.array(samples), np.array([0.0, 4.0, 8.0]))
    batch = timeline(1, 8.0)
    batch = replace(batch, tracks=(drive_track(batch, envelope, frequency=0.6),))
    with (
        closing(coherent_source(linear_drive_model())) as source,
        closing(Engine(1, families=("single_tone_gate",), backend="cpu")) as gate,
        pytest.raises(NoValidKernel) as failure,
    ):
        gate.evolve(source, (batch,), budget=2e-3)
    assert failure.value.attempted == ("single_tone_gate",)
